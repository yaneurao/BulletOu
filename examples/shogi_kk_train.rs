/*!
shogi_kk_train — KK-only standalone trainer for the KPPT-family KK component.

Exercises the `ShogiKk` sparse input (81 × 81 = 6,561 dims, max_active = 1)
in the simplest possible network:

  ShogiKk (sparse) -> linear -> 1 scalar  (dual-perspective concatenated)

There is no hidden layer, no SCReLU, no Layer Stack. KK by itself is too weak
to play strongly; this example produces `KK_synthesized.bin` to be combined
with separately-trained `KKP_synthesized.bin` and `KPP_synthesized.bin`
(see `shogi_kk_kkp_train` / `shogi_kpp_train`, or `bulletou --eval-type ...`).

Teacher data is given via `--teacher`: a file (`.hcpe` / `.hcpe3` / `.pack`
/ `.psv`), a directory of such files (all concatenated), or a
comma-separated combination. Format is inferred from the extension; all
files must share the same extension.

Usage:

    cargo run --release --features device-cuda --example shogi_kk_train -- \
        --teacher /data/shogi/train.hcpe \
        --output checkpoints/kk \
        --superbatches 10
*/

use std::path::PathBuf;

use bulletou_lib::{
    game::inputs::ShogiKk,
    nn::optimiser,
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader},
    },
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "shogi_kk_train")]
#[command(about = "KPPT KK-only standalone trainer (writes KK_synthesized.bin)")]
struct Args {
    /// Teacher data: file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), directory
    /// of such files (all concatenated), or comma-separated combination.
    #[arg(long)]
    teacher: String,

    /// Checkpoint output directory.
    #[arg(long, default_value = "checkpoints/shogi_kk")]
    output: PathBuf,

    /// Net identifier (prefix of the saved checkpoint subdirectory).
    #[arg(long, default_value = "shogi_kk")]
    net_id: String,

    /// Batch size.
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Batches per superbatch (default: ≈ 100M positions per superbatch).
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// Cap on the number of superbatches per epoch. If omitted, run each
    /// epoch until the dataloader reaches EOF (no cap). Mutually exclusive
    /// with `--max-epochs` in practical use.
    #[arg(long)]
    superbatches: Option<usize>,

    /// Number of epochs (= dataloader EOFs) to run. LR scheduler restarts at
    /// superbatch 1 each epoch.
    #[arg(long, default_value = "1")]
    max_epochs: usize,

    /// Starting superbatch counter (>1 to resume / extend).
    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    /// Initial Adam learning rate.
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// LR gamma (multiplicative drop).
    #[arg(long, default_value = "0.1")]
    lr_gamma: f32,

    /// LR step (apply gamma every N superbatches).
    #[arg(long, default_value = "8")]
    lr_step: usize,

    /// Start of the WDL linear schedule.
    #[arg(long, default_value = "0.0")]
    start_wdl: f32,

    /// End of the WDL linear schedule.
    #[arg(long, default_value = "1.0")]
    end_wdl: f32,

    /// Eval-to-score sigmoid scale.
    #[arg(long, default_value = "400")]
    scale: u32,

    /// Save every N superbatches.
    #[arg(long, default_value = "5")]
    save_rate: usize,

    /// Dataloader worker threads (CPU side).
    #[arg(long, default_value = "4")]
    threads: usize,

    /// GPU-side batch queue depth.
    #[arg(long, default_value = "32")]
    batch_queue_size: usize,

    /// Shuffle buffer size in megabytes (for hcpe / hcpe3 loaders).
    #[arg(long, default_value = "256")]
    buffer_mb: usize,

    /// Drop positions whose |score| >= this. Useful to exclude ±32000 mate
    /// stamps. Set to 0 to disable.
    #[arg(long, default_value = "32000")]
    score_drop_abs: u16,
}

fn main() {
    let args = Args::parse();

    // ------- Network --------
    //
    //   ShogiKk (6561 dims, 1-hot per perspective)
    //     |
    //     v   (linear: kk weights, 6561 -> 1 per perspective)
    //   stm_eval  ntm_eval
    //     \        /
    //      concat (2 dims)
    //         |
    //         v   (linear: out, 2 -> 1)
    //       y
    //
    // Quantisation (saved binary): Standard NNUE-ish.
    //   kk weights -> i16 with QA = 256
    //   out weights -> i16 with QB = 64
    //   out bias    -> i16 with QA * QB

    let qa: i16 = 256;
    let qb: i16 = 64;
    let qab: i16 = qa.checked_mul(qb).expect("qa*qb fits in i16");

    let save_format: Vec<SavedFormat> = vec![
        SavedFormat::id("kkw").round().quantise::<i16>(qa),
        SavedFormat::id("kkb").round().quantise::<i16>(qa),
        SavedFormat::id("outw").transpose().round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qab),
    ];

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKk)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let kk = builder.new_affine("kk", 6561, 1);
        let out = builder.new_affine("out", 2, 1);

        let stm_eval = kk.forward(stm_inputs);
        let ntm_eval = kk.forward(ntm_inputs);
        let combined = stm_eval.concat(ntm_eval);
        out.forward(combined)
    });

    // ------- Teacher data -------
    let data_files_owned = expand_teacher(&args.teacher).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

    let format = infer_data_format(&data_files_ref).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });

    // ------- Schedule --------
    let batches_per_superbatch =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));

    let end_superbatch = args.superbatches.unwrap_or(usize::MAX);

    let max_epochs = args.max_epochs.max(1);
    let output_dir = args.output.to_str().unwrap_or("checkpoints").to_string();
    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory: &output_dir,
        batch_queue_size: args.batch_queue_size,
        on_checkpoint_saved: None,
    };

    for epoch in 1..=max_epochs {
        if max_epochs > 1 {
            eprintln!("\n=== epoch {epoch} / {max_epochs} ===");
        }
        let net_id_for_epoch = if max_epochs > 1 {
            format!("{}-e{epoch}", args.net_id)
        } else {
            args.net_id.clone()
        };
        let schedule = TrainingSchedule {
            net_id: net_id_for_epoch,
            eval_scale: args.scale as f32,
            steps: TrainingSteps {
                batch_size: args.batch_size,
                batches_per_superbatch,
                start_superbatch: args.start_superbatch,
                end_superbatch,
            },
            wdl_scheduler: wdl::LinearWDL { start: args.start_wdl, end: args.end_wdl },
            lr_scheduler: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
            save_rate: args.save_rate,
        };

        match format {
            DataFormat::Hcpe => {
                let loader =
                    HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                trainer.run(&schedule, &settings, &loader);
            }
            DataFormat::Hcpe3 => {
                let loader =
                    Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                trainer.run(&schedule, &settings, &loader);
            }
            DataFormat::Pack => {
                let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                    .with_single_epoch(true);
                trainer.run(&schedule, &settings, &loader);
            }
            DataFormat::Psv => {
                let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
                trainer.run(&schedule, &settings, &loader);
            }
        }
    }
}
