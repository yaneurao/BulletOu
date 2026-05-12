/*!
shogi_kk_kkp_train — KKP-only standalone trainer for the KPPT family.

Despite the historical filename, this trains only the **KKP** weight tensor
and writes `KKP_synthesized.bin`. The KK component is trained separately by
`shogi_kk_train`; combining the two `.bin` files gives the KK + KKP portion
of a KPPT eval (the KPP portion comes from `shogi_kpp_train`).

Network (no hidden layer; KKP only):

    kkp weights (10,156,428 dims, perspective dual)
        |
        v
    sum (per perspective) -> concat (2) -> linear(out, 2 -> 1) -> sigmoid

Teacher data is given via `--teacher`: a file (`.hcpe` / `.hcpe3` / `.pack`
/ `.psv`), a directory of such files (all concatenated), or a
comma-separated combination. Format is inferred from the extension; all
files must share the same extension.

Usage:

    cargo run --release --features device-cuda --example shogi_kk_kkp_train -- \
        --teacher inbox/ref/sp_dr2-15K_20240210.hcpe \
        --output checkpoints/kkp \
        --superbatches 3 \
        --batches-per-superbatch 100 \
        --save-rate 1
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::{ShogiKk, ShogiKkp},
    nn::optimiser,
    teacher_path::{DataFormat, compute_auto_epoch_superbatches, expand_teacher, infer_data_format},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader},
        yaneuraou_kppt::save_yaneuraou_kppt,
    },
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "shogi_kk_kkp_train")]
#[command(about = "KPPT KKP-only standalone trainer (writes KKP_synthesized.bin)")]
struct Args {
    /// Teacher data: file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), directory
    /// of such files (all concatenated), or comma-separated combination.
    #[arg(long)]
    teacher: String,

    /// Checkpoint output directory.
    #[arg(long, default_value = "checkpoints/shogi_kkp")]
    output: PathBuf,

    /// Net identifier (prefix of the saved checkpoint subdirectory).
    #[arg(long, default_value = "shogi_kkp")]
    net_id: String,

    /// Mini-batch size (positions per gradient step).
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of mini-batches per superbatch. Default ≈ 100M positions per
    /// superbatch (100_000_000 / batch_size).
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// Number of superbatches to run. If omitted, the trainer runs for one
    /// epoch through the teacher data (computed from file sizes).
    #[arg(long)]
    superbatches: Option<usize>,

    /// Starting superbatch counter (>1 to resume / extend).
    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    /// Initial Adam learning rate.
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// LR gamma (multiplicative drop applied every `lr_step` superbatches).
    #[arg(long, default_value = "0.1")]
    lr_gamma: f32,

    /// LR step: apply `lr_gamma` every N superbatches.
    #[arg(long, default_value = "8")]
    lr_step: usize,

    /// Start of the WDL linear schedule (0 = pure eval, 1 = pure game result).
    #[arg(long, default_value = "0.0")]
    start_wdl: f32,

    /// End of the WDL linear schedule.
    #[arg(long, default_value = "1.0")]
    end_wdl: f32,

    /// Eval-to-score sigmoid scale.
    #[arg(long, default_value = "400")]
    scale: u32,

    /// f32 -> i32 quantisation scale for the YaneuraOu KPPT output.
    /// Provisionally `eval_scale * 10`. Will be tuned empirically.
    #[arg(long, default_value = "4000.0")]
    yaneuraou_quant_scale: f32,

    /// Save every N superbatches (1 = save every superbatch, 5 = every 5th).
    #[arg(long, default_value = "1")]
    save_rate: usize,

    /// Dataloader worker threads (CPU side).
    #[arg(long, default_value = "4")]
    threads: usize,

    /// GPU-side batch queue depth (number of batches the dataloader may
    /// stage ahead of training).
    #[arg(long, default_value = "32")]
    batch_queue_size: usize,

    /// Loader shuffle buffer size in megabytes.
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
    //   ShogiKk      (6,561 dims, 1-hot per perspective)
    //   ShogiKkp     (10,156,428 dims, ~38 active per perspective)
    //
    // But bullet's ValueTrainerBuilder only takes a single SparseInputType. To
    // combine KK and KKP we'd need to build one big sparse input by
    // concatenating the dim-spaces. Instead we train KKP alone here (it
    // dominates KK in expressivity anyway) and write only
    // `KKP_synthesized.bin`. A separate `shogi_kk_train` produces
    // `KK_synthesized.bin`.
    //
    // `save_yaneuraou_kppt` tolerates either or both of `kkw` / `kkpw` /
    // `kppw` being present in raw.bin, so a future combined run drops in
    // without changes to the writer.

    let _ = ShogiKk; // imported for symmetry; not used in this example

    let qa: i16 = 256;
    let qb: i16 = 64;
    let qab: i16 = qa.checked_mul(qb).expect("qa*qb fits in i16");

    let save_format: Vec<SavedFormat> = vec![
        SavedFormat::id("kkpw").round().quantise::<i16>(qa),
        SavedFormat::id("kkpb").round().quantise::<i16>(qa),
        SavedFormat::id("outw").transpose().round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qab),
    ];

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKkp)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let kkp = builder.new_affine("kkp", 81 * 81 * 1548, 1);
        let out = builder.new_affine("out", 2, 1);

        let stm_eval = kkp.forward(stm_inputs);
        let ntm_eval = kkp.forward(ntm_inputs);
        let combined = stm_eval.concat(ntm_eval);
        out.forward(combined)
    });

    let data_files_owned = expand_teacher(&args.teacher).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

    let format = infer_data_format(&data_files_ref).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });

    let batches_per_superbatch =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));

    let end_superbatch = args.superbatches.unwrap_or_else(|| {
        compute_auto_epoch_superbatches(&data_files_ref, format, args.batch_size, batches_per_superbatch)
    });

    let schedule = TrainingSchedule {
        net_id: args.net_id.clone(),
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

    // After every checkpoint, also write the YaneuraOu KPPT-format binaries
    // alongside raw.bin / quantised.bin in the saved superbatch directory.
    let net_id = args.net_id.clone();
    let output_dir_buf = args.output.clone();
    let yaneuraou_scale = args.yaneuraou_quant_scale;
    let on_checkpoint_saved = move |superbatch: usize| {
        let ckpt_dir = output_dir_buf.join(format!("{net_id}-{superbatch}"));
        match save_yaneuraou_kppt(&ckpt_dir, yaneuraou_scale) {
            Ok(()) => eprintln!("  also wrote KK_synthesized.bin / KKP_synthesized.bin in {}", ckpt_dir.display()),
            Err(e) => eprintln!("  WARN: failed to write YaneuraOu KPPT binaries in {}: {e}", ckpt_dir.display()),
        }
    };

    let output_dir = args.output.to_str().unwrap_or("checkpoints");
    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: args.batch_queue_size,
        on_checkpoint_saved: Some(&on_checkpoint_saved),
    };

    macro_rules! run_with_loader {
        ($loader:expr) => {{
            let loader = $loader;
            trainer.run(&schedule, &settings, &loader);
        }};
    }

    match format {
        DataFormat::Hcpe => {
            run_with_loader!(HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true))
        }
        DataFormat::Hcpe3 => {
            run_with_loader!(Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true))
        }
        DataFormat::Pack => {
            run_with_loader!(ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true))
        }
        DataFormat::Psv => {
            run_with_loader!(DirectSequentialDataLoader::new(&data_files_ref))
        }
    }
}
