/*!
shogi_kpp_train — KPP-only standalone trainer for the KPPT family.

Trains the KPP weight tensor and writes `KPP_synthesized.bin` (KPPT layout,
`int16_t × 2`, ~740 MB). The KK and KKP components are trained separately by
`shogi_kk_train` and `shogi_kk_kkp_train`; combining the three `.bin` files
gives a complete KPPT eval.

Network (no hidden layer; KPP only):

    kpp weights (194,100,624 dims, perspective dual, max_active = 703)
        |
        v
    sum (per perspective) -> concat (2) -> linear(out, 2 -> 1) -> sigmoid

Teacher data is given via `--teacher`: a file (`.hcpe` / `.hcpe3` / `.pack`
/ `.psv`), a directory of such files (all concatenated), or a
comma-separated combination. Format is inferred from the extension; all
files must share the same extension.

Memory considerations: the KPP weight tensor is ~776 MB at f32, plus optimiser
state (~2.3 GB total on GPU with AdamW). Sparse input batch buffers are also
~18x larger than the KKP example because `max_active = 703` (= C(38, 2)).

Usage:

    cargo run --release --example shogi_kpp_train -- \
        --teacher inbox/ref/sp_dr2-15K_20240210.hcpe \
        --output checkpoints/kpp \
        --superbatches 3 \
        --batches-per-superbatch 100 \
        --save-rate 1
*/

use std::path::PathBuf;

use bulletou_lib::{
    game::inputs::ShogiKpp,
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
        yaneuraou_kppt::save_yaneuraou_kppt,
    },
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "shogi_kpp_train")]
#[command(about = "KPPT KPP-only standalone trainer (writes KPP_synthesized.bin)")]
struct Args {
    /// Teacher data: file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), directory
    /// of such files (all concatenated), or comma-separated combination.
    #[arg(long)]
    teacher: String,

    /// Checkpoint output directory.
    #[arg(long, default_value = "checkpoints/shogi_kpp")]
    output: PathBuf,

    /// Net identifier (prefix of the saved checkpoint subdirectory).
    #[arg(long, default_value = "shogi_kpp")]
    net_id: String,

    /// Mini-batch size (positions per gradient step).
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of mini-batches per superbatch. Default ≈ 100M positions per
    /// superbatch (100_000_000 / batch_size).
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// Cap on the number of superbatches per epoch. If omitted, run each
    /// epoch until the dataloader reaches EOF.
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

    /// f32 -> i16 quantisation scale for the YaneuraOu KPP output. Note KPP
    /// values are `int16_t × 2` per YaneuraOu's `ValueKpp`, so the scale is an
    /// order of magnitude smaller than the i32 KK / KKP scale. Provisional;
    /// tune empirically.
    #[arg(long, default_value = "400.0")]
    yaneuraou_quant_scale: f32,

    /// Save every N superbatches (1 = save every superbatch, 5 = every 5th).
    #[arg(long, default_value = "1")]
    save_rate: usize,

    /// Dataloader worker threads (CPU side).
    #[arg(long, default_value = "4")]
    threads: usize,

    /// GPU-side batch queue depth.
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

    let qa: i16 = 256;
    let qb: i16 = 64;
    let qab: i16 = qa.checked_mul(qb).expect("qa*qb fits in i16");

    let save_format: Vec<SavedFormat> = vec![
        SavedFormat::id("kppw").round().quantise::<i16>(qa),
        SavedFormat::id("kppb").round().quantise::<i16>(qa),
        SavedFormat::id("outw").transpose().round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qab),
    ];

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKpp)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let kpp = builder.new_affine("kpp", 81 * 1548 * 1548, 1);
        let out = builder.new_affine("out", 2, 1);

        let stm_eval = kpp.forward(stm_inputs);
        let ntm_eval = kpp.forward(ntm_inputs);
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

    let end_superbatch = args.superbatches.unwrap_or(usize::MAX);

    let yaneuraou_scale = args.yaneuraou_quant_scale;
    let max_epochs = args.max_epochs.max(1);
    let output_dir_str = args.output.to_str().unwrap_or("checkpoints").to_string();

    for epoch in 1..=max_epochs {
        if max_epochs > 1 {
            eprintln!("\n=== epoch {epoch} / {max_epochs} ===");
        }
        let net_id_for_epoch = if max_epochs > 1 { format!("{}-e{epoch}", args.net_id) } else { args.net_id.clone() };
        let net_id_for_cb = net_id_for_epoch.clone();
        let output_dir_for_cb = args.output.clone();
        let on_checkpoint_saved = move |superbatch: usize| {
            let ckpt_dir = output_dir_for_cb.join(format!("{net_id_for_cb}-{superbatch}"));
            match save_yaneuraou_kppt(&ckpt_dir, yaneuraou_scale) {
                Ok(()) => eprintln!("  also wrote KPP_synthesized.bin in {}", ckpt_dir.display()),
                Err(e) => {
                    eprintln!("  WARN: failed to write YaneuraOu KPP binary in {}: {e}", ckpt_dir.display())
                }
            }
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
            save_epoch_end: true,
        };

        let settings = LocalSettings {
            threads: args.threads,
            test_set: None,
            output_directory: &output_dir_str,
            batch_queue_size: args.batch_queue_size,
            on_checkpoint_saved: Some(&on_checkpoint_saved),
        };

        match format {
            DataFormat::Hcpe => {
                let loader = HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                trainer.run(&schedule, &settings, &loader);
            }
            DataFormat::Hcpe3 => {
                let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
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
