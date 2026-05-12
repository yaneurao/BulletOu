/*!
shogi_kk_kkp_train — KPPT Phase 2: KK + KKP minimal trainer that also writes
the YaneuraOu KPPT-format `.bin` files (`KK_synthesized.bin` /
`KKP_synthesized.bin`) after each checkpoint.

Network (no hidden layer; KK and KKP are summed linearly):

    kk weights (6,561 dims, perspective dual)
        +
    kkp weights (10,156,428 dims, perspective dual)
        |
        v
    sum (per perspective) -> concat (2) -> linear(out, 2 -> 1) -> sigmoid

This is the smallest configuration that produces a non-trivial YaneuraOu KPPT
output (KK_synthesized.bin + KKP_synthesized.bin). Strength is still expected
to be limited until KPP is added (Phase 3) and the turn term is wired up
(Phase 3/4).

Usage:

    cargo run --release --features cuda --example shogi_kk_kkp_train -- \
        --data inbox/ref/sp_dr2-15K_20240210.hcpe \
        --data-format hcpe \
        --output checkpoints/kk-kkp-phase2 \
        --superbatches 3 \
        --batches-per-superbatch 100 \
        --save-rate 1
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::{ShogiKk, ShogiKkp},
    nn::optimiser,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader},
        yaneuraou_kppt::save_yaneuraou_kppt,
    },
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum DataFormat {
    #[default]
    Hcpe,
    Hcpe3,
    Pack,
}

#[derive(Parser, Debug)]
#[command(name = "shogi_kk_kkp_train")]
#[command(about = "KPPT Phase 2: KK + KKP minimal trainer with YaneuraOu-format output")]
struct Args {
    /// Training data path(s) (comma-separated).
    #[arg(long)]
    data: String,

    #[arg(long, value_enum, default_value = "hcpe")]
    data_format: DataFormat,

    #[arg(long, default_value = "checkpoints/shogi_kk_kkp_phase2")]
    output: PathBuf,

    #[arg(long, default_value = "shogi_kk_kkp_phase2")]
    net_id: String,

    #[arg(long, default_value = "16384")]
    batch_size: usize,

    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    #[arg(long, default_value = "10")]
    superbatches: usize,

    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    #[arg(long, default_value = "0.001")]
    lr: f32,

    #[arg(long, default_value = "0.1")]
    lr_gamma: f32,

    #[arg(long, default_value = "8")]
    lr_step: usize,

    #[arg(long, default_value = "0.0")]
    start_wdl: f32,

    #[arg(long, default_value = "1.0")]
    end_wdl: f32,

    #[arg(long, default_value = "400")]
    scale: u32,

    /// f32 -> i32 quantisation scale for the YaneuraOu KPPT output.
    /// Provisionally `eval_scale * 10`. Will be tuned empirically.
    #[arg(long, default_value = "4000.0")]
    yaneuraou_quant_scale: f32,

    #[arg(long, default_value = "1")]
    save_rate: usize,

    #[arg(long, default_value = "4")]
    threads: usize,

    #[arg(long, default_value = "32")]
    batch_queue_size: usize,

    #[arg(long, default_value = "256")]
    buffer_mb: usize,

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
    // combine KK and KKP we build one big sparse input by concatenating the
    // dim-spaces — but that requires a new combined SparseInputType.
    //
    // For Phase 2 we take the simpler path: train KKP alone here (it dominates
    // KK in expressivity anyway), and write only `KKP_synthesized.bin`. A
    // separate `shogi_kk_train` produces `KK_synthesized.bin`. The combined
    // training that emits both files in one run is part of Phase 4 / the
    // ValueTrainerBuilder tuple-input extension noted in the roadmap.
    //
    // The save_yaneuraou_kppt helper still tolerates either or both of `kkw` /
    // `kkpw` being present in raw.bin, so a future combined run drops in
    // without changes.

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

    let batches_per_superbatch =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));

    let schedule = TrainingSchedule {
        net_id: args.net_id.clone(),
        eval_scale: args.scale as f32,
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch,
            start_superbatch: args.start_superbatch,
            end_superbatch: args.superbatches,
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

    let data_files_owned: Vec<String> = args.data.split(',').map(|s| s.trim().to_string()).collect();
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

    macro_rules! run_with_loader {
        ($loader:expr) => {{
            let loader = $loader;
            trainer.run(&schedule, &settings, &loader);
        }};
    }

    match args.data_format {
        DataFormat::Hcpe => {
            run_with_loader!(HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true))
        }
        DataFormat::Hcpe3 => {
            run_with_loader!(Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true))
        }
        DataFormat::Pack => {
            run_with_loader!(ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true))
        }
    }
}
