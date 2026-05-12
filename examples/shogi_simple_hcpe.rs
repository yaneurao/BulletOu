/*!
shogi_simple_hcpe - Train a shogi NNUE directly from HCPE (HuffmanCodedPosAndEval) files.

This is a minimal example that exercises the new `HcpeDataLoader`. It uses the
same network shape as `shogi_simple.rs` (ShogiHalfKA_hm + single hidden layer +
SCReLU activation + dual-perspective), but reads `.hcpe` (dlshogi-style 38-byte
fixed-length records) instead of the formats `shogi_simple.rs` consumes (a
flat `.bin` dump of `PackedSfenValue` records, or YaneuraOu-ScriptCollection's `gensfen`
per-game variable-length `.pack`). All three eventually decode into the same
in-memory `PackedSfenValue` stream the trainer consumes. There is no
input-format option here — this example is *intentionally* hcpe-only, to keep
the code short and unambiguous.

For a richer training script (multiple feature sets, optimisers, win-rate
model, quantisation control, resume, etc.) see `shogi_simple.rs`, which
natively reads `.pack` and `.bin` (no conversion step needed).

Usage:

    # Required:
    #   --teacher <PATH>   .hcpe file, directory of .hcpe files, or
    #                      comma-separated combination
    #
    cargo run --release --features device-cuda --example shogi_simple_hcpe -- \
        --teacher /data/shogi/train.hcpe \
        --output checkpoints/my-hcpe-net \
        --superbatches 40

HCPE caveats (see crates/bullet_lib/src/value/loader/hcpe.rs for details):
  - HCPE has no game_ply, so Layer Stack's ply9 bucket cannot be used.
    (kingrank9 / progress8* buckets are fine, but this minimal example
    uses no bucketing at all.)
  - HCPE has no policy teacher (MoveVisits); value-only training.
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::ShogiHalfKA_hm,
    nn::optimiser,
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::HcpeDataLoader},
};
use clap::Parser;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(name = "shogi_simple_hcpe")]
#[command(about = "Minimal shogi NNUE training from HCPE files")]
struct Args {
    /// Teacher data: `.hcpe` file, directory of `.hcpe` files, or a
    /// comma-separated combination of either.
    #[arg(long)]
    teacher: String,

    /// Checkpoint output directory.
    #[arg(long, default_value = "checkpoints/shogi_simple_hcpe")]
    output: PathBuf,

    /// Net identifier (used as the checkpoint subdirectory name prefix).
    #[arg(long, default_value = "shogi_hcpe")]
    net_id: String,

    /// Batch size.
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of batches per superbatch. If unset, computed so that one
    /// superbatch ≈ 100M positions.
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// Number of superbatches (≈ epochs at default size).
    #[arg(long, default_value = "40")]
    superbatches: usize,

    /// Start of the superbatch counter (use >1 if resuming or appending).
    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    /// Initial learning rate (Adam).
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// LR schedule: multiply lr by gamma every `lr_step` superbatches.
    #[arg(long, default_value = "0.1")]
    lr_gamma: f32,

    /// LR schedule: step size (superbatches).
    #[arg(long, default_value = "18")]
    lr_step: usize,

    /// WDL ratio at the start of training (linear schedule).
    /// `wdl=0.0` means pure score loss; `1.0` means pure WDL loss.
    #[arg(long, default_value = "0.0")]
    start_wdl: f32,

    /// WDL ratio at the end of training.
    #[arg(long, default_value = "1.0")]
    end_wdl: f32,

    /// Eval-to-score scale (sigmoid scaling for the score loss).
    #[arg(long, default_value = "400")]
    scale: u32,

    /// Hidden layer 1 size (FT output per perspective).
    /// Total combined hidden width is `2 * l1` after dual-perspective concat.
    #[arg(long, default_value = "1024")]
    l1: usize,

    /// Save a checkpoint every `save_rate` superbatches.
    #[arg(long, default_value = "10")]
    save_rate: usize,

    /// Worker threads for the dataloader CPU pipeline.
    #[arg(long, default_value = "4")]
    threads: usize,

    /// GPU side batch queue depth (pipeline depth).
    #[arg(long, default_value = "32")]
    batch_queue_size: usize,

    /// HCPE shuffle buffer size in megabytes (decoded `PackedSfenValue` records).
    #[arg(long, default_value = "256")]
    buffer_mb: usize,

    /// Drop positions whose |score| >= score_drop_abs. Useful to exclude
    /// mate stamps (±32000) and similar outliers.
    /// Set to 0 to disable the filter.
    #[arg(long, default_value = "32000")]
    score_drop_abs: u16,
}

// =============================================================================
// main
// =============================================================================

fn main() {
    let args = Args::parse();

    // ----- Build network -----
    //
    // Architecture (matches the minimal half of shogi_simple.rs):
    //   ShogiHalfKA_hm (73,305-d sparse) -> L0 (l1 wide) -> SCReLU
    //                                                    -> concat(stm, ntm)
    //                                                    -> out (1)
    let input_size: usize = 73_305;
    let l1_size: usize = args.l1;

    // Quantisation constants (matches shogi_simple.rs Standard format).
    let qa: i16 = 256;
    let qb: i16 = 64;
    let qab: i16 = qa.checked_mul(qb).expect("qa*qb fits in i16");

    let save_format: Vec<SavedFormat> = vec![
        SavedFormat::id("l0w").round().quantise::<i16>(qa),
        SavedFormat::id("l0b").round().quantise::<i16>(qa),
        SavedFormat::id("outw").transpose().round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qab),
    ];

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiHalfKA_hm)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let l0 = builder.new_affine("l0", input_size, l1_size);
        let out = builder.new_affine("out", l1_size * 2, 1);

        let stm_hidden = l0.forward(stm_inputs).screlu();
        let ntm_hidden = l0.forward(ntm_inputs).screlu();
        let combined = stm_hidden.concat(ntm_hidden);
        out.forward(combined)
    });

    // ----- Training schedule -----
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

    // ----- Local settings -----
    let output_dir = args.output.to_str().unwrap_or("checkpoints");
    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: args.batch_queue_size,
        on_checkpoint_saved: None,
    };

    // ----- Data loader -----
    let data_files_owned = expand_teacher(&args.teacher).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();
    let format = infer_data_format(&data_files_ref).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    if format != DataFormat::Hcpe {
        eprintln!("error: shogi_simple_hcpe accepts only .hcpe files (got {format:?})");
        std::process::exit(2);
    }

    let loader = HcpeDataLoader::new_concat_multiple(
        &data_files_ref,
        args.buffer_mb,
        |_psv| true, // value-level filter is already applied via .score_drop_abs() above
    );

    // ----- Run training -----
    trainer.run(&schedule, &settings, &loader);
}
