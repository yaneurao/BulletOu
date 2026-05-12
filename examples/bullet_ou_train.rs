/*!
bullet_ou_train — unified trainer entry point for BulletOu.

A single binary that dispatches to the appropriate training routine via
`--eval-type`. The currently supported types are the three KPPT phases:

    bullet_ou_train --eval-type kppt-kk    ...   (Phase 1: KK only)
    bullet_ou_train --eval-type kppt-kkp   ...   (Phase 2: KKP only)
    bullet_ou_train --eval-type kppt-kpp   ...   (Phase 3: KPP only)

Each writes the corresponding YaneuraOu KPPT-format `.bin` file
(`KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin`) into
the checkpoint directory after every save. To assemble a complete YaneuraOu
KPPT eval (e.g. elmo(WCSC27)-style), run all three with the same data and
merge the three `.bin` files into one directory.

Joint training (one run that emits all three .bin files with shared gradient)
is a future addition (will appear as `--eval-type kppt`).

The standalone examples `shogi_kk_train` / `shogi_kk_kkp_train` /
`shogi_kpp_train` remain available as direct entry points; the unified
binary just dispatches to the same logic for users who prefer one CLI.

Input format is inferred from the file extension (`.hcpe` / `.hcpe3` /
`.pack`); mixed extensions are rejected.

Usage:

    cargo run --release --features cuda --example bullet_ou_train -- \
        --eval-type kppt-kk \
        --data inbox/ref/sp_dr2-15K_20240210.hcpe \
        --output checkpoints/kk-phase1 \
        --superbatches 10
*/

use std::path::{Path, PathBuf};

use bullet_lib::{
    game::inputs::{ShogiKk, ShogiKkp, ShogiKpp},
    nn::optimiser,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader},
        yaneuraou_kppt::{KppFormat, save_yaneuraou_eval},
    },
};
use clap::{Parser, ValueEnum};

// ----- eval-type ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EvalType {
    /// KPPT Phase 1: train only the KK term, write KK_synthesized.bin.
    KpptKk,
    /// KPPT Phase 2: train only the KKP term, write KKP_synthesized.bin.
    KpptKkp,
    /// KPPT Phase 3: train only the KPP term, write KPP_synthesized.bin
    /// (with turn channel — int16 × 2).
    KpptKpp,
    /// KPP_KKPT (factorised KPPT): train the KPP term and write
    /// `KPP_synthesized.bin` *without* the turn channel (int16 × 1, ~388 MB).
    /// KK and KKP for the KPP_KKPT family are identical to `kppt-kk` /
    /// `kppt-kkp` — use those for the other two components.
    KppKkptKpp,
}

impl EvalType {
    fn default_net_id(self) -> &'static str {
        match self {
            EvalType::KpptKk => "shogi_kk_phase1",
            EvalType::KpptKkp => "shogi_kk_kkp_phase2",
            EvalType::KpptKpp => "shogi_kpp_phase3",
            EvalType::KppKkptKpp => "shogi_kpp_kpp_kkpt",
        }
    }

    fn default_output(self) -> &'static str {
        match self {
            EvalType::KpptKk => "checkpoints/shogi_kk_phase1",
            EvalType::KpptKkp => "checkpoints/shogi_kk_kkp_phase2",
            EvalType::KpptKpp => "checkpoints/shogi_kpp_phase3",
            EvalType::KppKkptKpp => "checkpoints/shogi_kpp_kpp_kkpt",
        }
    }

    /// Suggested f32 -> i{16,32} quantisation scale for the YaneuraOu writer.
    /// KK / KKP are i32 (large dynamic range) so default 4000 = eval_scale * 10.
    /// KPP and KPP_KKPT KPP are i16 (small dynamic range) so default is an
    /// order of magnitude smaller.
    fn default_yaneuraou_quant_scale(self) -> f32 {
        match self {
            EvalType::KpptKk | EvalType::KpptKkp => 4000.0,
            EvalType::KpptKpp | EvalType::KppKkptKpp => 400.0,
        }
    }

    /// On-disk KPP layout to write at checkpoint time. KK / KKP eval types
    /// don't have a KPP file so this is ignored.
    fn kpp_format(self) -> KppFormat {
        match self {
            EvalType::KppKkptKpp => KppFormat::KppKkpt,
            _ => KppFormat::Kppt,
        }
    }
}

// ----- data format inference --------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataFormat {
    Hcpe,
    Hcpe3,
    Pack,
}

fn infer_data_format(paths: &[&str]) -> Result<DataFormat, String> {
    let mut found: Option<DataFormat> = None;
    for p in paths {
        let ext = Path::new(p).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
        let fmt = match ext.as_deref() {
            Some("hcpe") => DataFormat::Hcpe,
            Some("hcpe3") => DataFormat::Hcpe3,
            Some("pack") => DataFormat::Pack,
            _ => {
                return Err(format!(
                    "cannot infer data format from path: {p}\n  expected file extension: .hcpe / .hcpe3 / .pack"
                ));
            }
        };
        match found {
            None => found = Some(fmt),
            Some(prev) if prev == fmt => {}
            Some(prev) => {
                return Err(format!("mixed data formats: {p} is {fmt:?} but previous file(s) were {prev:?}"));
            }
        }
    }
    found.ok_or_else(|| "no data files provided".to_string())
}

// ----- CLI ---------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "bullet_ou_train")]
#[command(about = "BulletOu unified trainer (KPPT phases for now)")]
struct Args {
    /// Evaluation function type to train.
    #[arg(long, value_enum)]
    eval_type: EvalType,

    /// Training data file(s), comma-separated. Format (`.hcpe` / `.hcpe3` / `.pack`)
    /// is inferred from the extension.
    #[arg(long)]
    data: String,

    /// Checkpoint output directory. Defaults to a per-eval-type path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Net identifier (prefix of the saved checkpoint subdirectory).
    /// Defaults to a per-eval-type name.
    #[arg(long)]
    net_id: Option<String>,

    /// Mini-batch size (positions per gradient step).
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of mini-batches per superbatch. Default ≈ 100M positions per
    /// superbatch (100_000_000 / batch_size).
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// Number of superbatches to run (end_superbatch).
    #[arg(long, default_value = "10")]
    superbatches: usize,

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

    /// f32 -> integer quantisation scale for the YaneuraOu output. If
    /// omitted, an eval-type-specific default is used (4000 for KK/KKP,
    /// 400 for KPP).
    #[arg(long)]
    yaneuraou_quant_scale: Option<f32>,

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

impl Args {
    fn output_dir(&self) -> PathBuf {
        self.output.clone().unwrap_or_else(|| PathBuf::from(self.eval_type.default_output()))
    }

    fn net_id(&self) -> String {
        self.net_id.clone().unwrap_or_else(|| self.eval_type.default_net_id().to_string())
    }

    fn yaneuraou_scale(&self) -> f32 {
        self.yaneuraou_quant_scale.unwrap_or_else(|| self.eval_type.default_yaneuraou_quant_scale())
    }

    fn kpp_format(&self) -> KppFormat {
        self.eval_type.kpp_format()
    }
}

// ----- dispatch ----------------------------------------------------------

fn main() {
    let args = Args::parse();
    match args.eval_type {
        EvalType::KpptKk => run_kppt_kk(&args),
        EvalType::KpptKkp => run_kppt_kkp(&args),
        // Phase 3 trains the same network (KPP-only standalone) regardless of
        // whether the on-disk file is the KPPT (with turn channel) or
        // KPP_KKPT (factorised) variant. Only the writer differs, and that's
        // decided inside `run_training_inline!` via `args.kpp_format()`.
        EvalType::KpptKpp | EvalType::KppKkptKpp => run_kppt_kpp(&args),
    }
}

// `Trainer<G, O, S>` の concrete type は bullet API として直接露出していないので、
// 3 branch を generic helper でまとめる代わりに、共通の schedule / settings /
// loader dispatch をマクロで encapsulate する。
//
// 各 branch は: (a) save_format / weight ID を決め、(b) ValueTrainerBuilder で
// trainer を構築し、(c) `run_training_inline!(args, trainer)` を呼ぶ。
macro_rules! run_training_inline {
    ($args:expr, $trainer:expr) => {{
        let args: &Args = $args;
        let trainer = $trainer;

        let batches_per_superbatch =
            args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));

        let schedule = TrainingSchedule {
            net_id: args.net_id(),
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

        let net_id = args.net_id();
        let output_dir_buf = args.output_dir();
        let yaneuraou_scale = args.yaneuraou_scale();
        let kpp_format = args.kpp_format();
        let on_checkpoint_saved = move |superbatch: usize| {
            let ckpt_dir = output_dir_buf.join(format!("{net_id}-{superbatch}"));
            match save_yaneuraou_eval(&ckpt_dir, yaneuraou_scale, kpp_format) {
                Ok(()) => eprintln!("  also wrote YaneuraOu eval binary in {}", ckpt_dir.display()),
                Err(e) => {
                    eprintln!("  WARN: failed to write YaneuraOu eval binary in {}: {e}", ckpt_dir.display())
                }
            }
        };

        let output_dir_str = args.output_dir();
        let output_dir = output_dir_str.to_str().unwrap_or("checkpoints");
        let settings = LocalSettings {
            threads: args.threads,
            test_set: None,
            output_directory: output_dir,
            batch_queue_size: args.batch_queue_size,
            on_checkpoint_saved: Some(&on_checkpoint_saved),
        };

        let data_files_owned: Vec<String> = args.data.split(',').map(|s| s.trim().to_string()).collect();
        let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

        let format = infer_data_format(&data_files_ref).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });

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
                let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                trainer.run(&schedule, &settings, &loader);
            }
        }
    }};
}

// ----- KPPT Phase 1: KK --------------------------------------------------

fn run_kppt_kk(args: &Args) {
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

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT Phase 2: KKP -------------------------------------------------

fn run_kppt_kkp(args: &Args) {
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

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT Phase 3: KPP -------------------------------------------------

fn run_kppt_kpp(args: &Args) {
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

    run_training_inline!(args, &mut trainer);
}
