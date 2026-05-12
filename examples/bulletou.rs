/*!
bulletou — BulletOu trainer entry point.

Dispatches to the appropriate training routine via `--eval-type`. The
"family" eval-types train all three KPPT components (KK + KKP + KPP)
sequentially in a single invocation and assemble the result into
`<output>/final/`:

    bulletou --eval-type kppt            (KPPT family, KPP int16 × 2)
    bulletou --eval-type kpp-kkpt        (KPP_KKPT factorised, KPP int16)

To train a single component standalone (= for development / smoke testing):

    bulletou --eval-type kppt-kk         KK only
    bulletou --eval-type kppt-kkp        KKP only
    bulletou --eval-type kppt-kpp        KPP only, KPPT layout
    bulletou --eval-type kpp-kkpt-kpp    KPP only, KPP_KKPT layout

Teacher data is given via `--teacher`. The argument is either a single
file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), a directory containing such
files (all matching files are concatenated), or a comma-separated list
of either. Format is inferred from the file extension; all files must
share the same extension.

Usage:

    cargo run --release --features device-cuda --example bulletou -- \
        --eval-type kppt \
        --teacher /data/shogi/train_set/ \
        --output checkpoints/my-kppt \
        --superbatches 20
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::{ShogiKk, ShogiKkp, ShogiKpp},
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
        yaneuraou_kppt::{KppFormat, save_yaneuraou_eval},
    },
};
use clap::{Parser, ValueEnum};

// ----- eval-type ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EvalType {
    /// KPPT family: train KK, KKP, and KPP sequentially and assemble the
    /// three-file KPPT eval (`KK_synthesized.bin` / `KKP_synthesized.bin` /
    /// `KPP_synthesized.bin`) into `<output>/final/`.
    Kppt,
    /// KPP_KKPT family (factorised KPPT): same as `kppt` but KPP is written
    /// in the KPP_KKPT layout (no turn channel; half the KPP file size).
    KppKkpt,
    /// KPPT KK component only.
    KpptKk,
    /// KPPT KKP component only.
    KpptKkp,
    /// KPPT KPP component only (with turn channel; ~740 MB).
    KpptKpp,
    /// KPP_KKPT KPP component only (no turn channel; ~388 MB).
    KppKkptKpp,
}

impl EvalType {
    fn default_net_id(self) -> &'static str {
        match self {
            EvalType::Kppt => "shogi_kppt",
            EvalType::KppKkpt => "shogi_kpp_kkpt",
            EvalType::KpptKk => "shogi_kk",
            EvalType::KpptKkp => "shogi_kkp",
            EvalType::KpptKpp => "shogi_kpp",
            EvalType::KppKkptKpp => "shogi_kpp_factorised",
        }
    }

    fn default_output(self) -> &'static str {
        match self {
            EvalType::Kppt => "checkpoints/shogi_kppt",
            EvalType::KppKkpt => "checkpoints/shogi_kpp_kkpt",
            EvalType::KpptKk => "checkpoints/shogi_kk",
            EvalType::KpptKkp => "checkpoints/shogi_kkp",
            EvalType::KpptKpp => "checkpoints/shogi_kpp",
            EvalType::KppKkptKpp => "checkpoints/shogi_kpp_factorised",
        }
    }

    /// Suggested f32 -> i{16,32} quantisation scale for the YaneuraOu writer.
    /// KK / KKP entries are i32 (large dynamic range) so 4000 = eval_scale * 10.
    /// KPP entries are i16 (smaller dynamic range) so the scale is an
    /// order of magnitude smaller.
    fn default_yaneuraou_quant_scale(self) -> f32 {
        match self {
            EvalType::Kppt | EvalType::KppKkpt | EvalType::KpptKk | EvalType::KpptKkp => 4000.0,
            EvalType::KpptKpp | EvalType::KppKkptKpp => 400.0,
        }
    }

    /// On-disk KPP layout to write at checkpoint time. KK / KKP eval types
    /// don't have a KPP file so this is ignored.
    fn kpp_format(self) -> KppFormat {
        match self {
            EvalType::KppKkpt | EvalType::KppKkptKpp => KppFormat::KppKkpt,
            _ => KppFormat::Kppt,
        }
    }
}

// (teacher-path expansion and format inference live in
//  `bullet_lib::teacher_path` so the single-component examples can share them.)

// ----- CLI ---------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
#[command(name = "bulletou")]
#[command(about = "BulletOu unified trainer")]
struct Args {
    /// Evaluation function type to train.
    #[arg(long, value_enum)]
    eval_type: EvalType,

    /// Teacher data: either a single file (`.hcpe` / `.hcpe3` / `.pack` /
    /// `.psv`), a directory containing such files (all matching files are
    /// concatenated), or a comma-separated list of either. Format is
    /// inferred from the extension; all included files must share the same
    /// extension.
    #[arg(long)]
    teacher: String,

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
        EvalType::Kppt | EvalType::KppKkpt => run_kppt_all(&args),
        EvalType::KpptKk => run_kppt_kk(&args),
        EvalType::KpptKkp => run_kppt_kkp(&args),
        // KPP trains the same network for both the KPPT and KPP_KKPT layouts;
        // only the writer differs, selected inside `run_training_inline!` via
        // `args.kpp_format()`.
        EvalType::KpptKpp | EvalType::KppKkptKpp => run_kppt_kpp(&args),
    }
}

// ----- KPPT family: KK + KKP + KPP sequential dispatch -------------------

/// Run the three KPPT components (KK, KKP, KPP) sequentially, then assemble
/// the three resulting `.bin` files into `<output>/final/` so the engine has
/// a single directory to point at.
///
/// `--eval-type kppt` uses the KPPT KPP layout (int16 × 2, with turn channel).
/// `--eval-type kpp-kkpt` uses the KPP_KKPT KPP layout (int16, no turn channel).
fn run_kppt_all(args: &Args) {
    let output_dir = args.output_dir();
    let total_superbatches = args.superbatches;
    let last_sb = total_superbatches;

    let kpp_eval_type = match args.eval_type {
        EvalType::Kppt => EvalType::KpptKpp,
        EvalType::KppKkpt => EvalType::KppKkptKpp,
        _ => unreachable!("run_kppt_all called with non-family eval_type"),
    };

    eprintln!("=== bulletou: running {} family (3 components) ===", match args.eval_type {
        EvalType::Kppt => "KPPT",
        EvalType::KppKkpt => "KPP_KKPT",
        _ => unreachable!(),
    });

    for (label, child_eval_type, net_id) in [
        ("KK", EvalType::KpptKk, "kk"),
        ("KKP", EvalType::KpptKkp, "kkp"),
        ("KPP", kpp_eval_type, "kpp"),
    ] {
        eprintln!("\n=== [{label}] training ===");
        let mut child = args.clone();
        child.eval_type = child_eval_type;
        child.net_id = Some(net_id.to_string());
        // Force the child's yaneuraou_quant_scale default to match the child's
        // eval-type when the user didn't override it.
        if args.yaneuraou_quant_scale.is_none() {
            child.yaneuraou_quant_scale = Some(child_eval_type.default_yaneuraou_quant_scale());
        }
        match child_eval_type {
            EvalType::KpptKk => run_kppt_kk(&child),
            EvalType::KpptKkp => run_kppt_kkp(&child),
            EvalType::KpptKpp | EvalType::KppKkptKpp => run_kppt_kpp(&child),
            _ => unreachable!(),
        }
    }

    // Assemble the three .bin files into <output>/final/.
    let final_dir = output_dir.join("final");
    if let Err(e) = std::fs::create_dir_all(&final_dir) {
        eprintln!("error: failed to create {}: {e}", final_dir.display());
        std::process::exit(1);
    }
    let assembly_pairs = [
        ("kk", "KK_synthesized.bin"),
        ("kkp", "KKP_synthesized.bin"),
        ("kpp", "KPP_synthesized.bin"),
    ];
    for (net_id, filename) in assembly_pairs {
        let src = output_dir.join(format!("{net_id}-{last_sb}")).join(filename);
        let dst = final_dir.join(filename);
        if let Err(e) = std::fs::copy(&src, &dst) {
            eprintln!("error: failed to copy {} -> {}: {e}", src.display(), dst.display());
            std::process::exit(1);
        }
        eprintln!("  copied {} -> {}", src.display(), dst.display());
    }

    eprintln!("\n=== done. assembled eval at {} ===", final_dir.display());
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

        let data_files_owned = expand_teacher(&args.teacher).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });
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
            DataFormat::Psv => {
                let loader = DirectSequentialDataLoader::new(&data_files_ref);
                trainer.run(&schedule, &settings, &loader);
            }
        }
    }};
}

// ----- KPPT: KK ---------------------------------------------------------

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

// ----- KPPT: KKP --------------------------------------------------------

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

// ----- KPPT: KPP --------------------------------------------------------

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
