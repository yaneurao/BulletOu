/*!
bulletou — BulletOu trainer entry point.

Dispatches to the appropriate training routine via `--eval-type`. The
"family" eval-types train all three KPPT components (KK + KKP + KPP)
sequentially in a single invocation and assemble the result into
`<output>/final/`:

    bulletou --eval-type KPPT            (KPPT family, KPP int16 × 2)
    bulletou --eval-type KPP_KKPT        (KPP_KKPT factorised, KPP int16)

For NNUE eval types, the architecture is selected with `--arch`. Each
save produces a YaneuraOu / Stockfish nnue-pytorch-compatible `nn.bin`:

    bulletou --eval-type NNUE_HALFKP                    classic HalfKP NNUE (default --arch 256x2-32-32)
    bulletou --eval-type NNUE_HALFKP --arch 1024x2-8-64 larger HalfKP NNUE
    bulletou --eval-type NNUE_KP                        K+P NNUE (default --arch 256x2-32-32)
    bulletou --eval-type NNUE_HALFKPE9                  HalfKP with per-square effect-count buckets

Supported `--arch` presets (matching the per-arch directories under
YaneuraOu's NNUE engine binary distribution):

    256x2-32-32   384x2-8-96   512x2-8-64
    768x2-16-64   1024x2-8-32  1024x2-8-64

(YaneuraOu's KPPT engine requires all three of `KK_synthesized.bin` /
`KKP_synthesized.bin` / `KPP_synthesized.bin` to load an eval, so the
single-component trainers are internal helpers driven by `KPPT` /
`KPP_KKPT` rather than CLI options.)

Teacher data is given via `--teacher`. The argument is either a single
file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), a directory containing such
files (all matching files are concatenated), or a comma-separated list
of either. Format is inferred from the file extension; all files must
share the same extension.

Usage:

    # Build once
    cargo build --release --features device-cuda --example bulletou

    # Then run
    ./target/release/examples/bulletou \
        --eval-type KPPT \
        --teacher /data/shogi/train_set/ \
        --output checkpoints/my-kppt \
        --superbatches 20
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::{ShogiHalfKP, ShogiHalfKpe9, ShogiKk, ShogiKkp, ShogiKp, ShogiKpp, SparseInputType},
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
        nnue_save::{
            Activation as NnueActivation, NnueFeatureSet, ft_hash_bytes, header_bytes,
            l1_bias_scale, network_layer_hash_bytes, pad_weights_for_simd,
        },
        yaneuraou_kppt::{
            KppFormat, bundle_component_state, parse_model_weights_bin, save_yaneuraou_eval,
            unbundle_component_state,
        },
    },
};
use clap::{Parser, ValueEnum};

// ----- eval-type ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "SCREAMING_SNAKE_CASE")]
enum EvalType {
    /// KPPT family: train KK, KKP, and KPP sequentially and assemble the
    /// three-file KPPT eval (`KK_synthesized.bin` / `KKP_synthesized.bin` /
    /// `KPP_synthesized.bin`) per save. The engine requires all three
    /// files together; single-component sub-trainings are not exposed as
    /// CLI options (they exist as internal helpers driven by `KPPT`).
    Kppt,
    /// KPP_KKPT family (factorised KPPT): same as `KPPT` but KPP is written
    /// in the KPP_KKPT layout (no turn channel; half the KPP file size).
    KppKkpt,
    /// NNUE HalfKP. Original YaneuraOu halfkp_256x2-32-32 (Nasu-san PR #75,
    /// 2018): dual-perspective HalfKP feature transformer + 4-layer
    /// ClippedReLU network. Writes a YaneuraOu / Stockfish nnue-pytorch
    /// compatible `nn.bin` per save. Architecture is selected via `--arch`
    /// (default `256x2-32-32`).
    NnueHalfkp,
    /// NNUE K-P. YaneuraOu kp_256x2-32-32 — same 4-layer ClippedReLU network
    /// as halfkp_256x2-32-32, but the input is `FeatureSet<K, P>` (K = 162
    /// king features, P = 1548 piece features per perspective; 1710 total)
    /// instead of HalfKP's (king × piece) cross product. Architecture is
    /// selected via `--arch` (default `256x2-32-32`).
    NnueKp,
    /// NNUE HalfKPE9. YaneuraOu halfkpe9_* — HalfKP × 9 effect-count buckets
    /// (`per-square own/opponent attacker count, 0/1/2 clipped, 3×3=9
    /// combinations`). Input dim is 1,128,492 per perspective (= HalfKP ×
    /// 9). Same 4-layer ClippedReLU network as halfkp / kp. Requires
    /// piece-effect computation, which BulletOu's threat module already
    /// provides.
    NnueHalfkpe9,
}

/// Pre-set NNUE architecture sizes — `<L1>x2-<L2>-<L3>` in the textual CLI
/// form. The set matches the per-arch directories YaneuraOu ships its NNUE
/// engine binaries under (`NNUE_halfkp_256x2_32_32`, `NNUE_halfkp_512x2_8_64`,
/// …): network structure is fixed (4-layer ClippedReLU, dual-perspective);
/// only `(L1, L2, L3)` vary. The same arch presets are usable from both
/// `--eval-type NNUE_HALFKP` and `--eval-type NNUE_KP`; YaneuraOu's KP build
/// currently only ships `256x2_32_32`, but the trainer doesn't restrict you.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum NnueArch {
    /// L1=256, L2=32, L3=32 (classic Stockfish-style NNUE preset).
    #[clap(name = "256x2-32-32")]
    Arch256x2_32_32,
    /// L1=384, L2=8, L3=96.
    #[clap(name = "384x2-8-96")]
    Arch384x2_8_96,
    /// L1=512, L2=8, L3=64.
    #[clap(name = "512x2-8-64")]
    Arch512x2_8_64,
    /// L1=768, L2=16, L3=64.
    #[clap(name = "768x2-16-64")]
    Arch768x2_16_64,
    /// L1=1024, L2=8, L3=32.
    #[clap(name = "1024x2-8-32")]
    Arch1024x2_8_32,
    /// L1=1024, L2=8, L3=64.
    #[clap(name = "1024x2-8-64")]
    Arch1024x2_8_64,
}

impl NnueArch {
    /// `(l1, l2, l3)` triple.
    fn dims(self) -> (usize, usize, usize) {
        match self {
            NnueArch::Arch256x2_32_32 => (256, 32, 32),
            NnueArch::Arch384x2_8_96 => (384, 8, 96),
            NnueArch::Arch512x2_8_64 => (512, 8, 64),
            NnueArch::Arch768x2_16_64 => (768, 16, 64),
            NnueArch::Arch1024x2_8_32 => (1024, 8, 32),
            NnueArch::Arch1024x2_8_64 => (1024, 8, 64),
        }
    }

    /// The arch's CLI value as the user types it (e.g. `256x2-32-32`).
    /// Must stay in sync with the `#[clap(name = ...)]` attribute on each
    /// variant of [`NnueArch`].
    fn cli_name(self) -> &'static str {
        match self {
            NnueArch::Arch256x2_32_32 => "256x2-32-32",
            NnueArch::Arch384x2_8_96 => "384x2-8-96",
            NnueArch::Arch512x2_8_64 => "512x2-8-64",
            NnueArch::Arch768x2_16_64 => "768x2-16-64",
            NnueArch::Arch1024x2_8_32 => "1024x2-8-32",
            NnueArch::Arch1024x2_8_64 => "1024x2-8-64",
        }
    }
}

impl EvalType {
    fn default_net_id(self) -> &'static str {
        match self {
            EvalType::Kppt => "shogi_kppt",
            EvalType::KppKkpt => "shogi_kpp_kkpt",
            EvalType::NnueHalfkp => "shogi_nnue_halfkp",
            EvalType::NnueKp => "shogi_nnue_kp",
            EvalType::NnueHalfkpe9 => "shogi_nnue_halfkpe9",
        }
    }

    /// Does this eval type actually consume `--arch`? KPPT family eval
    /// types have a fixed architecture and ignore `--arch`; NNUE eval
    /// types use it.
    fn uses_arch(self) -> bool {
        match self {
            EvalType::Kppt | EvalType::KppKkpt => false,
            EvalType::NnueHalfkp | EvalType::NnueKp | EvalType::NnueHalfkpe9 => true,
        }
    }

    /// The eval-type's CLI value as the user typed it (e.g. `NNUE_HALFKP`).
    /// Used to derive the default `--output` directory name. Must stay in
    /// sync with the `#[clap(rename_all = "SCREAMING_SNAKE_CASE")]`
    /// attribute on [`EvalType`].
    fn cli_name(self) -> &'static str {
        match self {
            EvalType::Kppt => "KPPT",
            EvalType::KppKkpt => "KPP_KKPT",
            EvalType::NnueHalfkp => "NNUE_HALFKP",
            EvalType::NnueKp => "NNUE_KP",
            EvalType::NnueHalfkpe9 => "NNUE_HALFKPE9",
        }
    }

    /// On-disk KPP layout to write at checkpoint time. KK / KKP / NNUE eval
    /// types don't have a KPP file so this is ignored.
    fn kpp_format(self) -> KppFormat {
        match self {
            EvalType::KppKkpt => KppFormat::KppKkpt,
            _ => KppFormat::Kppt,
        }
    }
}

/// Default `--yaneuraou-quant-scale` for each KPPT component. Used inside
/// [`run_kppt_all`] to inject the component-appropriate scale into the
/// child `Args` before dispatching to [`run_kppt_kk`] / [`run_kppt_kkp`] /
/// [`run_kppt_kpp`]. The values exist as constants rather than as methods
/// on `EvalType` because the public CLI no longer exposes per-component
/// eval types.
///
/// - KK / KKP entries are i32 (large dynamic range) so 4000 = eval_scale * 10.
/// - KPP entries are i16 (smaller dynamic range) so the scale is an order
///   of magnitude smaller.
const KPPT_KK_DEFAULT_QUANT_SCALE: f32 = 4000.0;
const KPPT_KKP_DEFAULT_QUANT_SCALE: f32 = 4000.0;
const KPPT_KPP_DEFAULT_QUANT_SCALE: f32 = 400.0;

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

    /// Cap on the number of superbatches per epoch. If omitted, there is no
    /// cap (= run until the dataloader reaches EOF). Specify this to stop
    /// each epoch early (e.g. to fit a quick smoke test). Mutually exclusive
    /// with `--max-epochs` in practical use.
    #[arg(long)]
    superbatches: Option<usize>,

    /// Number of epochs to train. One epoch = one full pass through the
    /// teacher data (= one dataloader EOF). After each epoch the dataloader
    /// is rebuilt from scratch and the LR scheduler restarts at superbatch
    /// 1, so for example `--lr-step 8` applies independently within each
    /// epoch. Default 1.
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

    /// Lambda — weight on the teacher's evaluation score (vs the actual
    /// game result) in the loss target. Matches YaneuraOu's built-in
    /// trainer convention:
    ///
    ///     target = lambda * eval_score + (1 - lambda) * game_result
    ///
    /// where `eval_score` is the teacher engine's score after sigmoid
    /// and `game_result` is W/D/L = 1.0 / 0.5 / 0.0 from side-to-move
    /// perspective. So `lambda = 1.0` trains on pure eval, `lambda = 0.0`
    /// trains on pure W/D/L, and intermediate values mix the two.
    /// Default 1.0 (pure eval) matches YaneuraOu's traditional default.
    ///
    /// (WDL = Win/Draw/Loss is the three-valued game-result label each
    /// teacher position carries alongside the eval score.)
    #[arg(long, default_value = "1.0")]
    lambda: f32,

    /// Eval-to-score sigmoid scale.
    #[arg(long, default_value = "400")]
    scale: u32,

    /// f32 -> integer quantisation scale for the YaneuraOu KPPT output.
    /// If omitted, per-component defaults are used (4000 for KK/KKP, 400
    /// for KPP). Ignored by NNUE eval types.
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

    /// Network architecture preset for NNUE eval types. Format
    /// `<L1>x2-<L2>-<L3>`. Supported values mirror the per-arch
    /// directories under YaneuraOu's NNUE binary distribution
    /// (`256x2-32-32`, `384x2-8-96`, `512x2-8-64`, `768x2-16-64`,
    /// `1024x2-8-32`, `1024x2-8-64`). Ignored for KPPT / KPP_KKPT
    /// eval types.
    #[arg(long, default_value = "256x2-32-32")]
    arch: NnueArch,
}

impl Args {
    /// Resolve the checkpoint output directory.
    ///
    /// - `--output PATH` honours the user's choice as-is.
    /// - Otherwise the default is `checkpoints/<eval-type>-<arch>` for eval
    ///   types that consume `--arch` (the NNUE family), and
    ///   `checkpoints/<eval-type>` for the KPPT family (which doesn't).
    ///
    /// `<eval-type>` and `<arch>` are the literal CLI values the user
    /// would type (e.g. `NNUE_HALFKP`, `256x2-32-32`) so the dir name
    /// stays in sync with the flags.
    fn output_dir(&self) -> PathBuf {
        if let Some(p) = &self.output {
            return p.clone();
        }
        let mut path = PathBuf::from("checkpoints");
        if self.eval_type.uses_arch() {
            path.push(format!("{}-{}", self.eval_type.cli_name(), self.arch.cli_name()));
        } else {
            path.push(self.eval_type.cli_name());
        }
        path
    }

    fn net_id(&self) -> String {
        self.net_id.clone().unwrap_or_else(|| self.eval_type.default_net_id().to_string())
    }

    /// YaneuraOu integer-quantisation scale to multiply into f32 weights at
    /// save time. The KPPT components have different defaults
    /// (`KPPT_KK_DEFAULT_QUANT_SCALE` etc.); `run_kppt_all` injects the
    /// right value into each child Args before calling
    /// `run_kppt_kk` / `run_kppt_kkp` / `run_kppt_kpp`. By the time this is
    /// read inside `run_training_inline!`, `yaneuraou_quant_scale` is
    /// always populated (either by the user via the CLI flag or by the
    /// parent run helper).
    fn yaneuraou_scale(&self) -> f32 {
        self.yaneuraou_quant_scale
            .expect("yaneuraou_quant_scale must be set before invoking the KPPT trainer")
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
        EvalType::NnueHalfkp => run_halfkp(&args),
        EvalType::NnueKp => run_kp(&args),
        EvalType::NnueHalfkpe9 => run_halfkpe9(&args),
    }
}

/// Count numbered subdirectories under `output_dir` whose names parse as
/// `usize`. Used so a resumed run extends the numbering rather than
/// overwriting the previous run's checkpoint dirs.
fn count_existing_numbered_dirs(output_dir: &std::path::Path) -> usize {
    let Ok(rd) = std::fs::read_dir(output_dir) else { return 0 };
    rd.flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| name.parse::<usize>().is_ok())
        .count()
}

/// Find the latest numbered subdirectory under `output_dir` (4-or-more-digit
/// name parsable as `usize`) whose `state.bin` exists. Returns `None` if no
/// resumable checkpoint is found.
fn find_latest_state_bin(output_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut latest: Option<(usize, std::path::PathBuf)> = None;
    let rd = std::fs::read_dir(output_dir).ok()?;
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let Ok(n) = name.parse::<usize>() else { continue };
        let state_bin = path.join("state.bin");
        if !state_bin.is_file() {
            continue;
        }
        match &latest {
            None => latest = Some((n, state_bin)),
            Some((m, _)) if n > *m => latest = Some((n, state_bin)),
            _ => {}
        }
    }
    latest.map(|(_, p)| p)
}

// ----- KPPT family: KK + KKP + KPP sequential dispatch -------------------

/// Run the three KPPT components (KK, KKP, KPP) sequentially, then assemble
/// the three resulting `.bin` files into `<output>/final/` so the engine has
/// a single directory to point at.
///
/// `--eval-type KPPT` uses the KPPT KPP layout (int16 × 2, with turn channel).
/// `--eval-type KPP_KKPT` uses the KPP_KKPT KPP layout (int16, no turn channel).
fn run_kppt_all(args: &Args) {
    let output_dir = args.output_dir();

    eprintln!("=== bulletou: running {} family (3 components) ===", args.eval_type.cli_name());

    // ---- Resume support -------------------------------------------------
    // If `<output>` already contains a numbered dir with a `state.bin`,
    // unbundle each component's records into a per-component
    // `optimiser_state/` triplet under `<output>/.bulletou_resume/<comp>/`,
    // and let each child run_kppt_* call `trainer.load_from_checkpoint(<comp>)`
    // immediately after building its trainer.
    let resume_state_bin = find_latest_state_bin(&output_dir);
    let resume_dirs: Option<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> =
        resume_state_bin.as_ref().map(|state_bin_path| {
            eprintln!("=== resume detected: {} ===", state_bin_path.display());
            let bytes = std::fs::read(state_bin_path).unwrap_or_else(|e| {
                eprintln!("error: failed to read {}: {e}", state_bin_path.display());
                std::process::exit(1);
            });
            let records = parse_model_weights_bin(&bytes).unwrap_or_else(|e| {
                eprintln!("error: failed to parse state.bin: {e}");
                std::process::exit(1);
            });
            let resume_root = output_dir.join(".bulletou_resume");
            // Fresh extraction each run; old contents may correspond to a
            // different save point.
            let _ = std::fs::remove_dir_all(&resume_root);
            let mut paths: Vec<std::path::PathBuf> = Vec::new();
            for comp in ["kk", "kkp", "kpp"] {
                let comp_dir = resume_root.join(comp);
                unbundle_component_state(&records, comp, &comp_dir.join("optimiser_state")).unwrap_or_else(
                    |e| {
                        eprintln!("error: state.bin missing `{comp}/*` records: {e}");
                        std::process::exit(1);
                    },
                );
                paths.push(comp_dir);
            }
            (paths[0].clone(), paths[1].clone(), paths[2].clone())
        });

    // Each component gets its own child Args with the right net_id +
    // component-specific yaneuraou_quant_scale default (user override via
    // `--yaneuraou-quant-scale` is preserved). The parent's `args.eval_type`
    // (KPPT or KPP_KKPT) flows through unchanged so `args.kpp_format()`
    // inside `run_kppt_kpp` selects the right on-disk KPP layout.
    let make_child = |net_id: &str, default_quant_scale: f32| -> Args {
        let mut child = args.clone();
        child.net_id = Some(net_id.to_string());
        if child.yaneuraou_quant_scale.is_none() {
            child.yaneuraou_quant_scale = Some(default_quant_scale);
        }
        child
    };

    eprintln!("\n=== [KK] training ===");
    let child_kk = make_child("kk", KPPT_KK_DEFAULT_QUANT_SCALE);
    run_kppt_kk(&child_kk, resume_dirs.as_ref().map(|d| d.0.as_path()));

    eprintln!("\n=== [KKP] training ===");
    let child_kkp = make_child("kkp", KPPT_KKP_DEFAULT_QUANT_SCALE);
    run_kppt_kkp(&child_kkp, resume_dirs.as_ref().map(|d| d.1.as_path()));

    eprintln!("\n=== [KPP] training ===");
    let child_kpp = make_child("kpp", KPPT_KPP_DEFAULT_QUANT_SCALE);
    run_kppt_kpp(&child_kpp, resume_dirs.as_ref().map(|d| d.2.as_path()));

    // Cleanup the scratch resume dir if it was used.
    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    // Re-organise per-component checkpoint subdirs into a flat, zero-padded
    // series `0001/`, `0002/`, ..., each containing the three `.bin` files
    // at the corresponding save point. The original `kk-*/` / `kkp-*/` /
    // `kpp-*/` subdirs are removed after assembly.
    let ctx = LogContext::from_args(args);
    let prior_positions = read_prior_positions(&output_dir.join("learn.log"));
    match assemble_numbered_dirs(&output_dir, &ctx, &prior_positions) {
        Ok((_first_idx, last_idx)) => {
            // Append the new run's full loss history to a top-level
            // `<output>/learn.log` so the user has a single growing file
            // spanning all resumes. Per-save `<output>/0NNN/learn.log` files
            // are kept as snapshots.
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!(
                    "warning: failed to update {}: {e}",
                    output_dir.join("learn.log").display()
                );
            }
        }
        Err(e) => {
            eprintln!("error: failed to assemble numbered checkpoint dirs: {e}");
            std::process::exit(1);
        }
    }
}

/// CSV header for `learn.log`. Both the top-level `<output>/learn.log` and
/// each per-save `0NNN/learn.log` start with this line followed by data
/// rows. Column meanings (9 total):
///
/// - `eval`: mirror of the output-dir name (`<eval-type>[-<arch>]`)
///   plus a `/<component>` suffix for multi-component eval types. For
///   NNUE eval types (single-component) the column holds the eval-type
///   joined with the architecture, e.g. `NNUE_HALFKP-256x2-32-32`. For
///   KPPT-family eval types (which ignore `--arch`, three components
///   trained sequentially) it holds `KPPT/kk`, `KPPT/kkp`, `KPPT/kpp`
///   (or `KPP_KKPT/kk`, etc.).
/// - `epoch`: 1-indexed epoch counter within this run (`--max-epochs`).
/// - `superbatch`: 1-indexed superbatch within the current epoch.
///   Increments every `--batches-per-superbatch` batches (default 6104).
/// - `curr_batch`: 1-indexed batch counter within the current superbatch
///   (= the `curr_batch` field bullet records every 32 batches: 32, 64,
///   96, ...). Combine with `superbatch` for "(superbatch − 1) ×
///   batches_per_superbatch + curr_batch" to get the total batch count.
/// - `value_loss`: bullet's per-32-batch loss value at that point.
/// - `lr`: learning rate at that superbatch (StepLR-derived).
/// - `lambda`: `--lambda` value at that point (constant per run), formatted
///   to three decimal places (`1.000`, `0.500`, ...).
/// - `positions`: cumulative number of teacher positions consumed so far
///   for this component, including positions from prior runs detected
///   in the existing top-level `learn.log` (resume-aware). Within a run,
///   the value resets at epoch boundaries when `--max-epochs > 1` — a
///   known v1 limitation.
/// - `teacher`: the user's `--teacher` CLI value verbatim, RFC-4180
///   escaped (quoted if it contains a comma / quote / newline) so a
///   directory or comma-separated list is preserved as one CSV field.
const LEARN_LOG_HEADER: &str =
    "eval,epoch,superbatch,curr_batch,value_loss,lr,lambda,positions,teacher";

/// Bundle of parameters the enrichment functions need to turn bullet's
/// raw 3-column `log.txt` rows (`superbatch,curr_batch,loss`) into the
/// 9-column `learn.log` CSV rows defined by [`LEARN_LOG_HEADER`].
#[derive(Clone, Debug)]
struct LogContext {
    eval_type: &'static str,
    /// Arch suffix (`256x2-32-32` etc.) for NNUE eval types. Empty string for
    /// KPPT-family eval types since they ignore `--arch`. When non-empty it is
    /// joined into the `eval` column as `<eval-type>-<arch>`, matching the
    /// output-dir naming.
    arch: &'static str,
    lr_start: f32,
    lr_gamma: f32,
    lr_step: usize,
    lambda: f32,
    batch_size: usize,
    batches_per_superbatch: usize,
    teacher_csv: String,
}

impl LogContext {
    fn from_args(args: &Args) -> Self {
        let batches_per_superbatch =
            args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));
        Self {
            eval_type: args.eval_type.cli_name(),
            arch: if args.eval_type.uses_arch() { args.arch.cli_name() } else { "" },
            lr_start: args.lr,
            lr_gamma: args.lr_gamma,
            lr_step: args.lr_step,
            lambda: args.lambda,
            batch_size: args.batch_size,
            batches_per_superbatch,
            teacher_csv: csv_escape(&args.teacher),
        }
    }

    /// LR at a given superbatch — mirrors `bullet_lib::trainer::schedule::lr::StepLR`.
    fn lr_at(&self, superbatch: usize) -> f32 {
        let steps = superbatch.saturating_sub(1) / self.lr_step;
        self.lr_start * self.lr_gamma.powi(steps as i32)
    }

    /// Cumulative teacher positions consumed up to `(superbatch, curr_batch)`
    /// within the current epoch, plus the `position_offset` carried over
    /// from prior runs (read from the existing top-level `learn.log`).
    fn positions_at(&self, superbatch: usize, curr_batch: usize, position_offset: usize) -> usize {
        position_offset
            + (superbatch.saturating_sub(1) * self.batches_per_superbatch + curr_batch) * self.batch_size
    }
}

/// RFC 4180-ish CSV escape: wrap in double quotes and double inner quotes if
/// the value contains a comma, a double quote, or a newline. Otherwise pass
/// through unchanged. Used to keep the trailing `teacher` column parseable
/// when the user passed a comma-separated list to `--teacher`.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for ch in s.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
        out
    } else {
        s.to_string()
    }
}

/// Convert bullet's raw 3-column `log.txt` text (`superbatch,curr_batch,loss`
/// per line) into the enriched 9-column CSV body (no header). The header
/// (= [`LEARN_LOG_HEADER`]) is the caller's responsibility, so the same
/// body can be concatenated under a single header by `assemble_numbered_dirs`.
fn enrich_bullet_log_to_csv(
    raw: &str,
    ctx: &LogContext,
    epoch: usize,
    component: &str,
    position_offset: usize,
) -> String {
    let mut out = String::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() != 3 {
            continue;
        }
        let Ok(sb) = parts[0].parse::<usize>() else { continue };
        let Ok(b) = parts[1].parse::<usize>() else { continue };
        let loss = parts[2];
        let lr = ctx.lr_at(sb);
        let positions = ctx.positions_at(sb, b, position_offset);
        // Mirror the output-dir name (`<eval-type>[-<arch>]`) plus a
        // `/<component>` suffix for multi-component eval types (KPPT
        // family). NNUE rows are single-component so the slash is
        // omitted; KPPT-family eval types don't consume `--arch`.
        let head: std::borrow::Cow<'_, str> = if ctx.arch.is_empty() {
            std::borrow::Cow::Borrowed(ctx.eval_type)
        } else {
            std::borrow::Cow::Owned(format!("{}-{}", ctx.eval_type, ctx.arch))
        };
        let eval_field: std::borrow::Cow<'_, str> = if component == "nnue" {
            head
        } else {
            std::borrow::Cow::Owned(format!("{}/{}", head, component))
        };
        out.push_str(&format!(
            "{eval},{epoch},{sb},{b},{loss},{lr},{lambda:.3},{positions},{teacher}\n",
            eval = eval_field,
            lambda = ctx.lambda,
            teacher = ctx.teacher_csv,
        ));
    }
    out
}

/// Read the existing top-level `<output>/learn.log` and return the maximum
/// `positions` value seen per component. Used at the start of a run to
/// pick up the cumulative offset across resumes.
///
/// Returns an empty map if the file doesn't exist yet (= first run).
///
/// The parser uses `splitn(9, ',')` so any commas inside the trailing
/// `teacher` field (e.g. a comma-separated teacher list) don't disturb
/// the first 8 columns. Component is extracted from the `eval` column at
/// index 0: a slash-suffix (e.g. `KPPT/kk`) names the component
/// explicitly; absence of a slash means a single-component NNUE eval
/// type, which maps to the `"nnue"` component key. The `positions`
/// column is at index 7 in the 9-column layout.
fn read_prior_positions(top_level_log: &std::path::Path) -> std::collections::BTreeMap<String, usize> {
    let mut map = std::collections::BTreeMap::new();
    let Ok(content) = std::fs::read_to_string(top_level_log) else { return map };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("eval,") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(9, ',').collect();
        if parts.len() < 8 {
            continue;
        }
        let eval = parts[0];
        let component = eval.split_once('/').map(|(_, c)| c).unwrap_or("nnue");
        let Ok(positions) = parts[7].parse::<usize>() else { continue };
        let entry = map.entry(component.to_string()).or_insert(0);
        if positions > *entry {
            *entry = positions;
        }
    }
    map
}

/// Append the body of the latest save dir's `learn.log` (already enriched
/// 7-column CSV from `assemble_numbered_dirs` / `finalize_nnue_dirs`) onto
/// the top-level `<output>/learn.log`, writing the CSV header on first
/// file creation. The result is a single pure CSV — no section headers,
/// no separators — that pandas / Excel can load directly.
fn append_to_top_level_log(output_dir: &std::path::Path, last_idx: usize) -> std::io::Result<()> {
    use std::io::Write;
    let latest_log = output_dir.join(format!("{last_idx:04}")).join("learn.log");
    let body = std::fs::read_to_string(&latest_log)?;
    let top = output_dir.join("learn.log");
    let top_existed = top.is_file();
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&top)?;
    let body_no_header = body
        .strip_prefix(LEARN_LOG_HEADER)
        .and_then(|rest| rest.strip_prefix('\n').or(Some(rest)))
        .unwrap_or(body.as_str());
    if !top_existed {
        // first time: write the header before the first data block
        writeln!(file, "{LEARN_LOG_HEADER}")?;
    }
    file.write_all(body_no_header.as_bytes())?;
    if !body_no_header.ends_with('\n') {
        file.write_all(b"\n")?;
    }
    Ok(())
}

/// List checkpoint subdirs for `net_id_prefix` under `output_dir`, sorted by
/// `(epoch, sb)`. Subdir names are `<prefix>-<sb>` (single-epoch) or
/// `<prefix>-e<epoch>-<sb>` (multi-epoch). Each returned tuple is
/// `(epoch, sb, path)` so the caller knows which epoch/superbatch each
/// dir corresponds to when enriching its `log.txt` into `learn.log`.
fn list_component_checkpoints_sorted(
    output_dir: &std::path::Path,
    net_id_prefix: &str,
) -> Vec<(usize, usize, std::path::PathBuf)> {
    let mut entries: Vec<(usize, usize, std::path::PathBuf)> = Vec::new();
    let prefix = format!("{net_id_prefix}-");
    let Ok(rd) = std::fs::read_dir(output_dir) else { return Vec::new() };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        let parsed: Option<(usize, usize)> = (|| {
            if let Some(after_e) = rest.strip_prefix('e') {
                let (e_str, sb_str) = after_e.split_once('-')?;
                Some((e_str.parse().ok()?, sb_str.parse().ok()?))
            } else {
                rest.parse::<usize>().ok().map(|sb| (1, sb))
            }
        })();
        let Some((epoch, sb)) = parsed else { continue };
        entries.push((epoch, sb, path));
    }
    entries.sort();
    entries
}

/// Walk the per-component checkpoint subdirs (`kk-*` / `kkp-*` / `kpp-*`)
/// produced by the three children of `run_kppt_all`, and assemble them into
/// flat `<output>/0001/`, `0002/`, ... directories each containing the
/// three `.bin` files. Removes the per-component subdirs after assembly so
/// the user sees a clean numbered layout.
///
/// Returns `(first_idx, last_idx)` of the numbered dirs written in this run
/// (1-based, inclusive). On resume the range starts above the previously
/// existing count, so the caller can locate the latest dir to inspect.
fn assemble_numbered_dirs(
    output_dir: &std::path::Path,
    ctx: &LogContext,
    prior_positions: &std::collections::BTreeMap<String, usize>,
) -> std::io::Result<(usize, usize)> {
    let kk_dirs = list_component_checkpoints_sorted(output_dir, "kk");
    let kkp_dirs = list_component_checkpoints_sorted(output_dir, "kkp");
    let kpp_dirs = list_component_checkpoints_sorted(output_dir, "kpp");

    let n = kk_dirs.len().min(kkp_dirs.len()).min(kpp_dirs.len());
    if n == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "no checkpoint subdirs under {} (kk={}, kkp={}, kpp={})",
                output_dir.display(),
                kk_dirs.len(),
                kkp_dirs.len(),
                kpp_dirs.len()
            ),
        ));
    }
    if kk_dirs.len() != n || kkp_dirs.len() != n || kpp_dirs.len() != n {
        eprintln!(
            "  warning: component save counts differ (kk={}, kkp={}, kpp={}); using the common prefix of {n}",
            kk_dirs.len(),
            kkp_dirs.len(),
            kpp_dirs.len()
        );
    }

    // When resuming, do not overwrite the previous run's numbered dirs --
    // start at `existing_count + 1` so new saves extend the series.
    let existing_count = count_existing_numbered_dirs(output_dir);

    let prior_kk = prior_positions.get("kk").copied().unwrap_or(0);
    let prior_kkp = prior_positions.get("kkp").copied().unwrap_or(0);
    let prior_kpp = prior_positions.get("kpp").copied().unwrap_or(0);

    eprintln!(
        "\n=== assembling {n} checkpoint dir(s) under {} (starting at #{}) ===",
        output_dir.display(),
        existing_count + 1
    );
    for i in 0..n {
        let idx = existing_count + i + 1;
        let dst = output_dir.join(format!("{idx:04}"));
        std::fs::create_dir_all(&dst)?;
        let (kk_epoch, _kk_sb, kk_dir) = &kk_dirs[i];
        let (kkp_epoch, _kkp_sb, kkp_dir) = &kkp_dirs[i];
        let (kpp_epoch, _kpp_sb, kpp_dir) = &kpp_dirs[i];
        // engine-facing quantised .bin files
        std::fs::copy(kk_dir.join("KK_synthesized.bin"), dst.join("KK_synthesized.bin"))?;
        std::fs::copy(kkp_dir.join("KKP_synthesized.bin"), dst.join("KKP_synthesized.bin"))?;
        std::fs::copy(kpp_dir.join("KPP_synthesized.bin"), dst.join("KPP_synthesized.bin"))?;
        // bundle the three components' resume state (Adam weights + momentum + velocity)
        // into a single `state.bin` so the dir holds everything needed to resume.
        let mut state_buf: Vec<u8> = Vec::new();
        bundle_component_state(&mut state_buf, "kk", &kk_dir.join("optimiser_state"))?;
        bundle_component_state(&mut state_buf, "kkp", &kkp_dir.join("optimiser_state"))?;
        bundle_component_state(&mut state_buf, "kpp", &kpp_dir.join("optimiser_state"))?;
        std::fs::write(dst.join("state.bin"), &state_buf)?;
        // Each component's bullet `log.txt` is the raw
        // `superbatch,curr_batch,loss` CSV. Enrich each into the 9-column
        // `learn.log` format (header + data rows for kk, then kkp, then
        // kpp). Pure CSV, no separator between components — the
        // `eval` column's `<eval-type>/<component>` suffix distinguishes them.
        let mut log_buf = String::new();
        log_buf.push_str(LEARN_LOG_HEADER);
        log_buf.push('\n');
        for (label, epoch, dir, prior) in [
            ("kk", *kk_epoch, kk_dir, prior_kk),
            ("kkp", *kkp_epoch, kkp_dir, prior_kkp),
            ("kpp", *kpp_epoch, kpp_dir, prior_kpp),
        ] {
            let raw = std::fs::read_to_string(dir.join("log.txt")).unwrap_or_default();
            log_buf.push_str(&enrich_bullet_log_to_csv(&raw, ctx, epoch, label, prior));
        }
        std::fs::write(dst.join("learn.log"), log_buf)?;
        eprintln!("  -> {}/", dst.display());
    }

    // Remove the now-redundant per-component subdirs.
    for (_, _, d) in kk_dirs.iter().chain(kkp_dirs.iter()).chain(kpp_dirs.iter()) {
        if let Err(e) = std::fs::remove_dir_all(d) {
            eprintln!("  warning: failed to remove {}: {e}", d.display());
        }
    }

    Ok((existing_count + 1, existing_count + n))
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

        let data_files_owned = expand_teacher(&args.teacher).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });
        let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

        let format = infer_data_format(&data_files_ref).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });

        // --superbatches が未指定なら epoch ごとに loader EOF まで回す (= usize::MAX で
        // 上限なし、loader 側で EOF が来たら trainer.run が返る)。
        let end_superbatch = args.superbatches.unwrap_or(usize::MAX);

        let net_id_base = args.net_id();
        let output_dir_buf = args.output_dir();
        let yaneuraou_scale = args.yaneuraou_scale();
        let kpp_format = args.kpp_format();
        let max_epochs = args.max_epochs.max(1);

        let output_dir_str = args.output_dir();
        let output_dir = output_dir_str.to_str().unwrap_or("checkpoints");

        // Tracks whether bullet fired the save callback at least once across
        // all epochs. If 教師 is smaller than a single superbatch (or any
        // other reason no superbatch boundary is crossed), bullet writes no
        // checkpoint at all and we'd end up with an empty output dir. After
        // all epochs finish we check this flag and, if no save happened, do
        // a final fallback save so at least the current trainer state is
        // persisted. This is *not* an EOF-triggered save — it fires exactly
        // once per training run and only as a last resort.
        let saved_any = std::cell::Cell::new(false);
        // Remember the last per-epoch net_id we used so the fallback save can
        // reuse the same naming convention (so assembly pairs the dirs by
        // sort order alongside any future numbered checkpoints).
        let mut last_net_id_for_epoch: String = net_id_base.clone();
        // The error_record returned by the most recent `trainer.run` call.
        // bullet writes `log.txt` itself at each save, but if zero saves
        // happened we need to write it ourselves in the fallback path.
        let mut last_error_record: Vec<(usize, usize, f32)> = Vec::new();

        for epoch in 1..=max_epochs {
            if max_epochs > 1 {
                eprintln!("\n=== epoch {epoch} / {max_epochs} ===");
            }

            // checkpoint dir 名は max_epochs=1 のとき従来通り `<net_id>-<superbatch>`、
            // 複数 epoch のときは `<net_id>-e<epoch>-<superbatch>` で重複を避ける。
            let net_id_for_epoch = if max_epochs > 1 {
                format!("{net_id_base}-e{epoch}")
            } else {
                net_id_base.clone()
            };
            last_net_id_for_epoch = net_id_for_epoch.clone();

            let schedule = TrainingSchedule {
                net_id: net_id_for_epoch.clone(),
                eval_scale: args.scale as f32,
                steps: TrainingSteps {
                    batch_size: args.batch_size,
                    batches_per_superbatch,
                    start_superbatch: args.start_superbatch,
                    end_superbatch,
                },
                wdl_scheduler: wdl::ConstantWDL { value: 1.0 - args.lambda },
                lr_scheduler: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
                save_rate: args.save_rate,
            };

            let net_id_for_cb = net_id_for_epoch.clone();
            let output_dir_for_cb = output_dir_buf.clone();
            let saved_any_ref = &saved_any;
            let on_checkpoint_saved = move |superbatch: usize| {
                saved_any_ref.set(true);
                let ckpt_dir = output_dir_for_cb.join(format!("{net_id_for_cb}-{superbatch}"));
                match save_yaneuraou_eval(&ckpt_dir, yaneuraou_scale, kpp_format) {
                    Ok(()) => eprintln!("  also wrote YaneuraOu eval binary in {}", ckpt_dir.display()),
                    Err(e) => {
                        eprintln!("  WARN: failed to write YaneuraOu eval binary in {}: {e}", ckpt_dir.display())
                    }
                }
            };

            let settings = LocalSettings {
                threads: args.threads,
                test_set: None,
                output_directory: output_dir,
                batch_queue_size: args.batch_queue_size,
                on_checkpoint_saved: Some(&on_checkpoint_saved),
            };

            last_error_record = match format {
                DataFormat::Hcpe => {
                    let loader =
                        HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Hcpe3 => {
                    let loader =
                        Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Pack => {
                    let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                        .with_single_epoch(true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Psv => {
                    let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
                    trainer.run(&schedule, &settings, &loader)
                }
            };
        }

        // End-of-training fallback save (see the comment on `saved_any`):
        // executes only when bullet never crossed a superbatch boundary.
        if !saved_any.get() {
            let ckpt_dir = output_dir_buf.join(format!("{last_net_id_for_epoch}-1"));
            eprintln!(
                "  WARN: no superbatch completed during training (教師 < 1 superbatch); writing fallback save to {}",
                ckpt_dir.display()
            );
            let ckpt_dir_str = ckpt_dir.to_str().expect("checkpoint path is utf-8");
            trainer.save_to_checkpoint(ckpt_dir_str);
            // bullet's save loop normally writes `log.txt` itself, but for the
            // fallback path no save ever fired, so write the in-memory loss
            // record (same `superbatch,batch,loss` CSV format) ourselves.
            if let Err(e) = write_loss_csv(&ckpt_dir.join("log.txt"), &last_error_record) {
                eprintln!("  WARN: failed to write log.txt in {}: {e}", ckpt_dir.display());
            }
            match save_yaneuraou_eval(&ckpt_dir, yaneuraou_scale, kpp_format) {
                Ok(()) => eprintln!("  also wrote YaneuraOu eval binary in {}", ckpt_dir.display()),
                Err(e) => eprintln!("  WARN: failed to write YaneuraOu eval binary in {}: {e}", ckpt_dir.display()),
            }
        }
    }};
}

/// Write loss records as CSV (`superbatch,batch,loss`), matching the format
/// bullet writes to `log.txt` at each save. Used by the end-of-training
/// fallback save path.
fn write_loss_csv(path: &std::path::Path, records: &[(usize, usize, f32)]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (sb, b, loss) in records {
        writeln!(file, "{sb},{b},{loss}")?;
    }
    Ok(())
}

// ----- KPPT: KK ---------------------------------------------------------

fn run_kppt_kk(args: &Args, resume_dir: Option<&std::path::Path>) {
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

    if let Some(dir) = resume_dir {
        eprintln!("  [KK] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT: KKP --------------------------------------------------------

fn run_kppt_kkp(args: &Args, resume_dir: Option<&std::path::Path>) {
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

    if let Some(dir) = resume_dir {
        eprintln!("  [KKP] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT: KPP --------------------------------------------------------

fn run_kppt_kpp(args: &Args, resume_dir: Option<&std::path::Path>) {
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

    if let Some(dir) = resume_dir {
        eprintln!("  [KPP] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}

// ----- NNUE HalfKP ------------------------------------------------------

/// Convert a freshly-saved bullet checkpoint dir to the NNUE final layout:
///
/// - bundle `optimiser_state/{weights,momentum,velocity}.bin` into `state.bin`
///   (under the `"nnue"` component label, reusing the helpers originally
///   written for KPPT — they take a `component: &str` so they're generic),
/// - rename `quantised.bin` -> `nn.bin` (the contents are already the
///   YaneuraOu / Stockfish NNUE binary because the trainer's `save_format`
///   includes the version header and component hashes),
/// - delete the bullet-internal artefacts (`raw.bin`, original
///   `quantised.bin`, `optimiser_state/`).
///
/// `log.txt` is left in place; the final assembly step (`finalize_nnue_dirs`)
/// will rename it to `learn.log` alongside the dir's number-rename.
fn convert_save_dir_to_nnue_layout(dir: &std::path::Path) -> std::io::Result<()> {
    let optimiser_state = dir.join("optimiser_state");
    let mut state_buf: Vec<u8> = Vec::new();
    bundle_component_state(&mut state_buf, "nnue", &optimiser_state)?;
    std::fs::write(dir.join("state.bin"), &state_buf)?;

    let quantised = dir.join("quantised.bin");
    let nn = dir.join("nn.bin");
    std::fs::rename(&quantised, &nn)?;

    let _ = std::fs::remove_file(dir.join("raw.bin"));
    let _ = std::fs::remove_dir_all(&optimiser_state);
    Ok(())
}

/// Inline training loop for single-component NNUE eval types. Same shape as
/// `run_training_inline!` (epoch loop, `--max-epochs`, EOF-as-epoch boundary,
/// fallback save when no superbatch completes, in-memory loss record
/// returned by `trainer.run`), but with NNUE-specific save handling:
/// the per-save callback converts each bullet save dir to the
/// `nn.bin` + `state.bin` (+ `log.txt`) layout via
/// `convert_save_dir_to_nnue_layout`.
macro_rules! run_training_inline_nnue {
    ($args:expr, $trainer:expr) => {{
        let args: &Args = $args;
        let trainer = $trainer;

        let batches_per_superbatch =
            args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));

        let data_files_owned = expand_teacher(&args.teacher).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });
        let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

        let format = infer_data_format(&data_files_ref).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });

        let end_superbatch = args.superbatches.unwrap_or(usize::MAX);
        let net_id_base = args.net_id();
        let output_dir_buf = args.output_dir();
        let max_epochs = args.max_epochs.max(1);

        let output_dir_str = args.output_dir();
        let output_dir = output_dir_str.to_str().unwrap_or("checkpoints");

        let saved_any = std::cell::Cell::new(false);
        let mut last_net_id_for_epoch: String = net_id_base.clone();
        let mut last_error_record: Vec<(usize, usize, f32)> = Vec::new();

        for epoch in 1..=max_epochs {
            if max_epochs > 1 {
                eprintln!("\n=== epoch {epoch} / {max_epochs} ===");
            }
            let net_id_for_epoch = if max_epochs > 1 {
                format!("{net_id_base}-e{epoch}")
            } else {
                net_id_base.clone()
            };
            last_net_id_for_epoch = net_id_for_epoch.clone();

            let schedule = TrainingSchedule {
                net_id: net_id_for_epoch.clone(),
                eval_scale: args.scale as f32,
                steps: TrainingSteps {
                    batch_size: args.batch_size,
                    batches_per_superbatch,
                    start_superbatch: args.start_superbatch,
                    end_superbatch,
                },
                wdl_scheduler: wdl::ConstantWDL { value: 1.0 - args.lambda },
                lr_scheduler: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
                save_rate: args.save_rate,
            };

            let net_id_for_cb = net_id_for_epoch.clone();
            let output_dir_for_cb = output_dir_buf.clone();
            let saved_any_ref = &saved_any;
            let on_checkpoint_saved = move |superbatch: usize| {
                saved_any_ref.set(true);
                let ckpt_dir = output_dir_for_cb.join(format!("{net_id_for_cb}-{superbatch}"));
                match convert_save_dir_to_nnue_layout(&ckpt_dir) {
                    Ok(()) => eprintln!("  wrote NNUE nn.bin + state.bin in {}", ckpt_dir.display()),
                    Err(e) => {
                        eprintln!("  WARN: failed to convert save dir {}: {e}", ckpt_dir.display())
                    }
                }
            };

            let settings = LocalSettings {
                threads: args.threads,
                test_set: None,
                output_directory: output_dir,
                batch_queue_size: args.batch_queue_size,
                on_checkpoint_saved: Some(&on_checkpoint_saved),
            };

            last_error_record = match format {
                DataFormat::Hcpe => {
                    let loader =
                        HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Hcpe3 => {
                    let loader =
                        Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Pack => {
                    let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                        .with_single_epoch(true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Psv => {
                    let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
                    trainer.run(&schedule, &settings, &loader)
                }
            };
        }

        if !saved_any.get() {
            let ckpt_dir = output_dir_buf.join(format!("{last_net_id_for_epoch}-1"));
            eprintln!(
                "  WARN: no superbatch completed during training (教師 < 1 superbatch); writing fallback save to {}",
                ckpt_dir.display()
            );
            let ckpt_dir_str = ckpt_dir.to_str().expect("checkpoint path is utf-8");
            trainer.save_to_checkpoint(ckpt_dir_str);
            if let Err(e) = write_loss_csv(&ckpt_dir.join("log.txt"), &last_error_record) {
                eprintln!("  WARN: failed to write log.txt in {}: {e}", ckpt_dir.display());
            }
            match convert_save_dir_to_nnue_layout(&ckpt_dir) {
                Ok(()) => eprintln!("  wrote NNUE nn.bin + state.bin in {}", ckpt_dir.display()),
                Err(e) => eprintln!("  WARN: failed to convert save dir {}: {e}", ckpt_dir.display()),
            }
        }
    }};
}

/// Build the NNUE Standard output `save_format` for the trainer. The
/// returned vector, when consumed by `trainer.save_to_checkpoint` /
/// `save_quantised`, produces a `quantised.bin` that is byte-identical to
/// nnue-pytorch's NNUE file format (which YaneuraOu and Stockfish read).
/// `convert_save_dir_to_nnue_layout` later renames that file to `nn.bin`.
///
/// The layer-stack architecture (L0 -> ClippedReLU -> L1 -> ClippedReLU ->
/// L2 -> ClippedReLU -> Out) is shared across NNUE_HALFKP / NNUE_KP / ...,
/// so only the feature set varies between them. Pass the matching
/// [`NnueFeatureSet`] for the correct header / hash bytes.
fn build_nnue_save_format(
    feature_set: NnueFeatureSet,
    l1_size: usize,
    l2_size: usize,
    l3_size: usize,
) -> Vec<SavedFormat> {
    // Quantisation scales for the original-style NNUE (Nasu-san PR #75,
    // 2018) which uses ClippedReLU throughout:
    // - L0 weights/biases use qa=127 (CReLU output range is 0..127).
    // - L1/L2/Out weights use qb=64 (i8 row-major after .transpose()).
    // SqrClippedReLU (SCReLU) is a later, separate activation added in
    // PR #311 (2026) for SFNNwoPSQT-1536 and is NOT used here.
    let qa: i16 = 127;
    let qb: i16 = 64;

    let l1_input_dim = 2 * l1_size; // dual perspective concat
    let l1_bias = l1_bias_scale(NnueActivation::Crelu, /*pairwise=*/ false, qa, qb);

    vec![
        SavedFormat::custom(header_bytes(feature_set, l1_size, l2_size, l3_size)),
        SavedFormat::custom(ft_hash_bytes(feature_set, l1_size)),
        // L0: biases first, then weights (standard nnue-pytorch order).
        SavedFormat::id("l0b").round().quantise::<i16>(qa),
        SavedFormat::id("l0w").round().quantise::<i16>(qa),
        // Network layer hash (between FT and the FC stack).
        SavedFormat::custom(network_layer_hash_bytes(l1_size, l2_size, l3_size)),
        // L1: bias i32, weights i8 (row-major, 32B-padded). Stockfish /
        // YaneuraOu's SIMD inference pads each layer's input dim to a
        // multiple of 32.
        SavedFormat::id("l1b").round().quantise::<i32>(l1_bias),
        SavedFormat::id("l1w")
            .transpose()
            .transform({
                let out_dim = l2_size;
                let in_dim = l1_input_dim;
                move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
            })
            .round()
            .quantise::<i8>(qb),
        // L2: bias i32, weights i8 (row-major, padded). L2 input scale after
        // crelu_i32_to_u8 is always 127.
        SavedFormat::id("l2b").round().quantise::<i32>(127 * i32::from(qb)),
        SavedFormat::id("l2w")
            .transpose()
            .transform({
                let out_dim = l3_size;
                let in_dim = l2_size;
                move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
            })
            .round()
            .quantise::<i8>(qb),
        // Output: bias i32, weights i8 (row-major, padded).
        SavedFormat::id("outb").round().quantise::<i32>(127 * i32::from(qb)),
        SavedFormat::id("outw")
            .transpose()
            .transform({
                let out_dim = 1;
                let in_dim = l3_size;
                move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
            })
            .round()
            .quantise::<i8>(qb),
    ]
}

/// NNUE HalfKP training entry point.
///
/// 4-layer ClippedReLU network with dual-perspective HalfKP input. L1 /
/// L2 / L3 sizes come from `--arch` (`256x2-32-32`, `384x2-8-96`, …);
/// the layer structure and activation function are fixed across all
/// presets — only the sizes vary, matching YaneuraOu's per-arch
/// `halfkp_*.h` headers.
/// - Dual-perspective HalfKP feature transformer -> L1 (ClippedReLU)
/// - L1 -> L2 (ClippedReLU)
/// - L2 -> L3 (ClippedReLU)
/// - L3 -> 1 (eval scalar)
///
/// Per-save layout (after `convert_save_dir_to_nnue_layout`):
///   `<output>/<net_id>-<sb>/{nn.bin, state.bin, log.txt}`
/// then at end-of-training renamed to `<output>/0NNN/{nn.bin, state.bin, learn.log}`.
fn run_halfkp(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch.dims();
    let input_size = ShogiHalfKP.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_HALFKP ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    // ---- Resume support -------------------------------------------------
    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(&output_dir);
    let resume_dir: Option<std::path::PathBuf> = resume_state_bin.as_ref().map(|state_bin_path| {
        eprintln!("=== resume detected: {} ===", state_bin_path.display());
        let bytes = std::fs::read(state_bin_path).unwrap_or_else(|e| {
            eprintln!("error: failed to read {}: {e}", state_bin_path.display());
            std::process::exit(1);
        });
        let records = parse_model_weights_bin(&bytes).unwrap_or_else(|e| {
            eprintln!("error: failed to parse state.bin: {e}");
            std::process::exit(1);
        });
        let resume_root = output_dir.join(".bulletou_resume");
        let _ = std::fs::remove_dir_all(&resume_root);
        unbundle_component_state(&records, "nnue", &resume_root.join("optimiser_state")).unwrap_or_else(|e| {
            eprintln!("error: state.bin missing `nnue/*` records: {e}");
            std::process::exit(1);
        });
        resume_root
    });

    let save_format = build_nnue_save_format(NnueFeatureSet::HalfKp, l1_size, l2_size, l3_size);

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiHalfKP)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let l0 = builder.new_affine("l0", input_size, l1_size);
        let l1 = builder.new_affine("l1", l1_input_dim, l2_size);
        let l2 = builder.new_affine("l2", l2_size, l3_size);
        let out = builder.new_affine("out", l3_size, 1);

        let stm_hidden = l0.forward(stm_inputs).crelu();
        let ntm_hidden = l0.forward(ntm_inputs).crelu();
        let combined = stm_hidden.concat(ntm_hidden);
        let hidden1 = l1.forward(combined).crelu();
        let hidden2 = l2.forward(hidden1).crelu();
        out.forward(hidden2)
    });

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    // Cleanup the scratch resume dir if it was used.
    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    // Single-component finalisation: rename `<net_id>-*/` to `0NNN/` and
    // enrich each dir's bullet `log.txt` into the 7-column `learn.log`.
    let ctx = LogContext::from_args(args);
    let top_level_log = output_dir.join("learn.log");
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!(
                    "warning: failed to update {}: {e}",
                    output_dir.join("learn.log").display()
                );
            }
        }
        Err(e) => {
            eprintln!("error: failed to finalise NNUE checkpoint dirs: {e}");
            std::process::exit(1);
        }
    }
}

/// NNUE K-P training entry point. Structurally identical to [`run_halfkp`]
/// but uses YaneuraOu's `FeatureSet<K, P>` (`kp_256x2-32-32.h`) as input:
/// K (162 dims, both kings) + P (1548 dims, non-king pieces) = 1710 dims
/// per perspective. The network stack (L0 -> ClippedReLU -> L1 ->
/// ClippedReLU -> L2 -> ClippedReLU -> Out) is the same as halfkp_256x2-32-32.
fn run_kp(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch.dims();
    let input_size = ShogiKp.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_KP ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(&output_dir);
    let resume_dir: Option<std::path::PathBuf> = resume_state_bin.as_ref().map(|state_bin_path| {
        eprintln!("=== resume detected: {} ===", state_bin_path.display());
        let bytes = std::fs::read(state_bin_path).unwrap_or_else(|e| {
            eprintln!("error: failed to read {}: {e}", state_bin_path.display());
            std::process::exit(1);
        });
        let records = parse_model_weights_bin(&bytes).unwrap_or_else(|e| {
            eprintln!("error: failed to parse state.bin: {e}");
            std::process::exit(1);
        });
        let resume_root = output_dir.join(".bulletou_resume");
        let _ = std::fs::remove_dir_all(&resume_root);
        unbundle_component_state(&records, "nnue", &resume_root.join("optimiser_state")).unwrap_or_else(|e| {
            eprintln!("error: state.bin missing `nnue/*` records: {e}");
            std::process::exit(1);
        });
        resume_root
    });

    let save_format = build_nnue_save_format(NnueFeatureSet::Kp, l1_size, l2_size, l3_size);

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKp)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let l0 = builder.new_affine("l0", input_size, l1_size);
        let l1 = builder.new_affine("l1", l1_input_dim, l2_size);
        let l2 = builder.new_affine("l2", l2_size, l3_size);
        let out = builder.new_affine("out", l3_size, 1);

        let stm_hidden = l0.forward(stm_inputs).crelu();
        let ntm_hidden = l0.forward(ntm_inputs).crelu();
        let combined = stm_hidden.concat(ntm_hidden);
        let hidden1 = l1.forward(combined).crelu();
        let hidden2 = l2.forward(hidden1).crelu();
        out.forward(hidden2)
    });

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    let ctx = LogContext::from_args(args);
    let top_level_log = output_dir.join("learn.log");
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!(
                    "warning: failed to update {}: {e}",
                    output_dir.join("learn.log").display()
                );
            }
        }
        Err(e) => {
            eprintln!("error: failed to finalise NNUE checkpoint dirs: {e}");
            std::process::exit(1);
        }
    }
}

/// NNUE HalfKPE9 training entry point. Same 4-layer ClippedReLU network as
/// HalfKP / K-P, but the input is `ShogiHalfKpe9` (1,128,492 dims per
/// perspective = HalfKP × 9 effect-count buckets). The effect-count
/// computation is done once per training position by `ShogiHalfKpe9`'s
/// `map_features` using the threat module's `for_each_attack`.
fn run_halfkpe9(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch.dims();
    let input_size = ShogiHalfKpe9.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_HALFKPE9 ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(&output_dir);
    let resume_dir: Option<std::path::PathBuf> = resume_state_bin.as_ref().map(|state_bin_path| {
        eprintln!("=== resume detected: {} ===", state_bin_path.display());
        let bytes = std::fs::read(state_bin_path).unwrap_or_else(|e| {
            eprintln!("error: failed to read {}: {e}", state_bin_path.display());
            std::process::exit(1);
        });
        let records = parse_model_weights_bin(&bytes).unwrap_or_else(|e| {
            eprintln!("error: failed to parse state.bin: {e}");
            std::process::exit(1);
        });
        let resume_root = output_dir.join(".bulletou_resume");
        let _ = std::fs::remove_dir_all(&resume_root);
        unbundle_component_state(&records, "nnue", &resume_root.join("optimiser_state")).unwrap_or_else(|e| {
            eprintln!("error: state.bin missing `nnue/*` records: {e}");
            std::process::exit(1);
        });
        resume_root
    });

    let save_format = build_nnue_save_format(NnueFeatureSet::HalfKpe9, l1_size, l2_size, l3_size);

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiHalfKpe9)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let l0 = builder.new_affine("l0", input_size, l1_size);
        let l1 = builder.new_affine("l1", l1_input_dim, l2_size);
        let l2 = builder.new_affine("l2", l2_size, l3_size);
        let out = builder.new_affine("out", l3_size, 1);

        let stm_hidden = l0.forward(stm_inputs).crelu();
        let ntm_hidden = l0.forward(ntm_inputs).crelu();
        let combined = stm_hidden.concat(ntm_hidden);
        let hidden1 = l1.forward(combined).crelu();
        let hidden2 = l2.forward(hidden1).crelu();
        out.forward(hidden2)
    });

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    let ctx = LogContext::from_args(args);
    let top_level_log = output_dir.join("learn.log");
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!(
                    "warning: failed to update {}: {e}",
                    output_dir.join("learn.log").display()
                );
            }
        }
        Err(e) => {
            eprintln!("error: failed to finalise NNUE checkpoint dirs: {e}");
            std::process::exit(1);
        }
    }
}

/// Single-component analogue of `assemble_numbered_dirs`: list `<net_id>-*/`
/// (or `<net_id>-e<epoch>-<sb>/` for multi-epoch) under `output_dir`, sort
/// by (epoch, sb), rename them to `0NNN/` starting at `existing_count + 1`,
/// and enrich each dir's bullet-format `log.txt` into the 7-column CSV
/// `learn.log` shared with KPPT.
fn finalize_nnue_dirs(
    output_dir: &std::path::Path,
    ctx: &LogContext,
    net_id_prefix: &str,
    prior_position: usize,
) -> std::io::Result<(usize, usize)> {
    let src_dirs = list_component_checkpoints_sorted(output_dir, net_id_prefix);
    let n = src_dirs.len();
    if n == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "no checkpoint subdirs under {} (prefix `{net_id_prefix}-`)",
                output_dir.display()
            ),
        ));
    }

    let existing_count = count_existing_numbered_dirs(output_dir);

    eprintln!(
        "\n=== finalising {n} NNUE checkpoint dir(s) under {} (starting at #{}) ===",
        output_dir.display(),
        existing_count + 1
    );
    for (i, (epoch, _sb, src)) in src_dirs.iter().enumerate() {
        let idx = existing_count + i + 1;
        let dst = output_dir.join(format!("{idx:04}"));
        std::fs::rename(src, &dst)?;
        // Enrich bullet's `log.txt` (raw 3-col CSV) into 7-col `learn.log`.
        let log_txt = dst.join("log.txt");
        let learn_log = dst.join("learn.log");
        let raw = std::fs::read_to_string(&log_txt).unwrap_or_default();
        let body = enrich_bullet_log_to_csv(&raw, ctx, *epoch, "nnue", prior_position);
        let mut content = String::with_capacity(body.len() + LEARN_LOG_HEADER.len() + 1);
        content.push_str(LEARN_LOG_HEADER);
        content.push('\n');
        content.push_str(&body);
        std::fs::write(&learn_log, content)?;
        let _ = std::fs::remove_file(&log_txt);
        eprintln!("  -> {}/", dst.display());
    }
    Ok((existing_count + 1, existing_count + n))
}
