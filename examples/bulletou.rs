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
    bulletou --eval-type NNUE_KA2 --arch 256x2-64-64    K+A2 NNUE (e.g. wider hidden layers)
    bulletou --eval-type NNUE_HALFKPE9                  HalfKP with per-square effect-count buckets
    bulletou --eval-type NNUE_HALFKPVM                  HalfKP with file-mirror (~half input dims of HalfKP)
    bulletou --eval-type SFNN_HALFKA2HM --arch 1536x2-15-32 --layerstack king3-by-king3
    bulletou --eval-type SFNN_KA2       --arch 1536x2-15-32 --layerstack king3-by-king3
                                                        SFNN-1536 with HalfKA_hm2 + 9 LayerStacks
                                                        (= YaneuraOu YANEURAOU_ENGINE_NNUE_SFNNwoP1536)

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

use bulletou_lib::{
    game::inputs::{
        ShogiHalfKP, ShogiHalfKPvm, ShogiHalfKaHm1, ShogiHalfKaHm2, ShogiHalfKpe9, ShogiKa2, ShogiKk, ShogiKkp,
        ShogiKp, ShogiKpp, SparseInputType,
    },
    game::outputs::ShogiLayerStackBucket9,
    nn::{Affine, InitSettings, Shape, optimiser},
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    trainer::schedule::lr::LrScheduler,
    validate::{compute_sign_accuracy, read_random_hcpe_positions},
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
        nnue_save_sfnn1536::{Sfnn1536SaveParams, build_sfnn_1536_save_format},
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
    /// NNUE K-A2. YaneuraOu `FeatureSet<K, A2>` — same 4-layer ClippedReLU
    /// network as kp_256x2-32-32, but the piece feature is A2 (1629 dims,
    /// kings collapsed onto friend plane via v2 encoding) so both kings
    /// participate in the piece feature in addition to K (162 dims). Input
    /// total = 1791 dims per perspective. Same architecture knob (`--arch`)
    /// as NNUE_KP / NNUE_HALFKP. Matches YaneuraOu's
    /// `YANEURAOU_ENGINE_NNUE_ka2_*` build (single LayerStack, no SFNN
    /// post-FT structure).
    NnueKa2,
    /// NNUE HalfKPE9. YaneuraOu halfkpe9_* — HalfKP × 9 effect-count buckets
    /// (`per-square own/opponent attacker count, 0/1/2 clipped, 3×3=9
    /// combinations`). Input dim is 1,128,492 per perspective (= HalfKP ×
    /// 9). Same 4-layer ClippedReLU network as halfkp / kp. Requires
    /// piece-effect computation, which BulletOu's threat module already
    /// provides.
    NnueHalfkpe9,
    /// NNUE HalfKP_vm. YaneuraOu halfkpvm_* — HalfKP with file-mirror
    /// folding: king positions on files 6-9 are mirrored to files 1-4,
    /// halving the input dimension to 69,660 per perspective (= 45 king
    /// buckets × 1548 piece inputs). Same 4-layer ClippedReLU network as
    /// the rest of the NNUE family.
    NnueHalfkpvm,
    /// SFNN-1536 with `HalfKA_hm1` input (= strict v1, both kings on
    /// separate planes, 76,950 dim). LayerStacks family — uses a 9-stack
    /// MLP (FT → fc_0(L1+1 PSQT-shortcut) → CReLU + SqrCReLU concat →
    /// fc_1 → fc_2 → +PSQT bypass). Bucketing chosen via `--layerstack`.
    /// `--arch 1536x2-15-32` matches YaneuraOu's `sfnnwop-1536.h`; that
    /// is the only preset YaneuraOu currently ships, but other sizes
    /// can be trained for ablation (not engine-loadable).
    SfnnHalfka1hm,
    /// SFNN-1536 with `HalfKA_hm2` input (= strict v2, enemy king
    /// collapsed onto friend plane, 73,305 dim). This is the variant
    /// YaneuraOu's `YANEURAOU_ENGINE_NNUE_SFNNwoP1536` build actually
    /// uses. Identical network topology to `SFNN_HALFKA1HM`, only the
    /// input feature differs.
    SfnnHalfka2hm,
    /// SFNN-1536 with `K + A2` input (= YaneuraOu `FeatureSet<K, A2>`,
    /// 1791 dim). K (162 king features) + A2 (1629 piece features,
    /// kings collapsed onto friend plane). No file-mirror, so input
    /// dimension is much smaller than HalfKA_hm2 but representation
    /// is also weaker (no king-anchor cross product). Matches
    /// YaneuraOu's `YANEURAOU_ENGINE_NNUE_SFNNwoPSQT_ka2_*` build.
    /// Identical network topology and LayerStacks bucketing as the
    /// other SFNN variants.
    SfnnKa2,
}

/// NNUE architecture size — `<L1>x2-<L2>-<L3>` in the textual CLI form.
///
/// Network structure is fixed (4-layer ClippedReLU, dual-perspective for the
/// non-SFNN family; SFNN family uses a fixed sfnnwop-1536-style topology),
/// only `(L1, L2, L3)` vary. Free-form: any positive `(L1, L2, L3)` triple
/// is accepted as long as `L1 % 32 == 0` (SIMD-alignment requirement of
/// the FT padding).
///
/// Common presets that are known to work:
/// - `256x2-32-32`  (classic Stockfish-style NNUE; YaneuraOu KP / HalfKP default)
/// - `512x2-8-64`, `768x2-16-64`, `1024x2-8-64`  (larger HalfKP variants
///   matching YaneuraOu's NNUE binary directories)
/// - `1536x2-15-32` (SFNN-1536, matches `architectures/sfnnwop-1536.h`;
///   L2=15 + 1 PSQT-shortcut neuron is added automatically inside the
///   SFNN trainer)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NnueArch {
    l1: usize,
    l2: usize,
    l3: usize,
}

impl NnueArch {
    /// `(l1, l2, l3)` triple.
    fn dims(self) -> (usize, usize, usize) {
        (self.l1, self.l2, self.l3)
    }

    /// The arch's CLI value as the user types it (e.g. `256x2-32-32`).
    fn cli_name(self) -> String {
        format!("{}x2-{}-{}", self.l1, self.l2, self.l3)
    }
}

impl std::fmt::Display for NnueArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x2-{}-{}", self.l1, self.l2, self.l3)
    }
}

impl std::str::FromStr for NnueArch {
    type Err = String;

    /// Parse `<L1>x2-<L2>-<L3>` (e.g. `256x2-32-32`). The middle `x2`
    /// is required (it stands for the dual-perspective concat) and rejected
    /// otherwise. `L1` must be a positive multiple of 32 (FT SIMD alignment);
    /// `L2`, `L3` must be positive.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return Err(format!(
                "invalid arch `{s}`: expected `<L1>x2-<L2>-<L3>` (e.g. `256x2-32-32`)"
            ));
        }
        let l1_part = parts[0]
            .strip_suffix("x2")
            .ok_or_else(|| format!("invalid arch `{s}`: `{}` must end with `x2`", parts[0]))?;
        let l1: usize = l1_part
            .parse()
            .map_err(|_| format!("invalid arch `{s}`: L1 `{l1_part}` is not a positive integer"))?;
        let l2: usize = parts[1]
            .parse()
            .map_err(|_| format!("invalid arch `{s}`: L2 `{}` is not a positive integer", parts[1]))?;
        let l3: usize = parts[2]
            .parse()
            .map_err(|_| format!("invalid arch `{s}`: L3 `{}` is not a positive integer", parts[2]))?;
        if l1 == 0 || l2 == 0 || l3 == 0 {
            return Err(format!("invalid arch `{s}`: L1/L2/L3 must all be > 0"));
        }
        if l1 % 32 != 0 {
            return Err(format!(
                "invalid arch `{s}`: L1 (= {l1}) must be a multiple of 32 (FT SIMD-padding requirement)"
            ));
        }
        Ok(NnueArch { l1, l2, l3 })
    }
}

/// LayerStack bucketing scheme for the SFNN family. Selects which
/// per-position bucket index is used to choose the active MLP stack
/// from the LayerStacks array, and implicitly determines the **stack
/// count** (the network model uses one bucket per stack).
///
/// Currently `king3-by-king3` is the only choice — it matches YaneuraOu's
/// `stack_index_for_nnue` so the trained `nn.bin` is engine-loadable
/// and evaluation matches between training and inference. Other
/// schemes implemented in `bulletou_lib::game::outputs::ShogiLayerStackBucket9`
/// (e.g. `Ply9`, `Progress8*`) are intentionally not exposed here
/// because they cannot be used with YaneuraOu's engine; they remain
/// available to `examples/shogi_layerstack.rs` for rshogi-style research.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum LayerStackMode {
    /// 3 × 3 = 9 stacks, indexed by `(friend_king_rank/3, enemy_king_rank/3)`.
    /// Matches YaneuraOu `stack_index_for_nnue` byte-for-byte.
    #[default]
    #[clap(name = "king3-by-king3")]
    Kingrank3by3,
}

impl LayerStackMode {
    /// CLI value as the user types it.
    fn cli_name(self) -> &'static str {
        match self {
            LayerStackMode::Kingrank3by3 => "king3-by-king3",
        }
    }

    /// Number of LayerStacks this bucketing scheme produces.
    fn num_stacks(self) -> usize {
        match self {
            LayerStackMode::Kingrank3by3 => 9,
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
            EvalType::NnueKa2 => "shogi_nnue_ka2",
            EvalType::NnueHalfkpe9 => "shogi_nnue_halfkpe9",
            EvalType::NnueHalfkpvm => "shogi_nnue_halfkpvm",
            EvalType::SfnnHalfka1hm => "shogi_sfnn_halfka1hm",
            EvalType::SfnnHalfka2hm => "shogi_sfnn_halfka2hm",
            EvalType::SfnnKa2 => "shogi_sfnn_ka2",
        }
    }

    /// Does this eval type actually consume `--arch`? KPPT family eval
    /// types have a fixed architecture and ignore `--arch`; NNUE / SFNN
    /// eval types use it.
    fn uses_arch(self) -> bool {
        match self {
            EvalType::Kppt | EvalType::KppKkpt => false,
            EvalType::NnueHalfkp
            | EvalType::NnueKp
            | EvalType::NnueKa2
            | EvalType::NnueHalfkpe9
            | EvalType::NnueHalfkpvm
            | EvalType::SfnnHalfka1hm
            | EvalType::SfnnHalfka2hm
            | EvalType::SfnnKa2 => true,
        }
    }

    /// Does this eval type consume `--layerstack`? Only the SFNN family
    /// (LayerStacks-based architectures) does; the rest of the NNUE
    /// family is single-stack.
    fn uses_layerstack(self) -> bool {
        matches!(
            self,
            EvalType::SfnnHalfka1hm | EvalType::SfnnHalfka2hm | EvalType::SfnnKa2
        )
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
            EvalType::NnueKa2 => "NNUE_KA2",
            EvalType::NnueHalfkpe9 => "NNUE_HALFKPE9",
            EvalType::NnueHalfkpvm => "NNUE_HALFKPVM",
            EvalType::SfnnHalfka1hm => "SFNN_HALFKA1HM",
            EvalType::SfnnHalfka2hm => "SFNN_HALFKA2HM",
            EvalType::SfnnKa2 => "SFNN_KA2",
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
//  `bulletou_lib::teacher_path` so the single-component examples can share them.)

/// LR scheduler wrapper with an optional superbatch offset. Used to
/// decouple bullet's dataloader skip-ahead (which uses
/// `start_superbatch` to seek through the teacher data) from the LR
/// schedule (which should track the cumulative training progress
/// across rounds, even when each round uses a fresh teacher file).
///
/// - `Plain(s)`: behaves exactly like `s` — `lr(b, sb)` calls
///   `s.lr(b, sb)`. Used in the no-resume / same-teacher cases where
///   bullet's `start_superbatch` already represents the absolute sb.
/// - `Offset { inner, offset }`: shifts the sb input by `offset` so
///   `lr(b, sb)` returns `inner.lr(b, sb + offset)`. Used in the
///   teacher-changed resume case: `start_superbatch=1` (= dataloader
///   reads the new file from the beginning) plus `offset=last_sb`
///   keeps the LR schedule aligned with the cumulative training step
///   instead of restarting at the initial LR.
#[derive(Clone, Debug)]
enum AdjustableStepLR {
    Plain(lr::StepLR),
    Offset { inner: lr::StepLR, offset: usize },
    /// Positions-based step: drops LR by `gamma` every
    /// `positions_per_step` cumulative positions trained (across
    /// rounds, including `prior_positions` carried over from the
    /// existing top-level learn.log). The current (batch, sb)
    /// arguments only contribute the in-this-run positions; the
    /// scheduler is responsible for combining them with the prior
    /// carry-over to compute the absolute count.
    Positions {
        start: f32,
        gamma: f32,
        positions_per_step: u64,
        prior_positions: u64,
        batch_size: usize,
        batches_per_superbatch: usize,
    },
}

impl AdjustableStepLR {
    /// Compute LR for an absolute position count. Used by both bullet's
    /// `lr(batch, sb)` callback and the enrich path's `learn.log` lr
    /// column so they always agree.
    fn lr_at_positions(start: f32, gamma: f32, step: u64, total: u64) -> f32 {
        if step == 0 {
            // Defensive: a 0-step would divide by zero. Treat as "never drop".
            return start;
        }
        let n = (total / step) as i32;
        start * gamma.powi(n)
    }
}

impl LrScheduler for AdjustableStepLR {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        match self {
            Self::Plain(s) => s.lr(batch, superbatch),
            Self::Offset { inner, offset } => inner.lr(batch, superbatch + offset),
            Self::Positions {
                start,
                gamma,
                positions_per_step,
                prior_positions,
                batch_size,
                batches_per_superbatch,
            } => {
                let in_run = ((superbatch.saturating_sub(1) * batches_per_superbatch + batch) as u64)
                    * (*batch_size as u64);
                let total = prior_positions + in_run;
                Self::lr_at_positions(*start, *gamma, *positions_per_step, total)
            }
        }
    }

    fn colourful(&self) -> String {
        match self {
            Self::Plain(s) => s.colourful(),
            Self::Offset { inner, offset } => format!("{} (sb offset +{})", inner.colourful(), offset),
            Self::Positions {
                start, gamma, positions_per_step, prior_positions, ..
            } => format!(
                "start {start} gamma {gamma} drop every {positions_per_step} positions (cumulative, prior {prior_positions})"
            ),
        }
    }
}

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

    /// Suffix appended to the auto-derived output directory name. Useful
    /// for running multiple experiments with the same network /
    /// architecture but different hyperparameters: each run lands in
    /// its own directory like
    /// `checkpoints/<eval-type>-<arch>[-<layerstack>]-<tag>`.
    /// Ignored when `--output` is set explicitly (the user-provided
    /// path wins).
    #[arg(long)]
    tag: Option<String>,

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
    /// 1, so for example `--lr-step 1` applies independently within each
    /// epoch. Default 1.
    #[arg(long, default_value = "1")]
    max_epochs: usize,

    /// Starting superbatch counter for the LR scheduler and the trainer's
    /// internal superbatch numbering.
    ///
    /// When omitted (the normal case), it auto-resumes from the latest
    /// saved superbatch found under `--output`: the next run continues
    /// the LR schedule from `last_saved_sb + 1` instead of restarting at
    /// 1. If no saved checkpoints exist, training starts from sb 1 as
    /// before.
    ///
    /// Pass an explicit value to override (e.g. `--start-superbatch 1` to
    /// force the LR schedule to restart from the beginning even when
    /// resuming from a state.bin).
    #[arg(long)]
    start_superbatch: Option<usize>,

    /// Initial Adam learning rate.
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// LR gamma (multiplicative drop applied every `lr_step` superbatches).
    /// The default `0.9` (combined with `--lr-step 1`) gives a gentle
    /// per-superbatch decay suited to long-running NNUE training; pass
    /// e.g. `0.1` for the older aggressive 10× drop.
    #[arg(long, default_value = "0.9")]
    lr_gamma: f32,

    /// LR step: apply `lr_gamma` every N superbatches. Ignored when
    /// `--lr-step-positions` is set. Default `1` = decay every
    /// superbatch (with `--lr-gamma 0.9` ⇒ ~10× over 22 sb).
    #[arg(long, default_value = "1")]
    lr_step: usize,

    /// LR step in *positions* (cumulative across rounds) instead of
    /// superbatches. When set, `lr_gamma` is applied every N
    /// teacher positions actually trained, regardless of how many
    /// superbatches that took. Useful for round-per-file workflows
    /// where each round may train fewer than 1 superbatch's worth
    /// of positions — the sb-based `--lr-step` would step too
    /// quickly because each round increments the sb counter even
    /// when the round was partial.
    ///
    /// Example: `--lr-step-positions 800000000` drops LR every 800M
    /// positions. With save-rate=1 and a full 100M-positions
    /// superbatch, this matches `--lr-step 8` (8 × 100M = 800M).
    /// With 60M-positions teachers (= round-per-file), LR drops
    /// after ~13 rounds (= 800M / 60M).
    ///
    /// When omitted (default), the sb-based `--lr-step` is used.
    #[arg(long)]
    lr_step_positions: Option<u64>,

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

    /// Network architecture for NNUE eval types. Free-form format
    /// `<L1>x2-<L2>-<L3>` where `L1` is the per-perspective FT size
    /// (must be a multiple of 32 for SIMD alignment) and `L2`, `L3`
    /// are hidden-layer sizes. Common YaneuraOu-shipped sizes:
    /// `256x2-32-32`, `384x2-8-96`, `512x2-8-64`, `768x2-16-64`,
    /// `1024x2-8-32`, `1024x2-8-64`, plus `1536x2-15-32` for the
    /// SFNN family (matches `architectures/sfnnwop-1536.h`). Any
    /// other valid triple (e.g. `256x2-64-64`) is also accepted for
    /// experimentation, though only the YaneuraOu-shipped sizes can
    /// be loaded by stock engine builds. Ignored for KPPT / KPP_KKPT
    /// eval types.
    #[arg(long, default_value = "256x2-32-32")]
    arch: NnueArch,

    /// LayerStack bucketing scheme for the SFNN family. Only consulted
    /// when `--eval-type` is `SFNN_HALFKA1HM`, `SFNN_HALFKA2HM`, or `SFNN_KA2`.
    /// Currently only `king3-by-king3` is supported — it matches
    /// YaneuraOu's `stack_index_for_nnue` (3 friend-king-rank ×
    /// 3 enemy-king-rank = 9 stacks) so the trained `nn.bin` is
    /// loadable and evaluation matches between training and inference.
    #[arg(long, default_value = "king3-by-king3")]
    layerstack: LayerStackMode,

    /// Held-out test set (.hcpe only) for sign-agreement validation
    /// during training. When set, the trainer runs validation after
    /// each save event (= every `--save-rate` superbatches): random-
    /// picks `--test-positions` positions from this file, runs them
    /// through the model, and emits per-superbatch
    /// `test_value_accuracy` and `test_value_loss` columns into
    /// `learn.log`. Positions whose teacher score is 0 (draw stamp)
    /// or `|score| >= --score-drop-abs` (mate stamp) are excluded
    /// from both metrics.
    ///
    /// Only NNUE / SFNN eval types are supported (the network's raw
    /// output is a single scalar). KPPT family is skipped.
    #[arg(long)]
    test_teacher: Option<PathBuf>,

    /// Number of positions to sample from `--test-teacher` per save
    /// event.
    #[arg(long, default_value = "100000")]
    test_positions: usize,

    /// GPU batch size for the validation forward pass. Larger is faster
    /// but uses more VRAM. Independent of `--batch-size` (which
    /// controls training).
    #[arg(long, default_value = "1024")]
    test_batch_size: usize,

    /// Seed for the random sampler in `--test-teacher`. `0`
    /// (default) means "use a time-based seed" (= different sample
    /// each save event). Pass any non-zero value for a reproducible
    /// sample (same positions every time).
    #[arg(long, default_value = "0")]
    test_seed: u64,
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
            // Explicit --output wins; --tag is ignored to keep the
            // user-provided path verbatim.
            return p.clone();
        }
        let mut path = PathBuf::from("checkpoints");
        let mut name = self.eval_type.cli_name().to_string();
        if self.eval_type.uses_arch() {
            name.push('-');
            name.push_str(&self.arch.cli_name());
        }
        if self.eval_type.uses_layerstack() {
            name.push('-');
            name.push_str(self.layerstack.cli_name());
        }
        if let Some(tag) = &self.tag {
            if !tag.is_empty() {
                name.push('-');
                name.push_str(tag);
            }
        }
        path.push(name);
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
    if let Err(e) = record_invocation_to_tag_txt(&args) {
        eprintln!(
            "warning: failed to write tag.txt under {}: {e}",
            args.output_dir().display()
        );
    }
    match args.eval_type {
        EvalType::Kppt | EvalType::KppKkpt => run_kppt_all(&args),
        EvalType::NnueHalfkp => run_halfkp(&args),
        EvalType::NnueKp => run_kp(&args),
        EvalType::NnueKa2 => run_nnue_ka2(&args),
        EvalType::NnueHalfkpe9 => run_halfkpe9(&args),
        EvalType::NnueHalfkpvm => run_halfkpvm(&args),
        EvalType::SfnnHalfka1hm => run_sfnn_1536(&args, ShogiHalfKaHm1, NnueFeatureSet::HalfKaHm1),
        EvalType::SfnnHalfka2hm => run_sfnn_1536(&args, ShogiHalfKaHm2, NnueFeatureSet::HalfKaHm2),
        EvalType::SfnnKa2 => run_sfnn_1536(&args, ShogiKa2, NnueFeatureSet::Ka2),
    }
}

/// Record this process's argv into `<output>/tag.txt` so that, weeks
/// later, the user can recall which CLI invocation produced this
/// checkpoint directory. Always appends; one line per invocation
/// `<unix_ts>\t<arg0> <arg1> ...`. Resumes accumulate a history.
///
/// Failures are non-fatal — if we can't even create the output dir
/// here (permissions, broken path, …), the training step itself will
/// likely report the same problem in a clearer context, so we just
/// log a warning and let the run continue.
fn record_invocation_to_tag_txt(args: &Args) -> std::io::Result<()> {
    use std::io::Write;
    let output_dir = args.output_dir();
    std::fs::create_dir_all(&output_dir)?;
    let tag_path = output_dir.join("tag.txt");
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // argv joined with single spaces; quoting/escaping is intentionally
    // not applied — the line is for human eyeballing, not for re-execution.
    // (clap-parsed values are mostly path/identifier strings without
    // spaces; if the user did pass a quoted path, the original quoting
    // is lost by the time we see std::env::args, so reconstructing it is
    // best-effort regardless.)
    let cmdline: String = std::env::args().collect::<Vec<_>>().join(" ");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&tag_path)?;
    writeln!(f, "{ts}\t{cmdline}")?;
    Ok(())
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
    "eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr,lambda,positions,teacher";

/// Bundle of parameters the enrichment functions need to turn bullet's
/// raw 3-column `log.txt` rows (`superbatch,curr_batch,loss`) into the
/// 11-column `learn.log` CSV rows defined by [`LEARN_LOG_HEADER`].
#[derive(Clone, Debug)]
struct LogContext {
    eval_type: &'static str,
    /// Arch suffix (`256x2-32-32` etc.) for NNUE eval types. Empty string for
    /// KPPT-family eval types since they ignore `--arch`. When non-empty it is
    /// joined into the `eval` column as `<eval-type>-<arch>`, matching the
    /// output-dir naming.
    arch: String,
    lr_start: f32,
    lr_gamma: f32,
    lr_step: usize,
    lambda: f32,
    batch_size: usize,
    batches_per_superbatch: usize,
    teacher_csv: String,
    /// Offset added to bullet's local sb counter when computing the
    /// "absolute" superbatch shown in `learn.log` and used for LR
    /// lookup. Set to `last_saved_sb` when auto-resume runs into a
    /// changed `--teacher` (so bullet's local sb starts back at 1 to
    /// keep the dataloader fresh, but the LR / display sb stays
    /// monotonic across rounds). Otherwise 0.
    sb_offset: usize,
    /// When set, the `learn.log` `lr` column uses positions-based
    /// step (= drops `lr_gamma` every `lr_step_positions` cumulative
    /// positions) instead of the sb-based `lr_step`. Mirrors the
    /// `--lr-step-positions` CLI flag and the `AdjustableStepLR
    /// ::Positions` variant so the enriched log matches what the
    /// trainer actually used.
    lr_step_positions: Option<u64>,
}

impl LogContext {
    fn from_args(args: &Args) -> Self {
        let batches_per_superbatch =
            args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));
        Self {
            eval_type: args.eval_type.cli_name(),
            arch: if args.eval_type.uses_arch() { args.arch.cli_name() } else { String::new() },
            lr_start: args.lr,
            lr_gamma: args.lr_gamma,
            lr_step: args.lr_step,
            lambda: args.lambda,
            batch_size: args.batch_size,
            batches_per_superbatch,
            teacher_csv: csv_escape(&args.teacher),
            sb_offset: 0,
            lr_step_positions: args.lr_step_positions,
        }
    }

    /// LR at a given superbatch — mirrors `bulletou_lib::trainer::schedule::lr::StepLR`.
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

/// Per-superbatch validation result attached to a single save dir's
/// enriched `learn.log`. When `Some`, every row of that dir gets the
/// same `test_value_accuracy` / `test_value_loss` (validation runs once
/// per save event, so all rows in one save share the same metric).
/// When `None`, both columns are emitted as `-`.
#[derive(Clone, Copy, Debug)]
struct TestMetrics {
    accuracy: f32,
    loss: f32,
}

/// Convert bullet's raw 3-column `log.txt` text (`superbatch,curr_batch,loss`
/// per line) into the enriched 11-column CSV body (no header). The header
/// (= [`LEARN_LOG_HEADER`]) is the caller's responsibility, so the same
/// body can be concatenated under a single header by `assemble_numbered_dirs`.
///
/// The `train_value_loss` column carries bullet's loss (= the third field
/// of `log.txt`, which is the running average of training-loss over the
/// last 32 batches). `test_value_accuracy` and `test_value_loss` are the
/// per-superbatch held-out validation result from `--test-teacher`; both
/// are `-` when the caller passes `test_metrics = None`.
fn enrich_bullet_log_to_csv(
    raw: &str,
    ctx: &LogContext,
    epoch: usize,
    component: &str,
    position_offset: usize,
    test_metrics: Option<TestMetrics>,
) -> String {
    let mut out = String::new();
    let (test_acc_field, test_loss_field): (String, String) = match test_metrics {
        Some(m) => (format!("{:.6}", m.accuracy), format!("{:.6}", m.loss)),
        None => ("-".to_string(), "-".to_string()),
    };
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() != 3 {
            continue;
        }
        let Ok(local_sb) = parts[0].parse::<usize>() else { continue };
        let Ok(b) = parts[1].parse::<usize>() else { continue };
        let train_loss = parts[2];
        // Absolute (= cumulative across rounds) sb. Bullet's `log.txt`
        // carries its own internal sb counter (= local within the run);
        // when the macro had to reset bullet's sb=1 for a teacher-changed
        // resume, `ctx.sb_offset` shifts the displayed sb to the
        // continuation point so the column stays monotonic.
        let absolute_sb = local_sb + ctx.sb_offset;
        // `positions` keeps using bullet's local sb because position_offset
        // already carries the cumulative count from prior runs — the
        // formula then adds (local_sb-1)*sb_size + b*batch_size to the
        // carry-over to give an honest cumulative position count.
        let positions = ctx.positions_at(local_sb, b, position_offset);
        // LR: when `lr_step_positions` is set, drop based on the
        // cumulative `positions` count above (= matches the trainer's
        // `AdjustableStepLR::Positions` variant). Otherwise fall back
        // to the sb-based formula keyed by absolute_sb.
        let lr = match ctx.lr_step_positions {
            Some(step) => AdjustableStepLR::lr_at_positions(ctx.lr_start, ctx.lr_gamma, step, positions as u64),
            None => ctx.lr_at(absolute_sb),
        };
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
            "{eval},{epoch},{sb},{b},{ta},{tl},{train},{lr},{lambda:.3},{positions},{teacher}\n",
            eval = eval_field,
            sb = absolute_sb,
            ta = test_acc_field,
            tl = test_loss_field,
            train = train_loss,
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
/// The parser uses `splitn(11, ',')` so any commas inside the trailing
/// `teacher` field (e.g. a comma-separated teacher list) don't disturb
/// the first 10 columns. Component is extracted from the `eval` column
/// at index 0: a slash-suffix (e.g. `KPPT/kk`) names the component
/// explicitly; absence of a slash means a single-component NNUE eval
/// type, which maps to the `"nnue"` component key. The `positions`
/// column is at index 9 in the 11-column layout
/// (eval, epoch, superbatch, curr_batch, test_value_accuracy,
/// test_value_loss, train_value_loss, lr, lambda, **positions**, teacher).
fn read_prior_positions(top_level_log: &std::path::Path) -> std::collections::BTreeMap<String, usize> {
    let mut map = std::collections::BTreeMap::new();
    let Ok(content) = std::fs::read_to_string(top_level_log) else { return map };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("eval,") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(11, ',').collect();
        if parts.len() < 10 {
            continue;
        }
        let eval = parts[0];
        let component = eval.split_once('/').map(|(_, c)| c).unwrap_or("nnue");
        let Ok(positions) = parts[9].parse::<usize>() else { continue };
        let entry = map.entry(component.to_string()).or_insert(0);
        if positions > *entry {
            *entry = positions;
        }
    }
    map
}

/// Detect the latest saved superbatch number from the highest-numbered
/// `<output_dir>/<NNNN>/learn.log`. Used to auto-resume the LR scheduler
/// (and the trainer's internal sb counter) at `last_sb + 1` instead of
/// silently restarting from sb=1 when the user re-runs the same command
/// after Ctrl+C.
///
/// Returns `None` if there is no numbered dir, no `learn.log`, or no
/// parseable sb column — which collapses to "treat as a fresh run" by
/// the caller.
fn read_latest_saved_superbatch(output_dir: &std::path::Path) -> Option<usize> {
    let mut latest_idx: Option<usize> = None;
    for entry in std::fs::read_dir(output_dir).ok()?.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else { continue };
        let Ok(n) = name.parse::<usize>() else { continue };
        latest_idx = Some(latest_idx.map_or(n, |m| m.max(n)));
    }
    let n = latest_idx?;
    let learn_log = output_dir.join(format!("{n:04}")).join("learn.log");
    let content = std::fs::read_to_string(&learn_log).ok()?;
    // 11-column rows: eval, epoch, sb, batch, test_value_accuracy,
    // test_value_loss, train_value_loss, lr, lambda, positions, teacher.
    // sb is at column index 2. All rows in a single per-save dir share the
    // same sb (bullet flushes log.txt at save time and the dir captures one
    // save event), so taking the max is robust whether the schedule is
    // save_rate=1 or larger.
    let mut max_sb: Option<usize> = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("eval,") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(9, ',').collect();
        if parts.len() < 3 {
            continue;
        }
        let Ok(sb) = parts[2].parse::<usize>() else { continue };
        max_sb = Some(max_sb.map_or(sb, |m| m.max(sb)));
    }
    max_sb
}

/// Detect the teacher path recorded in the highest-numbered
/// `<output_dir>/<NNNN>/learn.log`. Used to decide whether
/// auto-resume's `start_superbatch` skip-ahead is safe: bullet's
/// dataloader skips `(start_sb - 1) * batches_per_sb` records at
/// startup, which only makes sense if the resume run uses the same
/// teacher file as the previous run. If the teacher changed, the new
/// (smaller) file may have fewer records than the requested skip,
/// causing `NoBatchesReceived` panic. We use the comparison result to
/// fall back to `start_superbatch=1` in the changed-teacher case while
/// still honouring the model+optimizer load from `state.bin`.
///
/// Returns the **trimmed** teacher field of the **last (= bottom) row**
/// in the latest dir's learn.log (which is the most recent `--teacher`
/// arg used for that save). Returns `None` if no row could be parsed.
fn read_latest_saved_teacher(output_dir: &std::path::Path) -> Option<String> {
    let mut latest_idx: Option<usize> = None;
    for entry in std::fs::read_dir(output_dir).ok()?.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else { continue };
        let Ok(n) = name.parse::<usize>() else { continue };
        latest_idx = Some(latest_idx.map_or(n, |m| m.max(n)));
    }
    let n = latest_idx?;
    let learn_log = output_dir.join(format!("{n:04}")).join("learn.log");
    let content = std::fs::read_to_string(&learn_log).ok()?;
    // Same 11-column layout as read_latest_saved_superbatch. teacher
    // is the trailing field (index 10). splitn(11, ',') keeps any
    // commas inside teacher (= comma-separated `--teacher` list)
    // as a single CSV field.
    let mut last_teacher: Option<String> = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("eval,") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(11, ',').collect();
        if parts.len() < 11 {
            continue;
        }
        last_teacher = Some(parts[10].trim().to_string());
    }
    last_teacher
}

/// Append the body of the latest save dir's `learn.log` (already enriched
/// 11-column CSV from `assemble_numbered_dirs` / `finalize_nnue_dirs`) onto
/// the top-level `<output>/learn.log`, writing the CSV header on first
/// file creation. The result is a single pure CSV — no section headers,
/// no separators — that pandas / Excel can load directly.
///
/// If the existing top-level `learn.log` was written by an older version
/// of `bulletou` (= a different header line than the current
/// [`LEARN_LOG_HEADER`]), this returns `InvalidData` so the caller can
/// alert the user to clear the old file rather than silently mixing
/// schemas in the same CSV.
fn append_to_top_level_log(output_dir: &std::path::Path, last_idx: usize) -> std::io::Result<()> {
    use std::io::Write;
    let latest_log = output_dir.join(format!("{last_idx:04}")).join("learn.log");
    let body = std::fs::read_to_string(&latest_log)?;
    let top = output_dir.join("learn.log");
    let top_existed = top.is_file();

    // Detect schema mismatch on existing file. The existing file is
    // assumed to start with a header line followed by data rows. If the
    // header doesn't match, refuse to mix formats.
    if top_existed {
        let mut head_buf = String::new();
        if let Ok(mut f) = std::fs::File::open(&top) {
            use std::io::Read as _;
            // Header is at most ~200 bytes; reading 1KB is plenty.
            let mut buf = [0u8; 1024];
            if let Ok(n) = f.read(&mut buf) {
                if let Ok(s) = std::str::from_utf8(&buf[..n]) {
                    if let Some(first_line) = s.lines().next() {
                        head_buf = first_line.to_string();
                    }
                }
            }
        }
        if !head_buf.is_empty() && head_buf != LEARN_LOG_HEADER {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{}: existing learn.log header has a different schema than this build expects.\n  \
                     existing: {head_buf}\n  expected: {LEARN_LOG_HEADER}\n  \
                     This build added/changed columns (e.g. test_value_accuracy / test_value_loss).\n  \
                     Rename or delete the old file (and the per-dir <NNNN>/learn.log if you want a clean restart) and re-run.",
                    top.display()
                ),
            ));
        }
    }

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
            // KPPT family does not run --test-teacher validation (out tensor
            // shape doesn't match the single-scalar assumption); always emit
            // `-` for the test_value_* columns by passing None.
            log_buf.push_str(&enrich_bullet_log_to_csv(&raw, ctx, epoch, label, prior, None));
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
                    // KPPT family does not get auto-resume yet (the 3-component
                    // assembly makes the bookkeeping non-trivial). Treat
                    // `--start-superbatch` as a plain default-1 flag for now.
                    start_superbatch: args.start_superbatch.unwrap_or(1),
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

/// Cache of test-set positions used for per-save validation. Loaded
/// once at the start of training (when `--test-teacher` is set) and
/// reused for every subsequent validation forward pass — the random
/// sampling happens once at load time, not on each save.
struct TestPositionsCache {
    positions: Vec<bulletou_lib::shogi::PackedSfenValue>,
    teacher_scores: Vec<i16>,
    teacher_results: Vec<i8>,
}

impl TestPositionsCache {
    /// `args.test_teacher` is `Some` and we successfully sampled
    /// positions: `Some(cache)`. Otherwise `None` (= no validation).
    fn try_load(args: &Args) -> Option<Self> {
        let test_path = args.test_teacher.as_ref()?;
        let path = match test_path.to_str() {
            Some(s) => s.to_string(),
            None => {
                eprintln!("  WARN: --test-teacher path is not valid UTF-8, skipping validation");
                return None;
            }
        };
        eprintln!(
            "  loading {} test positions from {} (seed={}) for per-superbatch validation...",
            args.test_positions, path, args.test_seed
        );
        match read_random_hcpe_positions(&path, args.test_positions, args.test_seed) {
            Ok(positions) => {
                let teacher_scores: Vec<i16> = positions.iter().map(|p| p.score()).collect();
                let teacher_results: Vec<i8> = positions.iter().map(|p| p.game_result()).collect();
                eprintln!("  ...{} test positions ready", positions.len());
                Some(Self { positions, teacher_scores, teacher_results })
            }
            Err(e) => {
                eprintln!("  WARN: failed to read --test-teacher {path}: {e}; per-superbatch validation disabled");
                None
            }
        }
    }
}

/// Run validation on the cached test positions and produce per-save
/// `TestMetrics`. Caller must already hold `&mut trainer` (= called
/// outside `trainer.run`).
fn run_one_test_pass(
    cache: &TestPositionsCache,
    args: &Args,
    trainer_outputs: Vec<f32>,
) -> TestMetrics {
    let cap = if args.score_drop_abs > 0 { Some(args.score_drop_abs) } else { None };
    let report = compute_sign_accuracy(
        &trainer_outputs,
        &cache.teacher_scores,
        &cache.teacher_results,
        cap,
        args.lambda,
        args.scale as f32,
    );
    let accuracy = if report.compared == 0 { f32::NAN } else { report.accuracy() };
    let loss = report.test_loss.unwrap_or(f32::NAN);
    eprintln!(
        "  test: accuracy={:.4}% ({}/{}, draws={}, mate={}), loss={:.6}",
        accuracy * 100.0,
        report.sign_matches,
        report.compared,
        report.draws_in_teacher,
        report.filtered_by_score_cap,
        loss,
    );
    TestMetrics { accuracy, loss }
}

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

        // Auto-resume: if --start-superbatch is unspecified and a previous
        // run left numbered checkpoint dirs under output_dir, continue the
        // sb counter (and therefore the LR schedule) from the last saved
        // superbatch + 1 instead of silently restarting at sb=1.
        //
        // When sb is auto-continued, `positions_at(sb, b, 0)` already gives
        // the correct cumulative position count (sb itself encodes the
        // history), so the prior-position offset is 0. When the user
        // explicitly passes --start-superbatch (signalling "reset sb to N"),
        // fall back to the legacy behaviour of carrying the prior position
        // sum forward from the existing top-level learn.log.
        let mut cb_ctx = LogContext::from_args(args);
        let cb_top_level_log = output_dir_buf.join("learn.log");
        let auto_resume_sb_raw = read_latest_saved_superbatch(&output_dir_buf);
        // Teacher-change detection: bullet's dataloader skips
        // `(start_sb - 1) * batches_per_sb` records at startup, which
        // assumes the resume run uses the same teacher data. If the
        // teacher path changed (e.g. the user is doing yane-distill
        // round-2 training on a new chunk), the new file may not have
        // enough records to reach the skip target → trainer.run hits
        // EOF before any batch is delivered → `NoBatchesReceived`
        // panic.
        //
        // Resolution: keep the LR schedule continuous (so the user does
        // not see LR jump back to start when they swap teacher files
        // mid-experiment), but tell bullet `start_superbatch=1` so the
        // dataloader reads the new file from the beginning.
        // `AdjustableStepLR::Offset` handles the LR side by shifting
        // the sb input by `last_saved_sb`. `cb_ctx.sb_offset` carries
        // the same shift through to the enriched `learn.log`'s sb /
        // lr columns so the displayed time-series stays monotonic.
        let prev_teacher = read_latest_saved_teacher(&output_dir_buf);
        let teacher_changed = match prev_teacher.as_deref() {
            Some(prev) => prev.trim() != args.teacher.trim(),
            None => false,
        };
        let user_set_start = args.start_superbatch.is_some();
        let (effective_start_superbatch, sb_offset_for_lr) = if user_set_start {
            // Explicit user value wins for both dataloader skip and LR.
            (args.start_superbatch.unwrap(), 0usize)
        } else if teacher_changed {
            if let Some(last_sb) = auto_resume_sb_raw {
                // Keep dataloader fresh (start_sb=1) but shift LR by
                // last_sb so the schedule continues from the correct
                // step. cb_ctx.sb_offset mirrors this shift in the log.
                (1usize, last_sb)
            } else {
                // Teacher changed but no prior dirs → fresh run.
                (1usize, 0)
            }
        } else if let Some(last_sb) = auto_resume_sb_raw {
            // Same teacher: legacy behaviour — let bullet's dataloader
            // skip ahead so training picks up where the previous run
            // left off in the SAME file.
            (last_sb + 1, 0)
        } else {
            // First run.
            (1usize, 0)
        };
        cb_ctx.sb_offset = sb_offset_for_lr;
        // When the LR scheduler is `AdjustableStepLR::Plain` and bullet's
        // sb already encodes the absolute step (= same-teacher resume or
        // fresh run), positions_at(sb, b, 0) gives the correct cumulative
        // count and prior_position should be 0. When we shifted via
        // sb_offset (= teacher-changed case) or when the user explicitly
        // overrode start_superbatch, sb is local-to-this-run so we need
        // the cumulative carry-over from the existing top-level log.
        //
        // For positions-based LR (`--lr-step-positions`), the `lr` column
        // is derived from the `positions` value directly, so we always
        // need the cumulative carry-over here so the enrich path's
        // `positions` matches what `AdjustableStepLR::Positions` saw.
        let want_cumulative_prior = user_set_start
            || teacher_changed
            || auto_resume_sb_raw.is_none()
            || args.lr_step_positions.is_some();
        let cb_prior_position = if want_cumulative_prior {
            read_prior_positions(&cb_top_level_log).get("nnue").copied().unwrap_or(0)
        } else {
            0
        };
        let cb_next_idx = std::cell::Cell::new(count_existing_numbered_dirs(&output_dir_buf) + 1);

        if !user_set_start {
            if teacher_changed {
                if let (Some(prev), Some(last_sb)) = (prev_teacher.as_deref(), auto_resume_sb_raw) {
                    eprintln!(
                        "  teacher path differs from previous run\n    previous: {prev}\n    current:  {}\n  \
                         dataloader will read the new file from the beginning (start_sb=1) but the LR\n  \
                         schedule continues from sb {} (model + optimiser are loaded from the latest\n  \
                         state.bin as usual). Pass --start-superbatch <N> to override.",
                        args.teacher,
                        last_sb + 1
                    );
                }
            } else if let Some(last_sb) = auto_resume_sb_raw {
                eprintln!(
                    "  auto-resuming from superbatch {} (last saved: {}); LR schedule continues. \
                     pass --start-superbatch to override.",
                    effective_start_superbatch, last_sb
                );
            }
        }

        // Build the LR scheduler:
        // - `--lr-step-positions <N>` if set: positions-based, decoupled
        //   from sb. Carries `prior_positions` (= cumulative trained
        //   across previous rounds) so the count is absolute.
        // - else: sb-based. `Plain` when bullet's sb is already
        //   absolute (= same-teacher resume / fresh run), `Offset`
        //   when we shifted to sb_offset (= teacher-changed case so
        //   bullet's local sb starts at 1 but the LR continues).
        let lr_scheduler_for_run = if let Some(positions_per_step) = args.lr_step_positions {
            let prior_positions = read_prior_positions(&cb_top_level_log)
                .get("nnue")
                .copied()
                .unwrap_or(0) as u64;
            AdjustableStepLR::Positions {
                start: args.lr,
                gamma: args.lr_gamma,
                positions_per_step,
                prior_positions,
                batch_size: args.batch_size,
                batches_per_superbatch,
            }
        } else if sb_offset_for_lr == 0 {
            AdjustableStepLR::Plain(lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step })
        } else {
            AdjustableStepLR::Offset {
                inner: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
                offset: sb_offset_for_lr,
            }
        };

        // Per-save validation cache: load test positions ONCE up front
        // (random-pick happens here), then reuse the same positions for
        // every save event. `None` if --test-teacher unset or load failed.
        let test_cache = TestPositionsCache::try_load(args);

        // Per-save incremental finalize: rename `<net_id>-<sb>` →
        // `<NNNN>/`, generate per-dir `learn.log` (with per-save test
        // metrics), append to top-level `learn.log`. Done OUTSIDE
        // `trainer.run` (= once per save chunk) so we can call
        // `trainer.eval_packed_batch` for validation between chunks.
        // Killing training mid-run still leaves a clean numbered layout
        // and a resumable top-level log.

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

            // Run the epoch in chunks of `save_rate` superbatches. Each
            // chunk ends at a save boundary, after which we validate (if
            // requested) and finalise the saved dir with the test metrics.
            let mut chunk_start = effective_start_superbatch;
            let chunk_size = args.save_rate.max(1);
            'epoch: loop {
                if chunk_start > end_superbatch { break; }
                let chunk_end = chunk_start.saturating_add(chunk_size).saturating_sub(1).min(end_superbatch);

                let schedule = TrainingSchedule {
                    net_id: net_id_for_epoch.clone(),
                    eval_scale: args.scale as f32,
                    steps: TrainingSteps {
                        batch_size: args.batch_size,
                        batches_per_superbatch,
                        start_superbatch: chunk_start,
                        end_superbatch: chunk_end,
                    },
                    wdl_scheduler: wdl::ConstantWDL { value: 1.0 - args.lambda },
                    lr_scheduler: lr_scheduler_for_run.clone(),
                    save_rate: args.save_rate,
                };

                // Per-chunk callback only records that the save happened
                // (no in-callback finalize). The actual rename + enrich
                // runs outside `trainer.run` so it can take per-save test
                // metrics computed via `trainer.eval_packed_batch`.
                let net_id_for_cb = net_id_for_epoch.clone();
                let output_dir_for_cb = output_dir_buf.clone();
                let saved_any_ref = &saved_any;
                let saved_dir_in_chunk: std::cell::RefCell<Option<std::path::PathBuf>> =
                    std::cell::RefCell::new(None);
                let saved_dir_ref = &saved_dir_in_chunk;
                let on_checkpoint_saved = move |superbatch: usize| {
                    saved_any_ref.set(true);
                    let ckpt_dir = output_dir_for_cb.join(format!("{net_id_for_cb}-{superbatch}"));
                    if let Err(e) = convert_save_dir_to_nnue_layout(&ckpt_dir) {
                        eprintln!("  WARN: failed to convert save dir {}: {e}", ckpt_dir.display());
                        return;
                    }
                    eprintln!("  wrote NNUE nn.bin + state.bin in {}", ckpt_dir.display());
                    *saved_dir_ref.borrow_mut() = Some(ckpt_dir);
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

                // Closure dropped → its borrow of saved_dir_in_chunk released.
                let saved_ckpt_dir = saved_dir_in_chunk.into_inner();
                let Some(ckpt_dir) = saved_ckpt_dir else {
                    // Save did not fire in this chunk → either EOF before
                    // crossing the save boundary, or chunk_size > available
                    // remaining sb. Either way the epoch is over.
                    break 'epoch;
                };

                // Per-save validation (if --test-teacher cached positions).
                let test_metrics = test_cache.as_ref().map(|cache| {
                    let outputs = trainer.eval_packed_batch(&cache.positions, args.test_batch_size);
                    run_one_test_pass(cache, args, outputs)
                });

                // Finalise this saved dir into <NNNN>/ with the test metrics.
                let idx = cb_next_idx.get();
                match finalize_one_nnue_dir(
                    &output_dir_buf,
                    &ckpt_dir,
                    &cb_ctx,
                    epoch,
                    idx,
                    cb_prior_position,
                    test_metrics,
                ) {
                    Ok(dst) => {
                        cb_next_idx.set(idx + 1);
                        if let Err(e) = append_to_top_level_log(&output_dir_buf, idx) {
                            eprintln!(
                                "  WARN: failed to update {}: {e}",
                                output_dir_buf.join("learn.log").display()
                            );
                        }
                        eprintln!("  -> {}/", dst.display());
                    }
                    Err(e) => {
                        eprintln!(
                            "  WARN: failed to finalise {} into NNNN/: {e}",
                            ckpt_dir.display()
                        );
                    }
                }

                chunk_start = chunk_end + 1;
            }
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
            // The fallback path bypasses the per-chunk callback flow, so
            // run validation here too (when --test-teacher is set) and
            // finalise the dir directly. Without this, the leftover dir
            // gets enriched by the post-macro `finalize_nnue_dirs` call
            // with `None` test_metrics, causing test_value_* columns to
            // come out as "-" even though --test-teacher was given.
            let test_metrics = test_cache.as_ref().map(|cache| {
                let outputs = trainer.eval_packed_batch(&cache.positions, args.test_batch_size);
                run_one_test_pass(cache, args, outputs)
            });
            let idx = cb_next_idx.get();
            match finalize_one_nnue_dir(
                &output_dir_buf,
                &ckpt_dir,
                &cb_ctx,
                /*epoch=*/ max_epochs,
                idx,
                cb_prior_position,
                test_metrics,
            ) {
                Ok(dst) => {
                    cb_next_idx.set(idx + 1);
                    if let Err(e) = append_to_top_level_log(&output_dir_buf, idx) {
                        eprintln!(
                            "  WARN: failed to update {}: {e}",
                            output_dir_buf.join("learn.log").display()
                        );
                    }
                    eprintln!("  -> {}/", dst.display());
                }
                Err(e) => {
                    eprintln!(
                        "  WARN: failed to finalise fallback save dir {} into NNNN/: {e}",
                        ckpt_dir.display()
                    );
                }
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
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
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
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
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

/// NNUE K-A2 training entry point. Mirrors `run_kp` exactly, only the input
/// feature differs: K (162 dims) + A2 (1629 dims, kings collapsed onto friend
/// plane via v2 encoding) = 1791 dims per perspective. Network topology is
/// the same 4-layer ClippedReLU as halfkp_256x2-32-32 / kp_256x2-32-32.
/// Architecture is selected via `--arch` (default `256x2-32-32`).
fn run_nnue_ka2(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch.dims();
    let input_size = ShogiKa2.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_KA2 ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
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

    let save_format = build_nnue_save_format(NnueFeatureSet::Ka2, l1_size, l2_size, l3_size);

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKa2)
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
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
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
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
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

/// NNUE HalfKP_vm training entry point. Identical wiring to `run_halfkp`,
/// only the input feature type swaps from `ShogiHalfKP` (125,388 dims) to
/// `ShogiHalfKPvm` (69,660 dims, file-mirror folded). The 4-layer
/// ClippedReLU network and quantisation pipeline are unchanged.
fn run_halfkpvm(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch.dims();
    let input_size = ShogiHalfKPvm.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_HALFKPVM ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
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

    let save_format = build_nnue_save_format(NnueFeatureSet::HalfKpvm, l1_size, l2_size, l3_size);

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiHalfKPvm)
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
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
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

/// SFNN-1536 / LayerStacks training entry point (= `SFNN_HALFKA1HM` and
/// `SFNN_HALFKA2HM`). Generic over the input feature so v1 (`HalfKA_hm1`,
/// 76,950 dim) and v2 (`HalfKA_hm2`, 73,305 dim) share the entire
/// training pipeline — only the `input` and `feature_set` arguments
/// differ between the two callers.
///
/// Network topology mirrors YaneuraOu `architectures/sfnnwop-1536.h`:
/// per-perspective FT (`l0`, 1536 dim) → CReLU → pairwise-mul → concat
/// across perspectives → bucket-specific L1 (= 16 = 15 hidden + 1 PSQT
/// shortcut neuron) → split off the PSQT neuron as a residual bypass
/// → `[SqrCReLU; CReLU]` concat → L2 (32 dim) → CReLU → L3 (scalar)
/// → add PSQT residual.
///
/// The bucket-specific layers `l1` and `l2` use the LayerStacks pattern:
/// the weight tensor is `(NUM_STACKS × out_dim, in_dim)` and a
/// `.select(output_buckets)` per forward pass picks the active slice
/// based on `ShogiLayerStackBucket9` (= `--layerstack`).
///
/// **Note**: the `nn.bin` written by this commit uses the standard
/// non-LayerStack `build_nnue_save_format` and is **not** YaneuraOu-
/// loadable yet. The LayerStacks-aware save layout (Phase 4) lands
/// in a follow-up commit.
fn run_sfnn_1536<I>(args: &Args, input: I, feature_set: NnueFeatureSet)
where
    I: SparseInputType<RequiredDataType = bulletou_lib::shogi::PackedSfenValue> + Copy,
{
    let (ft_size, l1_hidden, l2_size) = args.arch.dims();
    if ft_size != 1536 || l1_hidden != 15 || l2_size != 32 {
        eprintln!(
            "warning: --arch {} is being trained as an SFNN, but YaneuraOu only ships\n\
             1536x2-15-32 for the SFNNwoP family. The resulting nn.bin will not be\n\
             engine-loadable. Continuing for ablation purposes.",
            args.arch.cli_name()
        );
    }
    // L1 output = effective hidden + 1 PSQT-shortcut neuron (matches
    // yaneuraou `kHidden1Dims + 1` in sfnnwop-1536.h).
    let l1_out = l1_hidden + 1;
    let l1_effective = l1_hidden;
    let l2_in = l1_effective * 2; // [SqrCReLU; CReLU] concat
    let num_stacks = args.layerstack.num_stacks();
    let input_size = input.num_inputs();

    eprintln!(
        "=== bulletou: running {} ({}x2-{}-{} CReLU+SqrCReLU, dual-perspective, LayerStacks={} via {}) ===",
        args.eval_type.cli_name(),
        ft_size,
        l1_hidden,
        l2_size,
        num_stacks,
        args.layerstack.cli_name()
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

    // LayerStack bucket selector. `Kingrank9` matches YaneuraOu's
    // `stack_index_for_nnue` (3×3 = 9 buckets by king ranks).
    let bucket_impl = match args.layerstack {
        LayerStackMode::Kingrank3by3 => ShogiLayerStackBucket9::KingRank9,
    };

    // YaneuraOu SFNNwoP1536 互換 nn.bin の save format。`Sfnn1536SaveParams`
    // で feature set / 各層のサイズ / LayerStacks 数を受け、`bulletou_lib::value::nnue_save_sfnn1536`
    // が組み立てた `SavedFormat` 列を渡す。出力は `EvalDir` で yaneuraou
    // (`YANEURAOU_ENGINE_NNUE_SFNNwoP1536` ビルド) が load できる layout。
    let save_format = build_sfnn_1536_save_format(Sfnn1536SaveParams {
        feature_set,
        input_size,
        ft_size,
        l1_hidden,
        l2_size,
        num_stacks,
    });

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(input)
        .output_buckets(bucket_impl)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs, output_buckets| {
        let l0 = builder.new_affine("l0", input_size, ft_size);
        l0.init_with_effective_input_size(32);

        // L1: bucket-specific weights (zero-init) + shared factorised
        // counterpart `l1f`. Bullet's LayerStacks pattern keeps training
        // stable in the early epochs because `l1` starts at zero — all
        // bucket outputs are equal to `l1f` until per-bucket signal develops.
        let l1 = Affine {
            weights: builder.new_weights(
                "l1w",
                Shape::new(num_stacks * l1_out, ft_size),
                InitSettings::Zeroed,
            ),
            bias: builder.new_weights("l1b", Shape::new(num_stacks * l1_out, 1), InitSettings::Zeroed),
        };
        let l1f = builder.new_affine("l1f", ft_size, l1_out);
        let l2 = builder.new_affine("l2", l2_in, num_stacks * l2_size);
        let l3 = builder.new_affine("l3", l2_size, num_stacks);

        // Per-perspective FT → CReLU → pairwise-mul → concat. After the
        // pairwise-mul the dim is ft_size/2; concat of stm/ntm brings it
        // back to ft_size (matching `kInputDims = kTransformedFeatureDimensions`
        // in sfnnwop-1536.h).
        let stm = l0.forward(stm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
        let ntm = l0.forward(ntm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
        let combined = stm.concat(ntm);

        // L1 = bucket-selected + shared. The shared term is added to all
        // buckets so the model can learn before per-bucket signal accumulates.
        let l1_out_t = l1.forward(combined).select(output_buckets) + l1f.forward(combined);

        // Split the L1 output: rows 0..l1_effective are the hidden, the
        // last row is the PSQT shortcut neuron that bypasses everything
        // and adds straight into the final scalar.
        let l1_main = l1_out_t.slice_rows(0, l1_effective);
        let l1_skip = l1_out_t.slice_rows(l1_effective, l1_out);

        // [SqrCReLU; CReLU] pair, matching yaneuraou's
        // `memcpy(ac_sqr_0_out + kHidden1Dims, ac_0_out, ...)` concat layout.
        let l1_sqr = l1_main.abs_pow(2.0) * (127.0 / 128.0);
        let l2_input = l1_sqr.concat(l1_main).crelu();

        let l2_out_t = l2.forward(l2_input).select(output_buckets).crelu();
        let l3_out = l3.forward(l2_out_t).select(output_buckets);

        // PSQT bypass: final = L3(bucket) + PSQT shortcut neuron, matching
        // yaneuraou's `buf.fc_2_out[0] += buf.fc_0_out[kHidden1Dims]`.
        l3_out + l1_skip
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
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
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

    // Derive the YaneuraOu edition name that matches this trained nn.bin.
    // Format follows nnue_arch_gen.py / source/Makefile: the feature suffix
    // is the lowercase tag the python script dispatches on, and the dim
    // segments use underscores (not hyphens) so the resulting -D macro is a
    // valid C identifier (avoids clang -Wc99-extensions warning).
    let feature_suffix = match feature_set {
        NnueFeatureSet::HalfKaHm1 => "halfkahm1",
        NnueFeatureSet::HalfKaHm2 => "halfkahm2",
        NnueFeatureSet::Ka2 => "ka2",
        // run_sfnn_1536 is only invoked from EvalType::SfnnHalfka1hm /
        // SfnnHalfka2hm / SfnnKa2 dispatch in main(), so any other feature
        // set means a new EvalType variant was added without updating this
        // arm — fail loudly rather than print a wrong edition name.
        other => unreachable!("run_sfnn_1536 received unsupported feature set: {other:?}"),
    };
    let edition_name = format!(
        "YANEURAOU_ENGINE_NNUE_SFNNwoPSQT_{feature_suffix}_{ft_size}_{l1_hidden}_{l2_size}_ls{num_stacks}"
    );
    let legacy_alias_note = if matches!(feature_set, NnueFeatureSet::HalfKaHm2)
        && ft_size == 1536
        && l1_hidden == 15
        && l2_size == 32
        && num_stacks == 9
    {
        // sfnnwop-1536.h is special-cased in source/Makefile so this exact
        // architecture is also accepted under the shorter legacy alias.
        "\n  (legacy alias: YANEURAOU_ENGINE_NNUE_SFNNwoP1536)"
    } else {
        ""
    };

    eprintln!(
        "note: nn.bin in each save dir targets a YaneuraOu build with edition\n  \
             {edition_name}{legacy_alias_note}\n\
         Build it with `make normal YANEURAOU_EDITION=<edition>`."
    );
}

/// Single-component analogue of `assemble_numbered_dirs`: list `<net_id>-*/`
/// (or `<net_id>-e<epoch>-<sb>/` for multi-epoch) under `output_dir`, sort
/// by (epoch, sb), rename them to `0NNN/` starting at `existing_count + 1`,
/// and enrich each dir's bullet-format `log.txt` into the 7-column CSV
/// `learn.log` shared with KPPT.
/// Single-dir version of [`finalize_nnue_dirs`]: rename `src` to
/// `output_dir/<idx:04>/` and convert its raw `log.txt` to the enriched
/// `learn.log` in the new location. Used by the per-superbatch save callback
/// in [`run_training_inline_nnue`] so that `learn.log` and the `0001/`
/// numbered layout are in place even if training is killed mid-run.
fn finalize_one_nnue_dir(
    output_dir: &std::path::Path,
    src: &std::path::Path,
    ctx: &LogContext,
    epoch: usize,
    idx: usize,
    prior_position: usize,
    test_metrics: Option<TestMetrics>,
) -> std::io::Result<std::path::PathBuf> {
    let dst = output_dir.join(format!("{idx:04}"));
    std::fs::rename(src, &dst)?;
    let log_txt = dst.join("log.txt");
    let learn_log = dst.join("learn.log");
    let raw = std::fs::read_to_string(&log_txt).unwrap_or_default();
    let body = enrich_bullet_log_to_csv(&raw, ctx, epoch, "nnue", prior_position, test_metrics);
    let mut content = String::with_capacity(body.len() + LEARN_LOG_HEADER.len() + 1);
    content.push_str(LEARN_LOG_HEADER);
    content.push('\n');
    content.push_str(&body);
    std::fs::write(&learn_log, content)?;
    let _ = std::fs::remove_file(&log_txt);
    Ok(dst)
}

/// Sweep any remaining bullet-named (`<net_id>-<sb>`) checkpoint dirs that
/// were not finalised incrementally by the per-superbatch callback in
/// [`run_training_inline_nnue`]. In normal flow this is empty (the callback
/// finalises each dir as it is written); the only case where it has work
/// to do is the "教師 < 1 superbatch" fallback save, which writes its
/// bullet-named dir AFTER the training loop and so misses the callback.
///
/// Returns `(first_idx, last_idx)` of dirs finalised here, both `0` when
/// nothing was left to do.
fn finalize_nnue_dirs(
    output_dir: &std::path::Path,
    ctx: &LogContext,
    net_id_prefix: &str,
    prior_position: usize,
) -> std::io::Result<(usize, usize)> {
    let src_dirs = list_component_checkpoints_sorted(output_dir, net_id_prefix);
    let n = src_dirs.len();
    if n == 0 {
        return Ok((0, 0));
    }

    let existing_count = count_existing_numbered_dirs(output_dir);

    eprintln!(
        "\n=== finalising {n} leftover NNUE checkpoint dir(s) under {} (starting at #{}) ===",
        output_dir.display(),
        existing_count + 1
    );
    for (i, (epoch, _sb, src)) in src_dirs.iter().enumerate() {
        let idx = existing_count + i + 1;
        // Leftover dirs were not finalised by the per-save callback so we
        // also have no test metrics for them (validation runs in the
        // training loop, not in this fallback path).
        let dst = finalize_one_nnue_dir(output_dir, src, ctx, *epoch, idx, prior_position, None)?;
        eprintln!("  -> {}/", dst.display());
    }
    Ok((existing_count + 1, existing_count + n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn nnue_arch_parse_known_presets() {
        assert_eq!(NnueArch::from_str("256x2-32-32").unwrap().dims(), (256, 32, 32));
        assert_eq!(NnueArch::from_str("384x2-8-96").unwrap().dims(), (384, 8, 96));
        assert_eq!(NnueArch::from_str("512x2-8-64").unwrap().dims(), (512, 8, 64));
        assert_eq!(NnueArch::from_str("768x2-16-64").unwrap().dims(), (768, 16, 64));
        assert_eq!(NnueArch::from_str("1024x2-8-32").unwrap().dims(), (1024, 8, 32));
        assert_eq!(NnueArch::from_str("1024x2-8-64").unwrap().dims(), (1024, 8, 64));
        assert_eq!(NnueArch::from_str("1536x2-15-32").unwrap().dims(), (1536, 15, 32));
    }

    #[test]
    fn nnue_arch_parse_freeform_sizes() {
        // 新しい自由なサイズも受理される。
        assert_eq!(NnueArch::from_str("256x2-64-64").unwrap().dims(), (256, 64, 64));
        assert_eq!(NnueArch::from_str("2048x2-32-64").unwrap().dims(), (2048, 32, 64));
    }

    #[test]
    fn nnue_arch_cli_name_roundtrip() {
        for s in ["256x2-32-32", "1536x2-15-32", "256x2-64-64", "2048x2-32-64"] {
            let parsed = NnueArch::from_str(s).unwrap();
            assert_eq!(parsed.cli_name(), s);
            assert_eq!(parsed.to_string(), s);
        }
    }

    #[test]
    fn nnue_arch_parse_rejects_bad_format() {
        assert!(NnueArch::from_str("").is_err());
        assert!(NnueArch::from_str("256-32-32").is_err()); // x2 missing
        assert!(NnueArch::from_str("256x3-32-32").is_err()); // x3 not allowed
        assert!(NnueArch::from_str("256x2-32").is_err()); // L3 missing
        assert!(NnueArch::from_str("256x2-32-32-32").is_err()); // too many parts
        assert!(NnueArch::from_str("abcx2-32-32").is_err());
    }

    #[test]
    fn nnue_arch_parse_rejects_bad_dims() {
        // 0 dims はNG
        assert!(NnueArch::from_str("0x2-32-32").is_err());
        assert!(NnueArch::from_str("256x2-0-32").is_err());
        assert!(NnueArch::from_str("256x2-32-0").is_err());
        // L1 が 32 の倍数でない
        assert!(NnueArch::from_str("100x2-32-32").is_err());
        assert!(NnueArch::from_str("257x2-32-32").is_err());
    }

    /// `finalize_one_nnue_dir` が bullet 形式の checkpoint dir を `<NNNN>/`
    /// に rename し、`log.txt` を 9-column の `learn.log` に変換することを確認。
    /// per-superbatch save callback で呼ばれた場合の単発動作と等価。
    #[test]
    fn finalize_one_nnue_dir_renames_and_enriches() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-finalize-one-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("shogi_nnue_ka2-3");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("nn.bin"), b"dummy").unwrap();
        std::fs::write(src.join("state.bin"), b"dummy").unwrap();
        // bullet's raw 3-column log.txt (superbatch, batch, loss)
        std::fs::write(src.join("log.txt"), "3,32,0.123\n3,64,0.099\n").unwrap();

        let ctx = LogContext {
            eval_type: "SFNN_KA2",
            arch: "1536x2-15-32".to_string(),
            lr_start: 0.001,
            lr_gamma: 0.1,
            lr_step: 8,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/foo.hcpe".to_string(),
            sb_offset: 0,
            lr_step_positions: None,
        };

        let dst = finalize_one_nnue_dir(&tmp, &src, &ctx, /*epoch=*/ 1, /*idx=*/ 5, /*prior=*/ 0, /*test_metrics=*/ None)
            .expect("finalize ok");

        // src is gone, dst is `0005/`
        assert!(!src.exists(), "src dir should have been renamed away");
        assert_eq!(dst, tmp.join("0005"));
        assert!(dst.is_dir());
        // contents preserved
        assert!(dst.join("nn.bin").is_file());
        assert!(dst.join("state.bin").is_file());
        // log.txt removed, learn.log written with header + 2 rows
        assert!(!dst.join("log.txt").exists(), "log.txt should be deleted");
        let learn = std::fs::read_to_string(dst.join("learn.log")).unwrap();
        assert!(learn.starts_with(LEARN_LOG_HEADER), "learn.log should start with header");
        let body_lines: Vec<&str> = learn.lines().skip(1).filter(|l| !l.is_empty()).collect();
        assert_eq!(body_lines.len(), 2, "two body rows expected");
        // each row has 11 comma-separated fields (= LEARN_LOG_HEADER columns)
        // and the two test_value_* columns are "-" because we passed None.
        for row in &body_lines {
            assert_eq!(row.split(',').count(), 11, "row `{row}` should be 11 columns");
            assert!(row.starts_with("SFNN_KA2-1536x2-15-32,"));
            // Columns: eval, epoch, sb, batch, test_value_accuracy,
            // test_value_loss, train_value_loss, lr, lambda, positions, teacher
            // → indexes 4 and 5 should be "-"
            let cols: Vec<&str> = row.split(',').collect();
            assert_eq!(cols[4], "-", "test_value_accuracy should be '-' when no test_metrics");
            assert_eq!(cols[5], "-", "test_value_loss should be '-' when no test_metrics");
        }
        // cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `--tag` を指定すると、自動命名された出力フォルダ名の末尾に
    /// `-<tag>` が付くこと、`--output` 指定時は `--tag` が無視されて
    /// ユーザー指定パスがそのまま使われることを確認。
    #[test]
    fn output_dir_applies_tag_suffix() {
        use clap::Parser as _;

        // Baseline (no --tag, no --output): default name only.
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type", "NNUE_KP",
            "--teacher", "/dev/null",
        ])
        .unwrap();
        assert_eq!(
            args.output_dir(),
            std::path::PathBuf::from("checkpoints/NNUE_KP-256x2-32-32"),
        );

        // --tag appends `-<tag>` to the auto-derived name.
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type", "NNUE_KP",
            "--teacher", "/dev/null",
            "--tag", "lr0.001",
        ])
        .unwrap();
        assert_eq!(
            args.output_dir(),
            std::path::PathBuf::from("checkpoints/NNUE_KP-256x2-32-32-lr0.001"),
        );

        // --tag with SFNN: applied after the layerstack segment.
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type", "SFNN_KA2",
            "--arch", "1536x2-15-32",
            "--teacher", "/dev/null",
            "--tag", "exp7",
        ])
        .unwrap();
        assert_eq!(
            args.output_dir(),
            std::path::PathBuf::from(
                "checkpoints/SFNN_KA2-1536x2-15-32-king3-by-king3-exp7"
            ),
        );

        // Explicit --output wins; --tag is ignored.
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type", "NNUE_KP",
            "--teacher", "/dev/null",
            "--output", "/custom/path",
            "--tag", "ignored",
        ])
        .unwrap();
        assert_eq!(args.output_dir(), std::path::PathBuf::from("/custom/path"));

        // Empty --tag is treated as no tag (no trailing dash).
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type", "NNUE_KP",
            "--teacher", "/dev/null",
            "--tag", "",
        ])
        .unwrap();
        assert_eq!(
            args.output_dir(),
            std::path::PathBuf::from("checkpoints/NNUE_KP-256x2-32-32"),
        );
    }

    /// finalize_one_nnue_dir with Some(TestMetrics) emits actual values
    /// in the test_value_* columns rather than `-`.
    #[test]
    fn finalize_one_nnue_dir_emits_test_metrics_when_provided() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-finalize-with-metrics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("shogi_sfnn_ka2-7");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("nn.bin"), b"dummy").unwrap();
        std::fs::write(src.join("state.bin"), b"dummy").unwrap();
        std::fs::write(src.join("log.txt"), "7,32,0.111\n7,64,0.099\n").unwrap();

        let ctx = LogContext {
            eval_type: "NNUE_KA2",
            arch: "256x2-32-32".to_string(),
            lr_start: 0.001,
            lr_gamma: 0.1,
            lr_step: 8,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/foo.hcpe".to_string(),
            sb_offset: 0,
            lr_step_positions: None,
        };
        let metrics = TestMetrics { accuracy: 0.8765, loss: 0.0512 };
        let dst = finalize_one_nnue_dir(&tmp, &src, &ctx, 1, 9, 0, Some(metrics)).unwrap();
        let learn = std::fs::read_to_string(dst.join("learn.log")).unwrap();
        let body_lines: Vec<&str> = learn.lines().skip(1).filter(|l| !l.is_empty()).collect();
        assert_eq!(body_lines.len(), 2);
        for row in &body_lines {
            let cols: Vec<&str> = row.split(',').collect();
            assert_eq!(cols.len(), 11);
            // Expect formatted floats (6 decimal places per impl)
            assert_eq!(cols[4], "0.876500", "test_value_accuracy should be the metric");
            assert_eq!(cols[5], "0.051200", "test_value_loss should be the metric");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `LogContext.lr_step_positions` を設定すると、enrich の lr 列が
    /// positions-based formula で計算され、sb / lr_step とは独立に
    /// cumulative positions に応じて drop することを確認。
    #[test]
    fn enrich_with_lr_step_positions_uses_positions_based_formula() {
        let ctx = LogContext {
            eval_type: "NNUE_KP",
            arch: "256x2-32-32".to_string(),
            lr_start: 0.001,
            lr_gamma: 0.1,
            lr_step: 8, // sb-based, but should be ignored
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/round.hcpe".to_string(),
            sb_offset: 0,
            lr_step_positions: Some(800_000_000),
        };
        // batch=32 with prior=0: positions = 32*16384 = 524,288 → step 0 → lr 0.001
        // batch=6000 with prior=0: positions = 6000*16384 = 98,304,000 → step 0 → lr 0.001
        let raw = "1,32,0.07\n1,6000,0.06\n";
        let body = enrich_bullet_log_to_csv(raw, &ctx, 1, "nnue", 0, None);
        for row in body.lines() {
            let cols: Vec<&str> = row.split(',').collect();
            let lr: f32 = cols[7].parse().unwrap();
            assert!((lr - 0.001).abs() < 1e-7, "lr should still be 0.001, got {lr} from row {row}");
        }
        // Push position_offset to 800M; lr drops to 0.0001 once positions ≥ 800M
        let body2 = enrich_bullet_log_to_csv("1,32,0.05\n", &ctx, 1, "nnue", 800_000_000, None);
        let cols: Vec<&str> = body2.lines().next().unwrap().split(',').collect();
        let lr: f32 = cols[7].parse().unwrap();
        assert!((lr - 0.0001).abs() < 1e-7, "lr should drop to 0.0001 past 800M, got {lr}");
    }

    /// `LogContext.sb_offset` が enrich の sb / lr 列に正しく反映され、
    /// 教師変更による start_sb=1 reset でも sb 表示と LR が連続するか確認。
    #[test]
    fn enrich_with_sb_offset_emits_absolute_sb_and_shifted_lr() {
        let mut ctx = LogContext {
            eval_type: "NNUE_KP",
            arch: "256x2-32-32".to_string(),
            lr_start: 0.001,
            lr_gamma: 0.1,
            lr_step: 8,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/round2.hcpe".to_string(),
            sb_offset: 1, // = "round 1 saved sb 1, this run is the round 2 continuation"
            lr_step_positions: None,
        };
        // bullet's local sb in raw log is 1 (= first sb of round 2). With
        // sb_offset=1, the enriched row should display sb=2 and pull the
        // lr at sb=2 (= still 0.001 since 2/8 = 0).
        let raw = "1,32,0.07\n1,64,0.06\n";
        let body = enrich_bullet_log_to_csv(&raw, &ctx, /*epoch=*/ 1, "nnue", /*prior=*/ 60_000_000, None);
        let rows: Vec<&str> = body.lines().collect();
        assert_eq!(rows.len(), 2);
        // Each row: eval, epoch, sb, batch, ta, tl, train, lr, lambda, positions, teacher
        let cols0: Vec<&str> = rows[0].split(',').collect();
        assert_eq!(cols0[2], "2", "absolute sb (= 1 + offset 1)");
        assert_eq!(cols0[7], "0.001", "lr at absolute sb 2 (no decay yet)");
        // positions column: 60M (prior) + 0*sb_size + 32*16384 = 60_524_288
        assert_eq!(cols0[9], "60524288", "positions = prior + (local_sb-1)*sb_size + b*batch_size");

        // Cross a real LR boundary: sb_offset=8 puts absolute sb at 9 = past
        // step boundary, lr should drop by gamma (0.001 * 0.1 = 0.0001).
        ctx.sb_offset = 8;
        let body2 = enrich_bullet_log_to_csv(&"1,32,0.05\n", &ctx, 1, "nnue", 0, None);
        let cols: Vec<&str> = body2.lines().next().unwrap().split(',').collect();
        assert_eq!(cols[2], "9", "absolute sb (= 1 + offset 8)");
        // lr_at(9): steps = (9-1)/8 = 1 → 0.001 * 0.1^1 ≈ 0.0001 (f32 rounding)
        let lr_value: f32 = cols[7].parse().expect("lr column is a float");
        assert!(
            (lr_value - 0.0001).abs() < 1e-7,
            "expected lr ≈ 0.0001 at absolute sb 9, got {lr_value} (raw: {})",
            cols[7]
        );
    }

    /// `read_latest_saved_superbatch` が `<NNNN>/learn.log` の sb 列を
    /// 正しく拾うことを確認。auto-resume の出発点。
    #[test]
    fn read_latest_saved_superbatch_picks_max_sb() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-resume-sb-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        // 空 dir → None
        assert_eq!(read_latest_saved_superbatch(&tmp), None);

        // 0001/ だけあって learn.log 無し → None
        let d1 = tmp.join("0001");
        std::fs::create_dir(&d1).unwrap();
        assert_eq!(read_latest_saved_superbatch(&tmp), None);

        // 0001/learn.log の sb 列が 1 → 1 が返る
        std::fs::write(
            d1.join("learn.log"),
            format!(
                "{LEARN_LOG_HEADER}\nNNUE_KP-256x2-32-32,1,1,32,0.1,0.001,1.000,524288,t.hcpe\n\
                 NNUE_KP-256x2-32-32,1,1,64,0.09,0.001,1.000,1048576,t.hcpe\n"
            ),
        )
        .unwrap();
        assert_eq!(read_latest_saved_superbatch(&tmp), Some(1));

        // 0004/ も追加 (sb=4 のログ) → 最高番号 dir の sb が返る
        let d4 = tmp.join("0004");
        std::fs::create_dir(&d4).unwrap();
        std::fs::write(
            d4.join("learn.log"),
            format!(
                "{LEARN_LOG_HEADER}\nNNUE_KP-256x2-32-32,1,4,32,0.06,0.001,1.000,2097152,t.hcpe\n"
            ),
        )
        .unwrap();
        assert_eq!(read_latest_saved_superbatch(&tmp), Some(4));

        // 不正 dir 名 (foo/) は無視
        std::fs::create_dir(tmp.join("foo")).unwrap();
        assert_eq!(read_latest_saved_superbatch(&tmp), Some(4));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `read_latest_saved_teacher` が `<NNNN>/learn.log` の teacher 列
    /// (= 11-列の最終フィールド) を取り、auto-resume の教師変更検出に
    /// 使えることを確認。
    #[test]
    fn read_latest_saved_teacher_picks_last_teacher() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-resume-teacher-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        // 空 dir → None
        assert_eq!(read_latest_saved_teacher(&tmp), None);

        // 0001 だけあって learn.log 無し → None
        let d1 = tmp.join("0001");
        std::fs::create_dir(&d1).unwrap();
        assert_eq!(read_latest_saved_teacher(&tmp), None);

        // 11-列 row が 1 つあれば teacher が拾える
        std::fs::write(
            d1.join("learn.log"),
            format!(
                "{LEARN_LOG_HEADER}\nNNUE_KP-256x2-32-32,1,1,32,-,-,0.1,0.001,1.000,524288,foo.hcpe\n"
            ),
        )
        .unwrap();
        assert_eq!(read_latest_saved_teacher(&tmp), Some("foo.hcpe".to_string()));

        // 0004 を追加して bar.hcpe にしたら、最新 dir の teacher が返る
        let d4 = tmp.join("0004");
        std::fs::create_dir(&d4).unwrap();
        std::fs::write(
            d4.join("learn.log"),
            format!(
                "{LEARN_LOG_HEADER}\nNNUE_KP-256x2-32-32,1,4,32,0.6,0.05,0.06,0.001,1.000,2097152,bar.hcpe\n"
            ),
        )
        .unwrap();
        assert_eq!(read_latest_saved_teacher(&tmp), Some("bar.hcpe".to_string()));

        // 9-列のレガシー row は無視される (parts.len() < 11 で skip)
        let d5 = tmp.join("0005");
        std::fs::create_dir(&d5).unwrap();
        std::fs::write(
            d5.join("learn.log"),
            format!("{LEARN_LOG_HEADER}\nNNUE_KP,1,5,32,0.5,0.001,1.000,3000,legacy.hcpe\n"),
        )
        .unwrap();
        assert_eq!(read_latest_saved_teacher(&tmp), None, "legacy 9-col row should be skipped");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `finalize_nnue_dirs` がレガシー (= callback 未通過) dir をまとめて
    /// 処理し、空のときは `(0, 0)` を返すことを確認。
    #[test]
    fn finalize_nnue_dirs_handles_empty_gracefully() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-finalize-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let ctx = LogContext {
            eval_type: "NNUE_KA2",
            arch: "256x2-32-32".to_string(),
            lr_start: 0.001,
            lr_gamma: 0.1,
            lr_step: 8,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/foo.hcpe".to_string(),
            sb_offset: 0,
            lr_step_positions: None,
        };
        let res = finalize_nnue_dirs(&tmp, &ctx, "shogi_nnue_ka2", 0).unwrap();
        assert_eq!(res, (0, 0), "empty dir should return (0,0), not Err");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
