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

    bulletou --eval-type NNUE_HALFKP                                        classic HalfKP NNUE
    bulletou --eval-type NNUE_HALFKP --arch NNUE_halfkp_1024x2_8_64         larger HalfKP NNUE
    bulletou --eval-type NNUE_KP                                            K+P NNUE
    bulletou --eval-type NNUE_KA2 --arch NNUE_ka2_256x2_64_64               K+A2 NNUE
    bulletou --eval-type NNUE_HALFKPE9                  HalfKP with per-square effect-count buckets
    bulletou --eval-type NNUE_HALFKPVM                  HalfKP with file-mirror (~half input dims of HalfKP)
    bulletou --eval-type SFNN_HALFKA2HM --arch SFNN_halfkahm2_1536_15_32_k3k3
    bulletou --eval-type SFNN_HALFKA2   --arch SFNN_halfka2_1024_7_64_k3k3

`--arch` uses the YaneuraOu Makefile architecture name with the
`YANEURAOU_ENGINE_` prefix removed.

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

#[cfg(feature = "cuda-cpp-backend")]
use std::collections::BTreeMap;
#[cfg(feature = "cuda-cpp-backend")]
use std::path::Path;
use std::path::PathBuf;

use bullet_compiler::tensor::TValue;
#[cfg(feature = "cuda-cpp-backend")]
use bulletou_lib::value::nnue_save_sfnn1536::{
    FT_HASH_SFNN, KHASH_SFNN, NETWORK_HASH_SFNN, QA as SFNN_QA, QB as SFNN_QB,
};
use bulletou_lib::{
    game::inputs::{
        ShogiHalfKP, ShogiHalfKPvm, ShogiHalfKa2, ShogiHalfKaHm1, ShogiHalfKaHm2, ShogiHalfKpe9, ShogiKa2, ShogiKk,
        ShogiKkp, ShogiKp, ShogiKpp, SparseInputType,
    },
    game::outputs::{OutputBuckets, ShogiLayerStackBucket9},
    nn::{ExecutionContext, ModelNode, optimiser},
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    trainer::schedule::lr::LrScheduler,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, wdl},
        settings::LocalSettings,
    },
    validate::{
        ValidationLossKind, compute_sign_accuracy_with_loss, read_random_teacher_positions,
        read_teacher_positions_prefix,
    },
    value::{
        ValueTrainer, ValueTrainerBuilder,
        loader::{DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader},
        nnue_save::{
            Activation as NnueActivation, NnueFeatureSet, ft_hash_bytes, header_bytes, l1_bias_scale,
            network_layer_hash_bytes, pad_weights_for_simd, pad32 as nnue_pad32,
        },
        nnue_save_sfnn1536::{
            LEB128_MAGIC, NNUE_VERSION as SFNN_NNUE_VERSION, Sfnn1536SaveParams, build_sfnn_1536_save_format,
        },
        yaneuraou_kppt::{
            KppFormat, STATE_BACKEND_BULLET, bundle_component_state, parse_model_weights_bin, save_yaneuraou_eval,
            unbundle_component_state, write_state_backend_marker,
        },
    },
};
use clap::{ArgAction, Parser, ValueEnum};

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
    /// (default `NNUE_halfkp_256x2_32_32`).
    NnueHalfkp,
    /// NNUE K-P. YaneuraOu kp_256x2-32-32 — same 4-layer ClippedReLU network
    /// as halfkp_256x2-32-32, but the input is `FeatureSet<K, P>` (K = 162
    /// king features, P = 1548 piece features per perspective; 1710 total)
    /// instead of HalfKP's (king × piece) cross product. Architecture is
    /// selected via `--arch` (default `NNUE_kp_256x2_32_32`).
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
    /// fc_1 → fc_2 → +PSQT bypass). Bucketing is selected by the `--arch`
    /// LayerStack suffix.
    /// `--arch SFNN_halfkahm1_1536_15_32_k3k3` matches the corresponding
    /// YaneuraOu SFNN dynamic build.
    SfnnHalfka1hm,
    /// SFNN-1536 with `HalfKA_hm2` input (= strict v2, enemy king
    /// collapsed onto friend plane, 73,305 dim). This is the variant
    /// `YANEURAOU_ENGINE_SFNN1536` alias uses. Identical network topology
    /// to `SFNN_HALFKA1HM`, only the input feature differs.
    SfnnHalfka2hm,
    /// SFNN with `HalfKA2` input (= non-mirrored 81 king buckets, enemy king
    /// collapsed onto the friend-king plane). Matches YaneuraOu dynamic
    /// `SFNN_halfka2_*_k3k3` architecture names.
    SfnnHalfka2,
    /// SFNN-1536 with `K + A2` input (= YaneuraOu `FeatureSet<K, A2>`,
    /// 1791 dim). K (162 king features) + A2 (1629 piece features,
    /// kings collapsed onto friend plane). No file-mirror, so input
    /// dimension is much smaller than HalfKA_hm2 but representation
    /// is also weaker (no king-anchor cross product). Matches
    /// YaneuraOu's `YANEURAOU_ENGINE_SFNN_ka2_*` build.
    /// Identical network topology and LayerStacks bucketing as the
    /// other SFNN variants.
    SfnnKa2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
enum BackendKind {
    /// Existing generic Bullet backend. Supports every eval type.
    Bullet,
    /// Windows-native C++/CUDA fixed-layout backend bring-up path.
    CudaCpp,
}

impl BackendKind {
    fn cli_name(self) -> &'static str {
        match self {
            BackendKind::Bullet => "bullet",
            BackendKind::CudaCpp => "cuda-cpp",
        }
    }
}

/// LayerStack bucketing scheme for the SFNN family. Selects which
/// per-position bucket index is used to choose the active MLP stack
/// from the LayerStacks array, and implicitly determines the **stack
/// count** (the network model uses one bucket per stack).
///
/// Currently `k3k3(king3-by-king3)` is the only choice — it matches YaneuraOu's
/// `stack_index_for_nnue` so the trained `nn.bin` is engine-loadable
/// and evaluation matches between training and inference. Other
/// schemes implemented in `bulletou_lib::game::outputs::ShogiLayerStackBucket9`
/// (e.g. `Ply9`, `Progress8*`) are intentionally not exposed here
/// because they cannot be used with YaneuraOu's engine; they remain
/// available to `examples/shogi_layerstack.rs` for rshogi-style research.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LayerStackMode {
    /// 3 × 3 = 9 stacks, indexed by `(friend_king_rank/3, enemy_king_rank/3)`.
    /// Matches YaneuraOu `stack_index_for_nnue` byte-for-byte.
    #[default]
    Kingrank3by3,
}

impl LayerStackMode {
    /// Human-facing display name.
    fn cli_name(self) -> &'static str {
        match self {
            LayerStackMode::Kingrank3by3 => "k3k3(king3-by-king3)",
        }
    }

    /// Short YaneuraOu architecture suffix.
    fn arch_suffix(self) -> &'static str {
        match self {
            LayerStackMode::Kingrank3by3 => "k3k3",
        }
    }

    /// Number of LayerStacks this bucketing scheme produces.
    fn num_stacks(self) -> usize {
        match self {
            LayerStackMode::Kingrank3by3 => 9,
        }
    }
}

/// NNUE architecture name in the YaneuraOu Makefile form with the
/// `YANEURAOU_ENGINE_` prefix removed.
///
/// Examples:
/// - `NNUE_halfkp_256x2_32_32`
/// - `NNUE_ka2_256x2_64_64`
/// - `SFNN_halfka2_1024_7_64_k3k3`
/// - `SFNN_halfkahm2_1536_15_32_king3_by_king3`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NnueArch {
    family: NnueArchFamily,
    feature: NnueArchFeature,
    l1: usize,
    l2: usize,
    l3: usize,
    layerstack: Option<LayerStackMode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NnueArchFamily {
    Nnue,
    Sfnn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NnueArchFeature {
    Halfkp,
    Kp,
    Ka2,
    Halfkpe9,
    Halfkpvm,
    Halfka2,
    Halfkahm1,
    Halfkahm2,
}

impl NnueArch {
    fn new(
        family: NnueArchFamily,
        feature: NnueArchFeature,
        l1: usize,
        l2: usize,
        l3: usize,
        layerstack: Option<LayerStackMode>,
    ) -> Self {
        Self { family, feature, l1, l2, l3, layerstack }
    }

    /// `(l1, l2, l3)` triple.
    fn dims(self) -> (usize, usize, usize) {
        (self.l1, self.l2, self.l3)
    }

    /// The arch's canonical CLI value.
    fn cli_name(self) -> String {
        match self.family {
            NnueArchFamily::Nnue => {
                format!("NNUE_{}_{}x2_{}_{}", self.feature.arch_suffix(), self.l1, self.l2, self.l3)
            }
            NnueArchFamily::Sfnn => {
                let layerstack = self.layerstack.unwrap_or(LayerStackMode::Kingrank3by3);
                format!(
                    "SFNN_{}_{}_{}_{}_{}",
                    self.feature.arch_suffix(),
                    self.l1,
                    self.l2,
                    self.l3,
                    layerstack.arch_suffix()
                )
            }
        }
    }

    fn default_for_eval_type(eval_type: EvalType) -> Self {
        match eval_type {
            EvalType::NnueHalfkp => Self::new(NnueArchFamily::Nnue, NnueArchFeature::Halfkp, 256, 32, 32, None),
            EvalType::NnueKp => Self::new(NnueArchFamily::Nnue, NnueArchFeature::Kp, 256, 32, 32, None),
            EvalType::NnueKa2 => Self::new(NnueArchFamily::Nnue, NnueArchFeature::Ka2, 256, 32, 32, None),
            EvalType::NnueHalfkpe9 => Self::new(NnueArchFamily::Nnue, NnueArchFeature::Halfkpe9, 256, 32, 32, None),
            EvalType::NnueHalfkpvm => Self::new(NnueArchFamily::Nnue, NnueArchFeature::Halfkpvm, 256, 32, 32, None),
            EvalType::SfnnHalfka1hm => Self::new(
                NnueArchFamily::Sfnn,
                NnueArchFeature::Halfkahm1,
                1536,
                15,
                32,
                Some(LayerStackMode::Kingrank3by3),
            ),
            EvalType::SfnnHalfka2hm => Self::new(
                NnueArchFamily::Sfnn,
                NnueArchFeature::Halfkahm2,
                1536,
                15,
                32,
                Some(LayerStackMode::Kingrank3by3),
            ),
            EvalType::SfnnHalfka2 => Self::new(
                NnueArchFamily::Sfnn,
                NnueArchFeature::Halfka2,
                1536,
                15,
                32,
                Some(LayerStackMode::Kingrank3by3),
            ),
            EvalType::SfnnKa2 => {
                Self::new(NnueArchFamily::Sfnn, NnueArchFeature::Ka2, 1536, 15, 32, Some(LayerStackMode::Kingrank3by3))
            }
            EvalType::Kppt | EvalType::KppKkpt => {
                Self::new(NnueArchFamily::Nnue, NnueArchFeature::Halfkp, 256, 32, 32, None)
            }
        }
    }

    fn expected_eval_type(self) -> EvalType {
        match (self.family, self.feature) {
            (NnueArchFamily::Nnue, NnueArchFeature::Halfkp) => EvalType::NnueHalfkp,
            (NnueArchFamily::Nnue, NnueArchFeature::Kp) => EvalType::NnueKp,
            (NnueArchFamily::Nnue, NnueArchFeature::Ka2) => EvalType::NnueKa2,
            (NnueArchFamily::Nnue, NnueArchFeature::Halfkpe9) => EvalType::NnueHalfkpe9,
            (NnueArchFamily::Nnue, NnueArchFeature::Halfkpvm) => EvalType::NnueHalfkpvm,
            (NnueArchFamily::Sfnn, NnueArchFeature::Halfkahm1) => EvalType::SfnnHalfka1hm,
            (NnueArchFamily::Sfnn, NnueArchFeature::Halfkahm2) => EvalType::SfnnHalfka2hm,
            (NnueArchFamily::Sfnn, NnueArchFeature::Halfka2) => EvalType::SfnnHalfka2,
            (NnueArchFamily::Sfnn, NnueArchFeature::Ka2) => EvalType::SfnnKa2,
            (NnueArchFamily::Nnue, NnueArchFeature::Halfka2)
            | (NnueArchFamily::Nnue, NnueArchFeature::Halfkahm1)
            | (NnueArchFamily::Nnue, NnueArchFeature::Halfkahm2)
            | (NnueArchFamily::Sfnn, NnueArchFeature::Halfkp)
            | (NnueArchFamily::Sfnn, NnueArchFeature::Kp)
            | (NnueArchFamily::Sfnn, NnueArchFeature::Halfkpe9)
            | (NnueArchFamily::Sfnn, NnueArchFeature::Halfkpvm) => {
                unreachable!("unsupported parsed architecture combination: {self:?}")
            }
        }
    }

    fn validate_dims(self, original: &str) -> Result<Self, String> {
        if self.l1 == 0 || self.l2 == 0 || self.l3 == 0 {
            return Err(format!("invalid arch `{original}`: L1/L2/L3 must all be > 0"));
        }
        if self.l1 % 32 != 0 {
            return Err(format!(
                "invalid arch `{original}`: L1 (= {}) must be a multiple of 32 (FT SIMD-padding requirement)",
                self.l1
            ));
        }
        Ok(self)
    }
}

impl NnueArchFeature {
    fn parse(raw: &str, family: NnueArchFamily, original: &str) -> Result<Self, String> {
        let normalized = raw.to_ascii_lowercase();
        let feature = match normalized.as_str() {
            "halfkp" => NnueArchFeature::Halfkp,
            "kp" => NnueArchFeature::Kp,
            "ka2" => NnueArchFeature::Ka2,
            "halfkpe9" => NnueArchFeature::Halfkpe9,
            "halfkpvm" => NnueArchFeature::Halfkpvm,
            "halfka2" => NnueArchFeature::Halfka2,
            "halfkahm1" => NnueArchFeature::Halfkahm1,
            "halfkahm2" => NnueArchFeature::Halfkahm2,
            _ => return Err(format!("invalid arch `{original}`: unsupported feature `{raw}`")),
        };

        let supported = matches!(
            (family, feature),
            (NnueArchFamily::Nnue, NnueArchFeature::Halfkp)
                | (NnueArchFamily::Nnue, NnueArchFeature::Kp)
                | (NnueArchFamily::Nnue, NnueArchFeature::Ka2)
                | (NnueArchFamily::Nnue, NnueArchFeature::Halfkpe9)
                | (NnueArchFamily::Nnue, NnueArchFeature::Halfkpvm)
                | (NnueArchFamily::Sfnn, NnueArchFeature::Halfkahm1)
                | (NnueArchFamily::Sfnn, NnueArchFeature::Halfkahm2)
                | (NnueArchFamily::Sfnn, NnueArchFeature::Halfka2)
                | (NnueArchFamily::Sfnn, NnueArchFeature::Ka2)
        );
        if !supported {
            let family_name = match family {
                NnueArchFamily::Nnue => "NNUE",
                NnueArchFamily::Sfnn => "SFNN",
            };
            return Err(format!("invalid arch `{original}`: feature `{raw}` is not supported for {family_name}"));
        }

        Ok(feature)
    }

    fn arch_suffix(self) -> &'static str {
        match self {
            NnueArchFeature::Halfkp => "halfkp",
            NnueArchFeature::Kp => "kp",
            NnueArchFeature::Ka2 => "ka2",
            NnueArchFeature::Halfkpe9 => "halfkpe9",
            NnueArchFeature::Halfkpvm => "halfkpvm",
            NnueArchFeature::Halfka2 => "halfka2",
            NnueArchFeature::Halfkahm1 => "halfkahm1",
            NnueArchFeature::Halfkahm2 => "halfkahm2",
        }
    }
}

impl std::fmt::Display for NnueArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.cli_name())
    }
}

impl std::str::FromStr for NnueArch {
    type Err = String;

    /// Parse a YaneuraOu architecture name without the `YANEURAOU_ENGINE_`
    /// prefix. This is intentionally not compatible with the old
    /// `<L1>x2-<L2>-<L3>` shorthand.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.to_ascii_uppercase().starts_with("YANEURAOU_ENGINE_") {
            return Err(format!("invalid arch `{s}`: pass the architecture name without the YANEURAOU_ENGINE_ prefix"));
        }
        if s.eq_ignore_ascii_case("SFNN1536") {
            return NnueArch::new(
                NnueArchFamily::Sfnn,
                NnueArchFeature::Halfkahm2,
                1536,
                15,
                32,
                Some(LayerStackMode::Kingrank3by3),
            )
            .validate_dims(s);
        }

        let tokens: Vec<&str> = s.split('_').collect();
        if tokens.len() < 5 {
            return Err(format!(
                "invalid arch `{s}`: expected `NNUE_<feature>_<L1>x2_<L2>_<L3>` or `SFNN_<feature>_<FT>_<H1>_<H2>_k3k3`"
            ));
        }

        let family = match tokens[0].to_ascii_uppercase().as_str() {
            "NNUE" => NnueArchFamily::Nnue,
            "SFNN" => NnueArchFamily::Sfnn,
            other => {
                return Err(format!("invalid arch `{s}`: first component must be NNUE or SFNN, got `{other}`"));
            }
        };
        let feature = NnueArchFeature::parse(tokens[1], family, s)?;

        match family {
            NnueArchFamily::Nnue => {
                if tokens.len() != 5 {
                    return Err(format!("invalid arch `{s}`: expected `NNUE_<feature>_<L1>x2_<L2>_<L3>`"));
                }
                let l1_part = tokens[2]
                    .strip_suffix("x2")
                    .or_else(|| tokens[2].strip_suffix("X2"))
                    .ok_or_else(|| format!("invalid arch `{s}`: `{}` must end with `x2`", tokens[2]))?;
                let l1: usize = l1_part
                    .parse()
                    .map_err(|_| format!("invalid arch `{s}`: L1 `{l1_part}` is not a positive integer"))?;
                let l2: usize = tokens[3]
                    .parse()
                    .map_err(|_| format!("invalid arch `{s}`: L2 `{}` is not a positive integer", tokens[3]))?;
                let l3: usize = tokens[4]
                    .parse()
                    .map_err(|_| format!("invalid arch `{s}`: L3 `{}` is not a positive integer", tokens[4]))?;
                NnueArch::new(family, feature, l1, l2, l3, None).validate_dims(s)
            }
            NnueArchFamily::Sfnn => {
                if tokens.len() < 6 {
                    return Err(format!("invalid arch `{s}`: expected `SFNN_<feature>_<FT>_<H1>_<H2>_k3k3`"));
                }
                let l1: usize = tokens[2]
                    .parse()
                    .map_err(|_| format!("invalid arch `{s}`: FT `{}` is not a positive integer", tokens[2]))?;
                let l2: usize = tokens[3]
                    .parse()
                    .map_err(|_| format!("invalid arch `{s}`: H1 `{}` is not a positive integer", tokens[3]))?;
                let l3: usize = tokens[4]
                    .parse()
                    .map_err(|_| format!("invalid arch `{s}`: H2 `{}` is not a positive integer", tokens[4]))?;
                let layerstack = match tokens[5..].join("_").to_ascii_lowercase().as_str() {
                    "k3k3" | "king3_by_king3" => LayerStackMode::Kingrank3by3,
                    "ls9" => {
                        return Err(format!(
                            "invalid arch `{s}`: ls9 is no longer supported; use k3k3 or king3_by_king3"
                        ));
                    }
                    other => {
                        return Err(format!(
                            "invalid arch `{s}`: unsupported SFNN layer stack `{other}`; expected k3k3 or king3_by_king3"
                        ));
                    }
                };
                NnueArch::new(family, feature, l1, l2, l3, Some(layerstack)).validate_dims(s)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "SCREAMING_SNAKE_CASE")]
enum NerfEvalType {
    /// Generic SFNN layout. The concrete input feature does not affect
    /// nerf offsets because this command leaves the FeatureTransformer intact.
    Sfnn,
    /// SFNN with HalfKA_hm1 input.
    SfnnHalfka1hm,
    /// SFNN with HalfKA_hm2 input.
    SfnnHalfka2hm,
    /// SFNN with HalfKA2 input, used by dynamic YaneuraOu
    /// `SFNN_halfka2_*` builds.
    SfnnHalfka2,
    /// SFNN with K+A2 input.
    SfnnKa2,
}

impl Default for NerfEvalType {
    fn default() -> Self {
        NerfEvalType::Sfnn
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NerfLayerSet {
    fc0: bool,
    fc1: bool,
    fc2: bool,
}

impl Default for NerfLayerSet {
    fn default() -> Self {
        Self { fc0: false, fc1: true, fc2: true }
    }
}

impl std::str::FromStr for NerfLayerSet {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut set = Self { fc0: false, fc1: false, fc2: false };
        for raw in s.split(',') {
            let token = raw.trim().to_ascii_lowercase();
            if token.is_empty() {
                continue;
            }
            match token.as_str() {
                "all" => {
                    set.fc0 = true;
                    set.fc1 = true;
                    set.fc2 = true;
                }
                "fc0" | "l1" => set.fc0 = true,
                "fc1" | "l2" => set.fc1 = true,
                "fc2" | "out" | "output" => set.fc2 = true,
                _ => {
                    return Err(format!("invalid --layers token `{raw}`: expected comma-separated fc0,fc1,fc2 or all"));
                }
            }
        }
        if !(set.fc0 || set.fc1 || set.fc2) {
            return Err("--layers must select at least one of fc0,fc1,fc2".to_string());
        }
        Ok(set)
    }
}

impl std::fmt::Display for NerfLayerSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if self.fc0 {
            parts.push("fc0");
        }
        if self.fc1 {
            parts.push("fc1");
        }
        if self.fc2 {
            parts.push("fc2");
        }
        write!(f, "{}", parts.join(","))
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "bulletou nerf")]
#[command(about = "Post-process a YaneuraOu nn.bin by adding reproducible ±1 noise to selected i8 weights")]
struct NerfArgs {
    /// Input YaneuraOu-compatible `nn.bin`.
    #[arg(long)]
    input: PathBuf,

    /// Output path for the nerfed `nn.bin`.
    #[arg(long)]
    output: PathBuf,

    /// Evaluation file layout. Currently only SFNN-style layouts are
    /// supported; standard NNUE layouts can be added later.
    #[arg(long, value_enum, default_value = "SFNN")]
    eval_type: NerfEvalType,

    /// Network architecture in YaneuraOu Makefile form with the
    /// `YANEURAOU_ENGINE_` prefix removed, e.g.
    /// `SFNN_halfka2_1024_7_64_k3k3`.
    #[arg(long)]
    arch: NnueArch,

    /// Comma-separated layer list. Only i8 weights are changed; biases,
    /// FeatureTransformer, hashes, and padding weights are left intact.
    #[arg(long, default_value = "fc2,fc1")]
    layers: NerfLayerSet,

    /// Number of random +/-1 mutation attempts. The same weight may be
    /// selected multiple times.
    #[arg(long)]
    count: usize,

    /// RNG seed. Defaults to 1 for reproducible output.
    #[arg(long, default_value = "1")]
    seed: u64,
}

impl NerfArgs {
    fn effective_layerstack(&self) -> LayerStackMode {
        self.arch.layerstack.unwrap_or(LayerStackMode::Kingrank3by3)
    }

    fn validate_arch_flags(&self) -> Result<(), String> {
        if self.arch.family != NnueArchFamily::Sfnn {
            return Err(format!(
                "--arch {} is not an SFNN architecture; nerf currently supports only SFNN nn.bin layouts",
                self.arch.cli_name()
            ));
        }
        Ok(())
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
            EvalType::SfnnHalfka2 => "shogi_sfnn_halfka2",
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
            | EvalType::SfnnHalfka2
            | EvalType::SfnnKa2 => true,
        }
    }

    fn supports_nnue_pytorch_wrm_loss(self) -> bool {
        self.uses_arch()
    }

    /// Does this eval type use LayerStacks? Only the SFNN family
    /// (LayerStacks-based architectures) does; the rest of the NNUE family
    /// is single-stack.
    fn uses_layerstack(self) -> bool {
        matches!(self, EvalType::SfnnHalfka1hm | EvalType::SfnnHalfka2hm | EvalType::SfnnHalfka2 | EvalType::SfnnKa2)
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
            EvalType::SfnnHalfka2 => "SFNN_HALFKA2",
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

/// Positions-based LR scheduler. Drops `start * gamma^n` where
/// `n = (prior_positions + in_run_positions) / positions_per_step`.
/// Decoupled from bullet's superbatch counter so the schedule tracks
/// actual data trained across rounds, regardless of how the user
/// chunked the teacher into `--teacher` invocations.
///
/// `prior_positions` is carried over from the existing top-level
/// `learn.log` so a multi-round / resumed workflow has a continuous
/// position count. The `lr(batch, sb)` callback computes the in-run
/// contribution from bullet's `(batch, sb)` (with `sb` here being
/// bullet's local counter, which restarts at 1 each `trainer.run`
/// call in the chunk loop).
#[derive(Clone, Debug)]
struct GeometricLR {
    start: f32,
    min: f32,
    period_positions: u64,
    prior_positions: u64,
    batch_size: usize,
    batches_per_superbatch: usize,
}

impl GeometricLR {
    /// Pure formula: LR for a given total cumulative position count.
    /// Used by both `LrScheduler::lr` and the `learn.log` enrich path
    /// so the trainer's LR and the logged LR always agree.
    ///
    /// `lr(t) = start * (min/start)^t` where `t = (total % period) /
    /// period`. Geometric interpolation in log space: at t=0 → start
    /// (lr_max), t=1 → min (lr_min). Warm restart at cycle boundary
    /// (= each `period_positions`), mirroring `CosineLR`.
    fn lr_at_positions(start: f32, min: f32, period: u64, total: u64) -> f32 {
        if period == 0 {
            return start;
        }
        let in_cycle = (total % period) as f64;
        let t = in_cycle / period as f64;
        let s = start as f64;
        // min > 0 should be validated at CLI parse; clamp here defensively so
        // (min/start)^t doesn't degenerate to 0 for any t>0.
        let m = (min as f64).max(1e-12);
        let lr = s * (m / s).powf(t);
        lr as f32
    }
}

impl LrScheduler for GeometricLR {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let in_run =
            ((superbatch.saturating_sub(1) * self.batches_per_superbatch + batch) as u64) * (self.batch_size as u64);
        let total = self.prior_positions + in_run;
        Self::lr_at_positions(self.start, self.min, self.period_positions, total)
    }

    fn colourful(&self) -> String {
        format!(
            "geometric: start {} min {} period {} positions (cumulative, prior {})",
            self.start, self.min, self.period_positions, self.prior_positions
        )
    }
}

/// StepLR with an epoch-local warm restart. Drops LR by a fixed gamma every
/// `step_positions` trained positions within the current epoch, then resets
/// to `start` at the next epoch boundary.
#[derive(Clone, Debug)]
struct StepLR {
    start: f32,
    min: f32,
    gamma: f32,
    step_positions: u64,
    period_positions: u64,
    prior_positions: u64,
    batch_size: usize,
    batches_per_superbatch: usize,
}

impl StepLR {
    fn lr_at_positions(
        start: f32,
        min: f32,
        gamma: f32,
        step_positions: u64,
        period_positions: u64,
        total: u64,
    ) -> f32 {
        if step_positions == 0 {
            return start;
        }
        let epoch_pos = if period_positions == 0 { total } else { total % period_positions };
        let steps = epoch_pos / step_positions;
        let lr = (start as f64) * (gamma as f64).powf(steps as f64);
        (lr as f32).max(min)
    }
}

impl LrScheduler for StepLR {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let in_run =
            ((superbatch.saturating_sub(1) * self.batches_per_superbatch + batch) as u64) * (self.batch_size as u64);
        let total = self.prior_positions + in_run;
        Self::lr_at_positions(self.start, self.min, self.gamma, self.step_positions, self.period_positions, total)
    }

    fn colourful(&self) -> String {
        format!(
            "step: start {} min {} gamma {} every {} positions, period {} positions (prior {})",
            self.start, self.min, self.gamma, self.step_positions, self.period_positions, self.prior_positions
        )
    }
}

/// Cosine annealing with warm restart (SGDR style), positions-based.
///
/// Mirrors [`GeometricLR`] structurally so the two schedules can be
/// dropped into the same training run as alternatives. Within each
/// `period_positions`-long cycle the LR sweeps `start` → `min`
/// following the half-cosine curve; at cycle boundaries it snaps back
/// to `start` (warm restart). Bullet's `sb`-reset at each epoch lines
/// up with `in_run = 0` and the cycle index reset, so when
/// `period_positions == one epoch`, each epoch is exactly one full
/// cosine cycle — apples-to-apples with the stepwise schedule which
/// also resets at epoch boundaries.
#[derive(Clone, Debug)]
struct CosineLR {
    start: f32,
    min: f32,
    period_positions: u64,
    prior_positions: u64,
    batch_size: usize,
    batches_per_superbatch: usize,
}

impl CosineLR {
    /// Pure formula: LR for a given cumulative position count.
    /// Used by both `LrScheduler::lr` and the `learn.log` enrich path.
    fn lr_at_positions(start: f32, min: f32, period: u64, total: u64) -> f32 {
        if period == 0 {
            return start;
        }
        let in_cycle = (total % period) as f64;
        let t = in_cycle / period as f64;
        let cos_val = (std::f64::consts::PI * t).cos();
        let lr = min as f64 + 0.5 * (start - min) as f64 * (1.0 + cos_val);
        lr as f32
    }
}

impl LrScheduler for CosineLR {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let in_run =
            ((superbatch.saturating_sub(1) * self.batches_per_superbatch + batch) as u64) * (self.batch_size as u64);
        let total = self.prior_positions + in_run;
        Self::lr_at_positions(self.start, self.min, self.period_positions, total)
    }

    fn colourful(&self) -> String {
        format!(
            "cosine: start {} min {} period {} positions (cumulative, prior {})",
            self.start, self.min, self.period_positions, self.prior_positions
        )
    }
}

#[derive(Clone, Debug)]
struct FixedLR {
    value: f32,
}

impl LrScheduler for FixedLR {
    fn lr(&self, _batch: usize, _superbatch: usize) -> f32 {
        self.value
    }

    fn colourful(&self) -> String {
        format!("plateau: fixed lr {}", self.value)
    }
}

/// Wrapper enum so the macro can pass either schedule into bullet's
/// generic `Trainer::train_custom` (which takes a single
/// `impl LrScheduler` per run).
#[derive(Clone, Debug)]
enum LrSchedulerImpl {
    Step(StepLR),
    Geometric(GeometricLR),
    Cos(CosineLR),
    Fixed(FixedLR),
}

impl LrScheduler for LrSchedulerImpl {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        match self {
            LrSchedulerImpl::Step(s) => s.lr(batch, superbatch),
            LrSchedulerImpl::Geometric(s) => s.lr(batch, superbatch),
            LrSchedulerImpl::Cos(s) => s.lr(batch, superbatch),
            LrSchedulerImpl::Fixed(s) => s.lr(batch, superbatch),
        }
    }
    fn colourful(&self) -> String {
        match self {
            LrSchedulerImpl::Step(s) => s.colourful(),
            LrSchedulerImpl::Geometric(s) => s.colourful(),
            LrSchedulerImpl::Cos(s) => s.colourful(),
            LrSchedulerImpl::Fixed(s) => s.colourful(),
        }
    }
}

#[derive(Clone, Debug)]
struct PlateauLrState {
    current_lr: f32,
    min_lr: f32,
    factor: f32,
    min_delta: f32,
    monitor: PlateauMonitor,
    best: Option<PlateauMetrics>,
    final_min_run: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlateauMetrics {
    loss: f32,
    accuracy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PlateauAction {
    First { metrics: PlateauMetrics },
    Improved { old_best: PlateauMetrics, new_best: PlateauMetrics },
    Keep { metrics: PlateauMetrics, best: PlateauMetrics },
    Reduced { old_lr: f32, new_lr: f32, metrics: PlateauMetrics, best: PlateauMetrics },
    ScheduledFinal { old_lr: f32, min_lr: f32, metrics: PlateauMetrics, best: PlateauMetrics },
    FinalImproved { old_best: PlateauMetrics, new_best: PlateauMetrics },
    FinalRejected { metrics: PlateauMetrics, best: PlateauMetrics },
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "snake_case")]
enum PlateauMonitor {
    Loss,
    Accuracy,
    LossOrAccuracy,
}

impl PlateauMonitor {
    fn cli_name(self) -> &'static str {
        match self {
            PlateauMonitor::Loss => "loss",
            PlateauMonitor::Accuracy => "accuracy",
            PlateauMonitor::LossOrAccuracy => "loss_or_accuracy",
        }
    }

    fn improved(self, current: PlateauMetrics, best: PlateauMetrics, min_delta: f32) -> bool {
        let loss_improved = current.loss + min_delta < best.loss;
        let accuracy_improved = current.accuracy > best.accuracy;
        match self {
            PlateauMonitor::Loss => loss_improved,
            PlateauMonitor::Accuracy => accuracy_improved,
            PlateauMonitor::LossOrAccuracy => loss_improved || accuracy_improved,
        }
    }

    fn label(self) -> &'static str {
        match self {
            PlateauMonitor::Loss => "validation loss",
            PlateauMonitor::Accuracy => "validation accuracy",
            PlateauMonitor::LossOrAccuracy => "validation loss/accuracy",
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
enum TestSampleMode {
    Random,
    Sequential,
}

impl TestSampleMode {
    fn cli_name(self) -> &'static str {
        match self {
            TestSampleMode::Random => "random",
            TestSampleMode::Sequential => "sequential",
        }
    }
}

fn plateau_action_retries_teacher(action: PlateauAction) -> bool {
    matches!(action, PlateauAction::Reduced { .. } | PlateauAction::ScheduledFinal { .. })
}

fn plateau_action_rejects_update(action: PlateauAction) -> bool {
    matches!(
        action,
        PlateauAction::Reduced { .. } | PlateauAction::ScheduledFinal { .. } | PlateauAction::FinalRejected { .. }
    )
}

fn plateau_action_epoch_final_metrics(action: PlateauAction) -> Option<PlateauMetrics> {
    match action {
        PlateauAction::FinalImproved { new_best, .. } => Some(new_best),
        PlateauAction::FinalRejected { best, .. } => Some(best),
        _ => None,
    }
}

fn plateau_metrics_text(metrics: PlateauMetrics) -> String {
    format!("loss={:.6}, accuracy={:.6}", metrics.loss, metrics.accuracy)
}

fn epoch_final_should_stop(
    previous_metrics: Option<PlateauMetrics>,
    current_metrics: PlateauMetrics,
    monitor: PlateauMonitor,
    min_delta: f32,
) -> bool {
    matches!(previous_metrics, Some(previous) if !monitor.improved(current_metrics, previous, min_delta))
}

impl PlateauLrState {
    fn new(start_lr: f32, min_lr: f32, factor: f32, min_delta: f32, monitor: PlateauMonitor) -> Self {
        Self { current_lr: start_lr, min_lr, factor, min_delta, monitor, best: None, final_min_run: false }
    }

    fn observe(&mut self, metrics: PlateauMetrics) -> PlateauAction {
        if self.final_min_run {
            self.final_min_run = false;
            match self.best {
                Some(best) if self.monitor.improved(metrics, best, self.min_delta) => {
                    self.best = Some(metrics);
                    return PlateauAction::FinalImproved { old_best: best, new_best: metrics };
                }
                Some(best) => {
                    return PlateauAction::FinalRejected { metrics, best };
                }
                None => {
                    self.best = Some(metrics);
                    return PlateauAction::First { metrics };
                }
            }
        }

        match self.best {
            None => {
                self.best = Some(metrics);
                PlateauAction::First { metrics }
            }
            Some(best) if self.monitor.improved(metrics, best, self.min_delta) => {
                self.best = Some(metrics);
                PlateauAction::Improved { old_best: best, new_best: metrics }
            }
            Some(best) => {
                let old_lr = self.current_lr;
                if old_lr <= self.min_lr {
                    return PlateauAction::FinalRejected { metrics, best };
                }

                let new_lr = old_lr * self.factor;
                if new_lr < self.min_lr {
                    self.current_lr = self.min_lr;
                    self.final_min_run = true;
                    PlateauAction::ScheduledFinal { old_lr, min_lr: self.min_lr, metrics, best }
                } else if new_lr < old_lr {
                    self.current_lr = new_lr;
                    PlateauAction::Reduced { old_lr, new_lr, metrics, best }
                } else {
                    PlateauAction::Keep { metrics, best }
                }
            }
        }
    }
}

fn effective_max_epochs(args: &Args) -> usize {
    args.max_epochs.unwrap_or(usize::MAX).max(1)
}

fn print_epoch_banner(epoch: usize, max_epochs: usize) {
    if max_epochs == usize::MAX {
        eprintln!("\n=== epoch {epoch} / unlimited ===");
    } else if max_epochs > 1 {
        eprintln!("\n=== epoch {epoch} / {max_epochs} ===");
    }
}

/// Which LR schedule the trainer should follow. `Step` is the
/// Epoch-local fixed-gamma StepLR; `Geometric` is the old smooth epoch
/// schedule; `Cos` is SGDR-style cosine annealing with one cycle per epoch.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
enum LrScheduleKind {
    Step,
    Geometric,
    Cos,
    Plateau,
}

impl LrScheduleKind {
    fn cli_name(self) -> &'static str {
        match self {
            LrScheduleKind::Step => "step",
            LrScheduleKind::Geometric => "geometric",
            LrScheduleKind::Cos => "cos",
            LrScheduleKind::Plateau => "plateau",
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
enum OptimizerKind {
    Adamw,
    Radam,
    Ranger,
}

impl OptimizerKind {
    fn cli_name(self) -> &'static str {
        match self {
            OptimizerKind::Adamw => "adamw",
            OptimizerKind::Radam => "radam",
            OptimizerKind::Ranger => "ranger",
        }
    }
}

fn effective_lr_step_positions(args: &Args, batches_per_superbatch: usize) -> u64 {
    args.lr_step_positions
        .unwrap_or_else(|| (effective_batch_size(args) as u64).saturating_mul(batches_per_superbatch as u64))
}

const DEFAULT_LR_STEP_GAMMA: f32 = 0.992;
const DEFAULT_POSITIONS_PER_SUPERBATCH: usize = 100_000_000;
const DEFAULT_BATCH_SIZE: usize = 65_536;
const DEFAULT_SAVE_RATE: usize = 20;

fn effective_batch_size(args: &Args) -> usize {
    args.batch_size.unwrap_or(DEFAULT_BATCH_SIZE)
}

fn effective_save_rate(args: &Args) -> usize {
    args.save_rate.unwrap_or(DEFAULT_SAVE_RATE)
}

fn effective_save_epoch_end(args: &Args) -> bool {
    args.save_epoch_end && !args.no_save_epoch_end
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_default_cpu_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get().saturating_mul(2).clamp(4, 24)).unwrap_or(24)
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_effective_teacher_threads(args: &Args) -> usize {
    if args.threads == 4 { cuda_cpp_default_cpu_threads() } else { args.threads }
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_effective_loader_threads(args: &Args) -> usize {
    if args.loader_threads == 0 { cuda_cpp_default_cpu_threads() } else { args.loader_threads }
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_effective_batch_queue_size(args: &Args) -> usize {
    if args.batch_queue_size == 32 { 4 } else { args.batch_queue_size }
}

fn effective_batches_per_superbatch(args: &Args) -> Result<usize, String> {
    let batch_size = effective_batch_size(args);
    if batch_size == 0 {
        return Err("--batch-size must be > 0.".to_string());
    }
    let batches = args.positions_per_superbatch / batch_size;
    if batches == 0 {
        return Err(format!(
            "--positions-per-superbatch ({}) must be >= --batch-size ({}).",
            args.positions_per_superbatch, batch_size
        ));
    }
    Ok(batches)
}

fn effective_positions_per_superbatch(args: &Args) -> Result<usize, String> {
    Ok(effective_batches_per_superbatch(args)?.saturating_mul(effective_batch_size(args)))
}

fn effective_lr_step_gamma(args: &Args, batches_per_superbatch: usize) -> Result<(f32, bool), String> {
    if let Some(gamma) = args.lr_step_gamma {
        return Ok((gamma, false));
    }
    if args.lr_schedule != LrScheduleKind::Step {
        return Ok((DEFAULT_LR_STEP_GAMMA, false));
    }
    let Some(superbatches) = args.superbatches else {
        return Ok((DEFAULT_LR_STEP_GAMMA, false));
    };
    if args.lr <= 0.0 {
        return Err("--lr must be > 0 for automatic --lr-step-gamma.".to_string());
    }
    if args.lr_min <= 0.0 {
        return Err("--lr-min must be > 0 for automatic --lr-step-gamma.".to_string());
    }
    if args.lr_min > args.lr {
        return Err("--lr-min must be <= --lr for automatic --lr-step-gamma.".to_string());
    }

    let step_positions = effective_lr_step_positions(args, batches_per_superbatch);
    if step_positions == 0 {
        return Err("--lr-step-positions must be > 0.".to_string());
    }
    let total_positions = (superbatches as u128)
        .saturating_mul(batches_per_superbatch as u128)
        .saturating_mul(effective_batch_size(args) as u128);
    let steps = total_positions / (step_positions as u128);
    if steps == 0 {
        return Err(
            "automatic --lr-step-gamma needs at least one LR step; reduce --lr-step-positions or increase --superbatches."
                .to_string(),
        );
    }
    let gamma = ((args.lr_min as f64) / (args.lr as f64)).powf(1.0 / steps as f64) as f32;
    Ok((gamma, true))
}

// ----- CLI ---------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
#[command(name = "bulletou")]
#[command(about = "BulletOu unified trainer")]
#[command(
    after_help = "Subcommands:\n  nerf    Post-process a supported nn.bin by adding reproducible ±1 noise to selected i8 weights\n\nRun `bulletou nerf --help` for nerf-specific options."
)]
struct Args {
    /// Evaluation function type to train. Required for training; not
    /// needed (and ignored) when `--count-teacher` is used.
    #[arg(long, value_enum, required_unless_present = "count_teacher")]
    eval_type: Option<EvalType>,

    /// Training backend. `bullet` is the existing generic backend.
    /// `cuda-cpp` is the Windows-native C++/CUDA fixed-layout backend.
    #[arg(long, value_enum, default_value = "cuda-cpp")]
    backend: BackendKind,

    /// CUDA device index for the Windows-native C++/CUDA backend.
    #[arg(long, default_value = "0")]
    cuda_cpp_device: i32,

    /// Run the C++/CUDA backend bring-up smoke and exit.
    #[arg(long)]
    cuda_cpp_smoke: bool,

    /// Optional initial root-format NNUE weights/state stream for the Windows-native
    /// C++/CUDA direct trainer. If Ranger optimizer records are present, they
    /// are restored together with the weights.
    #[arg(long)]
    cuda_cpp_weights_bin: Option<PathBuf>,

    /// Temporary Windows-native C++/CUDA direct-trainer batch count.
    /// This currently runs NNUE_HALFKP fixed-layout train steps without
    /// production checkpoint/resume orchestration.
    #[arg(long)]
    cuda_cpp_train_steps: Option<usize>,

    /// Profile the first N C++/CUDA direct-trainer steps with CUDA events.
    /// Prints upload/forward/loss/backward/update GPU time per profiled step.
    #[arg(long, default_value = "0")]
    cuda_cpp_profile_steps: usize,

    /// In C++/CUDA direct-step mode, skip the final numbered checkpoint and
    /// `cuda-cpp-direct` full-state output. Useful for throughput/validation
    /// probes where writing multi-GB optimizer state would dominate disk use.
    #[arg(long)]
    cuda_cpp_skip_final_output: bool,

    /// Print CPU teacher batch preparation time for the Windows-native
    /// C++/CUDA backend. This disables the prepared-batch producer queue
    /// for clearer per-batch timings, so use it only for diagnosis.
    #[arg(long)]
    cuda_cpp_profile_teacher_prepare: bool,

    /// Read and print C++/CUDA direct-trainer loss every N steps. This
    /// synchronises the compute stream, so use 0 for throughput probes
    /// where only the final loss is needed.
    #[arg(long, default_value = "10")]
    cuda_cpp_loss_readback_interval: usize,

    /// Teacher data: either a single file (`.hcpe` / `.hcpe3` / `.pack` /
    /// `.psv`), a directory containing such files (all matching files are
    /// concatenated), or a comma-separated list of either. Format is
    /// inferred from the extension; all included files must share the same
    /// extension.
    #[arg(long)]
    teacher: String,

    /// Count teacher positions and exit without training. For fixed-record
    /// formats (HCPE / PSV) this just reads `file_size / record_size` per
    /// file (instant). HCPE3 / pack are variable-length and would need to
    /// walk every game; not yet supported by this flag.
    ///
    /// Use the printed total to pick `--superbatches N` for `geometric` / `cos`
    /// runs such that one epoch ≈ (or ≤) the teacher size. With cosine
    /// annealing, period auto-aligns to `--superbatches`.
    #[arg(long)]
    count_teacher: bool,

    /// Checkpoint output directory. Defaults to a per-eval-type path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Suffix appended to the auto-derived output directory name. Useful
    /// for running multiple experiments with the same network /
    /// architecture but different hyperparameters: each run lands in
    /// its own directory like
    /// `checkpoints/<eval-type>-<arch>[-<tag>]`.
    /// Ignored when `--output` is set explicitly (the user-provided
    /// path wins).
    #[arg(long)]
    tag: Option<String>,

    /// Net identifier (prefix of the saved checkpoint subdirectory).
    /// Defaults to a per-eval-type name.
    #[arg(long)]
    net_id: Option<String>,

    /// Mini-batch size (positions per gradient step). If omitted, BulletOu
    /// uses 65536 to match tatara's high-throughput recipe.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Target positions consumed per superbatch. The actual value is rounded
    /// down to a multiple of `--batch-size` because the trainer advances in
    /// whole mini-batches. Default = 100M positions.
    #[arg(long, default_value_t = DEFAULT_POSITIONS_PER_SUPERBATCH)]
    positions_per_superbatch: usize,

    /// Cap on the number of superbatches per epoch. If omitted, there is no
    /// cap (= run until the dataloader reaches EOF). Specify this to stop
    /// each epoch early (e.g. to fit a quick smoke test). For `plateau`,
    /// this is only a safety cap; the epoch normally ends when `--lr-min`
    /// is reached.
    #[arg(long)]
    superbatches: Option<usize>,

    /// Force loading the latest checkpoint under the output directory,
    /// even if its stored resume config is missing or differs from the
    /// current command line. Use this only when you intentionally want to
    /// continue an old run with changed training controls.
    #[arg(long, conflicts_with = "no_resume")]
    resume: bool,

    /// Refuse to load any checkpoint from the output directory. If the
    /// directory already contains a resumable checkpoint, the program
    /// stops instead of mixing a fresh run into the same checkpoint series.
    #[arg(long, conflicts_with = "resume")]
    no_resume: bool,

    /// Number of epochs to train. With explicit `--superbatches N`, one
    /// epoch is an LR/validation cycle of N superbatches; the teacher stream
    /// continues across epoch boundaries and wraps only at EOF. Without
    /// `--superbatches`, non-plateau schedules treat teacher EOF as the
    /// epoch end. With `plateau`, teacher EOF wraps back to the beginning
    /// inside the same epoch; the epoch ends when LR reaches `--lr-min` and
    /// the final min-LR retry has completed. If omitted, all LR schedules
    /// keep running without a fixed epoch cap.
    /// With a readable `--test-teacher`, training also stops before reaching
    /// this cap when an epoch-final validation run improves neither loss nor
    /// accuracy versus the previous epoch.
    #[arg(long, alias = "max-epoch")]
    max_epochs: Option<usize>,

    /// Initial optimizer learning rate. Default follows the tatara
    /// SFNN-1536 reference recipe.
    #[arg(long, default_value = "0.000875")]
    lr: f32,

    /// Optimizer used for training. The default is `ranger`, matching
    /// the tatara reference recipe and bullet-shogi's shogi examples.
    /// `ranger` is BulletOu's existing
    /// RAdam+Lookahead implementation; it is useful for ablation against
    /// nnue-pytorch's Ranger21, but is not a full Ranger21 clone.
    #[arg(long, value_enum, default_value = "ranger")]
    optimizer: OptimizerKind,

    /// LR schedule kind. `step` applies fixed gamma drops within one epoch
    /// and warm-restarts to `--lr` at epoch boundaries. `geometric` and `cos` sweep
    /// `--lr` (lr_max) → `--lr-min` over **one epoch**, warm-restarting to
    /// `--lr` at each epoch boundary. `plateau` keeps LR fixed during one
    /// superbatch, then reduces it when the validation monitor does not
    /// improve:
    ///
    /// - `step` (default) = `lr = max(lr_min, lr * gamma^n)` where n is
    ///   the number of completed `--lr-step-positions` intervals.
    ///   If `--lr-step-positions` is omitted, this is one gamma drop per
    ///   superbatch.
    /// - `geometric` = exponential interpolation in log space:
    ///   `lr(t) = lr_max * (lr_min/lr_max)^t` where t∈[0,1] is
    ///   "fraction of one epoch completed". Constant multiplicative
    ///   decay per batch.
    /// - `cos` = cosine annealing (SGDR-style):
    ///   `lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(πt))`.
    ///   Slower descent at the start and end, fastest in the middle.
    /// - `plateau` = ReduceLROnPlateau: after each saved superbatch,
    ///   if the `--lr-plateau-monitor` metric did not improve, multiply LR by
    ///   `--lr-plateau-factor` and retry the same teacher interval.
    ///   When the next LR would fall below `--lr-min`, train that
    ///   interval one final time at exactly `--lr-min` and end the epoch.
    ///
    /// For `step` / `geometric` / `cos`, the epoch period is set by
    /// `--superbatches N` (`N * sb_size`). Without `--superbatches`,
    /// `step` uses open-ended gamma=0.992, while `geometric` / `cos` fall
    /// back to the teacher's total position count for HCPE / PSV and require
    /// an explicit period for HCPE3 / pack. `plateau` is validation-driven.
    #[arg(long, value_enum, default_value = "step")]
    lr_schedule: LrScheduleKind,

    /// Floor LR. For `step` / `plateau`, this is the lower bound.
    /// For `geometric` / `cos`, this is reached at the end of each epoch
    /// before warm restart.
    /// Must be strictly positive for `step`, `geometric`, and `plateau`;
    /// cosine can use 0 but 1e-5 or 1e-6 is more typical.
    #[arg(long, default_value = "0.00001")]
    lr_min: f32,

    /// Multiplicative LR factor used by `--lr-schedule step`.
    /// If omitted and `--superbatches` is set, BulletOu computes gamma so
    /// LR reaches `--lr-min` within one epoch. If the epoch length is open-ended,
    /// tatara / bullet-shogi's default gamma=0.992 is used.
    #[arg(long)]
    lr_step_gamma: Option<f32>,

    /// Position interval for one `step` decay. If omitted, one
    /// BulletOu superbatch is used (the effective `--positions-per-superbatch`,
    /// rounded down to a multiple of `--batch-size`).
    /// Omit this to decay once per BulletOu superbatch.
    /// Use `100000000` for position-fixed comparisons against
    /// nnue-pytorch's default 100M-position epoch.
    #[arg(long)]
    lr_step_positions: Option<u64>,

    /// ReduceLROnPlateau factor used by `--lr-schedule plateau`.
    /// Must satisfy `0 < factor < 1`.
    #[arg(long, default_value = "0.5")]
    lr_plateau_factor: f32,

    /// Minimum validation-loss improvement required by
    /// `--lr-schedule plateau`. With the default 0, any strictly lower
    /// validation loss counts as an improvement.
    #[arg(long, default_value = "0.0")]
    lr_plateau_min_delta: f32,

    /// Metric used by `--lr-schedule plateau` to accept a superbatch.
    /// `loss` keeps the historical behaviour. `accuracy` accepts updates
    /// whose held-out sign accuracy increases. `loss_or_accuracy` accepts
    /// either a lower loss or a higher accuracy, and is the practical default.
    #[arg(long, value_enum, default_value = "loss_or_accuracy")]
    lr_plateau_monitor: PlateauMonitor,

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
    #[arg(long, default_value = "290")]
    scale: u32,

    /// Use nnue-pytorch's WRM value loss and target conversion for NNUE /
    /// SFNN scalar value networks. This uses the nodchip shogi defaults:
    /// nnue2score=600, offset=270, input scaling=340, output scaling=380,
    /// and |target-prediction|^2.5. Default is off for A/B comparisons.
    #[arg(long)]
    nnue_pytorch_wrm_loss: bool,

    /// Optimizer weight decay for the selected optimizer. Default follows
    /// the tatara SFNN-1536 reference recipe.
    #[arg(long, default_value = "0.0")]
    optimizer_weight_decay: f32,

    /// Optimizer epsilon override for the selected optimizer. If omitted,
    /// the optimizer's own default is used.
    #[arg(long)]
    optimizer_epsilon: Option<f32>,

    /// Optimizer beta1 override for the selected optimizer. If omitted,
    /// the optimizer's own default is used. This matters for `ranger`:
    /// bullet-shogi's Ranger default is beta1=0.99, not AdamW's 0.9.
    #[arg(long)]
    optimizer_beta1: Option<f32>,

    /// Optimizer beta2 override for the selected optimizer. If omitted,
    /// the optimizer's own default is used.
    #[arg(long)]
    optimizer_beta2: Option<f32>,

    /// Use nnue-pytorch-style layer-specific AdamW clipping for NNUE / SFNN
    /// scalar value networks. Hidden weights use +/-127/64, while only the
    /// final output weight tensor uses +/-127*127/(600*16). Default is off
    /// for A/B comparisons.
    #[arg(long)]
    nnue_pytorch_layer_clip: bool,

    /// Do not clip bias tensors in AdamW for NNUE / SFNN scalar value
    /// networks. nnue-pytorch's WeightClippingCallback clips only selected
    /// weight tensors, not biases. Default is off for A/B comparisons.
    #[arg(long)]
    nnue_pytorch_no_bias_clip: bool,

    /// f32 -> integer quantisation scale for the YaneuraOu KPPT output.
    /// If omitted, per-component defaults are used (4000 for KK/KKP, 400
    /// for KPP). Ignored by NNUE eval types.
    #[arg(long)]
    yaneuraou_quant_scale: Option<f32>,

    /// Save every N superbatches (1 = save every superbatch, 5 = every 5th).
    #[arg(long)]
    save_rate: Option<usize>,

    /// Also save the final superbatch of each epoch even when it is not on a save-rate boundary.
    #[arg(long, default_value_t = true, action = ArgAction::SetTrue)]
    save_epoch_end: bool,

    /// Disable the implicit checkpoint at the final superbatch of each epoch.
    #[arg(long = "no-save-epoch-end")]
    no_save_epoch_end: bool,

    /// Dataloader worker threads (CPU side).
    #[arg(long, default_value = "4")]
    threads: usize,

    /// GPU-side batch queue depth.
    #[arg(long, default_value = "32")]
    batch_queue_size: usize,

    /// Loader read buffer size in megabytes. PSV 1 record = 40 byte
    /// なので `--buffer-mb 4096` で 107M 局面 (= 約 1 superbatch 分) が
    /// read buffer に貯められる。BulletOu は学習時に追加シャッフルしないので、
    /// 教師ファイルは事前にシャッフル済みのものを渡すこと。
    ///
    /// RAM 消費: buffer のみで `buffer_mb` MB。学習中の他の構造
    /// (model / optimiser / batch queue) と合わせて peak はこの 2 倍弱を見込む。
    #[arg(long, default_value = "4096")]
    buffer_mb: usize,

    /// HCPE デコード並列度。`0` (デフォルト) で auto =
    /// `available_parallelism()` (= 論理コア数)。学習中に他プロセスへ
    /// CPU を譲りたいときなどに `--loader-threads 8` 等で上限を絞れる。
    /// 起動時に `read buffer ready: ... (N decode threads)` で
    /// 実際に使われた値が表示される。
    /// 現状 HCPE ローダーのみ反映 (HCPE3 / .pack / .psv は未対応)。
    #[arg(long, default_value = "0")]
    loader_threads: usize,

    /// Drop positions whose |score| >= this. Useful to exclude ±32000 mate
    /// stamps. Set to 0 to disable.
    #[arg(long, default_value = "32000")]
    score_drop_abs: u16,

    /// Network architecture for NNUE / SFNN eval types. Use the YaneuraOu
    /// Makefile architecture name after removing `YANEURAOU_ENGINE_`, e.g.
    /// `NNUE_halfkp_256x2_32_32` or `SFNN_halfka2_1024_7_64_k3k3`.
    /// If omitted, a per-eval-type default is used. Ignored for KPPT /
    /// KPP_KKPT eval types.
    #[arg(long)]
    arch: Option<NnueArch>,

    /// Scale multiplier for nnue-pytorch-compatible initialisation used by
    /// SFNN / LayerStack networks. The actual bound is
    /// `scale * sqrt(1 / fan_in)`. Values below 1.0 make the initial
    /// activations smaller and help diagnose early CReLU saturation.
    #[arg(long, default_value = "1.0")]
    nnue_pytorch_init_scale: f32,

    /// Add nnue-pytorch-style factorized shared L1 weights to SFNN /
    /// LayerStack networks. The shared term is zero-initialised and added to
    /// every bucket's L1 output during training, then folded into each bucket
    /// when saving `nn.bin`. Default is off to preserve historical BulletOu
    /// behaviour for A/B comparisons.
    #[arg(long)]
    sfnn_factorized_l1: bool,

    /// Dump SFNN / LayerStack activation saturation statistics on the
    /// held-out `--test-teacher` positions at startup and after each save.
    /// Currently supported for SFNN eval types only.
    #[arg(long)]
    dump_activation_stats: bool,

    /// Number of `--test-teacher` positions used for
    /// `--dump-activation-stats`. This is intentionally independent of
    /// `--test-positions` because the CPU-side diagnostic is heavier than
    /// normal GPU validation.
    #[arg(long, default_value = "1024")]
    activation_stats_positions: usize,

    /// Held-out test set (.hcpe / .psv) for sign-agreement validation
    /// during training. When set, the trainer runs validation after
    /// each save event (= every `--save-rate` superbatches): samples
    /// `--test-positions` positions from this file, runs them
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

    /// How to choose validation positions from `--test-teacher`.
    /// `sequential` reads the first `--test-positions` fixed records and
    /// is useful for byte-for-byte parity against external trainers.
    #[arg(long, value_enum, default_value = "random")]
    test_sample: TestSampleMode,

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
    /// `<eval-type>` and `<arch>` are the canonical CLI values the user
    /// would type (e.g. `NNUE_HALFKP`, `NNUE_halfkp_256x2_32_32`) so the
    /// dir name stays in sync with the flags.
    fn output_dir(&self) -> PathBuf {
        if let Some(p) = &self.output {
            // Explicit --output wins; --tag is ignored to keep the
            // user-provided path verbatim.
            return p.clone();
        }
        let mut path = PathBuf::from("checkpoints");
        let mut name = self.eval_type().cli_name().to_string();
        if self.eval_type().uses_arch() {
            name.push('-');
            name.push_str(&self.arch().cli_name());
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
        self.net_id.clone().unwrap_or_else(|| self.eval_type().default_net_id().to_string())
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
        self.yaneuraou_quant_scale.expect("yaneuraou_quant_scale must be set before invoking the KPPT trainer")
    }

    fn kpp_format(&self) -> KppFormat {
        self.eval_type().kpp_format()
    }

    fn arch(&self) -> NnueArch {
        self.arch.unwrap_or_else(|| NnueArch::default_for_eval_type(self.eval_type()))
    }

    fn effective_layerstack(&self) -> Option<LayerStackMode> {
        if !self.eval_type().uses_layerstack() {
            return None;
        }
        self.arch().layerstack.or(Some(LayerStackMode::Kingrank3by3))
    }

    fn validate_arch_flags(&self) -> Result<(), String> {
        let eval_type = self.eval_type();

        if !eval_type.uses_arch() {
            if self.arch.is_some() {
                return Err(format!(
                    "--arch is only valid with NNUE / SFNN eval types; --eval-type {} has a fixed KPPT layout",
                    eval_type.cli_name()
                ));
            }
            return Ok(());
        }

        let arch = self.arch();
        let expected = arch.expected_eval_type();
        if expected != eval_type {
            return Err(format!(
                "--arch {} implies --eval-type {}, but --eval-type {} was selected",
                arch.cli_name(),
                expected.cli_name(),
                eval_type.cli_name()
            ));
        }

        Ok(())
    }

    fn validate_backend_flags(&self) -> Result<(), String> {
        match self.backend {
            BackendKind::Bullet => Ok(()),
            BackendKind::CudaCpp => self.validate_cuda_cpp_backend_options(),
        }
    }

    fn validate_cuda_cpp_backend_options(&self) -> Result<(), String> {
        if !cfg!(feature = "cuda-cpp-backend") {
            return Err("--backend cuda-cpp requires building with --features cuda-cpp-backend".to_string());
        }
        if self.cuda_cpp_smoke {
            if self.cuda_cpp_train_steps.is_some() {
                return Err("--cuda-cpp-smoke and --cuda-cpp-train-steps cannot be used together".to_string());
            }
            if self.cuda_cpp_weights_bin.is_some() {
                return Err("--cuda-cpp-smoke and --cuda-cpp-weights-bin cannot be used together".to_string());
            }
            if self.cuda_cpp_profile_steps != 0 {
                return Err("--cuda-cpp-smoke and --cuda-cpp-profile-steps cannot be used together".to_string());
            }
            if self.cuda_cpp_skip_final_output {
                return Err("--cuda-cpp-smoke and --cuda-cpp-skip-final-output cannot be used together".to_string());
            }
            if self.cuda_cpp_profile_teacher_prepare {
                return Err(
                    "--cuda-cpp-smoke and --cuda-cpp-profile-teacher-prepare cannot be used together".to_string()
                );
            }
            return Ok(());
        }

        let eval_type = self.eval_type();
        if !matches!(
            eval_type,
            EvalType::Kppt | EvalType::KppKkpt | EvalType::NnueHalfkp | EvalType::NnueKp | EvalType::SfnnHalfka2
        ) {
            return Err(format!(
                "--backend cuda-cpp currently supports KPPT, KPP_KKPT, NNUE_HALFKP, NNUE_KP, or SFNN_HALFKA2 train steps only; {} is tracked by later tickets",
                eval_type.cli_name()
            ));
        }
        if eval_type.uses_arch() {
            let supported_arch = cuda_cpp_supported_arch(eval_type);
            if self.arch() != supported_arch {
                return Err(format!(
                    "--backend cuda-cpp currently supports only --arch {} for {}; got {}",
                    supported_arch.cli_name(),
                    eval_type.cli_name(),
                    self.arch().cli_name()
                ));
            }
        }

        if let Some(0) = self.cuda_cpp_train_steps {
            return Err("--cuda-cpp-train-steps must be > 0".to_string());
        }
        let production_schedule = cuda_cpp_uses_production_schedule(self);
        match (self.cuda_cpp_train_steps, production_schedule) {
            (Some(_), true) => {
                return Err(
                    "--backend cuda-cpp accepts either --cuda-cpp-train-steps or --superbatches, not both".to_string()
                );
            }
            (None, false) => {
                return Err("--backend cuda-cpp requires either --cuda-cpp-train-steps N for direct-step smoke mode \
                     or --superbatches N --max-epochs N for Windows-native production schedule mode"
                    .to_string());
            }
            _ => {}
        }
        if self.cuda_cpp_skip_final_output && self.cuda_cpp_train_steps.is_none() {
            return Err(
                "--cuda-cpp-skip-final-output is only valid with --cuda-cpp-train-steps direct-step mode".to_string()
            );
        }
        if matches!(eval_type, EvalType::Kppt | EvalType::KppKkpt) && self.cuda_cpp_skip_final_output {
            return Err(
                "--cuda-cpp-skip-final-output is currently supported for NNUE/SFNN direct trainers only".to_string()
            );
        }
        if matches!(eval_type, EvalType::Kppt | EvalType::KppKkpt) && self.cuda_cpp_profile_steps != 0 {
            return Err(
                "--cuda-cpp-profile-steps is currently supported for NNUE/SFNN direct trainers only".to_string()
            );
        }
        if production_schedule && self.max_epochs.is_none() {
            return Err(
                "--backend cuda-cpp production schedule mode currently requires --max-epochs to avoid an unbounded run"
                    .to_string(),
            );
        }

        if effective_batch_size(self) == 0 {
            return Err("--batch-size must be > 0 for --backend cuda-cpp".to_string());
        }
        if self.optimizer != OptimizerKind::Ranger {
            return Err("--backend cuda-cpp direct trainer currently supports only --optimizer ranger".to_string());
        }
        if self.cuda_cpp_weights_bin.is_some() && (self.resume || self.no_resume) {
            return Err(
                "--backend cuda-cpp: --cuda-cpp-weights-bin is an explicit initial state; do not combine it with --resume/--no-resume"
                    .to_string(),
            );
        }
        if self.nnue_pytorch_layer_clip || self.nnue_pytorch_no_bias_clip {
            return Err(
                "--backend cuda-cpp direct trainer does not yet implement per-layer/bias clipping flags".to_string()
            );
        }
        if self.test_teacher.is_some() && matches!(eval_type, EvalType::Kppt | EvalType::KppKkpt) {
            return Err(
                "--backend cuda-cpp direct final validation is currently wired only for NNUE_HALFKP, NNUE_KP, and SFNN_HALFKA2; KPPT/KPP_KKPT currently skip --test-teacher validation"
                    .to_string(),
            );
        }
        if !production_schedule
            && (self.max_epochs.is_some()
                || self.lr_schedule != LrScheduleKind::Step
                || self.lr_step_gamma.is_some()
                || self.lr_step_positions.is_some()
                || self.save_rate.is_some_and(|save_rate| save_rate != 1)
                || self.no_save_epoch_end)
        {
            return Err(
                "--backend cuda-cpp direct-step mode does not honor production schedule flags; use --superbatches with --max-epochs instead"
                    .to_string(),
            );
        }
        if production_schedule && self.lr_schedule == LrScheduleKind::Plateau {
            if self.test_teacher.is_none() {
                return Err(
                    "--backend cuda-cpp --lr-schedule plateau requires --test-teacher so validation metrics can be monitored"
                        .to_string(),
                );
            }
            if effective_save_rate(self) != 1 {
                return Err("--backend cuda-cpp --lr-schedule plateau requires --save-rate 1".to_string());
            }
        }

        Ok(())
    }

    /// Unwrap `eval_type`. clap's `required_unless_present = "count_teacher"`
    /// guarantees it's `Some` whenever training (= non-count_teacher) is
    /// taking place; if it isn't, there is a clap-side bug, not user error.
    fn eval_type(&self) -> EvalType {
        self.eval_type.expect("--eval-type required (clap should have validated)")
    }
}

type ValueLossFn = for<'a> fn(ModelNode<'a>, ModelNode<'a>) -> ModelNode<'a>;

fn sigmoid_mse_value_loss<'a>(output: ModelNode<'a>, target: ModelNode<'a>) -> ModelNode<'a> {
    output.sigmoid().squared_error(target)
}

fn nnue_pytorch_wrm_value_loss<'a>(output: ModelNode<'a>, target: ModelNode<'a>) -> ModelNode<'a> {
    const NNUE2SCORE: f32 = 600.0;
    const IN_OFFSET: f32 = 270.0;
    const IN_SCALING: f32 = 340.0;
    const POW_EXP: f32 = 2.5;

    let scorenet = output * NNUE2SCORE;
    let q = ((scorenet - IN_OFFSET) / IN_SCALING).sigmoid();
    let qm = ((-scorenet - IN_OFFSET) / IN_SCALING).sigmoid();
    let prediction = (1.0 + q - qm) * 0.5;
    prediction.power_error(target, POW_EXP)
}

fn value_loss_fn(args: &Args) -> ValueLossFn {
    if args.nnue_pytorch_wrm_loss { nnue_pytorch_wrm_value_loss } else { sigmoid_mse_value_loss }
}

fn validation_loss_kind(args: &Args) -> ValidationLossKind {
    if args.nnue_pytorch_wrm_loss {
        ValidationLossKind::NnuePytorchWrm
    } else {
        ValidationLossKind::SigmoidMse
    }
}

const BULLETOU_DEFAULT_ADAMW_CLIP: f32 = 1.98;
const NNUE_PYTORCH_HIDDEN_CLIP: f32 = 127.0 / 64.0;
const NNUE_PYTORCH_OUTPUT_CLIP: f32 = 127.0 * 127.0 / (600.0 * 16.0);
const ADAMW_UNCLIPPED_BIAS_LIMIT: f32 = 1.0e30;
#[cfg(feature = "cuda-cpp-backend")]
const STATE_BACKEND_CUDA_CPP: &str = "cuda-cpp";

fn adamw_params(args: &Args, clip: f32) -> optimiser::AdamWParams {
    let mut params = optimiser::AdamWParams {
        decay: args.optimizer_weight_decay,
        min_weight: -clip,
        max_weight: clip,
        ..Default::default()
    };
    if let Some(epsilon) = args.optimizer_epsilon {
        params.epsilon = epsilon;
    }
    if let Some(beta1) = args.optimizer_beta1 {
        params.beta1 = beta1;
    }
    if let Some(beta2) = args.optimizer_beta2 {
        params.beta2 = beta2;
    }
    params
}

fn radam_params(args: &Args, clip: f32) -> bullet_trainer::optimiser::radam::RAdamParams {
    let mut params = bullet_trainer::optimiser::radam::RAdamParams {
        decay: args.optimizer_weight_decay,
        clip: Some((-clip, clip)),
        ..Default::default()
    };
    if let Some(epsilon) = args.optimizer_epsilon {
        params.epsilon = epsilon;
    }
    if let Some(beta1) = args.optimizer_beta1 {
        params.beta1 = beta1;
    }
    if let Some(beta2) = args.optimizer_beta2 {
        params.beta2 = beta2;
    }
    params
}

fn ranger_params(args: &Args, clip: f32) -> optimiser::RangerParams {
    let mut params = optimiser::RangerParams {
        decay: args.optimizer_weight_decay,
        min_weight: -clip,
        max_weight: clip,
        ..Default::default()
    };
    if let Some(epsilon) = args.optimizer_epsilon {
        params.epsilon = epsilon;
    }
    if let Some(beta1) = args.optimizer_beta1 {
        params.beta1 = beta1;
    }
    if let Some(beta2) = args.optimizer_beta2 {
        params.beta2 = beta2;
    }
    params
}

trait BulletouOptimizer: optimiser::OptimiserType + Default {
    fn configure<Inp, Out>(
        args: &Args,
        trainer: &mut ValueTrainer<Self::Optimiser, Inp, Out>,
        output_weight_ids: &[&str],
        bias_ids: &[&str],
    ) where
        Inp: SparseInputType,
        Out: OutputBuckets<Inp::RequiredDataType>;
}

impl BulletouOptimizer for optimiser::AdamW {
    fn configure<Inp, Out>(
        args: &Args,
        trainer: &mut ValueTrainer<Self::Optimiser, Inp, Out>,
        output_weight_ids: &[&str],
        bias_ids: &[&str],
    ) where
        Inp: SparseInputType,
        Out: OutputBuckets<Inp::RequiredDataType>,
    {
        let global_clip =
            if args.nnue_pytorch_layer_clip { NNUE_PYTORCH_HIDDEN_CLIP } else { BULLETOU_DEFAULT_ADAMW_CLIP };
        trainer.optimiser.set_params(adamw_params(args, global_clip));

        if args.nnue_pytorch_layer_clip {
            let output_params = adamw_params(args, NNUE_PYTORCH_OUTPUT_CLIP);
            for id in output_weight_ids {
                trainer.optimiser.set_params_for_weight(id, output_params);
            }
        }

        if args.nnue_pytorch_no_bias_clip {
            let bias_params = adamw_params(args, ADAMW_UNCLIPPED_BIAS_LIMIT);
            for id in bias_ids {
                trainer.optimiser.set_params_for_weight(id, bias_params);
            }
        }
    }
}

impl BulletouOptimizer for optimiser::RAdam {
    fn configure<Inp, Out>(
        args: &Args,
        trainer: &mut ValueTrainer<Self::Optimiser, Inp, Out>,
        output_weight_ids: &[&str],
        bias_ids: &[&str],
    ) where
        Inp: SparseInputType,
        Out: OutputBuckets<Inp::RequiredDataType>,
    {
        let global_clip =
            if args.nnue_pytorch_layer_clip { NNUE_PYTORCH_HIDDEN_CLIP } else { BULLETOU_DEFAULT_ADAMW_CLIP };
        trainer.optimiser.set_params(radam_params(args, global_clip));

        if args.nnue_pytorch_layer_clip {
            let output_params = radam_params(args, NNUE_PYTORCH_OUTPUT_CLIP);
            for id in output_weight_ids {
                trainer.optimiser.set_params_for_weight(id, output_params);
            }
        }

        if args.nnue_pytorch_no_bias_clip {
            let bias_params = radam_params(args, ADAMW_UNCLIPPED_BIAS_LIMIT);
            for id in bias_ids {
                trainer.optimiser.set_params_for_weight(id, bias_params);
            }
        }
    }
}

impl BulletouOptimizer for optimiser::Ranger {
    fn configure<Inp, Out>(
        args: &Args,
        trainer: &mut ValueTrainer<Self::Optimiser, Inp, Out>,
        output_weight_ids: &[&str],
        bias_ids: &[&str],
    ) where
        Inp: SparseInputType,
        Out: OutputBuckets<Inp::RequiredDataType>,
    {
        let global_clip =
            if args.nnue_pytorch_layer_clip { NNUE_PYTORCH_HIDDEN_CLIP } else { BULLETOU_DEFAULT_ADAMW_CLIP };
        trainer.optimiser.set_params(ranger_params(args, global_clip));

        if args.nnue_pytorch_layer_clip {
            let output_params = ranger_params(args, NNUE_PYTORCH_OUTPUT_CLIP);
            for id in output_weight_ids {
                trainer.optimiser.set_params_for_weight(id, output_params);
            }
        }

        if args.nnue_pytorch_no_bias_clip {
            let bias_params = ranger_params(args, ADAMW_UNCLIPPED_BIAS_LIMIT);
            for id in bias_ids {
                trainer.optimiser.set_params_for_weight(id, bias_params);
            }
        }
    }
}

fn configure_optimizer<O, Inp, Out>(
    args: &Args,
    trainer: &mut ValueTrainer<O::Optimiser, Inp, Out>,
    output_weight_ids: &[&str],
    bias_ids: &[&str],
) where
    O: BulletouOptimizer,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    O::configure(args, trainer, output_weight_ids, bias_ids);
}

// ----- epoch period ------------------------------------------------------

/// Compute the warm-restart cycle period (= one epoch's positions), shared by
/// `step`, `geometric`, and `cos` schedules.
///
/// Semantics:
/// - If `--superbatches N` is set, period = `N * effective_sb_size`.
///   `effective_sb_size` is `--positions-per-superbatch` rounded down to
///   a multiple of `--batch-size`. This is the most common case: 1 cycle
///   per epoch.
/// - Otherwise (= unlimited sb cap), fall back to the teacher's total
///   position count. For HCPE / PSV (fixed-length records) we just
///   read `file_size / record_size` per file. HCPE3 / pack (variable
///   length) would need to walk every game so this combination is
///   rejected — the user is expected to set `--superbatches` for
///   variable-length teachers.
fn auto_epoch_period(
    args: &Args,
    format: DataFormat,
    data_files: &[String],
    batches_per_superbatch: usize,
) -> Result<u64, String> {
    let sb_size = (batches_per_superbatch as u64) * (effective_batch_size(args) as u64);
    if let Some(superbatches) = args.superbatches {
        return Ok(sb_size * (superbatches as u64));
    }
    let record_size: u64 = match format {
        DataFormat::Hcpe => 38,
        DataFormat::Psv => 40,
        DataFormat::Hcpe3 | DataFormat::Pack => {
            return Err(format!(
                "position-based LR schedule with variable-length teacher format \
                 ({format:?}) requires --superbatches to be set so the \
                 cosine cycle period is well-defined (no way to compute \
                 epoch length without walking the file). Use \
                 --count-teacher on an equivalent HCPE / PSV teacher \
                 to estimate the right --superbatches value."
            ));
        }
    };
    let mut total: u64 = 0;
    for p in data_files {
        let size = std::fs::metadata(p).map_err(|e| format!("stat {p}: {e}"))?.len();
        if size % record_size != 0 {
            return Err(format!(
                "{p}: size {size} not a multiple of {record_size} byte — \
                 possibly corrupted / truncated"
            ));
        }
        total += size / record_size;
    }
    Ok(total)
}

// ----- count-teacher -----------------------------------------------------

/// `--count-teacher` 実装: `--teacher` の指す全ファイルの局面数を集計して
/// stdout に出す。HCPE (38 byte 固定長) / PSV (40 byte 固定長) は file size
/// から即計算。HCPE3 / pack は可変長なので拒否 (= 別途 walker が必要)。
///
/// 1 sb ≒ 100M 局面に対して `--superbatches N` を選ぶための補助。
fn run_count_teacher(teacher: &str) -> Result<(), String> {
    let paths = expand_teacher(teacher)?;
    let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    let format = infer_data_format(&path_refs)?;
    let record_size: u64 = match format {
        DataFormat::Hcpe => 38,
        DataFormat::Psv => 40,
        DataFormat::Hcpe3 | DataFormat::Pack => {
            return Err(format!(
                "format {format:?} is variable-length; --count-teacher only supports \
                 fixed-length records (HCPE / PSV) currently. For HCPE3/pack you'd need \
                 to walk every game header."
            ));
        }
    };

    eprintln!("Counting {format:?} teacher files ({} byte/record)...", record_size);

    let mut total_positions: u64 = 0;
    let mut total_bytes: u64 = 0;
    for path in &paths {
        let meta = std::fs::metadata(path).map_err(|e| format!("failed to stat {path}: {e}"))?;
        let size = meta.len();
        if size % record_size != 0 {
            return Err(format!(
                "{path}: size {size} is not a multiple of {record_size} byte — \
                 possibly corrupted / truncated"
            ));
        }
        let positions = size / record_size;
        total_positions += positions;
        total_bytes += size;
        println!("  {:>14} positions  ({:>8.2} MB)  {path}", positions, size as f64 / (1024.0 * 1024.0),);
    }
    println!("---");
    println!(
        "Total: {total_positions} positions  ({:.2} GB)  across {} file(s)",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        paths.len(),
    );

    // Estimate sb count for the default settings (batch_size=65536, sb<=100M).
    let default_batch_size: u64 = DEFAULT_BATCH_SIZE as u64;
    let default_sb_size: u64 = (DEFAULT_POSITIONS_PER_SUPERBATCH as u64 / default_batch_size) * default_batch_size;
    // = floor(100M / batch_size) * batch_size = 99,942,400 for batch_size=65536.
    let full_sbs = total_positions / default_sb_size;
    let remainder = total_positions % default_sb_size;
    let partial_sb_fraction = remainder as f64 / default_sb_size as f64;
    println!(
        "Per-default-sb (= {:.0}M positions): {} full sb + {:.2} partial sb",
        default_sb_size as f64 / 1.0e6,
        full_sbs,
        partial_sb_fraction,
    );
    println!(
        "Suggested `--superbatches`: {} (= use {} full sb per epoch; ~{:.0}M positions leftover \
         carried to next epoch if loader wraps)",
        full_sbs.max(1),
        full_sbs.max(1),
        remainder as f64 / 1.0e6,
    );

    Ok(())
}

// ----- nerf ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NerfLayerId {
    Fc0,
    Fc1,
    Fc2,
}

#[derive(Debug, Clone, Copy)]
struct NerfCandidate {
    offset: usize,
    layer: NerfLayerId,
}

#[derive(Default, Debug, Clone)]
struct NerfReport {
    candidate_weights: usize,
    fc0_candidates: usize,
    fc1_candidates: usize,
    fc2_candidates: usize,
    selected: usize,
    changed: usize,
    saturated_noops: usize,
}

#[derive(Clone, Debug)]
struct NerfRng(u64);

impl NerfRng {
    fn from_seed(seed: u64) -> Self {
        Self(if seed == 0 {
            // Keep state non-zero even if the user explicitly passes 0.
            0xA076_1D64_78BD_642F
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn gen_index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() as usize) % upper
    }

    fn gen_delta(&mut self) -> i16 {
        if self.next_u64() & 1 == 0 { -1 } else { 1 }
    }
}

fn read_u32_le(bytes: &[u8], pos: usize, label: &str) -> Result<u32, String> {
    let end = pos.checked_add(4).ok_or_else(|| format!("{label}: offset overflow"))?;
    let slice = bytes.get(pos..end).ok_or_else(|| format!("{label}: truncated at byte {pos}"))?;
    Ok(u32::from_le_bytes(slice.try_into().expect("slice len checked")))
}

fn skip_leb128_block(bytes: &[u8], pos: usize, label: &str) -> Result<usize, String> {
    let magic_end = pos.checked_add(LEB128_MAGIC.len()).ok_or_else(|| format!("{label}: offset overflow"))?;
    if bytes.get(pos..magic_end) != Some(LEB128_MAGIC) {
        return Err(format!(
            "{label}: missing LEB128 magic at byte {pos}; this command currently expects an SFNN nn.bin"
        ));
    }
    let size_pos = magic_end;
    let payload_size = read_u32_le(bytes, size_pos, label)? as usize;
    let payload_start = size_pos + 4;
    let payload_end =
        payload_start.checked_add(payload_size).ok_or_else(|| format!("{label}: payload size overflow"))?;
    if payload_end > bytes.len() {
        return Err(format!(
            "{label}: payload claims {payload_size} byte(s) at byte {payload_start}, beyond file size {}",
            bytes.len()
        ));
    }
    Ok(payload_end)
}

fn sfnn_network_base_offset(bytes: &[u8]) -> Result<usize, String> {
    if bytes.len() < 12 {
        return Err("input is too small to contain an NNUE header".to_string());
    }
    let version = read_u32_le(bytes, 0, "NNUE header")?;
    if version != SFNN_NNUE_VERSION {
        return Err(format!("NNUE version mismatch: expected 0x{SFNN_NNUE_VERSION:08X}, got 0x{version:08X}"));
    }
    let desc_len = read_u32_le(bytes, 8, "NNUE header desc_len")? as usize;
    let desc_start = 12usize;
    let desc_end = desc_start.checked_add(desc_len).ok_or_else(|| "NNUE header desc_len overflow".to_string())?;
    if desc_end > bytes.len() {
        return Err(format!("NNUE header description claims {desc_len} byte(s), beyond file size {}", bytes.len()));
    }

    let ft_hash_pos = desc_end;
    let mut pos = ft_hash_pos.checked_add(4).ok_or_else(|| "FeatureTransformer hash offset overflow".to_string())?;
    if pos > bytes.len() {
        return Err("truncated before FeatureTransformer hash".to_string());
    }
    pos = skip_leb128_block(bytes, pos, "FeatureTransformer biases")?;
    pos = skip_leb128_block(bytes, pos, "FeatureTransformer weights")?;
    Ok(pos)
}

fn collect_sfnn_nerf_candidates(
    bytes: &[u8],
    arch: NnueArch,
    layerstack: LayerStackMode,
    layers: NerfLayerSet,
) -> Result<(Vec<NerfCandidate>, NerfReport), String> {
    let network_base = sfnn_network_base_offset(bytes)?;
    let (ft_size, hidden1, hidden2) = arch.dims();
    let l1_out = hidden1 + 1;
    let fc0_pad_in = nnue_pad32(ft_size);
    let fc1_real_in = hidden1 * 2;
    let fc1_pad_in = nnue_pad32(fc1_real_in);
    let fc2_pad_in = nnue_pad32(hidden2);
    let stack_count = layerstack.num_stacks();

    let fc0_bias_bytes = l1_out * 4;
    let fc0_weight_bytes = l1_out * fc0_pad_in;
    let fc1_bias_bytes = hidden2 * 4;
    let fc1_weight_bytes = hidden2 * fc1_pad_in;
    let fc2_bias_bytes = 4;
    let fc2_weight_bytes = fc2_pad_in;
    let stack_bytes =
        4 + fc0_bias_bytes + fc0_weight_bytes + fc1_bias_bytes + fc1_weight_bytes + fc2_bias_bytes + fc2_weight_bytes;
    let expected_len = network_base
        .checked_add(stack_bytes * stack_count)
        .ok_or_else(|| "SFNN network byte count overflow".to_string())?;
    if expected_len != bytes.len() {
        return Err(format!(
            "SFNN payload size mismatch for --arch {arch} / LayerStack {}: expected file size {expected_len}, got {}. \
             Check that --arch matches the engine architecture.",
            layerstack.cli_name(),
            bytes.len()
        ));
    }

    let mut out = Vec::new();
    let mut report = NerfReport::default();

    for stack in 0..stack_count {
        let mut pos = network_base + stack * stack_bytes;
        pos += 4; // Network hash.

        pos += fc0_bias_bytes;
        let fc0_weights = pos;
        if layers.fc0 {
            for o in 0..l1_out {
                for i in 0..ft_size {
                    out.push(NerfCandidate { offset: fc0_weights + o * fc0_pad_in + i, layer: NerfLayerId::Fc0 });
                    report.fc0_candidates += 1;
                }
            }
        }
        pos += fc0_weight_bytes;

        pos += fc1_bias_bytes;
        let fc1_weights = pos;
        if layers.fc1 {
            for o in 0..hidden2 {
                for i in 0..fc1_real_in {
                    out.push(NerfCandidate { offset: fc1_weights + o * fc1_pad_in + i, layer: NerfLayerId::Fc1 });
                    report.fc1_candidates += 1;
                }
            }
        }
        pos += fc1_weight_bytes;

        pos += fc2_bias_bytes;
        let fc2_weights = pos;
        if layers.fc2 {
            for i in 0..hidden2 {
                out.push(NerfCandidate { offset: fc2_weights + i, layer: NerfLayerId::Fc2 });
                report.fc2_candidates += 1;
            }
        }
        pos += fc2_weight_bytes;

        debug_assert_eq!(pos, network_base + (stack + 1) * stack_bytes);
    }

    report.candidate_weights = out.len();
    Ok((out, report))
}

fn nerf_sfnn_bytes(mut bytes: Vec<u8>, args: &NerfArgs) -> Result<(Vec<u8>, NerfReport), String> {
    let layerstack = args.effective_layerstack();
    let (candidates, mut report) = collect_sfnn_nerf_candidates(&bytes, args.arch, layerstack, args.layers)?;
    if args.count > 0 && candidates.is_empty() {
        return Err(format!(
            "--layers {} produced no candidate weights; choose at least one weight layer",
            args.layers
        ));
    }

    let mut rng = NerfRng::from_seed(args.seed);
    for _ in 0..args.count {
        let candidate = candidates[rng.gen_index(candidates.len())];
        match candidate.layer {
            NerfLayerId::Fc0 | NerfLayerId::Fc1 | NerfLayerId::Fc2 => {}
        }
        let old = bytes[candidate.offset] as i8;
        let delta = rng.gen_delta();
        let new = ((old as i16) + delta).clamp(i8::MIN as i16, i8::MAX as i16) as i8;
        if new == old {
            report.saturated_noops += 1;
        } else {
            bytes[candidate.offset] = new as u8;
            report.changed += 1;
        }
    }
    report.selected = args.count;
    Ok((bytes, report))
}

fn run_nerf(args: &NerfArgs) -> Result<NerfReport, String> {
    args.validate_arch_flags()?;
    if args.input == args.output {
        return Err("--input and --output must be different paths".to_string());
    }
    if args.output.exists() {
        let input_canon = std::fs::canonicalize(&args.input)
            .map_err(|e| format!("failed to canonicalize {}: {e}", args.input.display()))?;
        let output_canon = std::fs::canonicalize(&args.output)
            .map_err(|e| format!("failed to canonicalize {}: {e}", args.output.display()))?;
        if input_canon == output_canon {
            return Err("--input and --output resolve to the same file".to_string());
        }
    }
    if args.count == 0 {
        eprintln!("note: --count 0 copies the input without changing weights");
    }

    let bytes = std::fs::read(&args.input).map_err(|e| format!("failed to read {}: {e}", args.input.display()))?;
    let (nerfed, report) = nerf_sfnn_bytes(bytes, args)?;

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(&args.output, nerfed).map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;

    Ok(report)
}

// ----- dispatch ----------------------------------------------------------

fn main() {
    let mut raw_args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if raw_args.get(1).is_some_and(|arg| arg == std::ffi::OsStr::new("nerf")) {
        raw_args.remove(1);
        if let Some(program) = raw_args.get_mut(0) {
            *program = std::ffi::OsString::from("bulletou nerf");
        }
        let args = NerfArgs::parse_from(raw_args);
        match run_nerf(&args) {
            Ok(report) => {
                println!("nerf complete:");
                println!("  input              = {}", args.input.display());
                println!("  output             = {}", args.output.display());
                println!("  eval_type          = {:?}", args.eval_type);
                println!("  arch               = {}", args.arch);
                println!("  layerstack         = {}", args.effective_layerstack().cli_name());
                println!("  layers             = {}", args.layers);
                println!("  candidate_weights  = {}", report.candidate_weights);
                println!("    fc0              = {}", report.fc0_candidates);
                println!("    fc1              = {}", report.fc1_candidates);
                println!("    fc2              = {}", report.fc2_candidates);
                println!("  mutation_attempts  = {}", report.selected);
                println!("  changed            = {}", report.changed);
                println!("  saturated_noops    = {}", report.saturated_noops);
            }
            Err(e) => {
                eprintln!("error: nerf failed: {e}");
                std::process::exit(2);
            }
        }
        return;
    }

    let args = Args::parse();
    // `--count-teacher` operates standalone (no training): print position
    // counts for the supplied teacher path(s) and exit.
    if args.count_teacher {
        match run_count_teacher(&args.teacher) {
            Ok(()) => return,
            Err(e) => {
                eprintln!("error: --count-teacher failed: {e}");
                std::process::exit(2);
            }
        }
    }
    let batches_per_superbatch = effective_batches_per_superbatch(&args).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    if !(args.optimizer_weight_decay.is_finite() && args.optimizer_weight_decay >= 0.0) {
        eprintln!("error: --optimizer-weight-decay must be finite and >= 0.");
        std::process::exit(2);
    }
    if let Some(epsilon) = args.optimizer_epsilon {
        if !(epsilon.is_finite() && epsilon > 0.0) {
            eprintln!("error: --optimizer-epsilon must be finite and > 0.");
            std::process::exit(2);
        }
    }
    if let Some(beta1) = args.optimizer_beta1 {
        if !(beta1.is_finite() && beta1 > 0.0 && beta1 < 1.0) {
            eprintln!("error: --optimizer-beta1 must be finite and satisfy 0 < beta1 < 1.");
            std::process::exit(2);
        }
    }
    if let Some(beta2) = args.optimizer_beta2 {
        if !(beta2.is_finite() && beta2 > 0.0 && beta2 < 1.0) {
            eprintln!("error: --optimizer-beta2 must be finite and satisfy 0 < beta2 < 1.");
            std::process::exit(2);
        }
    }
    if !(args.nnue_pytorch_init_scale.is_finite() && args.nnue_pytorch_init_scale > 0.0) {
        eprintln!("error: --nnue-pytorch-init-scale must be finite and > 0.");
        std::process::exit(2);
    }
    if args.nnue_pytorch_init_scale != 1.0 && !args.eval_type().uses_layerstack() {
        eprintln!("error: --nnue-pytorch-init-scale currently applies to SFNN / LayerStack eval types only.");
        std::process::exit(2);
    }
    if args.sfnn_factorized_l1 && !args.eval_type().uses_layerstack() {
        eprintln!("error: --sfnn-factorized-l1 currently applies to SFNN / LayerStack eval types only.");
        std::process::exit(2);
    }
    if args.dump_activation_stats {
        if !args.eval_type().uses_layerstack() {
            eprintln!("error: --dump-activation-stats currently supports SFNN / LayerStack eval types only.");
            std::process::exit(2);
        }
        if args.activation_stats_positions == 0 {
            eprintln!("error: --activation-stats-positions must be > 0.");
            std::process::exit(2);
        }
    }
    // `geometric` and `cos` sweep from `--lr` (lr_max) down to `--lr-min`.
    // `step` and `plateau` reduce `--lr` multiplicatively down to
    // `--lr-min`. `--lr-min` must be > 0 for geometric/multiplicative schedules.
    // For cos, 0 is fine but unusual; warn rather than reject.
    if args.lr_min <= 0.0 {
        match args.lr_schedule {
            LrScheduleKind::Step | LrScheduleKind::Geometric | LrScheduleKind::Plateau => {
                eprintln!(
                    "error: --lr-min must be > 0 for --lr-schedule {}. \
                     1e-5 or 1e-6 is typical.",
                    match args.lr_schedule {
                        LrScheduleKind::Step => "step",
                        LrScheduleKind::Geometric => "geometric",
                        LrScheduleKind::Plateau => "plateau",
                        LrScheduleKind::Cos => unreachable!(),
                    }
                );
                std::process::exit(2);
            }
            LrScheduleKind::Cos => {
                eprintln!(
                    "  note: --lr-min 0.0 with --lr-schedule cos means lr \
                     literally touches 0 at each epoch end; 1e-5 or 1e-6 is \
                     usually preferred."
                );
            }
        }
    }
    if args.lr_schedule == LrScheduleKind::Step {
        if let Some(gamma) = args.lr_step_gamma {
            if !(gamma > 0.0 && gamma <= 1.0) {
                eprintln!("error: --lr-step-gamma must satisfy 0 < gamma <= 1 for --lr-schedule step.");
                std::process::exit(2);
            }
        }
        if let Err(e) = effective_lr_step_gamma(&args, batches_per_superbatch) {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
        if matches!(args.lr_step_positions, Some(0)) {
            eprintln!("error: --lr-step-positions must be > 0.");
            std::process::exit(2);
        }
        if args.lr <= 0.0 {
            eprintln!("error: --lr must be > 0 for --lr-schedule step.");
            std::process::exit(2);
        }
    }
    if args.lr_schedule != LrScheduleKind::Step {
        if let Some(gamma) = args.lr_step_gamma {
            if !(gamma > 0.0 && gamma <= 1.0) {
                eprintln!("error: --lr-step-gamma must satisfy 0 < gamma <= 1.");
                std::process::exit(2);
            }
        }
    }
    if args.nnue_pytorch_wrm_loss && !args.eval_type().supports_nnue_pytorch_wrm_loss() {
        eprintln!("error: --nnue-pytorch-wrm-loss currently applies to NNUE / SFNN eval types only.");
        std::process::exit(2);
    }
    if args.nnue_pytorch_layer_clip && !args.eval_type().uses_arch() {
        eprintln!("error: --nnue-pytorch-layer-clip currently applies to NNUE / SFNN eval types only.");
        std::process::exit(2);
    }
    if args.nnue_pytorch_no_bias_clip && !args.eval_type().uses_arch() {
        eprintln!("error: --nnue-pytorch-no-bias-clip currently applies to NNUE / SFNN eval types only.");
        std::process::exit(2);
    }
    if args.lr_schedule == LrScheduleKind::Plateau {
        if args.test_teacher.is_none() {
            eprintln!("error: --lr-schedule plateau requires --test-teacher so validation metrics can be monitored.");
            std::process::exit(2);
        }
        if effective_save_rate(&args) != 1 {
            eprintln!("error: --lr-schedule plateau requires --save-rate 1 for per-superbatch LR decisions.");
            std::process::exit(2);
        }
        if args.lr <= 0.0 {
            eprintln!("error: --lr must be > 0 for --lr-schedule plateau.");
            std::process::exit(2);
        }
        if args.lr < args.lr_min {
            eprintln!("error: --lr must be >= --lr-min for --lr-schedule plateau.");
            std::process::exit(2);
        }
        if !(args.lr_plateau_factor > 0.0 && args.lr_plateau_factor < 1.0) {
            eprintln!("error: --lr-plateau-factor must satisfy 0 < factor < 1.");
            std::process::exit(2);
        }
        if args.lr_plateau_min_delta < 0.0 {
            eprintln!("error: --lr-plateau-min-delta must be >= 0.");
            std::process::exit(2);
        }
        if matches!(args.eval_type(), EvalType::Kppt | EvalType::KppKkpt) {
            eprintln!("error: --lr-schedule plateau is currently supported for NNUE/SFNN eval types only.");
            std::process::exit(2);
        }
    }
    if let Err(e) = args.validate_arch_flags() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
    if let Err(e) = args.validate_backend_flags() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
    if args.backend == BackendKind::CudaCpp {
        if !args.cuda_cpp_smoke {
            prepare_resume_config_or_exit(&args);
            if let Err(e) = record_invocation_to_tag_txt(&args) {
                eprintln!("warning: failed to write tag.txt under {}: {e}", args.output_dir().display());
            }
        }
        if let Err(e) = run_cuda_cpp_backend(&args) {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
        return;
    }
    if args.batch_size.is_none() {
        eprintln!("  batch size = {} (auto for {})", effective_batch_size(&args), args.eval_type().cli_name());
    }
    if args.nnue_pytorch_wrm_loss {
        eprintln!("  nnue-pytorch WRM loss = enabled");
    }
    if args.optimizer != OptimizerKind::Ranger {
        eprintln!("  optimizer = {}", args.optimizer.cli_name());
    }
    if args.optimizer_weight_decay != 0.0 {
        eprintln!("  optimizer weight decay = {}", args.optimizer_weight_decay);
    }
    if let Some(epsilon) = args.optimizer_epsilon {
        eprintln!("  optimizer epsilon = {}", epsilon);
    }
    if let Some(beta1) = args.optimizer_beta1 {
        eprintln!("  optimizer beta1 = {}", beta1);
    }
    if let Some(beta2) = args.optimizer_beta2 {
        eprintln!("  optimizer beta2 = {}", beta2);
    }
    if args.lr_schedule == LrScheduleKind::Plateau {
        eprintln!("  plateau monitor = {}", args.lr_plateau_monitor.cli_name());
    }
    if args.lr_schedule == LrScheduleKind::Step {
        let (gamma, auto_gamma) = effective_lr_step_gamma(&args, batches_per_superbatch).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });
        let gamma_source = if args.lr_step_gamma.is_some() {
            "explicit"
        } else if auto_gamma {
            "auto"
        } else {
            "default"
        };
        eprintln!(
            "  step scheduler = gamma {} ({gamma_source}), step_positions {}, epoch_positions {}",
            gamma,
            args.lr_step_positions.map(|v| v.to_string()).unwrap_or_else(|| "one superbatch".to_string()),
            args.superbatches
                .map(|n| (n as u64)
                    .saturating_mul(effective_batch_size(&args) as u64)
                    .saturating_mul(batches_per_superbatch as u64)
                    .to_string())
                .unwrap_or_else(|| "open-ended".to_string())
        );
    }
    if args.nnue_pytorch_layer_clip {
        eprintln!(
            "  nnue-pytorch layer clipping = enabled (hidden +/-{:.6}, output weight +/-{:.6})",
            NNUE_PYTORCH_HIDDEN_CLIP, NNUE_PYTORCH_OUTPUT_CLIP
        );
    }
    if args.nnue_pytorch_no_bias_clip {
        eprintln!("  nnue-pytorch bias clipping = disabled");
    }
    prepare_resume_config_or_exit(&args);
    if let Err(e) = record_invocation_to_tag_txt(&args) {
        eprintln!("warning: failed to write tag.txt under {}: {e}", args.output_dir().display());
    }
    match args.eval_type() {
        EvalType::Kppt | EvalType::KppKkpt => run_kppt_all(&args),
        EvalType::NnueHalfkp => run_halfkp(&args),
        EvalType::NnueKp => run_kp(&args),
        EvalType::NnueKa2 => run_nnue_ka2(&args),
        EvalType::NnueHalfkpe9 => run_halfkpe9(&args),
        EvalType::NnueHalfkpvm => run_halfkpvm(&args),
        EvalType::SfnnHalfka1hm => run_sfnn_1536(&args, ShogiHalfKaHm1, NnueFeatureSet::HalfKaHm1),
        EvalType::SfnnHalfka2hm => run_sfnn_1536(&args, ShogiHalfKaHm2, NnueFeatureSet::HalfKaHm2),
        EvalType::SfnnHalfka2 => run_sfnn_1536(&args, ShogiHalfKa2, NnueFeatureSet::HalfKa2),
        EvalType::SfnnKa2 => run_sfnn_1536(&args, ShogiKa2, NnueFeatureSet::Ka2),
    }
}

fn run_cuda_cpp_backend(args: &Args) -> Result<(), String> {
    args.validate_cuda_cpp_backend_options()?;

    #[cfg(feature = "cuda-cpp-backend")]
    {
        if !args.cuda_cpp_smoke {
            return match args.eval_type() {
                EvalType::Kppt | EvalType::KppKkpt => run_cuda_cpp_kppt_direct_steps(args),
                EvalType::NnueHalfkp => run_cuda_cpp_halfkp_direct_steps(args),
                EvalType::NnueKp => run_cuda_cpp_kp_direct_steps(args),
                EvalType::SfnnHalfka2 => run_cuda_cpp_sfnn_halfka2_direct_steps(args),
                other => unreachable!("validated cuda-cpp direct trainer eval type reached dispatch: {other:?}"),
            };
        }

        use bulletou_cuda_cpp::{
            Context, Event, F32Buffer, F32UploadSlot, RAdamUpdateParams, RangerDeviceStateMut, RangerStateMut,
            RangerUpdateParams,
        };

        let device = args.cuda_cpp_device;
        eprintln!("  backend = cuda-cpp Windows-native smoke");
        let name = bulletou_cuda_cpp::device_name(device).map_err(|e| e.to_string())?;
        eprintln!("  cuda-cpp device {device}: {name}");

        let axpy = bulletou_cuda_cpp::axpy_host(device, 2.0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .map_err(|e| e.to_string())?;
        if axpy != vec![12.0, 24.0, 36.0, 48.0] {
            return Err(format!("cuda-cpp axpy smoke mismatch: {axpy:?}"));
        }

        let ctx = Context::new(device).map_err(|e| e.to_string())?;
        let x_dev = F32Buffer::from_host(&ctx, &[1.0, 2.0, 3.0, 4.0]).map_err(|e| e.to_string())?;
        let y_dev = F32Buffer::from_host(&ctx, &[10.0, 20.0, 30.0, 40.0]).map_err(|e| e.to_string())?;
        let out_dev = F32Buffer::new(&ctx, 4).map_err(|e| e.to_string())?;
        let start = Event::new(&ctx).map_err(|e| e.to_string())?;
        let stop = Event::new(&ctx).map_err(|e| e.to_string())?;
        start.record(&ctx).map_err(|e| e.to_string())?;
        bulletou_cuda_cpp::axpy_device(&ctx, 4, 2.0, &x_dev, &y_dev, &out_dev).map_err(|e| e.to_string())?;
        stop.record(&ctx).map_err(|e| e.to_string())?;
        stop.synchronize().map_err(|e| e.to_string())?;
        let _axpy_ms = stop.elapsed_ms_since(&start).map_err(|e| e.to_string())?;
        let axpy_device = out_dev.download(&ctx).map_err(|e| e.to_string())?;
        if axpy_device != vec![12.0, 24.0, 36.0, 48.0] {
            return Err(format!("cuda-cpp persistent axpy smoke mismatch: {axpy_device:?}"));
        }

        let graph_out = F32Buffer::new(&ctx, 4).map_err(|e| e.to_string())?;
        ctx.begin_capture().map_err(|e| e.to_string())?;
        bulletou_cuda_cpp::axpy_device(&ctx, 4, 2.0, &x_dev, &y_dev, &graph_out).map_err(|e| e.to_string())?;
        let graph = ctx.end_capture().map_err(|e| e.to_string())?;
        graph_out.fill(&ctx, 0.0).map_err(|e| e.to_string())?;
        graph.launch(&ctx).map_err(|e| e.to_string())?;
        graph.launch(&ctx).map_err(|e| e.to_string())?;
        ctx.synchronize().map_err(|e| e.to_string())?;
        let graph_axpy = graph_out.download(&ctx).map_err(|e| e.to_string())?;
        if graph_axpy != vec![12.0, 24.0, 36.0, 48.0] {
            return Err(format!("cuda-cpp graph axpy smoke mismatch: {graph_axpy:?}"));
        }

        let upload_ctx = Context::new(device).map_err(|e| e.to_string())?;
        let upload_x = F32UploadSlot::new(&upload_ctx, 4).map_err(|e| e.to_string())?;
        let upload_y = F32UploadSlot::new(&upload_ctx, 4).map_err(|e| e.to_string())?;
        upload_x.upload(&upload_ctx, &[1.0, 2.0, 3.0, 4.0]).map_err(|e| e.to_string())?;
        upload_y.upload(&upload_ctx, &[10.0, 20.0, 30.0, 40.0]).map_err(|e| e.to_string())?;
        let upload_out = F32Buffer::new(&ctx, 4).map_err(|e| e.to_string())?;
        let ready_x = upload_x.wait_on(&ctx).map_err(|e| e.to_string())?;
        let ready_y = upload_y.wait_on(&ctx).map_err(|e| e.to_string())?;
        bulletou_cuda_cpp::axpy_device(&ctx, 4, 2.0, ready_x, ready_y, &upload_out).map_err(|e| e.to_string())?;
        let upload_axpy = upload_out.download(&ctx).map_err(|e| e.to_string())?;
        if upload_axpy != vec![12.0, 24.0, 36.0, 48.0] {
            return Err(format!("cuda-cpp upload-slot axpy smoke mismatch: {upload_axpy:?}"));
        }

        let mut gradients = vec![0.25, -0.5, 1.0, -1.5];
        let mut weights = vec![0.1, -0.2, 0.3, -0.4];
        let mut momentum = vec![0.0; gradients.len()];
        let mut velocity = vec![0.0; gradients.len()];
        let mut slow_params = weights.clone();
        bulletou_cuda_cpp::ranger_update_host(
            device,
            RangerUpdateParams {
                radam: RAdamUpdateParams {
                    step: 1,
                    learning_rate: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    min_weight: -1.98,
                    max_weight: 1.98,
                    ..RAdamUpdateParams::default()
                },
                lookahead_alpha: 0.5,
                lookahead_period: 6,
            },
            RangerStateMut {
                gradients: &mut gradients,
                weights: &mut weights,
                momentum: &mut momentum,
                velocity: &mut velocity,
                slow_params: &mut slow_params,
            },
        )
        .map_err(|e| e.to_string())?;
        if !gradients.iter().all(|&g| g == 0.0) {
            return Err(format!("cuda-cpp ranger smoke did not reset gradients: {gradients:?}"));
        }

        let gradients_dev = F32Buffer::from_host(&ctx, &[0.25, -0.5, 1.0, -1.5]).map_err(|e| e.to_string())?;
        let weights_dev = F32Buffer::from_host(&ctx, &[0.1, -0.2, 0.3, -0.4]).map_err(|e| e.to_string())?;
        let momentum_dev = F32Buffer::from_host(&ctx, &[0.0; 4]).map_err(|e| e.to_string())?;
        let velocity_dev = F32Buffer::from_host(&ctx, &[0.0; 4]).map_err(|e| e.to_string())?;
        let slow_dev = F32Buffer::from_host(&ctx, &[0.1, -0.2, 0.3, -0.4]).map_err(|e| e.to_string())?;
        bulletou_cuda_cpp::ranger_update_device(
            &ctx,
            RangerUpdateParams {
                radam: RAdamUpdateParams {
                    step: 1,
                    learning_rate: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    min_weight: -1.98,
                    max_weight: 1.98,
                    ..RAdamUpdateParams::default()
                },
                lookahead_alpha: 0.5,
                lookahead_period: 6,
            },
            RangerDeviceStateMut {
                gradients: &gradients_dev,
                weights: &weights_dev,
                momentum: &momentum_dev,
                velocity: &velocity_dev,
                slow_params: &slow_dev,
            },
        )
        .map_err(|e| e.to_string())?;
        ctx.synchronize().map_err(|e| e.to_string())?;
        let gradients_device = gradients_dev.download(&ctx).map_err(|e| e.to_string())?;
        let weights_device = weights_dev.download(&ctx).map_err(|e| e.to_string())?;
        if !gradients_device.iter().all(|&g| g == 0.0) {
            return Err(format!("cuda-cpp persistent ranger smoke did not reset gradients: {gradients_device:?}"));
        }
        if weights_device != weights {
            return Err(format!(
                "cuda-cpp persistent ranger smoke mismatch: host={weights:?} device={weights_device:?}"
            ));
        }

        eprintln!("  cuda-cpp smoke = ok");
        Ok(())
    }

    #[cfg(not(feature = "cuda-cpp-backend"))]
    {
        let _ = args;
        Err("--backend cuda-cpp requires building with --features cuda-cpp-backend".to_string())
    }
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CudaCppNnueFeatureKind {
    Halfkp,
    Kp,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppNnueFeatureKind {
    fn train_label(self) -> &'static str {
        match self {
            Self::Halfkp => "NNUE_HALFKP",
            Self::Kp => "NNUE_KP",
        }
    }

    fn source_label(self) -> &'static str {
        match self {
            Self::Halfkp => "HalfKP",
            Self::Kp => "KP",
        }
    }

    fn feature_set(self) -> NnueFeatureSet {
        match self {
            Self::Halfkp => NnueFeatureSet::HalfKp,
            Self::Kp => NnueFeatureSet::Kp,
        }
    }

    fn base_input_size(self) -> usize {
        match self {
            Self::Halfkp => ShogiHalfKP.num_inputs(),
            Self::Kp => ShogiKp.num_inputs(),
        }
    }

    fn training_input_size(self) -> usize {
        match self {
            Self::Halfkp => ShogiHalfKP.num_inputs() + bulletou_lib::game::inputs::HALFKP_PIECE_INPUTS,
            Self::Kp => ShogiKp.num_inputs(),
        }
    }

    fn virtual_rows(self) -> usize {
        match self {
            Self::Halfkp => bulletou_lib::game::inputs::HALFKP_PIECE_INPUTS,
            Self::Kp => 0,
        }
    }

    fn max_active(self) -> usize {
        match self {
            Self::Halfkp => ShogiHalfKP.max_active(),
            Self::Kp => ShogiKp.max_active(),
        }
    }

    fn scratch_init_label(self) -> &'static str {
        match self {
            Self::Halfkp => "tatara-simple factorized scratch",
            Self::Kp => "deterministic NNUE_KP scratch",
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
struct CudaCppNnueTeacherBatch {
    batch: bulletou_lib::value::FastBatchHost,
    source: String,
    dataloader_pos: Option<bulletou_lib::value::TeacherDataloaderPos>,
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Clone, Copy)]
struct CudaCppNnueTeacherOptions {
    batch_size: usize,
    batch_index: usize,
    dataloader_resume_pos: Option<bulletou_lib::value::TeacherDataloaderPos>,
    loader_threads: usize,
    teacher_threads: usize,
    queue_depth: usize,
}

#[cfg(feature = "cuda-cpp-backend")]
fn for_each_cuda_cpp_nnue_teacher_batch<F>(
    args: &Args,
    feature_kind: CudaCppNnueFeatureKind,
    options: CudaCppNnueTeacherOptions,
    batch_count: usize,
    mut visitor: F,
) -> Result<usize, String>
where
    F: FnMut(CudaCppNnueTeacherBatch) -> Result<(), String>,
{
    match feature_kind {
        CudaCppNnueFeatureKind::Halfkp => {
            use bulletou_lib::value::{HalfkpTeacherBatchConfig, for_each_halfkp_teacher_fast_batch};
            let config = HalfkpTeacherBatchConfig {
                teacher: &args.teacher,
                batch_size: options.batch_size,
                batch_index: options.batch_index,
                dataloader_resume_pos: options.dataloader_resume_pos,
                buffer_mb: args.buffer_mb,
                loader_threads: options.loader_threads,
                threads: options.teacher_threads,
                queue_depth: options.queue_depth,
                lambda: args.lambda,
                scale: args.scale as f32,
                nnue_pytorch_wrm_loss: args.nnue_pytorch_wrm_loss,
                ft_factorize: false,
                score_drop_abs: (args.score_drop_abs > 0).then_some(args.score_drop_abs),
                profile_prepare: args.cuda_cpp_profile_teacher_prepare,
            };
            for_each_halfkp_teacher_fast_batch(&config, batch_count, |teacher_batch| {
                visitor(CudaCppNnueTeacherBatch {
                    batch: teacher_batch.batch,
                    source: teacher_batch.source,
                    dataloader_pos: teacher_batch.dataloader_pos,
                })
            })
            .map_err(|e| e.to_string())
        }
        CudaCppNnueFeatureKind::Kp => {
            use bulletou_lib::value::{KpTeacherBatchConfig, for_each_kp_teacher_fast_batch};
            let config = KpTeacherBatchConfig {
                teacher: &args.teacher,
                batch_size: options.batch_size,
                batch_index: options.batch_index,
                dataloader_resume_pos: options.dataloader_resume_pos,
                buffer_mb: args.buffer_mb,
                loader_threads: options.loader_threads,
                threads: options.teacher_threads,
                queue_depth: options.queue_depth,
                lambda: args.lambda,
                scale: args.scale as f32,
                nnue_pytorch_wrm_loss: args.nnue_pytorch_wrm_loss,
                score_drop_abs: (args.score_drop_abs > 0).then_some(args.score_drop_abs),
                profile_prepare: args.cuda_cpp_profile_teacher_prepare,
            };
            for_each_kp_teacher_fast_batch(&config, batch_count, |teacher_batch| {
                visitor(CudaCppNnueTeacherBatch {
                    batch: teacher_batch.batch,
                    source: teacher_batch.source,
                    dataloader_pos: teacher_batch.dataloader_pos,
                })
            })
            .map_err(|e| e.to_string())
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CudaCppKpptComponent {
    Kk,
    Kkp,
    Kpp,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppKpptComponent {
    fn label(self) -> &'static str {
        match self {
            Self::Kk => "KK",
            Self::Kkp => "KKP",
            Self::Kpp => "KPP",
        }
    }

    fn input_label(self) -> &'static str {
        match self {
            Self::Kk => "kk",
            Self::Kkp => "kkp",
            Self::Kpp => "kpp",
        }
    }

    fn table_weight_id(self) -> &'static str {
        match self {
            Self::Kk => "kkw",
            Self::Kkp => "kkpw",
            Self::Kpp => "kppw",
        }
    }

    fn table_bias_id(self) -> &'static str {
        match self {
            Self::Kk => "kkb",
            Self::Kkp => "kkpb",
            Self::Kpp => "kppb",
        }
    }

    fn input_size(self) -> usize {
        match self {
            Self::Kk => ShogiKk.num_inputs(),
            Self::Kkp => ShogiKkp.num_inputs(),
            Self::Kpp => ShogiKpp.num_inputs(),
        }
    }

    fn max_active(self) -> usize {
        match self {
            Self::Kk => ShogiKk.max_active(),
            Self::Kkp => ShogiKkp.max_active(),
            Self::Kpp => ShogiKpp.max_active(),
        }
    }

    fn default_quant_scale(self) -> f32 {
        match self {
            Self::Kk => KPPT_KK_DEFAULT_QUANT_SCALE,
            Self::Kkp => KPPT_KKP_DEFAULT_QUANT_SCALE,
            Self::Kpp => KPPT_KPP_DEFAULT_QUANT_SCALE,
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
struct CudaCppKpptTeacherBatch {
    batch: bulletou_lib::value::FastBatchHost,
    source: String,
    dataloader_pos: Option<bulletou_lib::value::TeacherDataloaderPos>,
}

#[cfg(feature = "cuda-cpp-backend")]
fn for_each_cuda_cpp_kppt_teacher_batch<F>(
    args: &Args,
    component: CudaCppKpptComponent,
    options: CudaCppNnueTeacherOptions,
    batch_count: usize,
    mut visitor: F,
) -> Result<usize, String>
where
    F: FnMut(CudaCppKpptTeacherBatch) -> Result<(), String>,
{
    use bulletou_lib::value::{KpptTeacherBatchConfig, for_each_kppt_teacher_fast_batch};

    let config = KpptTeacherBatchConfig {
        teacher: &args.teacher,
        batch_size: options.batch_size,
        batch_index: options.batch_index,
        dataloader_resume_pos: options.dataloader_resume_pos,
        buffer_mb: args.buffer_mb,
        loader_threads: options.loader_threads,
        threads: options.teacher_threads,
        queue_depth: options.queue_depth,
        lambda: args.lambda,
        scale: args.scale as f32,
        nnue_pytorch_wrm_loss: args.nnue_pytorch_wrm_loss,
        score_drop_abs: (args.score_drop_abs > 0).then_some(args.score_drop_abs),
        profile_prepare: args.cuda_cpp_profile_teacher_prepare,
    };

    match component {
        CudaCppKpptComponent::Kk => {
            for_each_kppt_teacher_fast_batch(ShogiKk, component.input_label(), &config, batch_count, |teacher_batch| {
                visitor(CudaCppKpptTeacherBatch {
                    batch: teacher_batch.batch,
                    source: teacher_batch.source,
                    dataloader_pos: teacher_batch.dataloader_pos,
                })
            })
        }
        CudaCppKpptComponent::Kkp => {
            for_each_kppt_teacher_fast_batch(ShogiKkp, component.input_label(), &config, batch_count, |teacher_batch| {
                visitor(CudaCppKpptTeacherBatch {
                    batch: teacher_batch.batch,
                    source: teacher_batch.source,
                    dataloader_pos: teacher_batch.dataloader_pos,
                })
            })
        }
        CudaCppKpptComponent::Kpp => {
            for_each_kppt_teacher_fast_batch(ShogiKpp, component.input_label(), &config, batch_count, |teacher_batch| {
                visitor(CudaCppKpptTeacherBatch {
                    batch: teacher_batch.batch,
                    source: teacher_batch.source,
                    dataloader_pos: teacher_batch.dataloader_pos,
                })
            })
        }
    }
    .map_err(|e| e.to_string())
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppKpptInitialWeights {
    shape: bulletou_cuda_cpp::KpptTableShape,
    table_w: Vec<f32>,
    table_b: Vec<f32>,
    outw: Vec<f32>,
    outb: Vec<f32>,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppKpptInitialWeights {
    fn validate(&self) -> Result<(), String> {
        self.as_host().validate().map_err(|e| e.to_string())
    }

    fn as_host(&self) -> bulletou_cuda_cpp::KpptTableForwardHostWeights<'_> {
        bulletou_cuda_cpp::KpptTableForwardHostWeights {
            shape: self.shape,
            table_w: &self.table_w,
            table_b: &self.table_b,
            outw: &self.outw,
            outb: &self.outb,
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppKpptInitialState {
    weights: CudaCppKpptInitialWeights,
    optimizer_states: Option<CudaCppKpptOptimizerState>,
    completed_steps: usize,
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppKpptOptimizerState {
    table_w: CudaCppRangerGroupState,
    table_b: CudaCppRangerGroupState,
    outw: CudaCppRangerGroupState,
    outb: CudaCppRangerGroupState,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppKpptOptimizerState {
    fn as_host(&self) -> bulletou_cuda_cpp::KpptTableRangerOptimizerHostStates<'_> {
        bulletou_cuda_cpp::KpptTableRangerOptimizerHostStates {
            table_w: self.table_w.as_host(),
            table_b: self.table_b.as_host(),
            outw: self.outw.as_host(),
            outb: self.outb.as_host(),
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_kppt_direct_steps(args: &Args) -> Result<(), String> {
    let schedule = cuda_cpp_run_schedule(args)?;
    let train_steps = schedule.total_steps;
    let batch_size = effective_batch_size(args);
    let device = args.cuda_cpp_device;
    let output_dir = args.output_dir();

    eprintln!(
        "  backend = cuda-cpp Windows-native direct {} trainer (KK + KKP + KPP, {train_steps} batch step{} each)",
        args.eval_type().cli_name(),
        if train_steps == 1 { "" } else { "s" }
    );
    if schedule.production {
        eprintln!(
            "  cuda-cpp KPPT schedule = production: max_epochs={}, superbatches={}, save_rate={}, save_epoch_end={}, batches_per_superbatch={}, lr={}",
            args.max_epochs.unwrap_or(1).max(1),
            args.superbatches.unwrap_or(1),
            effective_save_rate(args),
            effective_save_epoch_end(args),
            schedule.batches_per_superbatch,
            args.lr_schedule.cli_name()
        );
    } else {
        eprintln!("  cuda-cpp KPPT schedule = direct train-steps smoke mode");
    }
    let name = bulletou_cuda_cpp::device_name(device).map_err(|e| e.to_string())?;
    eprintln!("  cuda-cpp device {device}: {name}");
    eprintln!("  batch size = {batch_size}");
    eprintln!("  loss = {}", if args.nnue_pytorch_wrm_loss { "nnue-pytorch-wrm" } else { "sigmoid-mse" });

    for component in [CudaCppKpptComponent::Kk, CudaCppKpptComponent::Kkp, CudaCppKpptComponent::Kpp] {
        eprintln!("\n=== [cuda-cpp {}] training ===", component.label());
        run_cuda_cpp_kppt_component_direct_steps(args, component, &schedule)?;
    }

    let ctx = LogContext::from_args(args, schedule.lr_period);
    let prior_positions = read_prior_positions(&output_dir.join(SUMMARY_LEARN_LOG_NAME));
    match assemble_numbered_dirs(&output_dir, &ctx, &prior_positions, STATE_BACKEND_CUDA_CPP) {
        Ok((_first_idx, last_idx)) => {
            append_to_top_level_log(&output_dir, last_idx).map_err(|err| {
                format!("failed to update {}: {err}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display())
            })?;
        }
        Err(err) => return Err(format!("failed to assemble cuda-cpp KPPT numbered checkpoint dirs: {err}")),
    }

    Ok(())
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_kppt_component_direct_steps(
    args: &Args,
    component: CudaCppKpptComponent,
    schedule: &CudaCppRunSchedule,
) -> Result<(), String> {
    use bulletou_cuda_cpp::{
        Context, KpptTableTrainStepHostBatch, KpptTableTrainStepRunner, RAdamUpdateParams, RangerUpdateParams,
        ScalarLossKind,
    };

    let train_steps = schedule.total_steps;
    let batch_size = effective_batch_size(args);
    let max_active = component.max_active();
    let input_size = component.input_size();
    let device = args.cuda_cpp_device;
    let auto_resume_state_bin = cuda_cpp_auto_resume_state_bin(args);
    let ctx = Context::new(device).map_err(|e| e.to_string())?;

    eprintln!("  cuda-cpp {} input = dims={}, max_active={}", component.label(), input_size, max_active);
    let (mut runner, completed_step_offset) = {
        let initial_state = build_cuda_cpp_kppt_initial_state(args, component)?;
        if let Some(path) = args.cuda_cpp_weights_bin.as_deref() {
            let state_kind = if initial_state.optimizer_states.is_some() {
                "weights + Ranger optimizer state"
            } else if initial_state.completed_steps > 0 {
                "weights + step counters"
            } else {
                "weights only"
            };
            eprintln!("  initial {} state = {} ({state_kind})", component.label(), path.display());
        } else if let Some(path) = auto_resume_state_bin.as_deref() {
            eprintln!(
                "  initial {} state = {} (auto-resume weights + Ranger optimizer state)",
                component.label(),
                path.display()
            );
        } else {
            eprintln!("  initial {} weights = zero table, outw=[1,1], outb=0", component.label());
        }
        if initial_state.completed_steps > 0 {
            eprintln!("  initial {} completed optimizer steps = {}", component.label(), initial_state.completed_steps);
        }

        let initial_host_weights = initial_state.weights.as_host();
        let runner = match initial_state.optimizer_states.as_ref() {
            Some(optimizer_states) => KpptTableTrainStepRunner::with_optimizer_states(
                &ctx,
                initial_host_weights,
                optimizer_states.as_host(),
                batch_size,
                max_active,
            ),
            None => KpptTableTrainStepRunner::new(&ctx, initial_host_weights, batch_size, max_active),
        }
        .map_err(|e| e.to_string())?;
        (runner, initial_state.completed_steps)
    };

    let loss_kind =
        if args.nnue_pytorch_wrm_loss { ScalarLossKind::NnuePytorchWrm } else { ScalarLossKind::SigmoidMse };
    let output_inv_scale = 1.0_f32;
    let mut seen_steps = 0usize;
    let mut reported_loss_sum = 0.0_f64;
    let mut reported_loss_count = 0usize;
    let mut last_loss = 0.0_f32;
    let mut checkpoint_chunk_idx = 0usize;
    let mut last_dataloader_pos = None;
    let dataloader_resume_pos = cuda_cpp_auto_resume_dataloader_pos(args);
    if let Some(pos) = dataloader_resume_pos {
        eprintln!("  dataloader resume = byte_offset {}, plies {}", pos.byte_offset, pos.plies);
    }
    let teacher_threads = cuda_cpp_effective_teacher_threads(args);
    let loader_threads = cuda_cpp_effective_loader_threads(args);
    let batch_queue_size = cuda_cpp_effective_batch_queue_size(args);
    eprintln!(
        "  cuda-cpp {} teacher CPU = prepare_threads={}, loader_threads={}, batch_queue_size={}",
        component.label(),
        teacher_threads,
        loader_threads,
        batch_queue_size
    );
    let started = std::time::Instant::now();
    let teacher_options = CudaCppNnueTeacherOptions {
        batch_size,
        batch_index: 0,
        dataloader_resume_pos,
        loader_threads,
        teacher_threads,
        queue_depth: batch_queue_size,
    };

    for_each_cuda_cpp_kppt_teacher_batch(args, component, teacher_options, train_steps, |teacher_batch| {
        seen_steps += 1;
        last_dataloader_pos = teacher_batch.dataloader_pos;
        let optimizer_step = completed_step_offset + seen_steps;
        let checkpoint_chunk = schedule.chunks.get(checkpoint_chunk_idx);
        let is_checkpoint_step = checkpoint_chunk.is_some_and(|chunk| chunk.cumulative_steps == seen_steps);
        let fast = teacher_batch.batch;
        let params = {
            let ranger = ranger_params(args, BULLETOU_DEFAULT_ADAMW_CLIP);
            let step_index = seen_steps.saturating_sub(1);
            let learning_rate = schedule.lr_for_step(args, step_index, batch_size);
            RangerUpdateParams {
                radam: RAdamUpdateParams {
                    step: optimizer_step as u64,
                    learning_rate,
                    decay: ranger.decay,
                    beta1: ranger.beta1,
                    beta2: ranger.beta2,
                    epsilon: ranger.epsilon,
                    min_weight: ranger.min_weight,
                    max_weight: ranger.max_weight,
                    ..RAdamUpdateParams::default()
                },
                lookahead_alpha: ranger.alpha,
                lookahead_period: ranger.k as u64,
            }
        };
        let batch = KpptTableTrainStepHostBatch {
            stm_indices: &fast.stm,
            nstm_indices: &fast.nstm,
            targets: &fast.targets,
            entry_weights: &fast.weights,
            batch_size: fast.layout.batch_size,
            max_active: fast.layout.max_active,
        };
        let should_report = is_checkpoint_step
            || cuda_cpp_should_read_loss(seen_steps, train_steps, args.cuda_cpp_loss_readback_interval);
        runner
            .step_no_readback_with_loss_finalize(&ctx, params, loss_kind, output_inv_scale, batch, should_report)
            .map_err(|e| e.to_string())?;
        if should_report {
            let loss = runner.read_loss(&ctx).map_err(|e| e.to_string())?;
            last_loss = loss.mean;
            reported_loss_sum += f64::from(loss.mean);
            reported_loss_count += 1;
            eprintln!(
                "  cuda-cpp {} step {seen_steps:>6}/{train_steps:<6} loss_mean={:.8} source={}",
                component.label(),
                loss.mean,
                teacher_batch.source
            );
        }
        if is_checkpoint_step {
            let chunk = schedule.chunks[checkpoint_chunk_idx].clone();
            let dataloader_pos = cuda_cpp_direct_dataloader_pos(args, seen_steps, batch_size, last_dataloader_pos)?;
            if chunk.save_checkpoint {
                let trained_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
                let trained_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                let checkpoint_dir = write_cuda_cpp_kppt_component_checkpoint(
                    args,
                    component,
                    &trained_weights,
                    &trained_optimizer_states,
                    completed_step_offset + seen_steps,
                    last_loss,
                    &chunk,
                    schedule.batches_per_superbatch,
                    dataloader_pos,
                )?;
                eprintln!("  cuda-cpp {} component checkpoint = {}", component.label(), checkpoint_dir.display());
            } else {
                eprintln!(
                    "  cuda-cpp {} checkpoint skipped at epoch={}, superbatch={} (--no-save-epoch-end)",
                    component.label(),
                    chunk.epoch,
                    chunk.superbatch
                );
            }
            checkpoint_chunk_idx += 1;
        }
        Ok::<(), String>(())
    })?;

    ctx.synchronize().map_err(|e| e.to_string())?;
    if checkpoint_chunk_idx != schedule.chunks.len() {
        return Err(format!(
            "cuda-cpp {} schedule ended after {checkpoint_chunk_idx} checkpoints, expected {}",
            component.label(),
            schedule.chunks.len()
        ));
    }

    let elapsed = started.elapsed().as_secs_f64();
    let positions = seen_steps.saturating_mul(batch_size);
    let positions_per_sec = if elapsed > 0.0 { positions as f64 / elapsed } else { 0.0 };
    let reported_avg_loss = if reported_loss_count > 0 { reported_loss_sum / reported_loss_count as f64 } else { 0.0 };
    eprintln!(
        "  cuda-cpp {} train = ok: steps={seen_steps}, positions={positions}, elapsed={elapsed:.3}s, \
         throughput={positions_per_sec:.0} pos/s, reported_avg_loss={reported_avg_loss:.8}, last_loss={last_loss:.8}",
        component.label()
    );

    Ok(())
}

#[cfg(feature = "cuda-cpp-backend")]
fn build_cuda_cpp_kppt_initial_state(
    args: &Args,
    component: CudaCppKpptComponent,
) -> Result<CudaCppKpptInitialState, String> {
    if let Some(path) = args.cuda_cpp_weights_bin.as_deref() {
        return load_cuda_cpp_kppt_initial_state(path, component);
    }
    if let Some(path) = cuda_cpp_auto_resume_state_bin(args) {
        return load_cuda_cpp_kppt_initial_state(&path, component);
    }

    let weights = CudaCppKpptInitialWeights {
        shape: bulletou_cuda_cpp::KpptTableShape { input_size: component.input_size() },
        table_w: vec![0.0; component.input_size()],
        table_b: vec![0.0],
        outw: vec![1.0, 1.0],
        outb: vec![0.0],
    };
    weights.validate()?;
    Ok(CudaCppKpptInitialState { weights, optimizer_states: None, completed_steps: 0 })
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_kppt_initial_state(
    path: &Path,
    component: CudaCppKpptComponent,
) -> Result<CudaCppKpptInitialState, String> {
    let bytes = std::fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let records =
        parse_model_weights_bin(&bytes).map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let component_weights =
        bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, component.input_label(), "weights");
    let weights_records = if component_weights.is_empty() { &records } else { &component_weights };
    let weights = CudaCppKpptInitialWeights {
        shape: bulletou_cuda_cpp::KpptTableShape { input_size: component.input_size() },
        table_w: load_cuda_cpp_kppt_weight_record(weights_records, component.table_weight_id())?,
        table_b: load_cuda_cpp_kppt_weight_record(weights_records, component.table_bias_id())?,
        outw: load_cuda_cpp_kppt_weight_record(weights_records, "outw")?,
        outb: load_cuda_cpp_kppt_weight_record(weights_records, "outb")?,
    };
    weights.validate().map_err(|err| {
        format!("failed to load cuda-cpp {} weights from {}: {err}", component.label(), path.display())
    })?;
    let optimizer_states = load_cuda_cpp_kppt_optimizer_state(&records, component, &weights)?;
    let completed_steps = load_cuda_cpp_kppt_completed_steps(&records, component)?;
    Ok(CudaCppKpptInitialState { weights, optimizer_states, completed_steps })
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_kppt_weight_record(
    records: &BTreeMap<String, Vec<f32>>,
    id: &'static str,
) -> Result<Vec<f32>, String> {
    records.get(id).cloned().ok_or_else(|| format!("cuda-cpp KPPT state missing weight `{id}`"))
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_kppt_optimizer_state(
    records: &BTreeMap<String, Vec<f32>>,
    component: CudaCppKpptComponent,
    weights: &CudaCppKpptInitialWeights,
) -> Result<Option<CudaCppKpptOptimizerState>, String> {
    let comp = component.input_label();
    let momentum = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, comp, "momentum");
    let velocity = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, comp, "velocity");
    let slow = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, comp, "slow");
    let has_any = !momentum.is_empty() || !velocity.is_empty() || !slow.is_empty();
    if !has_any {
        return Ok(None);
    }
    if momentum.is_empty() || velocity.is_empty() || slow.is_empty() {
        return Err(format!(
            "cuda-cpp {} optimizer state is partial: expected {comp}/{{momentum,velocity,slow}}/* records",
            component.label()
        ));
    }

    Ok(Some(CudaCppKpptOptimizerState {
        table_w: load_cuda_cpp_kppt_ranger_group_state(
            component.label(),
            component.table_weight_id(),
            weights.table_w.len(),
            &momentum,
            &velocity,
            &slow,
        )?,
        table_b: load_cuda_cpp_kppt_ranger_group_state(
            component.label(),
            component.table_bias_id(),
            weights.table_b.len(),
            &momentum,
            &velocity,
            &slow,
        )?,
        outw: load_cuda_cpp_kppt_ranger_group_state(
            component.label(),
            "outw",
            weights.outw.len(),
            &momentum,
            &velocity,
            &slow,
        )?,
        outb: load_cuda_cpp_kppt_ranger_group_state(
            component.label(),
            "outb",
            weights.outb.len(),
            &momentum,
            &velocity,
            &slow,
        )?,
    }))
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_kppt_ranger_group_state(
    label: &'static str,
    id: &'static str,
    expected_len: usize,
    momentum: &BTreeMap<String, Vec<f32>>,
    velocity: &BTreeMap<String, Vec<f32>>,
    slow: &BTreeMap<String, Vec<f32>>,
) -> Result<CudaCppRangerGroupState, String> {
    Ok(CudaCppRangerGroupState {
        momentum: load_cuda_cpp_kppt_optimizer_record(label, "momentum", momentum, id, expected_len)?,
        velocity: load_cuda_cpp_kppt_optimizer_record(label, "velocity", velocity, id, expected_len)?,
        slow_params: load_cuda_cpp_kppt_optimizer_record(label, "slow", slow, id, expected_len)?,
    })
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_kppt_optimizer_record(
    label: &'static str,
    section: &'static str,
    records: &BTreeMap<String, Vec<f32>>,
    id: &'static str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let values =
        records.get(id).ok_or_else(|| format!("cuda-cpp KPPT/{label} optimizer state missing {section}/{id}"))?;
    if values.len() != expected_len {
        return Err(format!(
            "cuda-cpp KPPT/{label} optimizer state {section}/{id} has length {}, expected {}",
            values.len(),
            expected_len
        ));
    }
    Ok(values.clone())
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_kppt_completed_steps(
    records: &BTreeMap<String, Vec<f32>>,
    component: CudaCppKpptComponent,
) -> Result<usize, String> {
    let steps =
        bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, component.input_label(), "step_ranger");
    if steps.is_empty() {
        return Ok(0);
    }
    let ids = [component.table_weight_id(), component.table_bias_id(), "outw", "outb"];
    let mut completed_steps: Option<usize> = None;
    for id in ids {
        let values = steps.get(id).ok_or_else(|| {
            format!("cuda-cpp {} state missing {}/step_ranger/{id}", component.label(), component.input_label())
        })?;
        let value = values.first().copied().ok_or_else(|| {
            format!("cuda-cpp {} state {}/step_ranger/{id} is empty", component.label(), component.input_label())
        })?;
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "cuda-cpp {} state {}/step_ranger/{id} is invalid: {value}",
                component.label(),
                component.input_label()
            ));
        }
        let step = value.round() as usize;
        if let Some(prev) = completed_steps {
            if prev != step {
                return Err(format!(
                    "cuda-cpp {} state has inconsistent step_ranger counters: first={prev}, {id}={step}",
                    component.label()
                ));
            }
        } else {
            completed_steps = Some(step);
        }
    }
    Ok(completed_steps.unwrap_or(0))
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_kppt_component_checkpoint(
    args: &Args,
    component: CudaCppKpptComponent,
    weights: &bulletou_cuda_cpp::KpptTableTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::KpptTableRangerOptimizerStatesReadback,
    completed_steps: usize,
    train_loss: f32,
    chunk: &CudaCppScheduleChunk,
    curr_batch: usize,
    dataloader_pos: bulletou_lib::value::TeacherDataloaderPos,
) -> Result<std::path::PathBuf, String> {
    let output_dir = args.output_dir();
    std::fs::create_dir_all(&output_dir).map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;
    let net_id = if args.max_epochs.unwrap_or(1) > 1 || chunk.epoch > 1 {
        format!("{}-e{}", component.input_label(), chunk.epoch)
    } else {
        component.input_label().to_string()
    };
    let dir = output_dir.join(format!("{net_id}-{}", chunk.superbatch));
    if dir.exists() {
        return Err(format!("refusing to overwrite existing component checkpoint {}", dir.display()));
    }
    std::fs::create_dir_all(&dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    let optimiser_state_dir = dir.join("optimiser_state");
    std::fs::create_dir_all(&optimiser_state_dir)
        .map_err(|err| format!("failed to create {}: {err}", optimiser_state_dir.display()))?;

    write_cuda_cpp_kppt_component_optimizer_files(
        component,
        &optimiser_state_dir,
        weights,
        optimizer_states,
        completed_steps,
    )?;
    std::fs::write(dir.join("log.txt"), format!("{},{},{:.8}\n", chunk.superbatch, curr_batch, train_loss))
        .map_err(|err| format!("failed to write {}: {err}", dir.join("log.txt").display()))?;
    std::fs::write(dir.join("teacher.txt"), format!("{}\n", args.teacher))
        .map_err(|err| format!("failed to write {}: {err}", dir.join("teacher.txt").display()))?;
    std::fs::write(
        dir.join("dataloader_pos.txt"),
        format!("{},{}\n", dataloader_pos.byte_offset, dataloader_pos.plies),
    )
    .map_err(|err| format!("failed to write {}: {err}", dir.join("dataloader_pos.txt").display()))?;

    let quant_scale = args.yaneuraou_quant_scale.unwrap_or(component.default_quant_scale());
    save_yaneuraou_eval(&dir, quant_scale, args.kpp_format())
        .map_err(|err| format!("failed to write YaneuraOu {} eval in {}: {err}", component.label(), dir.display()))?;
    Ok(dir)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_kppt_component_optimizer_files(
    component: CudaCppKpptComponent,
    optimiser_state_dir: &Path,
    weights: &bulletou_cuda_cpp::KpptTableTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::KpptTableRangerOptimizerStatesReadback,
    completed_steps: usize,
) -> Result<(), String> {
    let table_weight_id = component.table_weight_id();
    let table_bias_id = component.table_bias_id();
    let weight_records = [
        (table_weight_id, weights.table_w.as_slice()),
        (table_bias_id, weights.table_b.as_slice()),
        ("outw", weights.outw.as_slice()),
        ("outb", weights.outb.as_slice()),
    ];
    std::fs::write(
        optimiser_state_dir.join("weights.bin"),
        bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(weight_records),
    )
    .map_err(|err| format!("failed to write {}: {err}", optimiser_state_dir.join("weights.bin").display()))?;

    let momentum_records = [
        (table_weight_id, optimizer_states.table_w.momentum.as_slice()),
        (table_bias_id, optimizer_states.table_b.momentum.as_slice()),
        ("outw", optimizer_states.outw.momentum.as_slice()),
        ("outb", optimizer_states.outb.momentum.as_slice()),
    ];
    std::fs::write(
        optimiser_state_dir.join("momentum.bin"),
        bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(momentum_records),
    )
    .map_err(|err| format!("failed to write {}: {err}", optimiser_state_dir.join("momentum.bin").display()))?;

    let velocity_records = [
        (table_weight_id, optimizer_states.table_w.velocity.as_slice()),
        (table_bias_id, optimizer_states.table_b.velocity.as_slice()),
        ("outw", optimizer_states.outw.velocity.as_slice()),
        ("outb", optimizer_states.outb.velocity.as_slice()),
    ];
    std::fs::write(
        optimiser_state_dir.join("velocity.bin"),
        bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(velocity_records),
    )
    .map_err(|err| format!("failed to write {}: {err}", optimiser_state_dir.join("velocity.bin").display()))?;

    let slow_records = [
        (table_weight_id, optimizer_states.table_w.slow_params.as_slice()),
        (table_bias_id, optimizer_states.table_b.slow_params.as_slice()),
        ("outw", optimizer_states.outw.slow_params.as_slice()),
        ("outb", optimizer_states.outb.slow_params.as_slice()),
    ];
    std::fs::write(
        optimiser_state_dir.join("slow.bin"),
        bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(slow_records),
    )
    .map_err(|err| format!("failed to write {}: {err}", optimiser_state_dir.join("slow.bin").display()))?;

    let step_text = format!(
        "{},{}\n{},{}\noutw,{}\noutb,{}\n",
        table_weight_id, completed_steps, table_bias_id, completed_steps, completed_steps, completed_steps
    );
    std::fs::write(optimiser_state_dir.join("step_ranger.txt"), step_text)
        .map_err(|err| format!("failed to write {}: {err}", optimiser_state_dir.join("step_ranger.txt").display()))
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_halfkp_direct_steps(args: &Args) -> Result<(), String> {
    run_cuda_cpp_nnue_direct_steps(args, CudaCppNnueFeatureKind::Halfkp)
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_kp_direct_steps(args: &Args) -> Result<(), String> {
    run_cuda_cpp_nnue_direct_steps(args, CudaCppNnueFeatureKind::Kp)
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_nnue_direct_steps(args: &Args, feature_kind: CudaCppNnueFeatureKind) -> Result<(), String> {
    use bulletou_cuda_cpp::{
        Context, NnueForwardHostWeights as CudaNnueForwardHostWeights, NnueForwardShape as CudaNnueForwardShape,
        NnueTrainStepHostBatch, NnueTrainStepRunner, RAdamUpdateParams, RangerUpdateParams, ScalarLossKind,
    };

    let schedule = cuda_cpp_run_schedule(args)?;
    let train_steps = schedule.total_steps;
    let batch_size = effective_batch_size(args);
    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let device = args.cuda_cpp_device;

    eprintln!(
        "  backend = cuda-cpp Windows-native direct {} trainer ({train_steps} batch step{})",
        feature_kind.train_label(),
        if train_steps == 1 { "" } else { "s" }
    );
    if schedule.production {
        eprintln!(
            "  cuda-cpp schedule = production: max_epochs={}, superbatches={}, save_rate={}, save_epoch_end={}, batches_per_superbatch={}, lr={}",
            args.max_epochs.unwrap_or(1).max(1),
            args.superbatches.unwrap_or(1),
            effective_save_rate(args),
            effective_save_epoch_end(args),
            schedule.batches_per_superbatch,
            args.lr_schedule.cli_name()
        );
    } else {
        eprintln!("  cuda-cpp schedule = direct train-steps smoke mode");
    }
    let name = bulletou_cuda_cpp::device_name(device).map_err(|e| e.to_string())?;
    eprintln!("  cuda-cpp device {device}: {name}");
    let auto_resume_state_bin = cuda_cpp_auto_resume_state_bin(args);
    let initial_state = build_nnue_initial_state_for_cuda_cpp(args, feature_kind)?;
    let initial_weights = &initial_state.weights;
    if let Some(path) = args.cuda_cpp_weights_bin.as_deref() {
        let state_kind = if initial_state.optimizer_states.is_some() {
            "weights + Ranger optimizer state"
        } else if initial_state.completed_steps > 0 {
            "weights + step counters"
        } else {
            "weights only"
        };
        eprintln!("  initial state = {} ({state_kind})", path.display());
    } else if let Some(path) = auto_resume_state_bin.as_deref() {
        eprintln!("  initial state = {} (auto-resume weights + Ranger optimizer state)", path.display());
    } else {
        eprintln!("  initial weights = {}", feature_kind.scratch_init_label());
    }
    if initial_state.completed_steps > 0 {
        eprintln!("  initial completed optimizer steps = {}", initial_state.completed_steps);
    }
    let input_size = initial_weights.shape.input_size;
    let max_active = feature_kind.max_active();
    eprintln!("  batch size = {batch_size}");
    match feature_kind {
        CudaCppNnueFeatureKind::Halfkp => eprintln!(
            "  arch = {} (factorized input {input_size}, implicit HalfKP piece rows, {l1_size}x2-{l2_size}-{l3_size})",
            args.arch().cli_name()
        ),
        CudaCppNnueFeatureKind::Kp => {
            eprintln!("  arch = {} (KP input {input_size}, {l1_size}x2-{l2_size}-{l3_size})", args.arch().cli_name())
        }
    }
    eprintln!("  loss = {}", if args.nnue_pytorch_wrm_loss { "nnue-pytorch-wrm" } else { "sigmoid-mse" });
    let profile_steps = args.cuda_cpp_profile_steps.min(train_steps);
    if profile_steps > 0 {
        eprintln!("  cuda-cpp profile steps = {profile_steps}");
    }
    eprintln!(
        "  cuda-cpp loss readback interval = {}",
        if args.cuda_cpp_loss_readback_interval == 0 {
            "final-only".to_string()
        } else {
            args.cuda_cpp_loss_readback_interval.to_string()
        }
    );

    let cuda_shape = CudaNnueForwardShape {
        input_size: initial_weights.shape.input_size,
        l1: initial_weights.shape.l1,
        l2: initial_weights.shape.l2,
        l3: initial_weights.shape.l3,
    };
    let ctx = Context::new(device).map_err(|e| e.to_string())?;
    let initial_host_weights = CudaNnueForwardHostWeights {
        shape: cuda_shape,
        l0w: &initial_weights.l0w,
        l0b: &initial_weights.l0b,
        l1w: &initial_weights.l1w,
        l1b: &initial_weights.l1b,
        l2w: &initial_weights.l2w,
        l2b: &initial_weights.l2b,
        outw: &initial_weights.outw,
        outb: &initial_weights.outb,
    };
    let mut runner = match initial_state.optimizer_states.as_ref() {
        Some(optimizer_states) => NnueTrainStepRunner::with_optimizer_states(
            &ctx,
            initial_host_weights,
            optimizer_states.as_host(),
            batch_size,
            max_active,
        ),
        None => NnueTrainStepRunner::new(&ctx, initial_host_weights, batch_size, max_active),
    }
    .map_err(|e| e.to_string())?;
    runner.warmup(&ctx).map_err(|e| e.to_string())?;
    eprintln!("  cuda-cpp warmup = done (NNUE dense-backward kernels)");
    let upload_ctx = Context::new(device).map_err(|e| e.to_string())?;
    eprintln!(
        "  cuda-cpp {} upload pipeline = enabled (2 pinned slots; non-profiled steps)",
        feature_kind.source_label()
    );

    let loss_kind =
        if args.nnue_pytorch_wrm_loss { ScalarLossKind::NnuePytorchWrm } else { ScalarLossKind::SigmoidMse };
    // Bullet's scalar value loss applies sigmoid directly to the model output;
    // `--scale` is used only while preparing the teacher target.
    let output_inv_scale = 1.0_f32;
    let mut seen_steps = 0usize;
    let mut reported_loss_sum = 0.0_f64;
    let mut reported_loss_count = 0usize;
    let mut last_loss = 0.0_f32;
    let completed_step_offset = initial_state.completed_steps;
    let mut profile_upload_ms = 0.0_f64;
    let mut profile_forward_ms = 0.0_f64;
    let mut profile_loss_ms = 0.0_f64;
    let mut profile_backward_ms = 0.0_f64;
    let mut profile_update_ms = 0.0_f64;
    let mut profile_total_ms = 0.0_f64;
    let mut profile_count = 0usize;
    let started = std::time::Instant::now();
    let mut last_dataloader_pos = None;
    let dataloader_resume_pos = cuda_cpp_auto_resume_dataloader_pos(args);
    if let Some(pos) = dataloader_resume_pos {
        eprintln!("  dataloader resume = byte_offset {}, plies {}", pos.byte_offset, pos.plies);
    }
    let teacher_threads = cuda_cpp_effective_teacher_threads(args);
    let loader_threads = cuda_cpp_effective_loader_threads(args);
    let batch_queue_size = cuda_cpp_effective_batch_queue_size(args);
    eprintln!(
        "  cuda-cpp teacher CPU = prepare_threads={}, loader_threads={}, batch_queue_size={}",
        teacher_threads, loader_threads, batch_queue_size
    );
    if args.cuda_cpp_profile_teacher_prepare {
        eprintln!("  cuda-cpp teacher prepare profiling = enabled (serial prepared-batch consumer)");
    }

    let teacher_options = CudaCppNnueTeacherOptions {
        batch_size,
        batch_index: 0,
        dataloader_resume_pos,
        loader_threads,
        teacher_threads,
        queue_depth: batch_queue_size,
    };

    if schedule.production && args.lr_schedule == LrScheduleKind::Plateau {
        let mut current_resume_pos = dataloader_resume_pos;
        let mut completed_steps = completed_step_offset;
        let mut accepted_steps_total = 0usize;
        let mut attempted_steps_total = 0usize;
        let mut last_checkpoint_metrics = None;
        let mut previous_epoch_final_metrics = if resume_enabled(args, &args.output_dir()) {
            read_latest_nnue_test_metrics_in_top_level_log(&args.output_dir().join(SUMMARY_LEARN_LOG_NAME))
        } else {
            None
        };
        let mut checkpoint_chunk_idx = 0usize;
        while checkpoint_chunk_idx < schedule.chunks.len() {
            let epoch = schedule.chunks[checkpoint_chunk_idx].epoch;
            let display_max_epochs = args.max_epochs.map(|n| n + epoch.saturating_sub(1)).unwrap_or(epoch);
            print_epoch_banner(epoch, display_max_epochs);
            let mut plateau_state = PlateauLrState::new(
                args.lr,
                args.lr_min,
                args.lr_plateau_factor,
                args.lr_plateau_min_delta,
                args.lr_plateau_monitor,
            );
            let mut plateau_epoch_final_metrics = None;
            while checkpoint_chunk_idx < schedule.chunks.len() && schedule.chunks[checkpoint_chunk_idx].epoch == epoch {
                let chunk = schedule.chunks[checkpoint_chunk_idx].clone();
                let snapshot_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
                let snapshot_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                let snapshot_completed_steps = completed_steps;
                let chunk_resume_pos = current_resume_pos;
                let mut chunk_seen_steps = 0usize;
                let mut chunk_last_pos = None;
                let chunk_teacher_options = CudaCppNnueTeacherOptions {
                    batch_size,
                    batch_index: 0,
                    dataloader_resume_pos: chunk_resume_pos,
                    loader_threads,
                    teacher_threads,
                    queue_depth: batch_queue_size,
                };
                eprintln!(
                    "  cuda-cpp plateau: epoch={}, superbatch={}, lr {}",
                    chunk.epoch, chunk.superbatch, plateau_state.current_lr
                );
                for_each_cuda_cpp_nnue_teacher_batch(
                    args,
                    feature_kind,
                    chunk_teacher_options,
                    chunk.steps,
                    |teacher_batch| {
                        chunk_seen_steps += 1;
                        attempted_steps_total += 1;
                        chunk_last_pos = teacher_batch.dataloader_pos;
                        let optimizer_step = snapshot_completed_steps + chunk_seen_steps;
                        let fast = teacher_batch.batch;
                        let ranger = ranger_params(args, BULLETOU_DEFAULT_ADAMW_CLIP);
                        let params = RangerUpdateParams {
                            radam: RAdamUpdateParams {
                                step: optimizer_step as u64,
                                learning_rate: plateau_state.current_lr,
                                decay: ranger.decay,
                                beta1: ranger.beta1,
                                beta2: ranger.beta2,
                                epsilon: ranger.epsilon,
                                min_weight: ranger.min_weight,
                                max_weight: ranger.max_weight,
                                ..RAdamUpdateParams::default()
                            },
                            lookahead_alpha: ranger.alpha,
                            lookahead_period: ranger.k as u64,
                        };
                        let batch = NnueTrainStepHostBatch {
                            stm_indices: &fast.stm,
                            nstm_indices: &fast.nstm,
                            targets: &fast.targets,
                            entry_weights: &fast.weights,
                            batch_size: fast.layout.batch_size,
                            max_active: fast.layout.max_active,
                        };
                        let finalize_loss = chunk_seen_steps == chunk.steps;
                        runner
                            .step_no_readback_with_loss_finalize(
                                &ctx,
                                params,
                                loss_kind,
                                output_inv_scale,
                                batch,
                                finalize_loss,
                            )
                            .map_err(|e| e.to_string())?;
                        Ok::<(), String>(())
                    },
                )
                .map_err(|e| e.to_string())?;
                if chunk_seen_steps != chunk.steps {
                    return Err(format!(
                        "cuda-cpp plateau chunk ended early: expected {} steps, saw {chunk_seen_steps}",
                        chunk.steps
                    ));
                }
                let loss = runner.read_loss(&ctx).map_err(|e| e.to_string())?;
                last_loss = loss.mean;
                reported_loss_sum += f64::from(loss.mean);
                reported_loss_count += 1;
                let trained_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
                let trained_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                let test_metrics =
                    run_cuda_cpp_nnue_final_validation(args, feature_kind, cuda_shape, &trained_weights)?.ok_or_else(
                        || "--backend cuda-cpp plateau requires readable --test-teacher metrics".to_string(),
                    )?;
                let action = plateau_state.observe(test_metrics.into());
                let reject_update = plateau_action_rejects_update(action);
                let retry_same_chunk = plateau_action_retries_teacher(action);
                let accepted_dataloader_pos = cuda_cpp_direct_dataloader_pos_from_base(
                    args,
                    chunk_seen_steps,
                    batch_size,
                    chunk_last_pos,
                    chunk_resume_pos,
                )?;

                if reject_update {
                    let snapshot_host_weights = CudaNnueForwardHostWeights {
                        shape: cuda_shape,
                        l0w: &snapshot_weights.l0w,
                        l0b: &snapshot_weights.l0b,
                        l1w: &snapshot_weights.l1w,
                        l1b: &snapshot_weights.l1b,
                        l2w: &snapshot_weights.l2w,
                        l2b: &snapshot_weights.l2b,
                        outw: &snapshot_weights.outw,
                        outb: &snapshot_weights.outb,
                    };
                    runner = NnueTrainStepRunner::with_optimizer_states(
                        &ctx,
                        snapshot_host_weights,
                        cuda_cpp_halfkp_optimizer_readback_as_host(&snapshot_optimizer_states),
                        batch_size,
                        max_active,
                    )
                    .map_err(|e| e.to_string())?;
                    completed_steps = snapshot_completed_steps;
                } else {
                    completed_steps = snapshot_completed_steps + chunk_seen_steps;
                    accepted_steps_total += chunk_seen_steps;
                    current_resume_pos = Some(accepted_dataloader_pos);
                    let checkpoint_dir = write_cuda_cpp_nnue_numbered_checkpoint(
                        args,
                        feature_kind,
                        cuda_shape,
                        &trained_weights,
                        &trained_optimizer_states,
                        completed_steps,
                        CudaCppCheckpointLog {
                            epoch: chunk.epoch,
                            superbatch: chunk.superbatch,
                            curr_batch: schedule.batches_per_superbatch,
                            train_steps: chunk.steps,
                            train_loss: last_loss,
                            test_metrics: Some(test_metrics),
                            lr_start: plateau_state.current_lr,
                            lr_end: plateau_state.current_lr,
                            dataloader_pos: accepted_dataloader_pos,
                        },
                    )?;
                    eprintln!("  cuda-cpp numbered checkpoint = {}", checkpoint_dir.display());
                    last_checkpoint_metrics = Some(test_metrics);
                }

                let current_lr = plateau_state.current_lr;
                let monitor_label = args.lr_plateau_monitor.label();
                match action {
                    PlateauAction::First { metrics } => {
                        eprintln!(
                            "  plateau: initial validation metrics = {}; lr stays {current_lr}",
                            plateau_metrics_text(metrics)
                        );
                    }
                    PlateauAction::Improved { old_best, new_best } => {
                        eprintln!(
                            "  plateau: {monitor_label} improved (best {} -> {}); lr stays {}",
                            plateau_metrics_text(old_best),
                            plateau_metrics_text(new_best),
                            current_lr,
                        );
                    }
                    PlateauAction::Keep { metrics, best } => {
                        eprintln!(
                            "  plateau: {monitor_label} did not improve (current {}, best {}); lr stays {}",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                            current_lr,
                        );
                    }
                    PlateauAction::Reduced { old_lr, new_lr, metrics, best } => {
                        eprintln!(
                            "  plateau: {monitor_label} did not improve (current {}, best {}); lr {old_lr} -> {new_lr}",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                        );
                    }
                    PlateauAction::ScheduledFinal { old_lr, min_lr, metrics, best } => {
                        eprintln!(
                            "  plateau: {monitor_label} did not improve (current {}, best {}); \
                             next lr would fall below lr_min, so one final superbatch will run at lr_min {min_lr} \
                             (old lr {old_lr})",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                        );
                    }
                    PlateauAction::FinalImproved { old_best, new_best } => {
                        plateau_epoch_final_metrics = plateau_action_epoch_final_metrics(action);
                        mark_latest_checkpoint_epoch_done(&args.output_dir());
                        eprintln!(
                            "  plateau: final lr_min superbatch improved {monitor_label} (best {} -> {}); \
                             accepting it and ending this epoch.",
                            plateau_metrics_text(old_best),
                            plateau_metrics_text(new_best),
                        );
                        checkpoint_chunk_idx += 1;
                        break;
                    }
                    PlateauAction::FinalRejected { metrics, best } => {
                        plateau_epoch_final_metrics = plateau_action_epoch_final_metrics(action);
                        mark_latest_checkpoint_epoch_done(&args.output_dir());
                        eprintln!(
                            "  plateau: final lr_min superbatch did not improve {monitor_label} (current {}, best {}); \
                             discarding it and ending this epoch.",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                        );
                        break;
                    }
                }

                if retry_same_chunk {
                    eprintln!(
                        "  cuda-cpp plateau: restored model + optimiser, then rewinding teacher to retry superbatch {} at lowered lr {}",
                        chunk.superbatch, plateau_state.current_lr
                    );
                    continue;
                }
                checkpoint_chunk_idx += 1;
            }
            while checkpoint_chunk_idx < schedule.chunks.len() && schedule.chunks[checkpoint_chunk_idx].epoch == epoch {
                checkpoint_chunk_idx += 1;
            }

            if let Some(current_metrics) = plateau_epoch_final_metrics {
                if epoch_final_should_stop(
                    previous_epoch_final_metrics,
                    current_metrics,
                    PlateauMonitor::LossOrAccuracy,
                    0.0,
                ) {
                    let previous_metrics = previous_epoch_final_metrics.expect("checked by predicate");
                    eprintln!(
                        "  plateau: epoch-final validation metrics did not improve from previous epoch \
                         (loss {:.6} -> {:.6}, accuracy {:.6} -> {:.6}); stopping training.",
                        previous_metrics.loss,
                        current_metrics.loss,
                        previous_metrics.accuracy,
                        current_metrics.accuracy
                    );
                    break;
                }
                previous_epoch_final_metrics = Some(current_metrics);
            }
        }

        ctx.synchronize().map_err(|e| e.to_string())?;
        let elapsed = started.elapsed().as_secs_f64();
        let positions = accepted_steps_total.saturating_mul(batch_size);
        let positions_per_sec = if elapsed > 0.0 { positions as f64 / elapsed } else { 0.0 };
        let reported_avg_loss =
            if reported_loss_count > 0 { reported_loss_sum / reported_loss_count as f64 } else { 0.0 };
        eprintln!(
            "  cuda-cpp plateau train = ok: accepted_steps={accepted_steps_total}, attempted_steps={attempted_steps_total}, \
             positions={positions}, elapsed={elapsed:.3}s, throughput={positions_per_sec:.0} pos/s, \
             reported_avg_loss={reported_avg_loss:.8}, last_loss={last_loss:.8}"
        );
        let final_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
        let final_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
        let direct_output_dir = args.output_dir().join("cuda-cpp-direct");
        write_cuda_cpp_nnue_direct_outputs(
            &direct_output_dir,
            feature_kind,
            cuda_shape,
            &final_weights,
            &final_optimizer_states,
            completed_steps,
        )?;
        eprintln!("  cuda-cpp direct output = {} (nn.bin, full-state weights.bin)", direct_output_dir.display());
        if let Some(metrics) = last_checkpoint_metrics {
            eprintln!(
                "  cuda-cpp final validation summary: test_value_accuracy={:.7}, test_value_loss={:.8}",
                metrics.accuracy, metrics.loss
            );
        }
        return Ok(());
    }

    let mut checkpoint_chunk_idx = 0usize;
    let mut last_checkpoint_metrics = None;
    let mut deferred_direct_checkpoint = None;
    for_each_cuda_cpp_nnue_teacher_batch(args, feature_kind, teacher_options, train_steps, |teacher_batch| {
        seen_steps += 1;
        last_dataloader_pos = teacher_batch.dataloader_pos;
        let optimizer_step = completed_step_offset + seen_steps;
        let checkpoint_chunk = schedule.chunks.get(checkpoint_chunk_idx);
        let is_checkpoint_step = checkpoint_chunk.is_some_and(|chunk| chunk.cumulative_steps == seen_steps);
        let fast = teacher_batch.batch;
        let params = {
            let ranger = ranger_params(args, BULLETOU_DEFAULT_ADAMW_CLIP);
            let step_index = seen_steps.saturating_sub(1);
            let learning_rate = schedule.lr_for_step(args, step_index, batch_size);
            RangerUpdateParams {
                radam: RAdamUpdateParams {
                    step: optimizer_step as u64,
                    learning_rate,
                    decay: ranger.decay,
                    beta1: ranger.beta1,
                    beta2: ranger.beta2,
                    epsilon: ranger.epsilon,
                    min_weight: ranger.min_weight,
                    max_weight: ranger.max_weight,
                    ..RAdamUpdateParams::default()
                },
                lookahead_alpha: ranger.alpha,
                lookahead_period: ranger.k as u64,
            }
        };
        let batch = NnueTrainStepHostBatch {
            stm_indices: &fast.stm,
            nstm_indices: &fast.nstm,
            targets: &fast.targets,
            entry_weights: &fast.weights,
            batch_size: fast.layout.batch_size,
            max_active: fast.layout.max_active,
        };
        let should_report =
            is_checkpoint_step || cuda_cpp_should_read_loss(seen_steps, train_steps, args.cuda_cpp_loss_readback_interval);
        if seen_steps <= profile_steps {
            let profile = runner
                .step_profiled_no_readback(&ctx, params, loss_kind, output_inv_scale, batch)
                .map_err(|e| e.to_string())?;
            profile_upload_ms += f64::from(profile.upload_ms);
            profile_forward_ms += f64::from(profile.forward_ms);
            profile_loss_ms += f64::from(profile.loss_ms);
            profile_backward_ms += f64::from(profile.backward_ms);
            profile_update_ms += f64::from(profile.update_ms);
            profile_total_ms += f64::from(profile.total_ms);
            profile_count += 1;
            eprintln!(
                "  profile_cuda_cpp step={seen_steps:<6} upload={:.3}ms forward={:.3}ms loss={:.3}ms \
                 backward={:.3}ms update={:.3}ms total={:.3}ms",
                profile.upload_ms,
                profile.forward_ms,
                profile.loss_ms,
                profile.backward_ms,
                profile.update_ms,
                profile.total_ms
            );
        } else {
            runner
                .step_pipelined_no_readback_with_loss_finalize(
                    &ctx,
                    &upload_ctx,
                    params,
                    loss_kind,
                    output_inv_scale,
                    batch,
                    should_report,
                )
                .map_err(|e| e.to_string())?;
        }
        if should_report {
            let loss = runner.read_loss(&ctx).map_err(|e| e.to_string())?;
            last_loss = loss.mean;
            reported_loss_sum += f64::from(loss.mean);
            reported_loss_count += 1;
            if completed_step_offset > 0 {
                eprintln!(
                    "  cuda-cpp step {seen_steps:>6}/{train_steps:<6} optimizer_step={optimizer_step} \
                     loss_mean={:.8} source={}",
                    loss.mean, teacher_batch.source
                );
            } else {
                eprintln!(
                    "  cuda-cpp step {seen_steps:>6}/{train_steps:<6} loss_mean={:.8} source={}",
                    loss.mean, teacher_batch.source
                );
            }
        }
        if is_checkpoint_step {
            let chunk = schedule.chunks[checkpoint_chunk_idx].clone();
            let dataloader_pos = cuda_cpp_direct_dataloader_pos(args, seen_steps, batch_size, last_dataloader_pos)?;
            if schedule.production {
                if chunk.save_checkpoint {
                    let trained_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
                    let trained_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                    let test_metrics = run_cuda_cpp_nnue_final_validation(args, feature_kind, cuda_shape, &trained_weights)?;
                    let checkpoint_dir = write_cuda_cpp_nnue_numbered_checkpoint(
                        args,
                        feature_kind,
                        cuda_shape,
                        &trained_weights,
                        &trained_optimizer_states,
                        completed_step_offset + seen_steps,
                        CudaCppCheckpointLog {
                            epoch: chunk.epoch,
                            superbatch: chunk.superbatch,
                            curr_batch: schedule.batches_per_superbatch,
                            train_steps: chunk.steps,
                            train_loss: last_loss,
                            test_metrics,
                            lr_start: chunk.lr_start,
                            lr_end: chunk.lr_end,
                            dataloader_pos,
                        },
                    )?;
                    eprintln!("  cuda-cpp numbered checkpoint = {}", checkpoint_dir.display());
                    if let Some(metrics) = test_metrics {
                        eprintln!(
                            "  cuda-cpp validation summary: epoch={}, superbatch={}, test_value_accuracy={:.7}, test_value_loss={:.8}",
                            chunk.epoch, chunk.superbatch, metrics.accuracy, metrics.loss
                        );
                    }
                    last_checkpoint_metrics = test_metrics;
                } else {
                    eprintln!(
                        "  cuda-cpp checkpoint skipped at epoch={}, superbatch={} (--no-save-epoch-end)",
                        chunk.epoch, chunk.superbatch
                    );
                }
            } else {
                deferred_direct_checkpoint = Some((chunk, dataloader_pos));
            }
            checkpoint_chunk_idx += 1;
        }
        Ok::<(), String>(())
    })
    .map_err(|e| e.to_string())?;

    ctx.synchronize().map_err(|e| e.to_string())?;
    let elapsed = started.elapsed().as_secs_f64();
    let positions = seen_steps.saturating_mul(batch_size);
    let positions_per_sec = if elapsed > 0.0 { positions as f64 / elapsed } else { 0.0 };
    let reported_avg_loss = if reported_loss_count > 0 { reported_loss_sum / reported_loss_count as f64 } else { 0.0 };
    eprintln!(
        "  cuda-cpp direct train = ok: steps={seen_steps}, positions={positions}, elapsed={elapsed:.3}s, \
         throughput={positions_per_sec:.0} pos/s, reported_avg_loss={reported_avg_loss:.8}, last_loss={last_loss:.8}"
    );
    if profile_count > 0 {
        let denom = profile_count as f64;
        eprintln!(
            "  cuda-cpp profile avg: steps={profile_count}, upload={:.3}ms forward={:.3}ms loss={:.3}ms \
             backward={:.3}ms update={:.3}ms total={:.3}ms",
            profile_upload_ms / denom,
            profile_forward_ms / denom,
            profile_loss_ms / denom,
            profile_backward_ms / denom,
            profile_update_ms / denom,
            profile_total_ms / denom
        );
    }

    let completed_steps = completed_step_offset + seen_steps;
    if args.cuda_cpp_skip_final_output {
        if let Some((chunk, _)) = deferred_direct_checkpoint {
            if let Some(metrics) = {
                let trained_weights = if args.test_teacher.is_some() {
                    Some(runner.read_weights(&ctx).map_err(|e| e.to_string())?)
                } else {
                    None
                };
                match trained_weights.as_ref() {
                    Some(weights) => run_cuda_cpp_nnue_final_validation(args, feature_kind, cuda_shape, weights)?,
                    None => None,
                }
            } {
                eprintln!(
                    "  cuda-cpp validation summary: epoch={}, superbatch={}, test_value_accuracy={:.7}, test_value_loss={:.8}",
                    chunk.epoch, chunk.superbatch, metrics.accuracy, metrics.loss
                );
                last_checkpoint_metrics = Some(metrics);
            }
        }
        if checkpoint_chunk_idx != schedule.chunks.len() {
            return Err(format!(
                "cuda-cpp schedule ended after {checkpoint_chunk_idx} checkpoints, expected {}",
                schedule.chunks.len()
            ));
        }
        if let Some(metrics) = last_checkpoint_metrics {
            eprintln!(
                "  cuda-cpp final validation summary: test_value_accuracy={:.7}, test_value_loss={:.8}",
                metrics.accuracy, metrics.loss
            );
        }
        eprintln!("  cuda-cpp final output skipped (--cuda-cpp-skip-final-output)");
        return Ok(());
    }
    let trained_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
    let trained_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
    if let Some((chunk, dataloader_pos)) = deferred_direct_checkpoint {
        let test_metrics = run_cuda_cpp_nnue_final_validation(args, feature_kind, cuda_shape, &trained_weights)?;
        let checkpoint_dir = write_cuda_cpp_nnue_numbered_checkpoint(
            args,
            feature_kind,
            cuda_shape,
            &trained_weights,
            &trained_optimizer_states,
            completed_steps,
            CudaCppCheckpointLog {
                epoch: chunk.epoch,
                superbatch: chunk.superbatch,
                curr_batch: chunk.steps,
                train_steps: chunk.steps,
                train_loss: last_loss,
                test_metrics,
                lr_start: chunk.lr_start,
                lr_end: chunk.lr_end,
                dataloader_pos,
            },
        )?;
        eprintln!("  cuda-cpp numbered checkpoint = {}", checkpoint_dir.display());
        if let Some(metrics) = test_metrics {
            eprintln!(
                "  cuda-cpp validation summary: epoch={}, superbatch={}, test_value_accuracy={:.7}, test_value_loss={:.8}",
                chunk.epoch, chunk.superbatch, metrics.accuracy, metrics.loss
            );
        }
        last_checkpoint_metrics = test_metrics;
    }
    let direct_output_dir = args.output_dir().join("cuda-cpp-direct");
    write_cuda_cpp_nnue_direct_outputs(
        &direct_output_dir,
        feature_kind,
        cuda_shape,
        &trained_weights,
        &trained_optimizer_states,
        completed_steps,
    )?;
    eprintln!("  cuda-cpp direct output = {} (nn.bin, full-state weights.bin)", direct_output_dir.display());
    if checkpoint_chunk_idx != schedule.chunks.len() {
        return Err(format!(
            "cuda-cpp schedule ended after {checkpoint_chunk_idx} checkpoints, expected {}",
            schedule.chunks.len()
        ));
    }
    if let Some(metrics) = last_checkpoint_metrics {
        eprintln!(
            "  cuda-cpp final validation summary: test_value_accuracy={:.7}, test_value_loss={:.8}",
            metrics.accuracy, metrics.loss
        );
    }

    Ok(())
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_sfnn_halfka2_direct_steps(args: &Args) -> Result<(), String> {
    use bulletou_cuda_cpp::{
        Context, RAdamUpdateParams, RangerUpdateParams, ScalarLossKind, SfnnTrainStepHostBatch, SfnnTrainStepRunner,
    };
    use bulletou_lib::value::{SfnnTeacherBatchConfig, for_each_sfnn_halfka2_teacher_fast_batch};

    let schedule = cuda_cpp_run_schedule(args)?;
    let train_steps = schedule.total_steps;
    let batch_size = effective_batch_size(args);
    let (ft_size, l1_hidden, l2_size) = args.arch().dims();
    let layerstack = args.effective_layerstack().unwrap_or(LayerStackMode::Kingrank3by3);
    let num_stacks = layerstack.num_stacks();
    let device = args.cuda_cpp_device;

    eprintln!(
        "  backend = cuda-cpp Windows-native direct SFNN_HALFKA2 trainer ({train_steps} batch step{})",
        if train_steps == 1 { "" } else { "s" }
    );
    if schedule.production {
        eprintln!(
            "  cuda-cpp SFNN schedule = production: max_epochs={}, superbatches={}, save_rate={}, save_epoch_end={}, batches_per_superbatch={}, lr={}",
            args.max_epochs.unwrap_or(1).max(1),
            args.superbatches.unwrap_or(1),
            effective_save_rate(args),
            effective_save_epoch_end(args),
            schedule.batches_per_superbatch,
            args.lr_schedule.cli_name()
        );
    } else {
        eprintln!("  cuda-cpp SFNN schedule = direct train-steps smoke mode");
    }
    let name = bulletou_cuda_cpp::device_name(device).map_err(|e| e.to_string())?;
    eprintln!("  cuda-cpp device {device}: {name}");
    let auto_resume_state_bin = cuda_cpp_auto_resume_state_bin(args);
    let initial_state = build_sfnn_halfka2_initial_state_for_cuda_cpp(args)?;
    let initial_weights = &initial_state.weights;
    if let Some(path) = args.cuda_cpp_weights_bin.as_deref() {
        let state_kind = if initial_state.optimizer_states.is_some() {
            "weights + Ranger optimizer state"
        } else if initial_state.completed_steps > 0 {
            "weights + step counters"
        } else {
            "weights only"
        };
        eprintln!("  initial state = {} ({state_kind})", path.display());
    } else if let Some(path) = auto_resume_state_bin.as_deref() {
        eprintln!("  initial state = {} (auto-resume weights + Ranger optimizer state)", path.display());
    } else {
        eprintln!("  initial weights = deterministic nnue-pytorch-style scratch");
    }
    if initial_state.completed_steps > 0 {
        eprintln!("  initial completed optimizer steps = {}", initial_state.completed_steps);
    }
    eprintln!("  batch size = {batch_size}");
    eprintln!(
        "  arch = {} (factorized HalfKA2 input {}, ft={ft_size}, l1_hidden={l1_hidden}, l2={l2_size}, stacks={num_stacks})",
        args.arch().cli_name(),
        initial_weights.shape.input_size
    );
    eprintln!("  l1 factorized shared term = {}", if initial_weights.l1fw.is_some() { "enabled" } else { "disabled" });
    eprintln!("  loss = {}", if args.nnue_pytorch_wrm_loss { "nnue-pytorch-wrm" } else { "sigmoid-mse" });
    let profile_steps = args.cuda_cpp_profile_steps.min(train_steps);
    if profile_steps > 0 {
        eprintln!("  cuda-cpp SFNN profile steps = {profile_steps}");
    }
    eprintln!(
        "  cuda-cpp SFNN loss readback interval = {}",
        if args.cuda_cpp_loss_readback_interval == 0 {
            "final-only".to_string()
        } else {
            args.cuda_cpp_loss_readback_interval.to_string()
        }
    );
    eprintln!("  cuda-cpp SFNN upload pipeline = enabled (2 slots; non-profiled steps)");

    let cuda_shape = initial_weights.shape;
    let ctx = Context::new(device).map_err(|e| e.to_string())?;
    let initial_host_weights = initial_weights.as_host();
    let max_active = ShogiHalfKa2.max_active();
    let mut runner = match initial_state.optimizer_states.as_ref() {
        Some(optimizer_states) => SfnnTrainStepRunner::with_optimizer_states(
            &ctx,
            initial_host_weights,
            optimizer_states.as_host(),
            batch_size,
            max_active,
        ),
        None => SfnnTrainStepRunner::new(&ctx, initial_host_weights, batch_size, max_active),
    }
    .map_err(|e| e.to_string())?;
    let upload_ctx = Context::new(device).map_err(|e| e.to_string())?;

    let loss_kind =
        if args.nnue_pytorch_wrm_loss { ScalarLossKind::NnuePytorchWrm } else { ScalarLossKind::SigmoidMse };
    // Bullet's scalar value loss applies sigmoid directly to the model output;
    // `--scale` is used only while preparing the teacher target.
    let output_inv_scale = 1.0_f32;
    let mut seen_steps = 0usize;
    let mut reported_loss_sum = 0.0_f64;
    let mut reported_loss_count = 0usize;
    let mut last_loss = 0.0_f32;
    let mut profile_upload_ms = 0.0_f64;
    let mut profile_forward_ms = 0.0_f64;
    let mut profile_loss_ms = 0.0_f64;
    let mut profile_backward_ms = 0.0_f64;
    let mut profile_update_ms = 0.0_f64;
    let mut profile_total_ms = 0.0_f64;
    let mut profile_bwd_zero_ms = 0.0_f64;
    let mut profile_bwd_l3_ms = 0.0_f64;
    let mut profile_bwd_l2_ms = 0.0_f64;
    let mut profile_bwd_l2_input_ms = 0.0_f64;
    let mut profile_bwd_l1_ms = 0.0_f64;
    let mut profile_bwd_l0_ms = 0.0_f64;
    let mut profile_bwd_total_ms = 0.0_f64;
    let mut profile_count = 0usize;
    let completed_step_offset = initial_state.completed_steps;
    let started = std::time::Instant::now();
    let mut last_dataloader_pos = None;
    let dataloader_resume_pos = cuda_cpp_auto_resume_dataloader_pos(args);
    if let Some(pos) = dataloader_resume_pos {
        eprintln!("  dataloader resume = byte_offset {}, plies {}", pos.byte_offset, pos.plies);
    }
    let teacher_threads = cuda_cpp_effective_teacher_threads(args);
    let loader_threads = cuda_cpp_effective_loader_threads(args);
    let batch_queue_size = cuda_cpp_effective_batch_queue_size(args);
    eprintln!(
        "  cuda-cpp SFNN teacher CPU = prepare_threads={}, loader_threads={}, batch_queue_size={}",
        teacher_threads, loader_threads, batch_queue_size
    );
    if args.cuda_cpp_profile_teacher_prepare {
        eprintln!("  cuda-cpp SFNN teacher prepare profiling = enabled (serial prepared-batch consumer)");
    }

    let config = SfnnTeacherBatchConfig {
        teacher: &args.teacher,
        batch_size,
        batch_index: 0,
        dataloader_resume_pos,
        buffer_mb: args.buffer_mb,
        loader_threads,
        threads: teacher_threads,
        queue_depth: batch_queue_size,
        lambda: args.lambda,
        scale: args.scale as f32,
        nnue_pytorch_wrm_loss: args.nnue_pytorch_wrm_loss,
        score_drop_abs: (args.score_drop_abs > 0).then_some(args.score_drop_abs),
        profile_prepare: args.cuda_cpp_profile_teacher_prepare,
    };

    if schedule.production && args.lr_schedule == LrScheduleKind::Plateau {
        let mut current_resume_pos = dataloader_resume_pos;
        let mut completed_steps = completed_step_offset;
        let mut accepted_steps_total = 0usize;
        let mut attempted_steps_total = 0usize;
        let mut last_checkpoint_metrics = None;
        let mut previous_epoch_final_metrics = if resume_enabled(args, &args.output_dir()) {
            read_latest_nnue_test_metrics_in_top_level_log(&args.output_dir().join(SUMMARY_LEARN_LOG_NAME))
        } else {
            None
        };
        let mut checkpoint_chunk_idx = 0usize;
        while checkpoint_chunk_idx < schedule.chunks.len() {
            let epoch = schedule.chunks[checkpoint_chunk_idx].epoch;
            let display_max_epochs = args.max_epochs.map(|n| n + epoch.saturating_sub(1)).unwrap_or(epoch);
            print_epoch_banner(epoch, display_max_epochs);
            let mut plateau_state = PlateauLrState::new(
                args.lr,
                args.lr_min,
                args.lr_plateau_factor,
                args.lr_plateau_min_delta,
                args.lr_plateau_monitor,
            );
            let mut plateau_epoch_final_metrics = None;
            while checkpoint_chunk_idx < schedule.chunks.len() && schedule.chunks[checkpoint_chunk_idx].epoch == epoch {
                let chunk = schedule.chunks[checkpoint_chunk_idx].clone();
                let snapshot_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
                let snapshot_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                let snapshot_completed_steps = completed_steps;
                let chunk_resume_pos = current_resume_pos;
                let mut chunk_seen_steps = 0usize;
                let mut chunk_last_pos = None;
                let chunk_config = SfnnTeacherBatchConfig {
                    teacher: &args.teacher,
                    batch_size,
                    batch_index: 0,
                    dataloader_resume_pos: chunk_resume_pos,
                    buffer_mb: args.buffer_mb,
                    loader_threads,
                    threads: teacher_threads,
                    queue_depth: batch_queue_size,
                    lambda: args.lambda,
                    scale: args.scale as f32,
                    nnue_pytorch_wrm_loss: args.nnue_pytorch_wrm_loss,
                    score_drop_abs: (args.score_drop_abs > 0).then_some(args.score_drop_abs),
                    profile_prepare: args.cuda_cpp_profile_teacher_prepare,
                };
                eprintln!(
                    "  cuda-cpp SFNN plateau: epoch={}, superbatch={}, lr {}",
                    chunk.epoch, chunk.superbatch, plateau_state.current_lr
                );
                for_each_sfnn_halfka2_teacher_fast_batch(&chunk_config, chunk.steps, |teacher_batch| {
                    chunk_seen_steps += 1;
                    attempted_steps_total += 1;
                    chunk_last_pos = teacher_batch.dataloader_pos;
                    let optimizer_step = snapshot_completed_steps + chunk_seen_steps;
                    let fast = teacher_batch.batch;
                    let ranger = ranger_params(args, BULLETOU_DEFAULT_ADAMW_CLIP);
                    let params = RangerUpdateParams {
                        radam: RAdamUpdateParams {
                            step: optimizer_step as u64,
                            learning_rate: plateau_state.current_lr,
                            decay: ranger.decay,
                            beta1: ranger.beta1,
                            beta2: ranger.beta2,
                            epsilon: ranger.epsilon,
                            min_weight: ranger.min_weight,
                            max_weight: ranger.max_weight,
                            ..RAdamUpdateParams::default()
                        },
                        lookahead_alpha: ranger.alpha,
                        lookahead_period: ranger.k as u64,
                    };
                    let batch = SfnnTrainStepHostBatch {
                        stm_indices: &fast.stm,
                        nstm_indices: &fast.nstm,
                        buckets: &fast.buckets,
                        targets: &fast.targets,
                        entry_weights: &fast.weights,
                        batch_size: fast.layout.batch_size,
                        max_active: fast.layout.max_active,
                    };
                    let finalize_loss = chunk_seen_steps == chunk.steps;
                    runner
                        .step_no_readback_with_loss_finalize(
                            &ctx,
                            params,
                            loss_kind,
                            output_inv_scale,
                            batch,
                            finalize_loss,
                        )
                        .map_err(|e| e.to_string())?;
                    Ok::<(), String>(())
                })
                .map_err(|e| e.to_string())?;
                if chunk_seen_steps != chunk.steps {
                    return Err(format!(
                        "cuda-cpp SFNN plateau chunk ended early: expected {} steps, saw {chunk_seen_steps}",
                        chunk.steps
                    ));
                }
                let loss = runner.read_loss(&ctx).map_err(|e| e.to_string())?;
                last_loss = loss.mean;
                reported_loss_sum += f64::from(loss.mean);
                reported_loss_count += 1;
                let trained_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
                let test_metrics = run_cuda_cpp_sfnn_halfka2_final_validation(args, cuda_shape, &trained_weights)?
                    .ok_or_else(|| {
                        "--backend cuda-cpp SFNN plateau requires readable --test-teacher metrics".to_string()
                    })?;
                let trained_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                let action = plateau_state.observe(test_metrics.into());
                let reject_update = plateau_action_rejects_update(action);
                let retry_same_chunk = plateau_action_retries_teacher(action);
                let accepted_dataloader_pos = cuda_cpp_direct_dataloader_pos_from_base(
                    args,
                    chunk_seen_steps,
                    batch_size,
                    chunk_last_pos,
                    chunk_resume_pos,
                )?;

                if reject_update {
                    runner = SfnnTrainStepRunner::with_optimizer_states(
                        &ctx,
                        cuda_cpp_sfnn_weights_readback_as_host(cuda_shape, &snapshot_weights),
                        cuda_cpp_sfnn_optimizer_readback_as_host(&snapshot_optimizer_states),
                        batch_size,
                        max_active,
                    )
                    .map_err(|e| e.to_string())?;
                    completed_steps = snapshot_completed_steps;
                } else {
                    completed_steps = snapshot_completed_steps + chunk_seen_steps;
                    accepted_steps_total += chunk_seen_steps;
                    current_resume_pos = Some(accepted_dataloader_pos);
                    let checkpoint_dir = write_cuda_cpp_sfnn_numbered_checkpoint(
                        args,
                        cuda_shape,
                        &trained_weights,
                        &trained_optimizer_states,
                        completed_steps,
                        CudaCppCheckpointLog {
                            epoch: chunk.epoch,
                            superbatch: chunk.superbatch,
                            curr_batch: schedule.batches_per_superbatch,
                            train_steps: chunk.steps,
                            train_loss: last_loss,
                            test_metrics: Some(test_metrics),
                            lr_start: plateau_state.current_lr,
                            lr_end: plateau_state.current_lr,
                            dataloader_pos: accepted_dataloader_pos,
                        },
                    )?;
                    eprintln!("  cuda-cpp SFNN numbered checkpoint = {}", checkpoint_dir.display());
                    last_checkpoint_metrics = Some(test_metrics);
                }

                let current_lr = plateau_state.current_lr;
                let monitor_label = args.lr_plateau_monitor.label();
                match action {
                    PlateauAction::First { metrics } => {
                        eprintln!(
                            "  plateau: initial validation metrics = {}; lr stays {current_lr}",
                            plateau_metrics_text(metrics)
                        );
                    }
                    PlateauAction::Improved { old_best, new_best } => {
                        eprintln!(
                            "  plateau: {monitor_label} improved (best {} -> {}); lr stays {}",
                            plateau_metrics_text(old_best),
                            plateau_metrics_text(new_best),
                            current_lr,
                        );
                    }
                    PlateauAction::Keep { metrics, best } => {
                        eprintln!(
                            "  plateau: {monitor_label} did not improve (current {}, best {}); lr stays {}",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                            current_lr,
                        );
                    }
                    PlateauAction::Reduced { old_lr, new_lr, metrics, best } => {
                        eprintln!(
                            "  plateau: {monitor_label} did not improve (current {}, best {}); lr {old_lr} -> {new_lr}",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                        );
                    }
                    PlateauAction::ScheduledFinal { old_lr, min_lr, metrics, best } => {
                        eprintln!(
                            "  plateau: {monitor_label} did not improve (current {}, best {}); \
                             next lr would fall below lr_min, so one final superbatch will run at lr_min {min_lr} \
                             (old lr {old_lr})",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                        );
                    }
                    PlateauAction::FinalImproved { old_best, new_best } => {
                        plateau_epoch_final_metrics = plateau_action_epoch_final_metrics(action);
                        mark_latest_checkpoint_epoch_done(&args.output_dir());
                        eprintln!(
                            "  plateau: final lr_min superbatch improved {monitor_label} (best {} -> {}); \
                             accepting it and ending this epoch.",
                            plateau_metrics_text(old_best),
                            plateau_metrics_text(new_best),
                        );
                        checkpoint_chunk_idx += 1;
                        break;
                    }
                    PlateauAction::FinalRejected { metrics, best } => {
                        plateau_epoch_final_metrics = plateau_action_epoch_final_metrics(action);
                        mark_latest_checkpoint_epoch_done(&args.output_dir());
                        eprintln!(
                            "  plateau: final lr_min superbatch did not improve {monitor_label} (current {}, best {}); \
                             discarding it and ending this epoch.",
                            plateau_metrics_text(metrics),
                            plateau_metrics_text(best),
                        );
                        break;
                    }
                }

                if retry_same_chunk {
                    eprintln!(
                        "  cuda-cpp SFNN plateau: restored model + optimiser, then rewinding teacher to retry superbatch {} at lowered lr {}",
                        chunk.superbatch, plateau_state.current_lr
                    );
                    continue;
                }
                checkpoint_chunk_idx += 1;
            }
            while checkpoint_chunk_idx < schedule.chunks.len() && schedule.chunks[checkpoint_chunk_idx].epoch == epoch {
                checkpoint_chunk_idx += 1;
            }

            if let Some(current_metrics) = plateau_epoch_final_metrics {
                if epoch_final_should_stop(
                    previous_epoch_final_metrics,
                    current_metrics,
                    PlateauMonitor::LossOrAccuracy,
                    0.0,
                ) {
                    let previous_metrics = previous_epoch_final_metrics.expect("checked by predicate");
                    eprintln!(
                        "  plateau: epoch-final validation metrics did not improve from previous epoch \
                         (loss {:.6} -> {:.6}, accuracy {:.6} -> {:.6}); stopping training.",
                        previous_metrics.loss,
                        current_metrics.loss,
                        previous_metrics.accuracy,
                        current_metrics.accuracy
                    );
                    break;
                }
                previous_epoch_final_metrics = Some(current_metrics);
            }
        }

        ctx.synchronize().map_err(|e| e.to_string())?;
        let elapsed = started.elapsed().as_secs_f64();
        let positions = accepted_steps_total.saturating_mul(batch_size);
        let positions_per_sec = if elapsed > 0.0 { positions as f64 / elapsed } else { 0.0 };
        let reported_avg_loss =
            if reported_loss_count > 0 { reported_loss_sum / reported_loss_count as f64 } else { 0.0 };
        eprintln!(
            "  cuda-cpp SFNN plateau train = ok: accepted_steps={accepted_steps_total}, attempted_steps={attempted_steps_total}, \
             positions={positions}, elapsed={elapsed:.3}s, throughput={positions_per_sec:.0} pos/s, \
             reported_avg_loss={reported_avg_loss:.8}, last_loss={last_loss:.8}"
        );
        let final_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
        let final_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
        let direct_output_dir = args.output_dir().join("cuda-cpp-direct");
        write_cuda_cpp_sfnn_halfka2_direct_outputs(
            &direct_output_dir,
            cuda_shape,
            &final_weights,
            &final_optimizer_states,
            completed_steps,
        )?;
        eprintln!("  cuda-cpp SFNN direct output = {} (nn.bin, full-state weights.bin)", direct_output_dir.display());
        if let Some(metrics) = last_checkpoint_metrics {
            eprintln!(
                "  cuda-cpp SFNN final validation summary: test_value_accuracy={:.7}, test_value_loss={:.8}",
                metrics.accuracy, metrics.loss
            );
        }
        return Ok(());
    }

    let mut checkpoint_chunk_idx = 0usize;
    let mut last_checkpoint_metrics = None;
    let mut deferred_direct_checkpoint = None;
    for_each_sfnn_halfka2_teacher_fast_batch(&config, train_steps, |teacher_batch| {
        seen_steps += 1;
        last_dataloader_pos = teacher_batch.dataloader_pos;
        let optimizer_step = completed_step_offset + seen_steps;
        let checkpoint_chunk = schedule.chunks.get(checkpoint_chunk_idx);
        let is_checkpoint_step = checkpoint_chunk.is_some_and(|chunk| chunk.cumulative_steps == seen_steps);
        let fast = teacher_batch.batch;
        let params = {
            let ranger = ranger_params(args, BULLETOU_DEFAULT_ADAMW_CLIP);
            let step_index = seen_steps.saturating_sub(1);
            let learning_rate = schedule.lr_for_step(args, step_index, batch_size);
            RangerUpdateParams {
                radam: RAdamUpdateParams {
                    step: optimizer_step as u64,
                    learning_rate,
                    decay: ranger.decay,
                    beta1: ranger.beta1,
                    beta2: ranger.beta2,
                    epsilon: ranger.epsilon,
                    min_weight: ranger.min_weight,
                    max_weight: ranger.max_weight,
                    ..RAdamUpdateParams::default()
                },
                lookahead_alpha: ranger.alpha,
                lookahead_period: ranger.k as u64,
            }
        };
        let batch = SfnnTrainStepHostBatch {
            stm_indices: &fast.stm,
            nstm_indices: &fast.nstm,
            buckets: &fast.buckets,
            targets: &fast.targets,
            entry_weights: &fast.weights,
            batch_size: fast.layout.batch_size,
            max_active: fast.layout.max_active,
        };
        let should_report =
            is_checkpoint_step || cuda_cpp_should_read_loss(seen_steps, train_steps, args.cuda_cpp_loss_readback_interval);
        if seen_steps <= profile_steps {
            let profile = runner
                .step_profiled_no_readback(&ctx, params, loss_kind, output_inv_scale, batch)
                .map_err(|e| e.to_string())?;
            profile_upload_ms += f64::from(profile.upload_ms);
            profile_forward_ms += f64::from(profile.forward_ms);
            profile_loss_ms += f64::from(profile.loss_ms);
            profile_backward_ms += f64::from(profile.backward_ms);
            profile_update_ms += f64::from(profile.update_ms);
            profile_total_ms += f64::from(profile.total_ms);
            profile_bwd_zero_ms += f64::from(profile.backward_stages.zero_ms);
            profile_bwd_l3_ms += f64::from(profile.backward_stages.l3_ms);
            profile_bwd_l2_ms += f64::from(profile.backward_stages.l2_ms);
            profile_bwd_l2_input_ms += f64::from(profile.backward_stages.l2_input_ms);
            profile_bwd_l1_ms += f64::from(profile.backward_stages.l1_ms);
            profile_bwd_l0_ms += f64::from(profile.backward_stages.l0_ms);
            profile_bwd_total_ms += f64::from(profile.backward_stages.total_ms);
            profile_count += 1;
            eprintln!(
                "  profile_cuda_cpp_sfnn step={seen_steps:<6} upload={:.3}ms forward={:.3}ms loss={:.3}ms \
                 backward={:.3}ms update={:.3}ms total={:.3}ms \
                 bwd[zero={:.3} l3={:.3} l2={:.3} l2in={:.3} l1={:.3} l0={:.3} total={:.3}]",
                profile.upload_ms,
                profile.forward_ms,
                profile.loss_ms,
                profile.backward_ms,
                profile.update_ms,
                profile.total_ms,
                profile.backward_stages.zero_ms,
                profile.backward_stages.l3_ms,
                profile.backward_stages.l2_ms,
                profile.backward_stages.l2_input_ms,
                profile.backward_stages.l1_ms,
                profile.backward_stages.l0_ms,
                profile.backward_stages.total_ms
            );
        } else {
            runner
                .step_pipelined_no_readback_with_loss_finalize(
                    &ctx,
                    &upload_ctx,
                    params,
                    loss_kind,
                    output_inv_scale,
                    batch,
                    should_report,
                )
                .map_err(|e| e.to_string())?;
        }
        if should_report {
            let loss = runner.read_loss(&ctx).map_err(|e| e.to_string())?;
            last_loss = loss.mean;
            reported_loss_sum += f64::from(loss.mean);
            reported_loss_count += 1;
            if completed_step_offset > 0 {
                eprintln!(
                    "  cuda-cpp SFNN step {seen_steps:>6}/{train_steps:<6} optimizer_step={optimizer_step} \
                     loss_mean={:.8} source={}",
                    loss.mean, teacher_batch.source
                );
            } else {
                eprintln!(
                    "  cuda-cpp SFNN step {seen_steps:>6}/{train_steps:<6} loss_mean={:.8} source={}",
                    loss.mean, teacher_batch.source
                );
            }
        }
        if is_checkpoint_step {
            let chunk = schedule.chunks[checkpoint_chunk_idx].clone();
            let dataloader_pos = cuda_cpp_direct_dataloader_pos(args, seen_steps, batch_size, last_dataloader_pos)?;
            if schedule.production {
                if chunk.save_checkpoint {
                    let trained_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
                    let test_metrics = run_cuda_cpp_sfnn_halfka2_final_validation(args, cuda_shape, &trained_weights)?;
                    let trained_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
                    let checkpoint_dir = write_cuda_cpp_sfnn_numbered_checkpoint(
                        args,
                        cuda_shape,
                        &trained_weights,
                        &trained_optimizer_states,
                        completed_step_offset + seen_steps,
                        CudaCppCheckpointLog {
                            epoch: chunk.epoch,
                            superbatch: chunk.superbatch,
                            curr_batch: schedule.batches_per_superbatch,
                            train_steps: chunk.steps,
                            train_loss: last_loss,
                            test_metrics,
                            lr_start: chunk.lr_start,
                            lr_end: chunk.lr_end,
                            dataloader_pos,
                        },
                    )?;
                    eprintln!("  cuda-cpp SFNN numbered checkpoint = {}", checkpoint_dir.display());
                    if let Some(metrics) = test_metrics {
                        eprintln!(
                            "  cuda-cpp SFNN validation summary: epoch={}, superbatch={}, test_value_accuracy={:.7}, test_value_loss={:.8}",
                            chunk.epoch, chunk.superbatch, metrics.accuracy, metrics.loss
                        );
                    }
                    last_checkpoint_metrics = test_metrics;
                } else {
                    eprintln!(
                        "  cuda-cpp SFNN checkpoint skipped at epoch={}, superbatch={} (--no-save-epoch-end)",
                        chunk.epoch, chunk.superbatch
                    );
                }
            } else {
                deferred_direct_checkpoint = Some((chunk, dataloader_pos));
            }
            checkpoint_chunk_idx += 1;
        }
        Ok::<(), String>(())
    })
    .map_err(|e| e.to_string())?;

    ctx.synchronize().map_err(|e| e.to_string())?;
    let elapsed = started.elapsed().as_secs_f64();
    let positions = seen_steps.saturating_mul(batch_size);
    let positions_per_sec = if elapsed > 0.0 { positions as f64 / elapsed } else { 0.0 };
    let reported_avg_loss = if reported_loss_count > 0 { reported_loss_sum / reported_loss_count as f64 } else { 0.0 };
    eprintln!(
        "  cuda-cpp SFNN direct train = ok: steps={seen_steps}, positions={positions}, elapsed={elapsed:.3}s, \
         throughput={positions_per_sec:.0} pos/s, reported_avg_loss={reported_avg_loss:.8}, last_loss={last_loss:.8}"
    );
    if profile_count > 0 {
        let denom = profile_count as f64;
        eprintln!(
            "  cuda-cpp SFNN profile avg: steps={profile_count}, upload={:.3}ms forward={:.3}ms loss={:.3}ms \
             backward={:.3}ms update={:.3}ms total={:.3}ms",
            profile_upload_ms / denom,
            profile_forward_ms / denom,
            profile_loss_ms / denom,
            profile_backward_ms / denom,
            profile_update_ms / denom,
            profile_total_ms / denom
        );
        eprintln!(
            "  cuda-cpp SFNN backward profile avg: zero={:.3}ms l3={:.3}ms l2={:.3}ms l2_input={:.3}ms \
             l1={:.3}ms l0={:.3}ms total={:.3}ms",
            profile_bwd_zero_ms / denom,
            profile_bwd_l3_ms / denom,
            profile_bwd_l2_ms / denom,
            profile_bwd_l2_input_ms / denom,
            profile_bwd_l1_ms / denom,
            profile_bwd_l0_ms / denom,
            profile_bwd_total_ms / denom
        );
    }
    let completed_steps = completed_step_offset + seen_steps;
    if args.cuda_cpp_skip_final_output {
        if let Some((chunk, _)) = deferred_direct_checkpoint {
            if let Some(metrics) = {
                let trained_weights = if args.test_teacher.is_some() {
                    Some(runner.read_weights(&ctx).map_err(|e| e.to_string())?)
                } else {
                    None
                };
                match trained_weights.as_ref() {
                    Some(weights) => run_cuda_cpp_sfnn_halfka2_final_validation(args, cuda_shape, weights)?,
                    None => None,
                }
            } {
                eprintln!(
                    "  cuda-cpp SFNN validation summary: epoch={}, superbatch={}, test_value_accuracy={:.7}, test_value_loss={:.8}",
                    chunk.epoch, chunk.superbatch, metrics.accuracy, metrics.loss
                );
                last_checkpoint_metrics = Some(metrics);
            }
        }
        if checkpoint_chunk_idx != schedule.chunks.len() {
            return Err(format!(
                "cuda-cpp SFNN schedule ended after {checkpoint_chunk_idx} checkpoints, expected {}",
                schedule.chunks.len()
            ));
        }
        if let Some(metrics) = last_checkpoint_metrics {
            eprintln!(
                "  cuda-cpp SFNN final validation summary: test_value_accuracy={:.7}, test_value_loss={:.8}",
                metrics.accuracy, metrics.loss
            );
        }
        eprintln!("  cuda-cpp SFNN final output skipped (--cuda-cpp-skip-final-output)");
        return Ok(());
    }
    let trained_weights = runner.read_weights(&ctx).map_err(|e| e.to_string())?;
    let trained_optimizer_states = runner.read_optimizer_states(&ctx).map_err(|e| e.to_string())?;
    if let Some((chunk, dataloader_pos)) = deferred_direct_checkpoint {
        let test_metrics = run_cuda_cpp_sfnn_halfka2_final_validation(args, cuda_shape, &trained_weights)?;
        let checkpoint_dir = write_cuda_cpp_sfnn_numbered_checkpoint(
            args,
            cuda_shape,
            &trained_weights,
            &trained_optimizer_states,
            completed_steps,
            CudaCppCheckpointLog {
                epoch: chunk.epoch,
                superbatch: chunk.superbatch,
                curr_batch: chunk.steps,
                train_steps: chunk.steps,
                train_loss: last_loss,
                test_metrics,
                lr_start: chunk.lr_start,
                lr_end: chunk.lr_end,
                dataloader_pos,
            },
        )?;
        eprintln!("  cuda-cpp SFNN numbered checkpoint = {}", checkpoint_dir.display());
        if let Some(metrics) = test_metrics {
            eprintln!(
                "  cuda-cpp SFNN validation summary: epoch={}, superbatch={}, test_value_accuracy={:.7}, test_value_loss={:.8}",
                chunk.epoch, chunk.superbatch, metrics.accuracy, metrics.loss
            );
        }
        last_checkpoint_metrics = test_metrics;
    }
    let direct_output_dir = args.output_dir().join("cuda-cpp-direct");
    write_cuda_cpp_sfnn_halfka2_direct_outputs(
        &direct_output_dir,
        cuda_shape,
        &trained_weights,
        &trained_optimizer_states,
        completed_steps,
    )?;
    eprintln!("  cuda-cpp SFNN direct output = {} (nn.bin, full-state weights.bin)", direct_output_dir.display());
    if checkpoint_chunk_idx != schedule.chunks.len() {
        return Err(format!(
            "cuda-cpp SFNN schedule ended after {checkpoint_chunk_idx} checkpoints, expected {}",
            schedule.chunks.len()
        ));
    }
    if let Some(metrics) = last_checkpoint_metrics {
        eprintln!(
            "  cuda-cpp SFNN final validation summary: test_value_accuracy={:.7}, test_value_loss={:.8}",
            metrics.accuracy, metrics.loss
        );
    }

    Ok(())
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_ranger_readback_as_host(
    state: &bulletou_cuda_cpp::RangerParamStateReadback,
) -> bulletou_cuda_cpp::RangerParamHostState<'_> {
    bulletou_cuda_cpp::RangerParamHostState {
        momentum: &state.momentum,
        velocity: &state.velocity,
        slow_params: &state.slow_params,
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_halfkp_optimizer_readback_as_host(
    states: &bulletou_cuda_cpp::NnueRangerOptimizerStatesReadback,
) -> bulletou_cuda_cpp::NnueRangerOptimizerHostStates<'_> {
    bulletou_cuda_cpp::NnueRangerOptimizerHostStates {
        l0w: cuda_cpp_ranger_readback_as_host(&states.l0w),
        l0b: cuda_cpp_ranger_readback_as_host(&states.l0b),
        l1w: cuda_cpp_ranger_readback_as_host(&states.l1w),
        l1b: cuda_cpp_ranger_readback_as_host(&states.l1b),
        l2w: cuda_cpp_ranger_readback_as_host(&states.l2w),
        l2b: cuda_cpp_ranger_readback_as_host(&states.l2b),
        outw: cuda_cpp_ranger_readback_as_host(&states.outw),
        outb: cuda_cpp_ranger_readback_as_host(&states.outb),
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_sfnn_weights_readback_as_host(
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    weights: &bulletou_cuda_cpp::SfnnTrainWeightsReadback,
) -> bulletou_cuda_cpp::SfnnForwardHostWeights<'_> {
    bulletou_cuda_cpp::SfnnForwardHostWeights {
        shape,
        l0w: &weights.l0w,
        l0b: &weights.l0b,
        l1w: &weights.l1w,
        l1b: &weights.l1b,
        l1fw: weights.l1fw.as_deref(),
        l1fb: weights.l1fb.as_deref(),
        l2w: &weights.l2w,
        l2b: &weights.l2b,
        l3w: &weights.l3w,
        l3b: &weights.l3b,
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_sfnn_optimizer_readback_as_host(
    states: &bulletou_cuda_cpp::SfnnRangerOptimizerStatesReadback,
) -> bulletou_cuda_cpp::SfnnRangerOptimizerHostStates<'_> {
    bulletou_cuda_cpp::SfnnRangerOptimizerHostStates {
        l0w: cuda_cpp_ranger_readback_as_host(&states.l0w),
        l0b: cuda_cpp_ranger_readback_as_host(&states.l0b),
        l1w: cuda_cpp_ranger_readback_as_host(&states.l1w),
        l1b: cuda_cpp_ranger_readback_as_host(&states.l1b),
        l1fw: states.l1fw.as_ref().map(cuda_cpp_ranger_readback_as_host),
        l1fb: states.l1fb.as_ref().map(cuda_cpp_ranger_readback_as_host),
        l2w: cuda_cpp_ranger_readback_as_host(&states.l2w),
        l2b: cuda_cpp_ranger_readback_as_host(&states.l2b),
        l3w: cuda_cpp_ranger_readback_as_host(&states.l3w),
        l3b: cuda_cpp_ranger_readback_as_host(&states.l3b),
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_nnue_final_validation(
    args: &Args,
    feature_kind: CudaCppNnueFeatureKind,
    shape: bulletou_cuda_cpp::NnueForwardShape,
    weights: &bulletou_cuda_cpp::NnueTrainWeightsReadback,
) -> Result<Option<TestMetrics>, String> {
    let Some(cache) = TestPositionsCache::try_load(args) else {
        return Ok(None);
    };
    if cache.positions.is_empty() {
        eprintln!(
            "  WARN: --test-teacher yielded no positions; cuda-cpp {} final validation skipped",
            feature_kind.source_label()
        );
        return Ok(None);
    }

    let validation_weights = cuda_cpp_nnue_weights_for_cpu_validation(feature_kind, shape, weights)?;
    let batch_size = args.test_batch_size.max(1);
    let mut outputs = Vec::with_capacity(cache.positions.len());
    let started = std::time::Instant::now();
    for positions in cache.positions.chunks(batch_size) {
        let batch = build_nnue_validation_fast_batch(feature_kind, positions)?;
        let mut chunk_outputs = validation_weights.forward_batch(&batch).map_err(|e| e.to_string())?;
        outputs.append(&mut chunk_outputs);
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "  cuda-cpp {} final validation forward = ok: positions={}, batch_size={}, elapsed={elapsed:.3}s",
        feature_kind.source_label(),
        outputs.len(),
        batch_size
    );

    Ok(Some(run_one_test_pass(&cache, args, outputs)))
}

#[cfg(feature = "cuda-cpp-backend")]
fn run_cuda_cpp_sfnn_halfka2_final_validation(
    args: &Args,
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    weights: &bulletou_cuda_cpp::SfnnTrainWeightsReadback,
) -> Result<Option<TestMetrics>, String> {
    let Some(cache) = TestPositionsCache::try_load(args) else {
        return Ok(None);
    };
    if cache.positions.is_empty() {
        eprintln!("  WARN: --test-teacher yielded no positions; cuda-cpp SFNN final validation skipped");
        return Ok(None);
    }

    let validation_weights = cuda_cpp_sfnn_weights_for_cpu_validation(shape, weights)?;
    let batch_size = args.test_batch_size.max(1);
    let mut outputs = Vec::with_capacity(cache.positions.len());
    let started = std::time::Instant::now();
    for positions in cache.positions.chunks(batch_size) {
        let batch = build_sfnn_halfka2_validation_fast_batch(positions)?;
        let mut chunk_outputs = validation_weights.forward_batch(&batch).map_err(|e| e.to_string())?;
        outputs.append(&mut chunk_outputs);
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "  cuda-cpp SFNN final validation forward = ok: positions={}, batch_size={}, elapsed={elapsed:.3}s",
        outputs.len(),
        batch_size
    );

    Ok(Some(run_one_test_pass(&cache, args, outputs)))
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_auto_resume_state_bin(args: &Args) -> Option<std::path::PathBuf> {
    if args.cuda_cpp_weights_bin.is_some() {
        return None;
    }
    let output_dir = args.output_dir();
    find_latest_state_bin(args, &output_dir)
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_auto_resume_dataloader_pos(args: &Args) -> Option<bulletou_lib::value::TeacherDataloaderPos> {
    if args.cuda_cpp_weights_bin.is_some() {
        return None;
    }
    let output_dir = args.output_dir();
    if !resume_enabled(args, &output_dir) {
        return None;
    }
    if cuda_cpp_resume_teacher_changed(args, &output_dir) {
        return None;
    }
    read_latest_dataloader_pos(&output_dir)
        .map(|(byte_offset, plies)| bulletou_lib::value::TeacherDataloaderPos { byte_offset, plies })
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_resume_teacher_changed(args: &Args, output_dir: &std::path::Path) -> bool {
    let Some(prev_teacher) = read_latest_saved_teacher(output_dir) else { return false };
    prev_teacher.trim() != resolve_teacher_for_log(&args.teacher).trim()
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_direct_dataloader_pos(
    args: &Args,
    seen_steps: usize,
    batch_size: usize,
    last_pos: Option<bulletou_lib::value::TeacherDataloaderPos>,
) -> Result<bulletou_lib::value::TeacherDataloaderPos, String> {
    cuda_cpp_direct_dataloader_pos_from_base(
        args,
        seen_steps,
        batch_size,
        last_pos,
        cuda_cpp_auto_resume_dataloader_pos(args),
    )
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_direct_dataloader_pos_from_base(
    args: &Args,
    seen_steps: usize,
    batch_size: usize,
    last_pos: Option<bulletou_lib::value::TeacherDataloaderPos>,
    base_resume_pos: Option<bulletou_lib::value::TeacherDataloaderPos>,
) -> Result<bulletou_lib::value::TeacherDataloaderPos, String> {
    if let Some(pos) = last_pos {
        return Ok(pos);
    }

    let paths = expand_teacher(&args.teacher)?;
    let path_refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
    let format = infer_data_format(&path_refs)?;
    let Some(record_size) = (match format {
        DataFormat::Hcpe => Some(bulletou_lib::value::loader::hcpe::HCPE_RECORD_SIZE),
        DataFormat::Psv => Some(std::mem::size_of::<bulletou_lib::shogi::PackedSfenValue>()),
        DataFormat::Hcpe3 | DataFormat::Pack => None,
    }) else {
        eprintln!(
            "  WARN: cuda-cpp direct checkpoint could not infer dataloader_pos for variable-length {format:?}; writing 0,0"
        );
        return Ok(bulletou_lib::value::TeacherDataloaderPos { byte_offset: 0, plies: 0 });
    };

    let mut total_bytes = 0u64;
    for path in &paths {
        let len = std::fs::metadata(path).map_err(|err| format!("failed to stat teacher {path}: {err}"))?.len();
        total_bytes =
            total_bytes.checked_add(len).ok_or_else(|| format!("teacher byte size overflow while adding {path}"))?;
    }
    if total_bytes == 0 {
        return Ok(bulletou_lib::value::TeacherDataloaderPos { byte_offset: 0, plies: 0 });
    }
    let base_byte_offset = base_resume_pos.map(|pos| pos.byte_offset).unwrap_or(0);
    let consumed_records = seen_steps
        .checked_mul(batch_size)
        .ok_or_else(|| format!("cuda-cpp dataloader_pos overflow: steps={seen_steps} batch_size={batch_size}"))?;
    let consumed_bytes = (consumed_records as u64)
        .checked_mul(record_size as u64)
        .ok_or_else(|| format!("cuda-cpp dataloader_pos byte overflow: records={consumed_records}"))?;
    Ok(bulletou_lib::value::TeacherDataloaderPos {
        byte_offset: base_byte_offset.wrapping_add(consumed_bytes) % total_bytes,
        plies: 0,
    })
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Clone, Copy, Debug)]
struct CudaCppCheckpointLog {
    epoch: usize,
    superbatch: usize,
    curr_batch: usize,
    train_steps: usize,
    train_loss: f32,
    test_metrics: Option<TestMetrics>,
    lr_start: f32,
    lr_end: f32,
    dataloader_pos: bulletou_lib::value::TeacherDataloaderPos,
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_nnue_numbered_checkpoint(
    args: &Args,
    feature_kind: CudaCppNnueFeatureKind,
    shape: bulletou_cuda_cpp::NnueForwardShape,
    weights: &bulletou_cuda_cpp::NnueTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::NnueRangerOptimizerStatesReadback,
    completed_steps: usize,
    log: CudaCppCheckpointLog,
) -> Result<std::path::PathBuf, String> {
    let output_dir = args.output_dir();
    std::fs::create_dir_all(&output_dir).map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;
    let idx = count_existing_numbered_dirs(&output_dir) + 1;
    let dir = output_dir.join(format!("{idx:04}"));
    if dir.exists() {
        return Err(format!("refusing to overwrite existing numbered checkpoint {}", dir.display()));
    }
    std::fs::create_dir_all(&dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    write_cuda_cpp_nnue_nn_bin(&dir.join("nn.bin"), feature_kind, shape, weights)?;
    write_cuda_cpp_halfkp_weights_bin(&dir.join("state.bin"), weights, optimizer_states, completed_steps)?;
    write_cuda_cpp_direct_checkpoint_metadata(&output_dir, idx, &dir, args, log)?;
    Ok(dir)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_sfnn_numbered_checkpoint(
    args: &Args,
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    weights: &bulletou_cuda_cpp::SfnnTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::SfnnRangerOptimizerStatesReadback,
    completed_steps: usize,
    log: CudaCppCheckpointLog,
) -> Result<std::path::PathBuf, String> {
    let output_dir = args.output_dir();
    std::fs::create_dir_all(&output_dir).map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;
    let idx = count_existing_numbered_dirs(&output_dir) + 1;
    let dir = output_dir.join(format!("{idx:04}"));
    if dir.exists() {
        return Err(format!("refusing to overwrite existing numbered checkpoint {}", dir.display()));
    }
    std::fs::create_dir_all(&dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    write_cuda_cpp_sfnn_halfka2_nn_bin(&dir.join("nn.bin"), shape, weights)?;
    write_cuda_cpp_sfnn_weights_bin(&dir.join("state.bin"), weights, optimizer_states, completed_steps)?;
    write_cuda_cpp_direct_checkpoint_metadata(&output_dir, idx, &dir, args, log)?;
    Ok(dir)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_direct_checkpoint_metadata(
    output_dir: &std::path::Path,
    idx: usize,
    dir: &std::path::Path,
    args: &Args,
    log: CudaCppCheckpointLog,
) -> Result<(), String> {
    std::fs::write(dir.join("teacher.txt"), format!("{}\n", args.teacher))
        .map_err(|err| format!("failed to write {}: {err}", dir.join("teacher.txt").display()))?;
    std::fs::write(
        dir.join("dataloader_pos.txt"),
        format!("{},{}\n", log.dataloader_pos.byte_offset, log.dataloader_pos.plies),
    )
    .map_err(|err| format!("failed to write {}: {err}", dir.join("dataloader_pos.txt").display()))?;

    let prior_positions =
        read_prior_positions(&output_dir.join(SUMMARY_LEARN_LOG_NAME)).get("nnue").copied().unwrap_or(0);
    let mut learn = String::new();
    learn.push_str(LEARN_LOG_HEADER);
    learn.push('\n');
    learn.push_str(&cuda_cpp_direct_learn_log_row(args, log, prior_positions));
    std::fs::write(dir.join("learn.log"), learn)
        .map_err(|err| format!("failed to write {}: {err}", dir.join("learn.log").display()))?;
    append_to_top_level_log(output_dir, idx)
        .map_err(|err| format!("failed to update {}: {err}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display()))
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_direct_learn_log_row(args: &Args, log: CudaCppCheckpointLog, prior_positions: usize) -> String {
    let eval_field = if args.eval_type().uses_arch() {
        format!("{}-{}", args.eval_type().cli_name(), args.arch().cli_name())
    } else {
        args.eval_type().cli_name().to_string()
    };
    let (test_accuracy, test_loss) = match log.test_metrics {
        Some(metrics) => (format!("{:.6}", metrics.accuracy), format!("{:.6}", metrics.loss)),
        None => ("-".to_string(), "-".to_string()),
    };
    let positions = prior_positions.saturating_add(log.train_steps.saturating_mul(effective_batch_size(args)));
    format!(
        "{eval},{epoch},{superbatch},{batch},{test_accuracy},{test_loss},{train_loss:.6},{lr_start:.6},{lr_end:.6},{lambda:.6},{positions},{teacher}\n",
        eval = eval_field,
        epoch = log.epoch,
        superbatch = log.superbatch,
        batch = log.curr_batch,
        train_loss = log.train_loss,
        lr_start = log.lr_start,
        lr_end = log.lr_end,
        lambda = args.lambda,
        teacher = csv_escape(&resolve_teacher_for_log(&args.teacher)),
    )
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_nnue_weights_for_cpu_validation(
    feature_kind: CudaCppNnueFeatureKind,
    shape: bulletou_cuda_cpp::NnueForwardShape,
    weights: &bulletou_cuda_cpp::NnueTrainWeightsReadback,
) -> Result<bulletou_lib::value::NnueForwardOwnedWeights, String> {
    use bulletou_lib::value::{
        NnueForwardOwnedWeights as CpuNnueForwardOwnedWeights, NnueForwardShape as CpuNnueForwardShape,
    };

    let base_input_size = feature_kind.base_input_size();
    let virtual_rows = feature_kind.virtual_rows();
    let factorized_input_size = base_input_size + virtual_rows;
    let l0w = if shape.input_size == base_input_size {
        weights.l0w.clone()
    } else if virtual_rows > 0 && shape.input_size == factorized_input_size {
        fold_halfkp_piece_factorized_l0w(&weights.l0w, base_input_size, virtual_rows, shape.l1)?
    } else {
        return Err(format!(
            "cannot validate {} cuda-cpp weights with input_size={}, expected {} or factorized {}",
            feature_kind.source_label(),
            shape.input_size,
            base_input_size,
            factorized_input_size
        ));
    };
    let cpu_shape = CpuNnueForwardShape { input_size: base_input_size, l1: shape.l1, l2: shape.l2, l3: shape.l3 };
    let validation_weights = CpuNnueForwardOwnedWeights {
        shape: cpu_shape,
        l0w,
        l0b: weights.l0b.clone(),
        l1w: weights.l1w.clone(),
        l1b: weights.l1b.clone(),
        l2w: weights.l2w.clone(),
        l2b: weights.l2b.clone(),
        outw: weights.outw.clone(),
        outb: weights.outb.clone(),
    };
    validation_weights.validate().map_err(|e| e.to_string())?;
    Ok(validation_weights)
}

#[cfg(all(feature = "cuda-cpp-backend", test))]
fn cuda_cpp_halfkp_weights_for_cpu_validation(
    shape: bulletou_cuda_cpp::NnueForwardShape,
    weights: &bulletou_cuda_cpp::NnueTrainWeightsReadback,
) -> Result<bulletou_lib::value::NnueForwardOwnedWeights, String> {
    cuda_cpp_nnue_weights_for_cpu_validation(CudaCppNnueFeatureKind::Halfkp, shape, weights)
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_sfnn_weights_for_cpu_validation(
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    weights: &bulletou_cuda_cpp::SfnnTrainWeightsReadback,
) -> Result<bulletou_lib::value::SfnnForwardOwnedWeights, String> {
    use bulletou_lib::value::{
        SfnnForwardOwnedWeights as CpuSfnnForwardOwnedWeights, SfnnForwardShape as CpuSfnnForwardShape,
    };

    let cpu_shape = CpuSfnnForwardShape {
        input_size: shape.input_size,
        ft_size: shape.ft_size,
        l1_hidden: shape.l1_hidden,
        l2_size: shape.l2_size,
        num_stacks: shape.num_stacks,
    };
    let mut l1w = weights.l1w.clone();
    let mut l1b = weights.l1b.clone();
    fold_cuda_cpp_sfnn_l1f_into_stacked_l1(
        shape,
        &mut l1w,
        &mut l1b,
        weights.l1fw.as_deref(),
        weights.l1fb.as_deref(),
    )?;

    let cpu_weights = CpuSfnnForwardOwnedWeights {
        shape: cpu_shape,
        l0w: weights.l0w.clone(),
        l0b: weights.l0b.clone(),
        l1w,
        l1b,
        l2w: weights.l2w.clone(),
        l2b: weights.l2b.clone(),
        l3w: weights.l3w.clone(),
        l3b: weights.l3b.clone(),
    };
    cpu_weights.validate().map_err(|e| e.to_string())?;
    Ok(cpu_weights)
}

#[cfg(feature = "cuda-cpp-backend")]
fn fold_cuda_cpp_sfnn_l1f_into_stacked_l1(
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    l1w: &mut [f32],
    l1b: &mut [f32],
    l1fw: Option<&[f32]>,
    l1fb: Option<&[f32]>,
) -> Result<(), String> {
    let l1_out = shape.l1_out();
    let expected_l1w = shape.num_stacks * l1_out * shape.ft_size;
    let expected_l1b = shape.num_stacks * l1_out;
    if l1w.len() != expected_l1w {
        return Err(format!("SFNN l1w length mismatch: got {}, expected {expected_l1w}", l1w.len()));
    }
    if l1b.len() != expected_l1b {
        return Err(format!("SFNN l1b length mismatch: got {}, expected {expected_l1b}", l1b.len()));
    }
    match (l1fw, l1fb) {
        (Some(shared_w), Some(shared_b)) => {
            let expected_l1fw = shape.ft_size * l1_out;
            if shared_w.len() != expected_l1fw {
                return Err(format!("SFNN l1fw length mismatch: got {}, expected {expected_l1fw}", shared_w.len()));
            }
            if shared_b.len() != l1_out {
                return Err(format!("SFNN l1fb length mismatch: got {}, expected {l1_out}", shared_b.len()));
            }
            let stack_stride = l1_out * shape.ft_size;
            for stack in 0..shape.num_stacks {
                let bias_base = stack * l1_out;
                for out_col in 0..l1_out {
                    l1b[bias_base + out_col] += shared_b[out_col];
                }

                let weight_base = stack * stack_stride;
                for out_col in 0..l1_out {
                    let row_base = weight_base + out_col * shape.ft_size;
                    for in_col in 0..shape.ft_size {
                        l1w[row_base + in_col] += shared_w[in_col * l1_out + out_col];
                    }
                }
            }
        }
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => {
            return Err("cuda-cpp SFNN weights have partial l1f state".to_string());
        }
    }
    Ok(())
}

#[cfg(feature = "cuda-cpp-backend")]
fn build_nnue_validation_fast_batch(
    feature_kind: CudaCppNnueFeatureKind,
    positions: &[bulletou_lib::shogi::PackedSfenValue],
) -> Result<bulletou_lib::value::FastBatchHost, String> {
    use bulletou_lib::game::inputs::{fill_halfkp_feature_indices, fill_kp_feature_indices};
    use bulletou_lib::value::{FastBatchHost, FastBatchLayout};

    let batch_size = positions.len();
    if batch_size == 0 {
        return Err(format!("{} validation batch must not be empty", feature_kind.source_label()));
    }
    let max_active = feature_kind.max_active();
    let input_size = feature_kind.base_input_size();
    let sparse_len = batch_size
        .checked_mul(max_active)
        .ok_or_else(|| format!("{} validation sparse batch length overflow", feature_kind.source_label()))?;
    let mut stm = vec![-1_i32; sparse_len];
    let mut nstm = vec![-1_i32; sparse_len];
    for (sample, pos) in positions.iter().enumerate() {
        let sparse_offset = sample * max_active;
        let (stm_count, nstm_count) = match feature_kind {
            CudaCppNnueFeatureKind::Halfkp => fill_halfkp_feature_indices(
                pos,
                &mut stm[sparse_offset..sparse_offset + max_active],
                &mut nstm[sparse_offset..sparse_offset + max_active],
            ),
            CudaCppNnueFeatureKind::Kp => fill_kp_feature_indices(
                pos,
                &mut stm[sparse_offset..sparse_offset + max_active],
                &mut nstm[sparse_offset..sparse_offset + max_active],
            ),
        };
        if stm_count > max_active {
            return Err(format!(
                "{} STM active feature count {stm_count} exceeded max_active {max_active}",
                feature_kind.source_label()
            ));
        }
        if nstm_count > max_active {
            return Err(format!(
                "{} NSTM active feature count {nstm_count} exceeded max_active {max_active}",
                feature_kind.source_label()
            ));
        }
        for &idx in &stm[sparse_offset..sparse_offset + stm_count] {
            if idx < 0 || idx as usize >= input_size {
                return Err(format!(
                    "{} STM feature index {idx} exceeded input size {input_size}",
                    feature_kind.source_label()
                ));
            }
        }
        for &idx in &nstm[sparse_offset..sparse_offset + nstm_count] {
            if idx < 0 || idx as usize >= input_size {
                return Err(format!(
                    "{} NSTM feature index {idx} exceeded input size {input_size}",
                    feature_kind.source_label()
                ));
            }
        }
    }

    let batch = FastBatchHost {
        layout: FastBatchLayout { batch_size, max_active, output_size: 1, hand_count_dim: 0 },
        stm,
        nstm,
        buckets: vec![0_i32; batch_size],
        targets: vec![0.0; batch_size],
        weights: vec![1.0; batch_size],
        hand_count: None,
    };
    batch.validate()?;
    Ok(batch)
}

#[cfg(feature = "cuda-cpp-backend")]
fn build_sfnn_halfka2_validation_fast_batch(
    positions: &[bulletou_lib::shogi::PackedSfenValue],
) -> Result<bulletou_lib::value::FastBatchHost, String> {
    use bulletou_lib::value::{FastBatchHost, FastBatchLayout};

    let batch_size = positions.len();
    if batch_size == 0 {
        return Err("SFNN validation batch must not be empty".to_string());
    }
    let max_active = ShogiHalfKa2.max_active();
    let input_size = ShogiHalfKa2.num_inputs();
    let sparse_len =
        batch_size.checked_mul(max_active).ok_or_else(|| "SFNN validation sparse batch length overflow".to_string())?;
    let mut stm = vec![-1_i32; sparse_len];
    let mut nstm = vec![-1_i32; sparse_len];
    let mut buckets = vec![0_i32; batch_size];
    for (sample, pos) in positions.iter().enumerate() {
        let mut j_stm = 0usize;
        let mut j_nstm = 0usize;
        let sparse_offset = sample * max_active;
        let mut error = None;
        ShogiHalfKa2.map_features_split(pos, |our_opt, opp_opt| {
            if let Some(our) = our_opt {
                if our >= input_size {
                    error = Some(format!("STM feature index {our} exceeded input size {input_size}"));
                    return;
                }
                if j_stm >= max_active {
                    error = Some(format!("STM active feature count exceeded max_active {max_active}"));
                    return;
                }
                stm[sparse_offset + j_stm] = our as i32;
                j_stm += 1;
            }
            if let Some(opp) = opp_opt {
                if opp >= input_size {
                    error = Some(format!("NSTM feature index {opp} exceeded input size {input_size}"));
                    return;
                }
                if j_nstm >= max_active {
                    error = Some(format!("NSTM active feature count exceeded max_active {max_active}"));
                    return;
                }
                nstm[sparse_offset + j_nstm] = opp as i32;
                j_nstm += 1;
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        buckets[sample] = i32::from(ShogiLayerStackBucket9::KingRank9.bucket(pos));
    }

    let batch = FastBatchHost {
        layout: FastBatchLayout { batch_size, max_active, output_size: 1, hand_count_dim: 0 },
        stm,
        nstm,
        buckets,
        targets: vec![0.0; batch_size],
        weights: vec![1.0; batch_size],
        hand_count: None,
    };
    batch.validate()?;
    Ok(batch)
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppSfnnInitialWeights {
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    l0w: Vec<f32>,
    l0b: Vec<f32>,
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l1fw: Option<Vec<f32>>,
    l1fb: Option<Vec<f32>>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    l3w: Vec<f32>,
    l3b: Vec<f32>,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppSfnnInitialWeights {
    fn validate(&self) -> Result<(), String> {
        let weights = bulletou_cuda_cpp::SfnnForwardHostWeights {
            shape: self.shape,
            l0w: &self.l0w,
            l0b: &self.l0b,
            l1w: &self.l1w,
            l1b: &self.l1b,
            l1fw: self.l1fw.as_deref(),
            l1fb: self.l1fb.as_deref(),
            l2w: &self.l2w,
            l2b: &self.l2b,
            l3w: &self.l3w,
            l3b: &self.l3b,
        };
        weights.validate().map_err(|e| e.to_string())
    }

    fn as_host(&self) -> bulletou_cuda_cpp::SfnnForwardHostWeights<'_> {
        bulletou_cuda_cpp::SfnnForwardHostWeights {
            shape: self.shape,
            l0w: &self.l0w,
            l0b: &self.l0b,
            l1w: &self.l1w,
            l1b: &self.l1b,
            l1fw: self.l1fw.as_deref(),
            l1fb: self.l1fb.as_deref(),
            l2w: &self.l2w,
            l2b: &self.l2b,
            l3w: &self.l3w,
            l3b: &self.l3b,
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppSfnnInitialState {
    weights: CudaCppSfnnInitialWeights,
    optimizer_states: Option<CudaCppSfnnOptimizerState>,
    completed_steps: usize,
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppSfnnOptimizerState {
    l0w: CudaCppRangerGroupState,
    l0b: CudaCppRangerGroupState,
    l1w: CudaCppRangerGroupState,
    l1b: CudaCppRangerGroupState,
    l1fw: Option<CudaCppRangerGroupState>,
    l1fb: Option<CudaCppRangerGroupState>,
    l2w: CudaCppRangerGroupState,
    l2b: CudaCppRangerGroupState,
    l3w: CudaCppRangerGroupState,
    l3b: CudaCppRangerGroupState,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppSfnnOptimizerState {
    fn as_host(&self) -> bulletou_cuda_cpp::SfnnRangerOptimizerHostStates<'_> {
        bulletou_cuda_cpp::SfnnRangerOptimizerHostStates {
            l0w: self.l0w.as_host(),
            l0b: self.l0b.as_host(),
            l1w: self.l1w.as_host(),
            l1b: self.l1b.as_host(),
            l1fw: self.l1fw.as_ref().map(CudaCppRangerGroupState::as_host),
            l1fb: self.l1fb.as_ref().map(CudaCppRangerGroupState::as_host),
            l2w: self.l2w.as_host(),
            l2b: self.l2b.as_host(),
            l3w: self.l3w.as_host(),
            l3b: self.l3b.as_host(),
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn build_sfnn_halfka2_initial_state_for_cuda_cpp(args: &Args) -> Result<CudaCppSfnnInitialState, String> {
    if let Some(path) = args.cuda_cpp_weights_bin.as_deref() {
        return load_cuda_cpp_sfnn_halfka2_initial_state(path, args);
    }
    if let Some(path) = cuda_cpp_auto_resume_state_bin(args) {
        return load_cuda_cpp_sfnn_halfka2_initial_state(&path, args);
    }

    Ok(CudaCppSfnnInitialState {
        weights: build_sfnn_halfka2_initial_weights_for_cuda_cpp(args)?,
        optimizer_states: None,
        completed_steps: 0,
    })
}

#[cfg(feature = "cuda-cpp-backend")]
fn build_sfnn_halfka2_initial_weights_for_cuda_cpp(args: &Args) -> Result<CudaCppSfnnInitialWeights, String> {
    let (ft_size, l1_hidden, l2_size) = args.arch().dims();
    let layerstack = args.effective_layerstack().unwrap_or(LayerStackMode::Kingrank3by3);
    let num_stacks = layerstack.num_stacks();
    let base_input_size = ShogiHalfKa2.num_inputs();
    let input_size = base_input_size + bulletou_lib::game::inputs::PIECE_INPUTS;
    let init_scale = args.nnue_pytorch_init_scale;
    let shape = bulletou_cuda_cpp::SfnnForwardShape { input_size, ft_size, l1_hidden, l2_size, num_stacks };
    let l1_out = shape.l1_out();
    let l2_in = shape.l2_in();

    let mut l0w =
        cuda_cpp_tatara_uniform_fan_in_init(base_input_size * ft_size, 0x5f11_e001, base_input_size, init_scale);
    l0w.resize(input_size * ft_size, 0.0);
    let l0b = cuda_cpp_tatara_uniform_fan_in_init(ft_size, 0x5f11_e002, base_input_size, init_scale);

    let l1_bound = init_scale * (1.0 / ft_size.max(1) as f32).sqrt();
    let l2_bound = init_scale * (1.0 / l2_in.max(1) as f32).sqrt();
    let l3_bound = init_scale * (1.0 / l2_size.max(1) as f32).sqrt();
    let l1w = cuda_cpp_tatara_stacked_row_major_bucket0_init(ft_size, l1_out, num_stacks, 0x5f11_e003, l1_bound);
    let l1b = cuda_cpp_tatara_stacked_bias_bucket0_init(l1_out, num_stacks, 0x5f11_e004, l1_bound);
    let l2w = cuda_cpp_tatara_stacked_row_major_bucket0_init(l2_in, l2_size, num_stacks, 0x5f11_e005, l2_bound);
    let l2b = cuda_cpp_tatara_stacked_bias_bucket0_init(l2_size, num_stacks, 0x5f11_e006, l2_bound);
    let l3w = cuda_cpp_tatara_stacked_row_major_bucket0_init(l2_size, 1, num_stacks, 0x5f11_e007, l3_bound);
    let l3b = vec![0.0; num_stacks];
    let (l1fw, l1fb) = if args.sfnn_factorized_l1 {
        (Some(vec![0.0; ft_size * l1_out]), Some(vec![0.0; l1_out]))
    } else {
        (None, None)
    };

    let weights = CudaCppSfnnInitialWeights { shape, l0w, l0b, l1w, l1b, l1fw, l1fb, l2w, l2b, l3w, l3b };
    weights.validate()?;
    Ok(weights)
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_sfnn_halfka2_initial_state(path: &Path, args: &Args) -> Result<CudaCppSfnnInitialState, String> {
    let bytes = std::fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let records =
        parse_model_weights_bin(&bytes).map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let component_weights = bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "weights");
    let weights = if component_weights.is_empty() { &records } else { &component_weights };

    let (ft_size, l1_hidden, l2_size) = args.arch().dims();
    let layerstack = args.effective_layerstack().unwrap_or(LayerStackMode::Kingrank3by3);
    let shape = bulletou_cuda_cpp::SfnnForwardShape {
        input_size: ShogiHalfKa2.num_inputs() + bulletou_lib::game::inputs::PIECE_INPUTS,
        ft_size,
        l1_hidden,
        l2_size,
        num_stacks: layerstack.num_stacks(),
    };
    let weights = load_cuda_cpp_sfnn_weights_from_records(shape, weights).map_err(|err| {
        format!(
            "failed to load cuda-cpp SFNN HalfKA2 weights from {} for arch {}: {err}",
            path.display(),
            args.arch().cli_name()
        )
    })?;
    let optimizer_states = load_cuda_cpp_sfnn_optimizer_state(&records, &weights)?;
    let completed_steps = load_cuda_cpp_sfnn_completed_steps(&records, &weights)?;

    Ok(CudaCppSfnnInitialState { weights, optimizer_states, completed_steps })
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_sfnn_weights_from_records(
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    records: &BTreeMap<String, Vec<f32>>,
) -> Result<CudaCppSfnnInitialWeights, String> {
    let base_input_size = ShogiHalfKa2.num_inputs();
    let factorized_input_size = base_input_size + bulletou_lib::game::inputs::PIECE_INPUTS;
    if shape.input_size != factorized_input_size {
        return Err(format!(
            "internal SFNN HalfKA2 shape uses input_size={}, expected factorized {}",
            shape.input_size, factorized_input_size
        ));
    }

    let mut l0w = load_cuda_cpp_weight_record(records, "l0w")?;
    let expected_factorized_l0w = shape.input_size * shape.ft_size;
    let expected_base_l0w = base_input_size * shape.ft_size;
    if l0w.len() == expected_base_l0w {
        l0w.resize(expected_factorized_l0w, 0.0);
    } else if l0w.len() != expected_factorized_l0w {
        return Err(format!(
            "weight l0w has length {}, expected {} or base-only {}",
            l0w.len(),
            expected_factorized_l0w,
            expected_base_l0w
        ));
    }

    let l1fw = match (records.get("l1fw"), records.get("l1fb")) {
        (Some(_), Some(_)) => Some(load_cuda_cpp_weight_record(records, "l1fw")?),
        (None, None) => None,
        (Some(_), None) => return Err("SFNN weights contain l1fw without l1fb".to_string()),
        (None, Some(_)) => return Err("SFNN weights contain l1fb without l1fw".to_string()),
    };
    let l1fb = if l1fw.is_some() { Some(load_cuda_cpp_weight_record(records, "l1fb")?) } else { None };

    let weights = CudaCppSfnnInitialWeights {
        shape,
        l0w,
        l0b: load_cuda_cpp_weight_record(records, "l0b")?,
        l1w: load_cuda_cpp_weight_record(records, "l1w")?,
        l1b: load_cuda_cpp_weight_record(records, "l1b")?,
        l1fw,
        l1fb,
        l2w: load_cuda_cpp_weight_record(records, "l2w")?,
        l2b: load_cuda_cpp_weight_record(records, "l2b")?,
        l3w: load_cuda_cpp_weight_record(records, "l3w")?,
        l3b: load_cuda_cpp_weight_record(records, "l3b")?,
    };
    weights.validate()?;
    Ok(weights)
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_weight_record(records: &BTreeMap<String, Vec<f32>>, id: &'static str) -> Result<Vec<f32>, String> {
    records.get(id).cloned().ok_or_else(|| format!("cuda-cpp state missing weight `{id}`"))
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_sfnn_optimizer_state(
    records: &BTreeMap<String, Vec<f32>>,
    weights: &CudaCppSfnnInitialWeights,
) -> Result<Option<CudaCppSfnnOptimizerState>, String> {
    let momentum = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, "nnue", "momentum");
    let velocity = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, "nnue", "velocity");
    let slow = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, "nnue", "slow");
    let has_any = !momentum.is_empty() || !velocity.is_empty() || !slow.is_empty();
    if !has_any {
        return Ok(None);
    }
    if momentum.is_empty() || velocity.is_empty() || slow.is_empty() {
        return Err(
            "cuda-cpp SFNN optimizer state is partial: expected nnue/{momentum,velocity,slow}/* records".to_string()
        );
    }

    let (l1fw, l1fb) = match (&weights.l1fw, &weights.l1fb) {
        (Some(l1fw), Some(l1fb)) => (
            Some(load_cuda_cpp_ranger_group_state_for("SFNN", "l1fw", l1fw.len(), &momentum, &velocity, &slow)?),
            Some(load_cuda_cpp_ranger_group_state_for("SFNN", "l1fb", l1fb.len(), &momentum, &velocity, &slow)?),
        ),
        (None, None) => (None, None),
        _ => return Err("cuda-cpp SFNN l1f weights are partial".to_string()),
    };

    Ok(Some(CudaCppSfnnOptimizerState {
        l0w: load_cuda_cpp_ranger_group_state_for("SFNN", "l0w", weights.l0w.len(), &momentum, &velocity, &slow)?,
        l0b: load_cuda_cpp_ranger_group_state_for("SFNN", "l0b", weights.l0b.len(), &momentum, &velocity, &slow)?,
        l1w: load_cuda_cpp_ranger_group_state_for("SFNN", "l1w", weights.l1w.len(), &momentum, &velocity, &slow)?,
        l1b: load_cuda_cpp_ranger_group_state_for("SFNN", "l1b", weights.l1b.len(), &momentum, &velocity, &slow)?,
        l1fw,
        l1fb,
        l2w: load_cuda_cpp_ranger_group_state_for("SFNN", "l2w", weights.l2w.len(), &momentum, &velocity, &slow)?,
        l2b: load_cuda_cpp_ranger_group_state_for("SFNN", "l2b", weights.l2b.len(), &momentum, &velocity, &slow)?,
        l3w: load_cuda_cpp_ranger_group_state_for("SFNN", "l3w", weights.l3w.len(), &momentum, &velocity, &slow)?,
        l3b: load_cuda_cpp_ranger_group_state_for("SFNN", "l3b", weights.l3b.len(), &momentum, &velocity, &slow)?,
    }))
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_sfnn_completed_steps(
    records: &BTreeMap<String, Vec<f32>>,
    weights: &CudaCppSfnnInitialWeights,
) -> Result<usize, String> {
    let mut ids = vec!["l0w", "l0b", "l1w", "l1b", "l2w", "l2b", "l3w", "l3b"];
    if weights.l1fw.is_some() {
        ids.push("l1fw");
        ids.push("l1fb");
    }
    load_cuda_cpp_completed_steps_for("SFNN", records, &ids)
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_tatara_stacked_row_major_bucket0_init(
    input_dim: usize,
    output_dim: usize,
    num_stacks: usize,
    seed: u64,
    half_width: f32,
) -> Vec<f32> {
    let bucket0 = cuda_cpp_tatara_uniform_abs_init(input_dim * output_dim, seed, half_width);
    let mut weights = vec![0.0; num_stacks * output_dim * input_dim];
    for stack in 0..num_stacks {
        let stack_base = stack * output_dim * input_dim;
        for out_col in 0..output_dim {
            let src_base = out_col * input_dim;
            let dst_base = stack_base + out_col * input_dim;
            weights[dst_base..dst_base + input_dim].copy_from_slice(&bucket0[src_base..src_base + input_dim]);
        }
    }
    weights
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_tatara_stacked_bias_bucket0_init(
    output_dim: usize,
    num_stacks: usize,
    seed: u64,
    half_width: f32,
) -> Vec<f32> {
    let bucket0 = cuda_cpp_tatara_uniform_abs_init(output_dim, seed, half_width);
    let mut bias = vec![0.0; num_stacks * output_dim];
    for stack in 0..num_stacks {
        let base = stack * output_dim;
        bias[base..base + output_dim].copy_from_slice(&bucket0);
    }
    bias
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppHalfkpInitialState {
    weights: bulletou_lib::value::NnueForwardOwnedWeights,
    optimizer_states: Option<CudaCppHalfkpOptimizerState>,
    completed_steps: usize,
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppHalfkpOptimizerState {
    l0w: CudaCppRangerGroupState,
    l0b: CudaCppRangerGroupState,
    l1w: CudaCppRangerGroupState,
    l1b: CudaCppRangerGroupState,
    l2w: CudaCppRangerGroupState,
    l2b: CudaCppRangerGroupState,
    outw: CudaCppRangerGroupState,
    outb: CudaCppRangerGroupState,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppHalfkpOptimizerState {
    fn as_host(&self) -> bulletou_cuda_cpp::NnueRangerOptimizerHostStates<'_> {
        bulletou_cuda_cpp::NnueRangerOptimizerHostStates {
            l0w: self.l0w.as_host(),
            l0b: self.l0b.as_host(),
            l1w: self.l1w.as_host(),
            l1b: self.l1b.as_host(),
            l2w: self.l2w.as_host(),
            l2b: self.l2b.as_host(),
            outw: self.outw.as_host(),
            outb: self.outb.as_host(),
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Debug, Clone, PartialEq)]
struct CudaCppRangerGroupState {
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    slow_params: Vec<f32>,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppRangerGroupState {
    fn as_host(&self) -> bulletou_cuda_cpp::RangerParamHostState<'_> {
        bulletou_cuda_cpp::RangerParamHostState {
            momentum: &self.momentum,
            velocity: &self.velocity,
            slow_params: &self.slow_params,
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn build_nnue_initial_state_for_cuda_cpp(
    args: &Args,
    feature_kind: CudaCppNnueFeatureKind,
) -> Result<CudaCppHalfkpInitialState, String> {
    if let Some(path) = args.cuda_cpp_weights_bin.as_deref() {
        return load_cuda_cpp_nnue_initial_state(path, args, feature_kind);
    }
    if let Some(path) = cuda_cpp_auto_resume_state_bin(args) {
        return load_cuda_cpp_nnue_initial_state(&path, args, feature_kind);
    }

    Ok(CudaCppHalfkpInitialState {
        weights: build_nnue_initial_weights_for_cuda_cpp(args, feature_kind)?,
        optimizer_states: None,
        completed_steps: 0,
    })
}

#[cfg(feature = "cuda-cpp-backend")]
fn build_nnue_initial_weights_for_cuda_cpp(
    args: &Args,
    feature_kind: CudaCppNnueFeatureKind,
) -> Result<bulletou_lib::value::NnueForwardOwnedWeights, String> {
    use bulletou_lib::value::{NnueForwardOwnedWeights, NnueForwardShape as FastNnueForwardShape};

    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let base_input_size = feature_kind.base_input_size();
    let virtual_rows = feature_kind.virtual_rows();
    let input_size = feature_kind.training_input_size();
    let l1_input_dim = 2 * l1_size;
    let shape = FastNnueForwardShape { input_size, l1: l1_size, l2: l2_size, l3: l3_size };
    let l0w = if virtual_rows == 0 {
        cuda_cpp_tatara_uniform_abs_init(input_size * l1_size, 0x5071_e001, 0.01)
    } else {
        let mut l0w = vec![0.0_f32; input_size * l1_size];
        let base_l0w = cuda_cpp_tatara_uniform_abs_init(base_input_size * l1_size, 0x5071_e001, 0.01);
        for row in 0..base_input_size {
            let src_start = row * l1_size;
            let dst_start = (virtual_rows + row) * l1_size;
            l0w[dst_start..dst_start + l1_size].copy_from_slice(&base_l0w[src_start..src_start + l1_size]);
        }
        l0w
    };

    let weights = NnueForwardOwnedWeights {
        shape,
        l0w,
        l0b: cuda_cpp_tatara_uniform_abs_init(l1_size, 0x5071_e002, 0.01),
        l1w: cuda_cpp_tatara_uniform_abs_init(l1_input_dim * l2_size, 0x5071_e003, 0.01),
        l1b: cuda_cpp_tatara_uniform_abs_init(l2_size, 0x5071_e004, 0.01),
        l2w: cuda_cpp_tatara_uniform_abs_init(l2_size * l3_size, 0x5071_e005, 0.01),
        l2b: cuda_cpp_tatara_uniform_abs_init(l3_size, 0x5071_e006, 0.01),
        outw: cuda_cpp_tatara_uniform_abs_init(l3_size, 0x5071_e007, 0.01),
        outb: cuda_cpp_tatara_uniform_abs_init(1, 0x5071_e008, 0.01),
    };
    weights.validate().map_err(|e| e.to_string())?;
    Ok(weights)
}

#[cfg(all(feature = "cuda-cpp-backend", test))]
fn build_halfkp_initial_weights_for_cuda_cpp(
    args: &Args,
) -> Result<bulletou_lib::value::NnueForwardOwnedWeights, String> {
    build_nnue_initial_weights_for_cuda_cpp(args, CudaCppNnueFeatureKind::Halfkp)
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_nnue_initial_state(
    path: &Path,
    args: &Args,
    feature_kind: CudaCppNnueFeatureKind,
) -> Result<CudaCppHalfkpInitialState, String> {
    use bulletou_lib::value::{NnueForwardOwnedWeights, NnueForwardShape as FastNnueForwardShape};

    let bytes = std::fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let records =
        parse_model_weights_bin(&bytes).map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let component_weights = bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "weights");
    let weights = if component_weights.is_empty() { &records } else { &component_weights };

    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let input_size = feature_kind.training_input_size();
    let shape = FastNnueForwardShape { input_size, l1: l1_size, l2: l2_size, l3: l3_size };
    let weights = NnueForwardOwnedWeights::from_weight_map(shape, weights).map_err(|err| {
        format!(
            "failed to load cuda-cpp {} weights from {} for arch {}: {err}",
            feature_kind.source_label(),
            path.display(),
            args.arch().cli_name()
        )
    })?;
    let optimizer_states = load_cuda_cpp_halfkp_optimizer_state(&records, &weights)?;
    let completed_steps = load_cuda_cpp_halfkp_completed_steps(&records)?;

    Ok(CudaCppHalfkpInitialState { weights, optimizer_states, completed_steps })
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_halfkp_optimizer_state(
    records: &BTreeMap<String, Vec<f32>>,
    weights: &bulletou_lib::value::NnueForwardOwnedWeights,
) -> Result<Option<CudaCppHalfkpOptimizerState>, String> {
    let momentum = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, "nnue", "momentum");
    let velocity = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, "nnue", "velocity");
    let slow = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, "nnue", "slow");
    let has_any = !momentum.is_empty() || !velocity.is_empty() || !slow.is_empty();
    if !has_any {
        return Ok(None);
    }
    if momentum.is_empty() || velocity.is_empty() || slow.is_empty() {
        return Err(
            "cuda-cpp HalfKP optimizer state is partial: expected nnue/{momentum,velocity,slow}/* records".to_string()
        );
    }

    Ok(Some(CudaCppHalfkpOptimizerState {
        l0w: load_cuda_cpp_ranger_group_state("l0w", weights.l0w.len(), &momentum, &velocity, &slow)?,
        l0b: load_cuda_cpp_ranger_group_state("l0b", weights.l0b.len(), &momentum, &velocity, &slow)?,
        l1w: load_cuda_cpp_ranger_group_state("l1w", weights.l1w.len(), &momentum, &velocity, &slow)?,
        l1b: load_cuda_cpp_ranger_group_state("l1b", weights.l1b.len(), &momentum, &velocity, &slow)?,
        l2w: load_cuda_cpp_ranger_group_state("l2w", weights.l2w.len(), &momentum, &velocity, &slow)?,
        l2b: load_cuda_cpp_ranger_group_state("l2b", weights.l2b.len(), &momentum, &velocity, &slow)?,
        outw: load_cuda_cpp_ranger_group_state("outw", weights.outw.len(), &momentum, &velocity, &slow)?,
        outb: load_cuda_cpp_ranger_group_state("outb", weights.outb.len(), &momentum, &velocity, &slow)?,
    }))
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_ranger_group_state(
    id: &'static str,
    expected_len: usize,
    momentum: &BTreeMap<String, Vec<f32>>,
    velocity: &BTreeMap<String, Vec<f32>>,
    slow: &BTreeMap<String, Vec<f32>>,
) -> Result<CudaCppRangerGroupState, String> {
    load_cuda_cpp_ranger_group_state_for("HalfKP", id, expected_len, momentum, velocity, slow)
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_ranger_group_state_for(
    label: &'static str,
    id: &'static str,
    expected_len: usize,
    momentum: &BTreeMap<String, Vec<f32>>,
    velocity: &BTreeMap<String, Vec<f32>>,
    slow: &BTreeMap<String, Vec<f32>>,
) -> Result<CudaCppRangerGroupState, String> {
    Ok(CudaCppRangerGroupState {
        momentum: load_cuda_cpp_optimizer_record_for(label, "momentum", momentum, id, expected_len)?,
        velocity: load_cuda_cpp_optimizer_record_for(label, "velocity", velocity, id, expected_len)?,
        slow_params: load_cuda_cpp_optimizer_record_for(label, "slow", slow, id, expected_len)?,
    })
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_optimizer_record_for(
    label: &'static str,
    section: &'static str,
    records: &BTreeMap<String, Vec<f32>>,
    id: &'static str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let values =
        records.get(id).ok_or_else(|| format!("cuda-cpp {label} optimizer state missing nnue/{section}/{id}"))?;
    if values.len() != expected_len {
        return Err(format!(
            "cuda-cpp {label} optimizer state nnue/{section}/{id} has length {}, expected {}",
            values.len(),
            expected_len
        ));
    }
    Ok(values.clone())
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_halfkp_completed_steps(records: &BTreeMap<String, Vec<f32>>) -> Result<usize, String> {
    load_cuda_cpp_completed_steps_for("HalfKP", records, &["l0w", "l0b", "l1w", "l1b", "l2w", "l2b", "outw", "outb"])
}

#[cfg(feature = "cuda-cpp-backend")]
fn load_cuda_cpp_completed_steps_for(
    label: &'static str,
    records: &BTreeMap<String, Vec<f32>>,
    ids: &[&'static str],
) -> Result<usize, String> {
    let steps = bulletou_lib::value::yaneuraou_kppt::extract_component_section(records, "nnue", "step_ranger");
    if steps.is_empty() {
        return Ok(0);
    }

    let mut completed_steps: Option<usize> = None;
    for &id in ids {
        let values = steps.get(id).ok_or_else(|| format!("cuda-cpp {label} state missing nnue/step_ranger/{id}"))?;
        let value =
            values.first().copied().ok_or_else(|| format!("cuda-cpp {label} state nnue/step_ranger/{id} is empty"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(format!("cuda-cpp {label} state nnue/step_ranger/{id} is invalid: {value}"));
        }
        let step = value.round() as usize;
        if let Some(prev) = completed_steps {
            if prev != step {
                return Err(format!(
                    "cuda-cpp {label} state has inconsistent step_ranger counters: first={prev}, {id}={step}"
                ));
            }
        } else {
            completed_steps = Some(step);
        }
    }

    Ok(completed_steps.unwrap_or(0))
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_nnue_direct_outputs(
    dir: &Path,
    feature_kind: CudaCppNnueFeatureKind,
    shape: bulletou_cuda_cpp::NnueForwardShape,
    weights: &bulletou_cuda_cpp::NnueTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::NnueRangerOptimizerStatesReadback,
    completed_steps: usize,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    write_cuda_cpp_nnue_nn_bin(&dir.join("nn.bin"), feature_kind, shape, weights)?;
    write_cuda_cpp_halfkp_weights_bin(&dir.join("weights.bin"), weights, optimizer_states, completed_steps)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_halfkp_weights_bin(
    path: &Path,
    weights: &bulletou_cuda_cpp::NnueTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::NnueRangerOptimizerStatesReadback,
    completed_steps: usize,
) -> Result<(), String> {
    let completed_steps = [completed_steps as f32];
    let mut bytes = write_state_backend_marker("cuda-cpp");
    let mut records: Vec<(&str, &[f32])> = vec![
        ("nnue/weights/l0w", weights.l0w.as_slice()),
        ("nnue/weights/l0b", weights.l0b.as_slice()),
        ("nnue/weights/l1w", weights.l1w.as_slice()),
        ("nnue/weights/l1b", weights.l1b.as_slice()),
        ("nnue/weights/l2w", weights.l2w.as_slice()),
        ("nnue/weights/l2b", weights.l2b.as_slice()),
        ("nnue/weights/outw", weights.outw.as_slice()),
        ("nnue/weights/outb", weights.outb.as_slice()),
    ];
    macro_rules! push_group_state {
        ($id:literal, $state:expr) => {{
            let state = $state;
            records.push((concat!("nnue/momentum/", $id), state.momentum.as_slice()));
            records.push((concat!("nnue/velocity/", $id), state.velocity.as_slice()));
            records.push((concat!("nnue/slow/", $id), state.slow_params.as_slice()));
        }};
    }
    push_group_state!("l0w", &optimizer_states.l0w);
    push_group_state!("l0b", &optimizer_states.l0b);
    push_group_state!("l1w", &optimizer_states.l1w);
    push_group_state!("l1b", &optimizer_states.l1b);
    push_group_state!("l2w", &optimizer_states.l2w);
    push_group_state!("l2b", &optimizer_states.l2b);
    push_group_state!("outw", &optimizer_states.outw);
    push_group_state!("outb", &optimizer_states.outb);
    records.extend([
        ("nnue/step_ranger/l0w", completed_steps.as_slice()),
        ("nnue/step_ranger/l0b", completed_steps.as_slice()),
        ("nnue/step_ranger/l1w", completed_steps.as_slice()),
        ("nnue/step_ranger/l1b", completed_steps.as_slice()),
        ("nnue/step_ranger/l2w", completed_steps.as_slice()),
        ("nnue/step_ranger/l2b", completed_steps.as_slice()),
        ("nnue/step_ranger/outw", completed_steps.as_slice()),
        ("nnue/step_ranger/outb", completed_steps.as_slice()),
    ]);
    bytes.extend_from_slice(&bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(records));
    std::fs::write(path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_sfnn_halfka2_direct_outputs(
    dir: &Path,
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    weights: &bulletou_cuda_cpp::SfnnTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::SfnnRangerOptimizerStatesReadback,
    completed_steps: usize,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    write_cuda_cpp_sfnn_halfka2_nn_bin(&dir.join("nn.bin"), shape, weights)?;
    write_cuda_cpp_sfnn_weights_bin(&dir.join("weights.bin"), weights, optimizer_states, completed_steps)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_sfnn_weights_bin(
    path: &Path,
    weights: &bulletou_cuda_cpp::SfnnTrainWeightsReadback,
    optimizer_states: &bulletou_cuda_cpp::SfnnRangerOptimizerStatesReadback,
    completed_steps: usize,
) -> Result<(), String> {
    let completed_steps = [completed_steps as f32];
    let mut bytes = write_state_backend_marker("cuda-cpp");
    let mut records: Vec<(&str, &[f32])> = vec![
        ("nnue/weights/l0w", weights.l0w.as_slice()),
        ("nnue/weights/l0b", weights.l0b.as_slice()),
        ("nnue/weights/l1w", weights.l1w.as_slice()),
        ("nnue/weights/l1b", weights.l1b.as_slice()),
        ("nnue/weights/l2w", weights.l2w.as_slice()),
        ("nnue/weights/l2b", weights.l2b.as_slice()),
        ("nnue/weights/l3w", weights.l3w.as_slice()),
        ("nnue/weights/l3b", weights.l3b.as_slice()),
    ];
    match (&weights.l1fw, &weights.l1fb) {
        (Some(l1fw), Some(l1fb)) => {
            records.push(("nnue/weights/l1fw", l1fw.as_slice()));
            records.push(("nnue/weights/l1fb", l1fb.as_slice()));
        }
        (None, None) => {}
        _ => return Err("cuda-cpp SFNN weights have partial l1f state".to_string()),
    }

    macro_rules! push_group_state {
        ($id:literal, $state:expr) => {{
            let state = $state;
            records.push((concat!("nnue/momentum/", $id), state.momentum.as_slice()));
            records.push((concat!("nnue/velocity/", $id), state.velocity.as_slice()));
            records.push((concat!("nnue/slow/", $id), state.slow_params.as_slice()));
        }};
    }
    push_group_state!("l0w", &optimizer_states.l0w);
    push_group_state!("l0b", &optimizer_states.l0b);
    push_group_state!("l1w", &optimizer_states.l1w);
    push_group_state!("l1b", &optimizer_states.l1b);
    match (&optimizer_states.l1fw, &optimizer_states.l1fb) {
        (Some(l1fw), Some(l1fb)) => {
            push_group_state!("l1fw", l1fw);
            push_group_state!("l1fb", l1fb);
        }
        (None, None) => {}
        _ => return Err("cuda-cpp SFNN optimizer states have partial l1f state".to_string()),
    }
    push_group_state!("l2w", &optimizer_states.l2w);
    push_group_state!("l2b", &optimizer_states.l2b);
    push_group_state!("l3w", &optimizer_states.l3w);
    push_group_state!("l3b", &optimizer_states.l3b);

    records.extend([
        ("nnue/step_ranger/l0w", completed_steps.as_slice()),
        ("nnue/step_ranger/l0b", completed_steps.as_slice()),
        ("nnue/step_ranger/l1w", completed_steps.as_slice()),
        ("nnue/step_ranger/l1b", completed_steps.as_slice()),
        ("nnue/step_ranger/l2w", completed_steps.as_slice()),
        ("nnue/step_ranger/l2b", completed_steps.as_slice()),
        ("nnue/step_ranger/l3w", completed_steps.as_slice()),
        ("nnue/step_ranger/l3b", completed_steps.as_slice()),
    ]);
    if weights.l1fw.is_some() {
        records.extend([
            ("nnue/step_ranger/l1fw", completed_steps.as_slice()),
            ("nnue/step_ranger/l1fb", completed_steps.as_slice()),
        ]);
    }

    bytes.extend_from_slice(&bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(records));
    std::fs::write(path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_sfnn_halfka2_nn_bin(
    path: &Path,
    shape: bulletou_cuda_cpp::SfnnForwardShape,
    weights: &bulletou_cuda_cpp::SfnnTrainWeightsReadback,
) -> Result<(), String> {
    use std::io::Write as _;

    let feature_set = NnueFeatureSet::HalfKa2;
    let base_input_size = feature_set.input_size();
    let virtual_rows = bulletou_lib::game::inputs::PIECE_INPUTS;
    if shape.input_size != base_input_size && shape.input_size != base_input_size + virtual_rows {
        return Err(format!(
            "cannot write SFNN HalfKA2 nn.bin for input_size={}, expected {} or factorized {}",
            shape.input_size,
            base_input_size,
            base_input_size + virtual_rows
        ));
    }
    let folded_l0w;
    let l0w_for_export: &[f32] = if shape.input_size == base_input_size {
        &weights.l0w
    } else {
        folded_l0w =
            fold_sfnn_halfka2_piece_factorized_l0w(&weights.l0w, base_input_size, virtual_rows, shape.ft_size)?;
        &folded_l0w
    };

    let file = std::fs::File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    let arch = format!(
        "ModelType=SFNNWithoutPsqt;Features={}[{}->{}x2],Network=SFNN-{}{{LayerStack={}}}",
        feature_set.display_name(),
        base_input_size,
        shape.ft_size,
        shape.ft_size,
        shape.num_stacks
    );
    writer
        .write_all(&SFNN_NNUE_VERSION.to_le_bytes())
        .and_then(|_| writer.write_all(&KHASH_SFNN.to_le_bytes()))
        .and_then(|_| writer.write_all(&(arch.len() as u32).to_le_bytes()))
        .and_then(|_| writer.write_all(arch.as_bytes()))
        .and_then(|_| writer.write_all(&FT_HASH_SFNN.to_le_bytes()))
        .map_err(|err| format!("failed to write SFNN nn.bin header {}: {err}", path.display()))?;

    write_sfnn_leb128_i16_chunk(&mut writer, path, "l0b", &weights.l0b, f32::from(SFNN_QA))?;
    write_sfnn_leb128_i16_chunk(&mut writer, path, "l0w", l0w_for_export, f32::from(SFNN_QA))?;

    let l1_out = shape.l1_out();
    let l2_in = shape.l2_in();
    let fc_bias_scale = f32::from(SFNN_QA) * f32::from(SFNN_QB);
    let fc_weight_scale = f32::from(SFNN_QB);
    let l1fw = weights.l1fw.as_deref();
    let l1fb = weights.l1fb.as_deref();
    if l1fw.is_some() != l1fb.is_some() {
        return Err("cuda-cpp SFNN weights have partial l1f state".to_string());
    }

    for stack in 0..shape.num_stacks {
        writer
            .write_all(&NETWORK_HASH_SFNN.to_le_bytes())
            .map_err(|err| format!("failed to write SFNN nn.bin stack hash {}: {err}", path.display()))?;

        let mut l1b_bytes = Vec::with_capacity(l1_out * std::mem::size_of::<i32>());
        for out_col in 0..l1_out {
            let mut value = weights.l1b[stack * l1_out + out_col];
            if let Some(shared_bias) = l1fb {
                value += shared_bias[out_col];
            }
            l1b_bytes.extend_from_slice(&sfnn_quantise_i32(value, fc_bias_scale).to_le_bytes());
        }
        write_nnue_bin_chunk(&mut writer, path, "sfnn l1b", &l1b_bytes)?;

        let l1_pad_in = nnue_pad32(shape.ft_size);
        let mut l1w_bytes = Vec::with_capacity(l1_out * l1_pad_in);
        for out_col in 0..l1_out {
            for in_col in 0..l1_pad_in {
                let q = if in_col < shape.ft_size {
                    let mut value = weights.l1w[stack * l1_out * shape.ft_size + out_col * shape.ft_size + in_col];
                    if let Some(shared_weights) = l1fw {
                        value += shared_weights[in_col * l1_out + out_col];
                    }
                    sfnn_quantise_i8(value, fc_weight_scale)
                } else {
                    0
                };
                l1w_bytes.push(q as u8);
            }
        }
        write_nnue_bin_chunk(&mut writer, path, "sfnn l1w", &l1w_bytes)?;

        let mut l2b_bytes = Vec::with_capacity(shape.l2_size * std::mem::size_of::<i32>());
        for out_col in 0..shape.l2_size {
            let value = weights.l2b[stack * shape.l2_size + out_col];
            l2b_bytes.extend_from_slice(&sfnn_quantise_i32(value, fc_bias_scale).to_le_bytes());
        }
        write_nnue_bin_chunk(&mut writer, path, "sfnn l2b", &l2b_bytes)?;

        let l2_pad_in = nnue_pad32(l2_in);
        let mut l2w_bytes = Vec::with_capacity(shape.l2_size * l2_pad_in);
        for out_col in 0..shape.l2_size {
            for in_col in 0..l2_pad_in {
                let q = if in_col < l2_in {
                    let value = weights.l2w[stack * shape.l2_size * l2_in + out_col * l2_in + in_col];
                    sfnn_quantise_i8(value, fc_weight_scale)
                } else {
                    0
                };
                l2w_bytes.push(q as u8);
            }
        }
        write_nnue_bin_chunk(&mut writer, path, "sfnn l2w", &l2w_bytes)?;

        let l3b_bytes = sfnn_quantise_i32(weights.l3b[stack], fc_bias_scale).to_le_bytes();
        write_nnue_bin_chunk(&mut writer, path, "sfnn l3b", &l3b_bytes)?;

        let l3_pad_in = nnue_pad32(shape.l2_size);
        let mut l3w_bytes = Vec::with_capacity(l3_pad_in);
        for in_col in 0..l3_pad_in {
            let q = if in_col < shape.l2_size {
                sfnn_quantise_i8(weights.l3w[stack * shape.l2_size + in_col], fc_weight_scale)
            } else {
                0
            };
            l3w_bytes.push(q as u8);
        }
        write_nnue_bin_chunk(&mut writer, path, "sfnn l3w", &l3w_bytes)?;
    }

    writer.flush().map_err(|err| format!("failed to flush SFNN nn.bin {}: {err}", path.display()))
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_cuda_cpp_nnue_nn_bin(
    path: &Path,
    feature_kind: CudaCppNnueFeatureKind,
    shape: bulletou_cuda_cpp::NnueForwardShape,
    weights: &bulletou_cuda_cpp::NnueTrainWeightsReadback,
) -> Result<(), String> {
    use std::io::Write as _;

    let feature_set = feature_kind.feature_set();
    let base_input_size = feature_kind.base_input_size();
    let virtual_rows = feature_kind.virtual_rows();
    let factorized_input_size = base_input_size + virtual_rows;
    if shape.input_size != base_input_size && shape.input_size != factorized_input_size {
        return Err(format!(
            "cannot write {} nn.bin for input_size={}, expected {} or factorized {}",
            feature_kind.source_label(),
            shape.input_size,
            base_input_size,
            factorized_input_size
        ));
    }
    let folded_l0w;
    let l0w_for_export: &[f32] = if shape.input_size == base_input_size {
        &weights.l0w
    } else if virtual_rows > 0 {
        folded_l0w = fold_halfkp_piece_factorized_l0w(&weights.l0w, base_input_size, virtual_rows, shape.l1)?;
        &folded_l0w
    } else {
        return Err(format!(
            "cannot write {} nn.bin for factorized input_size={} because the feature has no virtual rows",
            feature_kind.source_label(),
            shape.input_size
        ));
    };

    let qa: i16 = 127;
    let qb: i16 = 64;
    let l1_input_dim = shape.l1 * 2;
    let l1_bias = l1_bias_scale(NnueActivation::Crelu, false, qa, qb);

    let file = std::fs::File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    writer
        .write_all(&header_bytes(feature_set, shape.l1, shape.l2, shape.l3))
        .and_then(|_| writer.write_all(&ft_hash_bytes(feature_set, shape.l1)))
        .map_err(|err| format!("failed to write NNUE nn.bin header {}: {err}", path.display()))?;

    write_quantized_i16(&mut writer, path, "l0b", &weights.l0b, qa)?;
    write_quantized_i16(&mut writer, path, "l0w", l0w_for_export, qa)?;
    writer
        .write_all(&network_layer_hash_bytes(shape.l1, shape.l2, shape.l3))
        .map_err(|err| format!("failed to write NNUE nn.bin network hash {}: {err}", path.display()))?;

    write_quantized_i32(&mut writer, path, "l1b", &weights.l1b, l1_bias)?;
    let l1w = transpose_input_major_dense_weights(&weights.l1w, l1_input_dim, shape.l2)?;
    let l1w = pad_weights_for_simd(&l1w, shape.l2, l1_input_dim);
    write_quantized_i8(&mut writer, path, "l1w", &l1w, qb)?;

    write_quantized_i32(&mut writer, path, "l2b", &weights.l2b, 127 * i32::from(qb))?;
    let l2w = transpose_input_major_dense_weights(&weights.l2w, shape.l2, shape.l3)?;
    let l2w = pad_weights_for_simd(&l2w, shape.l3, shape.l2);
    write_quantized_i8(&mut writer, path, "l2w", &l2w, qb)?;

    write_quantized_i32(&mut writer, path, "outb", &weights.outb, 127 * i32::from(qb))?;
    let outw = transpose_input_major_dense_weights(&weights.outw, shape.l3, 1)?;
    let outw = pad_weights_for_simd(&outw, 1, shape.l3);
    write_quantized_i8(&mut writer, path, "outw", &outw, qb)?;

    writer.flush().map_err(|err| format!("failed to flush NNUE nn.bin {}: {err}", path.display()))
}

#[cfg(feature = "cuda-cpp-backend")]
fn fold_halfkp_piece_factorized_l0w(
    weights: &[f32],
    base_input_size: usize,
    virtual_rows: usize,
    l1: usize,
) -> Result<Vec<f32>, String> {
    let expected = base_input_size
        .checked_add(virtual_rows)
        .and_then(|rows| rows.checked_mul(l1))
        .ok_or_else(|| {
            format!("factorized HalfKP l0w shape overflow: base_input_size={base_input_size} virtual_rows={virtual_rows} l1={l1}")
        })?;
    if weights.len() != expected {
        return Err(format!("factorized HalfKP l0w length mismatch: expected {expected}, got {}", weights.len()));
    }
    let mut folded = vec![0.0_f32; base_input_size * l1];
    for row in 0..base_input_size {
        let piece = row % virtual_rows;
        let virtual_start = piece * l1;
        let base_start = (virtual_rows + row) * l1;
        let dst_start = row * l1;
        for col in 0..l1 {
            folded[dst_start + col] = weights[base_start + col] + weights[virtual_start + col];
        }
    }
    Ok(folded)
}

#[cfg(feature = "cuda-cpp-backend")]
fn fold_sfnn_halfka2_piece_factorized_l0w(
    weights: &[f32],
    base_input_size: usize,
    virtual_rows: usize,
    ft_size: usize,
) -> Result<Vec<f32>, String> {
    let expected =
        base_input_size.checked_add(virtual_rows).and_then(|rows| rows.checked_mul(ft_size)).ok_or_else(|| {
            format!(
                "factorized SFNN HalfKA2 l0w shape overflow: base_input_size={base_input_size} \
                 virtual_rows={virtual_rows} ft_size={ft_size}"
            )
        })?;
    if weights.len() != expected {
        return Err(format!("factorized SFNN HalfKA2 l0w length mismatch: expected {expected}, got {}", weights.len()));
    }
    let mut folded = vec![0.0_f32; base_input_size * ft_size];
    for row in 0..base_input_size {
        let virtual_row = base_input_size + row % virtual_rows;
        let base_start = row * ft_size;
        let virtual_start = virtual_row * ft_size;
        for col in 0..ft_size {
            folded[base_start + col] = weights[base_start + col] + weights[virtual_start + col];
        }
    }
    Ok(folded)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_sfnn_leb128_i16_chunk(
    writer: &mut impl std::io::Write,
    path: &Path,
    name: &'static str,
    values: &[f32],
    scale: f32,
) -> Result<(), String> {
    let mut payload = Vec::with_capacity(values.len() * std::mem::size_of::<i16>());
    for &value in values {
        push_sfnn_signed_leb128_i16(&mut payload, sfnn_quantise_i16(value, scale));
    }
    let mut block = Vec::with_capacity(LEB128_MAGIC.len() + std::mem::size_of::<u32>() + payload.len());
    block.extend_from_slice(LEB128_MAGIC);
    block.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    block.extend_from_slice(&payload);
    write_nnue_bin_chunk(writer, path, name, &block)
}

#[cfg(feature = "cuda-cpp-backend")]
fn push_sfnn_signed_leb128_i16(out: &mut Vec<u8>, value: i16) {
    let mut v = i32::from(value);
    loop {
        let byte = (v as u8) & 0x7f;
        v >>= 7;
        let sign_bit = byte & 0x40 != 0;
        let done = (v == 0 && !sign_bit) || (v == -1 && sign_bit);
        out.push(if done { byte } else { byte | 0x80 });
        if done {
            break;
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn sfnn_quantise_i16(value: f32, scale: f32) -> i16 {
    (f64::from(value) * f64::from(scale)).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

#[cfg(feature = "cuda-cpp-backend")]
fn sfnn_quantise_i32(value: f32, scale: f32) -> i32 {
    (f64::from(value) * f64::from(scale)).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

#[cfg(feature = "cuda-cpp-backend")]
fn sfnn_quantise_i8(value: f32, scale: f32) -> i8 {
    (f64::from(value) * f64::from(scale)).round().clamp(i8::MIN as f64, i8::MAX as f64) as i8
}

#[cfg(feature = "cuda-cpp-backend")]
fn transpose_input_major_dense_weights(
    weights: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<f32>, String> {
    let expected = input_dim
        .checked_mul(output_dim)
        .ok_or_else(|| format!("dense weight shape overflow: input_dim={input_dim} output_dim={output_dim}"))?;
    if weights.len() != expected {
        return Err(format!(
            "dense weight length mismatch: got {}, expected input_dim({input_dim}) * output_dim({output_dim}) = {expected}",
            weights.len()
        ));
    }
    let mut transposed = vec![0.0_f32; weights.len()];
    for input in 0..input_dim {
        for output in 0..output_dim {
            transposed[output * input_dim + input] = weights[input * output_dim + output];
        }
    }
    Ok(transposed)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_quantized_i8(
    writer: &mut impl std::io::Write,
    path: &Path,
    name: &'static str,
    values: &[f32],
    scale: i16,
) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len());
    for &value in values {
        let qf = (f64::from(scale) * f64::from(value)).round();
        if qf < f64::from(i8::MIN) || qf > f64::from(i8::MAX) {
            return Err(quantization_error(path, name, "i8", value, qf));
        }
        bytes.extend_from_slice(&(qf as i8).to_le_bytes());
    }
    write_nnue_bin_chunk(writer, path, name, &bytes)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_quantized_i16(
    writer: &mut impl std::io::Write,
    path: &Path,
    name: &'static str,
    values: &[f32],
    scale: i16,
) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i16>());
    for &value in values {
        let qf = (f64::from(scale) * f64::from(value)).round();
        if qf < f64::from(i16::MIN) || qf > f64::from(i16::MAX) {
            return Err(quantization_error(path, name, "i16", value, qf));
        }
        bytes.extend_from_slice(&(qf as i16).to_le_bytes());
    }
    write_nnue_bin_chunk(writer, path, name, &bytes)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_quantized_i32(
    writer: &mut impl std::io::Write,
    path: &Path,
    name: &'static str,
    values: &[f32],
    scale: i32,
) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i32>());
    for &value in values {
        let qf = (f64::from(scale) * f64::from(value)).round();
        if qf < f64::from(i32::MIN) || qf > f64::from(i32::MAX) {
            return Err(quantization_error(path, name, "i32", value, qf));
        }
        bytes.extend_from_slice(&(qf as i32).to_le_bytes());
    }
    write_nnue_bin_chunk(writer, path, name, &bytes)
}

#[cfg(feature = "cuda-cpp-backend")]
fn write_nnue_bin_chunk(
    writer: &mut impl std::io::Write,
    path: &Path,
    name: &'static str,
    bytes: &[u8],
) -> Result<(), String> {
    writer.write_all(bytes).map_err(|err| format!("failed to write NNUE nn.bin {name} chunk {}: {err}", path.display()))
}

#[cfg(feature = "cuda-cpp-backend")]
fn quantization_error(path: &Path, name: &'static str, target: &'static str, value: f32, quantized: f64) -> String {
    format!("failed to quantize NNUE nn.bin {name} value {value} -> {quantized} as {target} for {}", path.display())
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_tatara_uniform_abs_init(len: usize, seed: u64, half_width: f32) -> Vec<f32> {
    let mut rng = CudaCppTataraXorShift::new(seed);
    (0..len).map(|_| rng.next_signed_unit() * half_width).collect()
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_tatara_uniform_fan_in_init(len: usize, seed: u64, fan_in: usize, init_scale: f32) -> Vec<f32> {
    let fan_in = fan_in.max(1) as f32;
    let half_width = init_scale * (1.0 / fan_in).sqrt();
    cuda_cpp_tatara_uniform_abs_init(len, seed, half_width)
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_should_read_loss(step: usize, total_steps: usize, interval: usize) -> bool {
    if step == total_steps {
        true
    } else if interval == 0 {
        false
    } else {
        step == 1 || step % interval == 0
    }
}

fn cuda_cpp_uses_production_schedule(args: &Args) -> bool {
    args.superbatches.is_some()
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Clone, Debug)]
struct CudaCppScheduleChunk {
    epoch: usize,
    superbatch: usize,
    steps: usize,
    cumulative_steps: usize,
    save_checkpoint: bool,
    lr_start: f32,
    lr_end: f32,
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Clone, Debug)]
struct CudaCppRunSchedule {
    production: bool,
    total_steps: usize,
    batches_per_superbatch: usize,
    prior_positions: usize,
    lr_period: u64,
    lr_step_gamma: f32,
    lr_step_positions: u64,
    chunks: Vec<CudaCppScheduleChunk>,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppRunSchedule {
    fn lr_for_step(&self, args: &Args, step_index: usize, batch_size: usize) -> f32 {
        if self.production {
            cuda_cpp_lr_at_step(
                args,
                step_index,
                batch_size,
                self.prior_positions,
                self.lr_period,
                self.lr_step_gamma,
                self.lr_step_positions,
            )
        } else {
            args.lr
        }
    }
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_run_schedule(args: &Args) -> Result<CudaCppRunSchedule, String> {
    let batch_size = effective_batch_size(args);
    let default_lr_step_positions = effective_lr_step_positions(args, 1);
    if let Some(train_steps) = args.cuda_cpp_train_steps {
        let lr = args.lr;
        return Ok(CudaCppRunSchedule {
            production: false,
            total_steps: train_steps,
            batches_per_superbatch: train_steps,
            prior_positions: 0,
            lr_period: 0,
            lr_step_gamma: DEFAULT_LR_STEP_GAMMA,
            lr_step_positions: default_lr_step_positions,
            chunks: vec![CudaCppScheduleChunk {
                epoch: 1,
                superbatch: 1,
                steps: train_steps,
                cumulative_steps: train_steps,
                save_checkpoint: true,
                lr_start: lr,
                lr_end: lr,
            }],
        });
    }

    let superbatches = args
        .superbatches
        .ok_or_else(|| "--backend cuda-cpp production schedule requires --superbatches".to_string())?;
    let max_epochs = args
        .max_epochs
        .ok_or_else(|| "--backend cuda-cpp production schedule requires --max-epochs".to_string())?
        .max(1);
    let batches_per_superbatch = effective_batches_per_superbatch(args)?;
    let total_steps = max_epochs
        .checked_mul(superbatches)
        .and_then(|v| v.checked_mul(batches_per_superbatch))
        .ok_or_else(|| {
            format!(
                "cuda-cpp schedule step count overflow: max_epochs={max_epochs}, superbatches={superbatches}, batches_per_superbatch={batches_per_superbatch}"
            )
        })?;
    if total_steps == 0 {
        return Err("--backend cuda-cpp production schedule resolved to zero train steps".to_string());
    }
    let (lr_step_gamma, _) = effective_lr_step_gamma(args, batches_per_superbatch)?;
    let lr_step_positions = effective_lr_step_positions(args, batches_per_superbatch);
    let lr_period = (superbatches as u64)
        .checked_mul(batches_per_superbatch as u64)
        .and_then(|v| v.checked_mul(batch_size as u64))
        .ok_or_else(|| "cuda-cpp LR period overflow".to_string())?;
    let output_dir = args.output_dir();
    let top_level_log = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let resume_enabled = resume_enabled(args, &output_dir);
    let latest_superbatch = if resume_enabled { read_latest_saved_superbatch(&output_dir) } else { None };
    let prev_teacher = if resume_enabled { read_latest_saved_teacher(&output_dir) } else { None };
    let teacher_changed =
        prev_teacher.as_deref().is_some_and(|prev| prev.trim() != resolve_teacher_for_log(&args.teacher).trim());
    let prev_run_completed_epoch = latest_superbatch.map(|last_sb| last_sb >= superbatches).unwrap_or(false);
    let max_epoch_in_log =
        if resume_enabled { read_latest_epoch_in_top_level_log(&top_level_log).unwrap_or(0) } else { 0 };
    let mid_epoch_resume = !teacher_changed && !prev_run_completed_epoch && latest_superbatch.is_some();
    let epoch_offset = if mid_epoch_resume {
        max_epoch_in_log.saturating_sub(1)
    } else if resume_enabled {
        max_epoch_in_log
    } else {
        0
    };
    let first_epoch_start_superbatch = if teacher_changed || prev_run_completed_epoch {
        1usize
    } else {
        latest_superbatch.map(|last_sb| last_sb + 1).unwrap_or(1)
    };
    let prior_positions = if resume_enabled {
        let positions = read_prior_positions(&top_level_log);
        if matches!(args.eval_type(), EvalType::Kppt | EvalType::KppKkpt) {
            ["kk", "kkp", "kpp"].iter().filter_map(|component| positions.get(*component).copied()).min().unwrap_or(0)
        } else {
            positions.get("nnue").copied().unwrap_or(0)
        }
    } else {
        0
    };

    let mut chunks = Vec::new();
    let mut cumulative_steps = 0usize;
    let save_rate = effective_save_rate(args).max(1);
    let save_epoch_end = effective_save_epoch_end(args);
    for local_epoch in 1..=max_epochs {
        let epoch = epoch_offset + local_epoch;
        let mut first_superbatch = if local_epoch == 1 { first_epoch_start_superbatch } else { 1 };
        while first_superbatch <= superbatches {
            let save_boundary = first_superbatch.saturating_add(save_rate).saturating_sub(1);
            let (last_superbatch, save_checkpoint) =
                if save_boundary <= superbatches { (save_boundary, true) } else { (superbatches, save_epoch_end) };
            let superbatch_count = last_superbatch - first_superbatch + 1;
            let steps = superbatch_count.checked_mul(batches_per_superbatch).ok_or_else(|| {
                format!("cuda-cpp chunk step overflow at epoch={epoch}, superbatch={last_superbatch}")
            })?;
            let chunk_start_step = cumulative_steps;
            cumulative_steps = cumulative_steps.checked_add(steps).ok_or_else(|| {
                format!("cuda-cpp cumulative step overflow at epoch={epoch}, superbatch={last_superbatch}")
            })?;
            let chunk_end_step = cumulative_steps.saturating_sub(1);
            let lr_start = cuda_cpp_lr_at_step(
                args,
                chunk_start_step,
                batch_size,
                prior_positions,
                lr_period,
                lr_step_gamma,
                lr_step_positions,
            );
            let lr_end = cuda_cpp_lr_at_step(
                args,
                chunk_end_step,
                batch_size,
                prior_positions,
                lr_period,
                lr_step_gamma,
                lr_step_positions,
            );
            chunks.push(CudaCppScheduleChunk {
                epoch,
                superbatch: last_superbatch,
                steps,
                cumulative_steps,
                save_checkpoint,
                lr_start,
                lr_end,
            });
            first_superbatch = last_superbatch + 1;
        }
    }
    if chunks.is_empty() {
        return Err("--backend cuda-cpp production schedule has no remaining chunks to train".to_string());
    }

    let total_steps = chunks.iter().map(|chunk| chunk.steps).sum();
    Ok(CudaCppRunSchedule {
        production: true,
        total_steps,
        batches_per_superbatch,
        prior_positions,
        lr_period,
        lr_step_gamma,
        lr_step_positions,
        chunks,
    })
}

#[cfg(feature = "cuda-cpp-backend")]
fn cuda_cpp_lr_at_step(
    args: &Args,
    step_index: usize,
    batch_size: usize,
    prior_positions: usize,
    lr_period: u64,
    lr_step_gamma: f32,
    lr_step_positions: u64,
) -> f32 {
    let positions = (prior_positions as u64).saturating_add((step_index as u64).saturating_mul(batch_size as u64));
    match args.lr_schedule {
        LrScheduleKind::Step => {
            StepLR::lr_at_positions(args.lr, args.lr_min, lr_step_gamma, lr_step_positions, lr_period, positions)
        }
        LrScheduleKind::Geometric => GeometricLR::lr_at_positions(args.lr, args.lr_min, lr_period, positions),
        LrScheduleKind::Cos => CosineLR::lr_at_positions(args.lr, args.lr_min, lr_period, positions),
        LrScheduleKind::Plateau => args.lr,
    }
}

#[cfg(feature = "cuda-cpp-backend")]
#[derive(Clone, Debug)]
struct CudaCppTataraXorShift {
    state: u64,
}

#[cfg(feature = "cuda-cpp-backend")]
impl CudaCppTataraXorShift {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_unit(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 11) as f32 / ((1u64 << 53) as f32)
    }

    fn next_signed_unit(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

fn cuda_cpp_supported_arch(eval_type: EvalType) -> NnueArch {
    match eval_type {
        EvalType::NnueHalfkp => NnueArch::default_for_eval_type(EvalType::NnueHalfkp),
        EvalType::NnueKp => NnueArch::default_for_eval_type(EvalType::NnueKp),
        EvalType::SfnnHalfka2 => NnueArch::new(
            NnueArchFamily::Sfnn,
            NnueArchFeature::Halfka2,
            1024,
            7,
            64,
            Some(LayerStackMode::Kingrank3by3),
        ),
        other => unreachable!("cuda-cpp unsupported eval type reached arch selection: {other:?}"),
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
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // argv joined with single spaces; quoting/escaping is intentionally
    // not applied — the line is for human eyeballing, not for re-execution.
    // (clap-parsed values are mostly path/identifier strings without
    // spaces; if the user did pass a quoted path, the original quoting
    // is lost by the time we see std::env::args, so reconstructing it is
    // best-effort regardless.)
    let cmdline: String = std::env::args().collect::<Vec<_>>().join(" ");
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&tag_path)?;
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

const RESUME_CONFIG_NAME: &str = "resume-config.txt";

/// Small, stable signature for the training controls that must match for
/// implicit auto-resume. Teacher paths are intentionally not part of this:
/// continuing a trained model on a new teacher is a supported workflow, but
/// changing controls such as `--superbatches` or LR policy should require an
/// explicit `--resume`.
fn resume_signature(args: &Args) -> String {
    let positions_per_superbatch = effective_positions_per_superbatch(args).unwrap_or(DEFAULT_POSITIONS_PER_SUPERBATCH);
    let batches_per_superbatch = effective_batches_per_superbatch(args).unwrap_or(1);
    let lr_step_gamma =
        effective_lr_step_gamma(args, batches_per_superbatch).map(|(gamma, _)| gamma).unwrap_or(DEFAULT_LR_STEP_GAMMA);
    let arch = if args.eval_type().uses_arch() { args.arch().cli_name() } else { "-".to_string() };
    let superbatches = args.superbatches.map(|n| n.to_string()).unwrap_or_else(|| "none".to_string());
    let test_teacher =
        args.test_teacher.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".to_string());

    [
        "schema=bulletou-resume-v2".to_string(),
        format!("backend={}", args.backend.cli_name()),
        format!("eval_type={}", args.eval_type().cli_name()),
        format!("arch={arch}"),
        format!("net_id={}", args.net_id()),
        format!("batch_size={}", effective_batch_size(args)),
        format!("positions_per_superbatch={positions_per_superbatch}"),
        format!("superbatches={superbatches}"),
        format!("lr_schedule={}", args.lr_schedule.cli_name()),
        format!("optimizer={}", args.optimizer.cli_name()),
        format!("lr={:.9}", args.lr),
        format!("lr_min={:.9}", args.lr_min),
        format!("lr_step_gamma={lr_step_gamma:.9}"),
        format!(
            "lr_step_positions={}",
            args.lr_step_positions.map(|n| n.to_string()).unwrap_or_else(|| "none".to_string())
        ),
        format!("lr_plateau_factor={:.9}", args.lr_plateau_factor),
        format!("lr_plateau_min_delta={:.9}", args.lr_plateau_min_delta),
        format!("lr_plateau_monitor={}", args.lr_plateau_monitor.cli_name()),
        format!("lambda={:.9}", args.lambda),
        format!("scale={}", args.scale),
        format!("nnue_pytorch_wrm_loss={}", args.nnue_pytorch_wrm_loss),
        format!("optimizer_weight_decay={:.9}", args.optimizer_weight_decay),
        format!(
            "optimizer_epsilon={}",
            args.optimizer_epsilon.map(|v| format!("{v:.9}")).unwrap_or_else(|| "none".to_string())
        ),
        format!(
            "optimizer_beta1={}",
            args.optimizer_beta1.map(|v| format!("{v:.9}")).unwrap_or_else(|| "none".to_string())
        ),
        format!(
            "optimizer_beta2={}",
            args.optimizer_beta2.map(|v| format!("{v:.9}")).unwrap_or_else(|| "none".to_string())
        ),
        format!("nnue_pytorch_layer_clip={}", args.nnue_pytorch_layer_clip),
        format!("nnue_pytorch_no_bias_clip={}", args.nnue_pytorch_no_bias_clip),
        format!("save_rate={}", effective_save_rate(args)),
        format!("save_epoch_end={}", effective_save_epoch_end(args)),
        format!("score_drop_abs={}", args.score_drop_abs),
        format!("nnue_pytorch_init_scale={:.9}", args.nnue_pytorch_init_scale),
        format!("sfnn_factorized_l1={}", args.sfnn_factorized_l1),
        format!("test_teacher={test_teacher}"),
        format!("test_positions={}", args.test_positions),
        format!("test_batch_size={}", args.test_batch_size),
        format!("test_sample={}", args.test_sample.cli_name()),
        format!("test_seed={}", args.test_seed),
    ]
    .join("\n")
        + "\n"
}

fn write_resume_config(output_dir: &std::path::Path, args: &Args) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    std::fs::write(output_dir.join(RESUME_CONFIG_NAME), resume_signature(args))
}

fn resume_config_matches(output_dir: &std::path::Path, args: &Args) -> Result<bool, std::io::Error> {
    let stored = std::fs::read_to_string(output_dir.join(RESUME_CONFIG_NAME))?;
    Ok(stored.trim_end() == resume_signature(args).trim_end())
}

/// Find the latest numbered subdirectory under `output_dir` (4-or-more-digit
/// name parsable as `usize`) whose `state.bin` exists. Returns `None` if no
/// resumable checkpoint is found.
fn find_latest_state_bin_raw(output_dir: &std::path::Path) -> Option<std::path::PathBuf> {
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

fn latest_checkpoint_dir(output_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    find_latest_state_bin_raw(output_dir).and_then(|p| p.parent().map(|dir| dir.to_path_buf()))
}

fn latest_checkpoint_has_file(output_dir: &std::path::Path, filename: &str) -> bool {
    latest_checkpoint_dir(output_dir).map(|dir| dir.join(filename).is_file()).unwrap_or(false)
}

fn mark_latest_checkpoint_epoch_done(output_dir: &std::path::Path) {
    match latest_checkpoint_dir(output_dir) {
        Some(dir) => {
            let marker = dir.join(PLATEAU_EPOCH_DONE_NAME);
            if let Err(e) = std::fs::write(&marker, b"1\n") {
                eprintln!("  WARN: failed to write {}: {e}", marker.display());
            }
        }
        None => {
            eprintln!("  WARN: plateau epoch ended with no accepted checkpoint to mark complete");
        }
    }
}

fn resume_enabled(args: &Args, output_dir: &std::path::Path) -> bool {
    if args.no_resume {
        return false;
    }
    if find_latest_state_bin_raw(output_dir).is_none() {
        return false;
    }
    if args.resume {
        return true;
    }
    resume_config_matches(output_dir, args).unwrap_or(false)
}

fn find_latest_state_bin(args: &Args, output_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if resume_enabled(args, output_dir) { find_latest_state_bin_raw(output_dir) } else { None }
}

fn prepare_resume_config_or_exit(args: &Args) {
    let output_dir = args.output_dir();
    let latest_state = find_latest_state_bin_raw(&output_dir);

    if latest_state.is_some() && args.no_resume {
        eprintln!(
            "error: --no-resume was specified, but {} already contains a resumable checkpoint.\n  \
             Use a different --tag/--output, or remove/rename the existing checkpoint directory.",
            output_dir.display()
        );
        std::process::exit(2);
    }

    if latest_state.is_some() && !args.resume {
        match resume_config_matches(&output_dir, args) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!(
                    "error: auto-resume refused because {} differs from the current training controls.\n  \
                     output: {}\n  \
                     If this is intentional, rerun with --resume. For a new experiment, use a new --tag/--output.",
                    output_dir.join(RESUME_CONFIG_NAME).display(),
                    output_dir.display()
                );
                std::process::exit(2);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!(
                    "error: auto-resume refused because {} has old checkpoints but no {}.\n  \
                     This checkpoint was created before resume compatibility tracking.\n  \
                     If you really want to continue it, rerun with --resume. For a new experiment, use a new --tag/--output.",
                    output_dir.display(),
                    RESUME_CONFIG_NAME
                );
                std::process::exit(2);
            }
            Err(e) => {
                eprintln!("error: failed to read {}: {e}", output_dir.join(RESUME_CONFIG_NAME).display());
                std::process::exit(2);
            }
        }
    }

    if let Err(e) = write_resume_config(&output_dir, args) {
        eprintln!("warning: failed to write {}: {e}", output_dir.join(RESUME_CONFIG_NAME).display());
    }
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

    eprintln!("=== bulletou: running {} family (3 components) ===", args.eval_type().cli_name());

    // ---- Resume support -------------------------------------------------
    // If `<output>` already contains a numbered dir with a `state.bin`,
    // unbundle each component's records into a per-component
    // `optimiser_state/` triplet under `<output>/.bulletou_resume/<comp>/`,
    // and let each child run_kppt_* call `trainer.load_from_checkpoint(<comp>)`
    // immediately after building its trainer.
    let resume_state_bin = find_latest_state_bin(args, &output_dir);
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
                unbundle_component_state(&records, comp, &comp_dir.join("optimiser_state")).unwrap_or_else(|e| {
                    eprintln!("error: state.bin missing `{comp}/*` records: {e}");
                    std::process::exit(1);
                });
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
    let ctx = LogContext::from_args(args, 0);
    let prior_positions = read_prior_positions(&output_dir.join(SUMMARY_LEARN_LOG_NAME));
    match assemble_numbered_dirs(&output_dir, &ctx, &prior_positions, STATE_BACKEND_BULLET) {
        Ok((_first_idx, last_idx)) => {
            // Append the new run's per-sb summary rows to a top-level
            // `<output>/summary-learn.log` so the user has a single
            // growing file spanning all resumes. Per-save
            // `<output>/0NNN/learn.log` files are kept as snapshots.
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!("warning: failed to update {}: {e}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display());
            }
        }
        Err(e) => {
            eprintln!("error: failed to assemble numbered checkpoint dirs: {e}");
            std::process::exit(1);
        }
    }
}

/// CSV header for per-save `0NNN/learn.log`. The top-level
/// `<output>/summary-learn.log` uses [`SUMMARY_LEARN_LOG_HEADER`] because
/// it drops `curr_batch`. Column meanings (12 total):
///
/// - `eval`: mirror of the output-dir name (`<eval-type>[-<arch>]`)
///   plus a `/<component>` suffix for multi-component eval types. For
///   NNUE eval types (single-component) the column holds the eval-type
///   joined with the architecture, e.g.
///   `NNUE_HALFKP-NNUE_halfkp_256x2_32_32`. For
///   KPPT-family eval types (which ignore `--arch`, three components
///   trained sequentially) it holds `KPPT/kk`, `KPPT/kkp`, `KPPT/kpp`
///   (or `KPP_KKPT/kk`, etc.).
/// - `epoch`: 1-indexed epoch counter within this run (`--max-epochs`).
/// - `superbatch`: 1-indexed superbatch within the current epoch.
///   Increments every internal superbatch boundary
///   (= `--positions-per-superbatch` rounded down to whole batches).
/// - `curr_batch`: 1-indexed batch counter within the current superbatch
///   (= the `curr_batch` field bullet records every 32 batches: 32, 64,
///   96, ...). Combine with `superbatch` for "(superbatch − 1) ×
///   effective_batches_per_superbatch + curr_batch" to get the total batch count.
/// - `value_loss`: bullet's per-32-batch loss value at that point.
/// - `lr_start`: learning rate at the start of the row's interval.
/// - `lr_end`: learning rate used by the last batch in the row's interval.
/// - `lambda`: `--lambda` value at that point (constant per run), formatted
///   to three decimal places (`1.000`, `0.500`, ...).
/// - `positions`: cumulative number of teacher positions consumed so far
///   for this component, including positions from prior runs detected
///   in the existing top-level `summary-learn.log` (resume-aware).
/// - `teacher`: the user's `--teacher` CLI value verbatim, RFC-4180
///   escaped (quoted if it contains a comma / quote / newline) so a
///   directory or comma-separated list is preserved as one CSV field.
const LEARN_LOG_HEADER: &str = "eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher";

/// Schema for the top-level `<output>/summary-learn.log`. Same as
/// [`LEARN_LOG_HEADER`] but **without** the `curr_batch` column, because
/// the summary file holds only one row per superbatch (the closing
/// row), where `curr_batch` is always the last batch index of that sb
/// (= the effective superbatch boundary) and conveys no info.
const SUMMARY_LEARN_LOG_HEADER: &str = "eval,epoch,superbatch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher";

/// Filename of the top-level summary log inside `<output>/`. Per-save
/// dirs (`<output>/<NNNN>/`) keep the original per-batch `learn.log`;
/// the summary lives next to them so they don't shadow each other.
const SUMMARY_LEARN_LOG_NAME: &str = "summary-learn.log";
const PLATEAU_EPOCH_DONE_NAME: &str = "plateau_epoch_done.txt";

/// Bundle of parameters the enrichment functions need to turn bullet's
/// raw 3-column `log.txt` rows (`superbatch,curr_batch,loss`) into the
/// 12-column `learn.log` CSV rows defined by [`LEARN_LOG_HEADER`].
#[derive(Clone, Debug)]
struct LogContext {
    eval_type: &'static str,
    /// Canonical architecture name for NNUE eval types. Empty string for
    /// KPPT-family eval types since they ignore `--arch`. When non-empty it is
    /// joined into the `eval` column as `<eval-type>-<arch>`, matching the
    /// output-dir naming.
    arch: String,
    lr_start: f32,
    lambda: f32,
    batch_size: usize,
    batches_per_superbatch: usize,
    teacher_csv: String,
    /// Offset added to bullet's local epoch counter when emitting the
    /// `epoch` column in `learn.log`. Bullet's `for epoch in 1..=N`
    /// counter resets to 1 at every new `bulletou` invocation, so
    /// without an offset a continued-training run would write `epoch=1`
    /// rows after the previous run had already finished `epoch=3`.
    /// Set to: `max_epoch_in_summary_log` when starting a fresh epoch
    /// (= teacher changed / previous run completed cleanly / auto-resume
    /// crossed an epoch boundary), or `max_epoch_in_summary_log - 1`
    /// when resuming mid-epoch (so the resuming rows continue to display
    /// the *same* epoch number as the previous run's last partial save).
    /// 0 for fresh first runs.
    epoch_offset: usize,
    /// Which LR schedule the trainer is running. Switches the
    /// enrich-path lr formula between `StepLR::lr_at_positions` (step),
    /// `GeometricLR::lr_at_positions` (geometric), and
    /// `CosineLR::lr_at_positions` (cos).
    lr_schedule: LrScheduleKind,
    /// Period of one warm-restart cycle (= one epoch's worth of
    /// positions), shared by both schedules. Computed via
    /// [`auto_epoch_period`] at training start.
    lr_period: u64,
    /// Decay factor and interval used by `step`.
    lr_step_gamma: f32,
    lr_step_positions: u64,
    /// Floor LR reached at end of each cycle. Mirrors `--lr-min`.
    lr_min: f32,
    /// When set, the lr column uses this exact value. Used by
    /// ReduceLROnPlateau because its LR depends on previous validation
    /// losses, not only on position count.
    lr_override: Option<f32>,
}

/// Return `args.teacher` verbatim for the `teacher` column of `learn.log`.
///
/// Earlier this expanded a directory `--teacher` into the comma-joined
/// list of actual files used, but for a 50-file teacher dir this made
/// every row of the log absurdly wide. Just record what the user typed and let them
/// scroll back through tag.txt / shell history if they need the
/// individual filenames.
fn resolve_teacher_for_log(teacher: &str) -> String {
    teacher.to_string()
}

impl LogContext {
    /// `lr_cosine_period_override` is the cosine cycle period in
    /// positions, pre-computed by [`auto_epoch_period`] when the
    /// `lr_period_override` is the warm-restart cycle period (= one
    /// epoch's positions), pre-computed by [`auto_epoch_period`] when
    /// training is about to run. Both step and cos schedules share
    /// the same period. For non-training callers (post-training log
    /// enrich paths), pass 0; the lr column in enrich is only
    /// meaningful when we know what the trainer actually used.
    fn from_args(args: &Args, lr_period_override: u64) -> Self {
        let batches_per_superbatch = effective_batches_per_superbatch(args).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });
        let lr_step_gamma = effective_lr_step_gamma(args, batches_per_superbatch)
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(2);
            })
            .0;
        Self {
            eval_type: args.eval_type().cli_name(),
            arch: if args.eval_type().uses_arch() { args.arch().cli_name() } else { String::new() },
            lr_start: args.lr,
            lambda: args.lambda,
            batch_size: effective_batch_size(args),
            batches_per_superbatch,
            teacher_csv: csv_escape(&resolve_teacher_for_log(&args.teacher)),
            epoch_offset: 0,
            lr_schedule: args.lr_schedule,
            lr_period: lr_period_override,
            lr_step_gamma,
            lr_step_positions: effective_lr_step_positions(args, batches_per_superbatch),
            lr_min: args.lr_min,
            lr_override: None,
        }
    }

    /// Cumulative teacher positions consumed up to `(superbatch, curr_batch)`
    /// within the current epoch, plus the `position_offset` carried over
    /// from prior runs (read from the existing top-level `summary-learn.log`).
    fn positions_at(&self, superbatch: usize, curr_batch: usize, position_offset: usize) -> usize {
        position_offset + (superbatch.saturating_sub(1) * self.batches_per_superbatch + curr_batch) * self.batch_size
    }

    fn lr_at_positions(&self, positions: usize) -> f32 {
        match self.lr_schedule {
            LrScheduleKind::Step => StepLR::lr_at_positions(
                self.lr_start,
                self.lr_min,
                self.lr_step_gamma,
                self.lr_step_positions,
                self.lr_period,
                positions as u64,
            ),
            LrScheduleKind::Geometric => {
                GeometricLR::lr_at_positions(self.lr_start, self.lr_min, self.lr_period, positions as u64)
            }
            LrScheduleKind::Cos => {
                CosineLR::lr_at_positions(self.lr_start, self.lr_min, self.lr_period, positions as u64)
            }
            LrScheduleKind::Plateau => self.lr_override.unwrap_or(self.lr_start),
        }
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
/// enriched `learn.log`. When `Some`, only the LAST row of each
/// superbatch in that dir carries the metric (validation runs once per
/// save, so per-batch rows that are not at the sb boundary should not
/// claim a value they did not measure); other rows emit `-`. When the
/// caller passes `None`, every row's two metric columns are `-`.
#[derive(Clone, Copy, Debug)]
struct TestMetrics {
    accuracy: f32,
    loss: f32,
}

impl From<TestMetrics> for PlateauMetrics {
    fn from(value: TestMetrics) -> Self {
        Self { loss: value.loss, accuracy: value.accuracy }
    }
}

/// Convert bullet's raw 3-column `log.txt` text (`superbatch,curr_batch,loss`
/// per line) into the enriched 12-column CSV body (no header). The header
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
    last_superbatch_complete: bool,
) -> String {
    let mut out = String::new();
    let (test_acc_filled, test_loss_filled): (String, String) = match test_metrics {
        Some(m) => (format!("{:.6}", m.accuracy), format!("{:.6}", m.loss)),
        None => ("-".to_string(), "-".to_string()),
    };
    // Pre-parse so we can identify the last raw row of each superbatch.
    // Validation runs once per sb save, so the metric only applies to
    // the row that closes the sb; intermediate per-batch rows show `-`.
    let parsed: Vec<(usize, usize, &str)> = raw
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let parts: Vec<&str> = line.splitn(3, ',').collect();
            if parts.len() != 3 {
                return None;
            }
            let sb = parts[0].parse::<usize>().ok()?;
            let b = parts[1].parse::<usize>().ok()?;
            Some((sb, b, parts[2]))
        })
        .collect();
    let n = parsed.len();
    for (i, &(local_sb, b, train_loss)) in parsed.iter().enumerate() {
        let prev_b = if i > 0 && parsed[i - 1].0 == local_sb { parsed[i - 1].1 } else { 0 };
        let is_sb_boundary = i + 1 == n || parsed[i + 1].0 != local_sb;
        let boundary_is_complete = is_sb_boundary && (last_superbatch_complete || i + 1 < n);
        let display_b = if boundary_is_complete { ctx.batches_per_superbatch } else { b };
        let (test_acc_field, test_loss_field) =
            if is_sb_boundary { (test_acc_filled.as_str(), test_loss_filled.as_str()) } else { ("-", "-") };
        // Absolute epoch: bullet's `for epoch in 1..=max_epochs` counter
        // is local within the current run. `ctx.epoch_offset` carries
        // the cumulative completed-epoch count from previous runs so
        // continued-training rows display monotonically.
        let absolute_epoch = epoch + ctx.epoch_offset;
        // `positions` keeps using bullet's local sb because position_offset
        // already carries the cumulative count from prior runs — the
        // formula then adds (local_sb-1)*sb_size + b*batch_size to the
        // carry-over to give an honest cumulative position count.
        let positions = ctx.positions_at(local_sb, display_b, position_offset);
        let lr_start_positions = if is_sb_boundary {
            ctx.positions_at(local_sb, 0, position_offset)
        } else {
            ctx.positions_at(local_sb, prev_b, position_offset)
        };
        let lr_end_positions = positions.saturating_sub(ctx.batch_size);
        let lr_start = ctx.lr_at_positions(lr_start_positions);
        let lr_end = ctx.lr_at_positions(lr_end_positions);
        // Mirror the output-dir name (`<eval-type>[-<arch>]`) plus a
        // `/<component>` suffix for multi-component eval types (KPPT
        // family). NNUE rows are single-component so the slash is
        // omitted; KPPT-family eval types don't consume `--arch`.
        let head: std::borrow::Cow<'_, str> = if ctx.arch.is_empty() {
            std::borrow::Cow::Borrowed(ctx.eval_type)
        } else {
            std::borrow::Cow::Owned(format!("{}-{}", ctx.eval_type, ctx.arch))
        };
        let eval_field: std::borrow::Cow<'_, str> =
            if component == "nnue" { head } else { std::borrow::Cow::Owned(format!("{}/{}", head, component)) };
        // All numeric float columns are formatted at fixed 6 decimals
        // for readability. `train` arrives as bullet's raw string (e.g.
        // "0.07035521"); reparse and re-format if possible, otherwise
        // pass through unchanged so a future schema change in bullet's
        // log doesn't silently lose data.
        let train_field: std::borrow::Cow<'_, str> = match train_loss.parse::<f32>() {
            Ok(v) => std::borrow::Cow::Owned(format!("{v:.6}")),
            Err(_) => std::borrow::Cow::Borrowed(train_loss),
        };
        out.push_str(&format!(
            "{eval},{epoch},{sb},{b},{ta},{tl},{train},{lr_start:.6},{lr_end:.6},{lambda:.6},{positions},{teacher}\n",
            eval = eval_field,
            epoch = absolute_epoch,
            sb = local_sb,
            b = display_b,
            ta = test_acc_field,
            tl = test_loss_field,
            train = train_field,
            lambda = ctx.lambda,
            teacher = ctx.teacher_csv,
        ));
    }
    out
}

/// Read the existing top-level `<output>/summary-learn.log` and return the maximum
/// `positions` value seen per component. Used at the start of a run to
/// pick up the cumulative offset across resumes.
///
/// Returns an empty map if the file doesn't exist yet (= first run).
///
/// Reads the **summary** log [`SUMMARY_LEARN_LOG_NAME`] (`<output>/
/// summary-learn.log`). Schema is [`SUMMARY_LEARN_LOG_HEADER`] (11
/// columns, NO `curr_batch`):
///
///   eval, epoch, superbatch, test_value_accuracy, test_value_loss,
///   train_value_loss, lr_start, lr_end, lambda, **positions**, teacher
///
/// `positions` is at index 9 in the current schema. Older summary logs
/// used a single `lr` column and had `positions` at index 8; accept both
/// when reading offsets so users can still resume far enough to receive
/// the explicit schema-mismatch warning on append.
///
/// `splitn` keeps any commas inside
/// the trailing `teacher` field are preserved. Component is extracted
/// from the `eval` column at index 0: a slash-suffix (e.g. `KPPT/kk`)
/// names the component explicitly; absence of a slash maps to `"nnue"`.
fn read_prior_positions(top_level_log: &std::path::Path) -> std::collections::BTreeMap<String, usize> {
    let mut map = std::collections::BTreeMap::new();
    let Ok(content) = std::fs::read_to_string(top_level_log) else { return map };
    let mut positions_index = 9usize;
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("eval,") {
            positions_index = if line.contains(",lr_start,lr_end,") { 9 } else { 8 };
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(positions_index + 2, ',').collect();
        if parts.len() <= positions_index {
            continue;
        }
        let eval = parts[0];
        let component = eval.split_once('/').map(|(_, c)| c).unwrap_or("nnue");
        let Ok(positions) = parts[positions_index].parse::<usize>() else { continue };
        let entry = map.entry(component.to_string()).or_insert(0);
        if positions > *entry {
            *entry = positions;
        }
    }
    map
}

/// Read the maximum value of the `epoch` column (index 1) from the
/// top-level summary log. Used to compute `LogContext.epoch_offset` so
/// continued-training rows display monotonic epoch numbers rather than
/// resetting to 1 each bulletou invocation.
///
/// Returns `None` if the file does not exist or no row has a parseable
/// epoch — which collapses to "no previous epochs to carry forward" at
/// the call site.
fn read_latest_epoch_in_top_level_log(top_level_log: &std::path::Path) -> Option<usize> {
    let content = std::fs::read_to_string(top_level_log).ok()?;
    let mut max_epoch: Option<usize> = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("eval,") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 2 {
            continue;
        }
        let Ok(epoch) = parts[1].parse::<usize>() else { continue };
        max_epoch = Some(max_epoch.map_or(epoch, |m| m.max(epoch)));
    }
    max_epoch
}

/// Read the last parseable validation metrics from the NNUE rows in the
/// top-level summary log. For plateau resume after a cleanly completed
/// epoch, this is the previous epoch's final accepted validation state.
fn read_latest_nnue_test_metrics_in_top_level_log(top_level_log: &std::path::Path) -> Option<PlateauMetrics> {
    let content = std::fs::read_to_string(top_level_log).ok()?;
    let mut latest: Option<PlateauMetrics> = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("eval,") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(11, ',').collect();
        if parts.len() < 5 {
            continue;
        }
        let component = parts[0].split_once('/').map(|(_, c)| c).unwrap_or("nnue");
        if component != "nnue" {
            continue;
        }
        let Ok(accuracy) = parts[3].parse::<f32>() else { continue };
        let Ok(loss) = parts[4].parse::<f32>() else { continue };
        latest = Some(PlateauMetrics { loss, accuracy });
    }
    latest
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
    // 12-column rows: eval, epoch, sb, batch, test_value_accuracy,
    // test_value_loss, train_value_loss, lr_start, lr_end, lambda,
    // positions, teacher.
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

/// Read the highest-numbered `<output_dir>/<NNNN>/dataloader_pos.txt`
/// (= the dataloader's "I have processed up to this position" marker,
/// written at each save). Returns `(byte_offset, plies_within_unit)`.
///
/// Format on disk: `<byte_offset>,<plies>` (single line). For loaders
/// over fixed-length records (HCPE / PSV) the `plies` part is always
/// 0. For game-structured loaders (HCPE3 / pack), the pair points to
/// the start of a game header at `byte_offset` and the ply index
/// within that game where the next position to be expanded sits — so
/// resume seeks to the header, parses it, then fast-skips `plies`
/// MoveInfo entries before re-entering the normal expansion loop.
///
/// Backward-compatible with the legacy single-number format (= just
/// `<byte_offset>` on the line, plies inferred 0).
fn read_latest_dataloader_pos(output_dir: &std::path::Path) -> Option<(u64, usize)> {
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
    let pos_file = output_dir.join(format!("{n:04}")).join("dataloader_pos.txt");
    let content = std::fs::read_to_string(&pos_file).ok()?;
    let line = content.trim();
    if let Some((off, plies)) = line.split_once(',') {
        let off = off.trim().parse::<u64>().ok()?;
        let plies = plies.trim().parse::<usize>().ok()?;
        Some((off, plies))
    } else {
        let off = line.parse::<u64>().ok()?;
        Some((off, 0))
    }
}

/// Detect the teacher path recorded in the highest-numbered
/// `<output_dir>/<NNNN>/learn.log`. Used to decide whether auto-resume's
/// dataloader skip-ahead is safe: bullet's dataloader skips
/// `(start_sb - 1) * batches_per_sb` records at startup, which only
/// makes sense if the resume run uses the same teacher file as the
/// previous run. If the teacher changed, the new (smaller) file may
/// have fewer records than the requested skip, causing
/// `NoBatchesReceived` panic. We use the comparison result to fall back
/// to a fresh `start_sb=1` read in the changed-teacher case while still
/// honouring the model+optimizer load from `state.bin`.
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
    // Same 12-column layout as read_latest_saved_superbatch. teacher
    // is the trailing field (index 11). splitn(12, ',') keeps any
    // commas inside teacher (= comma-separated `--teacher` list)
    // as a single CSV field.
    let mut last_teacher: Option<String> = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("eval,") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(12, ',').collect();
        if parts.len() < 12 {
            continue;
        }
        last_teacher = Some(parts[11].trim().to_string());
    }
    last_teacher
}

/// Append the body of the latest save dir's `learn.log` (already enriched
/// 12-column CSV from `assemble_numbered_dirs` / `finalize_nnue_dirs`) onto
/// the top-level `<output>/summary-learn.log`, writing the CSV header on first
/// file creation. The result is a single pure CSV — no section headers,
/// no separators — that pandas / Excel can load directly.
///
/// To keep the top-level file readable as a sb-level summary (rather
/// than a per-batch dump), only the **last row** of each (eval, sb)
/// group from the per-dir log is appended — exactly one row per
/// superbatch save per component. The `curr_batch` column is also
/// dropped (the summary file uses [`SUMMARY_LEARN_LOG_HEADER`]) because
/// for sb-boundary rows it conveys no info (always the last batch
/// index of that sb). The full per-batch series is still available in
/// each `<NNNN>/learn.log`.
///
/// If the existing summary file was written by an older version of
/// `bulletou` with a different header, returns `InvalidData` so the
/// caller can alert the user to clear the old file rather than
/// silently mixing schemas.
fn append_to_top_level_log(output_dir: &std::path::Path, last_idx: usize) -> std::io::Result<()> {
    use std::io::Write;
    let latest_log = output_dir.join(format!("{last_idx:04}")).join("learn.log");
    let body = std::fs::read_to_string(&latest_log)?;
    let top = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let top_existed = top.is_file();

    // Detect schema mismatch on existing file. The summary file is
    // assumed to start with [`SUMMARY_LEARN_LOG_HEADER`] followed by
    // 11-col data rows.
    if top_existed {
        let mut head_buf = String::new();
        if let Ok(mut f) = std::fs::File::open(&top) {
            use std::io::Read as _;
            let mut buf = [0u8; 1024];
            if let Ok(n) = f.read(&mut buf) {
                if let Ok(s) = std::str::from_utf8(&buf[..n]) {
                    if let Some(first_line) = s.lines().next() {
                        head_buf = first_line.to_string();
                    }
                }
            }
        }
        if !head_buf.is_empty() && head_buf != SUMMARY_LEARN_LOG_HEADER {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{}: existing summary log header has a different schema than this build expects.\n  \
                     existing: {head_buf}\n  expected: {SUMMARY_LEARN_LOG_HEADER}\n  \
                     Rename or delete the old file (and the per-dir <NNNN>/learn.log if you want a clean restart) and re-run.",
                    top.display()
                ),
            ));
        }
    }

    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&top)?;
    // Per-dir learn.log is 12-col (includes curr_batch); strip its
    // header before filtering data rows.
    let body_no_header = body
        .strip_prefix(LEARN_LOG_HEADER)
        .and_then(|rest| rest.strip_prefix('\n').or(Some(rest)))
        .unwrap_or(body.as_str());
    if !top_existed {
        writeln!(file, "{SUMMARY_LEARN_LOG_HEADER}")?;
    }
    // Keep only the last row of each (eval, sb) group and drop the
    // `curr_batch` column (= index 3 in the 12-col per-dir layout).
    let lines: Vec<&str> = body_no_header.lines().filter(|l| !l.is_empty()).collect();
    let key_of = |line: &str| -> Option<(String, String)> {
        let parts: Vec<&str> = line.splitn(12, ',').collect();
        if parts.len() < 3 {
            return None;
        }
        Some((parts[0].to_string(), parts[2].to_string()))
    };
    let drop_curr_batch = |line: &str| -> Option<String> {
        // Split with splitn(12, ',') so the trailing `teacher` field
        // keeps any commas. Drop index 3 (`curr_batch`).
        let parts: Vec<&str> = line.splitn(12, ',').collect();
        if parts.len() < 12 {
            return None;
        }
        let mut out = String::with_capacity(line.len());
        for (i, p) in parts.iter().enumerate() {
            if i == 3 {
                continue;
            }
            if !out.is_empty() {
                out.push(',');
            }
            out.push_str(p);
        }
        Some(out)
    };
    for (i, line) in lines.iter().enumerate() {
        let Some(here) = key_of(line) else { continue };
        let next_differs = match lines.get(i + 1) {
            Some(next) => key_of(next).map(|k| k != here).unwrap_or(true),
            None => true,
        };
        if next_differs {
            let Some(summary_row) = drop_curr_batch(line) else { continue };
            file.write_all(summary_row.as_bytes())?;
            file.write_all(b"\n")?;
        }
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
    state_backend: &str,
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
        for name in ["teacher.txt", "dataloader_pos.txt"] {
            let src = kk_dir.join(name);
            if src.is_file() {
                std::fs::copy(src, dst.join(name))?;
            }
        }
        // bundle the three components' resume state (Adam weights + momentum + velocity)
        // into a single `state.bin` so the dir holds everything needed to resume.
        let mut state_buf: Vec<u8> = write_state_backend_marker(state_backend);
        bundle_component_state(&mut state_buf, "kk", &kk_dir.join("optimiser_state"))?;
        bundle_component_state(&mut state_buf, "kkp", &kkp_dir.join("optimiser_state"))?;
        bundle_component_state(&mut state_buf, "kpp", &kpp_dir.join("optimiser_state"))?;
        std::fs::write(dst.join("state.bin"), &state_buf)?;
        // Each component's bullet `log.txt` is the raw
        // `superbatch,curr_batch,loss` CSV. Enrich each into the current
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
            log_buf.push_str(&enrich_bullet_log_to_csv(&raw, ctx, epoch, label, prior, None, true));
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

        let batches_per_superbatch = effective_batches_per_superbatch(args).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
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

        // --superbatches が未指定なら epoch ごとに loader EOF まで回す (= usize::MAX で
        // 上限なし、loader 側で EOF が来たら trainer.run が返る)。
        let end_superbatch = args.superbatches.unwrap_or(usize::MAX);
        let teacher_single_epoch = args.superbatches.is_none() && !matches!(args.lr_schedule, LrScheduleKind::Plateau);

        let net_id_base = args.net_id();
        let output_dir_buf = args.output_dir();
        let yaneuraou_scale = args.yaneuraou_scale();
        let kpp_format = args.kpp_format();
        let max_epochs = effective_max_epochs(args);

        let output_dir_str = args.output_dir();
        let output_dir = output_dir_str.to_str().unwrap_or("checkpoints");

        // Warm-restart cycle period = one epoch's positions. `step`,
        // `geometric`, and `cos` use this period; `plateau` is validation-driven.
        let lr_period: u64 = if matches!(args.lr_schedule, LrScheduleKind::Plateau)
            || (matches!(args.lr_schedule, LrScheduleKind::Step) && args.superbatches.is_none())
        {
            0
        } else {
            auto_epoch_period(args, format, &data_files_owned, batches_per_superbatch).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(2);
            })
        };
        let lr_step_positions = effective_lr_step_positions(args, batches_per_superbatch);
        let lr_step_gamma = effective_lr_step_gamma(args, batches_per_superbatch)
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(2);
            })
            .0;

        // Tracks whether bullet fired the save callback at least once across
        // all epochs. If 教師 is smaller than a single superbatch (or any
        // other reason no superbatch boundary is crossed), bullet writes no
        // checkpoint at all and we'd end up with an empty output dir. After
        // all epochs finish we check this flag and, if no save happened, do
        // a final fallback save so at least the current trainer state is
        // persisted. This is *not* an EOF-triggered save — it fires exactly
        // once per training run and only as a last resort.
        let saved_any = std::cell::Cell::new(false);
        let trained_without_save = std::cell::Cell::new(false);
        // Remember the last per-epoch net_id we used so the fallback save can
        // reuse the same naming convention (so assembly pairs the dirs by
        // sort order alongside any future numbered checkpoints).
        let mut last_net_id_for_epoch: String = net_id_base.clone();
        // The error_record returned by the most recent `trainer.run` call.
        // bullet writes `log.txt` itself at each save, but if zero saves
        // happened we need to write it ourselves in the fallback path.
        let mut last_error_record: Vec<(usize, usize, f32)> = Vec::new();

        // First-time CUDA setup heads-up: see comment in the NNUE macro
        // for the full story. Same warning applies to KPPT training.
        eprintln!(
            "  note: first-batch run will trigger CUDA JIT-compile of GPU kernels\n  \
             (≈30–60s on first run for this GPU model; cached afterwards). \
             If both CPU and\n  GPU look idle for a minute, that is expected — \
             the NVIDIA driver is compiling."
        );

        // KPPT macro: persistent HCPE loader hoisting is NOT applied here
        // (= no cb_dataloader_resume_offset in scope, this code path doesn't
        // wire the saved-pointer resume). KPPT-with-HCPE is rare; the per-chunk
        // loader creation remains.
        let persistent_hcpe_loader: Option<HcpeDataLoader<fn(&bulletou_lib::shogi::PackedSfenValue) -> bool>> = None;

        for epoch in 1..=max_epochs {
            print_epoch_banner(epoch, max_epochs);

            // checkpoint dir 名は max_epochs=1 のとき従来通り `<net_id>-<superbatch>`、
            // 複数 epoch のときは `<net_id>-e<epoch>-<superbatch>` で重複を避ける。
            let net_id_for_epoch = if max_epochs > 1 { format!("{net_id_base}-e{epoch}") } else { net_id_base.clone() };
            last_net_id_for_epoch = net_id_for_epoch.clone();

            let schedule = TrainingSchedule {
                net_id: net_id_for_epoch.clone(),
                eval_scale: args.scale as f32,
                steps: TrainingSteps {
                    batch_size: effective_batch_size(args),
                    batches_per_superbatch,
                    // KPPT family does not get auto-resume yet (the 3-component
                    // assembly makes the bookkeeping non-trivial); always starts at sb=1.
                    start_superbatch: 1,
                    end_superbatch,
                },
                wdl_scheduler: wdl::ConstantWDL { value: 1.0 - args.lambda },
                // KPPT family uses positions-based LR for consistency
                // with the NNUE family, but does not currently track
                // cumulative `prior_positions` across resumes (KPPT
                // resume support is itself limited), so prior is 0.
                lr_scheduler: match args.lr_schedule {
                    LrScheduleKind::Step => LrSchedulerImpl::Step(StepLR {
                        start: args.lr,
                        min: args.lr_min,
                        gamma: lr_step_gamma,
                        step_positions: lr_step_positions,
                        period_positions: lr_period,
                        prior_positions: 0,
                        batch_size: effective_batch_size(args),
                        batches_per_superbatch,
                    }),
                    LrScheduleKind::Geometric => LrSchedulerImpl::Geometric(GeometricLR {
                        start: args.lr,
                        min: args.lr_min,
                        period_positions: lr_period,
                        prior_positions: 0,
                        batch_size: effective_batch_size(args),
                        batches_per_superbatch,
                    }),
                    LrScheduleKind::Cos => LrSchedulerImpl::Cos(CosineLR {
                        start: args.lr,
                        min: args.lr_min,
                        period_positions: lr_period,
                        prior_positions: 0,
                        batch_size: effective_batch_size(args),
                        batches_per_superbatch,
                    }),
                    LrScheduleKind::Plateau => LrSchedulerImpl::Fixed(FixedLR { value: args.lr }),
                },
                save_rate: effective_save_rate(args),
                save_epoch_end: effective_save_epoch_end(args),
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
                    // KPPT path: chunk-local loader (no persistent producer
                    // hoisting here; the macro lacks the resume-offset wiring
                    // that the NNUE macro has).
                    let _ = &persistent_hcpe_loader; // suppress unused
                    let loader = HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                        .with_loader_threads(args.loader_threads)
                        .with_single_epoch(teacher_single_epoch);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Hcpe3 => {
                    let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                        .with_single_epoch(teacher_single_epoch);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Pack => {
                    let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                        .with_single_epoch(teacher_single_epoch);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Psv => {
                    let loader =
                        DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(teacher_single_epoch);
                    trainer.run(&schedule, &settings, &loader)
                }
            };
            if !effective_save_epoch_end(args)
                && end_superbatch != usize::MAX
                && !end_superbatch.is_multiple_of(effective_save_rate(args).max(1))
                && error_record_completed_superbatch(&last_error_record, end_superbatch, batches_per_superbatch)
            {
                trained_without_save.set(true);
                eprintln!(
                    "  checkpoint skipped at epoch={}, superbatch={} (--no-save-epoch-end)",
                    epoch, end_superbatch
                );
            }
        }

        // End-of-training fallback save (see the comment on `saved_any`):
        // executes only when bullet never crossed a superbatch boundary,
        // except when the run intentionally ended on a train-only tail via
        // `--no-save-epoch-end`.
        if !saved_any.get() && !trained_without_save.get() {
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

fn error_record_completed_superbatch(
    records: &[(usize, usize, f32)],
    superbatch: usize,
    batches_per_superbatch: usize,
) -> bool {
    records.iter().any(|(sb, batch, _)| *sb == superbatch && *batch == batches_per_superbatch)
}

// ----- KPPT: KK ---------------------------------------------------------

fn run_kppt_kk(args: &Args, resume_dir: Option<&std::path::Path>) {
    match args.optimizer {
        OptimizerKind::Adamw => run_kppt_kk_impl::<optimiser::AdamW>(args, resume_dir),
        OptimizerKind::Radam => run_kppt_kk_impl::<optimiser::RAdam>(args, resume_dir),
        OptimizerKind::Ranger => run_kppt_kk_impl::<optimiser::Ranger>(args, resume_dir),
    }
}

fn run_kppt_kk_impl<O: BulletouOptimizer>(args: &Args, resume_dir: Option<&std::path::Path>) {
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
        .optimiser(O::default())
        .inputs(ShogiKk)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &[], &[]);

    if let Some(dir) = resume_dir {
        eprintln!("  [KK] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT: KKP --------------------------------------------------------

fn run_kppt_kkp(args: &Args, resume_dir: Option<&std::path::Path>) {
    match args.optimizer {
        OptimizerKind::Adamw => run_kppt_kkp_impl::<optimiser::AdamW>(args, resume_dir),
        OptimizerKind::Radam => run_kppt_kkp_impl::<optimiser::RAdam>(args, resume_dir),
        OptimizerKind::Ranger => run_kppt_kkp_impl::<optimiser::Ranger>(args, resume_dir),
    }
}

fn run_kppt_kkp_impl<O: BulletouOptimizer>(args: &Args, resume_dir: Option<&std::path::Path>) {
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
        .optimiser(O::default())
        .inputs(ShogiKkp)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &[], &[]);

    if let Some(dir) = resume_dir {
        eprintln!("  [KKP] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT: KPP --------------------------------------------------------

fn run_kppt_kpp(args: &Args, resume_dir: Option<&std::path::Path>) {
    match args.optimizer {
        OptimizerKind::Adamw => run_kppt_kpp_impl::<optimiser::AdamW>(args, resume_dir),
        OptimizerKind::Radam => run_kppt_kpp_impl::<optimiser::RAdam>(args, resume_dir),
        OptimizerKind::Ranger => run_kppt_kpp_impl::<optimiser::Ranger>(args, resume_dir),
    }
}

fn run_kppt_kpp_impl<O: BulletouOptimizer>(args: &Args, resume_dir: Option<&std::path::Path>) {
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
        .optimiser(O::default())
        .inputs(ShogiKpp)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &[], &[]);

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
    let mut state_buf: Vec<u8> = write_state_backend_marker(STATE_BACKEND_BULLET);
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
            "  loading {} test positions from {} (sample={}, seed={}) for validation...",
            args.test_positions,
            path,
            args.test_sample.cli_name(),
            if args.test_sample == TestSampleMode::Random {
                args.test_seed.to_string()
            } else {
                "-".to_string()
            }
        );
        let loaded = match args.test_sample {
            TestSampleMode::Random => read_random_teacher_positions(&path, args.test_positions, args.test_seed),
            TestSampleMode::Sequential => read_teacher_positions_prefix(&path, args.test_positions),
        };
        match loaded {
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

#[derive(Clone, Copy)]
struct SfnnActivationShape {
    ft_size: usize,
    l1_hidden: usize,
    l1_out: usize,
    l2_in: usize,
    l2_size: usize,
    num_stacks: usize,
}

#[derive(Clone, Copy)]
struct RunningActivationStats {
    count: usize,
    zero: usize,
    maxed: usize,
    sum: f64,
    sum_sq: f64,
    min: f32,
    max: f32,
}

impl Default for RunningActivationStats {
    fn default() -> Self {
        Self { count: 0, zero: 0, maxed: 0, sum: 0.0, sum_sq: 0.0, min: f32::INFINITY, max: f32::NEG_INFINITY }
    }
}

impl RunningActivationStats {
    fn observe(&mut self, value: f32) {
        const EPS: f32 = 1.0e-6;
        self.count += 1;
        if value <= EPS {
            self.zero += 1;
        }
        if value >= 1.0 - EPS {
            self.maxed += 1;
        }
        self.sum += f64::from(value);
        self.sum_sq += f64::from(value) * f64::from(value);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }

    fn variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_sq / self.count as f64 - mean * mean).max(0.0)
        }
    }

    fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn zero_ratio(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.zero as f64 / self.count as f64 }
    }

    fn max_ratio(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.maxed as f64 / self.count as f64 }
    }
}

struct LayerActivationStats {
    label: &'static str,
    all: RunningActivationStats,
    by_neuron: Vec<RunningActivationStats>,
}

impl LayerActivationStats {
    fn new(label: &'static str, neurons: usize) -> Self {
        Self {
            label,
            all: RunningActivationStats::default(),
            by_neuron: vec![RunningActivationStats::default(); neurons],
        }
    }

    fn observe(&mut self, neuron: usize, value: f32) {
        self.all.observe(value);
        self.by_neuron[neuron].observe(value);
    }

    fn print(&self) {
        let flat = self.by_neuron.iter().filter(|s| s.count > 0 && s.variance() < 1.0e-10).count();
        let max_stuck = self.by_neuron.iter().filter(|s| s.count > 0 && s.max_ratio() > 0.999).count();
        let zero_stuck = self.by_neuron.iter().filter(|s| s.count > 0 && s.zero_ratio() > 0.999).count();
        eprintln!(
            "    {:<16} values={} mean={:.6} std={:.6} min={:.6} max={:.6} zero={:.3}% maxclip={:.3}% flat_neurons={} zero_stuck={} max_stuck={}",
            self.label,
            self.all.count,
            self.all.mean(),
            self.all.stddev(),
            self.all.min,
            self.all.max,
            self.all.zero_ratio() * 100.0,
            self.all.max_ratio() * 100.0,
            flat,
            zero_stuck,
            max_stuck,
        );
    }
}

fn model_weight_f32<Opt, I>(trainer: &ValueTrainer<Opt, I, ShogiLayerStackBucket9>, id: &str) -> Option<Vec<f32>>
where
    Opt: bullet_trainer::optimiser::OptimiserState<ExecutionContext>,
    I: SparseInputType<RequiredDataType = bulletou_lib::shogi::PackedSfenValue>,
{
    match trainer.optimiser.model.get_weights(id) {
        Some(TValue::F32(values)) => Some(values),
        Some(_) => {
            eprintln!("  WARN: activation stats skipped: weight `{id}` is not f32");
            None
        }
        None => {
            eprintln!("  WARN: activation stats skipped: missing weight `{id}`");
            None
        }
    }
}

fn affine_sparse(weights: &[f32], bias: &[f32], rows: usize, active: &[usize], out: &mut [f32]) {
    out.copy_from_slice(&bias[..rows]);
    for &feature in active {
        let base = feature * rows;
        for row in 0..rows {
            out[row] += weights[base + row];
        }
    }
}

fn affine_stacked_selected(
    weights: &[f32],
    bias: &[f32],
    input: &[f32],
    output_size_per_bucket: usize,
    buckets: usize,
    bucket: usize,
    out: &mut [f32],
) {
    let rows = output_size_per_bucket * buckets;
    let row_base = bucket * output_size_per_bucket;
    out.copy_from_slice(&bias[row_base..row_base + output_size_per_bucket]);
    for (input_idx, &x) in input.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        let base = input_idx * rows + row_base;
        for row in 0..output_size_per_bucket {
            out[row] += weights[base + row] * x;
        }
    }
}

fn affine_add_dense(weights: &[f32], bias: &[f32], input: &[f32], rows: usize, out: &mut [f32]) {
    for row in 0..rows {
        out[row] += bias[row];
    }
    for (input_idx, &x) in input.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        let base = input_idx * rows;
        for row in 0..rows {
            out[row] += weights[base + row] * x;
        }
    }
}

fn observe_crelu(layer: &mut LayerActivationStats, values: &mut [f32]) {
    for (idx, value) in values.iter_mut().enumerate() {
        *value = value.clamp(0.0, 1.0);
        layer.observe(idx, *value);
    }
}

fn pairwise_mul_scaled(input: &[f32], output: &mut [f32]) {
    let half = input.len() / 2;
    debug_assert_eq!(output.len(), half);
    for idx in 0..half {
        output[idx] = input[idx] * input[idx + half] * (127.0 / 128.0);
    }
}

fn dump_sfnn_activation_stats<Opt, I>(
    args: &Args,
    trainer: &ValueTrainer<Opt, I, ShogiLayerStackBucket9>,
    input: I,
    bucket_impl: ShogiLayerStackBucket9,
    cache: &TestPositionsCache,
    shape: SfnnActivationShape,
) where
    Opt: bullet_trainer::optimiser::OptimiserState<ExecutionContext>,
    I: SparseInputType<RequiredDataType = bulletou_lib::shogi::PackedSfenValue> + Copy,
{
    let sample_size = cache.positions.len().min(args.activation_stats_positions);
    if sample_size == 0 {
        return;
    }

    let Some(l0w) = model_weight_f32(trainer, "l0w") else { return };
    let Some(l0b) = model_weight_f32(trainer, "l0b") else { return };
    let Some(l1w) = model_weight_f32(trainer, "l1w") else { return };
    let Some(l1b) = model_weight_f32(trainer, "l1b") else { return };
    let (l1fw, l1fb) = if args.sfnn_factorized_l1 {
        let Some(l1fw) = model_weight_f32(trainer, "l1fw") else { return };
        let Some(l1fb) = model_weight_f32(trainer, "l1fb") else { return };
        (Some(l1fw), Some(l1fb))
    } else {
        (None, None)
    };
    let Some(l2w) = model_weight_f32(trainer, "l2w") else { return };
    let Some(l2b) = model_weight_f32(trainer, "l2b") else { return };

    let mut l0_stats = LayerActivationStats::new("l0.crelu", shape.ft_size);
    let mut l1_stats = LayerActivationStats::new("l1.to_l2.crelu", shape.l2_in);
    let mut l2_stats = LayerActivationStats::new("l2.crelu", shape.l2_size);

    let mut stm_active = Vec::with_capacity(input.max_active());
    let mut ntm_active = Vec::with_capacity(input.max_active());
    let mut stm_l0 = vec![0.0; shape.ft_size];
    let mut ntm_l0 = vec![0.0; shape.ft_size];
    let mut stm_pair = vec![0.0; shape.ft_size / 2];
    let mut ntm_pair = vec![0.0; shape.ft_size / 2];
    let mut combined = vec![0.0; shape.ft_size];
    let mut l1_out = vec![0.0; shape.l1_out];
    let mut l2_input = vec![0.0; shape.l2_in];
    let mut l2_out = vec![0.0; shape.l2_size];

    for pos in cache.positions.iter().take(sample_size) {
        stm_active.clear();
        ntm_active.clear();
        input.map_features_split(pos, |stm, ntm| {
            if let Some(idx) = stm {
                stm_active.push(idx);
            }
            if let Some(idx) = ntm {
                ntm_active.push(idx);
            }
        });

        affine_sparse(&l0w, &l0b, shape.ft_size, &stm_active, &mut stm_l0);
        affine_sparse(&l0w, &l0b, shape.ft_size, &ntm_active, &mut ntm_l0);
        observe_crelu(&mut l0_stats, &mut stm_l0);
        observe_crelu(&mut l0_stats, &mut ntm_l0);

        pairwise_mul_scaled(&stm_l0, &mut stm_pair);
        pairwise_mul_scaled(&ntm_l0, &mut ntm_pair);
        combined[..shape.ft_size / 2].copy_from_slice(&stm_pair);
        combined[shape.ft_size / 2..].copy_from_slice(&ntm_pair);

        let bucket = bucket_impl.bucket(pos) as usize;
        affine_stacked_selected(&l1w, &l1b, &combined, shape.l1_out, shape.num_stacks, bucket, &mut l1_out);
        if let (Some(l1fw), Some(l1fb)) = (&l1fw, &l1fb) {
            affine_add_dense(l1fw, l1fb, &combined, shape.l1_out, &mut l1_out);
        }

        for idx in 0..shape.l1_hidden {
            let x = l1_out[idx];
            l2_input[idx] = x.abs().powi(2) * (127.0 / 128.0);
            l2_input[shape.l1_hidden + idx] = x;
        }
        observe_crelu(&mut l1_stats, &mut l2_input);

        affine_stacked_selected(&l2w, &l2b, &l2_input, shape.l2_size, shape.num_stacks, bucket, &mut l2_out);
        observe_crelu(&mut l2_stats, &mut l2_out);
    }

    eprintln!("  activation stats: sample={} / test_positions={}", sample_size, cache.positions.len());
    l0_stats.print();
    l1_stats.print();
    l2_stats.print();
}

/// Run validation on the cached test positions and produce per-save
/// `TestMetrics`. Caller must already hold `&mut trainer` (= called
/// outside `trainer.run`).
fn run_one_test_pass(cache: &TestPositionsCache, args: &Args, trainer_outputs: Vec<f32>) -> TestMetrics {
    let cap = if args.score_drop_abs > 0 { Some(args.score_drop_abs) } else { None };
    let report = compute_sign_accuracy_with_loss(
        &trainer_outputs,
        &cache.teacher_scores,
        &cache.teacher_results,
        cap,
        args.lambda,
        args.scale as f32,
        1.0,
        validation_loss_kind(args),
    );
    let accuracy = if report.compared == 0 { f32::NAN } else { report.accuracy() };
    let loss = report.test_loss.unwrap_or(f32::NAN);
    eprintln!(
        "  test: accuracy={:.4}% ({}/{} decisive; draws={} excluded; mate={} filtered), loss={:.6} (n={})",
        accuracy * 100.0,
        report.sign_matches,
        report.compared,
        report.drawn_games,
        report.filtered_by_score_cap,
        loss,
        report.loss_sampled,
    );
    TestMetrics { accuracy, loss }
}

/// returned by `trainer.run`), but with NNUE-specific save handling:
/// the per-save callback converts each bullet save dir to the
/// `nn.bin` + `state.bin` (+ `log.txt`) layout via
/// `convert_save_dir_to_nnue_layout`.
macro_rules! run_training_inline_nnue {
    ($args:expr, $trainer:expr) => {{
        run_training_inline_nnue!(@impl none, $args, $trainer);
    }};
    ($args:expr, $trainer:expr, $activation_stats:expr) => {{
        run_training_inline_nnue!(@impl some($activation_stats), $args, $trainer);
    }};
    (@dump none, $trainer:ident, $cache:ident) => {{
        let _ = $cache;
    }};
    (@dump some($activation_stats:expr), $trainer:ident, $cache:ident) => {{
        $activation_stats(&*$trainer, $cache);
    }};
    (@impl $stats_mode:ident $(($activation_stats:expr))?, $args:expr, $trainer:expr) => {{
        let args: &Args = $args;
        let trainer = $trainer;

        let batches_per_superbatch = effective_batches_per_superbatch(args).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
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

        let end_superbatch = args.superbatches.unwrap_or(usize::MAX);
        let net_id_base = args.net_id();
        let output_dir_buf = args.output_dir();
        let max_epochs = effective_max_epochs(args);
        // `--superbatches` が未指定の非 plateau だけは、従来通り「教師 1 周
        // = epoch」として EOF で epoch を終える。`--superbatches N` 指定時は
        // epoch は LR/validation の周期であり、教師 stream は EOF で先頭へ
        // cyclic に戻って epoch を跨いでも継続する。
        let teacher_single_epoch =
            args.superbatches.is_none() && !matches!(args.lr_schedule, LrScheduleKind::Plateau);

        let output_dir_str = args.output_dir();
        let output_dir = output_dir_str.to_str().unwrap_or("checkpoints");

        // Warm-restart cycle period = one epoch's positions. `step`,
        // `geometric`, and `cos` use this period; `plateau` is validation-driven.
        let lr_period: u64 = if matches!(args.lr_schedule, LrScheduleKind::Plateau)
            || (matches!(args.lr_schedule, LrScheduleKind::Step) && args.superbatches.is_none())
        {
            0
        } else {
            auto_epoch_period(args, format, &data_files_owned, batches_per_superbatch).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(2);
            })
        };
        let lr_step_positions = effective_lr_step_positions(args, batches_per_superbatch);
        let lr_step_gamma = effective_lr_step_gamma(args, batches_per_superbatch).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        }).0;

        let saved_any = std::cell::Cell::new(false);
        let trained_without_save = std::cell::Cell::new(false);
        let mut last_net_id_for_epoch: String = net_id_base.clone();
        let mut last_error_record: Vec<(usize, usize, f32)> = Vec::new();

        // Auto-resume: if a previous run left numbered checkpoint dirs
        // under output_dir, continue the sb counter (and therefore the LR
        // schedule) from the last saved superbatch + 1 instead of silently
        // restarting at sb=1.
        let mut cb_ctx = LogContext::from_args(args, lr_period);
        let cb_top_level_log = output_dir_buf.join(SUMMARY_LEARN_LOG_NAME);
        let resume_enabled = resume_enabled(args, &output_dir_buf);
        let auto_resume_sb_raw = if resume_enabled {
            read_latest_saved_superbatch(&output_dir_buf)
        } else {
            None
        };
        // Teacher-change detection: bullet's dataloader skips
        // `(start_sb - 1) * batches_per_sb` records at startup, which
        // assumes the resume run uses the same teacher data. If the
        // teacher path changed (e.g. the user is doing yane-distill
        // round-2 training on a new chunk), the new file may not have
        // enough records to reach the skip target → trainer.run hits
        // EOF before any batch is delivered → `NoBatchesReceived`
        // panic.
        //
        // Resolution: tell bullet `start_superbatch=1` so the dataloader
        // reads the new file from the beginning. The LR schedule is
        // already positions-based and uses `prior_positions` (carried
        // from the top-level log) for continuity, so it survives the sb
        // reset on its own.
        // Compare the *resolved* (= expanded file list) form on both
        // sides — `from_args` writes the resolved list to the log, so
        // raw `args.teacher` would never match a stored directory.
        let prev_teacher = if resume_enabled {
            read_latest_saved_teacher(&output_dir_buf)
        } else {
            None
        };
        let resolved_now = resolve_teacher_for_log(&args.teacher);
        let teacher_changed = match prev_teacher.as_deref() {
            Some(prev) => prev.trim() != resolved_now.trim(),
            None => false,
        };
        // "Previous run cleanly completed its --superbatches target": the
        // last saved sb (per-epoch counter) already reached end_superbatch.
        // Treat the new invocation as "add more epochs" — start the next
        // epoch from sb=1 with a fresh dataloader, instead of trying to
        // resume sb=last_sb+1 (which would skip past end_superbatch and
        // train zero batches).
        let prev_run_completed_epoch = if matches!(args.lr_schedule, LrScheduleKind::Plateau)
            && latest_checkpoint_has_file(&output_dir_buf, PLATEAU_EPOCH_DONE_NAME)
        {
            true
        } else {
            auto_resume_sb_raw.map(|last_sb| last_sb >= end_superbatch).unwrap_or(false)
        };
        // Displayed sb in `learn.log` is intrinsically per-epoch (= 1..N
        // each epoch), so no cross-run offset is applied. Only the
        // dataloader's bullet-internal start_sb differs by case below.
        let effective_start_superbatch = if teacher_changed {
            // Teacher changed: dataloader fresh (start_sb=1) so the
            // skip-ahead doesn't run past EOF of the new file.
            1usize
        } else if prev_run_completed_epoch {
            // Same teacher, previous run completed its epoch(s) cleanly:
            // start the next epoch from sb=1.
            1usize
        } else if let Some(last_sb) = auto_resume_sb_raw {
            // Same teacher, previous run died mid-epoch: let bullet's
            // dataloader skip ahead so this epoch finishes from
            // last_sb+1. Epoch≥2 will reset to sb=1 in the chunk loop.
            last_sb + 1
        } else {
            // First run.
            1usize
        };
        // Epoch display offset for continued-training runs. Bullet's
        // per-invocation `for epoch in 1..=max_epochs` resets to 1, so
        // without this offset the second run after "3 epochs done"
        // would write `epoch=1` over the top of `epoch=3` rows.
        //
        // - Fresh first run (no top-level log): offset = 0.
        // - Clean continuation (previous run finished its last epoch,
        //   teacher unchanged): offset = max_epoch (so new local
        //   epoch=1 displays as max_epoch+1).
        // - Teacher changed (new epoch against new data): same as
        //   clean continuation — offset = max_epoch.
        // - Mid-epoch resume (previous run died inside epoch K, this
        //   run finishes that same epoch K from last_sb+1): offset
        //   = max_epoch - 1, so this run's local epoch=1 displays as
        //   K (same as the previous partial save), and its local
        //   epoch=2 displays as K+1.
        let max_epoch_in_log = if resume_enabled {
            read_latest_epoch_in_top_level_log(&cb_top_level_log).unwrap_or(0)
        } else {
            0
        };
        let mid_epoch_resume = !teacher_changed && !prev_run_completed_epoch && auto_resume_sb_raw.is_some();
        cb_ctx.epoch_offset = if mid_epoch_resume { max_epoch_in_log.saturating_sub(1) } else { max_epoch_in_log };
        // The LR columns are derived from `positions` (= prior + per-row
        // offset within this run), so we always need the cumulative
        // carry-over from the existing top-level log here so the enrich
        // path's `positions` matches what `GeometricLR` saw.
        // この変数は `cb_top_level_log` (= `<output>/summary-learn.log`) の最大
        // positions 値を保持する。enrich path で「この run の sb 1 が始まる
        // 前までに何局面学習されたか」を表し、各 save 行の positions 列に
        // `prior + (sb-1)*sb_size + b*batch_size` で書き込まれる。
        //
        // bullet の sb counter は trainer.run 呼び出し毎にリセットされる
        // (= 各 epoch / 各 chunk で 1 から始まる) ので、epoch を跨ぐと
        // positions も sb counter と一緒にリセットされてしまう。これを
        // 防ぐため、各 epoch 開始時にこの値を learn.log から再読み込み
        // して累積を反映させる。`mut` で更新される。
        let mut cb_prior_position = if resume_enabled {
            read_prior_positions(&cb_top_level_log).get("nnue").copied().unwrap_or(0)
        } else {
            0
        };
        let cb_next_idx = std::cell::Cell::new(count_existing_numbered_dirs(&output_dir_buf) + 1);
        // HCPE データローダー再開用 byte offset。最新の checkpoint dir に
        // `dataloader_pos.txt` があればそこから読み、なければ 0 (= 先頭から)。
        // 教師が変わった場合は 0 にリセット (新教師の先頭から)。
        // (byte_offset, plies_within_unit) のペアで保持する。HCPE / PSV
        // (固定長レコード) では plies は常に 0。HCPE3 / pack (棋譜単位の
        // 可変長) では plies は「現在の game header から何手分進んだ位置か」。
        let latest_dataloader_pos = if resume_enabled {
            read_latest_dataloader_pos(&output_dir_buf)
        } else {
            None
        };
        let resume_teacher_from_latest_pos = !teacher_changed
            && latest_dataloader_pos.is_some()
            && !(prev_run_completed_epoch && teacher_single_epoch);
        let initial_dataloader_resume_exact = resume_teacher_from_latest_pos;
        let cb_dataloader_resume_offset = std::cell::Cell::new(if resume_teacher_from_latest_pos {
            latest_dataloader_pos.unwrap_or((0, 0))
        } else {
            // Teacher-changed runs and the old omitted-`--superbatches`
            // non-plateau mode start a new epoch as a new teacher pass.
            // With explicit `--superbatches N`, epoch is only an
            // LR/validation cycle, so clean continuations use the latest
            // dataloader position above instead of rewinding the teacher.
            (0u64, 0usize)
        });
        let cb_dataloader_resume_exact = std::cell::Cell::new(initial_dataloader_resume_exact);

        if teacher_changed {
            if let Some(prev) = prev_teacher.as_deref() {
                eprintln!(
                    "  teacher path differs from previous run\n    previous: {prev}\n    current:  {}\n  \
                     dataloader will read the new file from the beginning; new epochs start at sb=1.\n  \
                     model + optimiser are loaded from the latest state.bin as usual.",
                    args.teacher,
                );
            }
        } else if prev_run_completed_epoch {
            let sb_text = auto_resume_sb_raw
                .map(|last_sb| format!("{} superbatch{}", last_sb, if last_sb == 1 { "" } else { "es" }))
                .unwrap_or_else(|| "a plateau epoch".to_string());
            if resume_teacher_from_latest_pos {
                let (resume_off, resume_plies) = cb_dataloader_resume_offset.get();
                eprintln!(
                    "  previous run completed {sb_text} cleanly; \
                     continuing as additional epoch(s) from sb=1. \
                     The teacher stream continues from dataloader offset {resume_off}, plies {resume_plies}."
                );
            } else {
                eprintln!(
                    "  previous run completed {sb_text} cleanly; \
                     continuing as additional epoch(s) from sb=1, reading the teacher from the beginning."
                );
            }
        } else if let Some(last_sb) = auto_resume_sb_raw {
            eprintln!("  auto-resuming from superbatch {} (last saved: {}).", effective_start_superbatch, last_sb);
        }

        // Build the LR scheduler. Positions-based for both kinds:
        // `prior_positions` carries the cumulative-trained count from
        // the existing top-level log so the schedule stays continuous
        // across resumes. Bullet's local sb resets per `trainer.run`
        // call (= per epoch / chunk), but the cumulative position offset
        // keeps `step` continuous. `geometric` / `cos` still use their
        // configured cycle period.
        // Note: lr_scheduler_for_run は each-epoch で再構築する
        // (`for epoch in 1..=max_epochs` 内)。`cb_prior_position` が epoch
        // を跨いで前 epoch の累積を反映するため、各 epoch で正しい LR
        // を計算できる。

        // Per-save validation cache: load test positions ONCE up front
        // (random-pick happens here), then reuse the same positions for
        // every save event. `None` if --test-teacher unset or load failed.
        let test_cache = TestPositionsCache::try_load(args);
        if matches!(args.lr_schedule, LrScheduleKind::Plateau) && test_cache.is_none() {
            eprintln!("error: --lr-schedule plateau requires a readable --test-teacher.");
            std::process::exit(2);
        }
        if args.dump_activation_stats {
            if let Some(cache) = test_cache.as_ref() {
                run_training_inline_nnue!(@dump $stats_mode $(($activation_stats))?, trainer, cache);
            } else {
                eprintln!("  WARN: --dump-activation-stats requires a readable --test-teacher; activation stats disabled");
            }
        }

        // Per-save incremental finalize: rename `<net_id>-<sb>` →
        // `<NNNN>/`, generate per-dir `learn.log` (with per-save test
        // metrics), append to top-level `summary-learn.log`. Done OUTSIDE
        // `trainer.run` (= once per save chunk) so we can call
        // `trainer.eval_packed_batch` for validation between chunks.
        // Killing training mid-run still leaves a clean numbered layout
        // and a resumable top-level log.

        // First-time CUDA setup heads-up: bullet's display ends with
        // "Training on <GPU>" and the very next thing is the first batch
        // run, which triggers cuBLAS handle init + JIT-compilation of the
        // PTX kernels into SASS for the local GPU's compute capability.
        // On a fresh machine / fresh `~/.nv/ComputeCache` this can take
        // ~30–60s during which both CPU and GPU look idle — the work is
        // happening inside the NVIDIA driver and is not always visible
        // to Task Manager / nvidia-smi. Subsequent invocations reuse the
        // cached SASS and skip this delay. We can't progress-report on
        // the driver-side work itself, but at least telling the user it
        // is expected.
        eprintln!(
            "  note: first-batch run will trigger CUDA JIT-compile of GPU kernels\n  \
             (≈30–60s on first run for this GPU model; cached afterwards). \
             If both CPU and\n  GPU look idle for a minute, that is expected — \
             the NVIDIA driver is compiling."
        );

        // HCPE 専用: persistent loader。Producer thread を chunk loop の外で
        // 立ち上げ、複数 trainer.run 呼び出しを跨いで pre-fetch を継続する。
        // teacher-single-epoch mode では各 epoch 開始時に新しい loader を
        // spawn し直す。明示 `--superbatches N` では epoch は LR/validation
        // cycle なので、同じ loader を持ち越して教師ストリームを継続する。
        let hcpe_loader_buffer_records = effective_save_rate(args)
            .max(1)
            .min(end_superbatch)
            .saturating_mul(batches_per_superbatch)
            .saturating_mul(effective_batch_size(args));
        let new_hcpe_loader =
            |resume_off: u64,
             resume_exact: bool|
             -> Option<HcpeDataLoader<fn(&bulletou_lib::shogi::PackedSfenValue) -> bool>> {
                if matches!(format, DataFormat::Hcpe) {
                    let loader = HcpeDataLoader::new_concat_multiple(
                        &data_files_ref,
                        args.buffer_mb,
                        (|_| true) as fn(&bulletou_lib::shogi::PackedSfenValue) -> bool,
                    )
                    .with_buffer_records(hcpe_loader_buffer_records)
                    .with_loader_threads(args.loader_threads)
                    .with_single_epoch(teacher_single_epoch);
                    Some(if resume_exact {
                        loader.with_exact_resume_offset(resume_off)
                    } else {
                        loader.with_resume_offset(resume_off)
                    })
                } else {
                    None
                }
            };
        let mut persistent_hcpe_loader =
            new_hcpe_loader(cb_dataloader_resume_offset.get().0, cb_dataloader_resume_exact.get());
        let mut previous_epoch_final_metrics = if prev_run_completed_epoch && test_cache.is_some() {
            read_latest_nnue_test_metrics_in_top_level_log(&cb_top_level_log)
        } else {
            None
        };
        if let Some(metrics) = previous_epoch_final_metrics {
            eprintln!(
                "  previous completed epoch final validation metrics = loss {:.6}, accuracy {:.6}",
                metrics.loss, metrics.accuracy
            );
        }
        if !matches!(args.lr_schedule, LrScheduleKind::Plateau) && max_epochs > 1 && test_cache.is_none() {
            let fallback = if max_epochs == usize::MAX {
                "no epoch cap is set, so training will continue until interrupted."
            } else {
                "--max-epochs will be used."
            };
            eprintln!(
                "  note: epoch-final early stop requires a readable --test-teacher; \
                 validation metrics are unavailable, so {fallback}"
            );
        }
        let mut last_epoch_for_fallback = 1usize;
        for epoch in 1..=max_epochs {
            last_epoch_for_fallback = epoch;
            let display_epoch = epoch + cb_ctx.epoch_offset;
            let display_max_epochs =
                if max_epochs == usize::MAX { usize::MAX } else { max_epochs + cb_ctx.epoch_offset };
            print_epoch_banner(display_epoch, display_max_epochs);
            let mut stop_training_after_epoch = false;
            let mut plateau_epoch_final_metrics: Option<PlateauMetrics> = None;
            let mut epoch_final_metrics: Option<PlateauMetrics> = None;
            let mut plateau_state = if matches!(args.lr_schedule, LrScheduleKind::Plateau) {
                Some(PlateauLrState::new(
                    args.lr,
                    args.lr_min,
                    args.lr_plateau_factor,
                    args.lr_plateau_min_delta,
                    args.lr_plateau_monitor,
                ))
            } else {
                None
            };
            // Epoch boundary (epoch >= 2): in teacher-single-epoch mode,
            // a new epoch is a new pass over the teacher, so reset the
            // dataloader. When `--superbatches N` is set, epoch is an
            // LR/validation cycle; the teacher stream must continue from
            // its current position across epoch boundaries.
            if epoch > 1 && teacher_single_epoch {
                cb_dataloader_resume_offset.set((0, 0));
                cb_dataloader_resume_exact.set(false);
                if matches!(format, DataFormat::Hcpe) {
                    persistent_hcpe_loader = new_hcpe_loader(0, false);
                }
            }
            // Epoch >= 2: re-read the top-level summary-learn.log so `cb_prior_position`
            // picks up the cumulative positions across the previous epoch(s).
            // Without this, bullet's sb counter reset at trainer.run boundary
            // would also reset the `positions` column.
            if epoch > 1 {
                cb_prior_position =
                    read_prior_positions(&cb_top_level_log).get("nnue").copied().unwrap_or(cb_prior_position);
            }
            let net_id_for_epoch = if max_epochs > 1 || cb_ctx.epoch_offset > 0 {
                format!("{net_id_base}-e{display_epoch}")
            } else {
                net_id_base.clone()
            };
            last_net_id_for_epoch = net_id_for_epoch.clone();

            // Run the epoch in chunks of `save_rate` superbatches. Each
            // chunk ends at a save boundary, after which we validate (if
            // requested) and finalise the saved dir with the test metrics.
            //
            // chunk_start: epoch 1 honours the auto-resume / user-set
            // start sb so an interrupted previous epoch picks up where it
            // left off. Epoch>=2 always starts at displayed sb=1. This
            // resets the LR/validation cycle only; with explicit
            // `--superbatches N`, the dataloader position continues
            // across the boundary.
            let mut chunk_start = if epoch == 1 { effective_start_superbatch } else { 1 };
            let chunk_size = effective_save_rate(args).max(1);
            let plateau_rollback_dir = output_dir_buf.join(".bulletou_plateau_rollback");
            'epoch: loop {
                if chunk_start > end_superbatch {
                    break;
                }
                let chunk_resume_before = cb_dataloader_resume_offset.get();
                let chunk_resume_exact_before = cb_dataloader_resume_exact.get();
                let chunk_end = chunk_start.saturating_add(chunk_size).saturating_sub(1).min(end_superbatch);
                let lr_for_chunk = plateau_state.as_ref().map(|s| s.current_lr);
                if plateau_state.is_some() {
                    let _ = std::fs::remove_dir_all(&plateau_rollback_dir);
                    let rollback_path = plateau_rollback_dir.to_str().expect("checkpoint path is utf-8");
                    trainer.save_to_checkpoint(rollback_path);
                }
                let lr_scheduler_for_chunk = match args.lr_schedule {
                    LrScheduleKind::Step => LrSchedulerImpl::Step(StepLR {
                        start: args.lr,
                        min: args.lr_min,
                        gamma: lr_step_gamma,
                        step_positions: lr_step_positions,
                        period_positions: lr_period,
                        prior_positions: cb_prior_position as u64,
                        batch_size: effective_batch_size(args),
                        batches_per_superbatch,
                    }),
                    LrScheduleKind::Geometric => LrSchedulerImpl::Geometric(GeometricLR {
                        start: args.lr,
                        min: args.lr_min,
                        period_positions: lr_period,
                        prior_positions: cb_prior_position as u64,
                        batch_size: effective_batch_size(args),
                        batches_per_superbatch,
                    }),
                    LrScheduleKind::Cos => LrSchedulerImpl::Cos(CosineLR {
                        start: args.lr,
                        min: args.lr_min,
                        period_positions: lr_period,
                        prior_positions: cb_prior_position as u64,
                        batch_size: effective_batch_size(args),
                        batches_per_superbatch,
                    }),
                    LrScheduleKind::Plateau => {
                        LrSchedulerImpl::Fixed(FixedLR { value: lr_for_chunk.unwrap_or(args.lr) })
                    }
                };

                let schedule = TrainingSchedule {
                    net_id: net_id_for_epoch.clone(),
                    eval_scale: args.scale as f32,
                    steps: TrainingSteps {
                        batch_size: effective_batch_size(args),
                        batches_per_superbatch,
                        start_superbatch: chunk_start,
                        end_superbatch: chunk_end,
                    },
                    wdl_scheduler: wdl::ConstantWDL { value: 1.0 - args.lambda },
                    lr_scheduler: lr_scheduler_for_chunk,
                    save_rate: effective_save_rate(args),
                    save_epoch_end: effective_save_epoch_end(args),
                };

                // Per-chunk callback only records that the save happened
                // (no in-callback finalize). The actual rename + enrich
                // runs outside `trainer.run` so it can take per-save test
                // metrics computed via `trainer.eval_packed_batch`.
                let net_id_for_cb = net_id_for_epoch.clone();
                let output_dir_for_cb = output_dir_buf.clone();
                let saved_any_ref = &saved_any;
                let saved_dir_in_chunk: std::cell::RefCell<Option<std::path::PathBuf>> = std::cell::RefCell::new(None);
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

                // Resume pointer: 「ここまで処理した」を Consumer 側が書く
                // (offset, plies) のハンドル。trainer.run 後にこのペアを読んで
                // dataloader_pos.txt に書き出す。HCPE / PSV のような固定長
                // フォーマットでは plies は常に 0、HCPE3 / pack のような
                // 棋譜フォーマットでは ply 単位の正確な位置になる。
                let dl_offset_handle: std::cell::Cell<Option<std::sync::Arc<std::sync::atomic::AtomicU64>>> =
                    std::cell::Cell::new(None);
                let dl_plies_handle: std::cell::Cell<Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>> =
                    std::cell::Cell::new(None);

                last_error_record = match format {
                    DataFormat::Hcpe => {
                        // Persistent loader: chunk loop の外で 1 度だけ生成済み。
                        // 内部の producer thread が複数の trainer.run 呼び出しを
                        // 跨いで pre-fetch を継続する。
                        let loader = persistent_hcpe_loader
                            .as_ref()
                            .expect("persistent_hcpe_loader initialised for DataFormat::Hcpe");
                        dl_offset_handle.set(Some(loader.consumed_offset_handle()));
                        trainer.run(&schedule, &settings, loader)
                    }
                    DataFormat::Hcpe3 => {
                        let (resume_off, resume_plies) = cb_dataloader_resume_offset.get();
                        let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                            .with_buffer_records(hcpe_loader_buffer_records)
                            .with_single_epoch(teacher_single_epoch);
                        let loader = if cb_dataloader_resume_exact.get() {
                            loader.with_exact_resume_offset(resume_off, resume_plies)
                        } else {
                            loader.with_resume_offset(resume_off, resume_plies)
                        };
                        dl_offset_handle.set(Some(loader.consumed_offset_handle()));
                        dl_plies_handle.set(Some(loader.consumed_plies_handle()));
                        trainer.run(&schedule, &settings, &loader)
                    }
                    DataFormat::Pack => {
                        let (resume_off, resume_plies) = cb_dataloader_resume_offset.get();
                        let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                            .with_single_epoch(teacher_single_epoch);
                        let loader = if cb_dataloader_resume_exact.get() {
                            loader.with_exact_resume_offset(resume_off, resume_plies)
                        } else {
                            loader.with_resume_offset(resume_off, resume_plies)
                        };
                        dl_offset_handle.set(Some(loader.consumed_offset_handle()));
                        dl_plies_handle.set(Some(loader.consumed_plies_handle()));
                        trainer.run(&schedule, &settings, &loader)
                    }
                    DataFormat::Psv => {
                        // PSV is fixed 40-byte; DirectSequentialDataLoader
                        // already byte-seeks via `start_position * 40`, so
                        // no `dataloader_pos.txt` write is needed.
                        let loader = DirectSequentialDataLoader::new(&data_files_ref)
                            .with_single_epoch(teacher_single_epoch);
                        trainer.run(&schedule, &settings, &loader)
                    }
                };

                // Consumer が「ここまで処理した」を表す (offset, plies) を
                // 読み出し、次 chunk 用 resume として保持しておく。ファイル
                // 書き出しは finalize_one_nnue_dir 成功後にやる。
                let consumed_offset_val =
                    dl_offset_handle.take().map(|arc| arc.load(std::sync::atomic::Ordering::Acquire));
                let consumed_plies_val =
                    dl_plies_handle.take().map(|arc| arc.load(std::sync::atomic::Ordering::Acquire));
                if let Some(off) = consumed_offset_val {
                    cb_dataloader_resume_offset.set((off, consumed_plies_val.unwrap_or(0)));
                    cb_dataloader_resume_exact.set(true);
                }

                // Closure dropped → its borrow of saved_dir_in_chunk released.
                let saved_ckpt_dir = saved_dir_in_chunk.into_inner();
                let planned_train_only_tail = !effective_save_epoch_end(args)
                    && chunk_end == end_superbatch
                    && !chunk_end.is_multiple_of(effective_save_rate(args).max(1));
                if saved_ckpt_dir.is_none()
                    && planned_train_only_tail
                    && error_record_completed_superbatch(&last_error_record, chunk_end, batches_per_superbatch)
                {
                    trained_without_save.set(true);
                    eprintln!(
                        "  checkpoint skipped at epoch={}, superbatch={} (--no-save-epoch-end)",
                        display_epoch, chunk_end
                    );
                    chunk_start = chunk_end + 1;
                    continue 'epoch;
                }
                // 教師が sb 境界を跨がず途中で EOF した場合、bullet の save
                // callback は発火しない (= `saved_ckpt_dir` is None) が、
                // last_error_record には partial sb のロスが残っている。この
                // 場合は手動で trainer.save_to_checkpoint + log.txt 書き出し
                // を行い、partial sb 分も `0NNN/` ディレクトリ + learn.log
                // に残す。bullet が batch 0 個でも返ってきた (`last_error_record`
                // 空) ケースは、非 plateau では epoch 終了、plateau では教師先頭に
                // 巻き戻して同じ epoch を継続する。
                let (ckpt_dir, is_partial_save) = match saved_ckpt_dir {
                    Some(dir) => (dir, false),
                    None => {
                        if last_error_record.is_empty() {
                            if plateau_state.is_some() {
                                if chunk_resume_before == (0, 0) && chunk_resume_exact_before {
                                    eprintln!(
                                        "error: --lr-schedule plateau reached teacher EOF immediately after rewinding; \
                                         teacher appears empty or unreadable."
                                    );
                                    break 'epoch;
                                }
                                cb_dataloader_resume_offset.set((0, 0));
                                cb_dataloader_resume_exact.set(true);
                                if matches!(format, DataFormat::Hcpe) {
                                    persistent_hcpe_loader = new_hcpe_loader(0, true);
                                }
                                eprintln!(
                                    "  plateau: teacher EOF reached before a new superbatch; rewinding teacher to the beginning."
                                );
                                continue 'epoch;
                            }
                            break 'epoch;
                        }
                        let partial_dir = output_dir_buf.join(format!("{net_id_for_epoch}-{chunk_start}"));
                        eprintln!(
                            "  partial superbatch {} (teacher EOF mid-sb); \
                             saving partial state to {}",
                            chunk_start,
                            partial_dir.display(),
                        );
                        let s = partial_dir.to_str().expect("checkpoint path is utf-8");
                        trainer.save_to_checkpoint(s);
                        if let Err(e) = write_loss_csv(&partial_dir.join("log.txt"), &last_error_record) {
                            eprintln!("  WARN: failed to write log.txt in {}: {e}", partial_dir.display());
                        }
                        match convert_save_dir_to_nnue_layout(&partial_dir) {
                            Ok(()) => eprintln!("  wrote NNUE nn.bin + state.bin in {}", partial_dir.display()),
                            Err(e) => eprintln!("  WARN: failed to convert save dir {}: {e}", partial_dir.display()),
                        }
                        saved_any.set(true);
                        (partial_dir, true)
                    }
                };

                // Per-save validation (if --test-teacher cached positions).
                let test_metrics = test_cache.as_ref().map(|cache| {
                    let outputs = trainer.eval_packed_batch(&cache.positions, args.test_batch_size);
                    run_one_test_pass(cache, args, outputs)
                });
                if !matches!(args.lr_schedule, LrScheduleKind::Plateau) {
                    epoch_final_metrics = test_metrics.map(Into::into);
                }
                if args.dump_activation_stats {
                    if let Some(cache) = test_cache.as_ref() {
                        run_training_inline_nnue!(@dump $stats_mode $(($activation_stats))?, trainer, cache);
                    }
                }
                cb_ctx.lr_override = lr_for_chunk;

                let plateau_action = plateau_state.as_mut().map(|state| {
                    let metrics = test_metrics.expect("plateau requires --test-teacher");
                    state.observe(metrics.into())
                });
                let retry_same_chunk = plateau_action.map(plateau_action_retries_teacher).unwrap_or(false);
                let reject_checkpoint = plateau_action.map(plateau_action_rejects_update).unwrap_or(false);
                if reject_checkpoint {
                    let rollback_path = plateau_rollback_dir.to_str().expect("checkpoint path is utf-8");
                    trainer.load_from_checkpoint(rollback_path);
                    if let Err(e) = std::fs::remove_dir_all(&ckpt_dir) {
                        eprintln!("  WARN: failed to remove rejected plateau checkpoint {}: {e}", ckpt_dir.display());
                    }
                }
                let dataloader_pos_to_write = if retry_same_chunk {
                    Some(chunk_resume_before)
                } else {
                    consumed_offset_val.map(|off| (off, consumed_plies_val.unwrap_or(0)))
                };
                let plateau_epoch_done = matches!(plateau_action, Some(PlateauAction::FinalImproved { .. }));

                // Finalise this saved dir into <NNNN>/ with the test metrics.
                let idx = cb_next_idx.get();
                if !reject_checkpoint {
                    match finalize_one_nnue_dir(
                        &output_dir_buf,
                        &ckpt_dir,
                        &cb_ctx,
                        epoch,
                        idx,
                        cb_prior_position,
                        test_metrics,
                        !is_partial_save,
                    ) {
                        Ok(dst) => {
                            cb_next_idx.set(idx + 1);
                            if let Err(e) = append_to_top_level_log(&output_dir_buf, idx) {
                                eprintln!(
                                    "  WARN: failed to update {}: {e}",
                                    output_dir_buf.join(SUMMARY_LEARN_LOG_NAME).display()
                                );
                            }
                            // Consumer が「ここまで処理した」を表す pair を
                            // `dataloader_pos.txt` に書く (`<offset>,<plies>` 形式)。
                            // HCPE / PSV のような固定長レコードでは plies=0、HCPE3
                            // / pack のような棋譜単位なら ply 内の位置まで含めて
                            // 厳密に再開できる。
                            if let Some((off, plies)) = dataloader_pos_to_write {
                                let pos_file = dst.join("dataloader_pos.txt");
                                if let Err(e) = std::fs::write(&pos_file, format!("{off},{plies}\n")) {
                                    eprintln!("  WARN: failed to write {}: {e}", pos_file.display());
                                }
                            }
                            if plateau_epoch_done {
                                let marker = dst.join(PLATEAU_EPOCH_DONE_NAME);
                                if let Err(e) = std::fs::write(&marker, b"1\n") {
                                    eprintln!("  WARN: failed to write {}: {e}", marker.display());
                                }
                            }
                            eprintln!("  -> {}/", dst.display());
                        }
                        Err(e) => {
                            eprintln!("  WARN: failed to finalise {} into NNNN/: {e}", ckpt_dir.display());
                        }
                    }
                }

                if let Some(action) = plateau_action {
                    let current_lr = plateau_state.as_ref().map(|s| s.current_lr).unwrap_or(args.lr);
                    let monitor_label = args.lr_plateau_monitor.label();
                    match action {
                        PlateauAction::First { metrics } => {
                            eprintln!("  plateau: initial validation metrics = {}; lr stays {current_lr}", plateau_metrics_text(metrics));
                        }
                        PlateauAction::Improved { old_best, new_best } => {
                            eprintln!(
                                "  plateau: {monitor_label} improved (best {} -> {}); lr stays {}",
                                plateau_metrics_text(old_best),
                                plateau_metrics_text(new_best),
                                current_lr,
                            );
                        }
                        PlateauAction::Keep { metrics, best } => {
                            eprintln!(
                                "  plateau: {monitor_label} did not improve (current {}, best {}); lr stays {}",
                                plateau_metrics_text(metrics),
                                plateau_metrics_text(best),
                                current_lr,
                            );
                        }
                        PlateauAction::Reduced { old_lr, new_lr, metrics, best } => {
                            eprintln!(
                                "  plateau: {monitor_label} did not improve (current {}, best {}); lr {old_lr} -> {new_lr}",
                                plateau_metrics_text(metrics),
                                plateau_metrics_text(best),
                            );
                        }
                        PlateauAction::ScheduledFinal { old_lr, min_lr, metrics, best } => {
                            eprintln!(
                                "  plateau: {monitor_label} did not improve (current {}, best {}); \
                                 next lr would fall below lr_min, so one final superbatch will run at lr_min {min_lr} \
                                 (old lr {old_lr})",
                                plateau_metrics_text(metrics),
                                plateau_metrics_text(best),
                            );
                        }
                        PlateauAction::FinalImproved { old_best, new_best } => {
                            plateau_epoch_final_metrics = plateau_action_epoch_final_metrics(action);
                            eprintln!(
                                "  plateau: final lr_min superbatch improved {monitor_label} (best {} -> {}); \
                                 accepting it and ending this epoch.",
                                plateau_metrics_text(old_best),
                                plateau_metrics_text(new_best),
                            );
                            break 'epoch;
                        }
                        PlateauAction::FinalRejected { metrics, best } => {
                            plateau_epoch_final_metrics = plateau_action_epoch_final_metrics(action);
                            eprintln!(
                                "  plateau: final lr_min superbatch did not improve {monitor_label} (current {}, best {}); \
                                 discarding it and ending this epoch.",
                                plateau_metrics_text(metrics),
                                plateau_metrics_text(best),
                            );
                            mark_latest_checkpoint_epoch_done(&output_dir_buf);
                            break 'epoch;
                        }
                    }
                }

                if retry_same_chunk {
                    cb_dataloader_resume_offset.set(chunk_resume_before);
                    cb_dataloader_resume_exact.set(chunk_resume_exact_before);
                    if matches!(format, DataFormat::Hcpe) {
                        persistent_hcpe_loader = new_hcpe_loader(chunk_resume_before.0, chunk_resume_exact_before);
                    }
                    eprintln!(
                        "  plateau: restored model + optimiser, then rewinding teacher to retry superbatch {chunk_start} at lowered lr."
                    );
                    continue 'epoch;
                }

                // Partial save (= 教師 EOF mid-sb) was just finalised. Non-plateau
                // treats this as epoch end. Plateau keeps the same epoch alive by
                // wrapping the teacher to the beginning until lr_min final run.
                if is_partial_save {
                    if plateau_state.is_some() {
                        cb_dataloader_resume_offset.set((0, 0));
                        cb_dataloader_resume_exact.set(true);
                        if matches!(format, DataFormat::Hcpe) {
                            persistent_hcpe_loader = new_hcpe_loader(0, true);
                        }
                        eprintln!(
                            "  plateau: teacher EOF reached after partial superbatch; rewinding teacher to the beginning."
                        );
                        chunk_start = chunk_end + 1;
                        continue 'epoch;
                    }
                    break 'epoch;
                }

                chunk_start = chunk_end + 1;
            }
            let current_epoch_final_metrics = if matches!(args.lr_schedule, LrScheduleKind::Plateau) {
                plateau_epoch_final_metrics
            } else {
                epoch_final_metrics
            };
            if let Some(current_metrics) = current_epoch_final_metrics {
                if epoch_final_should_stop(
                    previous_epoch_final_metrics,
                    current_metrics,
                    PlateauMonitor::LossOrAccuracy,
                    0.0,
                ) {
                    let previous_metrics = previous_epoch_final_metrics.expect("checked by predicate");
                    eprintln!(
                        "  {}: epoch-final validation metrics did not improve from previous epoch \
                         (loss {:.6} -> {:.6}, accuracy {:.6} -> {:.6}); stopping training.",
                        args.lr_schedule.cli_name(),
                        previous_metrics.loss,
                        current_metrics.loss,
                        previous_metrics.accuracy,
                        current_metrics.accuracy
                    );
                    stop_training_after_epoch = true;
                }
                previous_epoch_final_metrics = Some(current_metrics);
            } else if matches!(args.lr_schedule, LrScheduleKind::Plateau) && max_epochs == usize::MAX {
                eprintln!(
                    "  plateau: epoch ended before epoch-final validation metrics were established; stopping unlimited run."
                );
                stop_training_after_epoch = true;
            }
            if stop_training_after_epoch {
                break;
            }
        }

        // End-of-training fallback save: skip it when `--no-save-epoch-end`
        // intentionally left a completed train-only tail without checkpoints.
        if !saved_any.get() && !trained_without_save.get() {
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
            if args.dump_activation_stats {
                if let Some(cache) = test_cache.as_ref() {
                    run_training_inline_nnue!(@dump $stats_mode $(($activation_stats))?, trainer, cache);
                }
            }
            let idx = cb_next_idx.get();
            match finalize_one_nnue_dir(
                &output_dir_buf,
                &ckpt_dir,
                &cb_ctx,
                /*epoch=*/ last_epoch_for_fallback,
                idx,
                cb_prior_position,
                test_metrics,
                false,
            ) {
                Ok(dst) => {
                    cb_next_idx.set(idx + 1);
                    if let Err(e) = append_to_top_level_log(&output_dir_buf, idx) {
                        eprintln!(
                            "  WARN: failed to update {}: {e}",
                            output_dir_buf.join(SUMMARY_LEARN_LOG_NAME).display()
                        );
                    }
                    eprintln!("  -> {}/", dst.display());
                }
                Err(e) => {
                    eprintln!("  WARN: failed to finalise fallback save dir {} into NNNN/: {e}", ckpt_dir.display());
                }
            }
        }
        let _ = std::fs::remove_dir_all(output_dir_buf.join(".bulletou_plateau_rollback"));
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
/// L2 / L3 sizes come from `--arch` (`NNUE_halfkp_256x2_32_32`, …);
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
    match args.optimizer {
        OptimizerKind::Adamw => run_halfkp_impl::<optimiser::AdamW>(args),
        OptimizerKind::Radam => run_halfkp_impl::<optimiser::RAdam>(args),
        OptimizerKind::Ranger => run_halfkp_impl::<optimiser::Ranger>(args),
    }
}

fn run_halfkp_impl<O: BulletouOptimizer>(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let input_size = ShogiHalfKP.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_HALFKP ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    // ---- Resume support -------------------------------------------------
    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(args, &output_dir);
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
        .optimiser(O::default())
        .inputs(ShogiHalfKP)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &["outw"], &["l0b", "l1b", "l2b", "outb"]);

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    // Cleanup the scratch resume dir if it was used.
    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    // Single-component finalisation: rename `<net_id>-*/` to `0NNN/` and
    // enrich each dir's bullet `log.txt` into the 7-column `learn.log`.
    let ctx = LogContext::from_args(args, 0);
    let top_level_log = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!("warning: failed to update {}: {e}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display());
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
    match args.optimizer {
        OptimizerKind::Adamw => run_kp_impl::<optimiser::AdamW>(args),
        OptimizerKind::Radam => run_kp_impl::<optimiser::RAdam>(args),
        OptimizerKind::Ranger => run_kp_impl::<optimiser::Ranger>(args),
    }
}

fn run_kp_impl<O: BulletouOptimizer>(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let input_size = ShogiKp.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_KP ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(args, &output_dir);
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
        .optimiser(O::default())
        .inputs(ShogiKp)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &["outw"], &["l0b", "l1b", "l2b", "outb"]);

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    let ctx = LogContext::from_args(args, 0);
    let top_level_log = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!("warning: failed to update {}: {e}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display());
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
/// Architecture is selected via `--arch` (default `NNUE_ka2_256x2_32_32`).
fn run_nnue_ka2(args: &Args) {
    match args.optimizer {
        OptimizerKind::Adamw => run_nnue_ka2_impl::<optimiser::AdamW>(args),
        OptimizerKind::Radam => run_nnue_ka2_impl::<optimiser::RAdam>(args),
        OptimizerKind::Ranger => run_nnue_ka2_impl::<optimiser::Ranger>(args),
    }
}

fn run_nnue_ka2_impl<O: BulletouOptimizer>(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let input_size = ShogiKa2.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_KA2 ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(args, &output_dir);
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
        .optimiser(O::default())
        .inputs(ShogiKa2)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &["outw"], &["l0b", "l1b", "l2b", "outb"]);

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    let ctx = LogContext::from_args(args, 0);
    let top_level_log = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!("warning: failed to update {}: {e}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display());
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
    match args.optimizer {
        OptimizerKind::Adamw => run_halfkpe9_impl::<optimiser::AdamW>(args),
        OptimizerKind::Radam => run_halfkpe9_impl::<optimiser::RAdam>(args),
        OptimizerKind::Ranger => run_halfkpe9_impl::<optimiser::Ranger>(args),
    }
}

fn run_halfkpe9_impl<O: BulletouOptimizer>(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let input_size = ShogiHalfKpe9.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_HALFKPE9 ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(args, &output_dir);
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
        .optimiser(O::default())
        .inputs(ShogiHalfKpe9)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &["outw"], &["l0b", "l1b", "l2b", "outb"]);

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    let ctx = LogContext::from_args(args, 0);
    let top_level_log = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!("warning: failed to update {}: {e}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display());
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
    match args.optimizer {
        OptimizerKind::Adamw => run_halfkpvm_impl::<optimiser::AdamW>(args),
        OptimizerKind::Radam => run_halfkpvm_impl::<optimiser::RAdam>(args),
        OptimizerKind::Ranger => run_halfkpvm_impl::<optimiser::Ranger>(args),
    }
}

fn run_halfkpvm_impl<O: BulletouOptimizer>(args: &Args) {
    let (l1_size, l2_size, l3_size) = args.arch().dims();
    let input_size = ShogiHalfKPvm.num_inputs();
    let l1_input_dim = 2 * l1_size;

    eprintln!(
        "=== bulletou: running NNUE_HALFKPVM ({}x2-{}-{} ClippedReLU, dual-perspective) ===",
        l1_size, l2_size, l3_size
    );

    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(args, &output_dir);
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
        .optimiser(O::default())
        .inputs(ShogiHalfKPvm)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

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

    configure_optimizer::<O, _, _>(args, &mut trainer, &["outw"], &["l0b", "l1b", "l2b", "outb"]);

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline_nnue!(args, &mut trainer);

    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    let ctx = LogContext::from_args(args, 0);
    let top_level_log = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!("warning: failed to update {}: {e}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display());
            }
        }
        Err(e) => {
            eprintln!("error: failed to finalise NNUE checkpoint dirs: {e}");
            std::process::exit(1);
        }
    }
}

/// SFNN / LayerStacks training entry point. Generic over the input feature so
/// HalfKA_hm1, HalfKA_hm2, HalfKA2, and K+A2 share the pipeline; only the
/// `input` and `feature_set` arguments differ between the callers.
///
/// Network topology mirrors YaneuraOu SFNN dynamic architecture:
/// per-perspective FT (`l0`) → CReLU → pairwise-mul → concat
/// across perspectives → bucket-specific L1 (= hidden + 1 PSQT
/// shortcut neuron) → split off the PSQT neuron as a residual bypass
/// → `[SqrCReLU; CReLU]` concat → L2 (32 dim) → CReLU → L3 (scalar)
/// → add PSQT residual.
///
/// The bucket-specific layers `l1` and `l2` use the LayerStacks pattern:
/// the weight tensor is `(NUM_STACKS × out_dim, in_dim)` and a
/// `.select(output_buckets)` per forward pass picks the active slice
/// based on `ShogiLayerStackBucket9` (= `--arch` LayerStack suffix).
///
fn run_sfnn_1536<I>(args: &Args, input: I, feature_set: NnueFeatureSet)
where
    I: SparseInputType<RequiredDataType = bulletou_lib::shogi::PackedSfenValue> + Copy,
{
    match args.optimizer {
        OptimizerKind::Adamw => run_sfnn_1536_impl::<optimiser::AdamW, I>(args, input, feature_set),
        OptimizerKind::Radam => run_sfnn_1536_impl::<optimiser::RAdam, I>(args, input, feature_set),
        OptimizerKind::Ranger => run_sfnn_1536_impl::<optimiser::Ranger, I>(args, input, feature_set),
    }
}

fn run_sfnn_1536_impl<O, I>(args: &Args, input: I, feature_set: NnueFeatureSet)
where
    O: BulletouOptimizer,
    I: SparseInputType<RequiredDataType = bulletou_lib::shogi::PackedSfenValue> + Copy,
{
    let arch = args.arch();
    let (ft_size, l1_hidden, l2_size) = arch.dims();
    // L1 output = effective hidden + 1 PSQT-shortcut neuron (matches
    // yaneuraou `kHidden1Dims + 1` in the SFNN architecture).
    let l1_out = l1_hidden + 1;
    let l1_effective = l1_hidden;
    let l2_in = l1_effective * 2; // [SqrCReLU; CReLU] concat
    let layerstack = args.effective_layerstack().unwrap_or(LayerStackMode::Kingrank3by3);
    let num_stacks = layerstack.num_stacks();
    let input_size = input.num_inputs();

    eprintln!(
        "=== bulletou: running {} ({}x2-{}-{} CReLU+SqrCReLU, dual-perspective, LayerStacks={} via {}) ===",
        args.eval_type().cli_name(),
        ft_size,
        l1_hidden,
        l2_size,
        num_stacks,
        layerstack.cli_name()
    );
    if args.nnue_pytorch_init_scale != 1.0 {
        eprintln!("  nnue-pytorch init scale = {}", args.nnue_pytorch_init_scale);
    }
    if args.sfnn_factorized_l1 {
        eprintln!("  SFNN factorized L1 shared term = enabled");
    }
    let output_dir = args.output_dir();
    let resume_state_bin = find_latest_state_bin(args, &output_dir);
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
    let bucket_impl = match layerstack {
        LayerStackMode::Kingrank3by3 => ShogiLayerStackBucket9::KingRank9,
    };

    // YaneuraOu SFNN 互換 nn.bin の save format。`Sfnn1536SaveParams`
    // で feature set / 各層のサイズ / LayerStacks 数を受け、`bulletou_lib::value::nnue_save_sfnn1536`
    // が組み立てた `SavedFormat` 列を渡す。出力は `EvalDir` で yaneuraou
    // (`YANEURAOU_ENGINE_SFNN_*` ビルド) が load できる layout。
    let save_format = build_sfnn_1536_save_format(Sfnn1536SaveParams {
        feature_set,
        input_size,
        ft_size,
        l1_hidden,
        l2_size,
        num_stacks,
        factorized_l1: args.sfnn_factorized_l1,
    });

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(O::default())
        .inputs(input)
        .output_buckets(bucket_impl)
        .save_format(&save_format)
        .loss_fn(value_loss_fn(args));

    if args.nnue_pytorch_wrm_loss {
        builder = builder.use_win_rate_model();
    }

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs, output_buckets| {
        let l0 = builder.new_affine("l0", input_size, ft_size);
        l0.init_nnue_pytorch_feature_transformer_scaled(input_size, args.nnue_pytorch_init_scale);

        // L1: yaneuraou's SFNN stores one independent `fc_0` per LayerStack.
        // With --sfnn-factorized-l1, also train nnue-pytorch's zero-initialised
        // shared L1 term and fold it into every bucket when saving `nn.bin`.
        // Match nnue-pytorch's StackedLinear initialisation: initialise bucket
        // 0 and copy it to every bucket. The output bias is zero-initialised.
        let l1 = builder.new_stacked_affine_nnue_pytorch_scaled(
            "l1",
            ft_size,
            l1_out,
            num_stacks,
            false,
            args.nnue_pytorch_init_scale,
        );
        let l1f = if args.sfnn_factorized_l1 {
            let l1f = builder.new_affine("l1f", ft_size, l1_out);
            l1f.init_zeroed();
            Some(l1f)
        } else {
            None
        };
        let l2 = builder.new_stacked_affine_nnue_pytorch_scaled(
            "l2",
            l2_in,
            l2_size,
            num_stacks,
            false,
            args.nnue_pytorch_init_scale,
        );
        let l3 = builder.new_stacked_affine_nnue_pytorch_scaled(
            "l3",
            l2_size,
            1,
            num_stacks,
            true,
            args.nnue_pytorch_init_scale,
        );

        // Per-perspective FT → CReLU → pairwise-mul → concat. After the
        // pairwise-mul the dim is ft_size/2; concat of stm/ntm brings it
        // back to ft_size (matching `kInputDims = kTransformedFeatureDimensions`
        // in sfnnwop-1536.h).
        let stm = l0.forward(stm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
        let ntm = l0.forward(ntm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
        let combined = stm.concat(ntm);

        let mut l1_out_t = l1.forward(combined).select(output_buckets);
        if let Some(l1f) = l1f {
            l1_out_t = l1_out_t + l1f.forward(combined);
        }

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

    let bias_ids: &[&str] = if args.sfnn_factorized_l1 {
        &["l0b", "l1b", "l1fb", "l2b", "l3b"]
    } else {
        &["l0b", "l1b", "l2b", "l3b"]
    };
    configure_optimizer::<O, _, _>(args, &mut trainer, &["l3w"], bias_ids);

    if let Some(dir) = resume_dir.as_ref() {
        eprintln!("  [NNUE] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    let activation_shape = SfnnActivationShape { ft_size, l1_hidden, l1_out, l2_in, l2_size, num_stacks };
    run_training_inline_nnue!(args, &mut trainer, |trainer, cache| {
        dump_sfnn_activation_stats(args, trainer, input, bucket_impl, cache, activation_shape);
    });

    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    let ctx = LogContext::from_args(args, 0);
    let top_level_log = output_dir.join(SUMMARY_LEARN_LOG_NAME);
    let prior_position = read_prior_positions(&top_level_log).get("nnue").copied().unwrap_or(0);
    match finalize_nnue_dirs(&output_dir, &ctx, &args.net_id(), prior_position) {
        // (0, 0) = nothing left to do (per-superbatch callback already
        // finalised everything during training). Top-level learn.log was
        // appended incrementally too, so skip the extra append here.
        Ok((_first_idx, 0)) => {}
        Ok((_first_idx, last_idx)) => {
            if let Err(e) = append_to_top_level_log(&output_dir, last_idx) {
                eprintln!("warning: failed to update {}: {e}", output_dir.join(SUMMARY_LEARN_LOG_NAME).display());
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
        NnueFeatureSet::HalfKa2 => "halfka2",
        NnueFeatureSet::Ka2 => "ka2",
        // run_sfnn_1536 is only invoked from SFNN eval-type dispatch in main(),
        // so any other feature set means a new EvalType variant was added
        // without updating this arm.
        other => unreachable!("run_sfnn_1536 received unsupported feature set: {other:?}"),
    };
    let edition_name =
        format!("YANEURAOU_ENGINE_SFNN_{feature_suffix}_{ft_size}_{l1_hidden}_{l2_size}_{}", layerstack.arch_suffix());
    let alias_note = if matches!(feature_set, NnueFeatureSet::HalfKaHm2)
        && ft_size == 1536
        && l1_hidden == 15
        && l2_size == 32
        && num_stacks == 9
    {
        "\n  (alias: YANEURAOU_ENGINE_SFNN1536)"
    } else {
        ""
    };

    eprintln!(
        "note: nn.bin in each save dir targets a YaneuraOu build with edition\n  \
             {edition_name}{alias_note}\n\
         Build it with `make normal YANEURAOU_EDITION=<edition>`."
    );
}

/// Single-component analogue of `assemble_numbered_dirs`: list `<net_id>-*/`
/// (or `<net_id>-e<epoch>-<sb>/` for multi-epoch) under `output_dir`, sort
/// by (epoch, sb), rename them to `0NNN/` starting at `existing_count + 1`,
/// and enrich each dir's bullet-format `log.txt` into the current CSV
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
    last_superbatch_complete: bool,
) -> std::io::Result<std::path::PathBuf> {
    let dst = output_dir.join(format!("{idx:04}"));
    std::fs::rename(src, &dst)?;
    let log_txt = dst.join("log.txt");
    let learn_log = dst.join("learn.log");
    let raw = std::fs::read_to_string(&log_txt).unwrap_or_default();
    let body =
        enrich_bullet_log_to_csv(&raw, ctx, epoch, "nnue", prior_position, test_metrics, last_superbatch_complete);
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
        let dst = finalize_one_nnue_dir(output_dir, src, ctx, *epoch, idx, prior_position, None, true)?;
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
        assert_eq!(NnueArch::from_str("NNUE_halfkp_256x2_32_32").unwrap().dims(), (256, 32, 32));
        assert_eq!(NnueArch::from_str("NNUE_halfkp_1024x2_8_64").unwrap().dims(), (1024, 8, 64));
        assert_eq!(NnueArch::from_str("SFNN_halfkahm2_1536_15_32_k3k3").unwrap().dims(), (1536, 15, 32));
        assert_eq!(
            NnueArch::from_str("sfnn_HALFKA2_1024_7_64_K3K3").unwrap().cli_name(),
            "SFNN_halfka2_1024_7_64_k3k3"
        );
        assert_eq!(NnueArch::from_str("SFNN1536").unwrap().cli_name(), "SFNN_halfkahm2_1536_15_32_k3k3");
    }

    #[test]
    fn nnue_arch_parse_freeform_sizes() {
        assert_eq!(NnueArch::from_str("NNUE_ka2_256x2_64_64").unwrap().dims(), (256, 64, 64));
        assert_eq!(NnueArch::from_str("SFNN_halfka2_2048_32_64_king3_by_king3").unwrap().dims(), (2048, 32, 64));
    }

    #[test]
    fn nnue_arch_cli_name_roundtrip() {
        for s in [
            "NNUE_halfkp_256x2_32_32",
            "NNUE_ka2_256x2_64_64",
            "SFNN_halfka2_1024_7_64_k3k3",
            "SFNN_halfkahm2_1536_15_32_k3k3",
        ] {
            let parsed = NnueArch::from_str(s).unwrap();
            assert_eq!(parsed.cli_name(), s);
            assert_eq!(parsed.to_string(), s);
        }
    }

    #[test]
    fn nnue_arch_parse_rejects_bad_format() {
        assert!(NnueArch::from_str("").is_err());
        assert!(NnueArch::from_str("256x2-32-32").is_err());
        assert!(NnueArch::from_str("YANEURAOU_ENGINE_SFNN_halfka2_1024_7_64_k3k3").is_err());
        assert!(NnueArch::from_str("NNUE_halfkp_256x3_32_32").is_err());
        assert!(NnueArch::from_str("NNUE_halfkp_256x2_32").is_err());
        assert!(NnueArch::from_str("NNUE_halfka2_256x2_32_32").is_err());
        assert!(NnueArch::from_str("SFNN_halfka2_1024_7_64_ls9").is_err());
    }

    #[test]
    fn nnue_arch_parse_rejects_bad_dims() {
        assert!(NnueArch::from_str("NNUE_halfkp_0x2_32_32").is_err());
        assert!(NnueArch::from_str("NNUE_halfkp_256x2_0_32").is_err());
        assert!(NnueArch::from_str("NNUE_halfkp_256x2_32_0").is_err());
        assert!(NnueArch::from_str("NNUE_halfkp_100x2_32_32").is_err());
        assert!(NnueArch::from_str("SFNN_halfka2_100_7_64_k3k3").is_err());
    }

    /// `finalize_one_nnue_dir` が bullet 形式の checkpoint dir を `<NNNN>/`
    /// に rename し、`log.txt` を current schema の `learn.log` に変換することを確認。
    /// per-superbatch save callback で呼ばれた場合の単発動作と等価。
    #[test]
    fn finalize_one_nnue_dir_renames_and_enriches() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-finalize-one-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
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
            arch: "SFNN_ka2_1536_15_32_k3k3".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/foo.hcpe".to_string(),
            epoch_offset: 0,
            lr_schedule: LrScheduleKind::Step,
            lr_period: 500_000_000,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.00001,
            lr_override: None,
        };

        let dst = finalize_one_nnue_dir(
            &tmp, &src, &ctx, /*epoch=*/ 1, /*idx=*/ 5, /*prior=*/ 0, /*test_metrics=*/ None, true,
        )
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
        // each row has 12 comma-separated fields (= LEARN_LOG_HEADER columns)
        // and the two test_value_* columns are "-" because we passed None.
        for row in &body_lines {
            assert_eq!(row.split(',').count(), 12, "row `{row}` should be 12 columns");
            assert!(row.starts_with("SFNN_KA2-SFNN_ka2_1536_15_32_k3k3,"));
            // Columns: eval, epoch, sb, batch, test_value_accuracy,
            // test_value_loss, train_value_loss, lr_start, lr_end, lambda,
            // positions, teacher
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
        let args = Args::try_parse_from(["bulletou", "--eval-type", "NNUE_KP", "--teacher", "/dev/null"]).unwrap();
        assert_eq!(args.output_dir(), std::path::PathBuf::from("checkpoints/NNUE_KP-NNUE_kp_256x2_32_32"),);

        // --tag appends `-<tag>` to the auto-derived name.
        let args =
            Args::try_parse_from(["bulletou", "--eval-type", "NNUE_KP", "--teacher", "/dev/null", "--tag", "lr0.001"])
                .unwrap();
        assert_eq!(args.output_dir(), std::path::PathBuf::from("checkpoints/NNUE_KP-NNUE_kp_256x2_32_32-lr0.001"),);

        // --tag with SFNN: applied after the architecture segment.
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_KA2",
            "--arch",
            "SFNN_ka2_1536_15_32_k3k3",
            "--teacher",
            "/dev/null",
            "--tag",
            "exp7",
        ])
        .unwrap();
        assert_eq!(args.output_dir(), std::path::PathBuf::from("checkpoints/SFNN_KA2-SFNN_ka2_1536_15_32_k3k3-exp7"),);

        // Explicit --output wins; --tag is ignored.
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_KP",
            "--teacher",
            "/dev/null",
            "--output",
            "/custom/path",
            "--tag",
            "ignored",
        ])
        .unwrap();
        assert_eq!(args.output_dir(), std::path::PathBuf::from("/custom/path"));

        // Empty --tag is treated as no tag (no trailing dash).
        let args = Args::try_parse_from(["bulletou", "--eval-type", "NNUE_KP", "--teacher", "/dev/null", "--tag", ""])
            .unwrap();
        assert_eq!(args.output_dir(), std::path::PathBuf::from("checkpoints/NNUE_KP-NNUE_kp_256x2_32_32"),);
    }

    #[test]
    fn resume_signature_distinguishes_superbatches_presence() {
        use clap::Parser as _;

        let with_superbatches = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--tag",
            "plateau-test",
            "--lr-schedule",
            "plateau",
            "--test-teacher",
            "/tmp/test.hcpe",
            "--superbatches",
            "19",
        ])
        .unwrap();
        let without_superbatches = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--tag",
            "plateau-test",
            "--lr-schedule",
            "plateau",
            "--test-teacher",
            "/tmp/test.hcpe",
        ])
        .unwrap();

        let sig_with = resume_signature(&with_superbatches);
        let sig_without = resume_signature(&without_superbatches);
        assert!(sig_with.contains("schema=bulletou-resume-v2"));
        assert!(sig_with.contains("backend=cuda-cpp"));
        assert!(sig_with.contains("positions_per_superbatch=99942400"));
        assert!(sig_with.contains("superbatches=19"));
        assert!(sig_with.contains("test_sample=random"));
        assert!(sig_without.contains("superbatches=none"));
        assert_ne!(sig_with, sig_without);
    }

    #[test]
    fn positions_per_superbatch_rounds_down_to_whole_batches() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--batch-size",
            "16384",
            "--positions-per-superbatch",
            "10000000",
        ])
        .unwrap();

        assert_eq!(effective_batches_per_superbatch(&args).unwrap(), 610);
        assert_eq!(effective_positions_per_superbatch(&args).unwrap(), 9_994_240);
    }

    #[test]
    fn omitted_batch_size_uses_tatara_sized_default_for_all_eval_types() {
        use clap::Parser as _;

        let nnue = Args::try_parse_from(["bulletou", "--eval-type", "NNUE_HALFKP", "--teacher", "/dev/null"]).unwrap();
        let sfnn = Args::try_parse_from(["bulletou", "--eval-type", "SFNN_HALFKA2", "--teacher", "/dev/null"]).unwrap();
        let kppt = Args::try_parse_from(["bulletou", "--eval-type", "KPPT", "--teacher", "/dev/null"]).unwrap();

        assert_eq!(effective_batch_size(&nnue), DEFAULT_BATCH_SIZE);
        assert_eq!(effective_batch_size(&sfnn), DEFAULT_BATCH_SIZE);
        assert_eq!(effective_batch_size(&kppt), DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn backend_defaults_to_cuda_cpp_path() {
        use clap::Parser as _;

        let args = Args::try_parse_from(["bulletou", "--eval-type", "NNUE_HALFKP", "--teacher", "/dev/null"]).unwrap();

        assert_eq!(args.backend, BackendKind::CudaCpp);
        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.unwrap_err().contains("--cuda-cpp-train-steps"));
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn explicit_bullet_backend_keeps_generic_path_available() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "bullet",
        ])
        .unwrap();

        assert_eq!(args.backend, BackendKind::Bullet);
        assert!(args.validate_backend_flags().is_ok());
    }

    #[test]
    fn cuda_cpp_backend_smoke_is_feature_gated() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-smoke",
        ])
        .unwrap();

        assert_eq!(args.backend, BackendKind::CudaCpp);
        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            let err = result.unwrap_err();
            assert!(err.contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_requires_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") {
            "--cuda-cpp-train-steps"
        } else {
            "cuda-cpp-backend"
        }));
    }

    #[test]
    fn cuda_cpp_backend_accepts_explicit_halfkp_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_explicit_kp_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_KP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_kppt_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "KPPT",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_kpp_kkpt_production_schedule() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "KPP_KKPT",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "1",
            "--max-epochs",
            "1",
            "--batch-size",
            "1024",
            "--positions-per-superbatch",
            "1024",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_rejects_kppt_validation_teacher() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "KPPT",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--test-teacher",
            "validation.psv",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") {
            "KPPT/KPP_KKPT currently skip"
        } else {
            "cuda-cpp-backend"
        }));
    }

    #[test]
    fn cuda_cpp_backend_rejects_kppt_profile_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "KPPT",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--cuda-cpp-profile-steps",
            "1",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") { "profile-steps" } else { "cuda-cpp-backend" }));
    }

    #[test]
    fn cuda_cpp_backend_accepts_halfkp_production_schedule() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "3",
            "--max-epochs",
            "2",
            "--save-rate",
            "2",
            "--batch-size",
            "1024",
            "--positions-per-superbatch",
            "4096",
            "--lr-schedule",
            "cos",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_halfkp_plateau_schedule() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "2",
            "--max-epochs",
            "1",
            "--batch-size",
            "1024",
            "--positions-per-superbatch",
            "1024",
            "--lr-schedule",
            "plateau",
            "--test-teacher",
            "yamaoka-floodgate.psv",
            "--save-rate",
            "1",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_plateau_requires_test_teacher() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "1",
            "--max-epochs",
            "1",
            "--lr-schedule",
            "plateau",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") { "--test-teacher" } else { "cuda-cpp-backend" }));
    }

    #[test]
    fn cuda_cpp_backend_plateau_requires_save_rate_one() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "2",
            "--max-epochs",
            "1",
            "--lr-schedule",
            "plateau",
            "--test-teacher",
            "yamaoka-floodgate.psv",
            "--save-rate",
            "2",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") { "--save-rate 1" } else { "cuda-cpp-backend" }));
    }

    #[test]
    fn cuda_cpp_backend_rejects_mixed_direct_and_production_schedule() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--superbatches",
            "1",
            "--max-epochs",
            "1",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") {
            "either --cuda-cpp-train-steps or --superbatches"
        } else {
            "cuda-cpp-backend"
        }));
    }

    #[test]
    fn cuda_cpp_backend_rejects_no_save_epoch_end_for_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--no-save-epoch-end",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") {
            "direct-step mode does not honor production schedule flags"
        } else {
            "cuda-cpp-backend"
        }));
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_run_schedule_chunks_save_rate_epoch_end_and_cos_lr() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "3",
            "--max-epochs",
            "2",
            "--save-rate",
            "2",
            "--batch-size",
            "1024",
            "--positions-per-superbatch",
            "4096",
            "--lr",
            "0.1",
            "--lr-min",
            "0.01",
            "--lr-schedule",
            "cos",
        ])
        .unwrap();

        args.validate_backend_flags().unwrap();
        let schedule = cuda_cpp_run_schedule(&args).unwrap();
        assert!(schedule.production);
        assert_eq!(schedule.batches_per_superbatch, 4);
        assert_eq!(schedule.total_steps, 24);
        assert_eq!(schedule.chunks.len(), 4);
        assert!(schedule.chunks.iter().all(|chunk| chunk.save_checkpoint));
        assert_eq!(schedule.chunks[0].epoch, 1);
        assert_eq!(schedule.chunks[0].superbatch, 2);
        assert_eq!(schedule.chunks[0].steps, 8);
        assert_eq!(schedule.chunks[0].cumulative_steps, 8);
        assert_eq!(schedule.chunks[1].epoch, 1);
        assert_eq!(schedule.chunks[1].superbatch, 3);
        assert_eq!(schedule.chunks[1].steps, 4);
        assert_eq!(schedule.chunks[1].cumulative_steps, 12);
        assert_eq!(schedule.chunks[2].epoch, 2);
        assert_eq!(schedule.chunks[2].superbatch, 2);
        assert_eq!(schedule.chunks[2].steps, 8);
        assert_eq!(schedule.chunks[2].cumulative_steps, 20);
        assert_eq!(schedule.chunks[3].epoch, 2);
        assert_eq!(schedule.chunks[3].superbatch, 3);
        assert_eq!(schedule.chunks[3].steps, 4);
        assert_eq!(schedule.chunks[3].cumulative_steps, 24);
        assert!((schedule.chunks[0].lr_start - 0.1).abs() < 1e-6);
        assert!(schedule.chunks[0].lr_end < schedule.chunks[0].lr_start);
        assert!((schedule.chunks[2].lr_start - 0.1).abs() < 1e-6, "LR should warm-restart at epoch 2");
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_run_schedule_no_save_epoch_end_trains_tail_without_checkpoint() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "3",
            "--max-epochs",
            "1",
            "--save-rate",
            "2",
            "--no-save-epoch-end",
            "--batch-size",
            "1024",
            "--positions-per-superbatch",
            "4096",
        ])
        .unwrap();

        args.validate_backend_flags().unwrap();
        let schedule = cuda_cpp_run_schedule(&args).unwrap();
        assert!(schedule.production);
        assert_eq!(schedule.batches_per_superbatch, 4);
        assert_eq!(schedule.total_steps, 12);
        assert_eq!(schedule.chunks.len(), 2);
        assert_eq!(schedule.chunks[0].superbatch, 2);
        assert_eq!(schedule.chunks[0].steps, 8);
        assert!(schedule.chunks[0].save_checkpoint);
        assert_eq!(schedule.chunks[1].superbatch, 3);
        assert_eq!(schedule.chunks[1].steps, 4);
        assert_eq!(schedule.chunks[1].cumulative_steps, 12);
        assert!(!schedule.chunks[1].save_checkpoint);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_run_schedule_resumes_mid_epoch_from_next_superbatch() {
        use clap::Parser as _;

        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-cuda-cpp-schedule-mid-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(tmp.join("0001")).unwrap();
        let output = tmp.to_str().unwrap();
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "teacher.hcpe",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "3",
            "--max-epochs",
            "1",
            "--save-rate",
            "1",
            "--batch-size",
            "64",
            "--positions-per-superbatch",
            "64",
            "--lr",
            "0.1",
            "--lr-min",
            "0.01",
            "--output",
            output,
        ])
        .unwrap();
        write_resume_config(&tmp, &args).unwrap();
        std::fs::write(tmp.join("0001").join("state.bin"), b"state").unwrap();
        std::fs::write(
            tmp.join("0001").join("learn.log"),
            format!(
            "{LEARN_LOG_HEADER}\nNNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,1,-,-,0.1,0.1,0.1,1.000000,64,teacher.hcpe\n"
        ),
        )
        .unwrap();
        std::fs::write(tmp.join(SUMMARY_LEARN_LOG_NAME), format!(
            "{SUMMARY_LEARN_LOG_HEADER}\nNNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,-,-,0.1,0.1,0.1,1.000000,64,teacher.hcpe\n"
        ))
        .unwrap();

        args.validate_backend_flags().unwrap();
        let schedule = cuda_cpp_run_schedule(&args).unwrap();
        assert!(schedule.production);
        assert_eq!(schedule.prior_positions, 64);
        assert_eq!(schedule.total_steps, 2);
        assert_eq!(schedule.chunks.len(), 2);
        assert_eq!(schedule.chunks[0].epoch, 1);
        assert_eq!(schedule.chunks[0].superbatch, 2);
        assert_eq!(schedule.chunks[1].epoch, 1);
        assert_eq!(schedule.chunks[1].superbatch, 3);
        assert!(schedule.chunks[0].lr_start < 0.1, "mid-epoch resume should continue LR inside the epoch");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_run_schedule_continues_after_completed_epoch() {
        use clap::Parser as _;

        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-cuda-cpp-schedule-clean-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(tmp.join("0001")).unwrap();
        let output = tmp.to_str().unwrap();
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "teacher.hcpe",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "3",
            "--max-epochs",
            "1",
            "--save-rate",
            "1",
            "--batch-size",
            "64",
            "--positions-per-superbatch",
            "64",
            "--lr",
            "0.1",
            "--lr-min",
            "0.01",
            "--output",
            output,
        ])
        .unwrap();
        write_resume_config(&tmp, &args).unwrap();
        std::fs::write(tmp.join("0001").join("state.bin"), b"state").unwrap();
        std::fs::write(
            tmp.join("0001").join("learn.log"),
            format!(
            "{LEARN_LOG_HEADER}\nNNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,3,1,-,-,0.1,0.1,0.1,1.000000,192,teacher.hcpe\n"
        ),
        )
        .unwrap();
        std::fs::write(tmp.join(SUMMARY_LEARN_LOG_NAME), format!(
            "{SUMMARY_LEARN_LOG_HEADER}\nNNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,3,-,-,0.1,0.1,0.1,1.000000,192,teacher.hcpe\n"
        ))
        .unwrap();

        args.validate_backend_flags().unwrap();
        let schedule = cuda_cpp_run_schedule(&args).unwrap();
        assert!(schedule.production);
        assert_eq!(schedule.prior_positions, 192);
        assert_eq!(schedule.total_steps, 3);
        assert_eq!(schedule.chunks.len(), 3);
        assert_eq!(schedule.chunks[0].epoch, 2);
        assert_eq!(schedule.chunks[0].superbatch, 1);
        assert_eq!(schedule.chunks[2].epoch, 2);
        assert_eq!(schedule.chunks[2].superbatch, 3);
        assert!((schedule.chunks[0].lr_start - 0.1).abs() < 1e-6, "completed epoch should warm-restart LR");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cuda_cpp_backend_accepts_sfnn_halfka2_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_sfnn_halfka2_production_schedule() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "1",
            "--max-epochs",
            "1",
            "--batch-size",
            "1024",
            "--positions-per-superbatch",
            "1024",
            "--sfnn-factorized-l1",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_sfnn_halfka2_plateau_schedule() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "1",
            "--max-epochs",
            "1",
            "--batch-size",
            "1024",
            "--positions-per-superbatch",
            "1024",
            "--lr-schedule",
            "plateau",
            "--test-teacher",
            "yamaoka-floodgate.psv",
            "--save-rate",
            "1",
            "--sfnn-factorized-l1",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_sfnn_final_validation_teacher() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--test-teacher",
            "yamaoka-floodgate.psv",
            "--test-positions",
            "65536",
            "--test-sample",
            "sequential",
        ])
        .unwrap();

        assert_eq!(args.test_sample, TestSampleMode::Sequential);
        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_auto_resume_dataloader_pos_skips_changed_teacher() {
        use clap::Parser as _;

        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-cuda-cpp-direct-teacher-change-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let dir = tmp.join("0001");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("state.bin"), b"").unwrap();
        std::fs::write(dir.join("dataloader_pos.txt"), "76,0\n").unwrap();
        std::fs::write(
            dir.join("learn.log"),
            format!(
                "{LEARN_LOG_HEADER}\nNNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,1,-,-,0.100000,0.000875,0.000875,1.000000,2,old.hcpe\n"
            ),
        )
        .unwrap();
        let output = tmp.to_str().unwrap();

        let same_teacher = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "old.hcpe",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--output",
            output,
            "--resume",
        ])
        .unwrap();
        assert_eq!(
            cuda_cpp_auto_resume_dataloader_pos(&same_teacher).unwrap(),
            bulletou_lib::value::TeacherDataloaderPos { byte_offset: 76, plies: 0 }
        );

        let changed_teacher = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "new.hcpe",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--output",
            output,
            "--resume",
        ])
        .unwrap();
        assert_eq!(cuda_cpp_auto_resume_dataloader_pos(&changed_teacher), None);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cuda_cpp_backend_accepts_resume_flags_for_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--resume",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_rejects_explicit_weights_with_resume_flags() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--cuda-cpp-weights-bin",
            "weights.bin",
            "--resume",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") {
            "--cuda-cpp-weights-bin"
        } else {
            "cuda-cpp-backend"
        }));
    }

    #[test]
    fn cuda_cpp_backend_accepts_halfkp_final_validation_teacher() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--test-teacher",
            "yamaoka-floodgate.psv",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_rejects_sfnn_default_arch() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") {
            "SFNN_halfka2_1024_7_64_k3k3"
        } else {
            "cuda-cpp-backend"
        }));
    }

    #[test]
    fn cuda_cpp_backend_accepts_sfnn_initial_weights_for_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--cuda-cpp-weights-bin",
            "weights.bin",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_profile_steps_for_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "3",
            "--cuda-cpp-profile-steps",
            "2",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
            assert_eq!(args.cuda_cpp_profile_steps, 2);
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_teacher_prepare_profile_flag() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "3",
            "--cuda-cpp-profile-teacher-prepare",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
            assert!(args.cuda_cpp_profile_teacher_prepare);
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_sfnn_profile_steps_for_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "3",
            "--cuda-cpp-profile-steps",
            "2",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
            assert_eq!(args.cuda_cpp_profile_steps, 2);
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_accepts_skip_final_output_for_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "3",
            "--cuda-cpp-skip-final-output",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
            assert!(args.cuda_cpp_skip_final_output);
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_backend_rejects_skip_final_output_for_production_schedule() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--superbatches",
            "1",
            "--max-epochs",
            "1",
            "--cuda-cpp-skip-final-output",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.unwrap_err().contains("--cuda-cpp-skip-final-output"));
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_loss_readback_interval_controls_reporting_steps() {
        assert!(cuda_cpp_should_read_loss(1, 50, 10));
        assert!(cuda_cpp_should_read_loss(10, 50, 10));
        assert!(!cuda_cpp_should_read_loss(11, 50, 10));
        assert!(cuda_cpp_should_read_loss(50, 50, 10));

        assert!(!cuda_cpp_should_read_loss(1, 50, 0));
        assert!(!cuda_cpp_should_read_loss(49, 50, 0));
        assert!(cuda_cpp_should_read_loss(50, 50, 0));
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_teacher_cpu_defaults_are_autotuned_for_gpu_backend() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let default_threads = cuda_cpp_default_cpu_threads();
        assert!((4..=24).contains(&default_threads));
        assert_eq!(cuda_cpp_effective_teacher_threads(&args), default_threads);
        assert_eq!(cuda_cpp_effective_loader_threads(&args), default_threads);
        assert_eq!(cuda_cpp_effective_batch_queue_size(&args), 4);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_teacher_cpu_explicit_non_defaults_are_preserved() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--threads",
            "10",
            "--loader-threads",
            "12",
            "--batch-queue-size",
            "8",
        ])
        .unwrap();

        assert_eq!(cuda_cpp_effective_teacher_threads(&args), 10);
        assert_eq!(cuda_cpp_effective_loader_threads(&args), 12);
        assert_eq!(cuda_cpp_effective_batch_queue_size(&args), 8);
    }

    #[test]
    fn cuda_cpp_backend_accepts_initial_weights_for_direct_steps() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--cuda-cpp-weights-bin",
            "weights.bin",
        ])
        .unwrap();

        let result = args.validate_backend_flags();
        if cfg!(feature = "cuda-cpp-backend") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().contains("cuda-cpp-backend"));
        }
    }

    #[test]
    fn cuda_cpp_smoke_rejects_initial_weights() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-smoke",
            "--cuda-cpp-weights-bin",
            "weights.bin",
        ])
        .unwrap();

        let err = args.validate_backend_flags().unwrap_err();
        assert!(err.contains(if cfg!(feature = "cuda-cpp-backend") {
            "--cuda-cpp-weights-bin"
        } else {
            "cuda-cpp-backend"
        }));
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_halfkp_initial_weights_use_tatara_factorized_shape() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let weights = build_halfkp_initial_weights_for_cuda_cpp(&args).unwrap();
        let base_input_size = ShogiHalfKP.num_inputs();
        let virtual_rows = bulletou_lib::game::inputs::HALFKP_PIECE_INPUTS;
        assert_eq!(weights.shape.input_size, base_input_size + virtual_rows);
        assert!(weights.l0w[..virtual_rows * weights.shape.l1].iter().all(|&v| v == 0.0));
        assert!(weights.l0w[virtual_rows * weights.shape.l1..].iter().any(|&v| v != 0.0));
        assert!(weights.l0b.iter().any(|&v| v != 0.0));
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_kp_initial_weights_use_direct_kp_shape() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_KP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
        ])
        .unwrap();

        let weights = build_nnue_initial_weights_for_cuda_cpp(&args, CudaCppNnueFeatureKind::Kp).unwrap();
        assert_eq!(weights.shape.input_size, ShogiKp.num_inputs());
        assert_eq!(weights.l0w.len(), ShogiKp.num_inputs() * weights.shape.l1);
        assert!(weights.l0w.iter().any(|&v| v != 0.0));
        assert!(weights.l0b.iter().any(|&v| v != 0.0));
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_sfnn_initial_weights_use_halfka2_factorized_shape() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "1",
            "--sfnn-factorized-l1",
        ])
        .unwrap();

        let weights = build_sfnn_halfka2_initial_weights_for_cuda_cpp(&args).unwrap();
        let base_input_size = ShogiHalfKa2.num_inputs();
        let virtual_rows = bulletou_lib::game::inputs::PIECE_INPUTS;
        let l1_out = weights.shape.l1_out();
        assert_eq!(weights.shape.input_size, base_input_size + virtual_rows);
        assert!(weights.l0w[..base_input_size * weights.shape.ft_size].iter().any(|&v| v != 0.0));
        assert!(weights.l0w[base_input_size * weights.shape.ft_size..].iter().all(|&v| v == 0.0));
        assert!(weights.l1fw.as_deref().unwrap().iter().all(|&v| v == 0.0));
        assert!(weights.l1fb.as_deref().unwrap().iter().all(|&v| v == 0.0));

        let stack_stride = l1_out * weights.shape.ft_size;
        assert_eq!(&weights.l1w[..stack_stride], &weights.l1w[stack_stride..2 * stack_stride]);
        assert_eq!(&weights.l1b[..l1_out], &weights.l1b[l1_out..2 * l1_out]);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_halfkp_loads_ranger_optimizer_state_records() {
        use bulletou_lib::value::{NnueForwardOwnedWeights, NnueForwardShape as FastNnueForwardShape};

        let weights = NnueForwardOwnedWeights {
            shape: FastNnueForwardShape { input_size: 2, l1: 2, l2: 2, l3: 1 },
            l0w: vec![0.0; 4],
            l0b: vec![0.0; 2],
            l1w: vec![0.0; 8],
            l1b: vec![0.0; 2],
            l2w: vec![0.0; 2],
            l2b: vec![0.0; 1],
            outw: vec![0.0; 1],
            outb: vec![0.0; 1],
        };
        let mut records: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for (id, len) in [
            ("l0w", weights.l0w.len()),
            ("l0b", weights.l0b.len()),
            ("l1w", weights.l1w.len()),
            ("l1b", weights.l1b.len()),
            ("l2w", weights.l2w.len()),
            ("l2b", weights.l2b.len()),
            ("outw", weights.outw.len()),
            ("outb", weights.outb.len()),
        ] {
            records.insert(format!("nnue/momentum/{id}"), vec![1.0; len]);
            records.insert(format!("nnue/velocity/{id}"), vec![2.0; len]);
            records.insert(format!("nnue/slow/{id}"), vec![3.0; len]);
            records.insert(format!("nnue/step_ranger/{id}"), vec![42.0]);
        }

        let state = load_cuda_cpp_halfkp_optimizer_state(&records, &weights).unwrap().unwrap();
        assert_eq!(state.l0w.momentum, vec![1.0; 4]);
        assert_eq!(state.l1w.velocity, vec![2.0; 8]);
        assert_eq!(state.outb.slow_params, vec![3.0]);
        assert_eq!(load_cuda_cpp_halfkp_completed_steps(&records).unwrap(), 42);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_halfkp_weights_bin_writes_full_ranger_state() {
        fn group_state(base: f32) -> bulletou_cuda_cpp::RangerParamStateReadback {
            bulletou_cuda_cpp::RangerParamStateReadback {
                momentum: vec![base],
                velocity: vec![base + 0.25],
                slow_params: vec![base + 0.5],
            }
        }

        let weights = bulletou_cuda_cpp::NnueTrainWeightsReadback {
            l0w: vec![10.0],
            l0b: vec![11.0],
            l1w: vec![12.0],
            l1b: vec![13.0],
            l2w: vec![14.0],
            l2b: vec![15.0],
            outw: vec![16.0],
            outb: vec![17.0],
        };
        let optimizer = bulletou_cuda_cpp::NnueRangerOptimizerStatesReadback {
            l0w: group_state(1.0),
            l0b: group_state(2.0),
            l1w: group_state(3.0),
            l1b: group_state(4.0),
            l2w: group_state(5.0),
            l2b: group_state(6.0),
            outw: group_state(7.0),
            outb: group_state(8.0),
        };
        let path = std::env::temp_dir().join(format!(
            "bulletou-test-cuda-cpp-state-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        write_cuda_cpp_halfkp_weights_bin(&path, &weights, &optimizer, 7).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let records = parse_model_weights_bin(&bytes).unwrap();

        assert_eq!(records["nnue/weights/l0w"], vec![10.0]);
        assert_eq!(records["nnue/momentum/l0w"], vec![1.0]);
        assert_eq!(records["nnue/velocity/outb"], vec![8.25]);
        assert_eq!(records["nnue/slow/l2b"], vec![6.5]);
        assert_eq!(records["nnue/step_ranger/outw"], vec![7.0]);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_sfnn_weights_bin_writes_full_ranger_state() {
        fn group_state(base: f32) -> bulletou_cuda_cpp::RangerParamStateReadback {
            bulletou_cuda_cpp::RangerParamStateReadback {
                momentum: vec![base],
                velocity: vec![base + 0.25],
                slow_params: vec![base + 0.5],
            }
        }

        let weights = bulletou_cuda_cpp::SfnnTrainWeightsReadback {
            l0w: vec![10.0],
            l0b: vec![11.0],
            l1w: vec![12.0],
            l1b: vec![13.0],
            l1fw: Some(vec![14.0]),
            l1fb: Some(vec![15.0]),
            l2w: vec![16.0],
            l2b: vec![17.0],
            l3w: vec![18.0],
            l3b: vec![19.0],
        };
        let optimizer = bulletou_cuda_cpp::SfnnRangerOptimizerStatesReadback {
            l0w: group_state(1.0),
            l0b: group_state(2.0),
            l1w: group_state(3.0),
            l1b: group_state(4.0),
            l1fw: Some(group_state(5.0)),
            l1fb: Some(group_state(6.0)),
            l2w: group_state(7.0),
            l2b: group_state(8.0),
            l3w: group_state(9.0),
            l3b: group_state(10.0),
        };
        let path = std::env::temp_dir().join(format!(
            "bulletou-test-cuda-cpp-sfnn-state-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        write_cuda_cpp_sfnn_weights_bin(&path, &weights, &optimizer, 11).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let records = parse_model_weights_bin(&bytes).unwrap();

        assert_eq!(records["nnue/weights/l1fw"], vec![14.0]);
        assert_eq!(records["nnue/momentum/l1fw"], vec![5.0]);
        assert_eq!(records["nnue/velocity/l3b"], vec![10.25]);
        assert_eq!(records["nnue/slow/l2w"], vec![7.5]);
        assert_eq!(records["nnue/step_ranger/l1fb"], vec![11.0]);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_sfnn_loads_ranger_optimizer_state_records() {
        let shape =
            bulletou_cuda_cpp::SfnnForwardShape { input_size: 4, ft_size: 2, l1_hidden: 1, l2_size: 2, num_stacks: 2 };
        let weights = CudaCppSfnnInitialWeights {
            shape,
            l0w: vec![0.0; shape.input_size * shape.ft_size],
            l0b: vec![0.0; shape.ft_size],
            l1w: vec![0.0; shape.num_stacks * shape.l1_out() * shape.ft_size],
            l1b: vec![0.0; shape.num_stacks * shape.l1_out()],
            l1fw: Some(vec![0.0; shape.ft_size * shape.l1_out()]),
            l1fb: Some(vec![0.0; shape.l1_out()]),
            l2w: vec![0.0; shape.num_stacks * shape.l2_size * shape.l2_in()],
            l2b: vec![0.0; shape.num_stacks * shape.l2_size],
            l3w: vec![0.0; shape.num_stacks * shape.l2_size],
            l3b: vec![0.0; shape.num_stacks],
        };
        let mut records: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for (id, len) in [
            ("l0w", weights.l0w.len()),
            ("l0b", weights.l0b.len()),
            ("l1w", weights.l1w.len()),
            ("l1b", weights.l1b.len()),
            ("l1fw", weights.l1fw.as_ref().unwrap().len()),
            ("l1fb", weights.l1fb.as_ref().unwrap().len()),
            ("l2w", weights.l2w.len()),
            ("l2b", weights.l2b.len()),
            ("l3w", weights.l3w.len()),
            ("l3b", weights.l3b.len()),
        ] {
            records.insert(format!("nnue/momentum/{id}"), vec![1.0; len]);
            records.insert(format!("nnue/velocity/{id}"), vec![2.0; len]);
            records.insert(format!("nnue/slow/{id}"), vec![3.0; len]);
            records.insert(format!("nnue/step_ranger/{id}"), vec![42.0]);
        }

        let state = load_cuda_cpp_sfnn_optimizer_state(&records, &weights).unwrap().unwrap();
        assert_eq!(state.l0w.momentum, vec![1.0; shape.input_size * shape.ft_size]);
        assert_eq!(state.l1fw.as_ref().unwrap().velocity, vec![2.0; shape.ft_size * shape.l1_out()]);
        assert_eq!(state.l3b.slow_params, vec![3.0; shape.num_stacks]);
        assert_eq!(load_cuda_cpp_sfnn_completed_steps(&records, &weights).unwrap(), 42);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_halfkp_factorized_l0w_fold_adds_virtual_piece_rows() {
        let base_input_size = 4;
        let virtual_rows = 2;
        let l1 = 3;
        let weights = vec![
            10.0, 11.0, 12.0, // virtual piece 0
            20.0, 21.0, 22.0, // virtual piece 1
            1.0, 2.0, 3.0, // base row 0 -> piece 0
            4.0, 5.0, 6.0, // base row 1 -> piece 1
            7.0, 8.0, 9.0, // base row 2 -> piece 0
            30.0, 31.0, 32.0, // base row 3 -> piece 1
        ];

        let folded = fold_halfkp_piece_factorized_l0w(&weights, base_input_size, virtual_rows, l1).unwrap();

        assert_eq!(
            folded,
            vec![
                11.0, 13.0, 15.0, //
                24.0, 26.0, 28.0, //
                17.0, 19.0, 21.0, //
                50.0, 52.0, 54.0,
            ]
        );
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_halfkp_validation_weights_fold_factorized_l0w() {
        let base_input_size = ShogiHalfKP.num_inputs();
        let virtual_rows = bulletou_lib::game::inputs::HALFKP_PIECE_INPUTS;
        let shape =
            bulletou_cuda_cpp::NnueForwardShape { input_size: base_input_size + virtual_rows, l1: 1, l2: 1, l3: 1 };
        let mut l0w = vec![0.0; shape.input_size * shape.l1];
        l0w[0] = 10.0;
        l0w[virtual_rows] = 1.0;
        let weights = bulletou_cuda_cpp::NnueTrainWeightsReadback {
            l0w,
            l0b: vec![0.0],
            l1w: vec![1.0, 1.0],
            l1b: vec![0.0],
            l2w: vec![1.0],
            l2b: vec![0.0],
            outw: vec![1.0],
            outb: vec![0.0],
        };

        let validation_weights = cuda_cpp_halfkp_weights_for_cpu_validation(shape, &weights).unwrap();

        assert_eq!(validation_weights.shape.input_size, base_input_size);
        assert_eq!(validation_weights.l0w[0], 11.0);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_sfnn_halfka2_factorized_l0w_fold_adds_virtual_piece_rows() {
        let base_input_size = 4;
        let virtual_rows = 2;
        let ft_size = 3;
        let weights = vec![
            1.0, 2.0, 3.0, // base row 0 -> piece 0
            4.0, 5.0, 6.0, // base row 1 -> piece 1
            7.0, 8.0, 9.0, // base row 2 -> piece 0
            30.0, 31.0, 32.0, // base row 3 -> piece 1
            10.0, 11.0, 12.0, // virtual piece 0
            20.0, 21.0, 22.0, // virtual piece 1
        ];

        let folded = fold_sfnn_halfka2_piece_factorized_l0w(&weights, base_input_size, virtual_rows, ft_size).unwrap();

        assert_eq!(
            folded,
            vec![
                11.0, 13.0, 15.0, //
                24.0, 26.0, 28.0, //
                17.0, 19.0, 21.0, //
                50.0, 52.0, 54.0,
            ]
        );
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_sfnn_validation_l1f_fold_adds_shared_l1_to_each_stack() {
        let shape =
            bulletou_cuda_cpp::SfnnForwardShape { input_size: 4, ft_size: 3, l1_hidden: 1, l2_size: 2, num_stacks: 2 };
        let l1_out = shape.l1_out();
        let mut l1w = vec![
            1.0, 2.0, 3.0, // stack0 out0
            4.0, 5.0, 6.0, // stack0 out1
            7.0, 8.0, 9.0, // stack1 out0
            10.0, 11.0, 12.0, // stack1 out1
        ];
        let mut l1b = vec![0.5, 1.5, 2.5, 3.5];
        // Layout: shared_w[in_col * l1_out + out_col].
        let l1fw = vec![0.1, 0.2, 1.0, 2.0, 10.0, 20.0];
        let l1fb = vec![100.0, 200.0];

        fold_cuda_cpp_sfnn_l1f_into_stacked_l1(shape, &mut l1w, &mut l1b, Some(&l1fw), Some(&l1fb)).unwrap();

        assert_eq!(l1_out, 2);
        assert_eq!(l1b, vec![100.5, 201.5, 102.5, 203.5]);
        assert_eq!(
            l1w,
            vec![
                1.1, 3.0, 13.0, // stack0 out0
                4.2, 7.0, 26.0, // stack0 out1
                7.1, 9.0, 19.0, // stack1 out0
                10.2, 13.0, 32.0, // stack1 out1
            ]
        );
    }

    #[test]
    fn removed_rust_cuda_backend_cli_value_is_rejected() {
        use clap::Parser as _;

        let parsed = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--backend",
            "cuda-oxide",
        ]);

        assert!(parsed.is_err());
    }

    #[test]
    fn batches_per_superbatch_cli_option_is_removed() {
        use clap::Parser as _;

        let parsed = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--batches-per-superbatch",
            "5086",
        ]);

        assert!(parsed.is_err());
    }

    #[test]
    fn max_epochs_omitted_is_unlimited_for_all_schedules() {
        use clap::Parser as _;

        for schedule in ["step", "geometric", "cos", "plateau"] {
            let args = Args::try_parse_from([
                "bulletou",
                "--eval-type",
                "NNUE_HALFKP",
                "--teacher",
                "/dev/null",
                "--lr-schedule",
                schedule,
                "--test-teacher",
                "/tmp/test.hcpe",
            ])
            .unwrap();
            assert_eq!(effective_max_epochs(&args), usize::MAX, "schedule={schedule}");
        }

        let capped = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--lr-schedule",
            "cos",
            "--max-epochs",
            "3",
        ])
        .unwrap();
        assert_eq!(effective_max_epochs(&capped), 3);
    }

    #[test]
    fn optimizer_flags_feed_params_and_resume_signature() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--optimizer-weight-decay",
            "0.0",
            "--optimizer-epsilon",
            "0.0000001",
            "--optimizer-beta1",
            "0.85",
            "--optimizer-beta2",
            "0.995",
        ])
        .unwrap();

        let params = adamw_params(&args, BULLETOU_DEFAULT_ADAMW_CLIP);
        assert_eq!(args.optimizer, OptimizerKind::Ranger);
        assert_eq!(params.decay, 0.0);
        assert_eq!(params.epsilon, 0.0000001);
        assert_eq!(params.beta1, 0.85);
        assert_eq!(params.beta2, 0.995);

        let radam = radam_params(&args, BULLETOU_DEFAULT_ADAMW_CLIP);
        assert_eq!(radam.decay, 0.0);
        assert_eq!(radam.epsilon, 0.0000001);
        assert_eq!(radam.beta1, 0.85);
        assert_eq!(radam.beta2, 0.995);

        let ranger = ranger_params(&args, BULLETOU_DEFAULT_ADAMW_CLIP);
        assert_eq!(ranger.decay, 0.0);
        assert_eq!(ranger.epsilon, 0.0000001);
        assert_eq!(ranger.beta1, 0.85);
        assert_eq!(ranger.beta2, 0.995);

        let sig = resume_signature(&args);
        assert!(sig.contains("optimizer=ranger"));
        assert!(sig.contains("optimizer_weight_decay=0.000000000"));
        assert!(sig.contains("optimizer_epsilon=0.000000100"));
        assert!(sig.contains("optimizer_beta1=0.850000024"));
        assert!(sig.contains("optimizer_beta2=0.995000005"));
    }

    #[test]
    fn default_optimizer_matches_bullet_shogi_ranger_defaults() {
        use clap::Parser as _;

        let args = Args::try_parse_from(["bulletou", "--eval-type", "NNUE_HALFKP", "--teacher", "/dev/null"]).unwrap();

        assert_eq!(args.optimizer, OptimizerKind::Ranger);
        let adamw = adamw_params(&args, BULLETOU_DEFAULT_ADAMW_CLIP);
        let ranger = ranger_params(&args, BULLETOU_DEFAULT_ADAMW_CLIP);
        assert_eq!(adamw.beta1, 0.9);
        assert_eq!(ranger.beta1, 0.99);
        assert_eq!(ranger.beta2, 0.999);
    }

    #[test]
    fn default_lr_and_step_schedule_match_tatara_recipe() {
        use clap::Parser as _;

        let args = Args::try_parse_from(["bulletou", "--eval-type", "NNUE_HALFKP", "--teacher", "/dev/null"]).unwrap();

        assert_eq!(args.lr, 0.000875);
        assert_eq!(args.lr_schedule, LrScheduleKind::Step);
        assert_eq!(args.lr_step_gamma, None);
        let batches_per_superbatch = effective_batches_per_superbatch(&args).unwrap();
        let (gamma, auto) = effective_lr_step_gamma(&args, batches_per_superbatch).unwrap();
        assert_eq!(gamma, DEFAULT_LR_STEP_GAMMA);
        assert!(!auto);
        assert_eq!(args.lr_step_positions, None);
        assert_eq!(args.optimizer_weight_decay, 0.0);
        assert_eq!(args.scale, 290);
        assert_eq!(args.save_rate, None);
        assert_eq!(effective_save_rate(&args), DEFAULT_SAVE_RATE);
        assert!(args.save_epoch_end);
        assert!(!args.no_save_epoch_end);
        assert!(effective_save_epoch_end(&args));
    }

    #[test]
    fn omitted_step_gamma_auto_reaches_lr_min_within_one_epoch() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--superbatches",
            "15",
            "--max-epochs",
            "2",
            "--lr",
            "0.000875",
            "--lr-min",
            "0.00001",
        ])
        .unwrap();

        let batches_per_superbatch = effective_batches_per_superbatch(&args).unwrap();
        let (gamma, auto) = effective_lr_step_gamma(&args, batches_per_superbatch).unwrap();
        let expected = (0.00001_f64 / 0.000875_f64).powf(1.0 / 15.0) as f32;
        assert!(auto);
        assert!((gamma - expected).abs() < 1e-9, "expected {expected}, got {gamma}");
    }

    #[test]
    fn max_epoch_alias_is_accepted() {
        use clap::Parser as _;

        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--max-epoch",
            "3",
        ])
        .unwrap();

        assert_eq!(args.max_epochs, Some(3));
    }

    #[test]
    fn removed_step_gamma_schedule_name_is_rejected() {
        use clap::Parser as _;

        let parsed = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "NNUE_HALFKP",
            "--teacher",
            "/dev/null",
            "--lr-schedule",
            "step_gamma",
        ]);

        assert!(parsed.is_err());
    }

    #[test]
    fn latest_checkpoint_has_file_checks_latest_numbered_state_dir() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-latest-marker-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let d1 = tmp.join("0001");
        std::fs::create_dir(&d1).unwrap();
        std::fs::write(d1.join("state.bin"), b"dummy").unwrap();
        std::fs::write(d1.join(PLATEAU_EPOCH_DONE_NAME), b"1\n").unwrap();
        assert!(latest_checkpoint_has_file(&tmp, PLATEAU_EPOCH_DONE_NAME));

        let d2 = tmp.join("0002");
        std::fs::create_dir(&d2).unwrap();
        std::fs::write(d2.join("state.bin"), b"dummy").unwrap();
        assert!(!latest_checkpoint_has_file(&tmp, PLATEAU_EPOCH_DONE_NAME));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// finalize_one_nnue_dir with Some(TestMetrics) emits actual values
    /// in the test_value_* columns rather than `-`.
    #[test]
    fn finalize_one_nnue_dir_emits_test_metrics_when_provided() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-finalize-with-metrics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("shogi_sfnn_ka2-7");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("nn.bin"), b"dummy").unwrap();
        std::fs::write(src.join("state.bin"), b"dummy").unwrap();
        std::fs::write(src.join("log.txt"), "7,32,0.111\n7,64,0.099\n").unwrap();

        let ctx = LogContext {
            eval_type: "NNUE_KA2",
            arch: "NNUE_ka2_256x2_32_32".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/foo.hcpe".to_string(),
            epoch_offset: 0,
            lr_schedule: LrScheduleKind::Geometric,
            lr_period: 500_000_000,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.00001,
            lr_override: None,
        };
        let metrics = TestMetrics { accuracy: 0.8765, loss: 0.0512 };
        let dst = finalize_one_nnue_dir(&tmp, &src, &ctx, 1, 9, 0, Some(metrics), true).unwrap();
        let learn = std::fs::read_to_string(dst.join("learn.log")).unwrap();
        let body_lines: Vec<&str> = learn.lines().skip(1).filter(|l| !l.is_empty()).collect();
        assert_eq!(body_lines.len(), 2);
        // Validation runs once per superbatch, so only the LAST row of
        // each sb (here: the second of two rows for sb=7) carries the
        // metric. Earlier rows show `-` for the test_value_* columns.
        let cols0: Vec<&str> = body_lines[0].split(',').collect();
        assert_eq!(cols0.len(), 12);
        assert_eq!(cols0[4], "-", "non-boundary row's test_value_accuracy should be `-`");
        assert_eq!(cols0[5], "-", "non-boundary row's test_value_loss should be `-`");
        let cols1: Vec<&str> = body_lines[1].split(',').collect();
        assert_eq!(cols1.len(), 12);
        assert_eq!(cols1[4], "0.876500", "sb-boundary row carries test_value_accuracy");
        assert_eq!(cols1[5], "0.051200", "sb-boundary row carries test_value_loss");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `append_to_top_level_log` should keep only the LAST row of each
    /// (eval, sb) group so the top-level summary stays sb-granularity
    /// even though per-dir `learn.log` is per-batch granularity.
    #[test]
    fn append_to_top_level_log_keeps_only_sb_boundary_rows() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-toplog-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let dir = tmp.join("0001");
        std::fs::create_dir(&dir).unwrap();
        // Per-dir learn.log: 3 rows for sb=1, 2 rows for sb=2.
        let body = format!(
            "{header}\n\
             E,1,1,32,-,-,0.10,0.001,0.0009,1.000,524288,t.hcpe\n\
             E,1,1,64,-,-,0.09,0.001,0.0008,1.000,1048576,t.hcpe\n\
             E,1,1,96,0.50,0.30,0.08,0.001,0.0007,1.000,1572864,t.hcpe\n\
             E,1,2,32,-,-,0.07,0.001,0.0006,1.000,2097152,t.hcpe\n\
             E,1,2,64,0.55,0.28,0.06,0.001,0.0005,1.000,2621440,t.hcpe\n",
            header = LEARN_LOG_HEADER,
        );
        std::fs::write(dir.join("learn.log"), body).unwrap();
        append_to_top_level_log(&tmp, 1).unwrap();
        let top = std::fs::read_to_string(tmp.join(SUMMARY_LEARN_LOG_NAME)).unwrap();
        let lines: Vec<&str> = top.lines().collect();
        assert_eq!(lines[0], SUMMARY_LEARN_LOG_HEADER, "first line is summary header (no curr_batch)");
        // Two data rows: the sb=1 boundary row (b=96) and the sb=2
        // boundary row (b=64); intermediate rows dropped, and the
        // curr_batch column itself is also stripped — leaving 11 cols.
        assert_eq!(lines.len(), 3, "header + one row per sb, got {lines:?}");
        let cols1: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(cols1.len(), 11, "summary row has 11 cols (no curr_batch)");
        assert_eq!(cols1[2], "1", "first kept row is sb=1");
        // Index 3 is now `test_value_accuracy` (was `curr_batch`).
        assert_eq!(cols1[3], "0.50", "col 3 is test_value_accuracy (curr_batch dropped)");
        assert_eq!(cols1[6], "0.001", "lr_start is preserved");
        assert_eq!(cols1[7], "0.0007", "lr_end is preserved");
        let cols2: Vec<&str> = lines[2].split(',').collect();
        assert_eq!(cols2.len(), 11);
        assert_eq!(cols2[2], "2", "second kept row is sb=2");
        assert_eq!(cols2[3], "0.55", "col 3 is test_value_accuracy");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    fn cuda_cpp_direct_checkpoint_metadata_writes_learn_and_summary_logs() {
        use clap::Parser as _;

        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-cuda-cpp-direct-meta-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let dir = tmp.join("0001");
        std::fs::create_dir_all(&dir).unwrap();
        let args = Args::try_parse_from([
            "bulletou",
            "--eval-type",
            "SFNN_HALFKA2",
            "--arch",
            "SFNN_halfka2_1024_7_64_k3k3",
            "--teacher",
            "teacher.psv",
            "--backend",
            "cuda-cpp",
            "--cuda-cpp-train-steps",
            "3",
            "--batch-size",
            "5",
            "--test-sample",
            "sequential",
        ])
        .unwrap();
        let metrics = TestMetrics { accuracy: 0.625, loss: 0.125 };

        write_cuda_cpp_direct_checkpoint_metadata(
            &tmp,
            1,
            &dir,
            &args,
            CudaCppCheckpointLog {
                epoch: 1,
                superbatch: 1,
                curr_batch: 3,
                train_steps: 3,
                train_loss: 0.25,
                test_metrics: Some(metrics),
                lr_start: args.lr,
                lr_end: args.lr,
                dataloader_pos: bulletou_lib::value::TeacherDataloaderPos { byte_offset: 600, plies: 0 },
            },
        )
        .unwrap();

        assert_eq!(std::fs::read_to_string(dir.join("teacher.txt")).unwrap(), "teacher.psv\n");
        assert_eq!(std::fs::read_to_string(dir.join("dataloader_pos.txt")).unwrap(), "600,0\n");
        let learn = std::fs::read_to_string(dir.join("learn.log")).unwrap();
        let learn_lines = learn.lines().collect::<Vec<_>>();
        assert_eq!(learn_lines[0], LEARN_LOG_HEADER);
        assert!(learn_lines[1].starts_with("SFNN_HALFKA2-SFNN_halfka2_1024_7_64_k3k3,1,1,3,"));
        assert!(learn_lines[1].contains(",0.625000,0.125000,0.250000,"));
        assert!(learn_lines[1].contains(",15,teacher.psv"));

        let summary = std::fs::read_to_string(tmp.join(SUMMARY_LEARN_LOG_NAME)).unwrap();
        let summary_lines = summary.lines().collect::<Vec<_>>();
        assert_eq!(summary_lines[0], SUMMARY_LEARN_LOG_HEADER);
        assert_eq!(summary_lines.len(), 2);
        assert_eq!(summary_lines[1].split(',').count(), 11);
        assert!(summary_lines[1].starts_with("SFNN_HALFKA2-SFNN_halfka2_1024_7_64_k3k3,1,1,"));

        let dir2 = tmp.join("0002");
        std::fs::create_dir_all(&dir2).unwrap();
        write_cuda_cpp_direct_checkpoint_metadata(
            &tmp,
            2,
            &dir2,
            &args,
            CudaCppCheckpointLog {
                epoch: 1,
                superbatch: 2,
                curr_batch: 2,
                train_steps: 2,
                train_loss: 0.5,
                test_metrics: None,
                lr_start: args.lr,
                lr_end: args.lr,
                dataloader_pos: bulletou_lib::value::TeacherDataloaderPos { byte_offset: 1000, plies: 0 },
            },
        )
        .unwrap();

        let learn2 = std::fs::read_to_string(dir2.join("learn.log")).unwrap();
        let learn2_lines = learn2.lines().collect::<Vec<_>>();
        let learn2_cols = learn2_lines[1].split(',').collect::<Vec<_>>();
        assert_eq!(learn2_cols[10], "25");

        let summary2 = std::fs::read_to_string(tmp.join(SUMMARY_LEARN_LOG_NAME)).unwrap();
        let summary2_lines = summary2.lines().collect::<Vec<_>>();
        assert_eq!(summary2_lines.len(), 3);
        let summary2_cols = summary2_lines[2].split(',').collect::<Vec<_>>();
        assert_eq!(summary2_cols[9], "25");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// geometric schedule の検証: 1 epoch 末で lr_min、cycle 跨ぎで
    /// warm restart して lr_max に戻る。
    #[test]
    fn geometric_lr_warm_restarts() {
        let max = 0.001f32;
        let min = 0.00001f32;
        let period = 500_000_000u64;
        // t=0 → lr_max
        let lr = GeometricLR::lr_at_positions(max, min, period, 0);
        assert!((lr - max).abs() < 1e-7, "t=0 should be lr_max, got {lr}");
        // t=0.5 → log-space midpoint = sqrt(max * min)
        let lr = GeometricLR::lr_at_positions(max, min, period, period / 2);
        let geomean = (max as f64 * min as f64).sqrt() as f32;
        assert!((lr - geomean).abs() < 1e-6, "t=0.5 should be geomean {geomean}, got {lr}");
        // Just before cycle end → near lr_min
        let lr = GeometricLR::lr_at_positions(max, min, period, period - 1);
        assert!(lr < min * 1.1, "near t=1 should approach lr_min, got {lr}");
        // Exact cycle boundary → warm restart to lr_max
        let lr = GeometricLR::lr_at_positions(max, min, period, period);
        assert!((lr - max).abs() < 1e-7, "cycle boundary should warm-restart to lr_max, got {lr}");
    }

    /// step は指定局面数ごとに gamma を掛け、epoch 境界で lr_max に戻る。
    /// lr_min を下回ったら floor する。
    #[test]
    fn step_lr_drops_by_fixed_gamma_and_restarts_each_epoch() {
        let max = 0.000875f32;
        let min = 0.00001f32;
        let gamma = 0.992f32;
        let step = 100_000_000u64;
        let period = step * 3;

        let lr0 = StepLR::lr_at_positions(max, min, gamma, step, period, 0);
        assert!((lr0 - max).abs() < 1e-9, "initial lr should be max, got {lr0}");

        let lr_mid = StepLR::lr_at_positions(max, min, gamma, step, period, step - 1);
        assert!((lr_mid - max).abs() < 1e-9, "drop should happen only at the step boundary, got {lr_mid}");

        let lr1 = StepLR::lr_at_positions(max, min, gamma, step, period, step);
        assert!((lr1 - max * gamma).abs() < 1e-9, "first boundary should apply gamma once, got {lr1}");

        let lr2 = StepLR::lr_at_positions(max, min, gamma, step, period, step * 2);
        assert!((lr2 - max * gamma * gamma).abs() < 1e-9, "second boundary should apply gamma twice, got {lr2}");

        let lr_restart = StepLR::lr_at_positions(max, min, gamma, step, period, period);
        assert!((lr_restart - max).abs() < 1e-9, "epoch boundary should restart to lr_max, got {lr_restart}");

        let lr_floor = StepLR::lr_at_positions(max, min, gamma, step, 0, step * 10_000);
        assert_eq!(lr_floor, min, "step should floor at lr_min");
    }

    #[test]
    fn step_lr_with_one_superbatch_step_drops_from_next_superbatch() {
        let scheduler = StepLR {
            start: 0.000875,
            min: 0.00001,
            gamma: 0.992,
            step_positions: 4 * 16_384,
            period_positions: 8 * 16_384,
            prior_positions: 0,
            batch_size: 16_384,
            batches_per_superbatch: 4,
        };

        assert!((scheduler.lr(0, 1) - 0.000875).abs() < 1e-12);
        assert!((scheduler.lr(3, 1) - 0.000875).abs() < 1e-12);
        assert!((scheduler.lr(0, 2) - 0.000875 * 0.992).abs() < 1e-12);
        assert!((scheduler.lr(0, 3) - 0.000875).abs() < 1e-12);
    }

    /// CosineLR の式: lr_min + 0.5*(lr_max-lr_min)*(1+cos(πt)) が
    /// t=0 で lr_max、t=0.5 で midpoint、t=1 で lr_min を取ることを確認。
    /// warm restart (cycle 跨ぎ) で lr_max に戻ることも確認。
    #[test]
    fn cosine_lr_formula_endpoints_and_restart() {
        let max = 0.001f32;
        let min = 0.00001f32;
        let period = 500_000_000u64;
        // t=0 → lr_max
        let lr = CosineLR::lr_at_positions(max, min, period, 0);
        assert!((lr - max).abs() < 1e-7, "t=0 should be lr_max, got {lr}");
        // t=0.5 → midpoint
        let lr = CosineLR::lr_at_positions(max, min, period, period / 2);
        let mid = min + 0.5 * (max - min);
        assert!((lr - mid).abs() < 1e-6, "t=0.5 should be midpoint {mid}, got {lr}");
        // Just before cycle end → near lr_min
        let lr = CosineLR::lr_at_positions(max, min, period, period - 1);
        assert!(lr < min + 1e-5, "near t=1 should approach lr_min, got {lr}");
        // Exactly at cycle boundary → restart at lr_max (cycle index increments,
        // in_cycle = 0).
        let lr = CosineLR::lr_at_positions(max, min, period, period);
        assert!((lr - max).abs() < 1e-7, "cycle boundary should warm-restart to lr_max, got {lr}");
        // Second cycle behaves the same way.
        let lr = CosineLR::lr_at_positions(max, min, period, period + period / 2);
        assert!((lr - mid).abs() < 1e-6, "cycle 2 midpoint same as cycle 1, got {lr}");
    }

    fn pm(loss: f32) -> PlateauMetrics {
        PlateauMetrics { loss, accuracy: 0.50 }
    }

    fn pma(loss: f32, accuracy: f32) -> PlateauMetrics {
        PlateauMetrics { loss, accuracy }
    }

    fn plateau_loss_state() -> PlateauLrState {
        PlateauLrState::new(0.001, 0.0001, 0.5, 0.0, PlateauMonitor::Loss)
    }

    #[test]
    fn plateau_lr_reduces_and_schedules_final_min_run() {
        let mut s = plateau_loss_state();
        assert_eq!(s.observe(pm(0.50)), PlateauAction::First { metrics: pm(0.50) });
        assert_eq!(s.observe(pm(0.49)), PlateauAction::Improved { old_best: pm(0.50), new_best: pm(0.49) });
        assert_eq!(
            s.observe(pm(0.50)),
            PlateauAction::Reduced { old_lr: 0.001, new_lr: 0.0005, metrics: pm(0.50), best: pm(0.49) }
        );
        assert!((s.current_lr - 0.0005).abs() < 1e-12);

        assert_eq!(
            s.observe(pm(0.51)),
            PlateauAction::Reduced { old_lr: 0.0005, new_lr: 0.00025, metrics: pm(0.51), best: pm(0.49) }
        );
        assert_eq!(
            s.observe(pm(0.52)),
            PlateauAction::Reduced { old_lr: 0.00025, new_lr: 0.000125, metrics: pm(0.52), best: pm(0.49) }
        );
        assert_eq!(
            s.observe(pm(0.53)),
            PlateauAction::ScheduledFinal { old_lr: 0.000125, min_lr: 0.0001, metrics: pm(0.53), best: pm(0.49) }
        );
        assert_eq!(s.current_lr, 0.0001);
        assert_eq!(s.observe(pm(0.54)), PlateauAction::FinalRejected { metrics: pm(0.54), best: pm(0.49) });
    }

    #[test]
    fn plateau_lr_accepts_final_min_run_only_when_it_improves() {
        let mut s = plateau_loss_state();
        assert_eq!(s.observe(pm(0.50)), PlateauAction::First { metrics: pm(0.50) });
        assert_eq!(s.observe(pm(0.49)), PlateauAction::Improved { old_best: pm(0.50), new_best: pm(0.49) });
        assert_eq!(
            s.observe(pm(0.50)),
            PlateauAction::Reduced { old_lr: 0.001, new_lr: 0.0005, metrics: pm(0.50), best: pm(0.49) }
        );
        assert_eq!(
            s.observe(pm(0.51)),
            PlateauAction::Reduced { old_lr: 0.0005, new_lr: 0.00025, metrics: pm(0.51), best: pm(0.49) }
        );
        assert_eq!(
            s.observe(pm(0.52)),
            PlateauAction::Reduced { old_lr: 0.00025, new_lr: 0.000125, metrics: pm(0.52), best: pm(0.49) }
        );
        assert_eq!(
            s.observe(pm(0.53)),
            PlateauAction::ScheduledFinal { old_lr: 0.000125, min_lr: 0.0001, metrics: pm(0.53), best: pm(0.49) }
        );
        assert_eq!(s.current_lr, 0.0001);
        assert_eq!(s.observe(pm(0.48)), PlateauAction::FinalImproved { old_best: pm(0.49), new_best: pm(0.48) });
    }

    #[test]
    fn plateau_accuracy_monitor_accepts_accuracy_improvement_even_if_loss_worsens() {
        let mut s = PlateauLrState::new(0.001, 0.0001, 0.5, 0.0, PlateauMonitor::Accuracy);
        assert_eq!(s.observe(pma(0.50, 0.55)), PlateauAction::First { metrics: pma(0.50, 0.55) });
        assert_eq!(
            s.observe(pma(0.60, 0.56)),
            PlateauAction::Improved { old_best: pma(0.50, 0.55), new_best: pma(0.60, 0.56) }
        );
    }

    #[test]
    fn plateau_loss_or_accuracy_monitor_accepts_either_improvement() {
        let mut s = PlateauLrState::new(0.001, 0.0001, 0.5, 0.0, PlateauMonitor::LossOrAccuracy);
        assert_eq!(s.observe(pma(0.50, 0.55)), PlateauAction::First { metrics: pma(0.50, 0.55) });
        assert_eq!(
            s.observe(pma(0.60, 0.56)),
            PlateauAction::Improved { old_best: pma(0.50, 0.55), new_best: pma(0.60, 0.56) }
        );
        assert_eq!(
            s.observe(pma(0.49, 0.54)),
            PlateauAction::Improved { old_best: pma(0.60, 0.56), new_best: pma(0.49, 0.54) }
        );
    }

    #[test]
    fn plateau_retry_actions_rewind_teacher() {
        assert!(plateau_action_retries_teacher(PlateauAction::Reduced {
            old_lr: 0.001,
            new_lr: 0.0005,
            metrics: pm(0.50),
            best: pm(0.49),
        }));
        assert!(plateau_action_retries_teacher(PlateauAction::ScheduledFinal {
            old_lr: 0.000125,
            min_lr: 0.0001,
            metrics: pm(0.53),
            best: pm(0.49),
        }));
        assert!(!plateau_action_retries_teacher(PlateauAction::Improved { old_best: pm(0.50), new_best: pm(0.49) }));
        assert!(!plateau_action_retries_teacher(PlateauAction::FinalRejected { metrics: pm(0.54), best: pm(0.49) }));
    }

    #[test]
    fn plateau_reject_actions_drop_checkpoint() {
        assert!(plateau_action_rejects_update(PlateauAction::Reduced {
            old_lr: 0.001,
            new_lr: 0.0005,
            metrics: pm(0.50),
            best: pm(0.49),
        }));
        assert!(plateau_action_rejects_update(PlateauAction::ScheduledFinal {
            old_lr: 0.000125,
            min_lr: 0.0001,
            metrics: pm(0.53),
            best: pm(0.49),
        }));
        assert!(plateau_action_rejects_update(PlateauAction::FinalRejected { metrics: pm(0.54), best: pm(0.49) }));
        assert!(!plateau_action_rejects_update(PlateauAction::FinalImproved {
            old_best: pm(0.49),
            new_best: pm(0.48),
        }));
    }

    #[test]
    fn plateau_epoch_final_metrics_uses_accepted_metrics() {
        assert_eq!(
            plateau_action_epoch_final_metrics(PlateauAction::FinalImproved { old_best: pm(0.49), new_best: pm(0.48) }),
            Some(pm(0.48))
        );
        assert_eq!(
            plateau_action_epoch_final_metrics(PlateauAction::FinalRejected { metrics: pm(0.54), best: pm(0.49) }),
            Some(pm(0.49))
        );
        assert_eq!(
            plateau_action_epoch_final_metrics(PlateauAction::Reduced {
                old_lr: 0.001,
                new_lr: 0.0005,
                metrics: pm(0.50),
                best: pm(0.49),
            }),
            None
        );
    }

    #[test]
    fn epoch_final_stops_when_monitor_does_not_improve() {
        assert!(!epoch_final_should_stop(None, pm(0.50), PlateauMonitor::Loss, 0.0));
        assert!(!epoch_final_should_stop(Some(pm(0.50)), pm(0.49), PlateauMonitor::Loss, 0.0));
        assert!(epoch_final_should_stop(Some(pm(0.50)), pm(0.50), PlateauMonitor::Loss, 0.0));
        assert!(epoch_final_should_stop(Some(pm(0.50)), pm(0.51), PlateauMonitor::Loss, 0.0));
        assert!(!epoch_final_should_stop(Some(pma(0.50, 0.55)), pma(0.60, 0.56), PlateauMonitor::LossOrAccuracy, 0.0));
        assert!(epoch_final_should_stop(Some(pma(0.50, 0.55)), pma(0.60, 0.55), PlateauMonitor::LossOrAccuracy, 0.0));
    }

    #[test]
    fn enrich_uses_plateau_lr_override() {
        let ctx = LogContext {
            eval_type: "NNUE_KP",
            arch: "NNUE_kp_256x2_32_32".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/c.hcpe".to_string(),
            epoch_offset: 0,
            lr_schedule: LrScheduleKind::Plateau,
            lr_period: 0,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.00001,
            lr_override: Some(0.00025),
        };
        let body = enrich_bullet_log_to_csv("1,32,0.10\n", &ctx, 1, "nnue", 50_000_000, None, false);
        let cols: Vec<&str> = body.lines().next().unwrap().split(',').collect();
        let lr_start: f32 = cols[7].parse().unwrap();
        let lr_end: f32 = cols[8].parse().unwrap();
        assert!((lr_start - 0.00025).abs() < 1e-12, "plateau lr_start should use exact override, got {lr_start}");
        assert!((lr_end - 0.00025).abs() < 1e-12, "plateau lr_end should use exact override, got {lr_end}");
    }

    /// enrich path が `LrScheduleKind::Cos` のとき CosineLR を呼ぶことを確認。
    #[test]
    fn enrich_uses_cosine_when_schedule_is_cos() {
        let ctx = LogContext {
            eval_type: "NNUE_KP",
            arch: "NNUE_kp_256x2_32_32".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/c.hcpe".to_string(),
            epoch_offset: 0,
            lr_schedule: LrScheduleKind::Cos,
            lr_period: 100_000_000,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.0,
            lr_override: None,
        };
        // batch=32, prior=0 → positions = 524,288 → t ≈ 0.00524 → lr ≈ lr_max
        let body = enrich_bullet_log_to_csv("1,32,0.10\n", &ctx, 1, "nnue", 0, None, false);
        let cols: Vec<&str> = body.lines().next().unwrap().split(',').collect();
        let lr_start: f32 = cols[7].parse().unwrap();
        assert!(lr_start > 0.0009, "near cycle start, lr_start should be near lr_max=0.001, got {lr_start}");
        // Push to half a cycle = 50M positions → midpoint = 0.0005
        let body = enrich_bullet_log_to_csv("1,32,0.10\n", &ctx, 1, "nnue", 50_000_000, None, false);
        let cols: Vec<&str> = body.lines().next().unwrap().split(',').collect();
        let lr_start: f32 = cols[7].parse().unwrap();
        assert!((lr_start - 0.0005).abs() < 1e-4, "midpoint should be ~0.0005, got {lr_start}");
    }

    #[test]
    fn enrich_full_save_uses_exact_superbatch_boundary_and_lr_range() {
        let ctx = LogContext {
            eval_type: "SFNN_HALFKA2",
            arch: "SFNN_halfka2_1024_7_64_k3k3".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 2543,
            teacher_csv: "teachers/c.hcpe".to_string(),
            epoch_offset: 0,
            lr_schedule: LrScheduleKind::Cos,
            lr_period: 2543 * 12 * 16384,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.00001,
            lr_override: None,
        };

        // bullet's raw log is emitted every 32 batches, so a full
        // superbatch of 2543 batches may have its last raw row at 2528.
        // For a completed save, the enriched boundary row must still use
        // the exact superbatch end.
        let body = enrich_bullet_log_to_csv("1,2528,0.10\n", &ctx, 1, "nnue", 0, None, true);
        let cols: Vec<&str> = body.lines().next().unwrap().split(',').collect();
        assert_eq!(cols[3], "2543", "completed boundary row should display the exact sb batch count");
        assert_eq!(cols[10], "41664512", "positions should be exact bps * batch_size, not last raw log batch");
        assert_eq!(cols[7], "0.001000", "lr_start of sb1 should be lr_max");
        let lr_end: f32 = cols[8].parse().unwrap();
        assert!(lr_end < 0.001 && lr_end > 0.00098, "lr_end should be near the cosine value at sb end, got {lr_end}");
    }

    /// enrich の sb 列が bullet の local sb をそのまま表示すること、および
    /// LR / positions 列が prior_positions オフセット込みで計算されることを
    /// 確認。継続学習 run (= 前回までの positions を carry-over) の整合性。
    #[test]
    fn enrich_emits_local_sb_with_prior_positions_offset() {
        let ctx = LogContext {
            eval_type: "NNUE_KP",
            arch: "NNUE_kp_256x2_32_32".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/round2.hcpe".to_string(),
            epoch_offset: 0,
            lr_schedule: LrScheduleKind::Geometric,
            lr_period: 800_000_000,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.00001,
            lr_override: None,
        };
        // bullet's local sb in raw log is 1; enrich displays the local sb
        // verbatim (no cross-run shift — sb is per-epoch by design).
        let raw = "1,32,0.07\n1,64,0.06\n";
        let body =
            enrich_bullet_log_to_csv(&raw, &ctx, /*epoch=*/ 1, "nnue", /*prior=*/ 60_000_000, None, false);
        let rows: Vec<&str> = body.lines().collect();
        assert_eq!(rows.len(), 2);
        // Each row: eval, epoch, sb, batch, ta, tl, train,
        // lr_start, lr_end, lambda, positions, teacher
        let cols0: Vec<&str> = rows[0].split(',').collect();
        assert_eq!(cols0[2], "1", "sb column = bullet's local sb");
        // positions = prior 60M + b*batch_size = 60M + 32*16384 = 60_524_288
        // With geometric schedule, lr decays smoothly from positions 0.
        // lr_start uses the beginning of this row interval, while
        // lr_end uses the final batch start within the interval.
        let lr_start: f32 = cols0[7].parse().expect("lr_start col is a float");
        let lr_end: f32 = cols0[8].parse().expect("lr_end col is a float");
        assert!(lr_start > lr_end, "geometric lr should decrease within the row interval");
        assert_eq!(cols0[10], "60524288", "positions = prior + (local_sb-1)*sb_size + b*batch_size");
    }

    /// `LogContext.epoch_offset` が enrich の epoch 列に正しく加算され、
    /// 追加学習 run (= 前回の max_epoch を引き継ぐ) で epoch 表示が連続する
    /// ことを確認。bullet 側の local epoch counter は invocation ごとに 1 から
    /// 始まるので、offset を足さないと 1 にリセットされてしまうのが背景。
    #[test]
    fn enrich_with_epoch_offset_emits_absolute_epoch() {
        let ctx = LogContext {
            eval_type: "NNUE_HALFKP",
            arch: "NNUE_halfkp_256x2_32_32".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/foo.hcpe".to_string(),
            epoch_offset: 3, // = "previous run completed epoch 1..3 cleanly"
            lr_schedule: LrScheduleKind::Step,
            lr_period: 100_000_000,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.00001,
            lr_override: None,
        };
        let raw = "1,32,0.07\n";
        // local epoch=1 + offset 3 → display epoch=4
        let body = enrich_bullet_log_to_csv(&raw, &ctx, /*epoch=*/ 1, "nnue", 0, None, false);
        let cols: Vec<&str> = body.lines().next().unwrap().split(',').collect();
        assert_eq!(cols[1], "4", "absolute epoch (= local 1 + offset 3)");
    }

    /// `read_latest_epoch_in_top_level_log` が summary-learn.log の epoch 列
    /// (= index 1) の最大値を拾うことを確認。複数行を読んで max を返す。
    #[test]
    fn read_latest_epoch_picks_max() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-epoch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let log = tmp.join("summary-learn.log");

        // 存在しない → None
        assert_eq!(read_latest_epoch_in_top_level_log(&log), None);

        // header + 3 行 (epoch 1, 2, 3) → max = 3
        std::fs::write(
            &log,
            "eval,epoch,superbatch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher\n\
             NNUE,1,6,-,-,0.1,0.001,0.0008,1.0,100000000,t.hcpe\n\
             NNUE,2,6,-,-,0.1,0.001,0.0008,1.0,200000000,t.hcpe\n\
             NNUE,3,6,-,-,0.1,0.001,0.0008,1.0,300000000,t.hcpe\n",
        )
        .unwrap();
        assert_eq!(read_latest_epoch_in_top_level_log(&log), Some(3));

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn read_latest_nnue_test_metrics_picks_last_numeric_metrics() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-metrics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let log = tmp.join("summary-learn.log");

        assert_eq!(read_latest_nnue_test_metrics_in_top_level_log(&log), None);
        std::fs::write(
            &log,
            "eval,epoch,superbatch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher\n\
             KPPT/kk,1,1,0.1,0.999,0.1,0.001,0.0008,1.0,10,t.hcpe\n\
             NNUE,1,1,-,-,0.1,0.001,0.0008,1.0,20,t.hcpe\n\
             NNUE,1,2,0.5,0.123456,0.1,0.001,0.0008,1.0,30,t.hcpe\n\
             NNUE,2,1,0.5,0.120000,0.1,0.001,0.0008,1.0,40,t.hcpe\n",
        )
        .unwrap();
        assert_eq!(
            read_latest_nnue_test_metrics_in_top_level_log(&log),
            Some(PlateauMetrics { loss: 0.120000, accuracy: 0.5 })
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// `read_latest_saved_superbatch` が `<NNNN>/learn.log` の sb 列を
    /// 正しく拾うことを確認。auto-resume の出発点。
    #[test]
    fn read_latest_saved_superbatch_picks_max_sb() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-resume-sb-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
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
                "{LEARN_LOG_HEADER}\nNNUE_KP-NNUE_kp_256x2_32_32,1,1,32,0.1,0.001,1.000,524288,t.hcpe\n\
                 NNUE_KP-NNUE_kp_256x2_32_32,1,1,64,0.09,0.001,1.000,1048576,t.hcpe\n"
            ),
        )
        .unwrap();
        assert_eq!(read_latest_saved_superbatch(&tmp), Some(1));

        // 0004/ も追加 (sb=4 のログ) → 最高番号 dir の sb が返る
        let d4 = tmp.join("0004");
        std::fs::create_dir(&d4).unwrap();
        std::fs::write(
            d4.join("learn.log"),
            format!("{LEARN_LOG_HEADER}\nNNUE_KP-NNUE_kp_256x2_32_32,1,4,32,0.06,0.001,1.000,2097152,t.hcpe\n"),
        )
        .unwrap();
        assert_eq!(read_latest_saved_superbatch(&tmp), Some(4));

        // 不正 dir 名 (foo/) は無視
        std::fs::create_dir(tmp.join("foo")).unwrap();
        assert_eq!(read_latest_saved_superbatch(&tmp), Some(4));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `read_latest_saved_teacher` が `<NNNN>/learn.log` の teacher 列
    /// (= 12-列の最終フィールド) を取り、auto-resume の教師変更検出に
    /// 使えることを確認。
    #[test]
    fn read_latest_saved_teacher_picks_last_teacher() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-test-resume-teacher-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        // 空 dir → None
        assert_eq!(read_latest_saved_teacher(&tmp), None);

        // 0001 だけあって learn.log 無し → None
        let d1 = tmp.join("0001");
        std::fs::create_dir(&d1).unwrap();
        assert_eq!(read_latest_saved_teacher(&tmp), None);

        // 12-列 row が 1 つあれば teacher が拾える
        std::fs::write(
            d1.join("learn.log"),
            format!(
                "{LEARN_LOG_HEADER}\nNNUE_KP-NNUE_kp_256x2_32_32,1,1,32,-,-,0.1,0.001,0.0009,1.000,524288,foo.hcpe\n"
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
                "{LEARN_LOG_HEADER}\nNNUE_KP-NNUE_kp_256x2_32_32,1,4,32,0.6,0.05,0.06,0.001,0.0008,1.000,2097152,bar.hcpe\n"
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
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let ctx = LogContext {
            eval_type: "NNUE_KA2",
            arch: "NNUE_ka2_256x2_32_32".to_string(),
            lr_start: 0.001,
            lambda: 1.0,
            batch_size: 16384,
            batches_per_superbatch: 6104,
            teacher_csv: "teachers/foo.hcpe".to_string(),
            epoch_offset: 0,
            lr_schedule: LrScheduleKind::Step,
            lr_period: 500_000_000,
            lr_step_gamma: 0.992,
            lr_step_positions: 100_000_000,
            lr_min: 0.00001,
            lr_override: None,
        };
        let res = finalize_nnue_dirs(&tmp, &ctx, "shogi_nnue_ka2", 0).unwrap();
        assert_eq!(res, (0, 0), "empty dir should return (0,0), not Err");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn empty_leb128_block() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(LEB128_MAGIC);
        out.extend_from_slice(&0u32.to_le_bytes());
        out
    }

    fn fake_sfnn_nn_bin(arch: NnueArch, stacks: usize) -> Vec<u8> {
        let (ft_size, hidden1, hidden2) = arch.dims();
        let mut bytes = Vec::new();
        let desc = b"test-sfnn";
        bytes.extend_from_slice(&SFNN_NNUE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0x3C20_3B32u32.to_le_bytes());
        bytes.extend_from_slice(&(desc.len() as u32).to_le_bytes());
        bytes.extend_from_slice(desc);
        bytes.extend_from_slice(&0x5F13_4AB8u32.to_le_bytes());
        bytes.extend_from_slice(&empty_leb128_block());
        bytes.extend_from_slice(&empty_leb128_block());

        let l1_out = hidden1 + 1;
        let fc0_pad_in = nnue_pad32(ft_size);
        let fc1_real_in = hidden1 * 2;
        let fc1_pad_in = nnue_pad32(fc1_real_in);
        let fc2_pad_in = nnue_pad32(hidden2);
        for _ in 0..stacks {
            bytes.extend_from_slice(&0x6333_718Au32.to_le_bytes());
            bytes.resize(bytes.len() + l1_out * 4, 0);
            bytes.resize(bytes.len() + l1_out * fc0_pad_in, 0);
            bytes.resize(bytes.len() + hidden2 * 4, 0);
            bytes.resize(bytes.len() + hidden2 * fc1_pad_in, 0);
            bytes.resize(bytes.len() + 4, 0);
            bytes.resize(bytes.len() + fc2_pad_in, 0);
        }
        bytes
    }

    #[test]
    fn nerf_layer_set_parses_comma_list() {
        let layers: NerfLayerSet = "fc2,fc1".parse().unwrap();
        assert!(!layers.fc0);
        assert!(layers.fc1);
        assert!(layers.fc2);

        let layers: NerfLayerSet = "all".parse().unwrap();
        assert!(layers.fc0);
        assert!(layers.fc1);
        assert!(layers.fc2);

        assert!("fc3".parse::<NerfLayerSet>().is_err());
    }

    #[test]
    fn sfnn_nerf_collects_only_real_weight_bytes() {
        let arch =
            NnueArch::new(NnueArchFamily::Sfnn, NnueArchFeature::Halfka2, 32, 1, 4, Some(LayerStackMode::Kingrank3by3));
        let bytes = fake_sfnn_nn_bin(arch, LayerStackMode::Kingrank3by3.num_stacks());
        let (candidates, report) =
            collect_sfnn_nerf_candidates(&bytes, arch, LayerStackMode::Kingrank3by3, "fc1,fc2".parse().unwrap())
                .unwrap();

        // fc1: 9 stacks * hidden2(4) * real input(hidden1*2 = 2)
        // fc2: 9 stacks * hidden2(4)
        assert_eq!(report.fc0_candidates, 0);
        assert_eq!(report.fc1_candidates, 9 * 4 * 2);
        assert_eq!(report.fc2_candidates, 9 * 4);
        assert_eq!(candidates.len(), report.fc1_candidates + report.fc2_candidates);
    }

    #[test]
    fn sfnn_nerf_allows_repeated_weight_edits() {
        let arch =
            NnueArch::new(NnueArchFamily::Sfnn, NnueArchFeature::Halfka2, 32, 1, 4, Some(LayerStackMode::Kingrank3by3));
        let input = fake_sfnn_nn_bin(arch, LayerStackMode::Kingrank3by3.num_stacks());
        let args = NerfArgs {
            input: PathBuf::from("in.nn"),
            output: PathBuf::from("out.nn"),
            eval_type: NerfEvalType::Sfnn,
            arch,
            layers: "fc2".parse().unwrap(),
            count: 50,
            seed: 123,
        };
        let (output, report) = nerf_sfnn_bytes(input.clone(), &args).unwrap();
        assert_eq!(report.fc2_candidates, 9 * 4);
        assert!(report.selected > report.fc2_candidates, "test should exercise repeated selection");
        assert_eq!(report.selected, 50);
        assert_eq!(report.changed, 50);
        assert_eq!(report.saturated_noops, 0);

        let diffs = input.iter().zip(output.iter()).filter(|(a, b)| a != b).count();
        assert!(diffs <= report.fc2_candidates, "final differing bytes cannot exceed the candidate set");
        assert!(diffs > 0, "at least one byte should differ after repeated edits");
    }
}
