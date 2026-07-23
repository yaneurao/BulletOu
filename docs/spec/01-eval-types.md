# 01. `--arch` training target specification

BulletOu no longer exposes `--eval-type` on the training CLI. The single
training selector is `--arch`:

- `--arch KPPT`
- `--arch KPP_KKPT`
- `--arch NNUE_<feature>_<L1>x2_<L2>_<L3>`
- `--arch SFNN_<feature>_<FT>_<H1>_<H2>[_gN|_cN_sMxG]_<k3k3|k9k9|hand64|hand64_k3k3|hand64_k9k9|hand256|hand256_k3k3|hand256_k9k9|hand1024|hand1024_k3k3|hand1024_k9k9>`

The old eval-type names still exist internally as checkpoint/log target
identifiers so existing output directory names and resume signatures remain
stable, but users do not pass them on the command line.

## Public training targets

| `--arch` form | inferred target | family | output files per save dir | engine-loadable |
|---|---|---|---|---|
| `KPPT` | `KPPT` | KPPT | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` | yes |
| `KPP_KKPT` | `KPP_KKPT` | KPPT (factorised KPP) | same, but KPP omits the side-to-move channel | yes |
| `NNUE_halfkp_<L1>x2_<L2>_<L3>` | `NNUE_HALFKP` | NNUE | `nn.bin` | yes |
| `NNUE_kp_<L1>x2_<L2>_<L3>` | `NNUE_KP` | NNUE | `nn.bin` | yes |
| `NNUE_ka2_<L1>x2_<L2>_<L3>` | `NNUE_KA2` | NNUE | `nn.bin` | yes, with matching `YANEURAOU_ENGINE_NNUE_ka2_*` build |
| `NNUE_halfkpe9_<L1>x2_<L2>_<L3>` | `NNUE_HALFKPE9` | NNUE | `nn.bin` | yes |
| `NNUE_halfkpvm_<L1>x2_<L2>_<L3>` | `NNUE_HALFKPVM` | NNUE | `nn.bin` | yes |
| `SFNN_halfkahm1_<FT>_<H1>_<H2>[_gN\|_cN_sMxG]_<stack>` | `SFNN_HALFKA1HM` | SFNN LayerStacks | `nn.bin` | ablation / experimental |
| `SFNN_halfkahm2_<FT>_<H1>_<H2>[_gN\|_cN_sMxG]_<stack>` | `SFNN_HALFKA2HM` | SFNN LayerStacks | `nn.bin` | yes, with matching YaneuraOu SFNN build |
| `SFNN_halfka2_<FT>_<H1>_<H2>[_gN\|_cN_sMxG]_<stack>` | `SFNN_HALFKA2` | SFNN LayerStacks | `nn.bin` | yes, with matching `YANEURAOU_ENGINE_SFNN_halfka2_*` build |
| `SFNN_ka2_<FT>_<H1>_<H2>[_gN\|_cN_sMxG]_<stack>` | `SFNN_KA2` | SFNN LayerStacks | `nn.bin` | yes, with matching `YANEURAOU_ENGINE_SFNN_ka2_*` build |

Every target writes `state.bin` for resume and `learn.log` for the per-save
loss/validation snapshot. Details are in
[04-checkpoint-layout.md](04-checkpoint-layout.md).

## Why target names are inferred from `--arch`

For NNUE/SFNN networks the architecture string already contains both pieces of
information that matter:

- the network family (`NNUE` or `SFNN`);
- the input feature module (`halfkp`, `kp`, `ka2`, `halfkpe9`,
  `halfkpvm`, `halfka2`, `halfkahm1`, `halfkahm2`).

Therefore `--arch SFNN_ka2_8192_7_64_g8_k3k3` unambiguously means target
`SFNN_KA2`. Requiring a second `--eval-type SFNN_KA2` flag made the CLI more
error-prone without adding information.

KPPT-family evals do not have a YaneuraOu NNUE architecture name, so they are
represented directly as special `--arch` values: `KPPT` and `KPP_KKPT`.

## Internal helper components

KPPT-family training internally runs the three components `kk`, `kkp`, and
`kpp` in sequence, but these are not public `--arch` values. YaneuraOu loads
the assembled triplet only, so single-component output directories are not
useful as user-facing training targets.

## Supported architecture grammar

### NNUE

```text
NNUE_<feature>_<L1>x2_<L2>_<L3>
```

Supported NNUE features:

- `halfkp`
- `kp`
- `ka2`
- `halfkpe9`
- `halfkpvm`

`L1` must be a multiple of 32 for feature-transformer SIMD padding.

### SFNN

```text
SFNN_<feature>_<FT>_<H1>_<H2>[_gN|_cN_sMxG]_<stack>
```

Supported SFNN features:

- `halfkahm1`
- `halfkahm2`
- `halfka2`
- `ka2`

`<stack>` selects the YaneuraOu-compatible LayerStack bucket algorithm:

- `k3k3` / `king3_by_king3`: 9 buckets by friend/enemy king ranks.
- `k9k9` / `king9_by_king9`: 81 buckets by exact friend/enemy king ranks.
- `hand64`: 64 buckets by side-to-move / non-side hand-score buckets.
- `hand64_k3k3` / `hand64_king3_by_king3`: 64 hand buckets × 9 king buckets = 576 stacks.
- `hand64_k9k9` / `hand64_king9_by_king9`: 64 hand buckets × 81 king buckets = 5184 stacks.
- `hand256`: 256 buckets by side-to-move / non-side 4-bit hand-presence buckets.
- `hand256_k3k3` / `hand256_king3_by_king3`: 256 hand buckets × 9 king buckets = 2304 stacks.
- `hand256_k9k9` / `hand256_king9_by_king9`: 256 hand buckets × 81 king buckets = 20736 stacks.
- `hand1024`: 1024 buckets by side-to-move / non-side 5-bit hand-presence buckets.
- `hand1024_k3k3` / `hand1024_king3_by_king3`: 1024 hand buckets × 9 king buckets = 9216 stacks.
- `hand1024_k9k9` / `hand1024_king9_by_king9`: 1024 hand buckets × 81 king buckets = 82944 stacks.

`gN` enables the grouped L1 variants, where the FT dimension and `H1 + 1` must
both be divisible by `N`. `_cN_sMxG` enables common+shard L1: the first `N` FT
channels are common to every L1 output group, then `G` shard blocks of `M`
channels follow. `N` may be `0`; for example `c0_s1024x8` is equivalent to a
pure 8-way grouped L1. It requires `N + M * G == FT`, `(H1 + 1) % G == 0`,
and both `N` and `M` to be multiples of 64. The `+1` is the PSQT shortcut
neuron added after H1.

For `_gN_` and `_cN_sMxG_`, BulletOu keeps `l1w` compact in `state.bin` and
optimizer state, then expands the L1 matrix to the dense YaneuraOu `fc_0`
layout when writing `nn.bin`.

Examples currently used for experiments:

- `SFNN_halfka2_4096_7_64_g4_k3k3`
- `SFNN_halfka2_4096_3_64_g4_k3k3`
- `SFNN_halfka2_8192_3_64_g4_k3k3`
- `SFNN_halfka2_8192_7_64_g8_k3k3`
- `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3`
- `SFNN_halfka2_4096_15_64_g16_k3k3`
- `SFNN_halfka2_1024_7_64_hand64`
- `SFNN_halfka2_1024_7_64_hand64_k3k3`
- `SFNN_halfka2_1024_7_64_k9k9`
- `SFNN_halfka2_1024_7_64_hand64_k9k9`
- `SFNN_halfka2_1024_7_64_hand256`
- `SFNN_halfka2_1024_7_64_hand256_k3k3`
- `SFNN_halfka2_1024_7_64_hand256_k9k9`
- `SFNN_halfka2_1024_7_64_hand1024`
- `SFNN_halfka2_1024_7_64_hand1024_k3k3`
- `SFNN_halfka2_1024_7_64_hand1024_k9k9`
- `SFNN_ka2_8192_7_64_g8_k3k3`
- `SFNN_ka2_32768_15_64_g16_k3k3`
- `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3`

## Default output directory

When `--output` is omitted, BulletOu derives the checkpoint root from the
inferred internal target and the arch value:

| command target | default `--output` |
|---|---|
| `--arch KPPT` | `checkpoints/KPPT` |
| `--arch KPP_KKPT` | `checkpoints/KPP_KKPT` |
| `--arch NNUE_halfkp_256x2_32_32` | `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32` |
| `--arch NNUE_kp_256x2_32_32` | `checkpoints/NNUE_KP-NNUE_kp_256x2_32_32` |
| `--arch SFNN_halfka2_1024_7_64_k3k3` | `checkpoints/SFNN_HALFKA2-SFNN_halfka2_1024_7_64_k3k3` |
| `--arch SFNN_ka2_8192_7_64_g8_k3k3` | `checkpoints/SFNN_KA2-SFNN_ka2_8192_7_64_g8_k3k3` |

`--tag TAG` appends `-TAG` to this auto-derived name. Explicit `--output`
always wins.

## Activation summary

KPPT and the classic NNUE targets use ClippedReLU. SFNN targets use the
LayerStacks path with the CReLU + SqrClippedReLU pair after `fc_0`; see
[05-activation-history.md](05-activation-history.md).
