# NNUE K-P Training

<a href="../../ja/shogi/kp.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

[Back to Reference index](../README.md)

`--arch NNUE_kp_256x2_32_32` trains the YaneuraOu `kp_256x2-32-32` NNUE — the same 4-layer ClippedReLU network as [HalfKP](halfkp.md), but with a different input feature set.

The architecture file in YaneuraOu source is `source/eval/nnue/architectures/kp_256x2-32-32.h`, declaring `RawFeatures = FeatureSet<Features::K, Features::P>`.

## Architecture

L1 / L2 / L3 sizes are selected via `--arch NNUE_kp_<L1>x2_<L2>_<L3>` (same common sizes as NNUE_HALFKP, see [§4.3](../tutorial/4-train.md#43-specifying---arch)). YaneuraOu currently ships its `NNUE_kp_*` engine binaries only for `256x2-32-32`; the trainer will happily produce others if you want to experiment. On the CLI, use the full architecture name such as `NNUE_kp_256x2_32_32`. Below shows the default:

```
Shogi position
        │
        │  K + P sparse input (1,710 dims per perspective)
        ▼
   L0 affine + ClippedReLU                ← shared between own / opponent perspectives
        ▼
   accumulator (256 dims × 2 perspectives = 512 dims concatenated)
        │
        │  L1 affine (512 → 32) + ClippedReLU
        ▼
        │  L2 affine (32 → 32) + ClippedReLU
        ▼
        │  Out affine (32 → 1)
        ▼
      eval (centipawn-ish scalar)
```

## Input features

`FeatureSet<K, P>` composes two YaneuraOu feature sets:

| Sub-feature | Dim | Max active | Hash | Meaning |
|---|---|---|---|---|
| **K** (`features/k.h`) | 162 (= 81 × 2) | 2 | `0xD3CEE169` | Own king at one of 81 squares + opp king at one of 81 squares |
| **P** (`features/p.h`) | 1548 (= `fe_end`) | 38 | `0x764CFB4B` | Non-king pieces as BonaPiece values |

Total per perspective: **1710 dims**, max **40** active features (2 kings + up to 38 non-king pieces).

`FeatureSet<Head, Tail>` (`feature_set.h`) places Tail indices first and shifts Head indices by `Tail::kDimensions`, so:

- Indices `0 .. 1547` are P features (non-king pieces' BonaPiece values; index 0 unused per BonaPiece convention).
- Indices `1548 .. 1628` are K-own-king features (own king at square 0..80, perspective-rotated).
- Indices `1629 .. 1709` are K-opp-king features (opp king at square 0..80, perspective-rotated).

The combined feature hash, used in the `nn.bin` header:
```
FeatureSet<K, P>::kHashValue
  = K::kHashValue ^ (P::kHashValue << 1) ^ (P::kHashValue >> 31)
  = 0xD3CEE169 ^ 0xEC99F696
  = 0x3F5717FF
```

## Difference vs. HalfKP

| | HalfKP | K-P |
|---|---|---|
| Input dim / perspective | 125,388 (= 81 × 1548) | 1,710 (= 162 + 1548) |
| Cross product? | Yes — every (king sq × piece) is its own feature | No — K and P are concatenated, no interaction at the feature level |
| L0 weight size | 125,388 × 256 | 1,710 × 256 |
| Expressive power | Higher (king × piece correlations baked in) | Lower (must learn correlations through L0+L1) |

KP was introduced alongside HalfKP in the same architecture family, with the same 4-layer ClippedReLU network. HalfKP became the canonical choice because the cross-product input produced stronger play; KP is kept for completeness and for ablation experiments.

## Actual usage

### Command

```bash
./target/release/examples/bulletou \
    --arch NNUE_kp_256x2_32_32 \
    --teacher teachers/ \
    --output checkpoints/my-kp
```

Everything else (training schedule flags, save layout, resume from `state.bin`, top-level `summary-learn.log`) is identical to [HalfKP](halfkp.md) — only `--arch` differs.

### Save layout

```
checkpoints/my-kp/
├── summary-learn.log
├── 0001/
│   ├── nn.bin                         ← YaneuraOu / Stockfish (nnue-pytorch) compatible NNUE binary
│   ├── state.bin                      ← resume data (weights + Ranger optimizer state)
│   └── learn.log
├── 0002/
├── ...
└── 000N/
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

### `nn.bin` format

Same binary layout as [HalfKP's `nn.bin`](halfkp.md#nnbin-format); the only differences are:

- Header `network_hash` and `feature_transformer_hash` differ (they include `FEATURE_HASH_KP = 0x3F5717FF` instead of HalfKP's `0x5D69D5B8`).
- Header `description` string starts `Features=K-P(Friend)[1710->256x2],...` instead of `Features=HalfKP(Friend)[125388->256x2],...`.
- L0 size: `1710 × 256` (i16) instead of `125388 × 256`.

L1 / L2 / Output layers are byte-identical between HalfKP and KP for the same `--arch` preset (same sizes, same i8 row-major SIMD-padded layout).

### Common CLI flags

| Flag | Meaning | Default |
|---|---|---|
| `--arch` | `NNUE_kp_256x2_32_32`, `NNUE_kp_384x2_8_96`, `NNUE_kp_512x2_8_64`, `NNUE_kp_768x2_16_64`, `NNUE_kp_1024x2_8_32`, `NNUE_kp_1024x2_8_64` | (required; target `NNUE_KP` is inferred) |
| `--teacher` | Teacher file / directory / comma-separated list | (required) |
| `--output` | Checkpoint parent directory | `checkpoints/<target>-<arch>` (e.g. `checkpoints/NNUE_KP-NNUE_kp_256x2_32_32`) |

See [HalfKP Training](halfkp.md) for the full flag list (it is identical between NNUE_HALFKP and NNUE_KP).
