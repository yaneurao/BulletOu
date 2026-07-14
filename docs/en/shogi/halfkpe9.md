# NNUE HalfKPE9 Training

<a href="../../ja/shogi/halfkpe9.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

[Back to Reference index](../README.md)

`--eval-type NNUE_HALFKPE9` trains YaneuraOu's `halfkpe9_*` evaluation function. HalfKP's `(own_king_sq, piece_bonapiece)` pair is augmented with the **per-square attacker counts (own / opponent, 0/1/2-clipped, 9 combinations)** for that piece's square. The 4-layer ClippedReLU dual-perspective network is the same as HalfKP / K-P.

Network structure is identical to HalfKP / K-P, but the input dimension is **9× HalfKP** (125,388 → 1,128,492), so the L0 weight matrix grows 9×. GPU memory and training time scale accordingly.

## Architecture

`--arch NNUE_halfkpe9_<L1>x2_<L2>_<L3>` selects L1 / L2 / L3 (same set of common sizes as NNUE_HALFKP, see [§4.3](../tutorial/4-train.md#43-specifying---arch)):

```
shogi position
       │
       ▼ HalfKPE9 sparse input (1,128,492 dims = 81 × 1548 × 9)
       │
       ▼ L0 affine + ClippedReLU       ← weights shared across own / opp perspectives
       │
       ▼ accumulator (L1 × 2 perspectives)
       │
       ▼ L1 affine + ClippedReLU
       ▼ L2 affine + ClippedReLU
       ▼ Out affine
       │
       ▼ eval (centipawn-ish scalar)
```

## Input features

`HalfKP × 9 effect-count buckets`:

| Axis | Range | Meaning |
|---|---|---|
| **king_sq** | 0..80 | own king square (perspective-rotated) |
| **bonapiece** | 0..1547 | piece BonaPiece value (perspective-rotated) |
| **effect bucket** | 0..8 | `(effect1 × 3 + effect2)` attacker-count combo |

`effect1` = number of **own-side** attackers on that piece's square (clipped to 0/1/2 from perspective). `effect2` = same for **opponent-side**.

Active index formula (matches YaneuraOu's `MakeIndex` exactly):

```
index = fe_end × king_sq + bonapiece
      + fe_end × SQ_NB × (effect1 × 3 + effect2)
```

- `fe_end` = 1548
- `SQ_NB` = 81

### Attack counting

Per training position, a 81 × 2 = 162-cell attacker-count table is built once (`compute_effect_counts`):
- All piece types (including king) are enumerated; `for_each_attack()` lists each piece's destination squares
- Slider pieces (bishop / rook / horse / dragon) handle blocker occlusion correctly (the existing routine inherited from the `shogi_halfka_hm_threat` module)

`for_each_attack` is the same utility used by the Threat-feature family — HalfKPE9 doesn't add new attack code.

### Hand pieces

Hand pieces don't sit on a board square, so `effect1 = effect2 = 0` (bucket 0). The `(king_sq, hand_bonapiece, 0, 0)` slice of the feature space is therefore HalfKP-equivalent.

### Dimensions and feature hash

| Item | Value |
|---|---|
| dim | 81 × 1548 × 9 = **1,128,492** |
| max_active | 38 |
| FEATURE_HASH | **`0x5D69D5B8`** (same as HalfKP Friend) |

YaneuraOu's `kHashValue` in `features/half_kpe9.h` is literally `0x5D69D5B9 ^ (Friend == 1)`, which collides with HalfKP's. **Identification is done via the description string (`HalfKPE9(Friend)`) and the input dimension**.

## Comparison vs HalfKP

| | HalfKP | HalfKPE9 |
|---|---|---|
| Input dim / perspective | 125,388 | 1,128,492 (= × 9) |
| L0 weight matrix (L1=256) | 125,388 × 256 ≈ 32M | 1,128,492 × 256 ≈ 290M |
| Attack info | no | own 0/1/2 × opp 0/1/2 (9 combos) |
| Expressive power | king pos × piece pos | king pos × piece pos × attack counts |
| Training time | baseline | several × HalfKP |

## Actual usage

### Command

```bash
# Build (once)
cargo build --release --features device-cuda --example bulletou

# Run
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKPE9 \
    --teacher teachers/
```

Default `--output` is `checkpoints/NNUE_HALFKPE9-NNUE_halfkpe9_256x2_32_32/`.

### Save layout

Identical to HalfKP:

```
checkpoints/NNUE_HALFKPE9-NNUE_halfkpe9_256x2_32_32/
├── learn.log                          ← 9-column CSV, cumulative across runs/resumes
├── 0001/
│   ├── nn.bin                         ← YaneuraOu / Stockfish (nnue-pytorch) compatible NNUE binary
│   ├── state.bin                      ← resume data
│   └── learn.log                      ← per-save snapshot
├── ...
└── 000N/
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

### `nn.bin` format

Same binary layout as [NNUE HalfKP's `nn.bin`](halfkp.md#nnbin-format). Differences:
- Description string begins `Features=HalfKPE9(Friend)[1128492->...x2]`
- Input dim is 1,128,492 (HalfKP's is 125,388)
- L0 weight matrix is 9× larger

L1 / L2 / Out layers are byte-identical to HalfKP for the same `--arch`.

### Common CLI flags

| Flag | Meaning | Default |
|---|---|---|
| `--eval-type` | `NNUE_HALFKPE9` | (required) |
| `--arch` | `NNUE_halfkpe9_256x2_32_32`<br>`NNUE_halfkpe9_384x2_8_96`<br>`NNUE_halfkpe9_512x2_8_64`<br>`NNUE_halfkpe9_768x2_16_64`<br>`NNUE_halfkpe9_1024x2_8_32`<br>`NNUE_halfkpe9_1024x2_8_64` | `NNUE_halfkpe9_256x2_32_32` |
| `--teacher` | Teacher file / directory / comma-separated | (required) |
| `--output` | Checkpoint parent directory | `checkpoints/<eval-type>-<arch>` |
| `--lambda` | Blend between teacher eval and W/D/L | 1.0 |

Full flag list: see [HalfKP Training](halfkp.md) (all four NNUE eval types share the same flags).

## Caveats

- **L0 is large — watch GPU memory**. A 1024x2 HalfKPE9 needs 9× the L0 memory of HalfKP. 16 GB+ GPU recommended.
- Attack counting runs on the dataloader threads (CPU side). Shares the LUT-based optimisations of the Threat-feature family.
- The receiving engine must be a YaneuraOu build configured for `halfkpe9_*` (e.g. `EVAL_KPP_NN_HALFKPE9`).
