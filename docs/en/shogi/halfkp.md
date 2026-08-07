# NNUE HalfKP Training

<a href="../../ja/shogi/halfkp.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

[Back to Reference index](../README.md)

`--arch NNUE_halfkp_256x2_32_32` trains the classic YaneuraOu HalfKP NNUE — dual-perspective HalfKP feature transformer + 4 ClippedReLU layers. This is the evaluation function family YaneuraOu has supported the longest.

The activation-function history (why ClippedReLU and not SCReLU) is documented in [`spec/05-activation-history.md`](../../spec/05-activation-history.md).

## Architecture

L1 / L2 / L3 sizes are selected via `--arch NNUE_halfkp_<L1>x2_<L2>_<L3>` (`L1` must be a multiple of 32). Common YaneuraOu-shipped sizes: `256x2-32-32` (default), `384x2-8-96`, `512x2-8-64`, `768x2-16-64`, `1024x2-8-32`, `1024x2-8-64`. On the CLI, use the full architecture name such as `NNUE_halfkp_256x2_32_32`. See [tutorial: Run the training](../tutorial/3-train.md) for the basic command shape. Below shows the default:

```
HalfKP sparse input (125,388 dims, per perspective)
        │
        │  L0 affine + ClippedReLU       ← shared between own / opponent perspectives
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

## Actual usage

### Prerequisites

- BulletOu built (`cargo build --release --features cuda-cpp-backend --example bulletou`)
- Training data (`.hcpe` / `.hcpe3` / `.pack` / `.psv` / `.bin`)

### Command

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_256x2_32_32 \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-halfkp
```

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader reaches EOF). For multi-epoch runs, set the epoch length with `--superbatches` and then pass `--max-epochs N`. `step` / `geometric` / `cos` restart to `--lr` at epoch boundaries.

### Save layout

```
checkpoints/my-halfkp/
├── summary-learn.log                  ← top-level cumulative log across runs/resumes
├── 0001/
│   ├── nn.bin                         ← YaneuraOu / Stockfish (nnue-pytorch) compatible NNUE binary
│   ├── state.bin                      ← resume data (weights + Ranger optimizer state)
│   └── learn.log                      ← snapshot of the training log at this save point
├── 0002/
│   ├── ...
├── ...
└── 000N/                              ← the most recent save
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

Point a YaneuraOu HalfKP engine at the latest numbered directory (`000N/nn.bin`). The engine ignores `state.bin`.

### `nn.bin` format

The file is the nnue-pytorch / Stockfish binary format, byte-identical to what `nnue-pytorch`'s `serialize.py` produces. Layout:

- Header: `NNUE_VERSION` = `0x7AF32F16` (u32 LE), `network_hash` (u32 LE), `desc_len` (u32 LE), `description` (UTF-8 bytes)
- Feature Transformer layer hash (u32 LE)
- L0 biases (i16 × L1)
- L0 weights (i16 × INPUT × L1)
- Network layer hash (u32 LE)
- L1: biases (i32 × L2), weights (i8 × L2 × pad32(L1×2), row-major)
- L2: biases (i32 × L3), weights (i8 × L3 × pad32(L2), row-major)
- Output: biases (i32 × 1), weights (i8 × 1 × pad32(L3), row-major)

`pad32(n) = ceil(n/32) * 32` aligns each layer's input dim to 32 bytes for SIMD inference. Quantisation: L0 uses `qa = 127` (ClippedReLU output range is 0..127), L1-Out use `qb = 64` for i8 weights.

### Common CLI flags

| Flag | Meaning | Default |
|---|---|---|
| `--arch` | `NNUE_halfkp_256x2_32_32`<br>`NNUE_halfkp_384x2_8_96`<br>`NNUE_halfkp_512x2_8_64`<br>`NNUE_halfkp_768x2_16_64`<br>`NNUE_halfkp_1024x2_8_32`<br>`NNUE_halfkp_1024x2_8_64` | (required; target `NNUE_HALFKP` is inferred) |
| `--teacher` | Teacher file (`.hcpe` / `.hcpe3` / `.pack` / `.psv` / `.bin`), a directory of such files, or comma-separated combination | (required) |
| `--output` | Checkpoint parent directory | `checkpoints/<target>-<arch>` (e.g. `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32`) |
| `--max-epochs` | Maximum number of epochs. If omitted: no fixed epoch cap | omitted |
| `--superbatches` | Cap superbatches per epoch | unlimited |
| `--batch-size` | Positions per gradient step. If omitted, BulletOu uses the tatara-aligned default | 65536 |
| `--positions-per-superbatch` | Target positions per superbatch. Effective value is rounded down to a multiple of `batch-size` | 100000000 |
| `--save-rate` | Save every N superbatches; epoch end is also saved by default | 20 |
| `--save-epoch-end` / `--no-save-epoch-end` | Keep or disable the implicit epoch-end save | on |
| `--lr` / `--lr-schedule` / `--lr-min` | LR schedule (`step` = StepLR, `geometric` = geometric, `cos` = cosine; see [Advanced: Adjust training settings](../advanced/tuning.md)) | 0.000875 / `step` / 0.00001 |
| `--lambda` | Blend weight between teacher eval and WDL (= Win/Draw/Loss game-result label). Matches YaneuraOu's `lambda` convention: `λ × teacher_eval + (1−λ) × game_result`. `λ=1.0` is pure eval, `λ=0.0` is pure WDL | 1.0 |
| `--loss-pow-exp` | Exponent `p` in `|prediction - target|^p` | 2.0 |
| `--wrm-nnue2score` | WRM loss coefficient that maps `network_output` to score scale | 600 |
| `--loss-sigmoid-mse` / `--scale` | Use plain sigmoid loss instead of WRM | off / 600 |
| `--fv-scale` | `FV_SCALE` assumed for quantized `nn.bin` checks/export | 40 |

For the loss details, see [Advanced: Loss scale and `FV_SCALE`](../advanced/scale-and-fv-scale.md). Activation is fixed to ClippedReLU (matching the original 2018 architecture).
