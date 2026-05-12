# NNUE HalfKP Training

<a href="../../ja/shogi/halfkp.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

[Back to Reference index](../README.md)

`--eval-type NNUE_HALFKP` trains the original YaneuraOu HalfKP NNUE (the `halfkp_256x2-32-32` architecture introduced by Nasu-san in YaneuraOu PR #75, 2018) — dual-perspective HalfKP feature transformer + 4 ClippedReLU layers. The "SqrClippedReLU" (`SCReLU`) activation was added later, in PR #311 (2026) for SFNNwoPSQT-1536, and is *not* used here.

## Architecture

Selected via `--arch` (currently only one preset, `256x2-32-32`):

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

- BulletOu built (`cargo build --release --features device-cuda --example bulletou`)
- Training data (`.hcpe` / `.hcpe3` / `.pack` / `.psv`)

### Command

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type NNUE_HALFKP \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-halfkp
```

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader reaches EOF). To run multiple passes, pass `--max-epochs N` — the LR scheduler restarts at the beginning of each epoch.

### Save layout

```
checkpoints/my-halfkp/
├── learn.log                          ← top-level cumulative log across runs/resumes
├── 0001/
│   ├── nn.bin                         ← YaneuraOu / Stockfish (nnue-pytorch) compatible NNUE binary
│   ├── state.bin                      ← resume data (weights + Adam moments)
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

### Resume

If `--output` already contains numbered dirs with `state.bin`, `bulletou` automatically resumes from the latest one. New saves continue the numbering. Just re-running the same command picks up where it left off.

### Common CLI flags

| Flag | Meaning | Default |
|---|---|---|
| `--eval-type` | `NNUE_HALFKP` | (required) |
| `--arch` | `256x2-32-32` (only preset for now) | `256x2-32-32` |
| `--teacher` | Teacher file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), a directory of such files, or comma-separated combination | (required) |
| `--output` | Checkpoint parent directory | `checkpoints/shogi_nnue_halfkp` |
| `--max-epochs` | Number of full passes through the teacher | 1 |
| `--superbatches` | Cap superbatches per epoch | unlimited |
| `--batches-per-superbatch` | Mini-batches per superbatch | ≈ 100M positions |
| `--save-rate` | Save every N superbatches | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | LR schedule | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | Linear WDL schedule | 0.0 / 1.0 |

Loss function is fixed to `sigmoid(eval).squared_error(target)`. Activation is fixed to ClippedReLU (matching the original 2018 architecture). These can be added as flags later if needed.
