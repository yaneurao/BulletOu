# 2. NNUE Tutorial — Train a Shogi NNUE

<a href="../../ja/tutorial/2-nnue-tutorial.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: end-to-end, train a shogi NNUE that a YaneuraOu-compatible engine can load.

This page assumes you have already completed [1. Quick Start](1-quickstart.md) — your toolchain works, and a smoke-test training succeeded.

## 2.1 What we will train

`bulletou --eval-type NNUE_HALFKP` trains a classical NNUE with HalfKP input features and a 4-layer SCReLU network:

```
shogi position
       │
       ▼ HalfKP sparse input (125,388 dims × 2 perspectives)
       │
       ▼ L0 affine + SCReLU       ← shared weights across own / opponent perspectives
       │
       ▼ accumulator (256 dims × 2 perspectives = 512 dims concatenated)
       │
       ▼ L1 affine (512 → 32) + SCReLU
       ▼ L2 affine (32 → 32) + SCReLU
       ▼ Out affine (32 → 1)
       │
       ▼ eval (centipawn-ish scalar)
```

The architecture is selected with `--arch` (only `256x2-32-32` for now — `x2` denotes dual-perspective, `256` is the accumulator size, `32-32` are the L2/L3 sizes). Equivalent to the small Stockfish-style NNUE.

This is not state of the art (which uses Layer Stack + threat features + a much larger FT), but it is enough to feel how training behaves and to plug into an engine for a sanity-check game.

## 2.2 Get training data

You need a `.pack`, `.hcpe`, `.hcpe3`, or `.psv` file.

- **Generate your own** — `.pack` is produced by the `gensfen` script in [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection); `.hcpe` / `.hcpe3` come from dlshogi-style generators. For this tutorial, 10–100 million positions is enough.
- **Use a shared dataset** — the shogi community shares files in all formats.

For this walkthrough we'll put teacher files under a `teachers/` directory next to the working directory:

```
teachers/
    teacher.pack
```

(`.hcpe` / `.hcpe3` / `.psv` work the same way. Format is inferred from the extension. You may also point `--teacher` at a directory, in which case all files of the same extension inside are concatenated.)

### Trying with a small subset first

Before running on a huge dataset, you can try a smaller subset by generating a smaller file from `gensfen`, or by limiting `--batches-per-superbatch` so each superbatch consumes less data (see §2.4).

## 2.3 Run NNUE training

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --output checkpoints/my-halfkp
```

(Use `--features device-rocm` instead of `cuda` for AMD GPUs.)

`--teacher` accepts:
- a single file (e.g. `teachers/teacher.pack`),
- a directory (above; all same-extension files inside are concatenated),
- or a comma-separated combination of the two.

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader reaches EOF). To run multiple passes, pass `--max-epochs N` — the LR scheduler restarts at the beginning of each epoch.

While it runs, you should see:

```
=== bulletou: running NNUE_HALFKP (256x2-32-32 SCReLU, dual-perspective) ===
Training Preamble
Net Name               : shogi_nnue_halfkp
Batch Size             : 16384
Batches / Superbatch   : 6104
Positions / Superbatch : 100007936
...
superbatch 1   pos = ... pos/s = ...   loss = ...
superbatch 2   ...
```

`pos/s` (positions per second) is the rough training-speed indicator. On a single RTX 4090 expect tens of millions of pos/s; on slower GPUs proportionally less.

## 2.4 Training schedule

The `superbatch` in the log is **the unit at which checkpoints and learning rate are updated**, about 100M positions by default.

Main flags:

| Flag | Meaning | Default |
|---|---|---|
| `--batch-size` | Positions per gradient step | 16384 |
| `--batches-per-superbatch` | Mini-batches per superbatch | `ceil(100M / batch-size)` (≒ 1 superbatch ≒ 100M positions) |
| `--superbatches` | Cap superbatches per epoch | unlimited (= run until EOF) |
| `--max-epochs` | Number of full passes through the teacher | 1 |
| `--save-rate` | Save a checkpoint every N superbatches | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR (multiply by `lr-gamma` every `lr-step` superbatches) | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL (blend ratio between eval score and game result) linearly interpolated | 0.0 / 1.0 |

Example (100M positions × 40 superbatches = 4 billion positions total):

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --output checkpoints/my-halfkp \
    --superbatches 40
```

If your teacher file is smaller than one superbatch (< 100M positions), lower `--batches-per-superbatch` (e.g. `1024` ⇒ 1 superbatch ≒ 16.78M positions) so multiple saves fire.

## 2.5 Inspect the output

After training finishes, `checkpoints/my-halfkp/` has the following layout:

```
checkpoints/my-halfkp/
├── learn.log                          ← top-level cumulative log across runs/resumes
├── 0001/
│   ├── nn.bin                         ← YaneuraOu / Stockfish (nnue-pytorch) compatible NNUE binary
│   ├── state.bin                      ← resume data (weights + Adam moments)
│   └── learn.log                      ← snapshot of the training log at this save point
├── 0002/
├── ...
└── 000N/                              ← the most recent save
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

The file you'll hand to the engine is `000N/nn.bin`.

## 2.6 Resume

Re-running the same command with the same `--output` automatically resumes from the latest `000N/state.bin` (new saves continue numbering from `000(N+1)/`). To start fresh, point `--output` at a different directory or delete the existing one.

## 2.7 Wire into an engine

Place the trained `000N/nn.bin` where the YaneuraOu engine expects its eval file (typically `eval/nn.bin`; the exact location is controlled by the `EvalDir` setting or similar — consult the engine's documentation), then launch the engine and confirm it loads with a quick `bench` or test game.

`state.bin` / `learn.log` are ignored by the engine but are worth keeping around for re-training and inspecting the loss curve.

## 2.8 Training other targets

- **KPPT** (the three-file `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` legacy eval): `--eval-type KPPT`, or `--eval-type KPP_KKPT` for the factorised variant. See [KPPT / KPP_KKPT Training](../shogi/kppt.md).
- Other NNUE variants (HalfKA / KP / SFNN+ls9 ...) will be added to `--eval-type` over time.

## 2.9 Where to go next

- [Reference: NNUE HalfKP Training](../shogi/halfkp.md) — `nn.bin` binary layout, quantisation, resume details
- [Reference: NNUE Basics](../1-basics.md) — the math behind perspective NNUE
- [Reference: Saved Networks](../4-saved-networks.md) — checkpoint layout, quantisation, transformation chains
- [Reference: KPPT / KPP_KKPT Training](../shogi/kppt.md) — training the legacy YaneuraOu evals
