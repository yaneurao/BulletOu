# 2. Running a training — make an evaluation function from real data

<a href="../../ja/tutorial/2-training.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: produce a YaneuraOu-loadable evaluation function from real training data and verify it in an engine.

This page assumes you have already completed [1. Quick Start](1-quickstart.md) — your toolchain works, and a smoke-test training succeeded.

We use **NNUE HalfKP as the running example** in this tutorial, but the same command shape applies to the other targets (NNUE K-P / KPPT / KPP_KKPT) by switching `--eval-type`.

## 2.1 Choosing what to train

`bulletou --eval-type <X>` selects which evaluation function to train. The currently public choices:

| `--eval-type` | What it trains | Output (per save) | `--arch` used? |
|---|---|---|---|
| **`NNUE_HALFKP`** ★ start here | Classic HalfKP NNUE — YaneuraOu's longest-standing evaluation function family. See [NNUE HalfKP Training](../shogi/halfkp.md). | `nn.bin` | yes |
| `NNUE_KP` | Same network as HalfKP, but the input keeps K and P as independent features. See [NNUE K-P Training](../shogi/kp.md). | `nn.bin` | yes |
| `KPPT` | Legacy three-file evaluation (elmo(WCSC27)-compatible). See [KPPT / KPP_KKPT Training](../shogi/kppt.md). | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` | no |
| `KPP_KKPT` | KPPT's factorised variant — only KPP changes (no turn channel, ~half size) | Same three files, only KPP layout differs | no |

Coming later: HalfKA, SFNN + ls9 (NNUEwoSQPT1536), and other variants.

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

## 2.3 Run the training

### Minimal command (NNUE HalfKP)

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

(Use `--features device-rocm` instead of `cuda` for AMD GPUs.)

That's it — no further flags needed. With `--output` omitted, checkpoints land under `checkpoints/NNUE_HALFKP-256x2-32-32/` (auto-derived from `--eval-type` and `--arch`). Pass `--output checkpoints/my-halfkp` (or any other path) to override.

### Specifying `--arch`

For NNUE eval types, the layer sizes are selected with `--arch <L1>x2-<L2>-<L3>`. The set mirrors the per-arch directories under YaneuraOu's NNUE engine binary distribution (`NNUE_halfkp_*`):

| `--arch` | L1 (accumulator) | L2 | L3 | Notes |
|---|---|---|---|---|
| `256x2-32-32` (default) | 256 | 32 | 32 | Classic small NNUE; fast to train, good for sanity checks |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | Medium |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | Larger (inference cost grows) |
| `1024x2-8-64` | 1024 | 8 | 64 | Larger |

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type NNUE_HALFKP \
    --arch 1024x2-8-64 \
    --teacher teachers/
```

Omitting `--arch` falls back to `256x2-32-32`. `NNUE_KP` accepts the same preset list (YaneuraOu ships only `NNUE_kp_256x2_32_32`, but the trainer doesn't restrict you).

(`halfkpe9` / `halfkpvm` — different *input feature sets*, not just different layer sizes — and `SFNNwoPSQT1536` are tracked as future `--eval-type` values, not reachable through `--arch` alone.)

### Training a KPPT eval

For KPPT-family eval types the architecture is fixed and `--arch` is ignored:

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type KPPT \
    --teacher teachers/
```

The default output dir is `checkpoints/KPPT/`. To get the factorised variant, swap to `--eval-type KPP_KKPT`.

### Passing teacher data

`--teacher` accepts:
- a single file (e.g. `teachers/teacher.pack`),
- a directory (above; all same-extension files inside are concatenated),
- or a comma-separated combination of the two.

### How long does training run

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader reaches EOF). To run multiple passes, pass `--max-epochs N` — the LR scheduler restarts at the beginning of each epoch.

### What you should see

While it runs:

```
=== bulletou: running NNUE_HALFKP (256x2-32-32 ClippedReLU, dual-perspective) ===
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

## 2.4 Training schedule (come back to this when you need to tune)

**All flags default to sensible values; you can ignore this section on the first run.** Return when you want to tune for teacher size or available compute.

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
    --superbatches 40
```

If your teacher file is smaller than one superbatch (< 100M positions), lower `--batches-per-superbatch` (e.g. `1024` ⇒ 1 superbatch ≒ 16.78M positions) so multiple saves fire.

## 2.5 Inspect the output

After training finishes the output directory (e.g. `checkpoints/NNUE_HALFKP-256x2-32-32/`) has the following layout:

```
checkpoints/NNUE_HALFKP-256x2-32-32/
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

`000N/` (the highest-numbered dir) holds the artefacts to hand to the engine.

For KPPT / KPP_KKPT, instead of `nn.bin` each numbered dir contains the three files `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` (all three are required together).

## 2.6 Resume

Stopping mid-training is fine: **re-run the exact same command with the same `--output` and `bulletou` automatically resumes from the latest `000N/state.bin`** (new saves continue from `000(N+1)/`). To start fresh, point `--output` at a different directory or delete the existing one.

## 2.7 Load into an engine

A minimum walkthrough for verifying the trained weights in a YaneuraOu engine.

### For NNUE evals (`nn.bin`)

Put the latest `000N/nn.bin` where the engine looks for its eval file. With YaneuraOu the path is set via the `EvalDir` USI option:

```
# After the engine starts, in the USI command line:
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/NNUE_HALFKP-256x2-32-32/0005
isready
bench
```

Alternatively, place `000N/nn.bin` as `eval/nn.bin` if your engine expects that relative path.

`isready` succeeding means the engine loaded the file. `bench` prints the hash of the loaded `nn.bin`, so a different number on each re-trained model confirms you're really using different weights.

### For KPPT-family evals (three-file set)

Point `EvalDir` at the latest `000N/` directory directly (it must contain all three files):

```
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/KPPT/0005
isready
bench
```

The engine refuses to load if any of the three files is missing.

### If the result is weak

The first training run uses a small teacher and few superbatches, so don't expect competitive strength. To get something usable in real play:
- Increase teacher size (100M → 1B+ positions)
- Run several epochs (e.g. `--max-epochs 3`)
- Increase `--save-rate` (e.g. 10) and only use the later saves

Per-eval-type hyperparameter advice lives in the reference docs ([halfkp.md](../shogi/halfkp.md) / [kp.md](../shogi/kp.md) / [kppt.md](../shogi/kppt.md)).

## 2.8 Where to go next

- [Reference: NNUE HalfKP Training](../shogi/halfkp.md) — `nn.bin` binary layout, quantisation, resume details
- [Reference: NNUE K-P Training](../shogi/kp.md) — input feature comparison vs HalfKP
- [Reference: KPPT / KPP_KKPT Training](../shogi/kppt.md) — legacy YaneuraOu evals
- [Specifications: spec/](../../../spec/) — eval-type matrix, binary layout, hash derivations
