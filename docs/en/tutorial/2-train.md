# 2. Run a training — make an evaluation function from real data

<a href="../../ja/tutorial/2-train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: produce a YaneuraOu-loadable evaluation function from real training data.

This page assumes you have already completed [1. Quick Start](1-quickstart.md) — your toolchain works, and a smoke-test training succeeded.

We use **NNUE HalfKP as the running example** in this tutorial, but the same command shape applies to the other targets (NNUE K-P / NNUE HalfKPE9 / KPPT / KPP_KKPT) by switching `--eval-type`.

## 2.1 Choosing what to train

`bulletou --eval-type <X>` selects which evaluation function to train. The currently public choices:

| `--eval-type` | What it trains | Output (per save) | `--arch` used? |
|---|---|---|---|
| **`NNUE_HALFKP`** ★ start here | Classic HalfKP NNUE — YaneuraOu's longest-standing evaluation function family. See [NNUE HalfKP Training](../shogi/halfkp.md). | `nn.bin` | yes |
| `NNUE_KP` | Same network as HalfKP, but the input keeps K and P as independent features. See [NNUE K-P Training](../shogi/kp.md). | `nn.bin` | yes |
| `NNUE_HALFKPE9` | HalfKP augmented with per-square attacker-count info (own/opp 0/1/2, 9 combos). See [NNUE HalfKPE9 Training](../shogi/halfkpe9.md). | `nn.bin` | yes |
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

Before running on a huge dataset, you can try a smaller subset by generating a smaller file from `gensfen`, or by limiting `--batches-per-superbatch` so each superbatch consumes less data (see [§3.1](3-tune.md#31-training-schedule)).

## 2.3 Run the training

### Build (one-off)

Build `bulletou` first. You only need this once, until the source changes:

```bash
cargo build --release --features device-cuda --example bulletou
```

(Use `--features device-rocm` instead of `cuda` for AMD GPUs. On Windows the binary is at `.\target\release\examples\bulletou.exe`; the run commands below use Unix-style paths — translate as needed.)

### Minimal command (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

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
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --arch 1024x2-8-64 \
    --teacher teachers/
```

Omitting `--arch` falls back to `256x2-32-32`. `NNUE_KP` / `NNUE_HALFKPE9` accept the same preset list.

(`halfkpvm` — a different *input feature set* — and `SFNNwoPSQT1536` are tracked as future `--eval-type` values, not reachable through `--arch` alone.)

### Training a KPPT eval

For KPPT-family eval types the architecture is fixed and `--arch` is ignored:

```bash
./target/release/examples/bulletou \
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

## 2.4 Stopping and resuming

Stop mid-training (Ctrl+C, machine reboot, whatever) and **re-run the exact same command with the same `--output` — `bulletou` automatically resumes from the latest `000N/state.bin`**.

```
checkpoints/.../
├── 0001/             ← from the previous run
├── 0002/
├── 0003/             ← latest save when training was interrupted
├── 0004/             ← the resumed run writes from here
└── 0005/
```

How it works:
- On startup, `bulletou` looks under `--output` for numbered dirs containing `state.bin`.
- The highest-numbered `state.bin` is loaded, restoring weights and Adam moments.
- New saves continue numbering from one past the existing maximum (`0004/` here).
- The cumulative `learn.log` keeps appending CSV rows for the resumed run. The LR scheduler restarts each run so the superbatch counter resets to 1, but the `positions` column continues from the previous run's max (read off the existing `learn.log` at startup).

This behaviour is identical for every eval-type (KPPT / KPP_KKPT / NNUE_HALFKP / NNUE_KP / NNUE_HALFKPE9 — all share the same mechanism). To start fresh, point `--output` at a different directory or delete the existing one.

---

Next:
- [3. Tune the training](3-tune.md) — adjust `--lambda`, `--lr`, `--superbatches`, etc. (optional)
- If you already have a trained model, jump to [4. Inspect and use the result](4-result.md)
