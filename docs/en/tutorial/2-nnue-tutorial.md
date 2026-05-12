# 2. NNUE Tutorial — Train a Shogi NNUE

<a href="../../ja/tutorial/2-nnue-tutorial.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: end-to-end, train a shogi NNUE that a YaneuraOu-compatible engine can load.

This page assumes you have already completed [1. Quick Start](1-quickstart.md) — your toolchain works, and a smoke-test training succeeded.

## 2.1 What we will train

A small NNUE with the following structure (the default in `shogi_simple.rs`):

```
shogi position
       │
       ▼ ShogiHalfKA_hm (73,305-dim sparse feature)
       │
       ▼ Feature Transformer (FT, hidden size = 1024 or 1536, perspective-doubled)
       │
       ▼ SCReLU activation
       │
       ▼ Linear → scalar score
```

This is the "smallest NNUE that is actually useful for shogi" point on the design space. It is far from state of the art (which uses Layer Stack + threat features + a much larger FT), but it is plenty good enough to feel how training behaves.

If you want to skip ahead to a stronger configuration, the example `shogi_layerstack.rs` is the production-quality variant with Layer Stack, bucket selection, optional Threat / HandThreat features, and WDL scheduling.

## 2.2 Get training data

You need a `.pack`, `.hcpe`, or `.hcpe3` file.

- **Generate your own** — `.pack` is produced by the `gensfen` script in [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection); `.hcpe` / `.hcpe3` come from dlshogi-style generators. For this tutorial, 10–100 million positions is enough.
- **Use a shared dataset** — the shogi community shares files in all three formats.

For this walkthrough we'll assume:

```
/data/shogi/raw.pack
```

(`.hcpe` / `.hcpe3` work the same way. Substitute your own path.)

### Trying with a small subset first

Before running on a huge dataset, you can try a smaller subset by generating a smaller `.pack` from `gensfen`, or by limiting `--batches-per-superbatch` so each superbatch consumes less data (see §2.4).

## 2.3 Run NNUE training

Pick the example matching your data format:

- **`shogi_simple`** — reads `.pack`
- **`shogi_simple_hcpe`** — reads `.hcpe`

### `.pack`

```bash
cargo run --release --features device-cuda --example shogi_simple -- \
    --data /data/shogi/raw.pack \
    --output checkpoints/my-first-shogi-net \
    --superbatches 40
```

(Use `--features device-rocm` instead of `cuda` for AMD GPUs.)

### `.hcpe`

```bash
cargo run --release --features device-cuda --example shogi_simple_hcpe -- \
    --teacher /data/shogi/raw.hcpe \
    --output checkpoints/my-first-shogi-net \
    --superbatches 40
```

HCPE-specific caveats:

- HCPE has no `game_ply`, so the Layer Stack `ply9` bucket cannot be used (this minimal example uses no bucketing).
- HCPE has no policy teacher; value-only training (use HCPE3 if you need a policy teacher).

While it runs, you should see:

```
superbatch 1 / 40   pos = ... pos/s = ...   loss = ...
superbatch 2 / 40   ...
```

`pos/s` (positions per second) is the rough training-speed indicator. On a single RTX 4090 the smoke-test configuration runs in the tens-of-millions of pos/s range; on slower GPUs proportionally less.

## 2.4 Training schedule

The `superbatch 1 / 40` in the log is **the unit at which checkpoints and learning rate are updated**, about 100M positions by default. Total training length is set by `--superbatches`.

Main flags:

| Flag | Meaning | Default |
|---|---|---|
| `--batch-size` | Positions per gradient step | 16384 |
| `--batches-per-superbatch` | Mini-batches per superbatch | `ceil(100M / batch-size)` (≒ 1 superbatch ≒ 100M positions) |
| `--superbatches` | Total superbatches | 10 |
| `--save-rate` | Save a checkpoint every N superbatches | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR (multiply by `lr-gamma` every `lr-step` superbatches) | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL (blend ratio between eval score and game result) interpolated linearly across `--superbatches` | 0.0 / 1.0 |

Example invocation:

```bash
--batch-size 16384 --batches-per-superbatch 6104 --superbatches 40
# = 1 superbatch ≒ 100M positions, total 4 billion positions
```

Scheduler details (Cosine / Linear / Warmup, etc.) live in the [reference](../).

## 2.5 Inspect the output

At every checkpoint (and at the end of training), BulletOu writes **`nn.bin`** under `checkpoints/my-first-shogi-net/` — this is the NNUE evaluation parameter file that a YaneuraOu engine loads at play time.

## 2.6 Try it in an engine

Place the trained `nn.bin` where the YaneuraOu engine expects its eval file (typically `eval/nn.bin`; the exact location is controlled by the `EvalDir` setting or similar — consult the engine's documentation), then launch the engine and confirm it loads with a quick `bench` or test game.

## 2.7 Stepping up to the production setup

When you are comfortable with `shogi_simple`, move to `shogi_layerstack` for stronger results:

```bash
cargo run --release --features device-cuda --example shogi_layerstack -- \
  --data /data/shogi/train.pack \
  --output checkpoints/my-layerstack-net \
  --feature ShogiHalfKaHmThreat \
  --bucket-mode progress8kpabs \
  --progress-coeff progress.bin \
  --start-wdl 0.0 --end-wdl 1.0
```

The pieces (Threat features, `progress.bin`, WDL scheduling) are explained in the [reference docs](../). Use the `shogi_simple` smoke test as your "is everything plumbed correctly?" check, then iterate with `shogi_layerstack`.

## 2.8 Where to go next

- [Reference: NNUE Basics](../1-basics.md) — the math behind perspective NNUE
- [Reference: Saved Networks](../4-saved-networks.md) — checkpoint layout, quantisation, transformation chains
- [Reference: KP-Absolute Progress](../shogi/kp-absolute-progress.md) — what `--bucket-mode progress8kpabs` actually does
- [Reference: shogi_progress_kpabs_train](../shogi/shogi_progress_kpabs_train.md) — how to train your own `progress.bin`
- [KPPT / KPP_KKPT Training](../shogi/kppt.md) — training the legacy YaneuraOu evals (reference)
