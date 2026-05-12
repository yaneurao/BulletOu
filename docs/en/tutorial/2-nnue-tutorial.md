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

You need a `.pack` file (PackedSfenValue, YaneuraOu's `gensfen` format).

Options:

- **Generate your own** with YaneuraOu's `gensfen` command. See YaneuraOu's documentation; a few hundred million positions is the typical scale, but for this tutorial even 10–100 million is enough.
- **Use a shared dataset**. The shogi community shares `.pack` files on various sites. Make sure the source is trustworthy.

For this walkthrough we'll assume:

```
/data/shogi/raw.pack
```

(Path can be anywhere; substitute your own.)

### Tiny test data first

If your full dataset is huge (tens of GB), it is much easier to first run with a small subset to make sure the command line works. The `bullet-utils` tool can split a `.pack` to a reasonable size:

```bash
cargo run --release --package bullet-utils -- \
  shuffle --input /data/shogi/raw.pack --output /tmp/small.pack \
  --record-size 40 --seed 42
# (Then take the first ~10 million records of /tmp/small.pack for your first run.)
```

(40 byte per record is the PackedSfenValue layout.)

## 2.3 Run NNUE training (shogi_simple)

The simplest end-to-end shogi training is:

```bash
cargo run --release --features cuda --example shogi_simple -- \
  --data /tmp/small.pack \
  --output checkpoints/my-first-shogi-net
```

(Use `--features rocm` instead of `cuda` for AMD GPUs.)

For real training you would replace `--data` with your full dataset and remove the `small.pack` step.

While it runs, you should see:

```
loaded 73305 input features (ShogiHalfKA_hm)
superbatch 1 / 40   pos = ... pos/s = ...   loss = ...
superbatch 2 / 40   ...
```

`pos/s` (positions per second) is the rough training-speed indicator. On a single RTX 4090 the smoke-test configuration runs in the tens-of-millions of pos/s range; on slower GPUs proportionally less.

## 2.4 Inspect the output

When training finishes (or at every saved checkpoint), `checkpoints/my-first-shogi-net/` will contain:

```
my-first-shogi-net-final/
├── raw.bin                ← float weights (resume from here)
├── quantised.bin          ← integer weights (rshogi-compatible)
└── optimiser_state/
    ├── weights.bin
    ├── moment1.bin
    └── ...
```

- `quantised.bin` is what an engine will load at play time.
- `raw.bin` and `optimiser_state/` together let you resume training from this exact point.

## 2.5 Try it in an engine

The exact integration depends on the engine. For YaneuraOu-compatible NNUE consumption, the typical steps are:

1. Convert `quantised.bin` to the engine's expected NN file format if needed (BulletOu writes the rshogi-compatible layout; YaneuraOu may need a thin adapter).
2. Place the file where the engine looks for it.
3. Run a quick game or `bench` to confirm it loads without error.

> Engine integration is currently outside the scope of BulletOu itself — the trainer's job ends at writing `quantised.bin`. Plumbing it into a specific engine is a per-engine task.

## 2.6 Stepping up to the production setup

When you are comfortable with `shogi_simple`, move to `shogi_layerstack` for stronger results:

```bash
cargo run --release --features cuda --example shogi_layerstack -- \
  --data /data/shogi/train.pack \
  --output checkpoints/my-layerstack-net \
  --feature ShogiHalfKaHmThreat \
  --bucket-mode progress8kpabs \
  --progress-coeff progress.bin \
  --start-wdl 0.0 --end-wdl 1.0
```

The pieces (Threat features, `progress.bin`, WDL scheduling) are explained in the [reference docs](../0-contents.md). Use the `shogi_simple` smoke test as your "is everything plumbed correctly?" check, then iterate with `shogi_layerstack`.

## 2.7 Where to go next

- [Reference: NNUE Basics](../1-basics.md) — the math behind perspective NNUE
- [Reference: Saved Networks](../4-saved-networks.md) — checkpoint layout, quantisation, transformation chains
- [Reference: KP-Absolute Progress](../shogi/kp-absolute-progress.md) — what `--bucket-mode progress8kpabs` actually does
- [Reference: shogi_progress_kpabs_train](../shogi/shogi_progress_kpabs_train.md) — how to train your own `progress.bin`
- [3. KPPT / KPP_KPPT Roadmap](3-kppt-roadmap.md) — what's planned for legacy-eval support
