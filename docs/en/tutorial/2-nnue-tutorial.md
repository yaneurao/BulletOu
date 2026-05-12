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

You need either:

- **`.pack`** — the **per-game variable-length** format produced by YaneuraOu's
  `gensfen` command. One file record = one game (start_flag + optional
  hcp/ply + (move16, eval) × moveNum + terminator). `ShogiPackLoader` expands
  each game into per-ply records on the fly.
- **`.hcpe`** — the dlshogi-style **38-byte fixed-length** record format
  (HCP + eval + bestMove16 + gameResult).
- **`.hcpe3`** — the dlshogi-style **per-game variable-length** format
  (game header + moveNum × MoveInfo + per-ply MoveVisits).

> ⚠️ `.pack` is **not** "a file of PackedSfenValue records". A `PackedSfenValue`
> is the 40-byte fixed-length internal unit that the trainer consumes once a
> loader has decoded the file. The two are different things — see the
> [Overview](0-overview.md#where-the-data-comes-from) for the distinction.

All three are supported; the choice depends on which generator you used (or
which shared dataset you have).

Options for obtaining data:

- **Generate your own** with YaneuraOu's `gensfen` (`.pack`) or dlshogi's data generator (`.hcpe` / `.hcpe3`). See each project's documentation; a few hundred million positions is the typical scale, but for this tutorial even 10–100 million is enough.
- **Use a shared dataset**. The shogi community shares `.pack`, `.hcpe`, and `.hcpe3` files on various sites. Make sure the source is trustworthy.

For this walkthrough we'll assume:

```
/data/shogi/raw.pack    # or
/data/shogi/raw.hcpe    # or
/data/shogi/raw.hcpe3
```

(Path can be anywhere; substitute your own.)

### Tiny test data first

If your full dataset is huge (tens of GB), it is convenient to first run with a small subset to make sure the command line works.

- For **`.hcpe`** (fixed 38-byte records) you can just take the head of the
  file:
  ```bash
  head -c $((38 * 10000000)) /data/shogi/raw.hcpe > /tmp/small.hcpe
  ```
  i.e. the first 10 million records.

- For **`.pack` / `.hcpe3`** (variable-length per game), you cannot byte-slice
  the file safely without breaking a game record boundary. Either generate a
  smaller `.pack` from `gensfen` directly, or use `--batches-per-superbatch`
  to cap how much data the trainer actually consumes per superbatch (covered
  in §2.3).

## 2.3 Run NNUE training

BulletOu provides two minimal examples depending on what your training data format is:

- **`shogi_simple`** — reads `.bin` (a flat dump of `PackedSfenValue` records,
  produced by bullet-utils' format converters) or `.pack` (per-game variable-
  length, from YaneuraOu's `gensfen`).
- **`shogi_simple_hcpe`** — reads `.hcpe` (dlshogi-style 38-byte fixed-length).

Pick whichever matches your data; otherwise the network shape and training loop are equivalent.

### Option A: `.pack` data (yaneurao gensfen)

```bash
cargo run --release --features cuda --example shogi_simple -- \
  --data /tmp/small.pack \
  --output checkpoints/my-first-shogi-net
```

(Use `--features rocm` instead of `cuda` for AMD GPUs.)

### Option B: `.hcpe` data (dlshogi-style)

```bash
cargo run --release --features cuda --example shogi_simple_hcpe -- \
  --data /data/shogi/train.hcpe \
  --output checkpoints/my-first-shogi-net-hcpe
```

`shogi_simple_hcpe` decodes each HCPE record (Apery-style HCP + eval + bestMove16 + gameResult) into the same internal PackedSfenValue used by `shogi_simple`, then feeds it into the same `ShogiHalfKA_hm` feature transformer and SCReLU + dual-perspective + 1-output network. There is no `--data-format` switch — the example is hcpe-only by design.

Caveats specific to HCPE:

- HCPE has no `game_ply`, so the Layer Stack `ply9` bucket cannot be used (this minimal example uses no bucketing).
- HCPE has no policy teacher (MoveVisits); value-only training. HCPE3 with policy is planned separately.

For real training you would replace `--data` with your full dataset and remove the `small.pack` / `small.hcpe` step.

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
