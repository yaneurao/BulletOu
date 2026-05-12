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

## 2.4 Training units — batch / superbatch / save / LR

The log line `superbatch 1 / 40` raises an obvious question: what *is* a superbatch, and what do `--batch-size`, `--batches-per-superbatch`, and `--superbatches` each mean? This section answers all of that at once. The concepts come from upstream jw1912/bullet and are inherited unchanged by bullet-shogi / BulletOu.

### 2.4.1 The three units

```
batch (= mini-batch, one gradient step)
  └─ 16384 positions: one forward + backward + optimizer step
        │
        │ × batches_per_superbatch
        ▼
superbatch
  └─ default ≈ 100M positions  (= 6104 batches × 16384 positions/batch)
        │
        │ × superbatches
        ▼
entire training (up to end_superbatch)
```

| CLI flag | Meaning | Default |
|---|---|---|
| `--batch-size` | Positions per gradient step (= mini-batch size). Sets GPU memory pressure and convergence behaviour. | `16384` |
| `--batches-per-superbatch` | Number of mini-batches that form one superbatch. **If unspecified, it is set to `ceil(100_000_000 / batch_size)`** automatically. | (auto) |
| `--superbatches` | Total superbatches to run (= `end_superbatch`). Sets the overall training length. | example-dependent; `10` in the KK/KKP examples |

The default formula for `batches_per_superbatch` is designed to **keep one superbatch at roughly 100M positions**. Changing `--batch-size` does not significantly change the positions-per-superbatch count. This is the implicit scale in upstream bullet's chess NNUE culture — `bullet/examples/progression/1_simple.rs` and friends hard-code `batches_per_superbatch: 6104`.

### 2.4.2 Is a superbatch the same as an epoch?

**Not exactly.** In standard ML terminology, an epoch is "one full pass through the dataset". A bullet superbatch, in contrast, is **a fixed ~100M-position chunk regardless of dataset size**.

- With 50M training positions, one superbatch sweeps through the data ~2× (loaders reshuffle and continue from the start when they hit EOF).
- With 1B training positions, one superbatch only touches ~10% of the data.

The accurate mental model: a superbatch is **the unit at which checkpoints / LR / WDL are updated**, not a pass through the data.

### 2.4.3 Checkpoint timing — `--save-rate`

```
--save-rate 1   →  save every superbatch
--save-rate 5   →  save every 5 superbatches
--save-rate 0   →  save only the final superbatch
```

Each save point creates a `checkpoints/<net-id>-<superbatch>/` directory containing `raw.bin` / `quantised.bin` / `optimiser_state/` (plus, in the KPPT examples, `KK_synthesized.bin` / `KKP_synthesized.bin`).

The final superbatch (`end_superbatch`) is always saved regardless of `--save-rate` (the `should_save` check is an OR).

### 2.4.4 LR scheduler time axis

Every bullet LR scheduler is a function `lr(batch, superbatch) -> f32`, and **virtually all of them key off `superbatch`** (only `Warmup` also uses the batch axis):

| Scheduler | Behaviour | CLI | Requires final superbatch? |
|---|---|---|---|
| `ConstantLR` | fixed value | (no direct flag) | no |
| `StepLR` | multiply by `gamma` every `step` superbatches | `--lr` / `--lr-gamma` / `--lr-step` (used by KK/KKP examples) | no |
| `DropLR` | multiply by `gamma` once at superbatch `drop` | — | no |
| `LinearDecayLR` | linear interpolate to `final_lr` by `final_superbatch` | — | **yes** |
| `CosineDecayLR` | same, cosine curve | — | **yes** |
| `ExponentialDecayLR` | same, exponential | — | **yes** |
| `Warmup<LR>` | linear warmup over N batches, then defer to inner | — | inner-dependent |

With the `shogi_kk_kkp_train` defaults `--lr 0.001 --lr-gamma 0.1 --lr-step 8`, the LR is `0.001` for superbatches 1–8, `0.0001` for 9–16, `0.00001` for 17–24, and so on — each `step` drops it by a factor of `gamma`. With `--superbatches 3` the drop never triggers and LR stays at `0.001` throughout.

### 2.4.5 WDL scheduler time axis

WDL (the blend ratio between eval score and game result label) is also indexed by superbatch. The default:

```
--start-wdl 0.0  --end-wdl 1.0
```

means "first superbatch trains on **eval only**, last superbatch trains on **game result only**, linear interpolation in between." The endpoint is `end_superbatch` (= `--superbatches`), so changing `--superbatches` automatically rescales the WDL ramp.

### 2.4.6 Worked example

```
--batch-size 16384
--batches-per-superbatch 100      ← much smaller than the default 6104
--superbatches 3
--save-rate 1
--lr 0.001 --lr-gamma 0.1 --lr-step 8
--start-wdl 0.0 --end-wdl 1.0
```

Concretely:

```
1 superbatch  = 100 batches × 16384 positions = 1,638,400 positions (≈ 1.6M)
total run     = 3 superbatches × 1.6M         = 4,915,200 positions (≈ 4.9M)
checkpoints   = saved at sb=1, sb=2, sb=3 (three saves)
LR            = 0.001 throughout (step=8 never triggers within 3 superbatches)
WDL           = 0.0 at sb=1, 0.5 at sb=2, 1.0 at sb=3
```

For real training at the "~100M positions per superbatch" scale, **omit `--batches-per-superbatch`** so it defaults to `6104` for the standard `--batch-size 16384`.

## 2.5 Inspect the output

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

## 2.6 Try it in an engine

The exact integration depends on the engine. For YaneuraOu-compatible NNUE consumption, the typical steps are:

1. Convert `quantised.bin` to the engine's expected NN file format if needed (BulletOu writes the rshogi-compatible layout; YaneuraOu may need a thin adapter).
2. Place the file where the engine looks for it.
3. Run a quick game or `bench` to confirm it loads without error.

> Engine integration is currently outside the scope of BulletOu itself — the trainer's job ends at writing `quantised.bin`. Plumbing it into a specific engine is a per-engine task.

## 2.7 Stepping up to the production setup

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

The pieces (Threat features, `progress.bin`, WDL scheduling) are explained in the [reference docs](../). Use the `shogi_simple` smoke test as your "is everything plumbed correctly?" check, then iterate with `shogi_layerstack`.

## 2.8 Where to go next

- [Reference: NNUE Basics](../1-basics.md) — the math behind perspective NNUE
- [Reference: Saved Networks](../4-saved-networks.md) — checkpoint layout, quantisation, transformation chains
- [Reference: KP-Absolute Progress](../shogi/kp-absolute-progress.md) — what `--bucket-mode progress8kpabs` actually does
- [Reference: shogi_progress_kpabs_train](../shogi/shogi_progress_kpabs_train.md) — how to train your own `progress.bin`
- [3. KPPT / KPP_KKPT Training](3-kppt-roadmap.md) — how to train legacy YaneuraOu evals (Phases 1–4 done)
