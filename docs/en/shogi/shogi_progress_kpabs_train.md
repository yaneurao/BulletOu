# shogi_progress_kpabs_train

<a href="../../ja/shogi/shogi_progress_kpabs_train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Tools for training `progress.bin` (the KP-Absolute progress model) used for LayerStack bucket selection.
The maintained helper is the CPU version. The old `bullet-gpu` CUDA helper was removed when BulletOu moved to the `cuda-cpp` training backend.

The generated `progress.bin` is referenced in LayerStack training as
`--bucket-mode progress8kpabs --progress-coeff progress.bin`.

---

## Model

Linear logistic regression over KP-Absolute features (king position × piece placement).

```
z      = Σ weights[kp_abs_index]
p      = sigmoid(z)
bucket = clamp(floor(p * 8), 0, 7)   // 8 values: 0..7
```

- Number of weights: `81 × 1548 = 125,388` (king position × `Eval::BonaPiece::fe_end`)

### Output file format

Size: **1,003,104 bytes** (= `8 × 81 × 1548`).
Elements are `f64` little-endian, array layout `weights[sq][bona_piece]`.

For the model design, BonaPiece numbering, and file format origins
(in [`yaneurao/YaneuraOu`](https://github.com/yaneurao/YaneuraOu)'s `old_engines/eval/progress/`
and [`nodchip/nnue-pytorch`](https://github.com/nodchip/nnue-pytorch)'s `tanuki_progress.cpp`),
see [`kp-absolute-progress.md`](kp-absolute-progress.md).

---

## Teacher-label modes

### Exact mode (`--game-relative`, recommended)

```
y = game_ply / (total_ply - 1)
```

- Normalised by the actual total move count of each game
- Game boundaries are detected when `game_ply` drops to or below the previous record's
- **Game-ordered data (pre-shuffle) is required**
- Bucket distribution tends to be uniform

### Approximate mode (default for CPU version)

```
y = clamp((game_ply - 1) / (ply_max - 1), 0, 1)
```

- Normalised by a fixed `ply_max` (CLI `--ply-max`, default 256)
- Usable on shuffled data
- Depending on the choice of `ply_max`, the bucket distribution may skew toward the first or second half
  (if `ply_max` is much larger than the actual game lengths, the late-game buckets become starved)

---

## Implementation variants

| Implementation | Binary name | Teacher mode | Training granularity | Backend |
|---|---|---|---|---|
| CPU | `shogi_progress_kpabs_train` | both exact and approximate | position-level minibatch (approximate) / 1 game = 1 step (exact) | single-thread CPU |

The removed CUDA helper solved the same convex optimisation problem, but it belonged to the retired `bullet-gpu` runtime and is no longer built.

Build:

```bash
cargo build --release --example shogi_progress_kpabs_train
```

---

## Data flow

```
self-play generation → raw.psv (game-ordered, pre-shuffle)
                          ├→ progress.bin training ← uses raw.psv here
                          └→ rescore → shuffle → train.psv → NNUE training
```

- Progress training does **not** use scores (eval values). Only the position (KP features) is used.
- Use data **before qsearch leaf replacement** (leaf replacement for the score teacher can change piece placements).

### Data feeding and file splitting

`--data` accepts either CSV or a directory containing `.bin` / `.pack`. When a directory is given, only `*.bin` / `*.pack` directly under it are processed.

After enumerating files, `pack_group_key()` partitions them by filename prefix
(`hao_depth_9_shuffled_*`, `shuffled_*`, otherwise grouped per file stem),
and `interleave_pack_groups()` performs **round-robin reordering** that pulls one file at a time from each group.

| Variant / mode | Data traversal | val/train split |
|---|---|---|
| CPU / approximate | `RoundRobinPackStream`: round-robin 1 record at a time across files | val_positions (the first N positions) → remaining `max_positions` is train (same stream) |
| CPU / exact | `MultiFileGameIterator`: sequential traversal in interleaved order, yielding per-game | **First 5%** (`packs.len() / 20`) is val, rest is train |
### Pitfall in automatic val splitting

Because the file split is based on the deterministic order produced by `interleave_pack_groups`, certain **game subsets** may be over-represented in val depending on the dataset composition.

For example, if "normal self-play files" and "specialised files (entering-king, etc.)" are concatenated and have different group keys, one of these subsets may end up clustered at the head or tail, producing a divergence between `val_loss` and the train distribution.

Workarounds:

- Unify the filename convention so `pack_group_key` returns the same group for all files
- Or set `--val-games` explicitly, and if needed, pre-shuffle the training dataset (i.e., produce mixed-source binary files in advance)

---

## Parameters

### CPU version (`shogi_progress_kpabs_train`)

| Parameter | Default | Description |
|---|---|---|
| `--data` | required | Comma-separated files or directories |
| `--output` | required | Path of the output `progress.bin` |
| `--game-relative` | false | Exact mode. Requires game-ordered data |
| `--max-positions` | 50,000,000 | Training samples per epoch (approximate mode) |
| `--val-positions` | 2,000,000 | Validation samples (approximate mode, taken from the head of the stream) |
| `--batch-size` | 4,096 | Minibatch size (approximate mode) |
| `--lr` | 0.0002 | Adam learning rate |
| `--epochs` | 1 | Number of training passes |
| `--ply-max` | 256 | Normalisation cap for approximate mode (ignored with `--game-relative`) |
| `--log-interval` | 100 | Log interval per batch (approximate mode) |
| `--max-games` | 0 (unlimited) | Number of training games per epoch (exact mode) |
| `--val-games` | 0 (auto) | Max games scanned per val pass (exact mode) |
| `--log-interval-games` | 1,000 | Log interval per game (exact mode) |
| `--save-each-epoch` | false | Also save `<output_stem>.eN.<ext>` after each epoch |

## Command examples

### CPU / exact mode

```bash
cargo run --release --example shogi_progress_kpabs_train -- \
  --data raw.psv \
  --output progress.bin \
  --game-relative \
  --max-games 0 \
  --val-games 0 \
  --epochs 1 \
  --lr 0.001 \
  --save-each-epoch
```

`--max-games 0` makes one full pass over the data. `--val-games 0` means "scan the entire val file group".

### CPU / approximate mode

```bash
cargo run --release --example shogi_progress_kpabs_train -- \
  --data train_shuffled.bin \
  --output progress.bin \
  --max-positions 50000000 \
  --val-positions 2000000 \
  --batch-size 4096 \
  --lr 0.0002 \
  --epochs 1 \
  --ply-max 256
```

Use this when shuffled data is already available. Adjust `--ply-max` to match actual game lengths in the data.

## Usage in NNUE training

```bash
cargo run --release --example shogi_layerstack -- \
  --data train.psv \
  --bucket-mode progress8kpabs \
  --progress-coeff progress.bin \
  ...
```

> The `progress.bin` used at training time **must match** the one used at inference time.
> Using a different `progress.bin` shifts bucket assignments and breaks consistency with the trained NN weights.

---

## Related documents

- [`kp-absolute-progress.md`](kp-absolute-progress.md) — KP-Absolute progress model (mathematics, BulletOu/bullet-shogi wiring, file format origins)
