# 9. LayerStack — pick a different sub-network per position

<a href="../../ja/tutorial/9-layerstack.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

A standard NNUE uses one MLP for every position. SFNN LayerStack models keep multiple small MLP stacks and select one stack from the position.

The important rule is simple: BulletOu and YaneuraOu must compute exactly the same LayerStack index.

## 9.1 LayerStack axes

YaneuraOu now treats LayerStack selection as three independent axes:

| Axis | Accepted tokens | Bucket count |
|---|---|---:|
| hand | omitted, `hand64`, `hand256`, `hand1024` | 1 / 64 / 256 / 1024 |
| king | omitted, `k3k3`, `k9k9`, `k21k21`, `k29k29` | 1 / 9 / 81 / 441 / 841 |
| progress | omitted, `progress2`, `progress3`, `progress4`, `progress8`, `progress16`, `progress32` | 1 / 2 / 3 / 4 / 8 / 16 / 32 |

The final stack count is:

```text
LayerStacks = hand_bucket_count * king_bucket_count * progress_bucket_count
```

The runtime index is composed in the same order as YaneuraOu:

```text
idx = hand_bucket
idx = idx * king_bucket_count + king_bucket
idx = idx * progress_bucket_count + progress_bucket
```

The architecture parser accepts tokens in any order, but BulletOu canonicalizes names as `hand`, then `king`, then `progress`. For example:

```text
SFNN_halfka2_1024_7_64_k3k3_hand256_progress16
```

is accepted and displayed/saved as:

```text
SFNN_halfka2_1024_7_64_hand256_k3k3_progress16
```

## 9.2 Examples

| `--arch` | LayerStacks | Meaning |
|---|---:|---|
| `SFNN_halfka2_1024_7_64` | 1 | single stack |
| `SFNN_halfka2_1024_7_64_k3k3` | 9 | king 3x3 |
| `SFNN_halfka2_1024_7_64_k29k29` | 841 | king 29x29 |
| `SFNN_halfka2_1024_7_64_hand256` | 256 | hand256 only |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 2304 | hand256 x k3k3 |
| `SFNN_halfka2_1024_7_64_progress8` | 8 | progress8 only |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 72 | k3k3 x progress8 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 36864 | hand256 x k3k3 x progress16 |

Grouped SFNN notation can be combined with the same suffixes:

```text
SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k3k3_progress16
```

## 9.3 King buckets

Long aliases such as `king3_by_king3`, `king9_by_king9`, `king21_by_king21`, and `king29_by_king29` are also accepted.

### `k3k3`

After perspective normalization, both kings' ranks are grouped into three bands:

|  | enemy rank 1-3 | enemy rank 4-6 | enemy rank 7-9 |
|---|---:|---:|---:|
| friend rank 1-3 | 0 | 1 | 2 |
| friend rank 4-6 | 3 | 4 | 5 |
| friend rank 7-9 | 6 | 7 | 8 |

### `k9k9`

Uses exact friend/enemy king ranks:

```text
bucket = friend_rank * 9 + enemy_rank
```

### `k21k21`

Each king square is mapped to 21 buckets:

```text
if rank < 3: single = 0
else if rank < 6: single = 1
else if rank < 7: single = 2
else: single = 3 + (rank - 7) * 9 + file

bucket = friend_single * 21 + enemy_single
```

### `k29k29`

Each king square is mapped to 29 buckets:

```text
if rank < 3: single = 0
else if rank < 6: single = 1
else: single = 2 + (rank - 6) * 9 + file

bucket = friend_single * 29 + enemy_single
```

## 9.4 Hand buckets

### `hand64`

Each side's hand is converted to a score, then bucketed:

```text
bucket_one_side = min((score + 3) / 4, 7)
bucket = side_to_move_bucket * 8 + non_side_bucket
```

Scores:

- pawn: 1
- lance/knight: 2
- silver/gold: 3
- bishop/rook: 5

### `hand256`

Each side uses four presence bits:

- bit0: has pawn/lance/knight
- bit1: has silver/gold
- bit2: has bishop
- bit3: has rook

The final bucket is:

```text
bucket = side_to_move_bucket * 16 + non_side_bucket
```

### `hand1024`

Each side uses five presence bits:

- bit0: has pawn
- bit1: has lance/knight
- bit2: has silver/gold
- bit3: has bishop
- bit4: has rook

The final bucket is:

```text
bucket = side_to_move_bucket * 32 + non_side_bucket
```

## 9.5 Progress buckets

`progressN` uses YaneuraOu's SFNN progress parameters to compute a scalar progress value in `0..255`, then maps it to the requested bucket count:

```text
progress_bucket = min(progress_0_255 * progress_bucket_count / 256,
                      progress_bucket_count - 1)
```

When `progressN` is used, the exported `nn.bin` layout is:

```text
NNUE header
FeatureTransformer section
Progress section
LayerStack network section 0
LayerStack network section 1
...
```

The Progress section is YaneuraOu-compatible:

```text
section_hash = 0x6f50524f  # "oPRO"
int32 bias_q16
int32 weights_q16[81][1548]
```

Use `--sfnn-progress-params <file>` only when you already have this q16 Progress::Parameters payload. The file may be either:

- `bias_q16 + weights_q16[81][1548]`
- `0x6f50524f + bias_q16 + weights_q16[81][1548]`

This is not the older experimental f64 `progress8kpabs` / `progress.bin` format. If `--sfnn-progress-params` is omitted for a `progressN` architecture, BulletOu writes zero progress parameters, matching YaneuraOu's dummy zero behavior; all positions then map to the neutral progress bucket.

Currently, `--sfnn-factorizer shared` works with `progressN`. `king=axis` / `hand=axis` factorizer terms are rejected with `progressN` because the current CUDA factorizer layout only models the two-axis hand/king decomposition.

## 9.6 Usage

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k3k3_progress8 \
    --teacher teachers/
```

For a combined hand/king/progress experiment:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand256_k3k3_progress16 \
    --teacher teachers/
```

Omitting `--output` puts checkpoints under a directory derived from the canonical architecture name.

## 9.7 Notes

- LayerStack is only meaningful for SFNN architectures.
- More buckets reduce the amount of data seen by each stack, so large combinations need much more teacher data.
- Checkpoint size and VRAM usage grow roughly with the number of LayerStacks.
- Build YaneuraOu with the same architecture suffix when loading the exported `nn.bin`.

---

Previous: [8. Load into an engine](8-engine.md)
