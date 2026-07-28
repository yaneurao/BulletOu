# 9. LayerStack — pick a different small network per position

<a href="../../ja/tutorial/9-layerstack.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

A standard NNUE uses the same MLP for every position. SFNN LayerStack models still share the FeatureTransformer, but select a different small MLP stack depending on the position.

The important rule is simple: BulletOu and YaneuraOu must compute exactly the same LayerStack index for the same position.

## 9.1 Big picture

LayerStack selection is the product of three independent axes.

| Axis | What it looks at | Accepted tokens | Bucket count |
|---|---|---|---:|
| hand | side-to-move / non-side hand pieces | omitted, `hand64`, `hand256`, `hand1024` | 1 / 64 / 256 / 1024 |
| king | side-to-move / non-side king positions | omitted, `k3k3`, `k9k9`, `k9k9z`, `k13k13z`, `k21k21`, `k29k29` | 1 / 9 / 81 / 81 / 169 / 441 / 841 |
| progress | game progress | omitted, `progress2`, `progress3`, `progress4`, `progress8`, `progress16`, `progress32` | 1 / 2 / 3 / 4 / 8 / 16 / 32 |

The final stack count is a product:

| Example | hand | king | progress | LayerStacks |
|---|---:|---:|---:|---:|
| `SFNN_halfka2_1024_7_64` | 1 | 1 | 1 | 1 |
| `SFNN_halfka2_1024_7_64_k3k3` | 1 | 9 | 1 | 9 |
| `SFNN_halfka2_1024_7_64_k9k9z` | 1 | 81 | 1 | 81 |
| `SFNN_halfka2_1024_7_64_k13k13z` | 1 | 169 | 1 | 169 |
| `SFNN_halfka2_1024_7_64_hand256` | 256 | 1 | 1 | 256 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 256 | 9 | 1 | 2304 |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 1 | 9 | 8 | 72 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 256 | 9 | 16 | 36864 |

YaneuraOu composes the runtime index in this order:

```text
idx = hand_bucket
idx = idx * king_bucket_count + king_bucket
idx = idx * progress_bucket_count + progress_bucket
```

So `hand256_k3k3_progress16` means:

```text
idx = (hand256_bucket * 9 + k3k3_bucket) * 16 + progress16_bucket
```

## 9.2 Architecture names

LayerStack tokens may be written in any order, but BulletOu canonicalizes saved names as `hand → king → progress`.

| Accepted input | Canonical name |
|---|---|
| `SFNN_halfka2_1024_7_64_k3k3_hand256_progress16` | `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` |
| `SFNN_halfka2_1024_7_64_progress8_k9k9z` | `SFNN_halfka2_1024_7_64_k9k9z_progress8` |
| `SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k13k13z` | unchanged |

Grouped SFNN / common+shard L1 notation can be combined with the same suffixes:

```text
SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k3k3_progress16
```

This means KA2 input, FT=3072, L1 = 1024 common + 256 x 8 shards, and LayerStack = `hand256 × k3k3 × progress16`.

## 9.3 King buckets

King buckets normalize the friend king and enemy king to the side-to-move perspective, bucket each king, then combine the pair.

| Token | Buckets per king | Final king buckets | Rough intent |
|---|---:|---:|---|
| omitted | 1 | 1 | do not split by king position |
| `k3k3` | 3 | 9 | lightest coarse rank split |
| `k9k9` | 9 | 81 | exact king rank, no file information |
| `k9k9z` | 9 | 81 | same count as `k9k9`, but keeps file/3 detail near home ranks |
| `k13k13z` | 13 | 169 | ranks 1-7 exact, ranks 8-9 split by file/3 |
| `k21k21` | 21 | 441 | full file detail on the deepest two home ranks |
| `k29k29` | 29 | 841 | full file detail on the deepest three home ranks |

Long aliases are also accepted, such as `king9_by_king9`, `king9z_by_king9z`, `king9zone_by_king9zone`, `king13z_by_king13z`, and `king13zone_by_king13zone`.

### `k3k3`

Ranks are grouped into three bands for each king.

|  | enemy rank 1-3 | enemy rank 4-6 | enemy rank 7-9 |
|---|---:|---:|---:|
| friend rank 1-3 | 0 | 1 | 2 |
| friend rank 4-6 | 3 | 4 | 5 |
| friend rank 7-9 | 6 | 7 | 8 |

### `k9k9`

This uses the exact rank of each king, but ignores file.

| One king | Value |
|---|---|
| rank 1 | 0 |
| rank 2 | 1 |
| ... | ... |
| rank 9 | 8 |

Final bucket:

```text
bucket = friend_rank * 9 + enemy_rank
```

### `k9k9z`

This also has 81 final buckets, but the meaning of one king's 9 buckets differs from `k9k9`. Far ranks are merged, while deeper home ranks keep file/3 information.

| rank | file 1-3 | file 4-6 | file 7-9 |
|---|---:|---:|---:|
| 1-3 | 0 | 0 | 0 |
| 4-6 | 1 | 1 | 1 |
| 7 | 2 | 2 | 2 |
| 8 | 3 | 4 | 5 |
| 9 | 6 | 7 | 8 |

Final bucket:

```text
bucket = friend_single * 9 + enemy_single
```

### `k13k13z`

Ranks 1-7 are kept separately. Ranks 8-9 are split into three file bands.

| rank | file 1-3 | file 4-6 | file 7-9 |
|---|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 1 | 1 | 1 |
| 3 | 2 | 2 | 2 |
| 4 | 3 | 3 | 3 |
| 5 | 4 | 4 | 4 |
| 6 | 5 | 5 | 5 |
| 7 | 6 | 6 | 6 |
| 8 | 7 | 8 | 9 |
| 9 | 10 | 11 | 12 |

Final bucket:

```text
bucket = friend_single * 13 + enemy_single
```

`k13k13z` is a middle ground: more rank information than `k9k9z`, but far fewer stacks than `k21k21` or `k29k29`.

### `k21k21` / `k29k29`

These keep full file detail on deeper home ranks.

| Token | rank 1-3 | rank 4-6 | rank 7 | rank 8 | rank 9 | Buckets per king |
|---|---:|---:|---:|---:|---:|---:|
| `k21k21` | 0 | 1 | 2 | 3-11 | 12-20 | 21 |
| `k29k29` | 0 | 1 | 2-10 | 11-19 | 20-28 | 29 |

Final buckets:

```text
k21k21: bucket = friend_single * 21 + enemy_single
k29k29: bucket = friend_single * 29 + enemy_single
```

## 9.4 Hand buckets

Hand buckets combine a side-to-move hand bucket with a non-side hand bucket.

| Token | Buckets per side | Final hand buckets | What it captures |
|---|---:|---:|---|
| omitted | 1 | 1 | no hand split |
| `hand64` | 8 | 64 | coarse material-in-hand score |
| `hand256` | 16 | 256 | four presence bits |
| `hand1024` | 32 | 1024 | five presence bits |

### `hand64`

Each side's hand is scored and rounded into 8 buckets.

| Piece | Score |
|---|---:|
| pawn | 1 |
| lance / knight | 2 |
| silver / gold | 3 |
| bishop / rook | 5 |

```text
one_side_bucket = min((score + 3) / 4, 7)
bucket = stm_bucket * 8 + non_stm_bucket
```

### `hand256` and `hand1024`

| Token | Bit | Meaning |
|---|---:|---|
| `hand256` | bit0 | has pawn/lance/knight |
| `hand256` | bit1 | has silver/gold |
| `hand256` | bit2 | has bishop |
| `hand256` | bit3 | has rook |
| `hand1024` | bit0 | has pawn |
| `hand1024` | bit1 | has lance/knight |
| `hand1024` | bit2 | has silver/gold |
| `hand1024` | bit3 | has bishop |
| `hand1024` | bit4 | has rook |

```text
hand256:  bucket = stm_4bit_bucket * 16 + non_stm_4bit_bucket
hand1024: bucket = stm_5bit_bucket * 32 + non_stm_5bit_bucket
```

## 9.5 Progress buckets

`progressN` uses YaneuraOu-compatible SFNN progress parameters to compute a scalar progress value in `0..255`, then maps that value to N buckets.

| Token | Progress buckets |
|---|---:|
| `progress2` | 2 |
| `progress3` | 3 |
| `progress4` | 4 |
| `progress8` | 8 |
| `progress16` | 16 |
| `progress32` | 32 |

```text
progress_bucket = min(progress_0_255 * progress_bucket_count / 256,
                      progress_bucket_count - 1)
```

When `progressN` is used, the exported `nn.bin` includes a Progress section.

| Section | Contents |
|---|---|
| header | NNUE/SFNN header |
| FeatureTransformer | L0 bias/weight |
| Progress | `0x6f50524f`, `bias_q16`, `weights_q16[81][1548]` |
| LayerStack network | stack 0, stack 1, ... |

Pass `--sfnn-progress-params <file>` only when you already have the q16 progress parameter payload. If omitted, BulletOu writes zero progress parameters, matching YaneuraOu's dummy-zero behavior.

Note: the current CUDA factorizer layout does not allow `progressN` together with `king=axis` or `hand=axis`. The `shared` factorizer can be used with `progressN`.

## 9.6 Choosing a bucket scheme

These are practical starting points, not universal rules.

| Goal | Candidate | Comment |
|---|---|---|
| quick sanity check | no suffix / `k3k3` | small and easy to debug |
| finer than `k3k3` without many more stacks | `k9k9`, `k9k9z`, `k13k13z` | `k9k9z` keeps the same 81 stack count as `k9k9` |
| detailed king-position specialization | `k21k21`, `k29k29` | needs much more teacher data per experiment |
| split by hand pieces | `hand64`, `hand256`, `hand1024` | grows very quickly when combined with king buckets |
| split by opening/middlegame/endgame | `progress8`, `progress16` | depends on progress parameters |

More buckets give the model more specialization capacity, but reduce the amount of teacher data seen by each stack. For example, `hand256_k13k13z` has `256 * 169 = 43264` stacks, which is already a very large model.

## 9.7 Usage examples

Try a king zone bucket:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k13k13z \
    --teacher teachers/
```

Combine hand and king zone buckets:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand256_k9k9z \
    --teacher teachers/
```

Combine hand, king, and progress:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand256_k3k3_progress16 \
    --teacher teachers/
```

When loading the exported `nn.bin`, build YaneuraOu with the same architecture suffix.

---

Previous: [8. Load into an engine](8-engine.md)
