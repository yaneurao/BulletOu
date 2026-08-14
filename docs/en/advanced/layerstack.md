# LayerStack — pick a different small network per position

<a href="../../ja/advanced/layerstack.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

A standard NNUE uses the same late network for every position. SFNN LayerStack models share the input features and the FeatureTransformer, then switch the small MLP stack behind it depending on the position.

For example, `k3k3` has 9 stacks selected by a coarse king-position bucket. `hand256_k3k3` combines 256 hand buckets with 9 king buckets, so it has `256 * 9 = 2304` stacks.

The critical rule is that BulletOu and YaneuraOu must compute exactly the same LayerStack index for the same position. If the index differs by even one, the exported `nn.bin` may load successfully, but the engine will read the wrong stack.

## 1. What LayerStack changes

LayerStack does not change the input feature set. The input features and FeatureTransformer are shared; only the late network is split into stacks.

| Part | Shared across stacks? | Meaning |
|---|---|---|
| Input features | shared | `halfka2` / `ka2` etc. do not change |
| FeatureTransformer | shared | L0 weights are common to all stacks |
| Late MLP | per-stack | each bucket owns a small L1/L2/L3-style network |
| Output value | one stack per position | the bucket index selects which stack produces the value |

In practice, each bucket axis gives the late network a different specialization.

| Split by | Intent |
|---|---|
| king position | specialize by own/enemy king placement |
| hand pieces | specialize by material in hand, attack pieces, or endgame-like positions |
| progress | specialize by opening/middlegame/endgame phase |

The trade-off is teacher density. More buckets give the model more capacity, but each stack receives fewer samples.

## 2. The three independent axes

BulletOu's SFNN LayerStack selection is the product of three independent axes.

| Split | What it looks at | Architecture-name parts | Bucket count |
|---|---|---|---:|
| hand | side-to-move / non-side hand pieces | omitted, `hand4`, `hand16`, `hand64`, `hand64z`, `hand256`, `hand1024` | 1 / 4 / 16 / 64 / 64 / 256 / 1024 |
| king | side-to-move / non-side king positions | omitted, `k3k3`, `k9k9`, `k9k9z`, `k13k13z`, `k21k21`, `k29k29` | 1 / 9 / 81 / 81 / 169 / 441 / 841 |
| progress | game progress | omitted, `progress2`, `progress3`, `progress4`, `progress8`, `progress16`, `progress32` | 1 / 2 / 3 / 4 / 8 / 16 / 32 |

The final stack count is a product.

| Architecture | hand | king | progress | LayerStacks |
|---|---:|---:|---:|---:|
| `SFNN_halfka2_1024_7_64` | 1 | 1 | 1 | 1 |
| `SFNN_halfka2_1024_7_64_k3k3` | 1 | 9 | 1 | 9 |
| `SFNN_halfka2_1024_7_64_k9k9z` | 1 | 81 | 1 | 81 |
| `SFNN_halfka2_1024_7_64_k13k13z` | 1 | 169 | 1 | 169 |
| `SFNN_halfka2_1024_7_64_k21k21` | 1 | 441 | 1 | 441 |
| `SFNN_halfka2_1024_7_64_k29k29` | 1 | 841 | 1 | 841 |
| `SFNN_halfka2_1024_7_64_hand4` | 4 | 1 | 1 | 4 |
| `SFNN_halfka2_1024_7_64_hand16` | 16 | 1 | 1 | 16 |
| `SFNN_halfka2_1024_7_64_hand64` | 64 | 1 | 1 | 64 |
| `SFNN_halfka2_1024_7_64_hand64z` | 64 | 1 | 1 | 64 |
| `SFNN_halfka2_1024_7_64_hand256` | 256 | 1 | 1 | 256 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 256 | 9 | 1 | 2304 |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 1 | 9 | 8 | 72 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 256 | 9 | 16 | 36864 |

YaneuraOu composes the index in `hand → king → progress` order.

```text
idx = hand_bucket
idx = idx * king_bucket_count + king_bucket
idx = idx * progress_bucket_count + progress_bucket
```

So `hand256_k3k3_progress16` means:

```text
idx = (hand256_bucket * 9 + k3k3_bucket) * 16 + progress16_bucket
```

## 3. Architecture names

The `hand256`, `k3k3`, and `progress8` parts may be written in any order. BulletOu organizes output directory names as `hand → king → progress`.

| Accepted input | Canonical name |
|---|---|
| `SFNN_halfka2_1024_7_64_k3k3_hand256_progress16` | `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` |
| `SFNN_halfka2_1024_7_64_progress8_k9k9z` | `SFNN_halfka2_1024_7_64_k9k9z_progress8` |
| `SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k13k13z` | unchanged |

For SFNN, the middle number `H1` also decides whether the L1 PSQT shortcut is present.

| Example | fc0 outputs | Shortcut |
|---|---:|---|
| `SFNN_halfka2_1024_7_64_k3k3` | 8 (`7 + 1`) | yes |
| `SFNN_halfka2_1024_8_64_k3k3` | 8 | no |
| `SFNN_halfka2_1024_15_64_k3k3` | 16 (`15 + 1`) | yes |
| `SFNN_halfka2_1024_16_64_k3k3` | 16 | no |

In short, `H1 = 8n - 1` uses one extra shortcut output, while `H1 = 8n` does not. Split-L1 names such as `c0_s1024x4` divide the actual fc0 output count, so `4096_7_64_c0_s1024x4` divides 8 outputs into 4 groups, and `4096_8_64_c0_s1024x4` also divides 8 outputs into 4 groups.

You may also omit all of these parts:

```text
SFNN_halfka2_1024_7_64
```

This means LayerStacks=1: no split by king position, hand pieces, or progress.

Split-L1 SFNN names can be combined with the same LayerStack parts.

```text
SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k3k3_progress16
```

This means KA2 input, FT=3072, L1 = 1024 common + 256 x 8 shards, and LayerStack = `hand256 × k3k3 × progress16`.

## 4. Coordinate rule for king buckets

King buckets bucket the side-to-move king and the non-side king separately, then combine the two single-king buckets.

The important detail is that each king square is normalized to that king's own side.

| Term | Meaning |
|---|---|
| side-to-move king | the king of the side to move |
| non-side king | the opponent king |
| normalized rank 1 | enemy-camp side from that king's perspective |
| normalized rank 9 | deepest home rank from that king's perspective |
| normalized file 1-9 | file after the same side-based normalization |

So both black and white kings use the same bucket rules. A king deep in its own camp is near normalized rank 8-9 regardless of color.

The king bucket generally has this form:

```text
king_bucket = stm_king_single_bucket * single_bucket_count
            + non_stm_king_single_bucket
```

For `k29k29`, one king has 29 buckets:

```text
king_bucket = stm_king29 * 29 + non_stm_king29
```

That is why `k29k29` has `29 * 29 = 841` LayerStacks.

## 5. King bucket variants

King bucket variants choose how much position detail to keep.

| Token | Buckets per king | Pair buckets | What it emphasizes |
|---|---:|---:|---|
| omitted | 1 | 1 | no king-position split |
| `k3k3` | 3 | 9 | coarse rank grouping |
| `k9k9` | 9 | 81 | exact king rank, no file information |
| `k9k9z` | 9 | 81 | same 81 stacks as `k9k9`, but spends detail on home-rank file zones |
| `k13k13z` | 13 | 169 | ranks 1-7 exact, ranks 8-9 split by file/3 |
| `k21k21` | 21 | 441 | ranks 8-9 keep full square detail |
| `k29k29` | 29 | 841 | ranks 7-9 keep full square detail |

The `z` in `k9k9z` / `k13k13z` means zone. These are not plain rank splits; they spend resolution on areas expected to matter more.

Long aliases are also accepted, such as `king9_by_king9`, `king9z_by_king9z`, `king9zone_by_king9zone`, `king13z_by_king13z`, `king13zone_by_king13zone`, `king21_by_king21`, and `king29_by_king29`.

## 6. `k3k3` — the coarse baseline

`k3k3` splits one king into three groups: enemy side, center, and home side. The two kings together produce `3 * 3 = 9` buckets.

| Normalized rank | Single-king bucket |
|---|---:|
| 1-3 | 0 |
| 4-6 | 1 |
| 7-9 | 2 |

Pair buckets:

| STM king \ non-STM king | rank 1-3 | rank 4-6 | rank 7-9 |
|---|---:|---:|---:|
| rank 1-3 | 0 | 1 | 2 |
| rank 4-6 | 3 | 4 | 5 |
| rank 7-9 | 6 | 7 | 8 |

`k3k3` is light and keeps high teacher density, so it is a good first comparison point. It does not distinguish files on the home ranks.

## 7. `k9k9` and `k9k9z` — same 81 stacks, different budget

Both `k9k9` and `k9k9z` use 9 buckets per king, so both have `9 * 9 = 81` stacks. The difference is how those 9 buckets are spent.

| Token | How the 9 buckets are used | Best when |
|---|---|---|
| `k9k9` | normalized ranks 1-9 directly | you want a plain rank split |
| `k9k9z` | coarse far ranks, file/3 detail on home ranks 8-9 | you want home-rank file information without increasing stack count |

### `k9k9`

`k9k9` maps the normalized rank directly:

| Normalized rank | Single-king bucket |
|---|---:|
| 1 | 0 |
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |
| 5 | 4 |
| 6 | 5 |
| 7 | 6 |
| 8 | 7 |
| 9 | 8 |

Files are ignored. On rank 9, file 1, file 5, and file 9 all map to bucket 8.

### `k9k9z`

`k9k9z` keeps the same 81 total stacks as `k9k9`, but reallocates the 9 single-king buckets. It merges ranks 1-6 aggressively and uses the saved resolution on file zones at ranks 8-9.

| Normalized rank | file 1-3 | file 4-6 | file 7-9 | Meaning |
|---|---:|---:|---:|---|
| 1-3 | 0 | 0 | 0 | enemy-side ranks merged |
| 4-6 | 1 | 1 | 1 | center ranks merged |
| 7 | 2 | 2 | 2 | home-side rank, no file detail yet |
| 8 | 3 | 4 | 5 | home rank split into three file zones |
| 9 | 6 | 7 | 8 | deepest home rank split into three file zones |

Compared with `k9k9`:

| Area | `k9k9` | `k9k9z` |
|---|---|---|
| ranks 1-6 | keeps exact rank | merges them coarsely |
| rank 7 | rank only | rank only |
| ranks 8-9 | rank only | rank + file/3 |
| stack count | 81 | 81 |

`k9k9z` is not a strict upgrade over `k9k9`. It spends the same 81-stack budget differently: less rank detail far from home, more file detail near home.

## 8. `k13k13z` — keep rank detail, zone only deep home ranks

`k13k13z` uses 13 buckets per king, so the pair has `13 * 13 = 169` stacks.

It keeps ranks 1-7 as rank buckets, then splits ranks 8-9 by file/3.

| Normalized rank | file 1-3 | file 4-6 | file 7-9 | Meaning |
|---|---:|---:|---:|---|
| 1 | 0 | 0 | 0 | exact rank |
| 2 | 1 | 1 | 1 | exact rank |
| 3 | 2 | 2 | 2 | exact rank |
| 4 | 3 | 3 | 3 | exact rank |
| 5 | 4 | 4 | 4 | exact rank |
| 6 | 5 | 5 | 5 | exact rank |
| 7 | 6 | 6 | 6 | exact rank |
| 8 | 7 | 8 | 9 | home rank split by file/3 |
| 9 | 10 | 11 | 12 | deepest home rank split by file/3 |

`k13k13z` is the middle ground when `k9k9z` is too coarse but `k21k21` / `k29k29` is too large.

| Variant | Stacks | Comment |
|---|---:|---|
| `k9k9z` | 81 | light; ranks 1-6 are coarse |
| `k13k13z` | 169 | keeps ranks 1-7, zones only ranks 8-9 |
| `k21k21` | 441 | full square detail on ranks 8-9 |
| `k29k29` | 841 | full square detail on ranks 7-9 |

## 9. `k21k21` and `k29k29` — detailed home-side king location

`k21k21` and `k29k29` keep full square detail on the deep home ranks.

### `k21k21`

`k21k21` uses 21 buckets per king.

| Normalized rank | file 1 | file 2 | file 3 | file 4 | file 5 | file 6 | file 7 | file 8 | file 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 7 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 8 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| 9 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |

Ranks 8-9 keep all `2 * 9 = 18` squares, while the other ranks are coarser. The pair has `21 * 21 = 441` stacks.

### `k29k29`

`k29k29` uses 29 buckets per king.

| Normalized rank | file 1 | file 2 | file 3 | file 4 | file 5 | file 6 | file 7 | file 8 | file 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 7 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 8 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| 9 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 |

Ranks 7-9 keep all `3 * 9 = 27` squares. The pair has `29 * 29 = 841` stacks.

`k29k29` is meant to model detailed king placement near home: whether the king is on rank 7 or 9, and whether it is on an edge or central file.

The cost is teacher density:

| Variant | LayerStacks | Compared with `k3k3` |
|---|---:|---:|
| `k3k3` | 9 | 1.0x |
| `k9k9` / `k9k9z` | 81 | 9.0x |
| `k13k13z` | 169 | 18.8x |
| `k21k21` | 441 | 49.0x |
| `k29k29` | 841 | 93.4x |

So `k29k29` can learn more slowly in early accuracy because each stack sees fewer samples. That is expected; it is not automatically a bug. Compare it with enough data, and consider factorizer settings for large king buckets.

## 10. Hand buckets

Hand buckets combine the side-to-move hand bucket and the non-side hand bucket.

| Token | Buckets per side | Final hand buckets | What it sees |
|---|---:|---:|---|
| omitted | 1 | 1 | ignores hands |
| `hand4` | 2 | 4 | bishop presence |
| `hand16` | 4 | 16 | pawn and bishop presence |
| `hand64` | 8 | 64 | 3 presence bits |
| `hand64z` | 8 | 64 | coarse hand-piece score zone |
| `hand256` | 16 | 256 | 4 presence bits |
| `hand1024` | 32 | 1024 | 5 presence bits |

`hand64` and `hand64z` both produce 64 final buckets, but they look at different information. `hand64` uses hand-piece presence groups; `hand64z` uses coarse hand-piece score zones.

### `hand4` and `hand16`

`hand4` / `hand16` are lightweight hand buckets. They build a one-side bucket, then combine the side-to-move and non-side buckets.

| Token | One-side bucket | Final bucket |
|---|---|---|
| `hand4` | `has bishop ? 1 : 0` | `stm_1bit * 2 + non_stm_1bit` |
| `hand16` | bit0=`has pawn`, bit1=`has bishop` | `stm_2bit * 4 + non_stm_2bit` |

Use `hand4` when you only want a very cheap bishop-in-hand split, and `hand16` when pawn-in-hand should also be visible.

### `hand64`

`hand64` represents one side's hand with three presence-bit groups.

| bit | Meaning |
|---:|---|
| bit0 | has pawn/lance/knight |
| bit1 | has gold/silver/rook |
| bit2 | has bishop |

The one-side bucket is a 3-bit value in `0..7`. The final bucket is:

```text
hand64_bucket = stm_3bit * 8 + non_stm_3bit
```

Use `hand64` when bishop-in-hand should be independent, but `hand256` / `hand1024` are too fine-grained.

### `hand64z`

`hand64z` scores one side's hand and rounds it into 8 buckets.

| Piece | Score |
|---|---:|
| pawn | 1 |
| lance / knight | 2 |
| silver / gold | 3 |
| bishop / rook | 5 |

One-side bucket is `min((score + 3) / 4, 7)`.

| One-side bucket | score |
|---:|---|
| 0 | 0 |
| 1 | 1-4 |
| 2 | 5-8 |
| 3 | 9-12 |
| 4 | 13-16 |
| 5 | 17-20 |
| 6 | 21-24 |
| 7 | 25 or more |

Final bucket:

```text
hand64z_bucket = stm_bucket * 8 + non_stm_bucket
```

`hand64z` is useful when you care more about coarse hand-piece amount than exact hand-piece groups. It has the same 64-bucket count as `hand64`, but it partitions positions differently.

### `hand256` and `hand1024`

`hand256` / `hand1024` encode hand-piece presence bits.

| Token | bit | Meaning |
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

| Token | One-side bucket | Final bucket |
|---|---|---|
| `hand256` | 4 bits, 0-15 | `stm_4bit * 16 + non_stm_4bit` |
| `hand1024` | 5 bits, 0-31 | `stm_5bit * 32 + non_stm_5bit` |

`hand256` is a lighter piece-type split. `hand1024` keeps pawn presence separate, but grows much faster when combined with king buckets.

## 11. Progress buckets

`progressN` computes a scalar progress value in `0..255`, then maps that value to N buckets.

Add `progress8` or `progress16` to the architecture name, and progress becomes the third LayerStack axis. The Progress section required by `progressN` is exported by BulletOu as part of `nn.bin`.

Current BulletOu uses deterministic material-progress parameters based on hand material and promoted major pieces. This keeps BulletOu training and YaneuraOu inference on the same progress bucket assignment.

| Token | Progress buckets |
|---|---:|
| `progress2` | 2 |
| `progress3` | 3 |
| `progress4` | 4 |
| `progress8` | 8 |
| `progress16` | 16 |
| `progress32` | 32 |

Bucket mapping:

```text
progress_bucket = min(progress_0_255 * progress_bucket_count / 256,
                      progress_bucket_count - 1)
```

For example, `progress8` splits `0..255` into roughly 32-point ranges. `progress16` uses roughly 16-point ranges.

When `progressN` is used, the exported `nn.bin` includes a Progress section.

| Section | Contents |
|---|---|
| header | NNUE/SFNN header |
| FeatureTransformer | L0 bias/weight |
| Progress | `0x6f50524f`, `bias_q16`, `weights_q16[81][1548]` |
| LayerStack network | stack 0, stack 1, ... |

`progressN` can be combined with factorizer settings. `--sfnn-factorizer axis` shares the single king and hand axes. `--sfnn-factorizer pair` also enables available two-axis factorizers such as `king-progress` and `hand-progress`.

For example, `SFNN_halfka2_1024_7_64_k3k3_hand64_progress2` can use:

```bash
--sfnn-factorizer pair
```

That setting enables the supported subset of `shared`, `king-axis`, `hand-axis`, `king-hand`, `king-progress`, and `hand-progress` for the selected architecture.

## 12. How large combinations get

The hand / king / progress axes multiply, so combinations can become huge quickly.

| Architecture-name part | LayerStacks | Comment |
|---|---:|---|
| none | 1 | no LayerStack split |
| `k3k3` | 9 | smallest king bucket |
| `k9k9z` | 81 | same size as `k9k9` |
| `k13k13z` | 169 | middle ground |
| `k29k29` | 841 | still reasonable if king-only |
| `hand4` | 4 | lightweight bishop-in-hand split |
| `hand16` | 16 | lightweight pawn/bishop-in-hand split |
| `hand64_k3k3` | 576 | light hand + king split |
| `hand64z_k3k3` | 576 | score-zone hand split + king |
| `hand256_k3k3` | 2304 | useful comparison point |
| `hand256_k9k9z` | 20736 | already large |
| `hand256_k13k13z` | 43264 | watch teacher data, VRAM, and checkpoint size |
| `hand1024_k29k29` | 861184 | extremely heavy |

Large choices affect not only training speed, but also save size, validation time, and teacher density per stack.

## 13. Choosing a bucket scheme

These are practical starting points, not universal rules.

| Goal | Candidate | Comment |
|---|---|---|
| quick sanity check | no extra part / `k3k3` | small and easy to debug |
| finer than `k3k3` | `k9k9` | plain rank split |
| keep 81 stacks but add home-rank file zones | `k9k9z` | same size as `k9k9`, different budget |
| `k9k9z` is too coarse | `k13k13z` | keeps ranks 1-7 and zones ranks 8-9 |
| detailed deepest home ranks | `k21k21` | lighter than `k29k29` |
| detailed ranks 7-9 | `k29k29` | high capacity, lower teacher density |
| split by hand pieces | `hand4`, `hand16`, `hand64`, `hand64z`, `hand256`, `hand1024` | grows quickly with king buckets |
| split by game phase | `progress8`, `progress16` | independent third axis combined with hand / king |

Larger buckets may show slower early accuracy because each stack receives fewer samples. Compare not only short-run accuracy, but also long-run loss, actual engine strength, checkpoint size, and training speed.

## 14. Usage examples

Smallest king bucket:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --teacher teachers/
```

Same 81 stacks as `k9k9`, but with home-rank file zones:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k9k9z \
    --teacher teachers/
```

Detailed `k29k29` king bucket:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k29k29 \
    --teacher teachers/
```

Combine hand and king zones:

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

When loading the exported `nn.bin`, build YaneuraOu with the same architecture name.

---

Previous: [Advanced guide](README.md)
