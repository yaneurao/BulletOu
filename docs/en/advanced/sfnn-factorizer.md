# SFNN factorizer

<a href="../../ja/advanced/sfnn-factorizer.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

This page explains `--sfnn-factorizer` for SFNN LayerStack architectures.

You do not need this page for a first training run. Read it when you want to compare architectures with many buckets, such as `hand1024`, `k29k29`, or `progress8`.

## 1. What the factorizer does

LayerStack switches the later network by position bucket. For example:

```text
SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4
```

This architecture has:

```text
hand1024 * k3k3 * progress4 = 1024 * 9 * 4 = 36,864 stacks
```

If every stack owns independent weights, the model becomes more expressive. The cost is teacher density: each individual stack receives fewer positions. Rare buckets can become noisy, and validation loss or post-quantization loss can become unstable.

The factorizer mitigates this by letting stacks share common components. Roughly, each stack weight is represented as a sum of individual and shared components.

```text
W_effective = W_base + W_shared + W_axis + W_pair
```

`W_effective` is the weight used by forward propagation. Here, `W` means one element of an SFNN L1/L2/L3 stack weight or bias tensor.

## 2. Bucket axes and stack index

BulletOu composes the LayerStack index in this order:

```text
stack = ((hand_bucket * king_bucket_count) + king_bucket) * progress_bucket_count
      + progress_bucket
```

If an architecture does not have an axis, that axis has one bucket.

| Axis | Examples | Meaning |
|---|---|---|
| hand | `hand4`, `hand16`, `hand64`, `hand64z`, `hand256`, `hand1024` | hand-piece state for side-to-move and opponent |
| king | `k3k3`, `k9k9`, `k9k9z`, `k13k13z`, `k21k21`, `k29k29` | king-position bucket |
| progress | `progress2`, `progress4`, `progress8`, `progress16`, `progress32` | game-progress bucket |

The factorizer uses these hand / king / progress axes to decide which stacks share components.

## 3. `shared`

`shared` adds one common component to every stack.

```text
W_effective[hand, king, progress]
  = W_base[hand, king, progress]
  + alpha_shared * W_shared
```

This is the coarsest sharing. BulletOu defaults to `--sfnn-factorizer shared`.

## 4. `axis`

`axis` adds components for each single bucket axis.

```text
W_effective[hand, king, progress]
  = W_base[hand, king, progress]
  + alpha_shared   * W_shared
  + alpha_hand     * W_hand_axis[hand]
  + alpha_king     * W_king_axis[king]
  + alpha_progress * W_progress_axis[progress]
```

Only axes present in the architecture are used. For example, plain `k3k3` has no hand/progress axis. `hand1024_k3k3_progress4` has all three axes.

### Hand-axis decomposition

A hand bucket is the product of the side-to-move hand bucket and the opponent hand bucket.

| Setting | Per-side bucket count `D` | Total hand buckets |
|---|---:|---:|
| `hand4` | 2 | 4 |
| `hand16` | 4 | 16 |
| `hand64` | 8 | 64 |
| `hand64z` | 8 | 64 |
| `hand256` | 16 | 256 |
| `hand1024` | 32 | 1024 |

The composition is:

```text
hand_bucket = stm_hand_bucket * D + non_stm_hand_bucket
```

`hand=axis` does not store 1024 independent axis components for `hand1024`. It decomposes the hand axis into the two per-side directions:

```text
W_hand_axis[hand_bucket]
  = W_hand_stm_axis[stm_hand_bucket]
  + W_hand_non_stm_axis[non_stm_hand_bucket]
```

For `hand1024`, each side has 32 buckets, so the hand-axis component count is `32 + 32 = 64`. This is much smaller than 1024 direct components and gives rare hand combinations a useful shared signal.

### King axis / progress axis

King and progress axes follow the same idea.

```text
W_king_axis[king_bucket]
W_progress_axis[progress_bucket]
```

For `k3k3`, the king bucket is a product of three side-to-move king zones and three opponent king zones. BulletOu internally decomposes king axis into those two directions. So `k3k3` has `3 + 3 = 6` king-axis components.

For `progress8`, the progress axis has 8 components.

## 5. `pair`

`pair` shares components for pairs of axes in addition to single axes.

```text
W_effective[hand, king, progress]
  = W_base[hand, king, progress]
  + alpha_shared   * W_shared
  + alpha_hand     * W_hand_axis[hand]
  + alpha_king     * W_king_axis[king]
  + alpha_progress * W_progress_axis[progress]
  + alpha_pair     * W_king_hand_pair[hand, king]
  + alpha_pair     * W_king_progress_pair[king, progress]
  + alpha_pair     * W_hand_progress_pair[progress, hand]
```

`--sfnn-factorizer pair` also enables `shared` and all available axis components. It does not mean “pair terms only”.

For `hand1024_k3k3_progress4`, the available pair terms are:

| Pair term | Sharing meaning | Component count |
|---|---|---:|
| `king-hand` | Same hand and king, shared across progress | `1024 * 9 = 9,216` |
| `king-progress` | Same king and progress, shared across hand | `9 * 4 = 36` |
| `hand-progress` | Same hand and progress, shared across king | `1024 * 4 = 4,096` |

For example, `hand-progress` means “use a component for the same hand state and progress bucket, regardless of king bucket.”

## 6. CLI settings

Common settings:

| Setting | Meaning |
|---|---|
| `--sfnn-factorizer shared` | Use one common component for every stack. Default |
| `--sfnn-factorizer none` | Disable factorizer |
| `--sfnn-factorizer axis` | Use every hand / king / progress single axis present in the architecture |
| `--sfnn-factorizer pair` | Use `shared`, available axis terms, and available pair terms |
| `--sfnn-factorizer king=axis,hand=axis` | Specify axes explicitly |
| `--sfnn-factorizer king-hand,hand-progress` | Specify pair terms explicitly |

Example using pair terms for `hand1024_k3k3_progress4`:

```bash
--arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4
--sfnn-factorizer pair
```

This enables the supported subset of:

```text
shared
king-axis
hand-axis
progress-axis
king-hand
king-progress
hand-progress
```

## 7. `--sfnn-factorizer-alpha`

`--sfnn-factorizer-alpha` controls how strongly factorizer terms are added during forward propagation.

```text
W_effective = W_base + alpha * W_factorizer
```

`alpha=1.0` is the standard value. `alpha=2.0` adds that component at twice its stored value, and the gradient into that factorizer tensor is also doubled. The accepted range is `0.0` to `10.0`.

Set every term to the same strength:

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha all=3.0
```

Set all single-axis terms and all pair terms:

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha axis=4.0,pair=4.0
```

Weaken only the hand-axis term:

```bash
--sfnn-factorizer axis
--sfnn-factorizer-alpha hand=0.80
```

`hand=` changes only the hand-axis strength. Pair terms such as `hand-progress` and `king-hand` are controlled by `pair=`.

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha hand=0.80,pair=2.0
```

In this example, hand-axis uses 0.8 and pair terms use 2.0.

You can combine `all=` with more specific keys. Later keys win.

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha all=3.0,pair=4.0
```

Here, `shared` and `axis` use 3.0, while `pair` uses 4.0.

## 8. Export to `nn.bin`

When BulletOu writes `nn.bin`, it folds factorizer terms into `W_effective`.

```text
W_export = W_base + alpha_shared * W_shared + ...
```

The engine does not need to know about the factorizer. It reads ordinary folded stack weights.

`state.bin` keeps base weights and factorizer tensors separately. When you change factorizer settings for continued training, check the startup log to confirm which tensors are active.

## 9. Choosing settings

| Situation | Good candidates |
|---|---|
| Plain `k3k3` is already stable | `shared` or `axis` |
| Many king buckets such as `k29k29` | `king=axis` |
| Using `hand1024` | `hand=axis`, or `pair` |
| Combining axes such as `hand1024_k3k3_progress4` | `pair` |
| qloss is unstable | Raise `alpha`, or try the saturation penalty |
| Factorizer feels too restrictive | Lower `alpha`, or do a short `none` fine-tune |

For large bucket configurations, `none` lets each stack learn independently, but rare buckets can drift. `axis` and `pair` restrict the model a little, in exchange for sharing evidence between related buckets.

## 10. Saturation penalty

With strong factorizer settings or many buckets, folded i8 weights in `nn.bin` can hit the quantization edge. If post-quantization loss or accuracy is the main problem, you can try:

```bash
--sfnn-saturation-penalty 1e-7
```

This is off by default.

## 11. Count-aware decay for rare buckets

Architectures such as `hand1024_k3k3_progress8` have many stacks, and some buckets may appear only rarely. If a rare bucket is allowed to learn a large independent residual from just a few positions, it can drift away from the shared structure.

BulletOu can pre-count bucket occurrences from the teacher data and use that count to apply a weaker or stronger decay to the base stack residual. Here, “residual” means the per-stack base weight, not the shared factorizer tensor.

First, create a count.bin file:

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\sojoteam_datasets `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --nn-bin C:\path\to\same-arch\nn.bin `
  --positions 500000000 `
  --buffer-mb 1024 `
  --read-buffers 3 `
  --output D:\BulletOu-snapshots\counts\hand1024-k3k3-progress4-count.bin
```

For architectures containing `progressN`, `--nn-bin` is required. The progress bucket is determined by the `bias_q16` / `weights_q16` values in the Progress section of `nn.bin`, so count.bin must be built against the same `nn.bin`. Architectures without `progressN` do not need `--nn-bin`.

When you use count.bin with a `progressN` architecture, keep the progress bucket assignment aligned with the `nn.bin` used to create the count file. If the progress parameters keep changing after count.bin is built, the bucket distribution represented by count.bin drifts away from the buckets used during training.

For count-aware fine-tuning from the checkpoint that produced count.bin, freeze progress:

```powershell
--sfnn-bucket-counts D:\...\count.bin `
--sfnn-freeze-progress
```

With `--sfnn-freeze-progress`, BulletOu does not update the progress parameters. Training also uses the same hard q16 Progress bucket rule that is exported to `nn.bin`. Validation batches can keep their GPU cache as long as the progress parameters do not change.

If you omit `--positions`, BulletOu scans every file in the teacher path once. Keep `--positions` when you want to sample only a prefix of a very large teacher set.

For fixed-size `.psv` / `.bin` records, BulletOu uses a dedicated fast path. It keeps several read buffers and reads into one buffer while counting another one. The implementation uses queues, but the buffers are reused like a ring buffer.

| Option | Meaning | Typical value |
|---|---|---|
| `--buffer-mb` | Size of one read buffer | default `1024` |
| `--read-buffers` | Number of read buffers | default `3`, minimum `2` |

Memory use is roughly `--buffer-mb × --read-buffers`. For example, `--buffer-mb 1024 --read-buffers 4` uses about 4GiB for read buffers. Larger is not always faster. If disk throughput is uneven, try `3` or `4`; for small cached inputs, smaller buffers may be faster.

Progress output separates the average speed from the recent interval speed:

```text
[count] ... avg_pos/s=... inst_pos/s=... read_wait=... count=...
```

`avg_pos/s` is the average since start. `inst_pos/s` is the speed of the latest progress interval. If `read_wait` is large, the run is mostly waiting for disk reads. If `count` is large, bucket decode/count is the main cost.

Then pass it during training:

```powershell
--sfnn-factorizer pair `
--sfnn-factorizer-alpha all=1.0 `
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\hand1024-k3k3-progress4-count.bin
```

`count.bin` stores counts for full LayerStack buckets. The same file can also be used for axis and pair factorizer confidence; BulletOu derives those counts by summing the stacks that touch each axis or pair row.

When `--sfnn-bucket-counts <count.bin>` is set and an SFNN factorizer is active, BulletOu enables the residual count gate by default. This is not an extra regularization loss. It directly scales the bucket-specific residual used by forward according to that bucket's count.

### Residual count gate

With a factorizer, the effective weight can be read as:

```text
W_effective =
    gate_stack * W_residual
  + shared_alpha * W_shared
  + axis_alpha   * confidence_axis * W_axis
  + pair_alpha   * confidence_pair * W_pair
```

`gate_stack` is computed per stack:

```text
residual_params_per_bucket = number of bucket-specific residual parameters per bucket
K = residual_params_per_bucket * --sfnn-residual-count-gate-confidence
gate_stack = count_stack / (count_stack + K)
```

When `--sfnn-bucket-counts` is set, the default `--sfnn-residual-count-gate-confidence` is `1.0`. This means: do not strongly trust a bucket-specific residual until that bucket has appeared about as many times as its own residual parameter count.

This gate is used consistently by training forward, gradient flow, GPU quantized validation, and `nn.bin` export. The qvalid path and exported `nn.bin` therefore use the same model formula.

Disable it explicitly when you want to load the count file only for statistics or for other count-confidence options:

```powershell
--sfnn-residual-count-gate-confidence 0
```

| count | Behavior |
|---:|---|
| `0` | Do not use the bucket-specific residual; rely on factorizer terms |
| `K` | Use 50% of the bucket-specific residual |
| `9K` | Use 90% of the bucket-specific residual |
| Very large count | `gate_stack` approaches 1 |

For compact-L1 SFNN storage, L1 is held as compact weights, so this gate applies to L2/L3 bucket-specific residuals. For dense L1 factorizer layouts, the same idea also applies to L1.

### Count-aware residual decay

Separately from the residual count gate, BulletOu can add a count-aware decay term to the optimizer gradient. This is experimental. Enable it with:

```powershell
--sfnn-bucket-counts D:\...\count.bin `
--sfnn-residual-count-confidence 1.0
```

The per-stack decay coefficient is:

```text
decay_stack = max_decay * min(1, sqrt((confidence_count + 1) / (count_stack + 1)))
```

`max_decay` is the maximum decay. If you enable residual count confidence and omit `--sfnn-residual-count-decay`, BulletOu uses `max_decay = 1e-7`. Override it with `--sfnn-residual-count-decay <value>` only when you need to tune the maximum decay itself.

`confidence_count` is computed from the model shape:

```text
residual_params_per_bucket = number of bucket-specific residual parameters per bucket
confidence_count = residual_params_per_bucket * --sfnn-residual-count-confidence
```

For example, `--sfnn-residual-count-confidence 1.0` means: do not trust a bucket-specific residual much until that bucket has appeared about as many times as its own residual parameter count. This is based on model degrees of freedom, not on a fraction of the total teacher positions.

| count | Behavior |
|---:|---|
| `count <= confidence_count` | Use the maximum `max_decay` |
| `count = 4 * confidence_count` | About `max_decay / 2` |
| Very large count | Almost no extra decay |

This decay does not directly affect factorizer tensors. `shared` / `axis` / `pair` components remain available, while the bucket-specific residual receives a count-dependent regularization gradient.

The residual count gate and residual count decay are separate controls. In most experiments, start with the gate alone and add decay only if you need stronger stabilization.

```text
gate:  scale W_residual in the forward formula
decay: add a regularization term to gradients before optimizer update
```

### Count-aware axis / pair confidence

These options dampen factorizer rows themselves:

```powershell
--sfnn-bucket-counts D:\...\count.bin `
--sfnn-axis-count-confidence 1.0 `
--sfnn-pair-count-confidence 1.0
```

The common axis option is used for king / hand / progress axis rows unless you override one of them:

```powershell
--sfnn-axis-count-confidence 2.0 `
--sfnn-king-axis-count-confidence 4.0 `
--sfnn-hand-axis-count-confidence 0.5 `
--sfnn-progress-axis-count-confidence 0.0
```

The common pair option is used for king-hand / king-progress / hand-progress pair rows unless you override one of them:

```powershell
--sfnn-pair-count-confidence 10.0 `
--sfnn-king-hand-pair-count-confidence 4.0 `
--sfnn-king-progress-pair-count-confidence 8.0 `
--sfnn-hand-progress-pair-count-confidence 20.0
```

For each axis or pair row, BulletOu sums the counts of all LayerStack buckets that use that row. It then multiplies the corresponding factorizer contribution by:

```text
confidence = count_term / (count_term + term_params * option_value)
```

`term_params` is the number of parameters held by one axis or pair row across L1/L2/L3. If the option value is `0`, the multiplier is `1` and the factorizer row is not damped. If a row has count `0` and the option is enabled, its multiplier is `0`.

For example, if `--sfnn-axis-count-confidence 2.0` and `--sfnn-king-axis-count-confidence 4.0` are both specified, king-axis rows use `4.0`; hand-axis and progress-axis rows use `2.0`. The same rule applies to pair rows.

With both alpha and count confidence, the effective weight is:

```text
W_effective =
    gate_stack * W_residual
  + shared_alpha * W_shared
  + axis_alpha   * confidence_axis * W_axis
  + pair_alpha   * confidence_pair * W_pair
```

This is useful when you want `shared` to remain as the broad prior, while axis or pair rows that have almost no observations are kept weak until the data supports them.

### `count.bin` file format

Normally you create this file with `bulletou.exe bucket-count`; you do not need to write it by hand. If you want to inspect it from another tool, the layout is below. All integer fields are little-endian.

| Order | Type | Meaning |
|---:|---|---|
| 1 | `u8[8]` | Magic bytes: ASCII `BOUCNT1\0` |
| 2 | `u32` | Version. Currently `1` |
| 3 | `u32` | Byte length of the architecture name |
| 4 | `u8[arch_len]` | UTF-8 architecture name, for example `SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4` |
| 5 | `u64` | Number of positions scanned |
| 6 | `u32` | Number of stacks |
| 7 | `u32[stack_count]` | Occurrence count for each LayerStack bucket |

`counts[i]` is the occurrence count for LayerStack bucket index `i`. When you pass the file with `--sfnn-bucket-counts`, BulletOu checks that the architecture name and stack count match the current `--arch`.

Counts are stored as `u32`, so one bucket cannot exceed 4,294,967,295 occurrences. If a bucket would overflow, count a smaller prefix with `--positions`.
