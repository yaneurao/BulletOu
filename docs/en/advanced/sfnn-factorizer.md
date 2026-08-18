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
  --positions 500000000 `
  --output D:\BulletOu-snapshots\counts\hand1024-k3k3-progress4-count.bin
```

Then pass it during training:

```powershell
--sfnn-factorizer pair `
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\hand1024-k3k3-progress4-count.bin `
--sfnn-residual-count-decay 1e-7 `
--sfnn-residual-count-decay-k 10000
```

The per-stack decay coefficient is:

```text
lambda_stack = lambda0 * min(1, sqrt((K + 1) / (count_stack + 1)))
```

`lambda0` is `--sfnn-residual-count-decay`, and `K` is `--sfnn-residual-count-decay-k`.

| count | Behavior |
|---:|---|
| `count <= K` | Use the maximum decay `lambda0` |
| `count = 4K` | About `lambda0 / 2` |
| Very large count | Almost no extra decay |

This does not directly decay the factorizer tensors. `shared` / `axis` / `pair` components remain available, while the bucket-specific residual is damped according to count. The effect is: trust the shared structure first, and let heavily observed buckets learn stronger individual residuals.

If you pass only `--sfnn-bucket-counts`, BulletOu validates the file and prints count statistics, but it does not change training. Set `--sfnn-residual-count-decay` above zero to enable the regularization.
