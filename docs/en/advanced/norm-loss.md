# Tensor-wise Norm loss

`--sfnn-norm-loss-strength` moves SFNN tensors toward L2 norm 1. Default: `0` (disabled). JSON: `"sfnn_norm_loss_strength": 0.0001`. Strength must be finite and non-negative; the target is fixed at 1. This is not hard normalization.

The reference is [bullet-shogi Ranger21 at commit 44941de4](https://github.com/nodchip/bullet-shogi/blob/44941de4/crates/acyclib/src/trainer/optimiser/ranger21.rs) and its `examples/shogi_layerstack.rs`, not Conductor. That example uses `1e-4`. **Only Norm loss is ported; BulletOu retains its existing optimizer, not the full Ranger21 algorithm.**

```text
n = sqrt(sum(w*w))
w *= 1 - learning_rate_norm * 2 * strength * (1 - 1/(n + 1e-7))
```

This is the decoupled correction approximately corresponding to `strength*(||W||₂-1)²`, not a gradient added to momentum/velocity. Zero tensors remain zero. Each stored tensor includes all buckets; this is not per-bucket, per-unit, or over folded weights.

| Tensor | Placement |
|---|---|
| FT weights, including factors | Excluded |
| FT bias | After RAdam/clipping, before Lookahead |
| L1 individual/shared weights and biases | Same |
| L2 weights and biases | Same |
| L3 weights | Same |
| L3 bias | Before RAdam |

BulletOu-specific L1 axis/pair tensors follow the L1 rule separately. Legacy L2/L3 factors, if updated, follow their layer's rule. Before placement uses the layer-adjusted LR; after placement retains the reference LR correction, using current optimizer betas and update step t:

```text
learning_rate_norm = lr * sqrt(1-beta2^t)
                    / ((1-beta1^t) * sqrt((1+beta2)^2+beta2^2))
```

Applied per optimizer update (every four batches for bpu=4). Frozen/excluded tensors are untouched. With dirty updates, Norm loss still scales unused buckets within the tensor; dirty optimizer-state rules remain unchanged.

Validation loss/qloss exclude this penalty. GPU computation avoids CPU weight readback, using about 1KiB scratch per context. Enabled mode adds full-tensor reads/writes and may slow training; disabled mode launches no dedicated kernels and allocates no Norm loss scratch. Reduction order differs from the reference, so bitwise equality is not guaranteed.

## Grid search

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
python .\scripts\grid_search.py `
  --settings-file .\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\20260915\grid-norm-loss `
  --grid sfnn_norm_loss_strength 0 0.000001 0.00001 0.0001
```

These four strengths are experimental candidates, not guaranteed optima. Strength is included in `grid_summary.csv`. Common settings (target offset, LR, epochs, initial checkpoint and teacher position) are preserved. Add `--dry-run` to inspect without training. Use a new grid root when the axis names differ from an existing grid.

For fine-tuning comparisons, use common `initial_state` and `initial_dataloader_pos`. Changing an existing trial's strength through resume is rejected; compare as a new condition. Checkpoint weight format is unchanged; older checkpoints without the option are interpreted as disabled and can resume unchanged.
