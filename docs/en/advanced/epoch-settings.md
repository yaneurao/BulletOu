# Per-epoch settings

## Switching L1 axis/shared factorization

```json
"sfnn_l1_factorizer": {
  "epoch1": "axis",
  "epoch5": "shared",
  "epoch8": "axis"
}
```

At epoch boundaries, axis-to-shared folds axis weights/biases into bucket base weights using current alpha/count coefficients. Lookahead slow weights receive the same transformation. Base/shared momentum, velocity and update steps survive; removed axis moments are discarded. Forward output is preserved apart from floating-point rounding, but subsequent optimizer dynamics are not equivalent.

Shared-to-axis retains base/shared parameters and adds zero axis weights, biases and optimizer states. It does not restore previously removed axes or extract them from base weights.

Supported for non-BN dense cuda-cpp SFNN production training, including k3k3, progress8 and combinations. Only existing architectural axes are used. Progress8 alone maps axes one-to-one to buckets, without cross-bucket sharing. BN/compact L1 and worker epoch schedules are unsupported. A zero residual gate with a nonzero axis coefficient raises an error instead of silently changing output.

Checkpoint restore checks actual axis presence to apply each boundary once. Future axis settings do not allocate axes in earlier shared epochs. `[L1 FACTORIZER]` records the transition and optimizer policy.

Use the common JSON in grid search. An explicit `--grid sfnn_l1_factorizer shared` overrides the entire schedule; omit that grid axis to retain epoch switching.

Standalone SFNN training and `grid_search.py` accept epoch maps in the common training JSON. Weights and optimizer state are retained; the process is not restarted.

```json
{
  "lr": {"epoch1": 0.000400, "epoch11": 0.000200},
  "lr_min": {"epoch1": 0.000050, "epoch11": 0.000030},
  "batches_per_update": {"epoch1": 1, "epoch11": 4},
  "sfnn_qat_l1": {"epoch1": true, "epoch11": false}
}
```

Merge these fields into a complete training configuration. Values apply from the specified epoch until the next boundary. Boolean transition directions work except for the one-way FT switch below. `epoch1` is required except for FT/L1/L2 revival enable controls, which default to false before their first entry. Keys must be `epoch1`, `epoch2`, etc. without leading zeros. Object order does not matter. Scalars still apply to every epoch.

`lr` and `lr_min` retain their within-epoch start/lower-limit meanings. Automatic step gamma is recalculated per epoch. Resume selects the resumed epoch's values; a separate run using `initial_state` starts at its epoch 1. Explicit CLI values and grid axes override the corresponding entire map. Settings are read at startup, not hot-reloaded.

`[epoch settings]` prints effective values at epoch boundaries. `grid_summary.csv` records resolved values per epoch; checkpoint `bulletou-settings.json` retains the original schedule.

## Supported fields

### Disable the FT factorizer at an epoch boundary

Non-BN `SFNN_halfka2` supports this one-way schedule (`ft_factorizer` is a legacy alias):

```json
"sfnn_ft_factorizer": {"epoch1": true, "epoch3": false}
```

Before the first batch of epoch 3, shared FT weights are folded into individual weights. Lookahead slow weights are folded separately and virtual rows are removed. Individual momentum/velocity and step counters survive; shared momentum/velocity is discarded. Effective weights are preserved (floating-point addition order can cause small differences), but future optimizer updates are not equivalent to ON mode. Units are not reset.

Use the same JSON with `--resume`: an ON checkpoint is folded on restoration into an OFF epoch; an already-OFF checkpoint is not folded again. OFF→ON, BN, other architectures, worker and plateau transitions are unsupported. A colored `[FT FACTORIZER]` line reports the conversion.

Grid search accepts the schedule in its common JSON. Remove any explicit `--grid ft_factorizer true` axis because it overrides the entire schedule. Settings are not hot-reloaded; resume from a saved checkpoint after changing them.

| Fields | Meaning |
|---|---|
| `lr`, `lr_min` | LR start and lower limit |
| `batches_per_update` | Accumulation batches per update |
| `ft_factorizer` / `sfnn_ft_factorizer` | Non-BN SFNN HalfKA2 ON→OFF automatic fold |
| `sfnn_qat_l1` | Enable/disable L1 QAT |
| `sfnn_bn_qat` | Enable/disable BN QAT; enable BN layers from startup, e.g. `{"epoch1":false,"epoch6":true}` |
| `sfnn_freeze_l1` | Freeze/unfreeze L1 |
| `sfnn_l1_lr_mult` | L1 LR multiplier |
| `sfnn_norm_loss_strength` | Norm regularization strength |
| `sfnn_saturation_penalty`, `sfnn_saturation_threshold` | Saturation penalty strength and threshold |
| `optimizer_weight_clip` | Weight clip width (0 disables) |
| `optimizer_weight_decay` | Weight decay strength |
| `bce_error_weight_k` | BCE error weighting (requires BCE) |

Unsupported maps fail explicitly. Architecture, batch size, teacher, non-FT factorizer structure and loss type cannot be scheduled. Worker/tuning, direct-step smoke and plateau are not supported.

## Accumulation alignment

Batches/SB are rounded down using **only the BPU active in that epoch**, preventing pending accumulated gradients at SB/checkpoint/epoch boundaries. Future settings do not affect current batch counts, LR periods, or optimizer update boundaries. With 40M positions, batch size 65,536, epoch1 BPU=1 and epoch11 BPU=4, epochs 1–10 use 610 batches/SB (39,976,960 positions); epochs 11 onward use 608 (39,845,888 positions). If max_epochs=5, the epoch11 entry has no effect on rounding.

The teacher shuffle window uses the settings active at startup and remains fixed within that execution; future BPUs do not affect it. Startup still validates the settings and builds the execution plan, but future values are not applied to current training computations.

[Grid search](grid-search.md) / [日本語](../../ja/advanced/epoch-settings.md)
## Unit revival at epoch boundaries

`sfnn_ft_revive`, `sfnn_l1_revive`, `sfnn_l2_revive`
run before the first training batch of each epoch where enabled. A scalar true runs every epoch.

```json
{
  "sfnn_l1_revive": {"epoch3": true, "epoch4": false},
  "sfnn_l2_revive": {"epoch3": true, "epoch4": false}
}
```

This example applies only at epoch3. Only these two settings may omit epoch1; the default before the first entry is false. Omitting epoch4:false keeps revival enabled for all later epochs.

Order is FT, L1, L2, with fresh 16-teacher-batch calibration using current weights. Require at least 1,024 positions per bucket and relative contribution strictly below the threshold. FT pairs must qualify in every bucket. See [contribution-based revival](l1-revive.md).

Mid-epoch resume skips revival; resume from the previous epoch end performs it. Interruption before saving recalibrates when restarting from the old checkpoint. FT factorizer and BN QAT transitions are applied before revival. Warmup epoch0 uses epoch1 settings.

L1 requires non-BN training; L2 supports non-BN or calibrated frozen-statistics BN QAT. Worker trials remain unsupported. Numbered audit CSVs and [REVIVE] epoch=N START/END messages record each pass.
