# Per-epoch settings

Standalone SFNN training and `grid_search.py` accept epoch maps in the common training JSON. Weights and optimizer state are retained; the process is not restarted.

```json
{
  "lr": {"epoch1": 0.000400, "epoch11": 0.000200},
  "lr_min": {"epoch1": 0.000050, "epoch11": 0.000030},
  "batches_per_update": {"epoch1": 1, "epoch11": 4},
  "sfnn_qat_l1": {"epoch1": true, "epoch11": false}
}
```

Merge these fields into a complete training configuration. Values apply from the specified epoch until the next boundary. Both boolean transition directions work. `epoch1` is required; keys must be `epoch1`, `epoch2`, etc. without leading zeros. Object order does not matter. Scalars still apply to every epoch.

`lr` and `lr_min` retain their within-epoch start/lower-limit meanings. Automatic step gamma is recalculated per epoch. Resume selects the resumed epoch's values; a separate run using `initial_state` starts at its epoch 1. Explicit CLI values and grid axes override the corresponding entire map. Settings are read at startup, not hot-reloaded.

`[epoch settings]` prints effective values at epoch boundaries. `grid_summary.csv` records resolved values per epoch; checkpoint `bulletou-settings.json` retains the original schedule.

## Supported fields

| Fields | Meaning |
|---|---|
| `lr`, `lr_min` | LR start and lower limit |
| `batches_per_update` | Accumulation batches per update |
| `sfnn_qat_l1` | Enable/disable L1 QAT |
| `sfnn_bn_qat` | Enable/disable BN QAT; enable BN layers from startup, e.g. `{"epoch1":false,"epoch6":true}` |
| `sfnn_freeze_l1` | Freeze/unfreeze L1 |
| `sfnn_l1_lr_mult` | L1 LR multiplier |
| `sfnn_norm_loss_strength` | Norm regularization strength |
| `sfnn_saturation_penalty`, `sfnn_saturation_threshold` | Saturation penalty strength and threshold |
| `optimizer_weight_clip` | Weight clip width (0 disables) |
| `optimizer_weight_decay` | Weight decay strength |
| `bce_error_weight_k` | BCE error weighting (requires BCE) |

Unsupported maps fail explicitly. Architecture, batch size, teacher, factorizer structure and loss type cannot be scheduled. Worker/tuning, direct-step smoke and plateau are not supported.

## Accumulation alignment

Batches/SB are rounded down using **only the BPU active in that epoch**, preventing pending accumulated gradients at SB/checkpoint/epoch boundaries. Future settings do not affect current batch counts, LR periods, or optimizer update boundaries. With 40M positions, batch size 65,536, epoch1 BPU=1 and epoch11 BPU=4, epochs 1–10 use 610 batches/SB (39,976,960 positions); epochs 11 onward use 608 (39,845,888 positions). If max_epochs=5, the epoch11 entry has no effect on rounding.

The teacher shuffle window uses the settings active at startup and remains fixed within that execution; future BPUs do not affect it. Startup still validates the settings and builds the execution plan, but future values are not applied to current training computations.

[Grid search](grid-search.md) / [日本語](../../ja/advanced/epoch-settings.md)
