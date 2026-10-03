# Revive constant L1 units at epoch start

These opt-in flags support the non-BN cuda-cpp SFNN trainer. Both default to false:

```json
"sfnn_l1_revive": true,
"sfnn_l1_revive_zero": true
```

`--sfnn-l1-revive` selects units whose normal **and** squared branches jointly output 1 at or above the threshold.
`--sfnn-l1-revive-zero` selects units whose two branches jointly output 0 at or above the threshold.
Zero upper-hit rate is NOT an always-zero activation. The skip output and squared-only saturation are excluded.
Grid syntax: `--grid sfnn_l1_revive false true` and `--grid sfnn_l1_revive_zero false true`.
Use a common `initial_state` checkpoint for A/B comparisons; scratch runs are also supported.

## Calibration and timing

Before the first training batch of each enabled epoch, infer 16 teacher batches at its data position,
without calibration shuffling or advancing the training cursor. This is not validation-set calibration.
Positions with zero training sample weight are excluded.
Both branches are inspected in the quantized proxy. A bucket must have at least 1,024 positions,
with the configured fraction satisfying the requested condition. This finite sample cannot prove constancy on all positions.

### Thresholds for L1 and L2

```json
"sfnn_l1_revive_threshold": 0.99,
"sfnn_l1_revive_zero_threshold": 0.99,
"sfnn_l2_revive_threshold": 0.99,
"sfnn_l2_revive_zero_threshold": 0.99
```

All default to 0.99, accept finite values in (0, 1], and use inclusive comparisons.
Use 1.0 for the previous all-observations criterion. Thresholds do not enable revival by themselves.
Epoch maps such as `{"epoch1": 1.0, "epoch9": 0.99}` require epoch1.
Grid example: `--grid sfnn_l1_revive_threshold 0.99 1.0`.
If very low thresholds allow both conditions, the enabled upper condition takes precedence.
Reviving a nonconstant unit is approximate: outputs may change on nonqualifying positions.
Stdout reports measured fractions and thresholds and prints a yellow WARNING for nonconstant selections.
Audit CSVs also store thresholds; existing audit files and source checkpoints are not overwritten.

Independent upper/zero completion flags are persisted in state.bin/weights.bin, including when no candidates exist.
Recalibrate at the next enabled epoch boundary. A mid-epoch checkpoint resume does not repeat revival.
All four revival controls support epoch maps, defaulting to false before the first entry and carrying values forward afterward.
`"sfnn_l1_revive": {"epoch3": true, "epoch4": false}` applies only to epoch3.
A scalar true applies at every epoch start (including warmup epoch0). See [epoch settings](epoch-settings.md).
Interrupting before saving reruns calibration from the old checkpoint.
The output folder receives `l1-revive.csv` with per-bucket/unit counts and selections; existing audits get numbered siblings.

## Mutation

1. Transfer both constant upper-branch contributions to L2 bias (nothing to transfer for zero units).
2. Glorot-uniform reinitialize the selected effective L1 input row; set bias to mean preactivation 0.5 on calibration inputs.
3. Set both outgoing L2 columns to ±1/64, retaining each old sign (zero becomes positive). These connections survive L2 QAT, avoiding a wait for outgoing masters to cross the quantization threshold before upstream gradients can flow.
4. Reset affected optimizer moments and independently compensate Lookahead slow weights.

Shared L1 weights remain unchanged: compensate through the selected bucket's residual row. Other units, FT and L3 are not reset.
If any units are selected, replay the same 16 teacher batches with the revived quantized proxy, measuring normal and squared branch means separately. Subtract each new connection times its branch mean from L2 bias; compensate Lookahead slow bias independently.
The training cursor does not advance, and large teacher input arrays are not retained. This extra inference pass occurs only at revival.
Like L2 revival, this is small-nonzero-connection plus mean compensation. It compensates mean L2 preactivation, not each position's output or mean final output. Even originally constant units are no longer pointwise-equivalent. Revival may saturate again and is not guaranteed to improve playing strength.

Supported: dense L1, factorizer none/shared, non-BN training including per-layer QAT, standalone/grid search.
BN, axis/pair, residual count gates, compact L1, legacy L2/L3 factorizers and worker trials are rejected explicitly.
The nn.bin inference format is unchanged; old checkpoints without completion flags are treated as unprocessed.

[L2 upper/zero revival](batch-normalization.md) also supports non-BN training. When all four flags are true, L1 revival runs before L2 calibration and revival.
