# Revive constant L1 units on restore

These opt-in flags support the non-BN cuda-cpp SFNN trainer. Both default to false:

```json
"sfnn_l1_revive": true,
"sfnn_l1_revive_zero": true
```

`--sfnn-l1-revive` selects units whose normal **and** squared branches always output 1.
`--sfnn-l1-revive-zero` selects units whose two branches always output 0.
Zero upper-hit rate is NOT an always-zero activation. The skip output and squared-only saturation are excluded.
Grid syntax: `--grid sfnn-l1-revive false true` and `--grid sfnn-l1-revive-zero false true`.
Use a common `initial_state` checkpoint for A/B comparisons; scratch runs are rejected.

## Calibration and timing

Once after restoring, before training, infer 16 teacher batches at the restored data position,
without calibration shuffling or advancing the training cursor. This is not validation-set calibration.
Positions with zero training sample weight are excluded.
Both branches are inspected in the quantized proxy. A bucket must have at least 1,024 positions,
with 100% of them satisfying the requested condition. This finite sample cannot prove constancy on all positions.

Independent upper/zero completion flags are persisted in state.bin/weights.bin, including when no candidates exist.
Subsequent resumes skip completed kinds; this does not run at each epoch boundary.
You can enable zero revival after an upper-only checkpoint. Interrupting before saving reruns calibration from the old checkpoint.
The output folder receives `l1-revive.csv` with per-bucket/unit counts and selections; existing audits get numbered siblings.

## Mutation

1. Transfer both constant upper-branch contributions to L2 bias (nothing to transfer for zero units).
2. Glorot-uniform reinitialize the selected effective L1 input row; set bias to mean preactivation 0.5 on calibration inputs.
3. Zero the two outgoing L2 columns. They remain trainable: incoming L1 gradients resume after those columns move from zero.
4. Reset affected optimizer moments and independently compensate Lookahead slow weights.

Shared L1 weights remain unchanged: compensate through the selected bucket's residual row. Other units, FT and L3 are not reset.
Unlike L2 revival's small nonzero outgoing connection, L1 uses zero columns to avoid an immediate fresh-unit contribution.
Bias transfer is algebraically equivalent where the original branches are constant, not a guarantee of exact quantized
or unseen-position parity. Revival may saturate again and is not guaranteed to improve playing strength.

Supported: dense L1, factorizer none/shared, non-BN training including per-layer QAT, standalone/grid search.
BN, axis/pair, residual count gates, compact L1, legacy L2/L3 factorizers and worker trials are rejected explicitly.
The nn.bin inference format is unchanged; old checkpoints without completion flags are treated as unprocessed.

[L2 upper/zero revival](batch-normalization.md) also supports non-BN training. When all four flags are true, L1 revival runs before L2 calibration and revival.
