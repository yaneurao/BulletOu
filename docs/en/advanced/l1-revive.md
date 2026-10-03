# Reviving low-contribution L1/L2 units at epoch boundaries

```json
"sfnn_l1_revive": true,
"sfnn_l2_revive": true,
"sfnn_l1_revive_contribution_threshold": 0.01,
"sfnn_l2_revive_contribution_threshold": 0.01
```

Enables default to false; thresholds default to 0.01, finite in `(0, 1]`.
Select strictly below the threshold: 0.01 means below 1%, not at or below 1%.
These are relative contribution scores, not saturation rates or percentages of playing strength.

$$U_i=\mathbb E[|h_i-\bar h_i|]\sum_j|w_{ji}|,\qquad R_i=U_i/\max_j U_j$$

Use empirical mean absolute deviation (MAD), not EMA or variance. Normalize within each layer/bucket.
L1 adds square and normal branch utilities, without cross-branch cancellation; exclude skip.
L2 uses its outgoing L3 weight. Activations and outgoing weights come from the quantized GPU proxy.
If every U in a bucket is zero, every R is zero and all units with sufficient coverage qualify.
This includes mixed constants (normal=0, square=1), zero outputs, and zero quantized outgoing weights.

## Calibration and timing

Before the first batch of each enabled epoch (including warmup epoch0), read 16 unshuffled teacher batches
from the current cursor, without advancing the training cursor. Exclude sample-weight-zero positions.
Require at least 1,024 positions per bucket. Neither validation data nor target labels are used.
Store outputs temporarily in CPU RAM for exact MAD (about 256MiB for 64 units and a million positions), not additional VRAM.
With both layers enabled, revive L1 first, then calibrate L2 on the resulting network.
Scores are finite-sample estimates, not guarantees over all possible positions.

```json
"sfnn_l1_revive": {"epoch3": true, "epoch4": false},
"sfnn_l2_revive": {"epoch3": true, "epoch4": false},
"sfnn_l1_revive_contribution_threshold": {"epoch1": 0.01, "epoch9": 0.02}
```

Enable maps default to false before their first entry; threshold maps require epoch1.
Checkpoint markers prevent duplicate mid-epoch revival; each new enabled epoch recalibrates.
Interruption before saving the modified checkpoint repeats calibration from the original checkpoint.

```powershell
python .\grid_search.py --settings-file settings.json --output-folder results `
  --grid sfnn_l1_revive true --grid sfnn_l1_revive_contribution_threshold 0.005 0.01 0.02
```

## Reset and mean compensation

Transfer each old branch's measured mean times its outgoing weight into the next bias (not an assumed 0/1).
Glorot-initialize incoming weights and center mean preactivation at 0.5. Preserve L1 shared weights by subtracting them from individual weights.
Restart outgoing connections at the old sign times 1/64, then subtract their new mean contribution from the downstream bias.
For L1, replay the same 16 batches after resetting to measure new branch means; L2 uses retained calibration inputs.
Compensate Lookahead slow connections separately and reset affected moments. This is approximate mean compensation, not pointwise equivalence or a guarantee of accuracy/strength preservation.

`l1-revive.csv` / `l2-revive.csv` record counts, upper/zero hits, U, relative score R, selection and threshold.
CSV R/threshold values are fractions (0..1); stdout uses percentages. Numbered audit files preserve previous records.

## Support and migration

Supports cuda-cpp dense SFNN, L1 none/shared, ordinary/per-layer-QAT training, standalone/grid.
L1 requires non-BN; L2 with BN requires calibrated L2 BN, BN QAT and frozen statistics.
Worker, ordinary NNUE, compact L1, axis/pair, residual count gates and legacy L2/L3 factorizers remain unsupported.
FT revival is not included. Checkpoint/nn.bin formats are unchanged.

`sfnn_l1_revive_zero`, `sfnn_l2_revive_zero` and the old per-layer `revive_threshold` / `revive_zero_threshold` options are removed.
Their presence produces a migration error. Remove them and use 0.01 above; do not carry over 0.99.
Existing checkpoints remain readable; remove obsolete keys from launch settings.
