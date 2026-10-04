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

FT/L1/L2 audits append to one `revive.csv` in the training output directory.
The first columns are `epoch,run,layer`. `run` starts at 1 within each epoch;
FT/L1/L2 share the same number in one epoch-start invocation, and rerunning that epoch uses the next number.
Remaining columns are `bucket,unit,pair,positions,upper_hits,zero_hits,contribution,relative_contribution,selected,contribution_threshold`.
FT uses `pair`; L1/L2 use `unit`. Inapplicable fields are blank.
An audit is written even when no units are selected. `selected` records the decision, not successful completion: the audit is persisted before modifying weights.
Existing rows are never overwritten or deleted. Legacy `ft-revive*.csv`, `l1-revive*.csv`, and `l2-revive*.csv` are retained without automatic migration.
CSV R/threshold values are fractions (0..1); stdout uses percentages.

### Revival randomness and reproducibility

FT/L1/L2 share a persistent revival-only random stream in the runner. A fixed seed initializes
the stream once for a fresh run, never per unit, layer, or epoch. Repeated revival consumes
new draws; an empty selection consumes none. Changing layer order or selected targets changes
subsequent draw assignments. Full training checkpoints (`state.bin`, etc.) and in-memory
snapshots preserve the state losslessly; resume continues from that saved state. Identical
saved state and draw order reproduce the same random values. `nn.bin` does not store this state.
Legacy checkpoints without the record start at the fixed seed and save the state thereafter.
Restarting from an older checkpoint rolls back the stream together with the weights.
Glorot bounds, bias compensation and signed 1/64 outgoing connections are unchanged; scratch
initialization options such as `sfnn_init_l1_glorot` do not control revival initialization.
Quantization-visible outgoing connections do not guarantee nonzero gradients in every case.

### Per-epoch summary

After all enabled layers successfully finish revival at epoch start, append one row to
`revive-summary.csv` in the training output directory. `epoch,run` matches the detailed audit;
reruns never remove earlier rows. No row is emitted when all layers are disabled.
Columns for `ft_`, `l1_`, then `l2_` have these suffixes:

| Suffix | Meaning |
| --- | --- |
| `eligible` | Eligible count, excluding insufficiently sampled buckets |
| `revived` | Count successfully reset, including mean compensation |
| `revive_rate` | `revived / eligible` |
| `mean_relative_contribution` | Mean pre-reset relative contribution of eligible targets |
| `contribution_threshold` | Applied threshold |

FT counts unique shared pairs. Its mean is the mean across pairs of each pair's maximum
relative contribution over buckets. FT has zero eligible pairs unless every bucket has
at least 1,024 positions. L1/L2 count bucket-specific units in sufficiently sampled buckets;
their mean is an unweighted unit average. L1 combines its two branches into one unit.
L1 is measured after FT revival, L2 after L1 revival. Disabled/unmeasured layers are blank.
With zero eligible targets, counts are zero and rate/mean are blank. Fractions and thresholds
use ten decimal places. Relative contribution is not a fraction of playing strength.
Uses existing calibration results without extra inference. Failed revival emits no completion row.
Summary write errors produce a yellow WARNING and do not stop training.
Old detailed audits are not backfilled because `selected` does not prove successful completion.

## Support and migration

Supports cuda-cpp dense SFNN, L1 none/shared/axis, ordinary/per-layer-QAT training, standalone/grid.
L1 requires non-BN; L2 with BN requires calibrated L2 BN, BN QAT and frozen statistics.
Worker, ordinary NNUE, compact L1, pair, residual count gates and legacy L2/L3 factorizers remain unsupported.
Axis revival retains shared/axis tensors and their optimizer states, subtracting their contribution
from selected bucket-specific weights/biases. Master and Lookahead are compensated separately,
including alpha and axis confidence. Unselected buckets are not reset through a common factor.
L2 revival leaves upstream L1 axis weights untouched. Epoch axis-to-shared switching remains supported.
FT revival is described below. Checkpoint/nn.bin formats are unchanged.

`sfnn_l1_revive_zero`, `sfnn_l2_revive_zero` and the old per-layer `revive_threshold` / `revive_zero_threshold` options are removed.
Their presence produces a migration error. Remove them and use 0.01 above; do not carry over 0.99.
Existing checkpoints remain readable; remove obsolete keys from launch settings.

## Shared FT product-pair revival

```json
"sfnn_ft_revive": {"epoch5": true, "epoch6": false, "epoch9": true, "epoch13": false},
"sfnn_ft_revive_contribution_threshold": 0.01
```

Runs at the start of epoch5 and epochs9–12. Use epoch10:false for epoch9 only.
Disabled by default and before the first schedule entry. The threshold defaults to 0.01,
must be finite in (0,1], and also supports an epoch schedule (epoch1 required).
Mid-epoch resume does not run it; the next enabled epoch boundary does.

Measure product pairs i × (i+FT width/2), not individual FT activations.
For each bucket, sum both perspectives' MAD(product) × sum(abs(outgoing L1 weights)),
including L1 skip. Normalize by that bucket's maximum pair utility (all zero => relative zero).
Select only pairs strictly below the threshold in EVERY bucket, not a frequency-weighted average.
If ANY bucket has fewer than 1,024 usable positions, warn and skip all FT revival.

Replay the same 16 teacher batches twice for exact empirical mean/MAD with the quantized GPU proxy;
ignore zero sample weights, do not use validation data, and do not advance the training cursor.
Only bucket×width aggregates are kept on CPU, not all FT activations.
Proxy/workspace VRAM and CPU weight/optimizer readback are still required, as in existing revival.

Reinitialize both selected FT columns: real feature weights are uniform within
max(sqrt(6/(real input count+FT width)),1/127), rounded to the 1/127 grid;
bias is 0.5 and selected virtual factorizer columns are zeroed. This is a revival-specific
initialization, not ordinary scratch initialization. Both perspectives' outgoing L1 connections
become signed 1/64; compensate L1 shared weights through the residual columns.
Transfer the measured old mean to L1 bias, replay the same teacher sample a third time,
and subtract the new mean contribution. Reset affected moments and compensate Lookahead separately.
This is approximate mean compensation, NOT pointwise-equivalent or guaranteed to preserve strength.

Order: FT, then L1, then L2. Audit: the shared `revive.csv` in the training output directory (`layer=FT`), fractions in CSV,
percentages on stdout. Counts may differ from offline validation-set integer nn.bin analysis.
Supports non-BN dense SFNN, L1 none/shared/axis, FT factorizer on/off, per-layer QAT, standalone/grid.
FT outgoing residuals compensate both shared and axis terms so effective new connections are signed 1/64.
BN, compact L1, pair, residual count gates and worker trials are unsupported.

```powershell
python .\grid_search.py --settings-file settings.json --output-folder results `
  --grid sfnn_ft_revive true --grid sfnn_ft_revive_contribution_threshold 0.005 0.01
```
