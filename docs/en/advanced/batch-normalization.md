# Experimental NNUE / SFNN batch normalization

## Ordinary NNUE

The CUDA C++ trainer supports BN for ordinary HalfKP, KP, KA2, HalfKPE9 and HalfKP_vm NNUE.
BN precedes each selected CReLU. FT perspectives remain **concatenated**, not SFNN product pooling.

| JSON key | Default | Meaning |
|---|---:|---|
| `nnue_bn_ft` | `false` | FT before CReLU; both perspectives share statistics and affine parameters |
| `nnue_bn_l1` | `false` | First dense hidden layer before CReLU |
| `nnue_bn_l2` | `false` | Second dense hidden layer before CReLU |
| `nnue_bn_gamma` | `0.25` | Initial trainable gamma |
| `nnue_bn_beta` | `0.5` | Initial trainable beta |
| `nnue_bn_momentum` | `0.1` | New-batch weight in running-statistic EMA, `(0,1]` |
| `nnue_bn_epsilon` | `0.00001` | Positive variance stabilizer |

Hidden layers use per-unit statistics without buckets. Equations, statistics and gamma/beta updates follow
the SFNN description below. Existing ordinary-NNUE training supports only `batches_per_update=1`.
Ordinary NNUE BN supports standalone/grid search, not plateau, worker mode, epoch-dependent BN settings or BN QAT.
Do not use `sfnn_bn_*` for ordinary NNUE.

Grid example (use an ordinary NNUE architecture in `settings.json`):

```powershell
python .\grid_search.py `
  --settings-file .\settings.json `
  --output-folder D:\BulletOu-snapshots\grid-nnue-bn `
  --grid nnue-bn-ft false true `
  --grid nnue-bn-l1 true `
  --grid nnue-bn-l2 true
```

Validation folds running statistics into weights. Export folds the same statistics before conventional
`nn.bin` quantization; no engine BN support or extra file is needed. Quantization error and clipping still apply.
This does not add/change the existing ordinary-NNUE validation metrics.
`state.bin` / full-state `weights.bin` retain unfused weights, BN statistics, gamma/beta and their optimizer state.
Resume with matching BN options/configuration. Disabling saved BN fails explicitly rather than discarding it.

## SFNN

Independent BN switches are available after the FT, L1, and L2 affine operations,
**before** their activations. All default to off. Unlike optimizer-coordinate
centering, BN changes the training forward pass.

| JSON key (CLI uses hyphens) | Default | Meaning |
|---|---:|---|
| `sfnn_bn_ft` | `false` | Before FT clamp and product pooling; both views share parameters/statistics |
| `sfnn_bn_l1` | `false` | Before normal/squared branches; excludes the skip output |
| `sfnn_bn_l2` | `false` | Before L2 clamp |
| `sfnn_bn_gamma` | `0.25` | Initial trainable gamma for enabled layers |
| `sfnn_bn_beta` | `0.5` | Initial trainable beta for enabled layers |
| `sfnn_bn_momentum` | `0.1` | New-batch EMA coefficient, in `(0,1]` |
| `sfnn_bn_epsilon` | `0.00001` | Positive variance stabilizer |

Gamma=0.25/beta=0.5 is an experimental initialization suited to the `[0,1]`
clamp range, not an established optimum. Gamma=1/beta=0 is also supported.
These options do not support epoch-dependent schedules.

## Semantics

\[
y=\gamma(z-\mu_B)/\sqrt{v_B+\varepsilon}+\beta.
\]

FT combines the two perspectives into one population per unit. L1/L2 have
independent statistics per bucket/unit. Statistics include every batch record,
without task-loss entry weighting. Training uses biased batch variance;
the running estimate uses unbiased variance. The first group observation with
at least two samples initializes running statistics; subsequent observations
use EMA. Empty groups leave statistics untouched. Singleton groups use running
statistics without updating them. Uninitialized running mean/variance are 0/1.

Backward differentiates the batch mean and variance; it is not STE. Gamma/beta
use the same Ranger update timing/LR as the model, with no weight decay or
weight clipping. Gradient accumulation includes gamma/beta, but BN statistics
are computed **per mini-batch**, not over the entire accumulation window.

## Inference and persistence

Validation uses running statistics. Quantized validation and nn.bin export fold
the same inference affine before quantization:

\[
r=\gamma/\sqrt{v_{running}+\varepsilon},\quad W'=rW,\quad
b'=r(b-\mu_{running})+\beta.
\]

L1 shared contributions are combined before bucket-wise folding. No nn.bin
format or engine inference change is required. Folding can exceed quantization
bounds; compare float/quantized metrics rather than assuming equivalence after
rounding/clipping.

state.bin retains unfused weights plus gamma/beta, running statistics, and
their optimizer states. Resume requires the same BN configuration; silently
discarding checkpoint BN is rejected. Adding BN to a non-BN checkpoint changes
the function; it is not a function-preserving conversion.

## Scope and cost

Supported: cuda-cpp standalone training/grid search, dense SFNN, none/shared
factorizers, gradient accumulation, and optional existing centering.
Not supported yet: worker, plateau, compact/grouped L1, count gates, layer
freezing/individual LR multipliers, L1-only QAT, L1 effective weight clipping, or FT/weight
saturation penalties. If L1-only QAT is requested without `sfnn_bn_qat`, a yellow WARNING is printed
and training continues with L1-only QAT disabled (effective `sfnn_qat_l1=false`), leaving
BN enabled and settings files unchanged. Other unsupported combinations still fail.
Each BN layer is limited to 65536 bucket/unit channels; FT has one group.
Validation batches must not exceed the training batch size.
`average-sfnn-state` and `compare-sfnn-quantization` reject BN state.bin files
instead of silently ignoring BN. Training validation/qvalid and nn.bin export
are supported.

FT width1024/batch65536 requires approximately 512 MiB extra VRAM just for
normalized activations across both views. BN adds GPU reductions/backward.
GPU qvalid folds BN and quantizes on-device without host weight readback/upload;
CPU-exact validation still runs on the CPU. With BN enabled, quantization uses
f64 scaling before rounding, matching the CPU nn.bin exporter at rounding boundaries.
Training reductions use 1024-row
chunks and coalesced 32-unit tiles for widths divisible by 32 (8 otherwise), preserving double-precision accumulation,
two-pass variance and EMA definitions. Inference applies running statistics
directly, without batch reductions. Reduction scratch adds about 1.6 MiB for
FT1024/L1=8/L2=64, 8 buckets, batch65536 (separate from saved activations).
FT training fuses BN application, clamping and pairwise multiplication to reduce
intermediate memory traffic, without additional VRAM. BN-disabled runs do not use these buffers. Reduction ordering can cause small
rounding differences. Playing-strength improvements have not been established.

Training also fuses L1 BN application with its normal/squared branches, and L2 BN
application with clamping. FT backward fuses BN gradient application with bias
gradient reduction and skips adding zero to absent feature gradients. Small-bucket
dense L1 uses a coalesced parameter-gradient reduction; shared L1 with 8/9 outputs
also specializes the input gradient. Other shapes retain the existing paths.
These optimizations need no extra VRAM and do not change BN definitions, statistics
update frequency, or gradient accumulation. Floating-point reduction order can
change training trajectories; bitwise-identical training is not guaranteed.

## Grid search

Use a new output directory. Explicitly disable unsupported options for **all**
conditions, including the non-BN baseline:

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260923-progress8-bceloss.json `
  --output-folder D:\BulletOu-snapshots\20260925\grid-progress8-bn `
  --grid sfnn-bn-ft false true `
  --grid sfnn-bn-l1 false true `
  --grid sfnn-bn-l2 false `
  --grid sfnn-qat-l1 false `
  --grid sfnn-l1-effective-weight-clip false `
  --grid sfnn-ft-saturation-penalty 0 `
  --grid sfnn-saturation-penalty 0
```

This runs four combinations. Change the L2 axis to `false true` for eight.
Other settings are inherited from the JSON. BN conditions appear in
grid_summary.csv. Rebuild BulletOu before using BN; do not replace a running
training executable.

## BN-aware QAT from scratch or fine tuning

`--sfnn-bn-qat` / JSON `"sfnn_bn_qat": true` defaults to **off**.
**Training from scratch is supported; no pre-trained BN checkpoint is required.**
`sfnn_bn_qat_freeze_stats` defaults to `false`: BN statistics update every mini-batch.

### Default: train-mode BN

For an effective weight \(W\) (including shared/factorizer contributions), use export
rounding/clipping \(Q\) with the pre-batch running scale:

\[
r=\gamma/\sqrt{v_{running}+\varepsilon},\qquad
\widetilde W=Q(rW)/r,\qquad y=\mathrm{BN}_{batch}(\widetilde W x+b).
\]

The calibration scale is detached: do not differentiate through it. Use identity STE
for weight rounding/clipping, and full BN backward through batch mean/variance and gamma/beta.
Uninitialized channels bypass weight fake quantization on their first observed batch to initialize
statistics; subsequent batches enable it. Zero-gamma channels also use raw weights to avoid division
by zero and allow gamma to recover. There is no fixed-superbatch warmup requirement.

Pre-BN biases stay raw in training: batch centering cancels them, and unscaling rounded folded biases
by tiny gamma would destroy numerical precision. Non-BN layers (including L3) fake-quantize both
weights and biases. Running statistics track the fake-quantized affine inputs. With BPU, each
mini-batch updates statistics and rebuilds the proxy; gradients accumulate. Activations remain float,
not bit-exact integer inference. Validation/export use master weights and running statistics;
export quantizes biases as usual.

### Explicit frozen-statistics mode (previous fine-tuning behavior)

Add `"sfnn_bn_qat_freeze_stats": true` / `--sfnn-bn-qat-freeze-stats` to use the following mode.
Only this mode requires a BN-trained `state.bin` or full-state `weights.bin`; each enabled layer
must have at least one initialized channel. Unseen buckets retain saved initial statistics.

This mode **freezes running means/variances** and trains raw FP32 weights,
biases and gamma/beta. Fold FT factorization, L1 shared weights and BN, then
fake-quantize FT/L1/L2/L3 weights and biases using the export scales, rounding
and clipping. Layers without BN are quantized too. The original FP32 master
parameters and optimizer state are preserved. Activations still use the float
training path: this is weight/bias QAT, not a bit-exact integer inference simulator.

Use identity STE through both rounding and clipping, including out-of-range
weights. For fixed statistics, with \(r=\gamma/\sqrt{v+\varepsilon}\),
\(W'=rW\), \(b'=r(b-\mu)+\beta\), and folded-coordinate gradients \(G_W,G_b\):

\[
\partial_W L=rG_W,\quad \partial_b L=rG_b,\quad
\partial_\beta L=G_b,\quad
\partial_\gamma L=\frac{\langle G_W,W\rangle+G_b(b-\mu)}{\sqrt{v+\varepsilon}}.
\]

Apply the factorizer/shared chain rule as well. No division by gamma is needed;
zero/negative gamma are supported. BPU accumulates folded gradients and pulls
them back once per optimizer update. Float validation keeps its original meaning;
quantized validation remains separate.

Keep the saved BN flags enabled. BN QAT replaces `sfnn_qat_l1` if both are set.
Centering, effective clipping and saturation penalties are unsupported with BN QAT;
worker mode remains unsupported. Enable/disable it explicitly on resume, not through
an epoch schedule. Settings files are never rewritten automatically.

For a scratch comparison, use common settings with BN enabled and no `initial_state`, and a new grid root:

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bn-training.json `
  --output-folder D:\BulletOu-snapshots\grid-bn-qat `
  --grid sfnn-bn-qat false true
```

By default both ON and OFF update statistics. Explicit frozen mode additionally changes
statistics behavior, so that comparison does not isolate quantization alone. Extra proxy weights require about 522 MiB
for HalfKA2 FT1024 (printed at startup). Checkpoint/nn.bin formats are unchanged;
no engine changes are required. Accuracy/playing-strength improvement is not guaranteed.

BN QAT fuses FT quantization and conversion back to training coordinates, avoiding
duplicate full-FT memory passes and identity gradient pullbacks in updating-statistics mode.
Quantization and BN statistics still update every mini-batch. Rounding is unchanged.
These optimizations require no additional option.
