# L2/L3 optimizer centering

The optional [persistent FT saturation penalty](ft-saturation-penalty.md) can be A/B tested alongside centering. This separate activation penalty is OFF by default.

## Optional folded L1 weight projection

`--sfnn-l1-effective-weight-clip` (JSON: `"sfnn_l1_effective_weight_clip": true`) is **OFF by default**. It is separate from the legacy optimizer clipping disabled by centering.

After each optimizer/Lookahead update and conversion back from centered coordinates, it projects `effective = residual + alpha*shared` into `[-2, 127/64]`, writing `residual = clamp(effective) - alpha*shared`. Shared weights remain unchanged. Both fast and slow weights are projected using their respective shared weights. In-range weights are left untouched; floating-point cancellation can introduce small numerical differences.

Only L1 weights are affected. Biases, FT/L2/L3, momentum/variance states, and identity STE are unchanged; no compensating bias shift is made. Supports cuda-cpp dense L1, none/shared, update-scope=all, no count gates or axis/pair. QAT/centering are optional. With gradient accumulation, projection occurs only on optimizer updates; frozen L1 (zero LR multiplier) is not projected. Loading a checkpoint alone does not project it: the first L1 update does. No checkpoint format changes, large scratch buffers, or host weight readbacks are required.

Epoch boolean schedules and resume toggling are supported. The option is recorded in the startup output and resume signature. A/B test in a new grid root:

```powershell
python .\grid_search.py `
  --settings-file <common-settings.json> `
  --output-folder <new-grid-root> `
  --grid sfnn_l1_effective_weight_clip false true
```

Keep other conditions identical. Judge quantized accuracy/loss, saturation, and playing strength, not merely agreement between FP32 and quantized metrics. Optimizer histories are retained when enabling this on resume.

## A/B testing L1 centering

`--sfnn-l1-center` (JSON: `"sfnn_l1_center": true`, default OFF) applies the same optimizer-coordinate transformation to L1. It is independent of, and can be combined with, L2/L3 centering.

```powershell
python .\grid_search.py `
  --settings-file <common-settings.json> `
  --output-folder <new-comparison-folder> `
  --grid sfnn_l2_l3_center true `
  --grid sfnn_l1_center false true
```

The GPU computes a global mean of the combined FT features feeding L1 across all batches in an update. The same mean applies to bucket-specific and shared L1 weights; it is not a per-bucket mean. BPU>1 and L1 QAT are supported. Dense L1 and factorizer none/shared are required; other restrictions below also apply. This does not center FT itself, normalize outputs, or penalize saturation. A reduction in saturation is not guaranteed.

Additional L1 scratch is approximately 132 KiB at FT width 1024, with no whole-weight CPU transfers. Forward and nn.bin retain the folded form. Epoch schedules and toggling on resume are supported without resetting optimizer state.

GPU mean reduction parallelizes rows as well as columns. L1 residual/shared coordinate transforms share one kernel launch before and after the optimizer update. All positions contribute to the mean; there is no subsampling. Reduction ordering can introduce floating-point roundoff differences, but centering equations, learning rates and optimizer definitions are unchanged.

The centering path avoids redundant mean accumulation, zeroing and copies for BPU=1; for BPU>1, final mean scaling uses a single kernel per layer. Benchmark your workload to determine the actual speedup. Set `optimizer_weight_clip: 0` for both A/B conditions to avoid also changing the clipping policy.

`--sfnn-l2-l3-center` (JSON: `"sfnn_l2_l3_center": true`) enables input-mean-centered optimizer coordinates for L2/L3. It is **off by default** and does not center FT/L1.

## Settings

Use these fields in your existing training JSON, keeping other training conditions unchanged:

```json
{
  "sfnn_l2_l3_center": true,
  "batches_per_update": 1,
  "sfnn_factorizer": "shared",
  "optimizer_weight_clip": 0,
  "optimizer_weight_decay": 0,
  "sfnn_norm_loss_strength": 0,
  "sfnn_saturation_penalty": 0,
  "sfnn_factorizer_residual_decay": 0
}
```

CLI: `--sfnn-l2-l3-center --batches-per-update 1 --optimizer-weight-clip 0`, also satisfying all constraints below. No user settings are automatically rewritten.

With a compatible common JSON, compare using:

```powershell
python .\grid_search.py `
  --settings-file <common-settings.json> `
  --output-folder <comparison-folder> `
  --grid sfnn_l2_l3_center false true
```

Use `--grid sfnn_l2_l3_center true` to run only the enabled condition. Both hyphenated and underscore key names are supported. The flag is recorded in `grid_summary.csv`. Epoch schedules such as `{"epoch1": false, "epoch2": true}` are supported.

## Comparing Glorot initialization with centering

Initialization is independently selectable per layer for cuda-cpp SFNN scratch training. Loaded checkpoint weights are never reinitialized. Use underscore keys in JSON and with `--grid`.

| JSON / grid key | Default | Effect when true |
|---|---|---|
| `sfnn_init_ft_glorot` | false | Glorot-uniform FT base weights |
| `sfnn_init_l1_glorot` | false | Glorot-uniform dense bucket L1 weights |
| `sfnn_init_l2_glorot` | false | Glorot-uniform L2 weights |
| `sfnn_init_l3_glorot` | false | Glorot-uniform L3 weights |
| `sfnn_init_ft_shared_zero` | true | Zero FT shared weights |
| `sfnn_init_l1_shared_zero` | false | Zero L1 shared weights |

`sfnn_init_l2_l3_glorot` was removed. Specify L2 and L3 separately.

```json
{
  "sfnn_init_ft_glorot": true,
  "sfnn_init_l1_glorot": true,
  "sfnn_init_l2_glorot": true,
  "sfnn_init_l3_glorot": true,
  "sfnn_init_ft_shared_zero": true,
  "sfnn_init_l1_shared_zero": true
}
```

Glorot half-width is `sqrt(6 / (fan_in + fan_out)) * nnue_pytorch_init_scale`. FT uses base feature count and FT width; dense L1 includes its skip output, if any; L2 uses denominator `2*H1 + H2`; L3 uses `H2 + 1`. Bucket count is not included. L2/L3 also apply their existing per-layer initialization scales. Bias initialization is unchanged.

With FT shared-zero false, initialize shared rows using the selected FT method and shared-row fan-in, with a separate random seed. Base FT random weights are unchanged. Without FT factorization, shared initialization has no effect. L1 shared-zero false retains uniform ±0.01. FT shared was already zero-initialized, so its default remains true.

Without FT Glorot, base FT retains ±`init_scale / sqrt(feature_count)`. HalfKA2 can therefore round entirely to zero under FT QAT. Non-BN FT QAT warns and stops if every folded, quantized FT weight and bias is zero. It never silently changes initialization or disables QAT. This check runs on activation/configuration change/restore, not as a CPU readback on every batch.

Four-way comparison:

```powershell
python .\grid_search.py `
  --settings-file <scratch-common-settings.json> `
  --output-folder <new-comparison-folder> `
  --grid sfnn_init_ft_glorot false true `
  --grid sfnn_init_ft_shared_zero false true
```

All six keys accept `--grid KEY false true` (64 combinations if all are varied). Remove checkpoint inputs for initialization experiments. Epoch-wise initialization changes are unsupported. This does not reproduce all Conductor conditions: BulletOu uses independent random weights per bucket.

## Centered update equations

Compute each layer's input mean `c` on GPU over all batches/positions in the optimizer update (not a separate mean per bucket). With `batches_per_update=4`, use all four batches, not just the final batch. Before updating, transform `beta = b + W*c` and `gW_center = gW - gb*c` using accumulated gradients. Run the optimizer in these coordinates, then restore `b = beta - W*c`. Lookahead slow weights/biases undergo the same coordinate conversion.

Forward, validation and nn.bin still use `W*x+b`. This is not BatchNorm or variance normalization. Moments are updated using centered gradients, so this is a different optimization algorithm from ordinary Ranger. It adds no penalty to the reported task loss.

GPU mean scratch is allocated on activation and reused (about 10 KiB for 1024/7/64). It does not copy entire weight tensors to the CPU.

## Supported combinations and resume

- cuda-cpp SFNN, `batches_per_update>=1`, `sfnn_update_scope=all`. Keep bpu identical across comparison conditions.
- L1 factorizer none/shared. FT factorization and L1 QAT are supported.
- Weight clipping is disabled while centering is active. Unspecified or positive clipping produces a warning and training continues; explicit zero avoids the warning. The JSON file is not rewritten. Weight decay, Norm loss, saturation penalty and factorizer residual decay must be zero.
- No bucket counts/count gates or axis/pair factors yet.
- L2 input/output widths each <=256; `buckets * L2 width <=65536`.
- Other unsupported combinations still fail explicitly. Epochs with centering off use the configured clipping policy again. To isolate centering in an A/B test, explicitly set `optimizer_weight_clip=0` in the common settings.

Checkpoint and nn.bin formats are unchanged; weights/biases are folded back before saving. You may switch the flag on resume, but optimizer moments are retained, not automatically reset. This changes training conditions and is not equivalent to an A/B comparison from scratch. Specify the desired flag in the resumed CLI/JSON.

Use a normal shell without the older `BULLETOU_EXPERIMENT_*` research environment variables. Saturation and validation improvements were observed in bounded experiments; long-term playing-strength improvement is not guaranteed.
