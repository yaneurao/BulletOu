# Quantized `nn.bin` checks

<a href="../../ja/advanced/quantized-nn-bin.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

Training-time validation normally uses the f32 weights in memory. The engine, however, plays with the exported quantized `nn.bin`.

Values printed during training by `--quantized-validation-rate` use a fast GPU proxy by default: BulletOu rounds weights to the same scale used by `nn.bin`, then evaluates those rounded weights with the f32 forward path. That is useful for watching the trend during training.

If you want exact integer-engine metrics during training, add `--quantized-validation-exact`. This uses the same CPU integer forward path as `quantized-test`, so it is much slower. In practice, use the GPU proxy for frequent monitoring and exact mode only when narrowing down candidates.

This page covers two commands for inspecting an exported `nn.bin` directly. These commands read the quantized `nn.bin` itself, so they are closer to what the engine will use.

| Command | Use |
|---|---|
| `quantized-test` | Measure quantized accuracy / loss |
| `calibrate-nn-bin` | Inspect output scale and fold an offset into the final bias |

## Test quantized accuracy / loss

### Compare L1 QAT on/off

SFNN dense L1 supports optional quantization-aware training with `--sfnn-qat-l1`.
It is **off by default**. In the top level of `bulletou-settings.json`, use:

```json
"sfnn_qat_l1": true
```

`false` or omission preserves ordinary training. The normal, worker and profiling paths support it;
grouped/common-shard L1 is rejected. Only **L1 weights and biases** are fake-quantized, not FT/L2/L3 or activations.
Factorizers and count gates are folded before quantization:

```text
W = gate * residual + alpha_shared * shared + sum(alpha_axis * confidence_axis * axis)
Q(W) = clamp(round(64 * W), -128, 127) / 64
Q(b) = clamp(round(8128 * b), INT32_MIN, INT32_MAX) / 8128
L1 output = Q(W) * input + Q(b)
```

Rounding uses nearest, ties away from zero, sharing the GPU quantized-validation kernels.
Backward input gradients also use `Q(W)`. FP32 master weights use **identity STE**: the derivative of
rounding/clamping is approximated as 1, even outside the clamp range. The factorizer/count-gate chain rule
is retained. Existing optimizer clipping and saturation penalty settings are not changed.

Masters and optimizer state stay FP32 in `state.bin`; the `nn.bin` format is unchanged.
Existing checkpoints can be fine-tuned with `--initial-state` and `--initial-dataloader-pos`, or QAT can
be toggled on `--resume`. Specify the desired QAT setting when restarting; it is logged and saved in settings.

`test_value_accuracy/loss` still evaluates the **unrounded FP32 model**, while `quantized_value_accuracy/loss`
still evaluates **all-layer quantization**. The loss target, scale, LR and metric definitions do not change.
Compare separate tags from identical weights, optimizer state and dataloader position, holding LR, data,
bpu and training length fixed. Check absolute quantized accuracy and engine strength, not just a smaller FP32/Q gap.

QAT uses GPU-only scratch cached until an update, restore or factorizer change.
Extra VRAM is `4 * stacks * L1_outputs * (FT_width + 1)` bytes: about 0.25 MiB for `1024_8_64_progress8`,
but proportional to the number of buckets for larger architectures. OFF allocates no QAT scratch.
Training overhead and accuracy improvements are not yet established by a full training benchmark.

### Evaluate a saved nn.bin

```powershell
.\target\release\examples\bulletou.exe quantized-test `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv
```

If `--test-positions` is omitted, BulletOu tests every position in the validation file. If it is set, choose the sampling mode with `--test-sample sequential` / `random` and `--test-seed`.

The reported `accuracy` is draw-excluded W/L sign agreement, matching YaneuraOu's `test eval_accuracy` command.

## Check output scale and offset

Different `nn.bin` files can have different final integer raw-output scales. YaneuraOu converts the final NNUE integer to an engine score as:

```text
engine_score = raw / FV_SCALE
```

So the same `FV_SCALE` can produce a different score range for different exported networks.

Use `calibrate-nn-bin` to run quantized forward on a validation set and choose `FV_SCALE` plus an offset.

| Item | Meaning |
|---|---|
| `estimated_fv_scale` | Diagnostic linear-fit scale between raw output and teacher score |
| `selected_fv_scale` | `FV_SCALE` with the lowest validation loss |
| `selected_offset` | Score offset with the lowest validation loss under the selected `FV_SCALE` |

Example:

```powershell
.\target\release\examples\bulletou.exe calibrate-nn-bin `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --output checkpoints\...\0002\nn2.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv `
  --fv-scale auto
```

If `--fv-scale` is omitted, BulletOu measures with `FV_SCALE=24`. This is the initial candidate for quantized validation. In the default WRM loss, `FV_SCALE` is not part of the training loss formula.

`--fv-scale auto` searches integer `FV_SCALE` values in `16..=40` by default. Use `--fv-scale-min`, `--fv-scale-max`, and `--fv-scale-step` to change that range.

If you pass an integer such as `--fv-scale 24`, BulletOu keeps that `FV_SCALE` fixed and searches only the offset.

Use `--objective` to choose how the offset is selected.

| Setting | Meaning |
|---|---|
| `--objective loss` | Choose the offset with the lowest validation loss. Default |
| `--objective accuracy` | Choose the offset with the highest sign-agreement accuracy |

For engine-strength testing, it is useful to create both the `loss` version and the `accuracy` version and compare them by games. The offset is only one global parameter shared by all LayerStacks, so this is cheap to test without retraining.

Example output:

```text
searched_fv_scales= 25
searched_offsets  = 257
searched_candidates= 6,425
selected_fv_scale = 16
estimated_fv_scale= 2.390  score ~= raw/2.390 +200.311
scale_fit         = samples 921,060  rmse 2271.179  r2 0.27811  current_fv_offset +27.783
selected_offset   = +26 Value
folded_raw_delta  = +416 l3b
before            = acc 63.2031%  loss_engine 0.07208891
after             = acc 62.8638%  loss_engine 0.07186714
```

`estimated_fv_scale` comes from the least-squares fit:

```text
teacher_score ~= raw / FV_SCALE + offset
```

It is a diagnostic value, not necessarily the loss-minimizing `FV_SCALE`. Use `selected_fv_scale` for the actual selected candidate.

`selected_offset` is the loss-reducing score offset under `selected_fv_scale`. The command writes that offset into the output `nn.bin` by adding `selected_offset * selected_fv_scale` to every final LayerStack bias.

The command does not write `FV_SCALE` itself into the `nn.bin`. When using the exported file in YaneuraOu, set the engine option `FV_SCALE` to the displayed `selected_fv_scale`.

Previous: [Advanced guide](README.md)
