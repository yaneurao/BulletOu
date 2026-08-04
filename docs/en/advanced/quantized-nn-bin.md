# Quantized `nn.bin` checks

<a href="../../ja/advanced/quantized-nn-bin.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Training-time validation normally uses the f32 weights in memory.
The engine, however, plays with the exported quantized `nn.bin`.

This page covers two commands for inspecting an exported `nn.bin` directly.

| Command | Use |
| --- | --- |
| `quantized-test` | Measure quantized accuracy / loss |
| `calibrate-nn-bin` | Inspect output scale and fold an offset into the final bias |

## Test quantized accuracy / loss

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

Use `calibrate-nn-bin` to run quantized forward on a validation set and inspect two values:

| Item | Meaning |
| --- | --- |
| `estimated_fv_scale` | Estimated `FV_SCALE` from a linear fit between raw output and teacher score |
| `selected_offset` | Score offset that gives the lowest validation loss under the supplied `--fv-scale` |

Example:

```powershell
.\target\release\examples\bulletou.exe calibrate-nn-bin `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --output checkpoints\...\0002\nn2.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv `
  --fv-scale 28
```

Example output:

```text
estimated_fv_scale= 27.832  score ~= raw/27.832 -12.345
scale_fit         = samples 921,060  rmse 620.123  r2 0.41234  current_fv_offset -9.876
selected_offset   = -10 Value
folded_raw_delta  = -280 l3b
before            = acc 62.7604%  loss_engine 0.12345678
after             = acc 62.8012%  loss_engine 0.12298765
```

`estimated_fv_scale` comes from the least-squares fit:

```text
teacher_score ~= raw / FV_SCALE + offset
```

It is a practical estimate of the `FV_SCALE` that matches this `nn.bin` to the teacher score scale.

`selected_offset` is the loss-reducing score offset under the `--fv-scale` you supplied. The command writes that offset into the output `nn.bin` by adding `selected_offset * FV_SCALE` to every final LayerStack bias.

The command does not write `FV_SCALE` itself into the `nn.bin`. When using the exported file in YaneuraOu, set the engine option `FV_SCALE` using the displayed `estimated_fv_scale` as a guide.

Previous: [Advanced guide](README.md)
