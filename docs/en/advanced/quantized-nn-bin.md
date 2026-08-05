# Quantized `nn.bin` checks

<a href="../../ja/advanced/quantized-nn-bin.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

Training-time validation normally uses the f32 weights in memory. The engine, however, plays with the exported quantized `nn.bin`.

This page covers two commands for inspecting an exported `nn.bin` directly.

| Command | Use |
|---|---|
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

`--fv-scale auto` searches integer `FV_SCALE` values in `16..=40` by default. Use `--fv-scale-min`, `--fv-scale-max`, and `--fv-scale-step` to change that range.

If you pass an integer such as `--fv-scale 28`, BulletOu keeps that `FV_SCALE` fixed and searches only the offset.

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
