# Loss scale and `FV_SCALE`

<a href="../../ja/advanced/scale-and-fv-scale.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

This page explains the score scale used by the training loss and the `FV_SCALE` used by YaneuraOu.

The important point is: in the default training loss, `FV_SCALE` is not part of the loss formula. `FV_SCALE` is used when checking or using the quantized `nn.bin`.

## 1. Default loss

BulletOu's default loss is the tatara-style WRM loss.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--wrm-nnue2score` | `600` | Converts `network_output` into a score-scale value |
| `--wrm-in-offset` | `270` | Prediction-side WRM offset |
| `--wrm-in-scaling` | `340` | Prediction-side WRM scaling |
| `--wrm-target-offset` | `270` | Teacher-side WRM offset |
| `--wrm-target-scaling` | `380` | Teacher-side WRM scaling |
| `--loss-pow-exp` | `2.0` | Exponent `p` in `|prediction - target|^p` |

The WRM function is:

```text
wrm(score; offset, scaling)
  = 0.5 * (1
           + sigmoid(( score - offset) / scaling)
           - sigmoid((-score - offset) / scaling))
```

where:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

The default loss is:

```text
score_net  = network_output * wrm_nnue2score

prediction = wrm(score_net;
                 wrm_in_offset,
                 wrm_in_scaling)

target     = wrm(teacher_score;
                 wrm_target_offset,
                 wrm_target_scaling)

loss       = |prediction - target|^loss_pow_exp
```

`teacher_score` is the eval score stored in the teacher data. With `--lambda 1.0`, game-result labels are not used in the training target.

## 2. Comparing zero-offset WRM

To test whether the WRM offset helps, set only the offsets to zero:

```powershell
--wrm-in-offset 0 `
--wrm-target-offset 0
```

The scaling values stay unchanged:

```text
prediction = wrm(network_output * 600; 0, 340)
target     = wrm(teacher_score;        0, 380)
```

Use a different `--tag` for comparison runs so the checkpoint directories do not overlap.

## 3. When `--scale` is used

`--scale` is used when you explicitly choose the plain sigmoid loss:

```powershell
--loss-sigmoid-mse `
--scale 600
```

That loss is:

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid((network_output * 8128 / fv_scale) / scale)
loss       = |prediction - target|^loss_pow_exp
```

In the plain sigmoid loss, `--fv-scale` affects the prediction-side output range. In the default WRM loss, `--fv-scale` is not part of the loss formula.

## 4. What `FV_SCALE` does

YaneuraOu converts the final integer NNUE output `raw` into an eval score as:

```text
engine_score = raw / FV_SCALE
```

For NNUE/SFNN `nn.bin` export, the approximate relation is:

```text
raw ≒ network_output * 8128
```

So if `FV_SCALE=24`:

```text
engine_score ≒ network_output * 8128 / 24
             ≒ network_output * 338.7
```

The `338.7` value is the output scale of the quantized `nn.bin` inside YaneuraOu. It is separate from the WRM training setting `--wrm-nnue2score 600`.

## 5. What to use

Start with the default WRM settings:

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --superbatches 324 `
  --max-epochs 28 `
  --tag sfnn-wrm-default
```

For the zero-offset WRM comparison, add:

```powershell
  --wrm-in-offset 0 `
  --wrm-target-offset 0
```

For the plain sigmoid loss, add:

```powershell
  --loss-sigmoid-mse `
  --scale 600 `
  --fv-scale 24
```

## 6. Checking the quantized network

Training `test_value_loss` is measured with f32 weights. The `nn.bin` used by YaneuraOu is quantized.

To measure quantized accuracy/loss or inspect the best `FV_SCALE`, use [Quantized `nn.bin` checks](quantized-nn-bin.md).
