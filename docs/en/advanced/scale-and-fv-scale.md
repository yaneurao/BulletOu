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

So if `FV_SCALE=40`:

```text
engine_score ≒ network_output * 8128 / 40
             ≒ network_output * 203.2
```

The `203.2` value is the output scale of the quantized `nn.bin` inside YaneuraOu. It is separate from the WRM training setting `--wrm-nnue2score 600`.

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
  --fv-scale 40
```

## 6. Checking the quantized network

Training `test_value_loss` is measured with f32 weights. The `nn.bin` used by YaneuraOu is quantized.

To measure quantized accuracy/loss or inspect the best `FV_SCALE`, use [Quantized `nn.bin` checks](quantized-nn-bin.md).

## 7. Matching a teacher score scale to an existing `nn.bin`

When you mix multiple teacher datasets, make sure their score magnitudes mean the same thing.

For example, two PSV datasets may both be produced by DL-based re-scoring, but if the DL win rate was converted back to eval scores with different coefficients, the same win rate will produce different `score` values. If you train on them as-is, “+100” and “+500” no longer mean the same thing across datasets.

`fit-teacher-scale` samples positions from a teacher PSV file, evaluates the same positions with a reference `nn.bin`, and estimates a multiplier `a` for the teacher scores.

```text
reference_score ≒ a * teacher_score
```

The multiplier is fitted with least squares through the origin:

```text
a = Σ(teacher_score * reference_score) / Σ(teacher_score^2)
```

Here `reference_score` is the quantized raw output of the reference `nn.bin`, converted back to the training score scale:

```text
reference_score = raw / 8128 * wrm_nnue2score
```

The default `wrm_nnue2score` is `600`. If the reference network was trained with a different `--wrm-nnue2score`, pass the same value to `fit-teacher-scale`.

Example:

```powershell
.\target\release\examples\bulletou.exe fit-teacher-scale `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --teacher C:\shogi\teacher\tayayan\good-testpsv20260717.psv `
  --nn-bin C:\path\to\sojo-trained\nn.bin `
  --sample-positions 100000
```

To inspect individual sampled rows, add an option such as `--dump-samples 10`.

Example output:

```text
scale_multiplier = 0.277969165
formula          = rescaled_score = round(teacher_score * 0.277969165)
```

This means: multiply the PSV scores by `0.277969165` to bring them closer to the training score scale of the reference `nn.bin`.

Use `rescale-psv` to write a converted PSV file:

```powershell
.\target\release\examples\bulletou.exe rescale-psv `
  --input C:\shogi\teacher\tayayan\good-testpsv20260717.psv `
  --output D:\teacher\tayayan-rescaled.psv `
  --scale-multiplier 0.277969165
```

`rescale-psv` changes only the PSV score field. The position, side to move, move, and game-result fields are preserved.

By default, scores with `|score| >= 32000` are treated as mate-like special values and are copied without scaling. To scale every score, use:

```powershell
  --preserve-score-abs 0
```
