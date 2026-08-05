# `--scale` and `--fv-scale`

<a href="../../ja/advanced/scale-and-fv-scale.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

This page explains the meaning of BulletOu's `--scale` and `--fv-scale`.

For NNUE/SFNN training, these two values should be treated as separate knobs.

| Option | Role | Default |
| --- | --- | --- |
| `--scale` | Converts teacher eval scores back into win-rate labels | `600` |
| `--fv-scale` | Assumed YaneuraOu `FV_SCALE` for the quantized NNUE/SFNN output range | `40` |

For example, if teacher scores were produced by rshogi `rescore_psv` with `scale=600`, BulletOu should also use `--scale 600`. If the exported `nn.bin` is intended to run in YaneuraOu with `FV_SCALE=40`, BulletOu should also train with `--fv-scale 40`. Both are the defaults, so you usually do not need to specify them.

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --superbatches 324 `
  --max-epochs 28 `
  --tag sfnn-example
```

This is equivalent to explicitly adding:

```powershell
  --scale 600 `
  --fv-scale 40
```

## 1. Teacher scores and win rates

Teacher eval scores can be interpreted as win-rate logits.

```text
winrate = sigmoid(score / scale)
```

`sigmoid(x)` means:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

With `scale=600`, the mapping is approximately:

| Teacher score | `score / 600` | Win-rate label |
| ---: | ---: | ---: |
| `-1200` | `-2.0` | `11.9%` |
| `-600` | `-1.0` | `26.9%` |
| `0` | `0.0` | `50.0%` |
| `+600` | `+1.0` | `73.1%` |
| `+1200` | `+2.0` | `88.1%` |

So if rshogi created teacher eval scores using `scale=600`, the natural way to turn those scores back into win-rate labels is also `scale=600`.

## 2. Where `203.2` comes from when `FV_SCALE=40`

NNUE/SFNN networks are shallow, so it can be useful for the network output range to be wider. If the exported network will be used by YaneuraOu with `FV_SCALE=40`, the saved `nn.bin` should satisfy:

```text
engine_score ≒ teacher_score
```

When BulletOu writes an NNUE/SFNN `nn.bin`, the f32 network output is approximately multiplied by `QA * QB` during quantization.

```text
QA = 127
QB = 64
QA * QB = 8128
```

If `raw` is the integer NNUE output inside YaneuraOu:

```text
raw ≒ network_output * 8128
```

YaneuraOu then divides by `FV_SCALE` to get the engine eval score.

```text
engine_score = raw / FV_SCALE
```

Therefore the f32 training-time network output corresponds to the YaneuraOu eval score as:

```text
engine_score ≒ network_output * 8128 / FV_SCALE
```

With `FV_SCALE=40`:

```text
engine_score ≒ network_output * 8128 / 40
             ≒ network_output * 203.2
```

The `203.2` here comes from:

```text
8128 / 40 = 203.2
```

It is the coefficient that maps `network_output` back to the YaneuraOu eval score. It is not the same thing as `--scale`, which maps teacher scores back to win-rate labels.

So to produce an engine eval score of `+600`, the network output should be:

```text
network_output ≒ 600 / 203.2
               ≒ 2.95
```

## 3. Why `--scale 203` is not the answer

After seeing `203.2`, it is tempting to train with `--scale 203`. That is not what we want. `--scale` controls the teacher-score to win-rate conversion, so putting `203` there changes the target labels.

If the teacher scores were created with `scale=600`, a teacher score of `+600` means:

```text
sigmoid(600 / 600) = sigmoid(1.0) = 0.731
```

But if BulletOu reads it with `--scale 203`, it becomes:

```text
sigmoid(600 / 203) = sigmoid(2.956) = 0.950
```

The teacher intended `+600` to mean about `73.1%`, but `--scale 203` reads it as about `95.0%`. That is what it means to distort the teacher win-rate labels.

This is why BulletOu keeps the two settings separate.

```text
--scale 600    # convert teacher scores back to win rates
--fv-scale 40  # train the network output range for FV_SCALE=40
```

## 4. BulletOu's loss formula

BulletOu aligns the teacher side and prediction side like this:

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid((network_output * 8128 / fv_scale) / scale)
loss       = |prediction - target|^p
```

`p` is controlled by `--loss-pow-exp`. The default is `2.0`, which is squared error in sigmoid space.

The formula means:

1. `teacher_score / scale` converts the teacher eval score back to a win-rate logit.
2. `network_output * 8128 / fv_scale` converts the network output to the eval score that YaneuraOu will see.
3. Dividing that score by `scale` puts the prediction into the same win-rate space as the target.
4. The loss is the difference between the two win rates.

For `--scale 600 --fv-scale 40`, consider a teacher score of `+600`.

```text
target = sigmoid(600 / 600)

prediction = sigmoid((network_output * 8128 / 40) / 600)
```

The loss is minimized when the values inside the two sigmoids match.

```text
(network_output * 8128 / 40) / 600 = 600 / 600
```

Rearranging:

```text
network_output * 8128 / 40 = 600
network_output = 600 * 40 / 8128
network_output ≒ 2.95
```

After quantization, YaneuraOu sees:

```text
raw ≒ 2.95 * 8128 ≒ 24000
engine_score = raw / 40 ≒ 600
```

So `--scale 600 --fv-scale 40` achieves both goals:

- the teacher win-rate label is read as `scale=600`;
- the NNUE/SFNN network output is widened for `FV_SCALE=40`.

## 5. How to choose the two values

The rule of thumb is simple.

| Goal | Setting |
| --- | --- |
| Teacher data was produced with `scale=600` | `--scale 600` |
| Run the exported network with YaneuraOu `FV_SCALE=40` | `--fv-scale 40` |
| Run the exported network with YaneuraOu `FV_SCALE=32` | `--fv-scale 32` |
| Teacher data was produced with another scale | Use that value for `--scale` |

Use the same `--fv-scale` during training as the `FV_SCALE` you plan to use in YaneuraOu. If the two values differ, the final eval score scale will differ too.

## 6. When using `--lambda`

When `--lambda` is not `1.0`, BulletOu mixes the label from the teacher score with the label from the game result.

```text
eval_label   = sigmoid(teacher_score / scale)
result_label = win ? 1.0 : draw ? 0.5 : 0.0

target = lambda * eval_label + (1 - lambda) * result_label
```

The prediction side is unchanged.

```text
prediction = sigmoid((network_output * 8128 / fv_scale) / scale)
```

For re-scored teacher data, game results are not always a reliable calibration target. Start with the default `--lambda 1.0`, which uses only the teacher eval score.

## 7. Notes

- `QA` and `QB` are quantization constants for `nn.bin` export. They are not normal user-tuned hyperparameters.
- `--fv-scale` is for NNUE/SFNN. KPPT-family targets use `--yaneuraou-quant-scale`.
- `--scale` should match the teacher score's win-rate model. Do not make `--scale` smaller just to widen the network output; that changes the target win rates.
- To change the network output range, tune `--fv-scale`, not `--scale`.
