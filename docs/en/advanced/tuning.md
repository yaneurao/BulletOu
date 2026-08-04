# Adjust training settings

<a href="../../ja/advanced/tuning.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-2563EB?style=flat-square"></a>

Read this after the command in [Tutorial §3 Run the training](../tutorial/3-train.md) works. For a first run, keep the defaults. Come back here when you want to change speed, save frequency, validation frequency, learning-rate schedule, loss, or SFNN factorizer settings.

## 1. Units used in the logs

BulletOu logs use `batch`, `superbatch`, and `epoch`. Keep these separate; many training-control flags are defined in terms of one of them.

| Name | Meaning |
|---|---|
| batch | Positions used for one GPU weight update. Default: `--batch-size 65536` |
| superbatch / sb | Progress, validation, and save unit. Its size is controlled by `--positions-per-superbatch` |
| epoch | A group of `--superbatches` superbatches. Learning rate returns to `--lr` at the epoch start |
| checkpoint | Saved files for resuming (`state.bin`) and for the engine (`nn.bin`) |
| validation | Accuracy/loss measured on `--test-teacher`, which should be separate from the training teacher |

Example:

```text
--batch-size 65536
--positions-per-superbatch 40000000
--superbatches 36
```

Here, one sb is `65536 × 610 = 39,976,960` positions. One epoch is 36 sb, or about 1.44 billion positions.

When `--superbatches` is set, an epoch is not “one pass over the teacher.” It is the boundary where learning rate, saving, and validation comparison are grouped. The teacher stream rewinds only when it reaches EOF.

## 2. Common options

The options you are most likely to change:

| Goal | Option | Example |
|---|---|---|
| Choose how many sb make one epoch | `--superbatches` | `--superbatches 36` |
| Choose positions per sb | `--positions-per-superbatch` | `--positions-per-superbatch 40000000` |
| Choose how many epochs to run | `--max-epochs` | `--max-epochs 3` |
| Save less often | `--save-rate` | `--save-rate 9999` usually leaves only epoch-end saves |
| Validate every sb | `--validation-rate` | `--validation-rate 1` |
| Use tatara-style StepLR | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
| Change WRM loss exponent | `--loss-pow-exp` | `--loss-pow-exp 2.5` |
| Disable SFNN factorizer | `--sfnn-factorizer` | `--sfnn-factorizer none` |

Fuller option table:

| Flag | What it changes | Default |
|---|---|---|
| `--backend` | Training implementation. Usually leave this as `cuda-cpp` | `cuda-cpp` |
| `--batch-size` | Positions per weight update. Larger batches use more VRAM and give a steadier gradient | 65536 |
| `--positions-per-superbatch` | Target positions per sb. Rounded down to a multiple of `batch-size` | 100000000 |
| `--teacher-shuffle-buffer-sbs` | How many sb of teacher positions to shuffle in RAM. `4` means two 4-sb buffers. `0` disables in-training shuffle | 1 |
| `--teacher-shuffle-buffer-batches` | Same shuffle buffer size, specified in batches. Usually use `--teacher-shuffle-buffer-sbs` instead | omitted |
| `--teacher-shuffle-seed` | Seed for in-training teacher shuffle | 0 |
| `--threads` | CPU workers for preparing positions. Set explicitly if CPU scheduling becomes a bottleneck | auto |
| `--loader-threads` | CPU workers for loading/decoding teacher files | auto |
| `--cuda-cpp-diagnostics-rate` | How often to write diagnostic timing logs for speed investigation | 1 |
| `--superbatches` | Number of sb in one epoch | omitted |
| `--max-epochs` | Maximum epochs to run | omitted |
| `--save-rate` | Save a checkpoint every N sb | 20 |
| `--validation-rate` | Validate every N sb. Independent from saving | same as `--save-rate` |
| `--test-positions` | Number of validation positions. If omitted, use all positions in `--test-teacher` | all |
| `--test-batch-size` | GPU batch size for validation. Lower only when validation runs out of VRAM | 65536 |
| `--save-epoch-end` / `--no-save-epoch-end` | Whether to save at the end of each epoch | on |
| `--lr` | Learning rate at epoch start | 0.000875 |
| `--lr-min` | Minimum learning rate | 0.00001 |
| `--lr-schedule` | Learning-rate schedule. Start with `step` | `step` |
| `--lr-step-gamma` | Multiplicative factor for `step` schedule | auto / 0.992 |
| `--lr-step-positions` | Positions between LR drops. Omitted means one drop per sb | omitted |
| `--lambda` | Blend between teacher score and game result | 1.0 |
| `--win-rate-model` | Convert teacher score to a win-rate label and train in that space | on |
| `--loss-sigmoid-mse` | Use MSE on `sigmoid(model_output)` instead of WRM, for comparison experiments | off |
| `--loss-pow-exp` | Exponent in WRM loss. `2.0` is squared error; `2.5` is also a useful experiment | 2.0 |
| `--wrm-nnue2score` | Multiplier that maps network output back to score scale for WRM prediction | 600 |
| `--wrm-target-calibration-positions` | Number of teacher-prefix positions used to estimate teacher-score → win-rate-label coefficients. `0` uses built-in coefficients without estimation | 100000 |
| `--wrm-target-offset` / `--wrm-target-scaling` | Manually set teacher-score → win-rate-label coefficients. Usually do not pass these | omitted |
| `--sfnn-factorizer` | How SFNN shares common components between buckets. Usually `shared` | `shared` |
| `--optimizer` | Optimizer. Usually leave as `ranger` | `ranger` |
| `--optimizer-weight-decay` | Weight decay | 0.0 |
| `--optimizer-epsilon` / `--optimizer-beta1` / `--optimizer-beta2` | Fine-grained optimizer coefficients for controlled experiments | omitted |

## 3. Learning-rate schedules

`--lr-schedule step` is the default. It applies:

```text
lr = lr * gamma
```

at a fixed interval. If `--lr-step-positions` is omitted, the interval is one sb. At the next epoch, LR returns to `--lr`.

To use tatara-style `gamma=0.992`, write:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --lr 0.000875 \
    --lr-min 0.00001 \
    --lr-schedule step \
    --lr-step-gamma 0.992 \
    --tag step-gamma-0992
```

If you omit `--lr-step-gamma` and set `--superbatches`, BulletOu computes a gamma that reaches `--lr-min` within one epoch.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --positions-per-superbatch 40000000 \
    --superbatches 36 \
    --max-epochs 3 \
    --lr 0.000875 \
    --lr-min 0.00001 \
    --lr-schedule step \
    --tag step-auto-gamma
```

In this example, each epoch’s 36 sb move from `--lr` toward `--lr-min`, then epoch 2 starts from `--lr` again.

`geometric` and `cos` are smooth schedules. Start with `step` unless you explicitly want to compare schedules.

| Value | Behavior |
|---|---|
| `step` | Drops LR in stairs every sb or every configured position interval |
| `geometric` | Drops LR by a tiny constant multiplier every batch |
| `cos` | Uses a cosine curve from `--lr` to `--lr-min` |
| `plateau` | Repeats the same teacher interval at a lower LR when validation does not improve |

## 4. Turning teacher data into training labels

Teacher files usually contain two useful pieces of information:

| Data | Meaning |
|---|---|
| Teacher score | The teacher engine’s numeric evaluation for the position |
| Game result | Whether the game eventually ended as win/draw/loss |

`--lambda` controls how these two sources are blended.

```text
training label = λ × label from teacher score + (1 - λ) × label from game result
```

| `--lambda` | Meaning |
|---|---|
| `1.0` | Use only teacher score. Default |
| `0.5` | Mix teacher score and game result evenly |
| `0.0` | Use only game result |

Start with `1.0`. Try values such as `0.5` or `0.7` only when you intentionally want game results to affect the training label.

## 5. WRM loss

WRM means win-rate model. BulletOu does not feed teacher scores directly into the loss. It first converts a teacher score such as `+300` into a 0–1 win-rate-like label, converts the network output into the same 0–1 space, then compares them. This is the default loss.

WRM changes three things:

| Item | What happens |
|---|---|
| Teacher score | Converted to a win-rate label |
| Network output | Converted to a win-rate prediction |
| Loss | Uses `abs(label - prediction)^p` |

`p` is `--loss-pow-exp`.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --loss-pow-exp 2.5 \
    --tag wrm-pow25
```

### Coefficients for teacher-score → win-rate-label conversion

Converting teacher scores to win-rate labels needs coefficients that match the teacher’s score scale. By default, BulletOu estimates them from the first 100,000 teacher positions.

```text
Look at teacher_score and game_result,
then estimate how many score points correspond to how much win probability
for this teacher data.
```

Change the sample size with `--wrm-target-calibration-positions`.

```bash
# Estimate from the first 300,000 positions
--wrm-target-calibration-positions 300000
```

Use `0` only when you do not want estimation.

```bash
--wrm-target-calibration-positions 0
```

That uses the built-in coefficients `offset=270`, `scaling=380`. If you must set the values explicitly for a controlled experiment, pass both:

```bash
--wrm-target-offset 270 --wrm-target-scaling 380
```

For normal training, do not pass these options. Use the default estimation.

### Check the score→win-rate shape on your teacher data

The relation between teacher score and game result can differ by dataset. This diagnostic command compares a plain sigmoid with WRM on the actual `(score, game_result)` statistics.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --analyze-score-winrate \
    --fit-positions 100000 \
    --analyze-positions 1000000 \
    --bin-size 50 \
    --score-winrate-csv score-winrate.csv
```

This command does not train. It fits both curves on the first `--fit-positions` positions, then reports BCE / Brier score and per-score-bucket empirical win rate on the following `--analyze-positions` positions.

| Output | Meaning |
|---|---|
| `sigmoid(score/s)` | Converts `score` to 0–1 with one scale parameter |
| `WRM(offset,scale)` | Uses offset and scale, allowing a flatter region near score 0 |
| `heldout_bce` | Lower is a better fit to game-result statistics |
| `heldout_brier` | Lower is a better probability prediction |
| `empirical` | `(wins + 0.5 * draws) / positions` in that score bucket |

If `delta(WRM - sigmoid)` is negative, WRM fits that teacher data better. If it is positive, the plain sigmoid fits better.

### `--wrm-nnue2score`

`--wrm-nnue2score` maps network output back to score scale before WRM prediction. The default is `600`. Change it only for explicit comparison experiments, for example when matching tatara settings.

### Comparing with sigmoid-MSE

To train with MSE on `sigmoid(model_output)` instead of WRM, pass:

```bash
--loss-sigmoid-mse
```

WRM and sigmoid-MSE use different formulas, so compare raw loss values only between runs with the same loss setting.

## 6. SFNN factorizer

SFNN architectures such as `k3k3` or `hand1024` create many buckets. More buckets give more expressive power, but fewer teacher positions reach each bucket.

The factorizer lets buckets share common components instead of making every bucket completely independent. This can reduce overfitting and make training more stable when teacher density is low.

| Setting | Meaning |
|---|---|
| `--sfnn-factorizer shared` | Share a common component across buckets. Default |
| `--sfnn-factorizer none` | Disable factorizer |
| `--sfnn-factorizer axis` | Enable every available bucket direction in the architecture. Example: `hand1024_k3k3` enables both king and hand |
| `--sfnn-factorizer king=axis,hand=axis` | Specify directions explicitly |
| `--sfnn-factorizer king=axis,hand=shared` | Use direction-specific sharing for king and only common sharing for hand |

Example:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k29k29 \
    --sfnn-factorizer king=axis \
    --tag k29-axis
```

If you change factorizer settings while continuing from saved data, pass `--resume` explicitly. BulletOu stops automatic resume when training settings change, so accidental continuation does not happen silently.

## 7. Save and validation frequency

Saving and validation are separate:

```bash
--save-rate 20 --validation-rate 1
```

This means “save every 20 sb, but validate every sb.”

Epoch-end saving is enabled by default. If one epoch is 36 sb and you want only epoch-end saves, use a large save rate:

```bash
--save-rate 9999
```

The save rate is never reached inside the 36-sb epoch, so the epoch-end save is the one that remains.

## 8. Reading training speed

Look at stdout `[train]` rows:

```text
[train]  epoch 1  sb 12/36  this-sb=39,976,960 pos (...)  wall=...s  train=...s  pos/s=...
```

| Field | Meaning |
|---|---|
| `wall` | Real elapsed time for that sb, including validation and saving |
| `train` | Training time only, excluding validation and saving |
| `pos/s` | Training speed computed from `train` |

If the GPU is idle but `pos/s` is low, teacher loading, decoding, or shuffling may be the bottleneck. Check:

- Whether teacher data is on slow storage
- Whether `--teacher-shuffle-buffer-sbs` is too large
- Whether `--threads` / `--loader-threads` are oversubscribing the CPU
- Whether `cuda-cpp-diagnostics.log` shows large teacher queue wait time

## 9. Optimizer

Usually leave `--optimizer ranger` and `--optimizer-weight-decay 0.0`.

When changing optimizer coefficients, change one condition at a time.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --optimizer ranger \
    --optimizer-weight-decay 0.0 \
    --optimizer-beta1 0.9 \
    --optimizer-beta2 0.999 \
    --optimizer-epsilon 0.0000001 \
    --tag optimizer-test
```

## 10. A good starting command

This SFNN example uses 36 sb per epoch, validates every sb, and keeps only epoch-end saves.

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher C:\shogi\teacher\sojo `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --positions-per-superbatch 40000000 `
  --superbatches 36 `
  --max-epochs 1 `
  --lr 0.000875 `
  --lr-min 0.000030 `
  --lr-schedule step `
  --optimizer ranger `
  --optimizer-weight-decay 0.0 `
  --save-rate 9999 `
  --validation-rate 1 `
  --tag sfnn-sojo-36sb
```

Next: [Continued training](additional-training.md)

Previous: [Advanced guide](README.md)
