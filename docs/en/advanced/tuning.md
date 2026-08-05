# Adjust training settings

<a href="../../ja/advanced/tuning.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

Read this after the command in [Tutorial 3: Run training](../tutorial/3-train.md) works. For a first run, keep the defaults. Come back here when you want to adjust speed, save frequency, validation frequency, learning rate, loss, or SFNN factorizer settings.

## 1. Units used in the logs

BulletOu logs use `batch`, `superbatch`, and `epoch`.

| Name | Meaning |
|---|---|
| batch | Positions used for one GPU weight update. Default: `--batch-size 65536` |
| superbatch / sb | Progress, validation, and save unit. Its size is controlled by `--positions-per-superbatch` |
| epoch | A group of `--superbatches` superbatches. Learning rate returns to `--lr` at the epoch start |
| checkpoint | Saved files for resuming (`state.bin`) and for the engine (`nn.bin`) |
| validation | Accuracy/loss measured on `--test-teacher` |

Example:

```text
--batch-size 65536
--positions-per-superbatch 40000000
--superbatches 36
```

Here, one sb is `65536 x 610 = 39,976,960` positions. One epoch is 36 sb, or about 1.44 billion positions.

## 2. Common options

| Goal | Option | Example |
|---|---|---|
| Choose how many sb make one epoch | `--superbatches` | `--superbatches 36` |
| Choose positions per sb | `--positions-per-superbatch` | `--positions-per-superbatch 40000000` |
| Choose how many epochs to run | `--max-epochs` | `--max-epochs 3` |
| Save less often | `--save-rate` | `--save-rate 9999` usually leaves only epoch-end saves |
| Validate every sb | `--validation-rate` | `--validation-rate 1` |
| Use tatara-style StepLR | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
| Try WRM loss | `--win-rate-model` | `--win-rate-model --loss-pow-exp 2.5` |
| Change the sigmoid loss exponent | `--loss-pow-exp` | `--loss-pow-exp 1.5` |
| Change SFNN factorizer | `--sfnn-factorizer` | `--sfnn-factorizer none` |

Fuller option table:

| Flag | What it changes | Default |
|---|---|---|
| `--backend` | Training backend. Usually leave this as `cuda-cpp` | `cuda-cpp` |
| `--batch-size` | Positions per weight update | 65536 |
| `--positions-per-superbatch` | Target positions per sb. Rounded down to a multiple of `batch-size` | 100000000 |
| `--teacher-shuffle-buffer-sbs` | How many sb of teacher positions to shuffle in RAM. `4` means two 4-sb buffers | 1 |
| `--teacher-shuffle-buffer-batches` | Same shuffle buffer size, specified in batches. Usually use `--teacher-shuffle-buffer-sbs` instead | omitted |
| `--teacher-shuffle-seed` | Seed for in-training teacher shuffle | 0 |
| `--threads` | CPU workers for preparing positions | auto |
| `--loader-threads` | CPU workers for loading/decoding teacher files | auto |
| `--cuda-cpp-diagnostics-rate` | How often to write diagnostic timing logs | 1 |
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
| `--scale` | Scale used in `sigmoid(score / scale)`. If omitted, BulletOu uses the fixed value 290 | omitted |
| `--win-rate-model` | Use the WRM curve on the prediction side | off |
| `--loss-pow-exp` | Exponent `p` in `|prediction - target|^p`. `2.0` is squared error | 2.0 |
| `--wrm-nnue2score` | Multiplier that maps network output back to score scale for WRM prediction | 600 |
| `--sfnn-factorizer` | How SFNN shares common components between buckets | `shared` |
| `--optimizer` | Optimizer | `ranger` |
| `--optimizer-weight-decay` | Weight decay | 0.0 |
| `--optimizer-epsilon` / `--optimizer-beta1` / `--optimizer-beta2` | Fine-grained optimizer coefficients | omitted |

## 3. Learning-rate schedules

`--lr-schedule step` applies:

```text
lr = lr * gamma
```

If `--lr-step-positions` is omitted, the interval is one sb. At the next epoch, LR returns to `--lr`.

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

## 4. Turning teacher data into training labels

Teacher files usually contain two useful pieces of information:

| Data | Meaning |
|---|---|
| Teacher score | The teacher engine's numeric evaluation for the position |
| Game result | Whether the game ended as win/draw/loss |

`--lambda` controls how these two sources are blended.

```text
training label = lambda * label_from_teacher_score + (1 - lambda) * label_from_game_result
```

| `--lambda` | Meaning |
|---|---|
| `1.0` | Use only teacher score. Default |
| `0.5` | Mix teacher score and game result evenly |
| `0.0` | Use only game result |

## 5. Loss and score scale

When no loss family is specified, BulletOu uses sigmoid probability loss:

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid(network_output)
loss       = |prediction - target|^p
```

`p` is controlled by `--loss-pow-exp`. The default is `2.0`, which is sigmoid-MSE.

```bash
# sigmoid-MSE
--loss-pow-exp 2.0

# experiments with a different error exponent
--loss-pow-exp 1.5
--loss-pow-exp 2.5
```

`scale` maps teacher scores into the 0–1 label space. If you omit `--scale`, BulletOu uses the fixed value `290`.

BulletOu does not estimate the training scale from game-result labels. Those labels are not always trustworthy: for example, a dataset may come from games between weak players and then be re-scored by a stronger deep-learning engine. In that case, the game result is not a reliable calibration target for the teacher score.

```bash
# Train with fixed scale 290
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --tag sigmoid-scale290
```

Fix scale for a comparison experiment:

```bash
--scale 600
```

## 6. Trying WRM loss

WRM means win-rate model. It still computes loss in 0–1 probability space, but it uses a WRM curve on the prediction side.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --win-rate-model \
    --loss-pow-exp 2.5 \
    --tag wrm-pow25
```

WRM also uses:

```text
loss = |prediction - target|^p
```

`--loss-pow-exp` applies to both sigmoid loss and WRM loss.

`--wrm-nnue2score` maps network output back to score scale before WRM prediction. The default is `600`. WRM uses the built-in fixed target curve; BulletOu does not estimate the WRM target curve from game-result labels during training.

## 7. Check the score → win-rate shape on your teacher data

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

This command does not train. It fits both curves on the first `--fit-positions` positions, then reports BCE / Brier score and per-score-bucket empirical win rate on the following `--analyze-positions` positions. Fitting and BCE / Brier score use only decisive win/loss records.

| Output | Meaning |
|---|---|
| `sigmoid(score/s)` | Converts `score` to 0–1 with one scale parameter |
| `WRM(offset,scale)` | Uses offset and scale |
| `heldout_bce` | Lower is a better fit to game-result statistics |
| `heldout_brier` | Lower is a better probability prediction |
| `empirical` | `wins / (wins + losses)` in that score bucket |

## 8. SFNN factorizer

SFNN architectures such as `k3k3` or `hand1024` create many buckets. More buckets give more expressive power, but fewer teacher positions reach each bucket.

The factorizer lets buckets share common components instead of making every bucket completely independent.

| Setting | Meaning |
|---|---|
| `--sfnn-factorizer shared` | Share a common component across buckets. Default |
| `--sfnn-factorizer none` | Disable factorizer |
| `--sfnn-factorizer axis` | Enable every available bucket direction in the architecture |
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

## 9. Save and validation frequency

Save and validation frequency are independent:

```bash
--save-rate 20 --validation-rate 1
```

This means "save a checkpoint every 20 sb, but measure accuracy/loss every sb."

Epoch-end saving is on by default. If you want only epoch-end saves, set `--save-rate` to a value that will not be reached inside one epoch.

```bash
--save-rate 9999
```

## 10. Reading speed logs

Use the `[train]` line:

```text
[train]  epoch 1  sb 12/36  this-sb=39,976,960 pos (...)  wall=...s  train=...s  pos/s=...
```

| Field | Meaning |
|---|---|
| `wall` | Real time for that sb, including validation and saving |
| `train` | Training time only, excluding validation and saving |
| `pos/s` | Training throughput computed from `train` |

If GPU utilization is low and `pos/s` is low, teacher loading, decoding, or shuffling may be the bottleneck. Check `cuda-cpp-diagnostics.log` for teacher queue wait time.
