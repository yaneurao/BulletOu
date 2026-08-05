# Adjust training settings

<a href="../../ja/advanced/tuning.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

Read this page after the command in [Tutorial 3: Run training](../tutorial/3-train.md) works. For a first run, keep the defaults. Come back here when you want to adjust speed, save frequency, validation frequency, learning rate, loss, or SFNN factorizer settings.

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
| Set StepLR decay | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
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
| `--scale` | Target scale used in `sigmoid(teacher_score / scale)`. If omitted, BulletOu uses 600 | omitted |
| `--fv-scale` | Engine-side `FV_SCALE` assumed when mapping NNUE/SFNN network output back to an eval score | 40 |
| `--loss-pow-exp` | Exponent `p` in `|prediction - target|^p`. `2.0` is squared error | 2.0 |
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

Example with `gamma=0.992`:

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

If you omit `--lr-step-gamma` and set `--superbatches`, BulletOu computes a gamma that moves from `--lr` toward `--lr-min` within one epoch.

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

BulletOu uses sigmoid probability loss. `--scale` controls how the teacher
score is converted back to a win-rate label. `--fv-scale` controls the
NNUE/SFNN output range that will be used by YaneuraOu after quantization.

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid((network_output * 8128 / fv_scale) / scale)
loss       = |prediction - target|^p
```

`8128` is `QA * QB` for the NNUE/SFNN export path (`QA=127`, `QB=64`).
For KPPT-family targets, `--fv-scale` is ignored because those targets use
their own `--yaneuraou-quant-scale` export path.

`p` is controlled by `--loss-pow-exp`. The default is `2.0`, which is squared error in sigmoid space.

```bash
# squared error
--loss-pow-exp 2.0

# experiments with a different error exponent
--loss-pow-exp 1.5
--loss-pow-exp 2.5
```

If your teacher scores were created from a win-rate model with `scale=600`,
leave BulletOu's `--scale` at the default `600`. If you also want the exported
`nn.bin` to run with `FV_SCALE=40`, leave `--fv-scale` at the default `40`.
This keeps the target win rate as `sigmoid(score / 600)` while training a
wider NNUE/SFNN network output range suitable for `FV_SCALE=40`.

For the derivation and worked examples, see [`--scale` and `--fv-scale`](scale-and-fv-scale.md).

BulletOu does not estimate the training scale from game-result labels. Those labels are not always trustworthy: for example, a dataset may come from games between weak players and then be re-scored by a stronger deep-learning engine. In that case, game result is not a reliable calibration target for the teacher score.

```bash
# Train with the default target scale 600 and FV_SCALE 40
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --tag sigmoid-scale600-fv40
```

Set fixed values explicitly for a comparison experiment:

```bash
--scale 600
--fv-scale 40
```

## 6. Checking the score-to-result shape in teacher data

This is a diagnostic command. It does not affect training.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --analyze-score-winrate \
    --fit-positions 100000 \
    --analyze-positions 1000000 \
    --bin-size 50 \
    --score-winrate-csv score-winrate.csv
```

The command fits the scale in `sigmoid(score / scale)` on the first `--fit-positions` positions, then reports BCE / Brier score and per-score-bucket empirical win rate on the following `--analyze-positions` positions. Fitting and BCE / Brier score use only decisive win/loss records.

| Output | Meaning |
|---|---|
| `sigmoid(score/s)` | Converts `score` to 0–1 with one scale parameter |
| `heldout_bce` | Lower is a better fit to game-result statistics |
| `heldout_brier` | Lower is a better probability prediction |
| `empirical` | `wins / (wins + losses)` in that score bucket |

## 7. SFNN factorizer

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

## 8. Save and validation frequency

Save and validation frequency are independent:

```bash
--save-rate 20 --validation-rate 1
```

This means "save a checkpoint every 20 sb, but measure accuracy/loss every sb."

Epoch-end saving is on by default. If you want only epoch-end saves, set `--save-rate` to a value that will not be reached inside one epoch.

```bash
--save-rate 9999
```

## 9. Reading speed logs

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
