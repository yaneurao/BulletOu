# Adjust training settings

<a href="../../ja/advanced/tuning.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

Read this page after the command in [Tutorial 3: Run training](../tutorial/3-train.md) works.

For a first run, keep the defaults. Come back here when you want to adjust speed, save frequency, validation frequency, learning rate, loss, or SFNN factorizer settings.

## 1. Units used in the logs

| Name | Meaning |
| --- | --- |
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
| --- | --- | --- |
| Choose how many sb make one epoch | `--superbatches` | `--superbatches 36` |
| Choose positions per sb | `--positions-per-superbatch` | `--positions-per-superbatch 40000000` |
| Choose how many epochs to run | `--max-epochs` | `--max-epochs 3` |
| Save less often | `--save-rate` | `--save-rate 9999` usually leaves only epoch-end saves |
| Put checkpoints on another drive | `--output-folder` | `--output-folder D:\checkpoints` |
| Validate every sb | `--validation-rate` | `--validation-rate 1` |
| Set StepLR decay | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
| Change the loss exponent | `--loss-pow-exp` | `--loss-pow-exp 2.5` |
| Change SFNN factorizer | `--sfnn-factorizer` | `--sfnn-factorizer none` |
| Change SFNN factorizer strength | `--sfnn-factorizer-alpha` | `--sfnn-factorizer-alpha king=0.90` |

Fuller option table:

| Flag | What it changes | Default |
| --- | --- | --- |
| `--backend` | Training backend. Usually leave this as `cuda-cpp` | `cuda-cpp` |
| `--output-folder` | Parent folder for checkpoints. Auto-derived directory names and `--tag` are still used | `checkpoints` |
| `--output` | Exact checkpoint directory. `--tag` is not used | omitted |
| `--batch-size` | Positions per weight update | 65536 |
| `--batches-per-update` | Accumulate N mini-batch gradients before one optimizer update | 1 |
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
| `--loss-pow-exp` | Exponent `p` in `|prediction - target|^p` | 2.0 |
| `--wrm-nnue2score` | WRM loss coefficient that maps `network_output` to score scale | 600 |
| `--wrm-in-offset` / `--wrm-in-scaling` | Prediction-side WRM curve | 270 / 340 |
| `--wrm-target-offset` / `--wrm-target-scaling` | Teacher-side WRM curve | 270 / 380 |
| `--loss-sigmoid-mse` | Use plain sigmoid loss instead of WRM | off |
| `--scale` | Target scale for `--loss-sigmoid-mse` | 600 |
| `--fv-scale` | `FV_SCALE` assumed for quantized `nn.bin` checks/export | 40 |
| `--sfnn-factorizer` | How SFNN shares common components between buckets | `shared` |
| `--sfnn-factorizer-alpha` | How strongly factorizer components contribute | 1.0 |
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
| --- | --- |
| Teacher score | The teacher engine's numeric evaluation for the position |
| Game result | Whether the game ended as win/draw/loss |

`--lambda` controls how these two sources are blended.

```text
training_label = lambda * label_from_teacher_score
               + (1 - lambda) * label_from_game_result
```

| `--lambda` | Meaning |
| --- | --- |
| `1.0` | Use only teacher score. Default |
| `0.5` | Mix teacher score and game result evenly |
| `0.0` | Use only game result |

For re-scored teacher data, game results are not always reliable calibration information for the teacher score. Start with `--lambda 1.0`.

## 5. Loss

The default is WRM loss. You do not need to pass a flag to enable it.

```text
score_net  = network_output * wrm_nnue2score
prediction = wrm(score_net;     wrm_in_offset,     wrm_in_scaling)
target     = wrm(teacher_score; wrm_target_offset, wrm_target_scaling)
loss       = |prediction - target|^loss_pow_exp
```

Default values:

```bash
--wrm-nnue2score 600
--wrm-in-offset 270
--wrm-in-scaling 340
--wrm-target-offset 270
--wrm-target-scaling 380
--loss-pow-exp 2.0
```

For a zero-offset WRM comparison:

```bash
--wrm-in-offset 0
--wrm-target-offset 0
```

For the plain sigmoid loss:

```bash
--loss-sigmoid-mse
--scale 600
--fv-scale 40
```

For the loss formula and how `FV_SCALE` fits in, see [Loss scale and `FV_SCALE`](scale-and-fv-scale.md).

## 6. Checking the score-to-result shape in teacher data

This is a diagnostic command. It does not train.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --analyze-score-winrate \
    --fit-positions 100000 \
    --analyze-positions 1000000 \
    --bin-size 50 \
    --score-winrate-csv score-winrate.csv
```

Output fields:

| Output | Meaning |
| --- | --- |
| `sigmoid(score/s)` | Converts `score` to 0–1 with one scale parameter |
| `heldout_bce` | Lower is a better fit to game-result statistics |
| `heldout_brier` | Lower is a better probability prediction |
| `empirical` | `wins / (wins + losses)` in that score bucket |

For re-scored teacher data, game result may reflect the original players more than the re-scoring engine. Do not automatically turn this diagnostic result into a training target.

## 7. SFNN factorizer

SFNN architectures such as `k3k3` or `hand1024` create many buckets. More buckets give more expressive power, but fewer teacher positions reach each bucket.

The factorizer lets buckets share common components. It can reduce overfitting and stabilize training when teacher density per bucket is low.

Conceptually, training uses an effective weight like this:

```text
W_effective = W_base + W_shared + W_axis + W_pair
```

`W_base` is owned by each bucket. `W_shared` is shared by all buckets. `W_axis` is shared along one bucket axis, such as king bucket or hand bucket. `W_pair` is shared by a pair of axes, such as `king-hand`, `king-progress`, or `hand-progress`.

| Setting | Meaning |
| --- | --- |
| `--sfnn-factorizer shared` | Share a common component across buckets. Default |
| `--sfnn-factorizer none` | Disable factorizer |
| `--sfnn-factorizer axis` | Enable every available bucket direction in the architecture |
| `--sfnn-factorizer pair` | Enable `axis` plus every available two-axis factorizer |
| `--sfnn-factorizer king=axis,hand=axis` | Specify directions explicitly |
| `--sfnn-factorizer king-hand,king-progress,hand-progress` | Specify two-axis factorizers explicitly |
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

Use `--sfnn-factorizer-alpha` when you want to change how strongly factorizer terms contribute.

```text
W_effective = W_base
            + alpha_shared * W_shared
            + alpha_king   * W_king_axis
            + alpha_hand   * W_hand_axis
            + alpha_pair   * W_pair
```

For example, to use only 90% of the king-axis factorizer:

```bash
--sfnn-factorizer king=axis
--sfnn-factorizer-alpha king=0.90
```

To set king and hand separately:

```bash
--sfnn-factorizer king=axis,hand=axis
--sfnn-factorizer-alpha king=0.90,hand=0.80
```

To set every factorizer term to the same strength:

```bash
--sfnn-factorizer axis
--sfnn-factorizer-alpha 0.90
```

When an architecture combines several bucket axes, such as `hand1024` and `progress8`, you can also try two-axis factorizers.

```bash
--arch SFNN_halfka2_1024_7_64_k3k3_hand1024_progress8
--sfnn-factorizer pair
```

This enables `shared`, `king-axis`, `hand-axis`, `king-hand`, `king-progress`, and `hand-progress` wherever the architecture has the required axes. Axes that do not exist in the architecture are ignored automatically.

`alpha=1.0` is the normal setting. With `alpha=0.0`, that factorizer term is not added in forward, and its gradient is also zero. This does not fold stored factorizer tensors into the base weights. If you want to continue training without factorizer terms, use `--sfnn-factorizer none`.

`alpha` can also be larger than `1.0`. The accepted range is `0.0` to `10.0`. For example, `king=2.0` adds the king-axis contribution at twice its stored value in forward, and the gradient into king-axis tensors is also doubled. Very large values can destabilize training, so treat this as an experimental tuning knob.

When BulletOu writes `nn.bin`, it folds weights using the `W_effective` formula above. So an `nn.bin` saved with `--sfnn-factorizer-alpha king=0.90` contains the king-axis contribution at 90% strength.

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
| --- | --- |
| `wall` | Real time for that sb, including validation and saving |
| `train` | Training time only, excluding validation and saving |
| `pos/s` | Training throughput computed from `train` |

If GPU utilization is low and `pos/s` is low, teacher loading, decoding, or shuffling may be the bottleneck. Check `cuda-cpp-diagnostics.log` for teacher queue wait time.

## 10. Gradient accumulation

Use `--batches-per-update N` when VRAM forces a smaller `--batch-size`, but you still want the optimizer to use a larger virtual batch.

Example:

```bash
--batch-size 16384
--batches-per-update 4
```

This reads four 16,384-position mini-batches, adds their gradients, and then applies one Ranger update. The optimizer sees a virtual batch of:

```text
16384 x 4 = 65536 positions
```

This is not the same as making each CUDA forward/backward pass as large as 65,536 positions, so it will not recover all throughput. It does reduce optimizer-update overhead and gives the optimizer a larger, less noisy gradient.

`--positions-per-superbatch` is rounded down to a multiple of `--batch-size`. For example, with:

```text
--positions-per-superbatch 40000000
--batch-size 65536
```

With `--batches-per-update 1`, BulletOu actually uses `610 * 65,536 = 39,976,960` positions for one sb.

When `--batches-per-update` is 2 or larger, BulletOu also rounds the mini-batch count down to a multiple of `--batches-per-update`. For example, with `--batches-per-update 4`, one sb uses 608 batches instead of 610 batches.

```text
608 * 65,536 = 39,845,888 positions
608 / 4 = 152 optimizer updates
```

This lets you keep user-facing commands simple, such as `--positions-per-superbatch 40000000`, without manually calculating values such as `39,845,888`.
