# 6. Tune the training — schedule and training target

<a href="../../ja/tutorial/6-tune.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Once a default-config run from [4. Run the training](4-train.md) is working, this page covers **the flags for tuning the training**. **The defaults are fine for a first run.** Come back here when you need to adjust things.

## 6.1 Training schedule

The `superbatch` in the log is **the unit at which checkpoints and learning rate are updated**, about 100M positions by default.

Main flags:

| Flag | Meaning | Default |
|---|---|---|
| `--backend` | Training backend. BulletOu training is Windows-native `cuda-cpp`; this option remains only for explicit scripts and currently accepts only `cuda-cpp` | `cuda-cpp` |
| `--batch-size` | Positions per gradient step. If omitted, BulletOu uses 65536 to match tatara | 65536 |
| `--positions-per-superbatch` | Target positions per superbatch. The actual value is rounded down to a multiple of `--batch-size` | 100000000 |
| `--teacher-shuffle-buffer-batches` | In-trainer teacher shuffle window. BulletOu allocates two CPU windows of `batch_size × N` positions each, consuming mini-batches from one while reading and Fisher-Yates shuffling the other. `N` must divide the effective `batches_per_superbatch`. `0` disables it | 0 |
| `--teacher-shuffle-seed` | Base seed for in-trainer teacher shuffle | 0 |
| `--cuda-cpp-diagnostics-rate` | SFNN per-superbatch diagnostics log. Writes teacher queue wait / load / prepare and representative CUDA stage timings to `cuda-cpp-diagnostics.log`. `1` profiles one CUDA step every sb, `N` every N superbatches, `0` disables it | 1 |
| `--superbatches` | Number of superbatches per epoch. For `geometric` / `cos`, this is the LR cycle length. For `step`, it is the epoch processing cap. For `plateau`, it is a safety cap | unlimited (= non-plateau runs until teacher EOF; plateau runs until `lr_min`) |
| `--max-epochs` | Maximum number of epochs. `--max-epoch` is also accepted as an alias. For `step` / `geometric` / `cos`, this is the number of LR cycles. For `plateau`, this caps plateau epochs. With `--test-teacher`, every schedule stops before the cap when epoch-final loss and accuracy both fail to improve | omitted = no fixed epoch cap |
| `--save-rate` | Save a checkpoint every N superbatches. By default, the final superbatch of each epoch is also saved even when it is not on a save-rate boundary. Plateau scheduling still requires `--save-rate 1` | 20 |
| `--validation-rate` | Run `--test-teacher` validation every N superbatches without necessarily saving a checkpoint. If omitted, it follows `--save-rate`. Plateau scheduling requires `--validation-rate 1` | same as `--save-rate` |
| `--save-epoch-end` / `--no-save-epoch-end` | Keep or disable the implicit checkpoint at the final superbatch of each epoch | on |
| `--lr` | Starting LR (lr_max; value at the start of each cycle) | 0.000875 |
| `--optimizer` | Optimizer. BulletOu currently exposes Ranger (RAdam+Lookahead), matching the tatara/bullet-shogi recipe | `ranger` |
| `--lr-schedule` | `step` (= staircase StepLR), `geometric` (= log-linear decay), `cos` (= cosine annealing), or `plateau` (= lower LR only when the validation monitor stops improving) | `step` |
| `--lr-min` | Floor LR. For `step` / `plateau`, this is the lower bound. For `geometric` / `cos`, this is reached at the end of each cycle | 0.00001 |
| `--lr-step-gamma` | Multiplicative LR factor for `step`. If omitted and `--superbatches` is set, BulletOu computes the value that reaches `--lr-min` from `--lr` within one epoch. If the epoch length is open-ended, it falls back to `0.992` | auto / 0.992 |
| `--lr-step-positions` | Positions per `step` LR drop. If omitted, one superbatch is used | omitted |
| `--lr-plateau-factor` | Factor multiplied into LR when the `plateau` monitor does not improve | 0.5 |
| `--lr-plateau-min-delta` | Minimum improvement used by the per-superbatch `plateau` decision | 0.0 |
| `--lr-plateau-monitor` | Validation metric used by `plateau`: `loss`, `accuracy`, or `loss_or_accuracy` | `loss_or_accuracy` |
| `--lambda` | Blend weight between teacher eval and W/D/L (see [§6.2](#62-training-target-lambda)) | 1.0 (= pure eval) |
| `--scale` | Eval-to-score sigmoid scale for the default sigmoid-MSE target | 290 |
| `--win-rate-model` | Use WRM (win-rate-model) target conversion and loss (see [§6.2](#wrm-win-rate-model-loss)) | off |
| `--loss-pow-exp` | Exponent `p` in the WRM error term `|prediction - target|^p`; used only with `--win-rate-model` | 2.0 |
| `--wrm-nnue2score` | WRM prediction-side scale. In `prediction = wrm(model_output × wrm_nnue2score)`, this sets `wrm_nnue2score` | 600 |
| `--sfnn-factorizer` | Select SFNN residual factorizer terms. `shared` is the default shared stack factorizer; `none` disables it; `axis` enables shared plus all available bucket-axis factorizer terms; combined forms such as `king=axis,hand=shared` are accepted for mixed king/hand bucket experiments. | `shared` |
| `--sfnn-factorized` / `--no-sfnn-factorized` | Compatibility aliases for enabling shared factorizer or disabling all SFNN factorizer terms. Prefer `--sfnn-factorizer shared` or `--sfnn-factorizer none` in new commands. | on |
| `--optimizer-weight-decay` | Weight decay for the selected optimizer | 0.0 |
| `--optimizer-epsilon` | Override epsilon for the selected optimizer. If omitted, the optimizer's own default is used | omitted |
| `--optimizer-beta1` | Override beta1 for the selected optimizer. If omitted, the optimizer's own default is used | omitted |
| `--optimizer-beta2` | Override beta2 for the selected optimizer. If omitted, the optimizer's own default is used | omitted |

Example (100M positions × 40 superbatches = 4 billion positions total):

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_256x2_32_32 \
    --teacher teachers/ \
    --superbatches 40
```

If your teacher file is smaller than one superbatch (< 100M positions), lower it with something like `--positions-per-superbatch 10000000` so multiple saves fire. The effective value is `floor(positions / batch_size) * batch_size`.

### Learning-rate evolution

The default `step` schedule is a staircase StepLR-style schedule within one epoch. If `--lr-step-positions` is omitted, BulletOu applies `lr *= gamma` once per superbatch, floors at `--lr-min`, and restarts back to `--lr` at the next epoch boundary.

If `--lr-step-gamma` is omitted while `--superbatches` is set, BulletOu automatically computes the `gamma` that reaches `--lr-min` from `--lr` within one epoch. For example, `--superbatches 15` normally means 15 LR steps to `lr_min`. If `--lr-step-positions` is set explicitly, the step count is derived from one epoch's positions divided by that interval. If the epoch length is open-ended, BulletOu uses `gamma=0.992`.

If `geometric` or `cos` is selected explicitly, the schedule sweeps from `--lr` (lr_max) down to `--lr-min` over one epoch, then warm-restarts back to lr_max at the next epoch's start. They differ only in the curve shape:

| schedule | formula | shape |
|---|---|---|
| `geometric` | `lr(t) = lr_max × (lr_min/lr_max)^t` | Log-linear — constant multiplicative drop per batch |
| `cos` | `lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(πt))` | Gentle at start/end, steepest in the middle |

`t = (cumulative_positions mod period) / period`, `period = one epoch's positions` (auto-derived).

**Period rules**:

| Situation | period |
|---|---|
| `--superbatches N` set | `N × sb_size` (= one epoch, the **recommended** setup) |
| Unlimited sb AND HCPE / PSV/.bin teacher | Total teacher position count (read from file sizes) |
| Unlimited sb AND HCPE3 / pack teacher | Error — variable-length format, set `--superbatches` explicitly |

When `--superbatches N` is set, an epoch is a validation / LR-control cycle, **not** one teacher pass. If the teacher reaches EOF in the middle of an epoch, BulletOu wraps to the teacher beginning and continues until N superbatches are complete. Conversely, epoch 2 does not rewind the teacher. It starts from the teacher position reached at the end of epoch 1. The teacher is a cyclic stream. `step` / `geometric` / `cos` restart LR back to `--lr` at epoch boundaries.

Example with `--superbatches 4 --lr 0.001 --lr-min 0.00001` (1 epoch = 4 sb ≒ 400M positions):

| Position within cycle | t | geometric | cos (cosine) |
|---|---|---|---|
| 0M (sb 1 start) | 0.0 | 0.001 | 0.001 |
| 100M (sb 2 start) | 0.25 | 0.000316 | 0.000856 |
| 200M (sb 3 start) | 0.5 | 0.000100 | 0.000505 (midpoint) |
| 300M (sb 4 start) | 0.75 | 0.0000316 | 0.000155 |
| 400M (sb 4 end) | 1.0 | 0.00001 | 0.00001 |
| Next epoch sb 1 | 0.0 | **0.001** ← warm restart | **0.001** ← warm restart |

The `geometric` schedule is **log-linear**: every batch multiplies lr by `(lr_min/lr_max)^(1/batches_per_epoch)` ≒ `0.99987`, a very smooth exponential decay.

⚠️ **`--lr-min` must be `> 0` for `step` / `geometric`**: `geometric` breaks when `lr_min = 0`, and `step` uses `lr_min` as its decay floor, so the CLI requires a positive value for both. `1e-5`–`1e-6` is typical. `cos` accepts 0 mathematically (with a warning).

Inspect `<NNNN>/learn.log`'s `lr_start` / `lr_end` columns to verify the actual lr trajectory ([§7.2](7-result.md#72-reading-the-training-log-learnlog)). Note that bullet's stdout `LR dropped to X` only prints at sb boundaries — for per-batch changes look at the per-dir log.

#### tatara / bullet-shogi / nnue-pytorch StepLR condition

`--lr-schedule step` is a staircase scheduler that applies `lr *= gamma` every configured number of positions. The old smooth BulletOu `step` schedule has been renamed to `geometric`. The current `step` schedule restarts back to `--lr` every epoch.

To force the same fixed `gamma=0.992` condition as tatara / bullet-shogi, spell it out explicitly:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --lr 0.000875 \
    --lr-schedule step \
    --lr-step-gamma 0.992 \
    --lr-min 0.00001 \
    --tag step-ablation
```

If `--lr-step-positions` is omitted, BulletOu drops LR once per superbatch. This corresponds to tatara's `lr_step=1` and bullet-shogi's `StepLR { gamma=0.992, step=1 }`. For position-fixed comparison runs, you can pass `--lr-step-positions 100000000` explicitly.

If instead you want BulletOu to choose `gamma` from the epoch length, leave `--lr-step-gamma` out:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --positions-per-superbatch 40000000 \
    --superbatches 15 \
    --max-epochs 3 \
    --lr 0.000875 \
    --lr-schedule step \
    --lr-min 0.00001 \
    --tag step-auto-gamma
```

Because `--lr-step-positions` is omitted here, LR decays once per superbatch and BulletOu internally uses `gamma = (lr_min / lr)^(1 / 15)` for each epoch. Epochs 2 and 3 start again from `--lr`.

#### ReduceLROnPlateau

Use `--lr-schedule plateau` when you want validation metrics to decide LR reductions instead of forcing a fixed cosine or geometric period.

After each saved superbatch, BulletOu evaluates `--test-teacher` and checks `test_value_loss` / `test_value_accuracy`. The update is accepted when the metric selected by `--lr-plateau-monitor` improves. If it does not improve, BulletOu discards that update, restores both model weights and optimizer state to the start of the superbatch, multiplies LR by `--lr-plateau-factor`, and retries the same teacher interval. When the next LR would go below `--lr-min`, it runs one final attempt at exactly `--lr-min`. That final attempt is accepted only if the monitor improves; otherwise it is discarded and the epoch ends.

`--lr-plateau-monitor` has three modes:

| Value | Acceptance rule |
|---|---|
| `loss` | Accept only when `test_value_loss` decreases. This is the historical ReduceLROnPlateau behaviour |
| `accuracy` | Accept only when `test_value_accuracy` increases |
| `loss_or_accuracy` | Accept when either loss decreases or accuracy increases. This is the default |

`--lr-plateau-min-delta` applies only to the loss side. Accuracy uses a strict increase.

If `--max-epochs` is omitted with `plateau`, there is no fixed epoch limit. After each epoch, BulletOu compares the epoch-final validation metrics that remained in `summary-learn.log` with the previous epoch's final metrics. If `test_value_loss` does not decrease and `test_value_accuracy` does not increase, training stops. `--lr-plateau-monitor` and `--lr-plateau-min-delta` affect only the per-superbatch plateau decision; epoch-to-epoch stopping always uses strict loss-or-accuracy improvement with no tolerance.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch NNUE_halfkp_256x2_32_32 \
    --lr 0.001 --lr-min 0.00001 \
    --lr-schedule plateau \
    --lr-plateau-factor 0.5 \
    --lr-plateau-monitor loss_or_accuracy
```

Constraints:

- `--test-teacher` is required.
- `--save-rate 1` is required because LR is decided once per superbatch.
- `--validation-rate 1` is also required for the same reason.
- `plateau` is currently supported for NNUE/SFNN eval types.

#### Comparing `step` vs `cos`

Run twice on the same teacher / same architecture and overlay the `summary-learn.log` curves. Both schedules share the same `--lr-min`, which makes apples-to-apples comparison easy:

```bash
# geometric decay
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch NNUE_kp_256x2_32_32 \
    --max-epochs 10 --superbatches 4 --tag 5G-geometric \
    --lr-schedule geometric --lr-min 0.00001

# cosine (one cycle per epoch)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch NNUE_kp_256x2_32_32 \
    --max-epochs 10 --tag 5G-cos --superbatches 4 \
    --lr-schedule cos --lr-min 0.00001
```

The two runs land in `checkpoints/NNUE_KP-NNUE_kp_256x2_32_32-5G-geometric/` and `-5G-cos/`. Load each `summary-learn.log` in pandas / Excel and compare the `test_value_accuracy` / `test_value_loss` columns to see which schedule helps more on your teacher.

### Count the teacher to pick `--superbatches`

For both `geometric` and `cos` schedules, you'll want one epoch to fit the teacher cleanly. That means knowing the teacher's total position count. BulletOu has a dedicated flag for that: `--count-teacher`. It reads `std::fs::metadata` only (no actual file content), so it's **instant even for hundreds of GB**:

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
```

Example output:
```
Counting Hcpe teacher files (38 byte/record)...
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0001.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0002.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0003.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0004.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0005.hcpe
---
Total: 461373440 positions  (16.71 GB)  across 5 file(s)
Per-default-sb (= 100M positions): 4 full sb + 0.61 partial sb
Suggested `--superbatches`: 4 (= use 4 full sb per epoch; ~61M positions leftover ...)
```

Then `--superbatches 4` gives:
- 1 epoch = 4 sb = 400M positions
- cos period = 400M (= exactly 1 epoch)
- `lr_min` lands at end of sb 4; warm restart to `lr_max` at sb 1 of the next epoch

The trailing 61M of teacher is not discarded. Epoch 1 reads the first 400M positions; epoch 2 starts from the remaining 61M, then wraps to the teacher beginning and continues. `--superbatches` decides where the LR cycle / validation epoch ends; it does not mean "rewind the teacher at every epoch".

#### Supported formats

| Format | Record size | `--count-teacher` |
|---|---|---|
| HCPE | 38 byte fixed | ✅ instant |
| PSV / `.bin` | 40 byte fixed | ✅ instant |
| HCPE3 | variable (game-structured) | ❌ not yet (would need to walk every game header) |
| pack | variable (game-structured) | ❌ same |

For HCPE3 / pack, pre-convert the corpus to HCPE / PSV/.bin, or set `--superbatches` manually.

### Multi-epoch training

`--max-epochs N` runs at most N epochs. For `step` / `geometric` / `cos`, this means N LR cycles. If omitted, there is no fixed epoch cap for any schedule; with `--test-teacher`, training still stops when epoch-final loss and accuracy both fail to improve. Without `--test-teacher`, non-plateau schedules keep looping over epochs until interrupted.

At each epoch boundary, the displayed superbatch counter and LR cycle reset. With explicit `--superbatches`, the teacher position does **not** reset; the teacher stream continues and wraps only at EOF. Only the old unlimited non-plateau mode (`--superbatches` omitted) treats teacher EOF as epoch end and starts the next epoch from the teacher beginning.

This is useful when you want each epoch to descend on its own LR schedule (a way to escape local minima in long training) without repeatedly training on the same prefix of the teacher. For `cos` schedule, setting `--superbatches N` automatically makes cycle = epoch (= canonical SGDR setup). When `--test-teacher` is set, every schedule compares epoch-final validation metrics with the previous epoch. If `test_value_loss` does not decrease and `test_value_accuracy` does not increase, training stops even before `--max-epochs` is reached.

## 6.2 Training target (`--lambda`)

Each teacher position carries **two labels**:

1. **Teacher eval** — the teacher engine's evaluation of that position (sigmoid-transformed).
2. **Game result** — the actual outcome of that game (W/D/L = 1.0 / 0.5 / 0.0, from side-to-move perspective).

`--lambda <λ>` controls how the loss target blends the two (matches YaneuraOu's built-in `lambda` convention):

```
target = λ × teacher_eval + (1 − λ) × game_result
```

| `--lambda` | Meaning |
|---|---|
| `1.0` (default) | 100% teacher eval, game result ignored |
| `0.5` | 50/50 blend (the classic elmo-style mix) |
| `0.0` | 100% game result, teacher eval ignored |
| `0.7` etc. | Any intermediate value is fine |

The default `1.0` (pure eval) is the safe starting point: the network learns to imitate the teacher engine's scores directly.

Lower `--lambda` to mix in the W/D/L game result. Pure-result training (`--lambda 0.0`) doesn't rely on teacher strength but has sparser gradients and slower convergence. A practical mix is usually `0.5–0.8`.

### WRM (win-rate-model) loss

Add `--win-rate-model` to use WRM target conversion and WRM loss instead of BulletOu's default MSE on `sigmoid(model_output)`.

This changes:

- teacher eval conversion to a win-rate target with `out_scaling=380` and `offset=270`
- network output conversion to a win-rate prediction with `nnue2score=600`, `in_scaling=340`, and `offset=270`
- loss formula to `abs(target - prediction)^p`, where `p` is `--loss-pow-exp`
- `test_value_loss` and `plateau` decisions to use the same WRM loss

`--loss-pow-exp` and `--wrm-nnue2score` follow tatara's convention. The `--loss-pow-exp` default is `2.0` (squared error). Use `2.5` for the commonly reported nnue-pytorch-style setting. The `--wrm-nnue2score` default is `600`.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-wrm-test \
    --win-rate-model \
    --loss-pow-exp 2.5 \
    --wrm-nnue2score 600
```

`--loss-pow-exp` and `--wrm-nnue2score` are used only when `--win-rate-model` is enabled. A WRM run's `test_value_loss` uses a different formula from the default loss, so do not compare the raw loss number directly against a non-WRM run. Compare runs with the same WRM setting, or use accuracy / engine strength.

### Optimizer Selection

`--optimizer` currently accepts only `ranger`, matching bullet-shogi's shogi examples and the tatara reference recipe.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger \
    --optimizer ranger
```

`ranger` is BulletOu's existing RAdam+Lookahead implementation. It is not a full clone of nodchip nnue-pytorch's Ranger21, so treat it as an ablation for narrowing the optimizer gap. To move it toward the nnue-pytorch condition, start with:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger-decay0 \
    --optimizer ranger \
    --optimizer-weight-decay 0.0 \
    --optimizer-beta1 0.9 \
    --optimizer-beta2 0.999 \
    --optimizer-epsilon 0.0000001
```

If `--optimizer-beta1`, `--optimizer-beta2`, or `--optimizer-epsilon` is omitted, BulletOu uses Ranger's defaults. In particular, `ranger` defaults to bullet-shogi's `beta1=0.99`, not the common Adam-style `0.9`.

### Optimizer Weight Decay

BulletOu's default setting uses `--optimizer-weight-decay 0.0`, matching the tatara SFNN-1536 reference run. To compare weight decay in isolation, keep the optimizer fixed and pass a non-zero value such as `--optimizer-weight-decay 0.01`.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger-decay001 \
    --optimizer-weight-decay 0.01
```

This does not change the loss formula, so `test_value_loss` is directly comparable with the default run. Treat it as a separate ON/OFF experiment from WRM loss (`--win-rate-model`).

### Optimizer Epsilon

If omitted, BulletOu uses the selected optimizer's own epsilon default. nodchip nnue-pytorch's Ranger21 uses `eps=1e-7`, so use `--optimizer-epsilon 0.0000001` to test only that difference.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger-eps1e-7 \
    --optimizer-epsilon 0.0000001
```

This is another optimizer-condition ablation. Compare it by itself first.

### Optimizer Beta

Optimizer `beta1` / `beta2` can also be set from the CLI. If omitted, BulletOu uses Ranger's defaults: `beta1=0.99`, `beta2=0.999`.

If you want to isolate only the optimizer momentum time constants, pass `--optimizer-beta1` / `--optimizer-beta2`.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-optimizer-beta-test \
    --optimizer-beta1 0.85 \
    --optimizer-beta2 0.995
```

This is not a Ranger21 compatibility mode by itself. Compare it by itself before combining it with weight decay or epsilon changes.

```bash
# elmo-style 50/50 blend on KPPT
./target/release/examples/bulletou \
    --arch KPPT \
    --teacher teachers/ \
    --lambda 0.5
```

(`WDL` = Win/Draw/Loss.)

---

Next: [7. Inspect the result](7-result.md) — check the output and read the training log

Previous: [4. Run the training](4-train.md)
