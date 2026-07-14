# 6. Tune the training — schedule and training target

<a href="../../ja/tutorial/6-tune.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Once a default-config run from [4. Run the training](4-train.md) is working, this page covers **the flags for tuning the training**. **The defaults are fine for a first run.** Come back here when you need to adjust things.

## 6.1 Training schedule

The `superbatch` in the log is **the unit at which checkpoints and learning rate are updated**, about 100M positions by default.

Main flags:

| Flag | Meaning | Default |
|---|---|---|
| `--batch-size` | Positions per gradient step | 16384 |
| `--batches-per-superbatch` | Mini-batches per superbatch | `ceil(100M / batch-size)` (≒ 1 superbatch ≒ 100M positions) |
| `--superbatches` | Cap superbatches per epoch. Usually unnecessary for `plateau` | unlimited (= non-plateau runs until EOF; plateau runs until `lr_min`) |
| `--max-epochs` | Maximum number of epochs. For `step` / `cos`, this is usually the number of teacher passes. For `plateau`, this caps the number of plateau epochs | omitted = `step` / `cos`: 1; `plateau`: until improvement stops |
| `--save-rate` | Save a checkpoint every N superbatches | 1 |
| `--lr` | Starting LR (lr_max; value at the start of each cycle) | 0.001 |
| `--lr-schedule` | `step` (= geometric / log-linear decay), `cos` (= cosine annealing), or `plateau` (= lower LR only when validation loss stops improving) | `step` |
| `--lr-min` | Floor LR. For `step` / `cos`, this is reached at the end of each cycle. For `plateau`, this is the final LR | 0.00001 |
| `--lr-plateau-factor` | Factor multiplied into LR when `plateau` validation loss does not improve | 0.5 |
| `--lr-plateau-min-delta` | Minimum improvement used by the per-superbatch `plateau` decision | 0.0 |
| `--lambda` | Blend weight between teacher eval and W/D/L (see [§6.2](#62-training-target-lambda)) | 1.0 (= pure eval) |
| `--nnue-pytorch-wrm-loss` | Use nnue-pytorch-compatible WRM loss (see [§6.2](#nnue-pytorch-compatible-wrm-loss)) | off |
| `--adamw-weight-decay` | AdamW weight decay. Try `0.0` when matching nnue-pytorch's optimizer condition | 0.01 |
| `--nnue-pytorch-layer-clip` | Use nnue-pytorch-compatible per-layer weight clipping | off |

Example (100M positions × 40 superbatches = 4 billion positions total):

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --superbatches 40
```

If your teacher file is smaller than one superbatch (< 100M positions), lower `--batches-per-superbatch` (e.g. `1024` ⇒ 1 superbatch ≒ 16.78M positions) so multiple saves fire.

### Learning-rate evolution

**Both** `step` and `cos` schedules sweep from `--lr` (lr_max) down to `--lr-min` over one epoch, then warm-restart back to lr_max at the next epoch's start. They differ only in the curve shape:

| schedule | formula | shape |
|---|---|---|
| `step` (default) | `lr(t) = lr_max × (lr_min/lr_max)^t` (geometric) | Log-linear — constant multiplicative drop per batch |
| `cos` | `lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(πt))` | Gentle at start/end, steepest in the middle |

`t = (cumulative_positions mod period) / period`, `period = one epoch's positions` (auto-derived).

**Period rules**:

| Situation | period |
|---|---|
| `--superbatches N` set | `N × sb_size` (= one epoch, the **recommended** setup) |
| Unlimited sb AND HCPE / PSV teacher | Total teacher position count (read from file sizes) |
| Unlimited sb AND HCPE3 / pack teacher | Error — variable-length format, set `--superbatches` explicitly |

Example with `--superbatches 4 --lr 0.001 --lr-min 0.00001` (1 epoch = 4 sb ≒ 400M positions):

| Position within cycle | t | step (geometric) | cos (cosine) |
|---|---|---|---|
| 0M (sb 1 start) | 0.0 | 0.001 | 0.001 |
| 100M (sb 2 start) | 0.25 | 0.000316 | 0.000856 |
| 200M (sb 3 start) | 0.5 | 0.000100 | 0.000505 (midpoint) |
| 300M (sb 4 start) | 0.75 | 0.0000316 | 0.000155 |
| 400M (sb 4 end) | 1.0 | 0.00001 | 0.00001 |
| Next epoch sb 1 | 0.0 | **0.001** ← warm restart | **0.001** ← warm restart |

The `step` schedule is **log-linear**: every batch multiplies lr by `(lr_min/lr_max)^(1/batches_per_epoch)` ≒ `0.99987`, a very smooth exponential decay.

⚠️ **`--lr-min` must be `> 0` for step**: the geometric formula `lr_max × (lr_min/lr_max)^t` collapses to 0 at any t>0 when `lr_min = 0`; the CLI rejects this at startup. `1e-5`–`1e-6` is typical. `cos` accepts 0 mathematically (with a warning).

Inspect `<NNNN>/learn.log`'s `lr` column to verify the actual lr trajectory ([§7.2](7-result.md#72-reading-the-training-log-learnlog)). Note that bullet's stdout `LR dropped to X` only prints at sb boundaries — for per-batch changes look at the per-dir log.

#### ReduceLROnPlateau

Use `--lr-schedule plateau` when you want validation loss to decide LR reductions instead of forcing a fixed cosine or geometric period.

After each saved superbatch, BulletOu evaluates `--test-teacher` and checks `test_value_loss`. If the loss did not improve, it discards that update, restores both model weights and Adam optimiser state to the start of the superbatch, multiplies LR by `--lr-plateau-factor`, and retries the same teacher interval. When the next LR would go below `--lr-min`, it runs one final attempt at exactly `--lr-min`. That final attempt is accepted only if validation loss improves; otherwise it is discarded and the epoch ends.

If `--max-epochs` is omitted with `plateau`, there is no fixed epoch limit. After each epoch, BulletOu compares the epoch-final `test_value_loss` that remained in `summary-learn.log` with the previous epoch's final loss. If it is **not strictly lower**, training stops. `--lr-plateau-min-delta` affects only the per-superbatch plateau decision; epoch-to-epoch stopping uses no tolerance.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_HALFKP \
    --lr 0.001 --lr-min 0.00001 \
    --lr-schedule plateau \
    --lr-plateau-factor 0.5
```

Constraints:

- `--test-teacher` is required.
- `--save-rate 1` is required because LR is decided once per superbatch.
- `plateau` is currently supported for NNUE/SFNN eval types.

#### Comparing `step` vs `cos`

Run twice on the same teacher / same architecture and overlay the `summary-learn.log` curves. Both schedules share the same `--lr-min`, which makes apples-to-apples comparison easy:

```bash
# stepwise (geometric decay)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --superbatches 4 --tag 5G-step \
    --lr-schedule step --lr-min 0.00001

# cosine (one cycle per epoch)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --tag 5G-cos --superbatches 4 \
    --lr-schedule cos --lr-min 0.00001
```

The two runs land in `checkpoints/NNUE_KP-256x2-32-32-5G-step/` and `-5G-cos/`. Load each `summary-learn.log` in pandas / Excel and compare the `test_value_accuracy` / `test_value_loss` columns to see which schedule helps more on your teacher.

### Count the teacher to pick `--superbatches`

For both `step` and `cos` schedules, you'll want one epoch to fit the teacher cleanly. That means knowing the teacher's total position count. BulletOu has a dedicated flag for that: `--count-teacher`. It reads `std::fs::metadata` only (no actual file content), so it's **instant even for hundreds of GB**:

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

The trailing 61M of teacher is not used (= each epoch uses the same first 400M). A small amount of waste is usually preferable to ragged cosine cycles.

#### Supported formats

| Format | Record size | `--count-teacher` |
|---|---|---|
| HCPE | 38 byte fixed | ✅ instant |
| PSV  | 40 byte fixed | ✅ instant |
| HCPE3 | variable (game-structured) | ❌ not yet (would need to walk every game header) |
| pack | variable (game-structured) | ❌ same |

For HCPE3 / pack, pre-convert the corpus to HCPE / PSV, or set `--superbatches` manually.

### Multi-epoch training

`--max-epochs N` runs through the teacher data N times. At each epoch boundary:
- The LR scheduler resets (superbatch counter back to 1, `lr = --lr`) — applies to both `step` and `cos`.
- The dataloader rewinds to the beginning of the data.

Effectively N restarted trainings on the same data. Useful when you want each epoch to descend on its own LR schedule (a way to escape local minima in long training). For `cos` schedule, setting `--superbatches N` automatically makes cycle = epoch (= canonical SGDR setup).

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

### nnue-pytorch-compatible WRM loss

Add `--nnue-pytorch-wrm-loss` to use nodchip nnue-pytorch's WRM (win-rate model) loss instead of BulletOu's default MSE on `sigmoid(model_output)`.

This changes:

- teacher eval conversion to a win-rate target with `out_scaling=380` and `offset=270`
- network output conversion to a win-rate prediction with `nnue2score=600`, `in_scaling=340`, and `offset=270`
- loss formula to `abs(target - prediction)^2.5`
- `test_value_loss` and `plateau` decisions to use the same WRM loss

This is an **experimental comparison flag** for reducing the training-condition gap between nnue-pytorch and BulletOu. You do not need it for a normal first run. When comparing ON/OFF, use the same teacher and architecture but a different `--tag`.

`test_value_loss` from a `--nnue-pytorch-wrm-loss` run uses a different formula from the default loss, so do not compare the raw loss number directly against a default-loss run. Compare runs with the same flag state.

Example:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type SFNN_HALFKA2 \
    --tag sfnn-wrm-test \
    --nnue-pytorch-wrm-loss
```

### AdamW weight decay

BulletOu's default AdamW uses `--adamw-weight-decay 0.01`. To move one step toward nodchip nnue-pytorch's optimizer condition while keeping AdamW, test only `--adamw-weight-decay 0.0`.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type SFNN_HALFKA2 \
    --tag sfnn-adamw-decay0 \
    --adamw-weight-decay 0.0
```

This does not change the loss formula, so `test_value_loss` is directly comparable with the default run. Treat it as a separate ON/OFF experiment from `--nnue-pytorch-wrm-loss`.

### nnue-pytorch layer clipping

Add `--nnue-pytorch-layer-clip` to move AdamW's weight clipping bounds closer to nnue-pytorch's quantisation scales.

- hidden weight: `[-127/64, 127/64]`
- final output weight: `[-127*127/(600*16), 127*127/(600*16)]`

This is an experimental NNUE / SFNN flag. It does not change the loss formula or weight decay, so compare it by itself before combining it with other experimental flags.

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type SFNN_HALFKA2 \
    --tag sfnn-layer-clip \
    --nnue-pytorch-layer-clip
```

```bash
# elmo-style 50/50 blend on KPPT
./target/release/examples/bulletou \
    --eval-type KPPT \
    --teacher teachers/ \
    --lambda 0.5
```

(`WDL` = Win/Draw/Loss.)

---

Next: [7. Inspect the result](7-result.md) — check the output and read the training log

Previous: [4. Run the training](4-train.md)
