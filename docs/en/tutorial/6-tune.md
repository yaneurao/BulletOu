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
| `--superbatches` | Cap superbatches per epoch | unlimited (= run until EOF) |
| `--max-epochs` | Number of full passes through the teacher | 1 |
| `--save-rate` | Save a checkpoint every N superbatches | 1 |
| `--lr` | Starting LR (lr_max) | 0.001 |
| `--lr-schedule` | `step` (exponential decay) or `cos` (cosine annealing + warm restart) | `step` |
| `--lr-gamma` / `--lr-step-positions` | (step only) multiply LR by `lr-gamma` every `lr-step-positions` cumulative positions | 0.9 / 100000000 |
| `--lr-cosine-period` / `--lr-min` | (cos only) one cosine cycle sweeps `--lr` → `--lr-min` over `--lr-cosine-period` positions, then warm-restarts | 500000000 / 0.0 |
| `--lambda` | Blend weight between teacher eval and W/D/L (see [§6.2](#62-training-target-lambda)) | 1.0 (= pure eval) |

Example (100M positions × 40 superbatches = 4 billion positions total):

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --superbatches 40
```

If your teacher file is smaller than one superbatch (< 100M positions), lower `--batches-per-superbatch` (e.g. `1024` ⇒ 1 superbatch ≒ 16.78M positions) so multiple saves fire.

### Learning-rate evolution — `--lr-schedule step` (default)

With `--lr 0.001 --lr-gamma 0.9 --lr-step-positions 100000000` (defaults), the LR drops by 0.9× every 100M **cumulative trained positions**:

| Cumulative positions | lr |
|---|---|
| 0 – 100M | 0.001 |
| 100M – 200M | 0.000900 |
| 200M – 300M | 0.000810 |
| 500M | 0.000591 |
| 1G | 0.000349 |
| 2.2G | 0.0001 (≒ 1/10 of starting LR) |

Pass an aggressive value like `--lr-gamma 0.1` for a 10× drop every 100M. For long runs the gentler `0.9`-class default is more typical.

You can verify the actual LR after the run by inspecting `learn.log`'s `lr` column ([§7.2 Reading the training log](7-result.md#72-reading-the-training-log-learnlog)).

### Learning-rate evolution — `--lr-schedule cos` (cosine annealing)

Pass `--lr-schedule cos` to use **cosine annealing with warm restart** (SGDR) instead of the stepwise schedule:

```
t  = (cumulative_positions mod cosine_period) / cosine_period
lr = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π · t))
```

Example with `--lr-cosine-period 500000000 --lr-min 0.00001`:

| Position within cycle | t | lr |
|---|---|---|
| 0M (cycle start) | 0.0 | 0.001 (= `--lr`, lr_max) |
| 125M | 0.25 | 0.000856 |
| 250M | 0.5 | 0.000505 (midpoint) |
| 375M | 0.75 | 0.000155 |
| 500M (cycle end) | 1.0 | 0.00001 (= `--lr-min`, lr_min) |
| 500M + 1 | 0.0 (next cycle) | **0.001** ← warm restart |

Set `--lr-cosine-period` equal to one epoch's worth of positions to get exactly one full cosine sweep per epoch, with the warm restart aligned to the epoch boundary.

#### Comparing `step` vs `cos`

Run twice on the same teacher / same architecture and overlay the `learn.log` curves. Use `--tag` to keep the output directories distinct:

```bash
# stepwise
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --tag 5G-step \
    --lr-schedule step --lr-step-positions 100000000 --lr-gamma 0.9

# cosine (one cycle per epoch)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --tag 5G-cos \
    --lr-schedule cos --lr-cosine-period 500000000 --lr-min 0.00001
```

The two runs land in `checkpoints/NNUE_KP-256x2-32-32-5G-step/` and `-5G-cos/`. Load each `learn.log` in pandas / Excel and compare the `test_value_accuracy` / `test_value_loss` columns to see which schedule helps more on your teacher.

### Multi-epoch training

`--max-epochs N` runs through the teacher data N times. At each epoch boundary:
- The LR scheduler resets (superbatch counter back to 1, `lr = --lr`) — applies to both `step` and `cos`.
- The dataloader rewinds to the beginning of the data.

Effectively N restarted trainings on the same data. Useful when you want each epoch to descend on its own LR schedule (a way to escape local minima in long training). For `cos` schedule, setting `--lr-cosine-period = epoch_size` is the canonical SGDR setup.

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
