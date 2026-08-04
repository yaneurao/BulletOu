# 5.5 Continued training (after a clean finish)

<a href="../../ja/tutorial/5b-additional-training.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

[§5 Interrupt & resume](5-resume.md) covered the case where training stopped and you want to continue. This page covers the case where **training finished cleanly and you want to add more training**.

Examples:
- 3 epochs done → look at the results, want 3 more epochs.
- Trained at batch_size 16384 → realised the 4090 has VRAM headroom, want to switch to 32768.
- Have saved weights → want to continue with a **different teacher**.
- Want to lower the LR and polish for a bit.

## 5.5.1 The simple rule: same `--tag` resumes

Continued training works with the same auto-resume mechanism as §5: **same `--tag` → auto-resume, different `--tag` → fresh start.**

```powershell
# Round 1: train 3 epochs
.\bulletou.exe --teacher c:\shogi\teacher\... `
    --arch NNUE_kp_256x2_32_32 `
    --tag round1 --max-epochs 3 --superbatches 6 `
    --lr-schedule step --lr-min 0.00001

# Round 2: 3 more epochs
.\bulletou.exe --teacher c:\shogi\teacher\... `
    --arch NNUE_kp_256x2_32_32 `
    --tag round1 --max-epochs 3 --superbatches 6 `
    --lr-schedule step --lr-min 0.00001
```

On the second launch:
- Detects the output dir `checkpoints/NNUE_KP-NNUE_kp_256x2_32_32-round1/`.
- Loads the latest `0018/state.bin` (weights + Ranger optimizer state).
- New saves start at `0019/`.
- `summary-learn.log` keeps growing — cumulative across runs.

Total effective epochs trained: 3 + 3 = 6.

⚠️ `--max-epochs` means **"epochs to run in this launch"**, not "total epochs to reach."

## 5.5.2 What you can / can't change between launches

### ✅ Safe to change

| Flag | Notes |
|---|---|
| `--batch-size` | state.bin is batch-size-independent; Adam state is per-parameter, portable. |
| `--positions-per-superbatch` | Changes sb size. The effective value is rounded down to a multiple of `batch_size`. |
| `--lr` | Start LR for `step`; lr_max for `geometric` / `cos`. |
| `--lr-min` | LR floor. |
| `--lr-schedule` (`step` / `geometric` / `cos` / `plateau`) | Changes how LR moves. Default is `step`. |
| `--max-epochs` | How many epochs in this launch. |
| `--superbatches` | LR cycle length for `geometric` / `cos`; per-epoch processing cap for `step`. |
| `--save-rate` | Changes checkpoint frequency only. If reusing an existing checkpoint, pass `--resume` explicitly. |
| `--validation-rate` | Changes validation frequency only. If reusing saved data, pass `--resume` explicitly. |
| `--lambda` | Teacher-target blend. |
| `--teacher` | BulletOu detects the teacher change and reads the new teacher from the start. See the LR note in [§5.5.4](#554-continue-with-a-different-teacher). |
| `--test-teacher` | Validation set swap. |
| `--sfnn-factorizer` | Changes how SFNN shares common components between buckets. If reusing saved data, pass `--resume` explicitly. |

### ❌ Don't change (model topology)

| Flag | Why |
|---|---|
| `--arch` | Changes the evaluation function or layer sizes, so saved weights no longer match. |
| `--arch` LayerStack part, such as `k3k3` or `hand1024` | Changes the number of SFNN branches, so saved weights no longer match. |
| `--sfnn-factorized` / `--no-sfnn-factorized` | Short forms for `--sfnn-factorizer shared` / `--sfnn-factorizer none`. Prefer `--sfnn-factorizer`. |
| `--tag` | Changing this lands you in a different output dir = fresh training. (Useful only when starting a new experiment.) |

To change any of these, pass a different `--tag` and run as a separate experiment.

## 5.5.3 Example: bump batch_size from 16384 to 32768

```powershell
# Continue with 3 more epochs at the larger batch size
.\bulletou.exe --teacher c:\shogi\teacher\... `
    --arch NNUE_kp_256x2_32_32 `
    --tag round1 --max-epochs 3 --superbatches 6 `
    --batch-size 32768 `
    --lr-schedule step --lr-min 0.00001
```

### `positions-per-superbatch` and `batch-size`

`--positions-per-superbatch` is the target position count. The effective `sb_size` is `floor(positions_per_superbatch / batch_size) * batch_size`.

| batch_size | positions_per_superbatch | effective sb_size |
|---|---|---|
| 16384 | 100,000,000 | 99,991,552 |
| 32768 | 100,000,000 | 99,975,168 |
| 65536 | 100,000,000 | 99,942,400 |

Changing `--batch-size` can slightly change the rounded effective sb_size. If you need an exactly matched LR cycle length, set `--positions-per-superbatch` explicitly as well.

### Optimizer state is slightly out of sync

Changing `--batch-size` keeps the weights reusable, but the optimizer has internal statistics from earlier updates. The first 1–2 sb may look slightly different. If the loss recovers quickly, this is not a problem.

## 5.5.4 Continue with a different teacher

A common workflow: train on a large weaker corpus, then continue on a smaller stronger corpus:

```powershell
# Distill on bulk teacher for 3 epochs
.\bulletou.exe --teacher c:\shogi\teacher\bulk\ `
    --arch NNUE_kp_256x2_32_32 `
    --tag distill `
    --max-epochs 3 --superbatches 6 `
    --lr-schedule step --lr-min 0.00001

# Continue on strong teacher with a smaller LR
.\bulletou.exe --teacher c:\shogi\teacher\strong\ `
    --arch NNUE_kp_256x2_32_32 `
    --tag distill `
    --max-epochs 2 --superbatches 4 `
    --lr 0.0001 --lr-min 0.000001 `
    --lr-schedule step
```

Teacher-change handling:
- bulletou reads the last `teacher` column in `summary-learn.log` and notices the path differs.
- Prints a warning and re-opens the dataloader at the new teacher's head.
- Resets `dataloader_pos.txt`.
- Adjusts the displayed sb counter (`cb_ctx.sb_offset`) so the log row stays monotonic.

For `step` / `geometric` / `cos`, every epoch starts from `--lr`. When changing teachers, pass explicit `--lr` / `--lr-min` values as needed, and use a different `--tag` when you want a separate experiment.

## 5.5.5 Cooling down with a smaller LR

After a near-converged run, a final polish at 1/10 the LR is a classic move:

```powershell
# Initial run
.\bulletou.exe --teacher ... --tag main `
    --max-epochs 3 --superbatches 6 `
    --lr 0.001 --lr-min 0.00001 `
    --lr-schedule step ...

# Polish: 1 more epoch at 1/10 LR
.\bulletou.exe --teacher ... --tag main `
    --max-epochs 1 --superbatches 6 `
    --lr 0.0001 --lr-min 0.000001 `
    --lr-schedule step ...
```

This is a simple final low-LR polish: small final step, no big swings.

## 5.5.6 Multiple launches vs one long run

Splitting 6 epochs into two 3-epoch launches is **functionally close** to one 6-epoch launch:

| Aspect | 2 launches | 1 launch |
|---|---|---|
| Total weight updates | Same | Same |
| LR cycles | `step` / `geometric` / `cos` restart from `--lr` at each epoch. | Same |
| CUDA initialization | Happens for each launch | Once |
| Intermediate checkpoints | Same (per-sb save) | Same |
| Interruption tolerance | Higher (each launch ends cleanly) | One long process |

In practice, 2–3 epoch chunks are often convenient: you get regular points to inspect results, adjust settings, or stop cleanly.

---

Next: [6. Adjust training settings](6-tune.md) — what `--lr` / `--superbatches` / `--lambda` actually mean.

Previous: [5. Interrupt & resume](5-resume.md)
