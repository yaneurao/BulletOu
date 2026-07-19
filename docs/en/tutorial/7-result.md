# 7. Inspect the result — output files and training log

<a href="../../ja/tutorial/7-result.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

What to do after (or during) training:
- check what's in the output directory
- read `learn.log` to confirm training is healthy

(Loading the trained eval into a YaneuraOu engine is covered in [8. Load into an engine](8-engine.md).)

## 7.1 Inspect the output

After training finishes the output directory (e.g. `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/`) has the following layout:

```
checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/
├── summary-learn.log                  ← top-level cumulative sb-level log across runs/resumes
├── 0001/
│   ├── nn.bin                         ← YaneuraOu / Stockfish (nnue-pytorch) compatible NNUE binary
│   ├── state.bin                      ← resume data (weights + Ranger optimizer state)
│   └── learn.log                      ← snapshot of the training log at this save point
├── 0002/
├── ...
└── 000N/                              ← the most recent save
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

`000N/` (the highest-numbered dir) holds the artefacts to hand to the engine.

For KPPT / KPP_KKPT, instead of `nn.bin` each numbered dir contains the three files `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` (all three are required together).

## 7.2 Reading the training log (`learn.log`)

The loss trajectory of every run, both during training and afterwards, is recorded in `<output>/summary-learn.log` (cumulative) and `<output>/0NNN/learn.log` (per-save snapshot). The files have different granularities: `summary-learn.log` has one row per validated/saved superbatch, while each `0NNN/learn.log` is a per-batch snapshot for a saved checkpoint.

`--validation-rate` controls how often held-out accuracy/loss are computed. It defaults to `--save-rate`, but you can set a smaller value (for example `--validation-rate 1 --save-rate 20`) to validate every superbatch while saving checkpoints less often. Validation-only summary rows do not have a matching numbered checkpoint directory; if training is interrupted, rows after the latest complete checkpoint are trimmed on resume because that unsaved model state cannot be restored.

### Which one to look at

- **Top-level `<output>/summary-learn.log`** — the **cumulative** file across all runs/resumes. Use this as the default.
- **Per-save `0NNN/learn.log`** — a snapshot up to that save point. Use this when you want to see "what did things look like at save 0005?".

The per-save `learn.log` keeps the 12-column per-batch schema shown below. The top-level `summary-learn.log` omits `curr_batch` and appends a rightmost `test_teacher` column so each validation accuracy/loss row records which `--test-teacher` file produced it.

### Sample CSV

```csv
eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,32,-,-,0.6234,0.001000,0.000999,1.000000,2097152,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,64,-,-,0.5891,0.000999,0.000998,1.000000,4194304,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,96,-,-,0.5510,0.000998,0.000997,1.000000,6291456,teachers/
...
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,2,32,-,-,0.4523,0.000934,0.000933,1.000000,102039552,teachers/
...
```

Bullet writes **one row every 32 batches**. With the default (`--positions-per-superbatch 100000000`, omitted `--batch-size` = 65536), the effective superbatch is 1525 batches (= 99,942,400 positions), so that's about 48 rows per superbatch. If you explicitly pass `--batch-size 16384`, the effective superbatch is 6103 batches (= 99,991,552 positions), or about 191 rows. Once `curr_batch` reaches the final batch in the effective superbatch, `superbatch` increments by 1 and `curr_batch` restarts from 1.

### Column meanings

| Column | Meaning | Example |
|---|---|---|
| `eval` | mirror of the output-dir name (`<target>[-<arch>]`) plus a `/<component>` suffix for multi-component (KPPT-family) rows | `NNUE_HALFKP-NNUE_halfkp_256x2_32_32` / `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` |
| `epoch` | within-run epoch (1-indexed) | `1` |
| `superbatch` | within-epoch superbatch (1-indexed). +1 every effective `--positions-per-superbatch` positions | `1`, `2`, ... |
| `curr_batch` | within-superbatch batch (1-indexed). Bullet logs every 32 batches | `32`, `64`, ..., `1525` |
| `test_value_accuracy` | Validation accuracy from `--test-teacher`. Filled only on sb-boundary rows; otherwise `-` | `0.583784` |
| `test_value_loss` | Validation loss from `--test-teacher`. Filled only on sb-boundary rows; otherwise `-` | `0.129676` |
| `train_value_loss` | bullet's per-32-batch averaged loss | `0.234` |
| `lr_start` | LR at the start of this row's interval. In summary rows, the superbatch start LR | `0.001000` |
| `lr_end` | LR used by the last batch in this row's interval. In summary rows, the superbatch-end-side LR | `0.000934` |
| `lambda` | the `--lambda` value (constant per run, fixed 6-decimal) | `1.000000` |
| `positions` | cumulative teacher positions (**carries across resumes**) | `2097152` |
| `teacher` | the `--teacher` value | `teachers/` |
| `test_teacher` | top-level `summary-learn.log` only: the `--test-teacher` filename used for validation, or `-` when unset | `test.hcpe` |

NNUE/SFNN targets embed `--arch` in the `eval` column (matching the output-dir name). KPPT-family targets use the fixed `KPPT` / `KPP_KKPT` names, so the column is just `<target>/<component>`.

Full spec: [`spec/04-checkpoint-layout.md`](../../spec/04-checkpoint-layout.md#learnlog-フォーマット).

### Read with pandas

```python
import pandas as pd

df = pd.read_csv("checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/summary-learn.log")
print(df.shape)        # total rows
print(df.tail())       # last few rows
print(df["train_value_loss"].describe())   # loss stats
```

The CSV header gives pandas the column names automatically.

### Sanity-check list

A healthy training run typically shows:

1. **`train_value_loss` is monotonically decreasing (roughly)**
   - Drops sharply at first, then slowly tapers
   - You should see a visible drop per superbatch consumed
   - No drop after a full superbatch ⇒ `--lr` may be too large, or the teacher is too small for the model
   - **Periodic loss spikes** or local loss bias almost always mean the teacher file wasn't pre-shuffled. BulletOu does not shuffle teacher positions during training. Fix: see [§3.2 Pre-shuffle the teacher file](3-data.md#pre-shuffle-the-teacher-file)

2. **`lr_start` / `lr_end` follow the configured schedule**
   - `--lr-schedule step` (default): multiply lr by `gamma` once per superbatch, floor at `--lr-min`, and restart to `--lr` at epoch boundaries. `gamma` is either explicit or auto-computed from one epoch's length.
   - `--lr-schedule geometric`: geometric (= log-linear) decay from `--lr` (lr_max) to `--lr-min` over one epoch (= `--superbatches × sb_size` positions), warm-restarting back to lr_max at each epoch boundary.
   - `--lr-schedule cos`: cosine annealing sweeping `--lr` (lr_max) → `--lr-min` over one epoch (= `--superbatches × sb_size` positions), then warm-restarts to `--lr` at each epoch boundary.
   - If it isn't moving as expected, double-check the LR flags ([§6.1 Training schedule](6-tune.md#61-training-schedule)).

3. **`positions` is monotonically increasing** (within a run and across resumes)
   - One completed superbatch ≒ 100M (= `--positions-per-superbatch` rounded down to a multiple of `--batch-size`)
   - Cross-check against your teacher size to confirm "is all of the teacher being consumed?"

4. **`superbatch` advances as expected**
   - With a teacher smaller than 100M positions, `superbatch` stays at 1 for the whole run (fallback save fires once at the end). That's by design.
   - With a larger teacher, `superbatch` should increment every time `curr_batch` reaches the final batch in the effective superbatch.
   - If `superbatch` is stuck at 1 and `curr_batch` plateaus far below the final batch in the effective superbatch, the dataloader may be cut short (e.g. the old HCPE polarity bug).

### Quick plot

```python
import matplotlib.pyplot as plt

# positions as the time axis
plt.figure(figsize=(12, 4))
plt.plot(df["positions"], df["train_value_loss"])
plt.xlabel("positions"); plt.ylabel("train_value_loss")
plt.title("training loss curve")
plt.savefig("loss_curve.png")
```

### KPPT case (kk / kkp / kpp recorded side-by-side)

For KPPT-family eval types, each save writes kk → kkp → kpp logs back-to-back. Their rows share `(epoch, superbatch, curr_batch, positions)` but differ in the component portion of the `eval` column (`KPPT/kk` / `KPPT/kkp` / `KPPT/kpp`), so filter before plotting:

```python
for c in ["kk", "kkp", "kpp"]:
    sub = df[df["eval"] == f"KPPT/{c}"]
    plt.plot(sub["positions"], sub["value_loss"], label=c)
plt.legend(); plt.xlabel("positions"); plt.ylabel("loss")
```

The KK component is a much smaller network than KKP / KPP, so don't compare absolute loss values — look at the **trend per component**.

To split the `eval` column into family / component as separate columns:

```python
df[["family", "component"]] = df["eval"].str.split("/", n=1, expand=True)
df["component"] = df["component"].fillna("nnue")   # NNUE rows have no slash
```

### Reading the log after a resume

After a resume, the new run's rows are appended verbatim. In the new run:
- `epoch` restarts at 1
- `superbatch` restarts at 1
- **`positions` continues from the previous run's max**

Using `positions` as the time axis gives you a continuous loss curve across resume boundaries (the plotting snippet above already does this).

`epoch` / `superbatch` are within-run counters, so the same numbers appear multiple times after a resume. To find a run boundary, look for a row where `(epoch, superbatch)` resets back to a low value while `positions` keeps going up.

## 7.3 Where to go next

- [8. Load into an engine](8-engine.md) — verify the trained weights in a YaneuraOu engine
- [Reference: NNUE HalfKP Training](../shogi/halfkp.md) — `nn.bin` binary layout, quantisation, resume details
- [Reference: NNUE K-P Training](../shogi/kp.md) — comparison vs HalfKP, input feature structure
- [Reference: NNUE HalfKPE9 Training](../shogi/halfkpe9.md) — HalfKP with attacker-count buckets
- [Reference: KPPT / KPP_KKPT Training](../shogi/kppt.md) — legacy YaneuraOu evals
- [Specifications: spec/](../../spec/) — target matrix, binary layout, hash derivations, `learn.log` format

---

Previous: [6. Tune the training](6-tune.md)
