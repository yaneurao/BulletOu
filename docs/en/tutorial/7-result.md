# 7. Inspect the result — output files and training log

<a href="../../ja/tutorial/7-result.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

What to do after (or during) training:
- check what's in the output directory
- read `learn.log` to confirm training is healthy

(Loading the trained eval into a YaneuraOu engine is covered in [8. Load into an engine](8-engine.md).)

## 7.1 Inspect the output

After training finishes the output directory (e.g. `checkpoints/NNUE_HALFKP-256x2-32-32/`) has the following layout:

```
checkpoints/NNUE_HALFKP-256x2-32-32/
├── learn.log                          ← top-level cumulative log across runs/resumes
├── 0001/
│   ├── nn.bin                         ← YaneuraOu / Stockfish (nnue-pytorch) compatible NNUE binary
│   ├── state.bin                      ← resume data (weights + Adam moments)
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

The loss trajectory of every run, both during training and afterwards, is recorded in `<output>/learn.log` (cumulative) and `<output>/0NNN/learn.log` (per-save snapshot). Both are the **same 9-column CSV** format.

### Which one to look at

- **Top-level `<output>/learn.log`** — the **cumulative** file across all runs/resumes. Use this as the default.
- **Per-save `0NNN/learn.log`** — a snapshot up to that save point. Use this when you want to see "what did things look like at save 0005?".

### Sample CSV

```csv
eval,epoch,superbatch,curr_batch,value_loss,lr,lambda,positions,teacher
NNUE_HALFKP-256x2-32-32,1,1,32,0.6234,0.001,1.000,524288,teachers/
NNUE_HALFKP-256x2-32-32,1,1,64,0.5891,0.001,1.000,1048576,teachers/
NNUE_HALFKP-256x2-32-32,1,1,96,0.5510,0.001,1.000,1572864,teachers/
...
NNUE_HALFKP-256x2-32-32,1,2,32,0.4523,0.001,1.000,100532224,teachers/
...
```

Bullet writes **one row every 32 batches**. With the default `--batches-per-superbatch ≒ 6104`, that's about 191 rows per superbatch. Once `curr_batch` reaches `batches_per_superbatch` (default 6104), `superbatch` increments by 1 and `curr_batch` restarts from 1.

### Column meanings

| Column | Meaning | Example |
|---|---|---|
| `eval` | mirror of the output-dir name (`<eval-type>[-<arch>]`) plus a `/<component>` suffix for multi-component (KPPT-family) rows | `NNUE_HALFKP-256x2-32-32` / `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` |
| `epoch` | within-run epoch (1-indexed) | `1` |
| `superbatch` | within-epoch superbatch (1-indexed). +1 every `--batches-per-superbatch` (default 6104) batches | `1`, `2`, ... |
| `curr_batch` | within-superbatch batch (1-indexed). Bullet logs every 32 batches | `32`, `64`, ..., `6104` |
| `value_loss` | bullet's per-32-batch averaged loss | `0.234` |
| `lr` | learning rate at that point (StepLR-derived) | `0.001` |
| `lambda` | the `--lambda` value (constant per run, fixed 3-decimal) | `1.000` |
| `positions` | cumulative teacher positions (**carries across resumes**) | `524288` |
| `teacher` | the `--teacher` value | `teachers/` |

NNUE eval types embed `--arch` in the `eval` column (matching the output-dir name). KPPT-family eval types don't consume `--arch`, so the column is just `<eval-type>/<component>`.

Full spec: [`spec/04-checkpoint-layout.md`](../../spec/04-checkpoint-layout.md#learnlog-フォーマット).

### Read with pandas

```python
import pandas as pd

df = pd.read_csv("checkpoints/NNUE_HALFKP-256x2-32-32/learn.log")
print(df.shape)        # total rows
print(df.tail())       # last few rows
print(df["value_loss"].describe())   # loss stats
```

The CSV header gives pandas the column names automatically.

### Sanity-check list

A healthy training run typically shows:

1. **`value_loss` is monotonically decreasing (roughly)**
   - Drops sharply at first, then slowly tapers
   - You should see a visible drop per superbatch consumed
   - No drop after a full superbatch ⇒ `--lr` may be too large, or the teacher is too small for the model
   - **Periodic loss spikes** (jumping sharply every few hundred batches) almost always mean the teacher file wasn't pre-shuffled. The shuffle buffer crosses a region boundary (default 256MB buffer ≒ every ~410 batches), and the distribution shifts. Fix: see [§3.2 Pre-shuffle the teacher file](3-data.md#pre-shuffle-the-teacher-file)

2. **`lr` follows the configured schedule**
   - `--lr-schedule step` (default): geometric (= log-linear) decay from `--lr` (lr_max) to `--lr-min` over one epoch (= `--superbatches × sb_size` positions), warm-restarting back to lr_max at each epoch boundary.
   - `--lr-schedule cos`: cosine annealing sweeping `--lr` (lr_max) → `--lr-min` over one epoch (= `--superbatches × sb_size` positions), then warm-restarts to `--lr` at each epoch boundary.
   - If it isn't moving as expected, double-check the LR flags ([§6.1 Training schedule](6-tune.md#61-training-schedule)).

3. **`positions` is monotonically increasing** (within a run and across resumes)
   - One completed superbatch ≒ 100M (= `--batches-per-superbatch × --batch-size`)
   - Cross-check against your teacher size to confirm "is all of the teacher being consumed?"

4. **`superbatch` advances as expected**
   - With a teacher smaller than 100M positions, `superbatch` stays at 1 for the whole run (fallback save fires once at the end). That's by design.
   - With a larger teacher, `superbatch` should increment every time `curr_batch` reaches 6104 (= `--batches-per-superbatch` default).
   - If `superbatch` is stuck at 1 and `curr_batch` plateaus far below `batches_per_superbatch`, the dataloader may be cut short (e.g. the old HCPE polarity bug).

### Quick plot

```python
import matplotlib.pyplot as plt

# positions as the time axis
plt.figure(figsize=(12, 4))
plt.plot(df["positions"], df["value_loss"])
plt.xlabel("positions"); plt.ylabel("value_loss")
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
- [Specifications: spec/](../../spec/) — eval-type matrix, binary layout, hash derivations, `learn.log` format

---

Previous: [6. Tune the training](6-tune.md)
