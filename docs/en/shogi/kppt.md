# KPPT / KPP_KKPT Training

<a href="../../ja/shogi/kppt.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

BulletOu trains YaneuraOu's legacy KPPT-family evaluation functions and writes the corresponding three-file binary set (`KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin`). This page describes how to use it.

## Why support KPPT / KPP_KKPT at all

YaneuraOu has a family of older (pre-NNUE) evaluation functions:

- **KK** — king vs king
- **KKP** — king × king × piece
- **KPP** — king × piece × piece (the Apery / Bonanza original)
- **KPPT** — KPP + turn-tensor T (= with a side-to-move channel)
- **KPP_KKPT** — KPPT factorised: KPP without turn, the turn term lives in KK and KKP

Still useful today:
- Improve / re-train classical evals as a research baseline.
- Take advantage of BulletOu's GPU pipeline -- this was historically CPU-only and very slow (**100x+ speed-up** for training).
- Run apples-to-apples comparisons between classical and NNUE evals on the same data.
- Reproduce historically important evals such as elmo(WCSC27).

## Structural difference from NNUE

NNUE is a "sparse feature transformer + small MLP" — a standard NN shape. KPPT is a different family: **no hidden layer at all, just a sum of three huge sparse lookup tables (KK / KKP / KPP)** for the evaluation score.

## Output files

Training writes a three-file set into the checkpoint directory:

| File | Size |
|---|---|
| `KK_synthesized.bin` | 51 KB |
| `KKP_synthesized.bin` | 77 MB |
| `KPP_synthesized.bin` (KPPT) | 740 MB |
| `KPP_synthesized.bin` (KPP_KKPT) | 388 MB |

KPP_KKPT is the factorised variant of KPPT: the KPP file drops the turn channel, halving its size. KK and KKP files are identical between the two families.

## Actual usage

### Prerequisites

- BulletOu built (`cargo build --release --features device-cuda --example bulletou`)
- Training data (`.hcpe` / `.hcpe3` / `.pack`)
- 4 GB+ of free GPU memory (KPP training uses ~2.3 GB)

### KPPT (elmo-compatible)

`--eval-type KPPT` trains all three components (KK / KKP / KPP) in one invocation and assembles the three resulting `.bin` files into `<output>/final/`:

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type KPPT \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-kppt
```

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader returns EOF). To run multiple passes, pass `--max-epochs N` — the LR scheduler restarts at the beginning of each epoch.

When training finishes, the saved checkpoints are laid out as zero-padded numbered directories, one per save point, each containing the three `.bin` files:

```
checkpoints/my-kppt/
├── learn.log                          ← top-level cumulative log across all runs/resumes
├── 0001/
│   ├── KK_synthesized.bin
│   ├── KKP_synthesized.bin
│   ├── KPP_synthesized.bin
│   ├── state.bin                      ← resume data (weights + Adam moments for all 3 components)
│   └── learn.log                      ← snapshot of the training log at this save point
├── 0002/
│   ├── ...
├── ...
└── 000N/                              ← the most recent save
    ├── KK_synthesized.bin
    ├── KKP_synthesized.bin
    ├── KPP_synthesized.bin
    ├── state.bin
    └── learn.log
```

`learn.log` format (per-save snapshot): bullet's per-component `log.txt` files concatenated with section headers:

```
# component: kk
1,32,0.234
1,64,0.231
...
# component: kkp
1,32,0.156
...
# component: kpp
1,32,0.245
...
```

Each row is `<superbatch>,<curr_batch>,<loss>` CSV. Bullet writes one row every 32 batches.

The top-level `<output>/learn.log` accumulates one section per run, prefixed with a header that records the wall-clock time and the range of numbered dirs produced:

```
# === run @ 2026-05-12T15:30:00Z saved 0001/-0005/ ===
# component: kk
1,32,0.234
...
# === run @ 2026-05-12T18:42:00Z saved 0006/-0010/ ===
# component: kk
1,32,0.118
...
```

On resume the superbatch counter restarts at 1 (the LR scheduler restarts each run); each section's header tells you which run the rows belong to.

Point a YaneuraOu KPPT engine at the latest numbered directory (`000N/`). The engine ignores `state.bin`.

### Resume

If `--output` already contains numbered dirs with `state.bin`, `bulletou` automatically resumes from the latest one. New saves continue the numbering (e.g. if the previous run produced `0001/`..`0005/`, the resumed run writes `0006/`, `0007/`, ...).

Just re-running the same command picks up where it left off. To start fresh, point `--output` at a different directory or delete the existing one.

### KPP_KKPT (factorised)

`--eval-type KPP_KKPT` produces a factorised eval (KPP without the turn channel, ~half the KPP file size). KK and KKP files are byte-identical to KPPT.

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type KPP_KKPT \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --superbatches 20
```

### Common KPPT-family CLI flags

| Flag | Meaning | Default |
|---|---|---|
| `--eval-type` | `KPPT` (3-component sequential) / `KPP_KKPT` (factorised) | (required) |
| `--teacher` | Teacher file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), a directory of such files, or comma-separated combination | (required) |
| `--output` | Checkpoint parent directory | `checkpoints/<eval-type>` (e.g. `checkpoints/KPPT`, `checkpoints/KPP_KKPT`) |
| `--net-id` | Prefix of the saved checkpoint subdirectory name | per-eval-type default |
| `--batch-size` | Positions per gradient step | 16384 |
| `--batches-per-superbatch` | Mini-batches per superbatch | `ceil(100M / batch-size)` |
| `--superbatches` | Cap on superbatches per epoch. Omit for no cap (run until dataloader EOF) | (no cap) |
| `--max-epochs` | Number of epochs to run (= dataloader EOFs). LR scheduler restarts at the start of each epoch | 1 |
| `--save-rate` | Save every N superbatches | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR scheduler | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL linear interpolation | 0.0 / 1.0 |
| `--yaneuraou-quant-scale` | f32 → i{16,32} quantisation scale | 4000 (KK/KKP), 400 (KPP) |
| `--score-drop-abs` | Drop positions where `|score| >= N` (mate-stamp filter) | 32000 |

For the meaning of the scheduling units, see [§2.4 Training schedule](../tutorial/2-nnue-tutorial.md#24-training-schedule).

## Memory requirements

KK and KKP training are tiny on GPU memory (any 4 GB+ card is plenty).

**KPP uses roughly 2.3 GB of GPU memory**, so an **8 GB+ GPU is recommended**.

## Hyperparameter guidance

KPPT historically uses:

- elmo-style WDL teaching (`--start-wdl 0.5 --end-wdl 0.5`, mid-range)
- stronger weight decay
- smaller learning rate (`--lr 1e-4` to `1e-3`)

`bulletou`'s defaults are NNUE-oriented (`--start-wdl 0.0 --end-wdl 1.0`, `--lr 1e-3`). For production-quality KPPT, adjust WDL and learning rate along the above lines.

## Related

- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kk.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kk.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kkp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kkp.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kpp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kpp.rs)
- [BulletOu source: `crates/bullet_lib/src/value/yaneuraou_kppt.rs`](../../../crates/bullet_lib/src/value/yaneuraou_kppt.rs)
- [YaneuraOu source: `source/eval/kppt/evaluate_kppt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kppt/evaluate_kppt.h)
- [YaneuraOu source: `source/eval/kpp_kkpt/evaluate_kpp_kkpt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kpp_kkpt/evaluate_kpp_kkpt.h)
