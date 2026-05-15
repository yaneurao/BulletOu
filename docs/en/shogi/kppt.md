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
./target/release/examples/bulletou \
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

`learn.log` is a 9-column CSV with a header row, the same format used by every eval-type:

```
eval,epoch,superbatch,curr_batch,value_loss,lr,lambda,positions,teacher
KPPT/kk,1,1,32,0.234,0.001,1.000,524288,teachers/
KPPT/kk,1,1,64,0.232,0.001,1.000,1048576,teachers/
...
KPPT/kkp,1,1,32,0.156,0.001,1.000,524288,teachers/
...
KPPT/kpp,1,1,32,0.245,0.001,1.000,524288,teachers/
...
```

The `eval` column uses the **`<eval-type>/<component>`** format, which distinguishes the kk / kkp / kpp components for KPPT-family rows. KPPT-family eval types ignore `--arch`, so no arch suffix appears here (unlike NNUE rows, which embed it as `NNUE_HALFKP-256x2-32-32`).

Per-save snapshot `0NNN/learn.log` and the top-level `<output>/learn.log` use the exact same format. The top-level accumulates rows across resumes; `positions` is cumulative across resumes (the start of a resumed run picks up from the previous run's max positions per component). The columns are described in detail in [`spec/04-checkpoint-layout.md`](../../spec/04-checkpoint-layout.md).

Point a YaneuraOu KPPT engine at the latest numbered directory (`000N/`). The engine ignores `state.bin`.

Resume / restart behaviour is identical across every eval-type; see [tutorial 5. Stop and resume](../tutorial/5-resume.md) for details.

### KPP_KKPT (factorised)

`--eval-type KPP_KKPT` produces a factorised eval (KPP without the turn channel, ~half the KPP file size). KK and KKP files are byte-identical to KPPT.

```bash
./target/release/examples/bulletou \
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
| `--lr` / `--lr-schedule` / `--lr-gamma` / `--lr-step-positions` | LR scheduler (`step` or `cos`, full details in [§6.1](../tutorial/6-tune.md#61-training-schedule)) | 0.001 / `step` / 0.9 / 100000000 |
| `--lambda` | Blend weight between teacher eval and WDL (= Win/Draw/Loss game-result label). Matches YaneuraOu's `lambda` convention: `λ × teacher_eval + (1−λ) × game_result`. `λ=1.0` is pure eval, `λ=0.0` is pure WDL | 1.0 |
| `--yaneuraou-quant-scale` | f32 → i{16,32} quantisation scale | 4000 (KK/KKP), 400 (KPP) |
| `--score-drop-abs` | Drop positions where `|score| >= N` (mate-stamp filter) | 32000 |

For the meaning of the scheduling units, see [§6.1 Training schedule](../tutorial/6-tune.md#61-training-schedule).

## Memory requirements

KK and KKP training are tiny on GPU memory (any 4 GB+ card is plenty).

**KPP uses roughly 2.3 GB of GPU memory**, so an **8 GB+ GPU is recommended**.

## Hyperparameter guidance

KPPT historically uses:

- elmo-style WDL teaching (`--lambda 0.5` or so, blending eval and game result 50:50)
- stronger weight decay
- smaller learning rate (`--lr 1e-4` to `1e-3`)

`bulletou`'s default is pure eval (`--lambda 1.0`, `--lr 1e-3`). For production-quality KPPT, adjust `--lambda` and the learning rate along the above lines.

## Related

- [BulletOu source: `crates/bulletou_lib/src/game/inputs/shogi_kk.rs`](../../../crates/bulletou_lib/src/game/inputs/shogi_kk.rs)
- [BulletOu source: `crates/bulletou_lib/src/game/inputs/shogi_kkp.rs`](../../../crates/bulletou_lib/src/game/inputs/shogi_kkp.rs)
- [BulletOu source: `crates/bulletou_lib/src/game/inputs/shogi_kpp.rs`](../../../crates/bulletou_lib/src/game/inputs/shogi_kpp.rs)
- [BulletOu source: `crates/bulletou_lib/src/value/yaneuraou_kppt.rs`](../../../crates/bulletou_lib/src/value/yaneuraou_kppt.rs)
- [YaneuraOu source: `source/eval/kppt/evaluate_kppt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kppt/evaluate_kppt.h)
- [YaneuraOu source: `source/eval/kpp_kkpt/evaluate_kpp_kkpt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kpp_kkpt/evaluate_kpp_kkpt.h)
