# 3. KPPT / KPP_KKPT Training

<a href="../../ja/tutorial/3-kppt-roadmap.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

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

NNUE is a "**sparse feature transformer + small MLP**" -- a standard NN shape.

KPPT is "**a sum of huge sparse embedding tables, no hidden layer**":

```
eval(pos) = KK[bk][wk]
          + Σ_i KKP[bk][wk][p_i]
          + Σ_{i<j} KPP[bk][p_i][p_j]
          + (turn term T)
```

No hidden layer in the NN sense -- just a sum of large lookup tables.

The biggest table (`KPP`) is `81 × 1548 × 1548 = 194,100,624` dims = 776 MB at f32, ~2.3 GB on GPU including Adam state.

## File formats (confirmed from YaneuraOu source)

From `source/eval/kppt/evaluate_kppt.h` and `eval/kpp_kkpt/evaluate_kpp_kkpt.h`:

| File | KPPT type | KPP_KKPT type | Size |
|---|---|---|---|
| `KK_synthesized.bin` | `int32_t kk[81][81][2]` | identical | 51 KB |
| `KKP_synthesized.bin` | `int32_t kkp[81][81][1548][2]` | identical | 77 MB |
| `KPP_synthesized.bin` | `int16_t kpp[81][1548][1548][2]` | `int16_t kpp[81][1548][1548]` | **740 MB / 388 MB** |

The trailing `[2]` is `[stm_independent, stm_dependent]` (turn-independent + turn-dependent term).
- **KPPT**: KPP has a turn channel.
- **KPP_KKPT**: KPP has *no* turn channel; the turn term lives in KK and KKP only.

BulletOu trains **only `[0]` (the turn-independent term)**; `[1]` (the turn-dependent term) is written as 0.

## Actual usage

### Prerequisites

- BulletOu built (`cargo build --release --features cuda --example bullet_ou_train`)
- Training data (`.hcpe` / `.hcpe3` / `.pack`)
- 4 GB+ of free GPU memory (KPP training uses ~2.3 GB)

### KPPT (elmo-compatible, `int16_t × 2` KPP)

Train the three components (KK / KKP / KPP) independently:

```bash
# KK training -> KK_synthesized.bin
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kk \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kppt \
    --net-id kk \
    --superbatches 20

# KKP training -> KKP_synthesized.bin
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kkp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kppt \
    --net-id kkp \
    --superbatches 20

# KPP training -> KPP_synthesized.bin (KPPT layout)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kpp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kppt \
    --net-id kpp \
    --superbatches 20
```

Three `.bin` files have now been written; assemble them:

```bash
mkdir -p checkpoints/my-kppt/final
cp checkpoints/my-kppt/kk-20/KK_synthesized.bin   checkpoints/my-kppt/final/
cp checkpoints/my-kppt/kkp-20/KKP_synthesized.bin checkpoints/my-kppt/final/
cp checkpoints/my-kppt/kpp-20/KPP_synthesized.bin checkpoints/my-kppt/final/
```

Point a YaneuraOu KPPT engine at `checkpoints/my-kppt/final/`.

### KPP_KKPT (factorised, `int16_t × 1` KPP)

KK and KKP files are byte-identical to KPPT, so the first two commands are the **same**. Only the KPP writer differs:

```bash
# KK training (identical to KPPT)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kk \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --net-id kk

# KKP training (identical to KPPT)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kkp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --net-id kkp

# KPP training, KPP_KKPT layout (no turn channel, half the size)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kpp-kkpt-kpp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --net-id kpp
```

Assemble the three files the same way as the KPPT case.

### Training a single component standalone

For smoke-testing or development, three single-component examples are also available; they contain the same logic `bullet_ou_train` dispatches to:

```bash
cargo run --release --features cuda --example shogi_kpp_train -- \
    --data inbox/ref/small.hcpe \
    --output checkpoints/kpp-smoke \
    --superbatches 3 \
    --batches-per-superbatch 100
```

### Common KPPT-family CLI flags

| Flag | Meaning | Default |
|---|---|---|
| `--eval-type` | `kppt-kk` / `kppt-kkp` / `kppt-kpp` / `kpp-kkpt-kpp` | (required) |
| `--data` | Training file (`.hcpe` / `.hcpe3` / `.pack`; comma-separated for multiple) | (required) |
| `--output` | Checkpoint parent directory | per-eval-type default |
| `--net-id` | Prefix of the saved checkpoint subdirectory name | per-eval-type default |
| `--batch-size` | Positions per gradient step | 16384 |
| `--batches-per-superbatch` | Mini-batches per superbatch | `ceil(100M / batch-size)` |
| `--superbatches` | Total superbatches | 10 |
| `--save-rate` | Save every N superbatches | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR scheduler | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL linear interpolation | 0.0 / 1.0 |
| `--yaneuraou-quant-scale` | f32 → i{16,32} quantisation scale | 4000 (KK/KKP), 400 (KPP) |
| `--score-drop-abs` | Drop positions where `|score| >= N` (mate-stamp filter) | 32000 |

For the meaning of the scheduling units, see [2.4 Training units](2-nnue-tutorial.md#24-training-units--batch--superbatch--save--lr).

## Memory requirements

| Component | Weights | f32 weights | + Adam (3× state) | Suggested GPU mem |
|---|---|---|---|---|
| KK | 6,561 | 26 KB | 78 KB | almost anything |
| KKP | 10,156,428 | 40 MB | 120 MB | 4 GB+ |
| KPP | 194,100,624 | 776 MB | 2.33 GB | **8 GB+ recommended** (sparse buffers add ~100 MB more) |

`max_active = 703` for KPP (= C(38, 2), the unordered pair count of non-king BonaPieces), so at `batch_size = 16384` the GPU-side sparse index buffer is ~92 MB.

## Hyperparameter guidance

KPPT historically uses:

- elmo-style WDL teaching (`--start-wdl 0.5 --end-wdl 0.5`, mid-range)
- stronger weight decay
- smaller learning rate (`--lr 1e-4` to `1e-3`)

`bullet_ou_train`'s defaults are NNUE-oriented (`--start-wdl 0.0 --end-wdl 1.0`, `--lr 1e-3`). For production-quality KPPT, adjust WDL and learning rate along the above lines.

## Related

- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kk.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kk.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kkp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kkp.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kpp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kpp.rs)
- [BulletOu source: `crates/bullet_lib/src/value/yaneuraou_kppt.rs`](../../../crates/bullet_lib/src/value/yaneuraou_kppt.rs)
- [YaneuraOu source: `source/eval/kppt/evaluate_kppt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kppt/evaluate_kppt.h)
- [YaneuraOu source: `source/eval/kpp_kkpt/evaluate_kpp_kkpt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kpp_kkpt/evaluate_kpp_kkpt.h)
