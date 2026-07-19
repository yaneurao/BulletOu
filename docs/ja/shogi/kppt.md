# KPPT / KPP_KKPT 学習

<a href="../../en/shogi/kppt.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

BulletOu はやねうら王の旧 KPPT 系評価関数 (`KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` の 3 ファイル組) を学習・出力できる。本ページではその使い方を説明する。

## なぜ KPPT / KPP_KKPT に対応するのか

やねうら王には NNUE 以前から旧来の評価関数系列がある:

- **KK** — 玉 × 玉
- **KKP** — 玉 × 玉 × 駒
- **KPP** — 玉 × 駒 × 駒 (Apery / Bonanza 流の元祖)
- **KPPT** — KPP + 手番テンソル T (= 手番チャンネル付き)
- **KPP_KKPT** — KPPT の factorise 版 (KPP は手番なし、手番項は KK / KKP 側に factorise)

これらは今でも価値がある:
- 古典評価関数を改良・再学習し、研究ベースラインにする
- BulletOu の GPU パイプラインで、歴史的に CPU 専用でとても遅かった学習を加速 (CPU → GPU で **100 倍以上**の高速化)
- 同じ学習データで古典評価関数と NNUE を比較する
- elmo(WCSC27) などの歴史的に重要な評価関数を再現

## NNUE との構造の違い

NNUE は「疎特徴量変換器 + 小さい MLP」という普通の NN 形状。一方 KPPT は **隠れ層を持たず、巨大な疎ルックアップテーブル (KK / KKP / KPP) の和だけ**で評価値を作る、別系統の評価関数。

## 出力ファイル

学習後、checkpoint ディレクトリ配下に 3 ファイル組が書き出される:

| ファイル | サイズ |
|---|---|
| `KK_synthesized.bin` | 51 KB |
| `KKP_synthesized.bin` | 77 MB |
| `KPP_synthesized.bin` (KPPT) | 740 MB |
| `KPP_synthesized.bin` (KPP_KKPT) | 388 MB |

KPP_KKPT は KPPT の factorise 版で、KPP ファイルから手番チャンネルを省いたぶんサイズが半分になっている。KK と KKP は両者で同じ。

## 実際の使い方

### 必要なもの

- BulletOu をビルド済み (`cargo build --release --features cuda-cpp-backend --example bulletou`)
- 学習データ (`.hcpe` / `.hcpe3` / `.pack` のいずれか)
- 4 GB+ の空き GPU メモリ (KPP 学習は ~2.3 GB を使う)

### KPPT (elmo 互換)

`--arch KPPT` を指定すると KK / KKP / KPP の 3 component を **1 コマンドで連続学習** し、各 save を `<output>/0001/`, `<output>/0002/`, ... の番号付き checkpoint directory に集約する。

```bash
./target/release/examples/bulletou \
    --arch KPPT \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-kppt
```

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--superbatches` で epoch 長を決めたうえで `--max-epochs 3` のように指定する。`step` / `geometric` / `cos` は epoch 境界で `--lr` に戻る。

完了すると、各 save 単位で `0001/`, `0002/`, ... と 4 桁番号のディレクトリが並び、それぞれに 3 ファイルが入る:

```
checkpoints/my-kppt/
├── summary-learn.log                  ← トップレベルの通算ログ (全 run / resume を連結)
├── 0001/
│   ├── KK_synthesized.bin
│   ├── KKP_synthesized.bin
│   ├── KPP_synthesized.bin
│   ├── state.bin                      ← resume 用の重み + Ranger optimizer state (3 component ぶん)
│   └── learn.log                      ← この save 時点の学習ログの snapshot
├── 0002/
│   ├── ...
├── ...
└── 000N/                              ← 最新 (= 最後に保存された) save
    ├── KK_synthesized.bin
    ├── KKP_synthesized.bin
    ├── KPP_synthesized.bin
    ├── state.bin
    └── learn.log
```

`learn.log` はヘッダ行つき CSV で、すべての target で同じフォーマット:

```
eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher
KPPT/kk,1,1,32,-,-,0.234,0.001000,0.000999,1.000000,524288,teachers/
KPPT/kk,1,1,64,-,-,0.232,0.000999,0.000998,1.000000,1048576,teachers/
...
KPPT/kkp,1,1,32,-,-,0.156,0.001000,0.000999,1.000000,524288,teachers/
...
KPPT/kpp,1,1,32,-,-,0.245,0.001000,0.000999,1.000000,524288,teachers/
...
```

`eval` 列は **`<target>/<component>`** 形式で、KPPT 系では `kk` / `kkp` / `kpp` を区別する。KPPT 系の行には NNUE/SFNN architecture 接尾辞は付かない (NNUE 系では `NNUE_HALFKP-NNUE_halfkp_256x2_32_32` のように arch も結合される)。

各 save の `0NNN/learn.log` snapshot とトップレベル `<output>/summary-learn.log` は列数が違うが、列の意味は同じ。`summary-learn.log` は `curr_batch` を除いた sb 境界行だけを保持する。`positions` は resume またぎで累積される (新規 run 開始時、既存トップレベル log からその component の最大 positions を読み取って続きから書く)。各列の意味は [`spec/04-checkpoint-layout.md`](../../spec/04-checkpoint-layout.md) を参照。

最新の `000N/` (= 最大番号) をやねうら王の KPPT エンジンの eval ディレクトリに設定すれば対局可能 (`state.bin` は engine からは無視される)。

中断・再開の挙動は target 横断で同じなので、[チュートリアル 5. 中断・再開](../tutorial/5-resume.md) を参照。

### KPP_KKPT (factorised)

`--arch KPP_KKPT` を指定すれば、KPP のみ手番チャンネルを省いた layout (約半分のサイズ) で書き出される。KK / KKP は KPPT と byte-identical。

```bash
./target/release/examples/bulletou \
    --arch KPP_KKPT \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --superbatches 20
```

### 主要 CLI フラグ (KPPT 系すべて共通)

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--arch` | `KPPT` (3-component sequential) / `KPP_KKPT` (factorised) | (required) |
| `--teacher` | 教師ファイル (`.hcpe` / `.hcpe3` / `.pack` / `.psv`)、またはそれらが入ったディレクトリ、カンマ区切りで併用可 | (必須) |
| `--output` | チェックポイント親ディレクトリ | `checkpoints/<target>` (例: `checkpoints/KPPT`、`checkpoints/KPP_KKPT`) |
| `--net-id` | チェックポイント subdir 名のプレフィクス | target 別自動 |
| `--batch-size` | 1 gradient step あたりの局面数 | 65536 |
| `--positions-per-superbatch` | 1 superbatch の目標局面数。実効値は `batch-size` の倍数へ切り捨て | 100000000 |
| `--superbatches` | 1 epoch あたりの superbatch 上限。省略時は上限なし (dataloader EOF まで) | (上限なし) |
| `--max-epochs` | epoch を何回実行するか。`--superbatches` 指定時の epoch は教師1周ではなく LR/validation cycle。`step` / `geometric` / `cos` は epoch 境界で `--lr` に戻る | 1 |
| `--save-rate` | N superbatch ごとに保存。デフォルトでは epoch 末尾も保存 | 20 |
| `--save-epoch-end` / `--no-save-epoch-end` | epoch 末尾の暗黙 save を有効/無効にする | on |
| `--lr` / `--lr-schedule` / `--lr-min` | LR スケジューラ (`step` = tatara/bullet-shogi 互換 StepLR、`geometric` = geometric、`cos` = cosine、詳細は [§6.1](../tutorial/6-tune.md#61-学習スケジュール)) | 0.000875 / `step` / 0.00001 |
| `--lambda` | 教師 eval と対局結果 (WDL = Win/Draw/Loss) のブレンド比 (やねうら王内蔵学習器の `lambda` と同じ慣例): `λ × 教師eval + (1−λ) × 対局結果`。`λ=1.0` で純 eval、`λ=0.0` で純 WDL | 1.0 |
| `--scale` | デフォルトの sigmoid-MSE target で使う eval-to-score sigmoid scale | 290 |
| `--yaneuraou-quant-scale` | f32 → i{16,32} 量子化スケール | 4000 (KK/KKP), 400 (KPP) |
| `--score-drop-abs` | `|score| >= N` の局面を除外 (詰み手スコア対策) | 32000 |

学習単位の意味は [§6.1 学習スケジュール](../tutorial/6-tune.md#61-学習スケジュール) を参照。

## メモリ要件

KK / KKP の学習は GPU メモリをほとんど使わない (4 GB GPU でも余裕)。

**KPP は約 2.3 GB の GPU メモリを使う**ので、**8 GB+ の GPU 推奨**。

## ハイパーパラメータの指針

KPPT は歴史的に以下の組み合わせが多い:

- ELMO 式 WDL 教師 (`--lambda 0.5` 程度の中間値、eval と対局結果を 50:50 で混合)
- 強めの weight decay
- 小さめの learning rate (`--lr 1e-4 〜 1e-3`)

`bulletou` のデフォルトは純 eval (`--lambda 1.0`) と tatara 寄せの `--lr 8.75e-4` になっている。KPPT で実用品質を狙う場合は `--lambda` と学習率を上記の方針で調整する。

## 関連

- [BulletOu source: `crates/bulletou_lib/src/game/inputs/shogi_kk.rs`](../../../crates/bulletou_lib/src/game/inputs/shogi_kk.rs)
- [BulletOu source: `crates/bulletou_lib/src/game/inputs/shogi_kkp.rs`](../../../crates/bulletou_lib/src/game/inputs/shogi_kkp.rs)
- [BulletOu source: `crates/bulletou_lib/src/game/inputs/shogi_kpp.rs`](../../../crates/bulletou_lib/src/game/inputs/shogi_kpp.rs)
- [BulletOu source: `crates/bulletou_lib/src/value/yaneuraou_kppt.rs`](../../../crates/bulletou_lib/src/value/yaneuraou_kppt.rs)
- [YaneuraOu source: `source/eval/kppt/evaluate_kppt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kppt/evaluate_kppt.h)
- [YaneuraOu source: `source/eval/kpp_kkpt/evaluate_kpp_kkpt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kpp_kkpt/evaluate_kpp_kkpt.h)
