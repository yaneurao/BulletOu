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

- BulletOu をビルド済み (`cargo build --release --features device-cuda --example bulletou`)
- 学習データ (`.hcpe` / `.hcpe3` / `.pack` のいずれか)
- 4 GB+ の空き GPU メモリ (KPP 学習は ~2.3 GB を使う)

### KPPT (elmo 互換)

`--eval-type KPPT` を指定すると KK / KKP / KPP の 3 component を **1 コマンドで連続学習** し、最後に `<output>/final/` に 3 ファイルを集約する。

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type KPPT \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-kppt
```

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--max-epochs 3` のように指定する (各 epoch 開始時に LR がリセットされる)。

完了すると、各 save 単位で `0001/`, `0002/`, ... と 4 桁番号のディレクトリが並び、それぞれに 3 ファイルが入る:

```
checkpoints/my-kppt/
├── learn.log                          ← トップレベルの通算ログ (全 run / resume を連結)
├── 0001/
│   ├── KK_synthesized.bin
│   ├── KKP_synthesized.bin
│   ├── KPP_synthesized.bin
│   ├── state.bin                      ← resume 用の重み + Adam moments (3 component ぶん)
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

各 save 配下の `learn.log` のフォーマット (3 component の bullet 既存 log.txt をセクションヘッダ付きで連結):

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

各行は `<superbatch>,<curr_batch>,<loss>` の CSV。bullet は 32 batch ごとに 1 行記録する。

トップレベルの `<output>/learn.log` には、1 run ごとに 1 section が追記される。section 頭にその run の wall-clock 時刻と生成された numbered dir の範囲が入る:

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

resume すると superbatch カウンタは 1 から再開される (run ごとに LR scheduler が reset されるため)。トップレベルの learn.log ではセクションヘッダで run の境界を判別する。

最新の `000N/` (= 最大番号) をやねうら王の KPPT エンジンの eval ディレクトリに設定すれば対局可能 (`state.bin` は engine からは無視される)。

### 中断・再開

`--output` で指定した dir に `0001/` 等の numbered dir + `state.bin` が既に存在する場合、起動時に自動的に **最新の `state.bin` から resume** します。新しい save は既存番号の続きから書かれる (例: 前回 5 個保存していたら新規 save は `0006/` から)。

同じコマンドを実行するだけで、前回の重みを引き継いで学習が続行されます。新規学習にしたい場合は `--output` を別の dir にするか、既存 dir を削除してください。

### KPP_KKPT (factorised)

`--eval-type KPP_KKPT` を指定すれば、KPP のみ手番チャンネルを省いた layout (約半分のサイズ) で書き出される。KK / KKP は KPPT と byte-identical。

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type KPP_KKPT \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --superbatches 20
```

### 単体 component だけ学習する

開発・動作確認用に、`--eval-type KPPT_KK` / `KPPT_KKP` / `KPPT_KPP` / `KPP_KKPT_KPP` で 1 component だけ学習することもできる:

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type KPPT_KPP \
    --teacher inbox/ref/small.hcpe \
    --output checkpoints/kpp-smoke \
    --superbatches 3 \
    --batches-per-superbatch 100
```

### 主要 CLI フラグ (KPPT 系すべて共通)

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--eval-type` | `KPPT` (3 component 連続学習) / `KPP_KKPT` (factorised 版) / `KPPT_KK` / `KPPT_KKP` / `KPPT_KPP` / `KPP_KKPT_KPP` | (必須) |
| `--teacher` | 教師ファイル (`.hcpe` / `.hcpe3` / `.pack` / `.psv`)、またはそれらが入ったディレクトリ、カンマ区切りで併用可 | (必須) |
| `--output` | チェックポイント親ディレクトリ | eval-type 別自動 |
| `--net-id` | チェックポイント subdir 名のプレフィクス | eval-type 別自動 |
| `--batch-size` | 1 gradient step あたりの局面数 | 16384 |
| `--batches-per-superbatch` | 1 superbatch の mini-batch 数 | `ceil(100M / batch-size)` |
| `--superbatches` | 1 epoch あたりの superbatch 上限。省略時は上限なし (dataloader EOF まで) | (上限なし) |
| `--max-epochs` | 教師データを何周回すか (= dataloader EOF を何回踏むか)。各 epoch 開始時に LR スケジューラがリセットされる | 1 |
| `--save-rate` | N superbatch ごとに保存 | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR スケジューラ | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL 線形補間 | 0.0 / 1.0 |
| `--yaneuraou-quant-scale` | f32 → i{16,32} 量子化スケール | 4000 (KK/KKP), 400 (KPP) |
| `--score-drop-abs` | `|score| >= N` の局面を除外 (詰み手スコア対策) | 32000 |

学習単位の意味は [§2.4 学習スケジュール](../tutorial/2-nnue-tutorial.md#24-学習スケジュール) を参照。

## メモリ要件

KK / KKP の学習は GPU メモリをほとんど使わない (4 GB GPU でも余裕)。

**KPP は約 2.3 GB の GPU メモリを使う**ので、**8 GB+ の GPU 推奨**。

## ハイパーパラメータの指針

KPPT は歴史的に以下の組み合わせが多い:

- ELMO 式 WDL 教師 (`--start-wdl 0.5 --end-wdl 0.5` 等の中程度設定)
- 強めの weight decay
- 小さめの learning rate (`--lr 1e-4 〜 1e-3`)

一方 `bulletou` のデフォルトは NNUE 寄り (`--start-wdl 0.0 --end-wdl 1.0`、`--lr 1e-3`)。KPPT で実用品質を狙う場合は WDL と学習率を上記の方針で調整する。

## 関連

- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kk.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kk.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kkp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kkp.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kpp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kpp.rs)
- [BulletOu source: `crates/bullet_lib/src/value/yaneuraou_kppt.rs`](../../../crates/bullet_lib/src/value/yaneuraou_kppt.rs)
- [YaneuraOu source: `source/eval/kppt/evaluate_kppt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kppt/evaluate_kppt.h)
- [YaneuraOu source: `source/eval/kpp_kkpt/evaluate_kpp_kkpt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kpp_kkpt/evaluate_kpp_kkpt.h)
