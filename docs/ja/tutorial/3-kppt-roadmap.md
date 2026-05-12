# 3. KPPT / KPP_KKPT 学習

<a href="../../en/tutorial/3-kppt-roadmap.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

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

NNUE は「**疎特徴量変換器 + 小さい MLP**」 — 普通の NN 形状。

KPPT は「**巨大な疎 embedding テーブルの和、隠れ層なし**」:

```
eval(pos) = KK[bk][wk]
          + Σ_i KKP[bk][wk][p_i]
          + Σ_{i<j} KPP[bk][p_i][p_j]
          + (手番項 T)
```

NN 的な「隠れ層」がない。巨大なルックアップテーブルの和だけ。

最大の `KPP` テーブルは `81 × 1548 × 1548 = 194,100,624` 次元、f32 で 776 MB、Adam 状態込みで GPU 上 2.3 GB。

## ファイルフォーマット (やねうら王ソース確認済み)

YaneuraOu の `source/eval/kppt/evaluate_kppt.h` と `eval/kpp_kkpt/evaluate_kpp_kkpt.h` 由来:

| ファイル | KPPT 型 | KPP_KKPT 型 | サイズ |
|---|---|---|---|
| `KK_synthesized.bin` | `int32_t kk[81][81][2]` | 同左 | 51 KB |
| `KKP_synthesized.bin` | `int32_t kkp[81][81][1548][2]` | 同左 | 77 MB |
| `KPP_synthesized.bin` | `int16_t kpp[81][1548][1548][2]` | `int16_t kpp[81][1548][1548]` | **740 MB / 388 MB** |

末尾の `[2]` は `[stm_independent, stm_dependent]` (手番無関係項 + 手番依存項)。
- **KPPT**: KPP も手番項あり (`[2]`)
- **KPP_KKPT**: KPP は手番項なし。手番は KK と KKP 側にだけ存在

BulletOu は **`[0]` (手番無関係項) のみ** を学習し、`[1]` (手番依存項) は 0 で書く。

## 実際の使い方

### 必要なもの

- BulletOu をビルド済み (`cargo build --release --features cuda --example bullet_ou_train`)
- 学習データ (`.hcpe` / `.hcpe3` / `.pack` のいずれか)
- 4 GB+ の空き GPU メモリ (KPP 学習は ~2.3 GB を使う)

### KPPT (elmo 互換、`int16_t × 2` 形式の KPP)

KK / KKP / KPP の 3 component をそれぞれ独立に学習する:

```bash
# KK 学習 → KK_synthesized.bin
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kk \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kppt \
    --net-id kk \
    --superbatches 20

# KKP 学習 → KKP_synthesized.bin
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kkp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kppt \
    --net-id kkp \
    --superbatches 20

# KPP 学習 → KPP_synthesized.bin (KPPT 形式)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kpp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kppt \
    --net-id kpp \
    --superbatches 20
```

3 つの `.bin` ファイルが出るのでまとめる:

```bash
mkdir -p checkpoints/my-kppt/final
cp checkpoints/my-kppt/kk-20/KK_synthesized.bin   checkpoints/my-kppt/final/
cp checkpoints/my-kppt/kkp-20/KKP_synthesized.bin checkpoints/my-kppt/final/
cp checkpoints/my-kppt/kpp-20/KPP_synthesized.bin checkpoints/my-kppt/final/
```

`checkpoints/my-kppt/final/` をやねうら王の KPPT エンジンの eval ディレクトリに設定すれば対局可能。

### KPP_KKPT (factorised、`int16_t × 1` 形式の KPP)

KK と KKP は KPPT と同じファイル形式なので **同じコマンド** を使う。違うのは KPP の writer だけ:

```bash
# KK 学習 (= KPPT と同じ)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kk \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --net-id kk

# KKP 学習 (= KPPT と同じ)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kppt-kkp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --net-id kkp

# KPP 学習 (KPP_KKPT 形式 = 手番チャンネルなし、半分のサイズ)
cargo run --release --features cuda --example bullet_ou_train -- \
    --eval-type kpp-kkpt-kpp \
    --data /path/to/train.hcpe \
    --output checkpoints/my-kpp-kkpt \
    --net-id kpp
```

3 ファイルを集める手順は KPPT と同じ。

### 単体 component の学習

`shogi_kk_train` / `shogi_kk_kkp_train` / `shogi_kpp_train` という単体 example も用意してある (これらは bullet_ou_train の内部で呼んでいるのと同じロジック)。動作確認用に:

```bash
cargo run --release --features cuda --example shogi_kpp_train -- \
    --data inbox/ref/small.hcpe \
    --output checkpoints/kpp-smoke \
    --superbatches 3 \
    --batches-per-superbatch 100
```

### 主要 CLI フラグ (KPPT 系すべて共通)

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--eval-type` | `kppt-kk` / `kppt-kkp` / `kppt-kpp` / `kpp-kkpt-kpp` | (必須) |
| `--data` | 教師ファイル (`.hcpe` / `.hcpe3` / `.pack`、複数指定はカンマ区切り) | (必須) |
| `--output` | チェックポイント親ディレクトリ | eval-type 別自動 |
| `--net-id` | チェックポイント subdir 名のプレフィクス | eval-type 別自動 |
| `--batch-size` | 1 gradient step あたりの局面数 | 16384 |
| `--batches-per-superbatch` | 1 superbatch の mini-batch 数 | `ceil(100M / batch-size)` |
| `--superbatches` | 走らせる superbatch 数 | 10 |
| `--save-rate` | N superbatch ごとに保存 | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR スケジューラ | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL 線形補間 | 0.0 / 1.0 |
| `--yaneuraou-quant-scale` | f32 → i{16,32} 量子化スケール | 4000 (KK/KKP), 400 (KPP) |
| `--score-drop-abs` | `|score| >= N` の局面を除外 (詰み手スコア対策) | 32000 |

学習単位の意味は [2.4 学習の単位](2-nnue-tutorial.md#24-学習の単位--batch--superbatch--save--lr-の関係) を参照。

## メモリ要件

| Component | 重みパラメータ数 | f32 重み | + Adam (3× state) | 推奨 GPU メモリ |
|---|---|---|---|---|
| KK | 6,561 | 26 KB | 78 KB | ほぼ何でも |
| KKP | 10,156,428 | 40 MB | 120 MB | 4 GB+ |
| KPP | 194,100,624 | 776 MB | 2.33 GB | **8 GB+ 推奨 (バッチ buffer 込みで 3 GB ほど)** |

`max_active = 703` (KPP の 1 局面あたりアクティブ特徴数 = C(38, 2)) なので、batch_size 16384 で GPU 側の sparse index buffer は約 92 MB。

## ハイパーパラメータの指針

KPPT は歴史的に以下の組み合わせが多い:

- ELMO 式 WDL 教師 (`--start-wdl 0.5 --end-wdl 0.5` 等の中程度設定)
- 強めの weight decay
- 小さめの learning rate (`--lr 1e-4 〜 1e-3`)

一方 `bullet_ou_train` のデフォルトは NNUE 寄り (`--start-wdl 0.0 --end-wdl 1.0`、`--lr 1e-3`)。KPPT で実用品質を狙う場合は WDL と学習率を上記の方針で調整する。

## 関連

- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kk.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kk.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kkp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kkp.rs)
- [BulletOu source: `crates/bullet_lib/src/game/inputs/shogi_kpp.rs`](../../../crates/bullet_lib/src/game/inputs/shogi_kpp.rs)
- [BulletOu source: `crates/bullet_lib/src/value/yaneuraou_kppt.rs`](../../../crates/bullet_lib/src/value/yaneuraou_kppt.rs)
- [YaneuraOu source: `source/eval/kppt/evaluate_kppt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kppt/evaluate_kppt.h)
- [YaneuraOu source: `source/eval/kpp_kkpt/evaluate_kpp_kkpt.h`](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/kpp_kkpt/evaluate_kpp_kkpt.h)
