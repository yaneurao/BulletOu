# 2. NNUE チュートリアル — 将棋 NNUE を学習する

<a href="../../en/tutorial/2-nnue-tutorial.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: やねうら王互換エンジンが読み込める将棋 NNUE をエンドツーエンドで学習する。

この章は [1. クイックスタート](1-quickstart.md) を完了している前提 — ツールチェーンが動き、smoke test の学習が成功した状態。

## 2.1 何を学習するか

`bulletou --eval-type NNUE_HALFKP` で、やねうら王に最初に入った NNUE 評価関数 `halfkp_256x2-32-32` (那須さんの 2018 年 PR #75) と同じ構成 — HalfKP 入力 + 全層 ClippedReLU の 4 層 NNUE を学習する:

```
将棋の局面
       │
       ▼ HalfKP sparse 特徴量 (125,388 次元、自玉 / 相手玉の 2 perspective)
       │
       ▼ L0 affine + ClippedReLU       ← 両 perspective で重み共有
       │
       ▼ accumulator (256 次元 × 2 perspective = 連結して 512 次元)
       │
       ▼ L1 affine (512 → 32) + ClippedReLU
       ▼ L2 affine (32 → 32) + ClippedReLU
       ▼ Out affine (32 → 1)
       │
       ▼ eval (centipawn ベースのスカラー)
```

アーキテクチャは `--arch` で選ぶが、本チュートリアル時点では `256x2-32-32` の 1 種類のみ (`x2` は dual-perspective、`256` は accumulator size、`32-32` は L2/L3 のサイズ)。

(SqrClippedReLU / SCReLU は別系統で、2026 年の PR #311 (SFNNwoPSQT-1536) で導入された新しい活性化関数。`NNUE_HALFKP` は使わない。)

最強構成 (Layer Stack + threat 特徴量 + 大きい FT) には届かないが、学習の挙動を体感するのと、エンジンに繋いで対局確認するには十分。

## 2.2 学習データを用意する

`.pack` / `.hcpe` / `.hcpe3` / `.psv` のいずれかのファイルが必要。

- **自分で生成** — [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection) の `gensfen` スクリプトで `.pack` を出力するか、dlshogi 系のデータ生成で `.hcpe` / `.hcpe3` を作る。チュートリアル目的なら 1000 万〜1 億局面で十分。
- **共有データセットを使う** — 将棋コミュニティでは各フォーマットのデータが共有されている。

本チュートリアルでは作業ディレクトリ直下に `teachers/` を作り、その下に教師ファイルを置く構成を仮定する:

```
teachers/
    teacher.pack
```

(`.hcpe` / `.hcpe3` / `.psv` でも同様に動く。フォーマットは拡張子から自動判別される。複数ファイル混在もディレクトリ指定で OK だが、すべて同じ拡張子であること。)

### 小さなサブセットで動作確認したい場合

巨大なデータセット (数十 GB) でいきなり動かす前に、小さなサブセットで試したいときは、`gensfen` 等で小さめのファイルを生成するか、`--batches-per-superbatch` を指定して 1 superbatch あたりの消費量を絞る (§2.4 参照)。

## 2.3 NNUE 学習を走らせる

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

(AMD GPU なら `--features device-cuda` を `--features device-rocm` に。)

`--output` を省略しているので、checkpoint は `checkpoints/NNUE_HALFKP-256x2-32-32/` 配下に書かれる (`--eval-type` と `--arch` の値から自動命名)。別の名前にしたい場合は `--output checkpoints/my-halfkp` のように明示する。

`--teacher` には:
- 1 つのファイル (`teachers/teacher.pack` のようなフルパス)
- ディレクトリ (上記例。中の同一拡張子ファイルがすべて連結される)
- カンマ区切り複数指定

のいずれも渡せる。

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--max-epochs 3` のように指定する (各 epoch 開始時に LR がリセットされる)。

動いていれば以下のような出力が流れる:

```
=== bulletou: running NNUE_HALFKP (256x2-32-32 ClippedReLU, dual-perspective) ===
Training Preamble
Net Name               : shogi_nnue_halfkp
Batch Size             : 16384
Batches / Superbatch   : 6104
Positions / Superbatch : 100007936
...
superbatch 1   pos = ... pos/s = ...   loss = ...
superbatch 2   ...
```

`pos/s` (1 秒あたり処理局面数) が学習速度の目安。RTX 4090 1 枚で数千万 pos/s 出る。下位 GPU では比例して低下。

## 2.4 学習スケジュール

ログに出てくる `superbatch` は **checkpoint や学習率を更新するためのまとまり**で、デフォルトで約 1 億局面ぶん。

主要なフラグ:

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--batch-size` | 1 gradient step あたりの局面数 | 16384 |
| `--batches-per-superbatch` | 1 superbatch を構成する mini-batch 数 | `ceil(100M / batch-size)` (≒ 1 superbatch ≒ 1 億局面) |
| `--superbatches` | epoch あたりの superbatch 数の上限 | 上限なし (= EOF まで) |
| `--max-epochs` | 教師データを何周するか | 1 |
| `--save-rate` | N superbatch ごとに checkpoint を保存 | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR (`lr-step` superbatch ごとに `lr-gamma` 倍) | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL (eval スコア vs 対局結果の blend 比率) を線形補間 | 0.0 / 1.0 |

実行例 (1 億局面 × 40 superbatch = 計 40 億局面):

```bash
cargo run --release --features device-cuda --example bulletou -- \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --superbatches 40
```

教師ファイルが 1 superbatch 未満 (≒ 1 億局面未満) しか無い場合は `--batches-per-superbatch` を小さくする (例: `1024` で 1 superbatch ≒ 1670 万局面) と、何回も save が走るようになる。

## 2.5 出力を確認する

学習完了後、`checkpoints/NNUE_HALFKP-256x2-32-32/` 配下は以下のレイアウト:

```
checkpoints/NNUE_HALFKP-256x2-32-32/
├── learn.log                          ← 全 run / resume を連結した累積ログ
├── 0001/
│   ├── nn.bin                         ← やねうら王 / Stockfish 互換 NNUE バイナリ
│   ├── state.bin                      ← resume 用の重み + Adam moments
│   └── learn.log                      ← この save 時点の学習ログ snapshot
├── 0002/
├── ...
└── 000N/                              ← 最新 (= 最後に保存された) save
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

最新の `000N/nn.bin` がやねうら王エンジンに渡すファイル。

## 2.6 中断・再開

同じ `--output` を指定してもう一度同じコマンドを走らせると、`bulletou` は自動的に最新 `000N/state.bin` から resume する (新 save は `000(N+1)/` から続く)。新規学習にしたい場合は `--output` を別の dir にするか、既存 dir を削除する。

## 2.7 エンジンに組み込む

やねうら王エンジンが eval ファイルを探す場所 (通常 `eval/nn.bin`) に学習結果の `000N/nn.bin` を置き、エンジンを起動して `bench` や簡易対局でロードを確認する。具体的なファイル配置はエンジンの設定 (`EvalDir` 等) に依存するのでエンジン側のドキュメントを参照。

`state.bin` / `learn.log` はエンジンからは無視されるが、再学習や loss 推移の確認用に残しておくと便利。

## 2.8 別のターゲットを学習したい場合

- **NNUE K-P** (HalfKP と同じ 4 層 ClippedReLU で入力だけ違う、軽量モデル): `--eval-type NNUE_KP`。詳細は [NNUE K-P 学習](../shogi/kp.md)
- **KPPT** (`KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` の 3 ファイル組): `--eval-type KPPT` または factorised 版の `--eval-type KPP_KKPT`。詳細は [KPPT / KPP_KKPT 学習](../shogi/kppt.md)
- 他の NNUE バリアント (HalfKA / SFNN+ls9 等) は順次 `--eval-type` に追加予定

## 2.9 次のステップ

- [リファレンス: NNUE HalfKP 学習](../shogi/halfkp.md) — `nn.bin` のバイナリレイアウト、量子化、resume の詳細
- [リファレンス: NNUE の基礎](../1-basics.md) — perspective NNUE の数式
- [リファレンス: 学習済みネットワーク](../4-saved-networks.md) — checkpoint レイアウト、量子化、変換チェーン
- [リファレンス: KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習
