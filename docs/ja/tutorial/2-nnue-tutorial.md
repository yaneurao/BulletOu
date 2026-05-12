# 2. NNUE チュートリアル — 将棋 NNUE を学習する

<a href="../../en/tutorial/2-nnue-tutorial.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: やねうら王互換エンジンが読み込める将棋 NNUE をエンドツーエンドで学習する。

この章は [1. クイックスタート](1-quickstart.md) を完了している前提 — ツールチェーンが動き、smoke test の学習が成功した状態。

## 2.1 何を学習するか

以下の構造の小さな NNUE を学習する (`shogi_simple.rs` のデフォルト):

```
将棋の局面
       │
       ▼ ShogiHalfKA_hm (73,305 次元 sparse 特徴量)
       │
       ▼ Feature Transformer (FT、隠れ層サイズ 1024 または 1536、perspective で 2 倍化)
       │
       ▼ SCReLU 活性化
       │
       ▼ Linear → スカラー score
```

これは「将棋で実用最小の NNUE」位置にある構成。最強構成 (Layer Stack + threat 特徴量 + 大きな FT) には遠く及ばないが、学習の挙動を体感するには十分。

より強い構成をすぐ試したい場合は、`shogi_layerstack.rs` 例が本番品質のバリアント (Layer Stack、bucket 選択、Threat / HandThreat 特徴量オプション、WDL スケジュール対応)。

## 2.2 学習データを用意する

`.pack` / `.hcpe` / `.hcpe3` のいずれかのファイルが必要。

- **自分で生成** — [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection) の `gensfen` スクリプトで `.pack` を出力するか、dlshogi 系のデータ生成で `.hcpe` / `.hcpe3` を作る。チュートリアル目的なら 1000 万〜1 億局面で十分。
- **共有データセットを使う** — 将棋コミュニティでは各フォーマットのデータが共有されている。

本チュートリアルでは以下を仮定:

```
/data/shogi/raw.pack
```

(`.hcpe` / `.hcpe3` でも同様に動く。パスは自分の環境に読み替える。)

### 小さなサブセットで動作確認したい場合

巨大なデータセット (数十 GB) でいきなり動かす前に、小さなサブセットで試したいときは、`gensfen` 等で小さめのファイルを生成するか、`--batches-per-superbatch` を指定して 1 superbatch あたりの消費量を絞る (§2.4 参照)。

## 2.3 NNUE 学習を走らせる

データ形式に応じて example を選ぶ:

- **`shogi_simple`** — `.pack` を読む
- **`shogi_simple_hcpe`** — `.hcpe` を読む

### `.pack` の場合

```bash
cargo run --release --features cuda --example shogi_simple -- \
    --data /data/shogi/raw.pack \
    --output checkpoints/my-first-shogi-net \
    --superbatches 40
```

(AMD GPU なら `--features cuda` を `--features rocm` に。)

### `.hcpe` の場合

```bash
cargo run --release --features cuda --example shogi_simple_hcpe -- \
    --data /data/shogi/raw.hcpe \
    --output checkpoints/my-first-shogi-net \
    --superbatches 40
```

HCPE 固有の制約:

- HCPE には `game_ply` 情報がないので、`game_ply` を使う bucket (Layer Stack の `ply9` 等) は使えない (この最小例は bucket を使わない)
- HCPE には policy teacher が存在しないので value 学習のみ。policy 教師込みの学習が必要なら HCPE3 を使う

動いていれば以下のような出力:

```
superbatch 1 / 40   pos = ... pos/s = ...   loss = ...
superbatch 2 / 40   ...
```

`pos/s` (1 秒あたり処理局面数) が学習速度の目安。RTX 4090 1 枚で smoke test 構成なら数千万 pos/s 出る。下位 GPU では比例して低下。

## 2.4 学習スケジュール

ログに出てくる `superbatch 1 / 40` の「superbatch」は **checkpoint や学習率を更新するためのまとまり**で、デフォルトで約 1 億局面ぶん。学習の長さは `--superbatches` で指定する。

主要なフラグ:

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--batch-size` | 1 gradient step あたりの局面数 | 16384 |
| `--batches-per-superbatch` | 1 superbatch を構成する mini-batch 数 | `ceil(100M / batch-size)` (≒ 1 superbatch ≒ 1 億局面) |
| `--superbatches` | 走らせる superbatch の総数 | 10 |
| `--save-rate` | N superbatch ごとに checkpoint を保存 | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR (`lr-step` superbatch ごとに `lr-gamma` 倍) | 0.001 / 0.1 / 8 |
| `--start-wdl` / `--end-wdl` | WDL (eval スコア vs 対局結果の blend 比率) を `--superbatches` の区間で線形補間 | 0.0 / 1.0 |

実行例:

```bash
--batch-size 16384 --batches-per-superbatch 6104 --superbatches 40
# = 1 superbatch ≒ 1 億局面、合計 40 億局面
```

スケジューラの詳細 (Cosine / Linear / Warmup 等) は [リファレンス](../) を参照。

## 2.5 出力を確認する

学習完了 (および各 checkpoint) のたびに、`checkpoints/my-first-shogi-net/` 配下に **`nn.bin`** が書き出される。これがやねうら王エンジンが対局時に読み込む NNUE 評価関数パラメーターファイル。

## 2.6 エンジンに組み込む

やねうら王エンジンが eval ファイルを探す場所 (通常 `eval/nn.bin`) に学習結果の `nn.bin` を置き、エンジンを起動して `bench` や簡易対局でロードを確認する。具体的なファイル配置はエンジンの設定 (`EvalDir` 等) に依存するのでエンジン側のドキュメントを参照。

## 2.7 本番構成にステップアップする

`shogi_simple` に慣れたら、`shogi_layerstack` でより強い学習に移る:

```bash
cargo run --release --features cuda --example shogi_layerstack -- \
  --data /data/shogi/train.pack \
  --output checkpoints/my-layerstack-net \
  --feature ShogiHalfKaHmThreat \
  --bucket-mode progress8kpabs \
  --progress-coeff progress.bin \
  --start-wdl 0.0 --end-wdl 1.0
```

各部品 (Threat 特徴量、`progress.bin`、WDL スケジュール) は [リファレンス](../) で説明されている。`shogi_simple` を「全部正しく繋がっているか」の確認用に使い、その後 `shogi_layerstack` で本格イテレーションを回す、というのが推奨フロー。

## 2.8 次のステップ

- [リファレンス: NNUE の基礎](../1-basics.md) — perspective NNUE の数式
- [リファレンス: 学習済みネットワーク](../4-saved-networks.md) — checkpoint レイアウト、量子化、変換チェーン
- [リファレンス: KP 絶対進行度](../shogi/kp-absolute-progress.md) — `--bucket-mode progress8kpabs` が実際に何をやっているか
- [リファレンス: shogi_progress_kpabs_train](../shogi/shogi_progress_kpabs_train.md) — 自分で `progress.bin` を学習する方法
- [KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習 (リファレンス)
