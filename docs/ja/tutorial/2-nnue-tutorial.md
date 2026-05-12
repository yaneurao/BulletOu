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

以下のいずれかが必要:

- **`.pack`** — やねうら王の `gensfen` が出す **ゲーム単位の可変長フォーマット**。1 ファイルレコード = 1 ゲーム (start_flag + (平手以外なら) hcp/ply + (move16, eval) × moveNum + 終局マーカー)。`ShogiPackLoader` がゲームを ply 単位に展開する。
- **`.hcpe`** — dlshogi 系の **38 byte 固定長レコード**形式 (HCP + eval + bestMove16 + gameResult)。
- **`.hcpe3`** — dlshogi 系の **ゲーム単位可変長**形式 (ゲームヘッダ + moveNum × MoveInfo + ply ごとの MoveVisits)。

> ⚠️ `.pack` は「PackedSfenValue が連続したファイル」では **ない**。`PackedSfenValue` は **トレーナの内部単位** (40 byte 固定長レコード)、`.pack` は **ファイル形式** で別物。詳細は [概要](0-overview.md#学習データはどこから来るか) 参照。

3 つすべてに対応。自分が使うジェネレータ (または手持ちの共有データセット) に合わせて選ぶ。

入手方法:

- **自分で生成** — やねうら王の `gensfen` (`.pack`) または dlshogi 系のデータ生成 (`.hcpe` / `.hcpe3`) を使う。各プロジェクトのドキュメント参照。典型的な規模は数億局面だが、チュートリアル目的なら 1000 万〜1 億局面で十分
- **共有データセットを使う** — 将棋コミュニティでは `.pack` / `.hcpe` / `.hcpe3` のいずれも共有されている。出所が信頼できることを確認

本チュートリアルでは以下を仮定:

```
/data/shogi/raw.pack    # または
/data/shogi/raw.hcpe    # または
/data/shogi/raw.hcpe3
```

(パスは任意。自分の環境に合わせて読み替える。)

### まずは小さなテストデータで

データセットが巨大 (数十 GB) なときは、最初に小さなサブセットで動作確認すると楽。

- **`.hcpe`** (固定 38 byte) は単に先頭を切り出せばよい:
  ```bash
  head -c $((38 * 10000000)) /data/shogi/raw.hcpe > /tmp/small.hcpe
  ```
  これで先頭 1000 万レコード分。

- **`.pack` / `.hcpe3`** (ゲーム単位の可変長) は、バイト単位で切り出すとゲーム境界が壊れるので NG。`gensfen` で直接小さな `.pack` を生成するか、`--batches-per-superbatch` で 1 superbatch あたりの消費量を抑えるかのいずれか (§2.3 参照)。

## 2.3 NNUE 学習を走らせる

データ形式に応じて、最小例が 2 つ用意されている:

- **`shogi_simple`** — `.bin` (bullet-utils の変換で生成される `PackedSfenValue` 連続ファイル) または `.pack` (やねうら王 `gensfen` のゲーム単位可変長) を読み込む。
- **`shogi_simple_hcpe`** — `.hcpe` (dlshogi 系の 38 byte 固定長) を読み込む。

データに合わせて選ぶ。ネット構造と学習ループはほぼ同じ。

### パターン A: `.pack` (やねうら王 gensfen) の場合

```bash
cargo run --release --features cuda --example shogi_simple -- \
  --data /tmp/small.pack \
  --output checkpoints/my-first-shogi-net
```

(AMD GPU なら `--features cuda` を `--features rocm` に。)

### パターン B: `.hcpe` (dlshogi 系) の場合

```bash
cargo run --release --features cuda --example shogi_simple_hcpe -- \
  --data /data/shogi/train.hcpe \
  --output checkpoints/my-first-shogi-net-hcpe
```

`shogi_simple_hcpe` は、各 HCPE レコード (Apery 系 HCP + eval + bestMove16 + gameResult) を内部で PackedSfenValue にデコードしてから、`shogi_simple` と同じ `ShogiHalfKA_hm` 特徴量 + SCReLU + dual-perspective + 出力 1 のネットに流す。`--data-format` のような切り替えはなく、ファイルは hcpe 固定 (シンプルさを優先した設計)。

HCPE 固有の制約:

- HCPE には `game_ply` 情報がないので、Layer Stack の `ply9` bucket は使えない (この最小例は bucket を使わない)
- HCPE には policy teacher (MoveVisits) がない。value 学習のみが対象。policy 教師込みは HCPE3 で対応予定

本格学習では `--data` をフルデータセットに差し替え、`small.pack` / `small.hcpe` 工程を省略する。

動いていれば以下のような出力:

```
loaded 73305 input features (ShogiHalfKA_hm)
superbatch 1 / 40   pos = ... pos/s = ...   loss = ...
superbatch 2 / 40   ...
```

`pos/s` (1 秒あたり処理局面数) が学習速度の目安。RTX 4090 1 枚で smoke test 構成なら数千万 pos/s 出る。下位 GPU では比例して低下。

## 2.4 出力を確認する

学習完了 (または checkpoint 保存) のたびに、`checkpoints/my-first-shogi-net/` に以下が出る:

```
my-first-shogi-net-final/
├── raw.bin                ← float 重み (ここから再開可能)
├── quantised.bin          ← 整数重み (rshogi 互換)
└── optimiser_state/
    ├── weights.bin
    ├── moment1.bin
    └── ...
```

- `quantised.bin` を対局時にエンジンが読む
- `raw.bin` と `optimiser_state/` の組み合わせで、ここから学習を厳密に再開できる

## 2.5 エンジンに組み込む

具体的な手順はエンジンに依存する。やねうら王互換 NNUE 消費の典型手順:

1. 必要なら `quantised.bin` をエンジンが想定する NN ファイル形式に変換する (BulletOu は rshogi 互換レイアウトで書く。やねうら王が読むには薄い変換層が要るかもしれない)
2. エンジンが探す場所にファイルを置く
3. ちょっとした対局や `bench` でロードできることを確認

> エンジンへの組み込み自体は現状 BulletOu の範囲外 — トレーナーの仕事は `quantised.bin` を書くところで終わる。特定エンジンへ繋ぐのはエンジンごとの作業。

## 2.6 本番構成にステップアップする

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

各部品 (Threat 特徴量、`progress.bin`、WDL スケジュール) は [リファレンス](../0-contents.md) で説明されている。`shogi_simple` を「全部正しく繋がっているか」の確認用に使い、その後 `shogi_layerstack` で本格イテレーションを回す、というのが推奨フロー。

## 2.7 次のステップ

- [リファレンス: NNUE の基礎](../1-basics.md) — perspective NNUE の数式
- [リファレンス: 学習済みネットワーク](../4-saved-networks.md) — checkpoint レイアウト、量子化、変換チェーン
- [リファレンス: KP 絶対進行度](../shogi/kp-absolute-progress.md) — `--bucket-mode progress8kpabs` が実際に何をやっているか
- [リファレンス: shogi_progress_kpabs_train](../shogi/shogi_progress_kpabs_train.md) — 自分で `progress.bin` を学習する方法
- [3. KPPT / KPP_KPPT ロードマップ](3-kppt-roadmap.md) — 旧評価関数対応の計画
