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

## 2.4 学習の単位 — batch / superbatch / save / LR の関係

ログに出てくる `superbatch 1 / 40` の「superbatch」は何か、`--batch-size` と `--batches-per-superbatch` と `--superbatches` がそれぞれ何を意味するのか、ここで一括して整理する。これは本家 jw1912/bullet 由来の概念で、bullet-shogi / BulletOu でもそのまま使われている。

### 2.4.1 3 つの単位

```
batch (= mini-batch, 1 gradient step)
  └─ 16384 局面 を 1 回 forward + backward + optimizer step
        │
        │ × batches_per_superbatch 回
        ▼
superbatch
  └─ デフォルト ≈ 100M 局面 (= 6104 batches × 16384 局面/batch)
        │
        │ × superbatches 回
        ▼
学習全体 (end_superbatch まで)
```

| CLI フラグ | 意味 | デフォルト |
|---|---|---|
| `--batch-size` | 1 gradient step の局面数 (= mini-batch size)。GPU メモリと収束特性を決める | `16384` |
| `--batches-per-superbatch` | 1 superbatch を構成する mini-batch 数。**未指定なら `ceil(100_000_000 / batch_size)`** が入る | (自動) |
| `--superbatches` | 走らせる superbatch の総数 (= `end_superbatch`)。学習全体の長さを決める | 例によるが KK/KKP 例では `10` |

`batches_per_superbatch` のデフォルト式は **「1 superbatch ≒ 1 億局面に揃える」** 設計。`--batch-size` をいじっても 1 superbatch あたりの局面数 (≈100M) はほぼ不変になる。これは本家 bullet のチェス NNUE 文化での暗黙のスケール感で、`bullet/examples/progression/1_simple.rs` 等では `batches_per_superbatch: 6104` がハードコードされている。

### 2.4.2 superbatch は「epoch」と同じか?

**完全には一致しない**。標準 ML 用語の epoch は「データセット全体を 1 周」だが、bullet の superbatch は **データセットサイズに関係なく「~100M 局面」固定**。

- 教師データが 50M 局面しかなければ、1 superbatch でデータ 2 周ぶん回す (loader が末尾に達したら頭から再シャッフルする実装)
- 教師データが 1B 局面あれば、1 superbatch でデータの 1/10 しか触れない

実用上は **「checkpoint / LR / WDL の更新タイミングの単位」** と捉えるのが正確。

### 2.4.3 checkpoint 保存タイミング — `--save-rate`

```
--save-rate 1   →  毎 superbatch で保存
--save-rate 5   →  5 superbatch ごとに保存
--save-rate 0   →  最終 superbatch のみ保存 (途中保存なし)
```

各保存ポイントで `checkpoints/<net-id>-<superbatch>/` ディレクトリが作られ、その中に `raw.bin` / `quantised.bin` / `optimiser_state/` (＋ KPPT 系の例では `KK_synthesized.bin` / `KKP_synthesized.bin`) が書き出される。

最終 superbatch (`end_superbatch`) は `--save-rate` の値に関わらず必ず保存される (`should_save` の OR 条件)。

### 2.4.4 LR スケジューラの時間軸

bullet の LR スケジューラはすべて `lr(batch, superbatch) -> f32` で値を返す関数で、**ほぼすべて `superbatch` をキー**にする (`Warmup` だけ batch 軸も使う)。詳細:

| スケジューラ | 挙動 | CLI | 終端 superbatch 要求 |
|---|---|---|---|
| `ConstantLR` | 固定 | (該当フラグなし) | × |
| `StepLR` | `step` superbatch ごとに `gamma` 倍 | `--lr` / `--lr-gamma` / `--lr-step` (現状 KK/KKP 例で使用) | × |
| `DropLR` | `drop` superbatch で 1 回だけ `gamma` 倍 | — | × |
| `LinearDecayLR` | `final_superbatch` まで線形補間 | — | **要** |
| `CosineDecayLR` | 同、cosine 曲線 | — | **要** |
| `ExponentialDecayLR` | 同、指数補間 | — | **要** |
| `Warmup<LR>` | 最初の N batch だけ線形立ち上げ後、内側スケジューラへ | — | 内側依存 |

`shogi_kk_kkp_train` のデフォルト `--lr 0.001 --lr-gamma 0.1 --lr-step 8` だと、1〜8 superbatch は `0.001`、9〜16 は `0.0001`、17〜24 は `0.00001`、... と毎 8 superbatch で 1/10 になる。`--superbatches 3` だと一度も下がらず学習が終わる。

### 2.4.5 WDL スケジューラの時間軸

WDL (Win/Draw/Loss = ゲーム結果ラベルへの blend 比率) も `superbatch` を時間軸として動く。デフォルト:

```
--start-wdl 0.0  --end-wdl 1.0
```

これは「最初の superbatch は **eval スコアだけ**、最後の superbatch は **対局結果だけ**、間は線形補間」という指定。終端は `end_superbatch` (= `--superbatches`) を使うので、`--superbatches` を変えると WDL の傾きも自動で追従する。

### 2.4.6 具体例

```
--batch-size 16384
--batches-per-superbatch 100      ← 通常はもっと大きい (6104 がデフォルト)
--superbatches 3
--save-rate 1
--lr 0.001 --lr-gamma 0.1 --lr-step 8
--start-wdl 0.0 --end-wdl 1.0
```

意味:

```
1 superbatch  = 100 batches × 16384 局面 = 1,638,400 局面 (≈ 1.6M)
学習全体     = 3 superbatches × 1.6M     = 4,915,200 局面 (≈ 4.9M)
checkpoint   = sb=1, sb=2, sb=3 で計 3 回保存
LR           = 全 superbatch 通じて 0.001 (8 sb 毎の drop が発火しない)
WDL          = sb=1 で 0.0、sb=2 で 0.5、sb=3 で 1.0
```

なお、本格学習で「1 superbatch ≒ 100M 局面」スケールに戻すなら `--batches-per-superbatch` を **指定しない**のが正解 (= 自動で 6104 になる)。

## 2.5 出力を確認する

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

## 2.6 エンジンに組み込む

具体的な手順はエンジンに依存する。やねうら王互換 NNUE 消費の典型手順:

1. 必要なら `quantised.bin` をエンジンが想定する NN ファイル形式に変換する (BulletOu は rshogi 互換レイアウトで書く。やねうら王が読むには薄い変換層が要るかもしれない)
2. エンジンが探す場所にファイルを置く
3. ちょっとした対局や `bench` でロードできることを確認

> エンジンへの組み込み自体は現状 BulletOu の範囲外 — トレーナーの仕事は `quantised.bin` を書くところで終わる。特定エンジンへ繋ぐのはエンジンごとの作業。

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

各部品 (Threat 特徴量、`progress.bin`、WDL スケジュール) は [リファレンス](../0-contents.md) で説明されている。`shogi_simple` を「全部正しく繋がっているか」の確認用に使い、その後 `shogi_layerstack` で本格イテレーションを回す、というのが推奨フロー。

## 2.8 次のステップ

- [リファレンス: NNUE の基礎](../1-basics.md) — perspective NNUE の数式
- [リファレンス: 学習済みネットワーク](../4-saved-networks.md) — checkpoint レイアウト、量子化、変換チェーン
- [リファレンス: KP 絶対進行度](../shogi/kp-absolute-progress.md) — `--bucket-mode progress8kpabs` が実際に何をやっているか
- [リファレンス: shogi_progress_kpabs_train](../shogi/shogi_progress_kpabs_train.md) — 自分で `progress.bin` を学習する方法
- [3. KPPT / KPP_KPPT ロードマップ](3-kppt-roadmap.md) — 旧評価関数対応の計画
