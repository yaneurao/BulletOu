# shogi_progress_kpabs_train

<a href="../../en/shogi/shogi_progress_kpabs_train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

LayerStack の bucket 選択に使う `progress.bin`（KP-absolute 進行度モデル）を学習するツール。
現在保守しているのは CPU 版のみ。旧 `bullet-gpu` CUDA 版は、BulletOu の学習 backend を `cuda-cpp` に移行したタイミングで削除した。

生成した `progress.bin` は LayerStack 学習で
`--bucket-mode progress8kpabs --progress-coeff progress.bin` として参照する。

---

## モデル

KP-absolute 特徴量（玉位置 × 駒配置）に対する線形ロジスティック回帰。

```
z      = Σ weights[kp_abs_index]
p      = sigmoid(z)
bucket = clamp(floor(p * 8), 0, 7)   // 結果は 0〜7 の 8 値
```

- 重みの数: `81 × 1548 = 125,388`（玉位置 × `Eval::BonaPiece::fe_end`）

### 出力ファイル形式

サイズ **1,003,104 bytes**（= `8 × 81 × 1548`）。
要素は `f64` little-endian、配列レイアウトは `weights[sq][bona_piece]`。

モデル設計・BonaPiece 番号付け・ファイル形式の出典
（[`yaneurao/YaneuraOu`](https://github.com/yaneurao/YaneuraOu) の `old_engines/eval/progress/`
および [`nodchip/nnue-pytorch`](https://github.com/nodchip/nnue-pytorch) の `tanuki_progress.cpp`）は
[`kp-absolute-progress.md`](kp-absolute-progress.md) を参照。

---

## 教師値モード

### 厳密モード (`--game-relative`、推奨)

```
y = game_ply / (total_ply - 1)
```

- 各対局の実際の総手数で正規化
- 対局境界は `game_ply` が前レコード以下になったら新対局として検出
- **対局順保持データ（シャッフル前）が必須**
- bucket 分布が均一になりやすい

### 近似モード（CPU 版デフォルト）

```
y = clamp((game_ply - 1) / (ply_max - 1), 0, 1)
```

- 固定の `ply_max` で正規化（CLI `--ply-max`、既定 256）
- シャッフル済みデータでも使える
- `ply_max` の選び方によっては bucket 分布が後半 / 前半に偏ることがある
  （実際の対局より長く `ply_max` を取りすぎると後半 bucket が枯れる）

---

## 実装バリアント

| 実装 | バイナリ名 | 教師値モード | 学習粒度 | バックエンド |
|---|---|---|---|---|
| CPU 版 | `shogi_progress_kpabs_train` | 近似 / 厳密 両対応 | 局面単位ミニバッチ（近似）/ 1 game = 1 step（厳密） | 単スレッド CPU |

削除済みの CUDA 版も同じ凸最適化問題を解いていたが、旧 `bullet-gpu` runtime 依存だったため現在はビルド対象外。

ビルド:

```bash
cargo build --release --example shogi_progress_kpabs_train
```

---

## データの流れ

```
棋譜生成 → raw.psv（対局順保持、シャッフル前）
              ├→ progress.bin 学習 ← ここで raw.psv を使う
              └→ リスコア → シャッフル → train.psv → NNUE学習
```

- 進行度学習にスコア（評価値）は使わない。局面の駒配置（KP 特徴量）のみ使用
- qsearch leaf 置換が入る前のデータを使う（教師スコア向けの leaf 置換は駒配置を変えうるため）

### データ供給とファイル分割

`--data` には CSV または `.bin` / `.pack` を含むディレクトリを渡せる。
ディレクトリ指定時は直下の `*.bin` / `*.pack` のみが対象。

ファイル列挙後、`pack_group_key()` でファイル名のプレフィックス
（`hao_depth_9_shuffled_*`、`shuffled_*`、それ以外は file stem ごとに別グループ）に
基づくグループを作り、`interleave_pack_groups()` で各グループから 1 つずつ
取り出す **round-robin 並べ替え** を行う。

| バリアント / モード | データ走査 | val/train 分割 |
|---|---|---|
| CPU 版・近似モード | `RoundRobinPackStream`：ファイル間を 1 レコードずつ round-robin | val_positions（先頭から N 局面）→ 残り max_positions が train（同一ストリーム）|
| CPU 版・厳密モード | `MultiFileGameIterator`：interleave 後の順序でファイル順次走査、対局単位で yield | **先頭 5%**（`packs.len() / 20`）を val に、残りを train |

### val 自動分割の落とし穴

ファイル分割は `interleave_pack_groups` 後の決定的順序に基づくため、
データセットの構成によっては val 側に**特定の対局種別**が偏ることがある。

例えば「通常自己対局ファイル群」と「特殊局面（入玉等）特化ファイル群」を
連結して与えた場合、グループ key が異なれば末尾（または先頭）に
特定種別が集中して置かれ、val_loss と train の分布が乖離する可能性がある。

回避策:

- ファイル名規則を統一して `pack_group_key` が同一グループを返すようにする
- もしくは `--val-games` を明示し、必要なら学習対象データセット側を
  事前にシャッフル（=複数ソース混合バイナリを生成）する

---

## パラメータ

### CPU 版 (`shogi_progress_kpabs_train`)

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--data` | (必須) | カンマ区切りのファイルまたはディレクトリ |
| `--output` | (必須) | 出力 `progress.bin` のパス |
| `--game-relative` | false | 厳密モード。対局順保持データが必要 |
| `--max-positions` | 50,000,000 | 1 epoch あたりの学習サンプル数（近似モード） |
| `--val-positions` | 2,000,000 | 検証用サンプル数（近似モード、ストリーム先頭から取得） |
| `--batch-size` | 4,096 | ミニバッチサイズ（近似モード） |
| `--lr` | 0.0002 | Adam の学習率 |
| `--epochs` | 1 | 学習パス数 |
| `--ply-max` | 256 | 近似モードの正規化上限（`--game-relative` 時は無視） |
| `--log-interval` | 100 | バッチごとのログ出力間隔（近似モード） |
| `--max-games` | 0 (無制限) | 厳密モードの 1 epoch あたり学習対局数 |
| `--val-games` | 0 (auto) | 厳密モードの val ファイル走査時の最大対局数 |
| `--log-interval-games` | 1,000 | 対局ごとのログ出力間隔（厳密モード） |
| `--save-each-epoch` | false | 各 epoch 後に `<output_stem>.eN.<ext>` を追加保存 |

## コマンド例

### CPU 版・厳密モード

```bash
cargo run --release --example shogi_progress_kpabs_train -- \
  --data raw.psv \
  --output progress.bin \
  --game-relative \
  --max-games 0 \
  --val-games 0 \
  --epochs 1 \
  --lr 0.001 \
  --save-each-epoch
```

`--max-games 0` で全データを 1 周。`--val-games 0` は「val ファイル群を全走査」の意味。

### CPU 版・近似モード

```bash
cargo run --release --example shogi_progress_kpabs_train -- \
  --data train_shuffled.bin \
  --output progress.bin \
  --max-positions 50000000 \
  --val-positions 2000000 \
  --batch-size 4096 \
  --lr 0.0002 \
  --epochs 1 \
  --ply-max 256
```

シャッフル済みデータが既にある場合に使う。`--ply-max` は実データの対局長に合わせて調整する。

## NNUE 学習での使用

```bash
cargo run --release --example shogi_layerstack -- \
  --data train.psv \
  --bucket-mode progress8kpabs \
  --progress-coeff progress.bin \
  ...
```

> 学習時に与えた `progress.bin` と推論時に使う `progress.bin` は**一致させる**こと。
> 異なる `progress.bin` を使うと bucket 割当が変わり、学習済み NN の重みと不整合になる。

---

## 関連ドキュメント

- [`kp-absolute-progress.md`](kp-absolute-progress.md) — KP-absolute 進行度モデル（数学・bullet-shogi 内 wiring・ファイル形式の出典）
