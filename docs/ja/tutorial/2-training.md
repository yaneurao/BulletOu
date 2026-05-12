# 2. 学習を走らせる — 実データで評価関数を作る

<a href="../../en/tutorial/2-training.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: やねうら王互換エンジンが読み込める評価関数バイナリを、実際の教師データから学習する。

この章は [1. クイックスタート](1-quickstart.md) を完了している前提 — ツールチェーンが動き、smoke test の学習が成功した状態。

本チュートリアルでは **NNUE HalfKP を例に** 解説するが、`--eval-type` を切り替えるだけで他のターゲット (NNUE K-P / KPPT / KPP_KKPT) も同じコマンド形式で学習できる。

## 2.1 学習対象を選ぶ

`bulletou --eval-type <X>` で学習する評価関数を選ぶ。現在公開されている `<X>`:

| `--eval-type` | 何を学習するか | 出力ファイル (per save) | `--arch` を使うか |
|---|---|---|---|
| **`NNUE_HALFKP`** ★初心者はここから | 古典的な HalfKP NNUE。やねうら王がもっとも長く採用している評価関数形式。詳細は [NNUE HalfKP 学習](../shogi/halfkp.md) | `nn.bin` | 使う |
| `NNUE_KP` | HalfKP と同じ NN だが入力が K + P の独立特徴。詳細は [NNUE K-P 学習](../shogi/kp.md) | `nn.bin` | 使う |
| `KPPT` | 旧来の KK + KKP + KPP 3 ファイル組 (elmo(WCSC27) 互換)。詳細は [KPPT / KPP_KKPT 学習](../shogi/kppt.md) | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` | 使わない |
| `KPP_KKPT` | KPPT の factorised 版 (KPP のみ手番チャンネルなし、サイズ半減) | 同上 (KPP layout のみ違う) | 使わない |

将来 `--eval-type` に追加予定: HalfKA / SFNN + ls9 (NNUEwoSQPT1536) など。

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

## 2.3 学習を走らせる

### ビルド (1 回だけ)

まず `bulletou` をビルドする。ソースに変更が無ければ初回 1 回だけで OK:

```bash
cargo build --release --features device-cuda --example bulletou
```

(AMD GPU なら `--features device-cuda` を `--features device-rocm` に。Windows の場合、生成されるバイナリは `.\target\release\examples\bulletou.exe`。以下のコマンド例は Unix 形式で書くので適宜読み替え。)

### 最小コマンド (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

これだけで動く。`--output` を省略しているので、checkpoint は `checkpoints/NNUE_HALFKP-256x2-32-32/` 配下に書かれる (`--eval-type` と `--arch` の値から自動命名)。別の場所に書きたい場合は `--output checkpoints/my-halfkp` のように明示する。

### `--arch` を指定する

NNUE 系 eval-type ではネットワーク層サイズを `--arch <L1>x2-<L2>-<L3>` で選ぶ。やねうら王が配布しているエンジンバイナリのディレクトリ名 (`NNUE_halfkp_*` のサフィックス) に揃えてあり、以下が選択可能:

| `--arch` | L1 (accumulator) | L2 | L3 | 用途の目安 |
|---|---|---|---|---|
| `256x2-32-32` (デフォルト) | 256 | 32 | 32 | 古典的な小型 NNUE。学習時間が短く挙動確認向き |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | 中型 |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | 大型 (推論コストは増える) |
| `1024x2-8-64` | 1024 | 8 | 64 | 大型 |

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --arch 1024x2-8-64 \
    --teacher teachers/
```

`--arch` を省略するとデフォルト `256x2-32-32` が適用される。`NNUE_KP` でも同じ preset 群が指定可能 (やねうら王が配布しているのは `NNUE_kp_256x2_32_32` のみだが、学習側は他 preset でも生成可能)。

(`halfkpe9` / `halfkpvm` のように **入力特徴量自体が違う variant**、および `SFNNwoPSQT1536` は別 `--eval-type` として今後追加予定。`--arch` だけでは到達できない。)

### KPPT を学習する

KPPT 系では `--arch` 不要 (architecture は固定):

```bash
./target/release/examples/bulletou \
    --eval-type KPPT \
    --teacher teachers/
```

デフォルト出力先は `checkpoints/KPPT/`。factorised 版にしたければ `--eval-type KPP_KKPT` に変えるだけ。

### 教師データの渡し方

`--teacher` には:
- 1 つのファイル (`teachers/teacher.pack` のようなフルパス)
- ディレクトリ (上記例。中の同一拡張子ファイルがすべて連結される)
- カンマ区切り複数指定

のいずれも渡せる。

### 学習がどこまで進むか

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--max-epochs 3` のように指定する (各 epoch 開始時に LR がリセットされる)。

### 期待される出力

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

## 2.4 学習スケジュール (必要になったら戻ってきて読む)

**最初は全部デフォルト値で問題ない**。教師のサイズや学習リソースに応じて調整したくなったら、このセクションに戻る。

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
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --superbatches 40
```

教師ファイルが 1 superbatch 未満 (≒ 1 億局面未満) しか無い場合は `--batches-per-superbatch` を小さくする (例: `1024` で 1 superbatch ≒ 1670 万局面) と、何回も save が走るようになる。

## 2.5 出力を確認する

学習完了後、出力ディレクトリ (例: `checkpoints/NNUE_HALFKP-256x2-32-32/`) は以下のレイアウト:

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

最新の `000N/` (= 最大番号のディレクトリ) がエンジンに渡す成果物が入ったフォルダ。

KPPT / KPP_KKPT の場合は `nn.bin` の代わりに `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` の 3 ファイル組が `000N/` 配下に入る (3 ファイル全部必要)。

## 2.6 中断・再開

学習途中で `Ctrl+C` で止めたり、マシンの再起動などで中断しても、**同じ `--output` で同じコマンドをもう一度実行するだけで、自動的に最新 `000N/state.bin` から学習が続行される**。

```
checkpoints/.../
├── 0001/             ← 前回の最初の save
├── 0002/
├── 0003/             ← 中断時点で最新だった save
├── 0004/             ← 再開後ここから書かれる
└── 0005/
```

仕組み:
- `bulletou` 起動時、`--output` 配下に番号付き dir + `state.bin` があれば検出
- 最大番号の `state.bin` から重みと Adam moments を復元
- 新 save は既存最大番号の次から書く (前例で `0003/` まであれば `0004/` から)
- `learn.log` (累積版) には新 run 用の section が追記される (LR scheduler が reset されるため superbatch カウンタは 1 から再開)

この挙動は eval-type 横断 (KPPT / KPP_KKPT / NNUE_HALFKP / NNUE_KP すべて同じ仕組み)。新規学習にしたい場合は `--output` を別の dir にするか、既存 dir を削除する。

## 2.7 エンジンに組み込む

学習結果をやねうら王エンジンで動作確認する最小手順。

### NNUE 系 (`nn.bin`)

最新の `000N/nn.bin` をエンジンが探す場所に置く。やねうら王の場合、`EvalDir` オプションでパスを指定する:

```
# エンジン起動後、USI コマンドで:
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/NNUE_HALFKP-256x2-32-32/0005
isready
bench
```

または、`eval/nn.bin` という相対パスでエンジン側に置く場合は、`000N/nn.bin` をそのファイル名で配置する。

`isready` でロードが通れば学習結果が認識できている。`bench` の出力に nn.bin のハッシュが出るので、毎回違う数字になっていれば確かに違う重みを load していることが分かる。

### KPPT 系 (`KK_synthesized.bin` 等の 3 ファイル組)

最新 `000N/` ディレクトリそのものを `EvalDir` に指定する (3 ファイルが揃った状態のディレクトリを指す):

```
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/KPPT/0005
isready
bench
```

3 ファイルすべてが揃っていない場合エンジンは load に失敗する点に注意。

### 学習結果が弱いとき

最初の学習は小さな教師で短い superbatch しか回していないので、評価の質はあまり期待しないこと。本格対局できるレベルにするには:
- 教師サイズを増やす (1 億 → 10 億局面以上)
- `--max-epochs 3` 程度で複数周回す
- `--save-rate` を大きく (例: 10) して、後半の save だけを使う

詳細なハイパーパラメータ調整は各 eval-type のリファレンス ([halfkp.md](../shogi/halfkp.md) / [kp.md](../shogi/kp.md) / [kppt.md](../shogi/kppt.md)) を参照。

## 2.8 次のステップ

- [リファレンス: NNUE HalfKP 学習](../shogi/halfkp.md) — `nn.bin` のバイナリレイアウト、量子化、resume の詳細
- [リファレンス: NNUE K-P 学習](../shogi/kp.md) — HalfKP との比較、入力 feature の構造
- [リファレンス: KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習
- [仕様: spec/](../../../spec/) — eval-type 一覧 / バイナリレイアウト / hash 計算式
