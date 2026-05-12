# 2. 学習を走らせる — 実データで評価関数を作る

<a href="../../en/tutorial/2-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: やねうら王互換エンジンが読み込める評価関数バイナリを、実際の教師データから学習する。

この章は [1. クイックスタート](1-quickstart.md) を完了している前提 — ツールチェーンが動き、smoke test の学習が成功した状態。

本チュートリアルでは **NNUE HalfKP を例に** 解説するが、`--eval-type` を切り替えるだけで他のターゲット (NNUE K-P / NNUE HalfKPE9 / KPPT / KPP_KKPT) も同じコマンド形式で学習できる。

## 2.1 学習対象を選ぶ

`bulletou --eval-type <X>` で学習する評価関数を選ぶ。現在公開されている `<X>`:

| `--eval-type` | 何を学習するか | 出力ファイル (per save) | `--arch` を使うか |
|---|---|---|---|
| **`NNUE_HALFKP`** ★初心者はここから | 古典的な HalfKP NNUE。やねうら王がもっとも長く採用している評価関数形式。詳細は [NNUE HalfKP 学習](../shogi/halfkp.md) | `nn.bin` | 使う |
| `NNUE_KP` | HalfKP と同じ NN だが入力が K + P の独立特徴。詳細は [NNUE K-P 学習](../shogi/kp.md) | `nn.bin` | 使う |
| `NNUE_HALFKPE9` | HalfKP に利き数情報 (自軍/敵軍 0/1/2 の 9 通り) を多重化した拡張版。詳細は [NNUE HalfKPE9 学習](../shogi/halfkpe9.md) | `nn.bin` | 使う |
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

### 教師ファイルは事前にシャッフルしておく

> ⚠️ **重要**: 教師ファイルは BulletOu に渡す前にシャッフルしておくこと。

bullet のローダーは **メモリ内 shuffle buffer (デフォルト 256 MB ≒ HCPE で 670 万局面)** で局面を Fisher-Yates シャッフルしてから batch に切り出す。これは **buffer 内シャッフルのみ** で buffer をまたいだクロスシャッフルはしないため、教師ファイル先頭から順に 670 万局面ずつ読みながら局所的にシャッフルしているだけになる。

`gensfen` / dlshogi-style 生成器の出力は **同一対局内の局面が連続して並んでいる** のが普通なので、ファイル全体をシャッフルしないまま学習すると、buffer 境界 (≒ 410 batch ごと、`--batch-size 16384` の場合) で分布が突然変わって **loss が周期的に跳ねる** ことになる。

対策:
- **`.hcpe` / `.hcpe3`**: dlshogi のシャッフルスクリプトを使うのが簡単。HCPE は固定長レコード (38 byte) なのでバイト単位のランダムシャッフルで OK。dlshogi リポジトリの `utils/` 配下にツールが用意されている。
- **`.pack`**: `gensfen` の出力時点でシャッフルオプションを有効にする、もしくは出力後に PSV に変換して shuffle。
- **緊急回避** (シャッフル前のファイルしか手元にないとき): `--buffer-mb` をファイルサイズと同等以上に上げて 1 バッファに全件収める。例: 1.94 GB の `.hcpe` (≒ 5100 万局面) なら `--buffer-mb 2048` で buffer 境界が無くなる。GPU メモリではなく **CPU 側 RAM** を食うので、メモリに余裕がある環境向け。

### 小さなサブセットで動作確認したい場合

巨大なデータセット (数十 GB) でいきなり動かす前に、小さなサブセットで試したいときは、`gensfen` 等で小さめのファイルを生成するか、`--batches-per-superbatch` を指定して 1 superbatch あたりの消費量を絞る ([§3.1](3-tune.md#31-学習スケジュール) 参照)。

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

`--arch` を省略するとデフォルト `256x2-32-32` が適用される。`NNUE_KP` / `NNUE_HALFKPE9` でも同じ preset 群が指定可能。

(`halfkpvm` のように **入力特徴量自体が違う variant**、および `SFNNwoPSQT1536` は別 `--eval-type` として今後追加予定。`--arch` だけでは到達できない。)

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

## 2.4 中断・再開

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
- `learn.log` (累積版) には新 run の CSV 行がそのまま追記される。LR scheduler は run ごとに reset されるため superbatch カウンタは 1 から再開するが、`positions` 列は累積される (新 run 開始時に既存 `learn.log` の最大 positions を読み取って続きから書く)

この挙動は eval-type 横断 (KPPT / KPP_KKPT / NNUE_HALFKP / NNUE_KP / NNUE_HALFKPE9 すべて同じ仕組み)。新規学習にしたい場合は `--output` を別の dir にするか、既存 dir を削除する。

---

次へ:
- [3. 学習をチューニング](3-tune.md) — `--lambda`、`--lr`、`--superbatches` 等で学習を調整する (任意)
- 学習結果がもう手元にあるなら [4. 結果を確認・活用する](4-result.md) へ
