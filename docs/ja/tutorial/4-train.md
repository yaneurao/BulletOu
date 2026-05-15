# 4. 学習を走らせる — `bulletou` コマンドの実行

<a href="../../en/tutorial/4-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 用意した教師データから、やねうら王互換エンジンが読み込める評価関数バイナリを学習する。

この章は [3. 教師データを用意する](3-data.md) を完了している前提 — 教師ファイル (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) が用意され、できれば事前にシャッフルされている状態。

## 4.1 ビルド (1 回だけ)

まず `bulletou` をビルドする。ソースに変更が無ければ初回 1 回だけで OK:

```bash
cargo build --release --features device-cuda --example bulletou
```

(AMD GPU なら `--features device-cuda` を `--features device-rocm` に。Windows の場合、生成されるバイナリは `.\target\release\examples\bulletou.exe`。以下のコマンド例は Unix 形式で書くので適宜読み替え。)

## 4.2 最小コマンド (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

これだけで動く。`--output` を省略しているので、checkpoint は `checkpoints/NNUE_HALFKP-256x2-32-32/` 配下に書かれる (`--eval-type` と `--arch` の値から自動命名)。別の場所に書きたい場合は `--output checkpoints/my-halfkp` のように明示する。

## 4.3 `--arch` を指定する

NNUE 系 eval-type ではネットワーク層サイズを `--arch <L1>x2-<L2>-<L3>` で **自由に**指定できる。`L1` (perspective ごとの accumulator サイズ) は **32 の倍数** (FT SIMD パディング要件) で正の整数、`L2` / `L3` は正の整数なら何でも受け付ける。

やねうら王が配布しているエンジンバイナリのディレクトリ名 (`NNUE_halfkp_*` のサフィックス) と一致するよく使われるサイズは以下:

| `--arch` | L1 (accumulator) | L2 | L3 | 用途の目安 |
|---|---|---|---|---|
| `256x2-32-32` (デフォルト) | 256 | 32 | 32 | 古典的な小型 NNUE。学習時間が短く挙動確認向き |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | 中型 |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | 大型 (推論コストは増える) |
| `1024x2-8-64` | 1024 | 8 | 64 | 大型 |
| `1536x2-15-32` | 1536 | 15 | 32 | SFNN-1536 (`architectures/sfnnwop-1536.h` 一致) |

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --arch 1024x2-8-64 \
    --teacher teachers/
```

`--arch` を省略するとデフォルト `256x2-32-32` が適用される。同じ `--arch` フラグが全ての NNUE / SFNN eval-type (`NNUE_HALFKP`, `NNUE_KP`, `NNUE_KA2`, `NNUE_HALFKPE9`, `NNUE_HALFKPVM`, `SFNN_*`) で使える。上記の表に無いサイズ (例: `--arch 256x2-64-64`) も実験用途で受け付けるが、学習結果の `nn.bin` を load できるのは「同じ triple のアーキテクチャヘッダで build したやねうら王」だけ。`make` に対応する edition 名を渡してビルドする必要がある (詳細は [§8 Engine](8-engine.md))。

## 4.4 SFNN-1536 (やねうら王 NNUEwoSQPT1536) を学習する

やねうら王の **`YANEURAOU_ENGINE_NNUE_SFNNwoP1536` ビルド** に load させる評価関数を学習したい場合は、専用の `--eval-type` を使う:

```bash
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch 1536x2-15-32 \
    --layerstack king3-by-king3 \
    --teacher teachers/
```

通常の NNUE と違って **9 個のサブネットを局面ごとに使い分ける** (LayerStacks=9) ので `--layerstack` フラグが要る点だけ毛色が違う。使い方の説明は [§9 LayerStack](9-layerstack.md)、アーキテクチャ / 量子化 / `nn.bin` レイアウトの仕様は [リファレンス: SFNN-1536](../shogi/sfnn-1536.md)。

## 4.5 KPPT を学習する

KPPT 系では `--arch` 不要 (architecture は固定):

```bash
./target/release/examples/bulletou \
    --eval-type KPPT \
    --teacher teachers/
```

デフォルト出力先は `checkpoints/KPPT/`。factorised 版にしたければ `--eval-type KPP_KKPT` に変えるだけ。

## 4.6 教師データの渡し方

`--teacher` には:
- 1 つのファイル (`teachers/teacher.pack` のようなフルパス)
- ディレクトリ (上記例。中の同一拡張子ファイルがすべて連結される)
- カンマ区切り複数指定

のいずれも渡せる。

## 4.7 学習がどこまで進むか

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--max-epochs 3` のように指定する (各 epoch 開始時に LR がリセットされる)。

## 4.8 期待される出力

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

---

次へ:
- 学習を中断したり再開したい場合は [5. 中断・再開](5-resume.md)
- 学習のスケジュールや教師ターゲットを調整したい場合は [6. 学習をチューニング](6-tune.md)
- 学習結果がもう手元にあるなら [7. 結果を確認](7-result.md) へ

前へ: [3. 教師データを用意する](3-data.md)
