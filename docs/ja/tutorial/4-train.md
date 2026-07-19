# 4. 学習を走らせる — `bulletou` コマンドの実行

<a href="../../en/tutorial/4-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 用意した教師データから、やねうら王互換エンジンが読み込める評価関数バイナリを学習する。

この章は [3. 教師データを用意する](3-data.md) を完了している前提 — 教師ファイル (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) が用意され、事前にシャッフルされている状態。

## 4.1 ビルド (1 回だけ)

まず `bulletou` をビルドする。ソースに変更が無ければ初回 1 回だけで OK:

```bash
cargo build --release --features cuda-cpp-backend --example bulletou
```

Windows の場合、生成されるバイナリは `.\target\release\examples\bulletou.exe`。以下のコマンド例は Unix 形式で書くので適宜読み替え。

## 4.2 最小コマンド (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

これだけで動く。`--output` を省略しているので、checkpoint は `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/` 配下に書かれる (`--eval-type` と `--arch` の値から自動命名)。別の場所に書きたい場合は `--output checkpoints/my-halfkp` のように明示する。

## 4.3 `--arch` を指定する

NNUE / SFNN 系 eval-type では `--arch` に **やねうら王の Makefile edition 名から `YANEURAOU_ENGINE_` を取り除いた名前**を指定する。たとえば HalfKP の 256x2-32-32 なら `NNUE_halfkp_256x2_32_32`、K-P の 256x2-32-32 なら `NNUE_kp_256x2_32_32`、SFNN なら `SFNN_halfka2_1024_7_64_k3k3` のように書く。古い短縮形 `256x2-32-32` は受け付けない。

NNUE 系のサイズ部分は `<L1>x2_<L2>_<L3>` で、`L1` (perspective ごとの accumulator サイズ) は **32 の倍数** (FT SIMD パディング要件) で正の整数、`L2` / `L3` は正の整数なら何でも受け付ける。よく使われるサイズは以下:

| サイズサフィックス | L1 (accumulator) | L2 | L3 | 用途の目安 |
|---|---|---|---|---|
| `256x2-32-32` (デフォルト) | 256 | 32 | 32 | 古典的な小型 NNUE。学習時間が短く挙動確認向き |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | 中型 |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | 大型 (推論コストは増える) |
| `1024x2-8-64` | 1024 | 8 | 64 | 大型 |
| `SFNN_halfkahm2_1536_15_32_k3k3` | 1536 | 15 | 32 | k3k3(king3-by-king3) LayerStacks の SFNN-1536 |

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --arch NNUE_halfkp_1024x2_8_64 \
    --teacher teachers/
```

`--arch` を省略すると eval-type ごとのデフォルトが適用される。たとえば `NNUE_HALFKP` は `NNUE_halfkp_256x2_32_32`、`NNUE_KP` は `NNUE_kp_256x2_32_32`。上記の表に無いサイズも実験用途で受け付けるが、学習結果の `nn.bin` を load できるのは「同じ architecture ヘッダで build したやねうら王」だけ。`make` に対応する edition 名を渡してビルドする必要がある (詳細は [§8 Engine](8-engine.md))。

## 4.4 SFNN-1536 (やねうら王 NNUEwoSQPT1536) を学習する

やねうら王の **`YANEURAOU_ENGINE_SFNN1536` ビルド** に load させる評価関数を学習したい場合は、専用の `--eval-type` を使う:

```bash
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

通常の NNUE と違って **9 個のサブネットを局面ごとに使い分ける** (LayerStacks=9)。`--arch` の `k3k3` suffix がやねうら王互換の LayerStack 方式を選ぶ。使い方の説明は [§9 LayerStack](9-layerstack.md)、アーキテクチャ / 量子化 / `nn.bin` レイアウトの仕様は [リファレンス: SFNN-1536](../shogi/sfnn-1536.md)。

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

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--superbatches` で epoch 長を決めたうえで `--max-epochs 3` のように指定する。`step` / `geometric` / `cos` は epoch 境界で `--lr` に戻る。

教師サイズが事前にわかっていると `--superbatches N` で「1 epoch = N sb」を明示できる ([§6.1 学習スケジュール](6-tune.md#61-学習スケジュール) 参照)。教師の総局面数を一瞬で数える `--count-teacher` フラグがある:

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
# → "Total: 461373440 positions, suggested --superbatches 4"
```

cosine annealing (`--lr-schedule cos`) を使うときは特に重要 — 1 cycle が 1 epoch とぴったり合うように `--superbatches` を選ぶと、各 epoch 末で lr_min に着地、次 epoch 頭で lr_max に warm restart、というきれいなサイクルになる。この場合、教師データ自体は epoch 境界で先頭へ戻らない。教師EOFに到達したときだけ先頭へ戻る cyclic stream として扱われる。

## 4.8 期待される出力

動いていれば以下のような出力が流れる:

```
=== bulletou: running NNUE_HALFKP (256x2-32-32 ClippedReLU, dual-perspective) ===
Training Preamble
Net Name               : shogi_nnue_halfkp
Batch Size             : 65536
Batches / Superbatch   : 1525
Positions / Superbatch : 99942400
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
