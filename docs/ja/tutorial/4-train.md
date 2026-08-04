# 4. 学習を走らせる — `bulletou` コマンドの実行

<a href="../../en/tutorial/4-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 用意した教師データから、やねうら王エンジンが読み込める評価関数バイナリを学習する。

このページは [3. 教師データを用意する](3-data.md) を完了している前提です。教師ファイル (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) が用意され、教師局面が事前シャッフル済み、または `--teacher-shuffle-buffer-sbs` で学習時にシャッフルされる状態を想定する。

## 4.1 ビルドする

まず `bulletou` をビルドする。ソースを変更していなければ初回だけで十分だが、BulletOu のソースを更新した直後は `.\target\release\examples\bulletou.exe` に更新が反映されていないので、必ず再ビルドする。

```bash
cargo build --release --features cuda-cpp-backend --example bulletou
```

Windows の場合、生成されるバイナリは `.\target\release\examples\bulletou.exe`。以下のコマンド例は Unix 形式で書いているので、Windows では適宜読み替える。

## 4.2 最小コマンド (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_256x2_32_32 \
    --teacher teachers/
```

これだけで動く。`--output` を省略した場合、学習結果は `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/` 配下に書かれる。この名前は `--arch` から自動生成される。別の場所に出力したい場合は `--output checkpoints/my-halfkp` のように明示する。

## 4.3 `--arch` を指定する

学習対象は `--arch` だけで指定する。KPPT 系なら `KPPT` または `KPP_KKPT`、NNUE / SFNN 系なら、やねうら王の Makefile edition 名から `YANEURAOU_ENGINE_` を取り除いた名前を指定する。

たとえば HalfKP の 256x2-32-32 は `NNUE_halfkp_256x2_32_32`、K-P の 256x2-32-32 は `NNUE_kp_256x2_32_32`、SFNN なら `SFNN_halfka2_1024_7_64_k3k3` のように書く。短縮形 `256x2-32-32` は受け付けない。

NNUE 名のサイズ部分は `<L1>x2_<L2>_<L3>`。ざっくり言うと、`L1` が最初の大きな層、`L2` / `L3` が後段の小さな層です。`L1` は 32 の正の倍数にする必要があります。よく使う例は次の通り。

| サイズ・arch例 | L1 / FT | L2 | L3 | 備考 |
|---|---:|---:|---:|---|
| `256x2-32-32` | 256 | 32 | 32 | 古典的な小型 NNUE。学習時間が短く、挙動確認向き |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | 中型 |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | 大型。推論コストは増える |
| `1024x2-8-64` | 1024 | 8 | 64 | 大型 |
| `SFNN_halfkahm2_1536_15_32_k3k3` | 1536 | 15 | 32 | k3k3 (king3-by-king3) LayerStack の SFNN-1536 |
| `SFNN_halfka2_1024_7_64` | 1024 | 7 | 64 | 局面で分岐しない SFNN (`LayerStacks = 1`) |
| `SFNN_halfka2_4096_3_64_c0_s1024x4_k3k3` | 4096 | 3 | 64 | grouped SFNN L1。4096 を 4 group に分け、各 group が 1024 -> 1 |
| `SFNN_halfka2_8192_3_64_c0_s2048x4_k3k3` | 8192 | 3 | 64 | grouped SFNN L1。8192 を 4 group に分け、各 group が 2048 -> 1 |
| `SFNN_halfka2_4096_7_64_c0_s1024x4_k3k3` | 4096 | 7 | 64 | grouped SFNN L1。4096 を 4 group に分ける |
| `SFNN_halfka2_1024_7_64_hand64` | 1024 | 7 | 64 | やねうら王 hand64 LayerStack bucket (64 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k3k3` | 1024 | 7 | 64 | hand64 × k3k3 LayerStack bucket (576 stacks。かなり大きい) |
| `SFNN_halfka2_1024_7_64_k9k9` | 1024 | 7 | 64 | king9-by-king9 LayerStack bucket (81 stacks) |
| `SFNN_halfka2_1024_7_64_k9k9z` | 1024 | 7 | 64 | king9-zone-by-king9-zone LayerStack bucket (81 stacks) |
| `SFNN_halfka2_1024_7_64_k13k13z` | 1024 | 7 | 64 | king13-zone-by-king13-zone LayerStack bucket (169 stacks) |
| `SFNN_halfka2_1024_7_64_k21k21` | 1024 | 7 | 64 | king21-by-king21 LayerStack bucket (441 stacks) |
| `SFNN_halfka2_1024_7_64_k29k29` | 1024 | 7 | 64 | king29-by-king29 LayerStack bucket (841 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k9k9` | 1024 | 7 | 64 | hand64 × k9k9 LayerStack bucket (5184 stacks。非常に大きい) |
| `SFNN_halfka2_1024_7_64_hand64_k21k21` | 1024 | 7 | 64 | hand64 × k21k21 LayerStack bucket (28224 stacks。巨大) |
| `SFNN_halfka2_1024_7_64_hand64_k29k29` | 1024 | 7 | 64 | hand64 × k29k29 LayerStack bucket (53824 stacks。巨大) |
| `SFNN_halfka2_1024_7_64_hand256` | 1024 | 7 | 64 | hand256 の手駒有無 LayerStack bucket (256 stacks) |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 1024 | 7 | 64 | hand256 × k3k3 LayerStack bucket (2304 stacks。非常に大きい) |
| `SFNN_halfka2_1024_7_64_hand1024` | 1024 | 7 | 64 | hand1024 の手駒有無 LayerStack bucket (1024 stacks) |
| `SFNN_halfka2_1024_7_64_hand1024_k3k3` | 1024 | 7 | 64 | hand1024 × k3k3 LayerStack bucket (9216 stacks。巨大) |
| `SFNN_halfka2_1024_7_64_progress8` | 1024 | 7 | 64 | 進行度だけで 8 分岐する SFNN |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 1024 | 7 | 64 | k3k3 × progress8 LayerStack bucket (72 stacks) |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 1024 | 7 | 64 | hand256 × k3k3 × progress16 LayerStack bucket (36864 stacks) |
| `SFNN_ka2_4096_15_64_c0_s256x16_k3k3` | 4096 | 15 | 64 | 軽量な KA2 入力を使う grouped SFNN |
| `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3` | 8192 | 7 | 64 | pure grouped L1 の common+shard 表記。0 common + 1024 × 8 shards |
| `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3` | 3072 | 7 | 64 | common+shard SFNN L1。1024 common + 256 × 8 shards |

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_1024x2_8_64 \
    --teacher teachers/
```

`--arch` は「何の評価関数を学習するか」を決める一番重要な指定です。学習結果の `nn.bin` をやねうら王で読むには、やねうら王側も同じ名前の architecture でビルドする必要があります。対応する edition 名を `make` に渡してビルドしてください。詳しくは [§8 Engine](8-engine.md) を参照。

SFNN の L1 層を分割したい場合は、名前の途中に `_cN_sMxG` を入れます。たとえば `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3` は、`FT=8192`, `L1 hidden=7`, `L2=64`, `L1 を 1024 channel × 8 個に分ける` という意味です。

共通部分を持たせる場合も同じ形式です。たとえば `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3` は、全分岐で共有する 1024 channel と、256 channel × 8 個の分割部分を持つという意味です。`k3k3` などを省略すると、局面で分岐しない `LayerStacks = 1` になります。

`hand64/hand256/hand1024` は持ち駒、`k3k3/k9k9/k21k21/k29k29` などは玉位置、`progress2/3/4/8/16/32` は進行度で分岐する指定です。これらは組み合わせられます。例: `hand256_k3k3_progress16`。指定順は `hand256_k3k3_progress16` でも `k3k3_hand256_progress16` でも受け付けますが、出力ディレクトリ名では `hand → king → progress` の順に整理されます。`ka2` / `halfka2` などの入力特徴名から、BulletOu が必要な内部設定を自動で選びます。

## 4.4 SFNN-1536 (やねうら王 NNUEwoSQPT1536) を学習する

やねうら王の **`YANEURAOU_ENGINE_SFNN1536` build** に読み込ませる評価関数を学習したい場合は、対応する architecture 名を指定する。

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

通常の NNUE と違い、SFNN は局面の種類ごとに後段ネットワークを切り替えます。`--arch` の `k3k3` は、玉位置で 3×3 に分ける指定です。使い方は [§9 LayerStack](9-layerstack.md)、量子化や `nn.bin` の詳細は [リファレンス: SFNN-1536](../shogi/sfnn-1536.md) を参照。

## 4.5 KPPT を学習する

KPPT 系では、`--arch KPPT` または `--arch KPP_KKPT` を指定する。

```bash
./target/release/examples/bulletou \
    --arch KPPT \
    --teacher teachers/
```

デフォルト出力先は `checkpoints/KPPT/`。KPP を小さくした形式で学習したい場合は `--arch KPP_KKPT` に変えるだけでよい。

## 4.6 教師データの渡し方

`--teacher` には次のいずれも渡せる。

- 1つのファイル。例: `teachers/teacher.pack`
- ディレクトリ。中にある同一拡張子のファイルがすべて連結される
- カンマ区切りの複数指定

## 4.7 学習はどこまで走るか

`--superbatches` と `--max-epochs` を省略すると、教師データを1周するまで学習する。複数 epoch 回したい場合は、`--superbatches` で「1 epoch は何 sb か」を決めたうえで `--max-epochs N` を指定する。`step` / `geometric` / `cos` は epoch 境界で `--lr` に戻る。

教師サイズが事前にわかっているなら、`--superbatches N` で「1 epoch = N sb」を明示できる。詳しくは [§6.1 まず覚える単位](6-tune.md#61-まず覚える単位) を参照。教師の総局面数は `--count-teacher` で確認できる。

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
# -> "Total: 461373440 positions, suggested --superbatches 4"
```

`--lr-schedule cos` を使うときは特に重要です。`--superbatches` を決めておくと、各 epoch 末で `lr_min` に着地し、次 epoch の先頭で `--lr` に戻ります。なお、教師データ自体は epoch 境界では先頭に戻りません。教師ファイルの末尾に到達したときだけ先頭に戻ります。

## 4.8 期待される出力

正常に動いていれば、次のような出力が流れる。

```text
=== bulletou: running NNUE_HALFKP (256x2-32-32 ClippedReLU, dual-perspective) ===
Training Preamble
Net Name               : shogi_nnue_halfkp
Batch Size             : 65536
Batches / Superbatch   : 1525
Positions / Superbatch : 99942400
...
  cuda-cpp loss progress log = checkpoints/.../cuda-cpp-progress.log (checkpoint/validation/final only)
  [save]  epoch 1  sb 1/36  this-sb=... pos (...)  total=... pos  sb_time=...s  pos/s=...
  [valid]  epoch 1 sb 1  test_value_accuracy=..., test_value_loss=..., elapsed=...
  cuda-cpp SFNN direct train = ok: steps=..., positions=..., train_elapsed=...s, elapsed=...s, pos/s=...
```

stdout に出る `pos/s` は、保存・検証・ログ書き込みの時間を除いた学習速度です。デフォルトでは、loss は保存・検証・終了時だけ取得します。細かい batch loss を診断したい場合だけ `--cuda-cpp-loss-readback-interval N` を指定します。

`pos/s` は 1秒あたりに処理した局面数で、学習速度の目安。RTX 4090 なら構成によって数百万〜数千万 pos/s 程度が目安。低位 GPU では比例して低下する。

## 4.9 書き出した `nn.bin` の量子化 accuracy を測る

学習中の検証は、基本的に f32 の重みで測ります。一方、やねうら王で実際に使うのは、保存時に整数化された `nn.bin` です。整数化後の符号一致率を確認したいときは `quantized-test` を使います。

```powershell
.\target\release\examples\bulletou.exe quantized-test `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv
```

`--test-positions` を省略すると検証ファイルの全局面を使う。`--test-positions N` を指定した場合は、`--test-sample sequential` / `random` と `--test-seed` でサンプル方法を選べる。

出力される `accuracy` は、やねうら王の `test eval_accuracy` と同じく、引き分けを除外した勝ち負けの符号一致率です。SFNN の整数化後計算を CPU で再現するため、学習中の検証より遅いですが、たまに確認する用途なら十分です。

## 4.10 `nn.bin` の出力 scale と offset を確認する

`nn.bin` ごとに、整数 NNUE の最終 raw output の大きさは少し変わります。やねうら王は最終的に

```text
engine_score = raw / FV_SCALE
```

として評価値に戻すため、同じ `FV_SCALE` でも `nn.bin` によって評価値の振れ幅が変わることがあります。

`calibrate-nn-bin` は、検証局面に対して量子化後の forward を行い、次の2つを調べます。

| 項目 | 意味 |
| --- | --- |
| `estimated_fv_scale` | 教師評価値に raw output を線形に合わせたときの推定 `FV_SCALE` |
| `selected_offset` | 指定された `FV_SCALE` のもとで、loss が一番小さくなる評価値 offset |

例:

```powershell
.\target\release\examples\bulletou.exe calibrate-nn-bin `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --output checkpoints\...\0002\nn2.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv `
  --fv-scale 28
```

出力例:

```text
estimated_fv_scale= 27.832  score ~= raw/27.832 -12.345
scale_fit         = samples 921,060  rmse 620.123  r2 0.41234  current_fv_offset -9.876
selected_offset   = -10 Value
folded_raw_delta  = -280 l3b
before            = acc 62.7604%  loss_engine 0.12345678
after             = acc 62.8012%  loss_engine 0.12298765
```

`estimated_fv_scale` は、`raw` と教師評価値の関係を

```text
teacher_score ~= raw / FV_SCALE + offset
```

として最小二乗で合わせた推定値です。これは「この `nn.bin` なら、やねうら王側の `FV_SCALE` はこの値が自然」という目安です。

`selected_offset` は、指定した `--fv-scale` のまま loss を下げるための補正値です。この補正は `--output` の `nn.bin` に書き込まれます。具体的には、全 LayerStack の最終 bias に `selected_offset * FV_SCALE` を加えます。

`FV_SCALE` 自体は、このコマンドでは `nn.bin` に書き込みません。やねうら王で使うときは、表示された `estimated_fv_scale` を参考にして、エンジンオプションの `FV_SCALE` を設定してください。

---

次へ:

- 学習を中断したり再開したい場合は [5. 中断・再開](5-resume.md)
- 学習率・保存頻度・loss を調整したい場合は [6. 学習設定を調整する](6-tune.md)
- 学習済みモデルを確認したい場合は [7. 結果を確認](7-result.md)

前へ: [3. 教師データを用意する](3-data.md)
