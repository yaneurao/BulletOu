# 4. 学習を走らせる — `bulletou` コマンドの実行

<a href="../../en/tutorial/4-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 用意した教師データから、やねうら王互換エンジンが読み込める評価関数バイナリを学習する。

このページは [3. 教師データを用意する](3-data.md) を完了している前提です。教師ファイル (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) が用意され、事前にシャッフルされている状態を想定する。

## 4.1 ビルドする

まず `bulletou` をビルドする。ソースを変更していなければ初回だけで十分だが、BulletOu のソースを更新した直後は既存の `.\target\release\examples\bulletou.exe` が古いままなので、必ず再ビルドする。

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

これだけで動く。`--output` を省略した場合、checkpoint は `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/` 配下に書かれる。この名前は `--arch` から自動生成される。別の場所に出力したい場合は `--output checkpoints/my-halfkp` のように明示する。

## 4.3 `--arch` を指定する

学習対象は `--arch` だけで指定する。KPPT 系なら `KPPT` または `KPP_KKPT`、NNUE / SFNN 系なら、やねうら王の Makefile edition 名から `YANEURAOU_ENGINE_` を取り除いた名前を指定する。

たとえば HalfKP の 256x2-32-32 は `NNUE_halfkp_256x2_32_32`、K-P の 256x2-32-32 は `NNUE_kp_256x2_32_32`、SFNN なら `SFNN_halfka2_1024_7_64_k3k3` のように書く。古い短縮形 `256x2-32-32` は受け付けない。

NNUE 名のサイズ部分は `<L1>x2_<L2>_<L3>`。`L1` は perspective ごとの accumulator サイズで、FT SIMD padding の都合により 32 の正の倍数である必要がある。`L2` / `L3` は正の整数なら受け付ける。よく使う例は次の通り。

| サイズ・arch例 | L1 / FT | L2 | L3 | 備考 |
|---|---:|---:|---:|---|
| `256x2-32-32` | 256 | 32 | 32 | 古典的な小型 NNUE。学習時間が短く、挙動確認向き |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | 中型 |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | 大型。推論コストは増える |
| `1024x2-8-64` | 1024 | 8 | 64 | 大型 |
| `SFNN_halfkahm2_1536_15_32_k3k3` | 1536 | 15 | 32 | k3k3 (king3-by-king3) LayerStack の SFNN-1536 |
| `SFNN_halfka2_1024_7_64` | 1024 | 7 | 64 | bucket suffix なしの single stack SFNN (`LayerStacks = 1`) |
| `SFNN_halfka2_4096_3_64_c0_s1024x4_k3k3` | 4096 | 3 | 64 | grouped SFNN L1。4096 を 4 group に分け、各 group が 1024 -> 1 |
| `SFNN_halfka2_8192_3_64_c0_s2048x4_k3k3` | 8192 | 3 | 64 | grouped SFNN L1。8192 を 4 group に分け、各 group が 2048 -> 1 |
| `SFNN_halfka2_4096_7_64_c0_s1024x4_k3k3` | 4096 | 7 | 64 | grouped SFNN L1。4096 を 4 group に分ける |
| `SFNN_halfka2_1024_7_64_hand64` | 1024 | 7 | 64 | やねうら王 hand64 LayerStack bucket (64 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k3k3` | 1024 | 7 | 64 | hand64 × k3k3 LayerStack bucket (576 stacks。かなり大きい) |
| `SFNN_halfka2_1024_7_64_k9k9` | 1024 | 7 | 64 | king9-by-king9 LayerStack bucket (81 stacks) |
| `SFNN_halfka2_1024_7_64_k21k21` | 1024 | 7 | 64 | king21-by-king21 LayerStack bucket (441 stacks) |
| `SFNN_halfka2_1024_7_64_k29k29` | 1024 | 7 | 64 | king29-by-king29 LayerStack bucket (841 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k9k9` | 1024 | 7 | 64 | hand64 × k9k9 LayerStack bucket (5184 stacks。非常に大きい) |
| `SFNN_halfka2_1024_7_64_hand64_k21k21` | 1024 | 7 | 64 | hand64 × k21k21 LayerStack bucket (28224 stacks。巨大) |
| `SFNN_halfka2_1024_7_64_hand64_k29k29` | 1024 | 7 | 64 | hand64 × k29k29 LayerStack bucket (53824 stacks。巨大) |
| `SFNN_halfka2_1024_7_64_hand256` | 1024 | 7 | 64 | hand256 の手駒有無 LayerStack bucket (256 stacks) |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 1024 | 7 | 64 | hand256 × k3k3 LayerStack bucket (2304 stacks。非常に大きい) |
| `SFNN_halfka2_1024_7_64_hand1024` | 1024 | 7 | 64 | hand1024 の手駒有無 LayerStack bucket (1024 stacks) |
| `SFNN_halfka2_1024_7_64_hand1024_k3k3` | 1024 | 7 | 64 | hand1024 × k3k3 LayerStack bucket (9216 stacks。巨大) |
| `SFNN_halfka2_1024_7_64_progress8` | 1024 | 7 | 64 | progress8 LayerStack bucket。進行度 axis のみ |
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

`--arch` は学習対象と内部 target family の single source of truth なので、通常の学習では必須。表にないサイズも実験用途では受け付けるが、学習結果の `nn.bin` を読み込めるのは、同じ architecture header で build したやねうら王だけ。対応する edition 名を `make` に渡してビルドする。詳しくは [§8 Engine](8-engine.md) を参照。

grouped SFNN は、任意の LayerStack suffix の前に `_cN_sMxG` を置いて表現できる。たとえば `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3` は、`FT=8192`, `L1 hidden=7`, `L2=64`, `L1 を 1024 channel × 8 shard に分割` という意味。

common 部分が非ゼロの common+shard L1 も同じ形式。たとえば `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3` は、1024 common channel + 256 channel × 8 shard。suffix を省略すると single stack (`LayerStacks = 1`) になる。

suffix を付ける場合は、独立した `hand64/hand256/hand1024`, `k3k3/k9k9/k21k21/k29k29`, `progress2/3/4/8/16/32` axis を組み合わせられる。例: `hand256_k3k3_progress16`。parser はこれらの token を任意順で受け付け、内部では `hand`, `king`, `progress` の順に canonicalize する。`ka2` / `halfka2` などの feature token から内部 target は自動的に決まる。

## 4.4 SFNN-1536 (やねうら王 NNUEwoSQPT1536) を学習する

やねうら王の **`YANEURAOU_ENGINE_SFNN1536` build** に読み込ませる評価関数を学習したい場合は、対応する architecture 名を指定する。

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

通常の NNUE と違い、SFNN は局面ごとに選択される複数の sub-network を使う。`--arch` の `k3k3` suffix は、やねうら王互換の LayerStack 方式を選ぶ指定。使い方は [§9 LayerStack](9-layerstack.md)、architecture / 量子化 / `nn.bin` layout の仕様は [リファレンス: SFNN-1536](../shogi/sfnn-1536.md) を参照。

## 4.5 KPPT を学習する

KPPT 系では固定 target 名を `--arch` に指定する。

```bash
./target/release/examples/bulletou \
    --arch KPPT \
    --teacher teachers/
```

デフォルト出力先は `checkpoints/KPPT/`。factorised 版にしたければ `--arch KPP_KKPT` に変えるだけでよい。

## 4.6 教師データの渡し方

`--teacher` には次のいずれも渡せる。

- 1つのファイル。例: `teachers/teacher.pack`
- ディレクトリ。中にある同一拡張子のファイルがすべて連結される
- カンマ区切りの複数指定

## 4.7 学習はどこまで走るか

`--superbatches` と `--max-epochs` を省略すると、教師データを1周するまで、つまり dataloader が EOF を返すまで学習する。複数 epoch 回したい場合は、`--superbatches` で epoch 長を決めたうえで `--max-epochs N` を指定する。`step` / `geometric` / `cos` は epoch 境界で `--lr` に戻る。

教師サイズが事前にわかっているなら、`--superbatches N` で「1 epoch = N sb」を明示できる。詳しくは [§6.1 学習スケジュール](6-tune.md#61-学習スケジュール) を参照。教師の総局面数は `--count-teacher` で確認できる。

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
# -> "Total: 461373440 positions, suggested --superbatches 4"
```

`--lr-schedule cos` を使うときは特に重要。1 cycle が 1 epoch と一致するように `--superbatches` を選ぶと、各 epoch 末で `lr_min` に着地し、次 epoch の先頭で `lr_max` に warm restart するきれいな周期になる。このモードでも、教師データ自体は epoch 境界では先頭に戻らない。教師 EOF に到達したときだけ先頭に戻る cyclic stream として扱われる。

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
  cuda-cpp loss progress log = checkpoints/.../cuda-cpp-progress.log (step 1, every 10 step(s), checkpoint, final)
  cuda-cpp SFNN checkpoint: epoch=1 sb=1/36 batch=2543/2543 positions=41664512 pos/s=... dir=checkpoints/.../0001
  cuda-cpp SFNN validation summary: epoch=1, superbatch=1, test_value_accuracy=..., test_value_loss=...
  cuda-cpp SFNN direct train = ok: steps=..., positions=..., train_elapsed=...s, elapsed=...s, throughput=... pos/s, ...
```

cuda-cpp backend の stdout に出る `pos/s` は、checkpoint file save / validation / loss readback / progress-log write の時間を除外した純粋な学習 throughput。batch ごとの loss は stdout には流さず、`<output>/cuda-cpp-progress.log` に CSV として追記される。

`pos/s` は 1秒あたりに処理した局面数で、学習速度の目安。RTX 4090 なら構成によって数百万〜数千万 pos/s 程度が目安。低位 GPU では比例して低下する。

---

次へ:

- 学習を中断したり再開したい場合は [5. 中断・再開](5-resume.md)
- 学習スケジュールや教師ターゲットを調整したい場合は [6. 学習をチューニング](6-tune.md)
- 学習済みモデルを確認したい場合は [7. 結果を確認](7-result.md)

前へ: [3. 教師データを用意する](3-data.md)
