# `--scale` と `--fv-scale`

<a href="../../en/advanced/scale-and-fv-scale.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

このページでは、BulletOu の `--scale` と `--fv-scale` の意味を説明します。

結論から言うと、NNUE/SFNN の学習ではこの2つを分けて考える必要があります。

| オプション | 役割 | デフォルト |
| --- | --- | --- |
| `--scale` | 教師評価値を勝率ラベルへ戻す係数 | `600` |
| `--fv-scale` | 量子化後にやねうら王側で使う `FV_SCALE` を想定した出力レンジ | `40` |

たとえば、rshogi の `rescore_psv` で `scale=600` を使って教師評価値を作った場合、BulletOu でも `--scale 600` を使うのが自然です。一方、やねうら王側で `FV_SCALE=40` として使いたいなら、BulletOu の学習時にも `--fv-scale 40` を指定します。どちらもデフォルトなので、通常は省略できます。

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --superbatches 324 `
  --max-epochs 28 `
  --tag sfnn-example
```

このコマンドは、明示的に書けば次と同じ意味です。

```powershell
  --scale 600 `
  --fv-scale 40
```

## 1. 教師評価値と勝率

教師データの評価値は、内部的には「勝率のlogit」として扱えます。

```text
winrate = sigmoid(score / scale)
```

`sigmoid(x)` は次の関数です。

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

たとえば `scale=600` なら、評価値と勝率はおおよそ次の対応になります。

| 教師評価値 | `score / 600` | 勝率ラベル |
| ---: | ---: | ---: |
| `-1200` | `-2.0` | `11.9%` |
| `-600` | `-1.0` | `26.9%` |
| `0` | `0.0` | `50.0%` |
| `+600` | `+1.0` | `73.1%` |
| `+1200` | `+2.0` | `88.1%` |

つまり、rshogi が `scale=600` で勝率から評価値を作ったなら、その評価値を勝率に戻すときも `scale=600` を使うのが筋です。

## 2. `FV_SCALE=40` から出てくる `203.2`

NNUE/SFNN は層が浅いため、network output の範囲をある程度大きく取ったほうがよいことがあります。やねうら王側で `FV_SCALE=40` として使いたい場合、学習後の `nn.bin` は次の関係を満たしてほしいです。

```text
engine_score ≒ teacher_score
```

SFNN/NNUE の `nn.bin` 書き出しでは、f32 の network output は量子化によっておおよそ `QA * QB` 倍されます。

```text
QA = 127
QB = 64
QA * QB = 8128
```

そのため、やねうら王側の整数NNUE出力を `raw` とすると、

```text
raw ≒ network_output * 8128
```

やねうら王側では最後に `FV_SCALE` で割って評価値にします。

```text
engine_score = raw / FV_SCALE
```

したがって、学習中の f32 network output と、やねうら王側の評価値は次のように対応します。

```text
engine_score ≒ network_output * 8128 / FV_SCALE
```

`FV_SCALE=40` なら、

```text
engine_score ≒ network_output * 8128 / 40
             ≒ network_output * 203.2
```

ここで出てくる `203.2` は、

```text
8128 / 40 = 203.2
```

という「network output をやねうら王側の評価値へ戻すための係数」です。教師評価値を勝率ラベルへ戻すための `--scale` とは役割が違います。

つまり、評価値 `+600` を出したいなら、network output はだいたい次の値になります。

```text
network_output ≒ 600 / 203.2
               ≒ 2.95
```

## 3. なぜ `--scale 203` ではだめなのか

前節の `203.2` を見て、「では `--scale 203` で学習すればよいのでは？」と思うかもしれません。しかし、`--scale` は教師評価値を勝率ラベルへ戻す係数なので、ここに `203` を入れると教師の勝率ラベルが変わってしまいます。

教師評価値が `scale=600` で作られている場合、教師評価値 `+600` は次の勝率を意味します。

```text
sigmoid(600 / 600) = sigmoid(1.0) = 0.731
```

しかし、BulletOuで `--scale 203` として読むと、

```text
sigmoid(600 / 203) = sigmoid(2.956) = 0.950
```

になります。

元の教師は「勝率73.1%」のつもりで `+600` と書いているのに、`--scale 203` で読むと「勝率95.0%」として扱ってしまいます。これが「教師データの勝率を歪める」という意味です。

そのため、BulletOuでは `--scale` と `--fv-scale` を分けています。

```text
--scale 600    # 教師評価値を勝率へ戻す
--fv-scale 40  # network outputをFV_SCALE=40向けの範囲にする
```

## 4. BulletOu のloss式

BulletOu は、教師側とprediction側を次のように揃えます。

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid((network_output * 8128 / fv_scale) / scale)
loss       = |prediction - target|^p
```

`p` は `--loss-pow-exp` です。デフォルトは `2.0` なので、sigmoid空間での二乗誤差です。

この式の意味は次の通りです。

1. `teacher_score / scale` で、教師評価値を勝率のlogitへ戻す
2. `network_output * 8128 / fv_scale` で、network outputをやねうら王側の評価値へ戻す
3. その評価値をさらに `/ scale` して、教師と同じ勝率空間へ戻す
4. 2つの勝率の差をlossにする

`--scale 600 --fv-scale 40` のとき、教師評価値 `+600` に対してlossが最小になる条件を見てみます。

```text
target = sigmoid(600 / 600)

prediction = sigmoid((network_output * 8128 / 40) / 600)
```

lossが最小になるのは、sigmoidの中身が一致するときです。

```text
(network_output * 8128 / 40) / 600 = 600 / 600
```

両辺を整理すると、

```text
network_output * 8128 / 40 = 600
network_output = 600 * 40 / 8128
network_output ≒ 2.95
```

このとき、量子化後のやねうら王側では、

```text
raw ≒ 2.95 * 8128 ≒ 24000
engine_score = raw / 40 ≒ 600
```

となります。

つまり、`--scale 600 --fv-scale 40` は次の2つを同時に満たします。

- 教師評価値の勝率ラベルは `scale=600` として正しく読む
- network output は `FV_SCALE=40` で運用しやすい範囲に広げる

## 5. `--scale` と `--fv-scale` の選び方

基本方針はシンプルです。

| 目的 | 指定 |
| --- | --- |
| 教師データが `scale=600` で作られている | `--scale 600` |
| やねうら王で `FV_SCALE=40` として使う | `--fv-scale 40` |
| やねうら王で `FV_SCALE=32` として使う | `--fv-scale 32` |
| 教師データが別のscaleで作られている | そのscaleを `--scale` に指定する |

`--fv-scale` は、学習後にやねうら王で使う `FV_SCALE` と同じ値にしてください。学習時と実行時で違う値にすると、評価値のスケールがずれます。

## 6. `--lambda` を使う場合

`--lambda` を `1.0` 以外にすると、教師評価値から作るラベルと、勝敗結果から作るラベルを混ぜます。

```text
eval_label   = sigmoid(teacher_score / scale)
result_label = win ? 1.0 : draw ? 0.5 : 0.0

target = lambda * eval_label + (1 - lambda) * result_label
```

prediction側は同じです。

```text
prediction = sigmoid((network_output * 8128 / fv_scale) / scale)
```

通常のre-score教師では、勝敗結果が教師評価値の較正に使えるとは限りません。そのため、まずはデフォルトの `--lambda 1.0`、つまり教師評価値だけを見る設定を推奨します。

## 7. 注意点

- `QA` と `QB` は `nn.bin` 書き出し用の量子化定数です。通常、ユーザーが変更する値ではありません。
- `--fv-scale` は NNUE/SFNN 用です。KPPT系は `--yaneuraou-quant-scale` を使います。
- `--scale` は教師の勝率モデルに合わせる値です。network outputを広げたいからといって `--scale` を小さくすると、教師の勝率ラベルが変わってしまいます。
- network outputの範囲を変えたい場合は、`--scale` ではなく `--fv-scale` を調整してください。
