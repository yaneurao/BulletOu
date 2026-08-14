# loss の scale と `FV_SCALE`

<a href="../../en/advanced/scale-and-fv-scale.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

このページでは、学習中の score scale と、やねうら王で使う `FV_SCALE` の関係を説明します。

結論から言うと、通常の学習では `FV_SCALE` は loss の式には入りません。`FV_SCALE` は書き出した `nn.bin` を量子化後にどう評価値へ戻すか、また量子化後の検証をどう測るかに使います。

## 1. デフォルトの loss

BulletOu のデフォルト loss は tatara と同じ形の WRM です。

| オプション | デフォルト | 意味 |
| --- | ---: | --- |
| `--wrm-nnue2score` | `600` | `network_output` を評価値スケールへ戻す係数 |
| `--wrm-in-offset` | `270` | prediction 側 WRM の offset |
| `--wrm-in-scaling` | `340` | prediction 側 WRM の scaling |
| `--wrm-target-offset` | `270` | teacher 側 WRM の offset |
| `--wrm-target-scaling` | `380` | teacher 側 WRM の scaling |
| `--loss-pow-exp` | `2.0` | `|prediction - target|^p` の `p` |

WRM 関数は次の形です。

```text
wrm(score; offset, scaling)
  = 0.5 * (1
           + sigmoid(( score - offset) / scaling)
           - sigmoid((-score - offset) / scaling))
```

`sigmoid(x)` は次の関数です。

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

デフォルト loss は次のようになります。

```text
score_net  = network_output * wrm_nnue2score

prediction = wrm(score_net;
                 wrm_in_offset,
                 wrm_in_scaling)

target     = wrm(teacher_score;
                 wrm_target_offset,
                 wrm_target_scaling)

loss       = |prediction - target|^loss_pow_exp
```

`teacher_score` は教師データに入っている評価値です。教師データの勝敗項は、`--lambda 1.0` のときは学習targetに使いません。

## 2. offset を 0 にして比較する

WRM の offset が必要かどうかを比較したい場合は、offset だけを 0 にします。

```powershell
--wrm-in-offset 0 `
--wrm-target-offset 0
```

この指定では scaling はそのままです。

```text
prediction = wrm(network_output * 600; 0, 340)
target     = wrm(teacher_score;        0, 380)
```

比較実験では、`--tag` を変えて別の checkpoint フォルダにしてください。

## 3. `--scale` はいつ使うか

`--scale` は、WRM を使わずに単純 sigmoid loss で学習するときの値です。

```powershell
--loss-sigmoid-mse `
--scale 600
```

このときの式は次のようになります。

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid((network_output * 8128 / fv_scale) / scale)
loss       = |prediction - target|^loss_pow_exp
```

単純 sigmoid loss では `--fv-scale` が prediction 側の出力レンジに関係します。WRM loss では `--fv-scale` は loss の式には入りません。

## 4. `FV_SCALE` は何をしているか

やねうら王側では、量子化後の整数出力 `raw` を `FV_SCALE` で割って評価値にします。

```text
engine_score = raw / FV_SCALE
```

NNUE/SFNN の `nn.bin` 書き出しでは、おおよそ次の関係になります。

```text
raw ≒ network_output * 8128
```

そのため、`FV_SCALE=40` なら、

```text
engine_score ≒ network_output * 8128 / 40
             ≒ network_output * 203.2
```

ここで出てくる `203.2` は、量子化後の `nn.bin` をやねうら王で動かしたときの出力スケールです。WRM loss の `--wrm-nnue2score 600` とは別の値です。

## 5. 何を指定すればよいか

まずはデフォルトの WRM で十分です。

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --superbatches 324 `
  --max-epochs 28 `
  --tag sfnn-wrm-default
```

offset なしWRMと比較するなら、次だけ追加します。

```powershell
  --wrm-in-offset 0 `
  --wrm-target-offset 0
```

単純 sigmoid loss を試すなら、次を追加します。

```powershell
  --loss-sigmoid-mse `
  --scale 600 `
  --fv-scale 40
```

## 6. 量子化後の確認

学習中の `test_value_loss` は f32 weight に対する loss です。実際にやねうら王で使う `nn.bin` は量子化されています。

量子化後の accuracy / loss や、適切な `FV_SCALE` を見たい場合は、応用編の [`nn.bin` の量子化検証](quantized-nn-bin.md) を使います。

## 7. 教師データの score scale を既存 `nn.bin` に合わせる

複数の教師データを混ぜるときは、score の絶対値が同じ意味になっているかを確認してください。

たとえば、どちらも DL 系モデルで re-score した PSV であっても、DL の勝率を評価値へ戻すときの係数が違うと、同じ勝率の局面でも `score` の大きさが変わります。そのまま追加学習すると、教師データごとに「+100 点」「+500 点」の意味がずれてしまいます。

`fit-teacher-scale` は、教師 PSV から局面をサンプリングし、指定した `nn.bin` で同じ局面を評価して、教師 score に掛ける係数 `a` を推定します。

```text
nn_score ≒ a * teacher_score
```

`a` は原点を通る最小二乗で求めます。

```text
a = Σ(teacher_score * nn_score) / Σ(teacher_score^2)
```

ここで `nn_score` は、指定した `nn.bin` の量子化後出力を `--fv-scale` で評価値へ戻したものです。つまり、この係数は参照する `nn.bin` と `--fv-scale` に依存します。

例:

```powershell
.\target\release\examples\bulletou.exe fit-teacher-scale `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --teacher C:\shogi\teacher\tayayan\good-testpsv20260717.psv `
  --nn-bin C:\path\to\sojo-trained\nn.bin `
  --sample-positions 100000 `
  --fv-scale 40
```

出力例:

```text
scale_multiplier = 0.094085538
formula          = rescaled_score = round(teacher_score * 0.094085538)
```

この結果は、「この PSV の score を `0.094085538` 倍すると、指定した `nn.bin` の評価値 scale に近づく」という意味です。

実際に PSV を変換するには `rescale-psv` を使います。

```powershell
.\target\release\examples\bulletou.exe rescale-psv `
  --input C:\shogi\teacher\tayayan\good-testpsv20260717.psv `
  --output D:\teacher\tayayan-rescaled.psv `
  --scale-multiplier 0.094085538
```

`rescale-psv` は PSV の score 欄だけを書き換えます。局面、手番、手、勝敗項などはそのままです。

デフォルトでは、`|score| >= 32000` の値は mate などの特殊な印として扱い、倍率を掛けずに保存します。すべての score を変換したい場合だけ、次のように指定します。

```powershell
  --preserve-score-abs 0
```
