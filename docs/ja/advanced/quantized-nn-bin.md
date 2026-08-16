# 量子化後の `nn.bin` を検証する

<a href="../../en/advanced/quantized-nn-bin.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習中の検証は、通常はメモリ上のf32重みで行います。一方、やねうら王で実際に使うのは、保存時に整数化された `nn.bin` です。

学習中に `--quantized-validation-rate` で出る値は、デフォルトでは高速化のためにGPU上の近似検証を使います。重みを `nn.bin` と同じ量子化単位に丸め、その重みをf32 forwardで評価するため、学習中の傾向を見る用途に向いています。

整数演算まで含めて正確に測りたい場合は、学習コマンドに `--quantized-validation-exact` を足します。この場合は `quantized-test` と同じCPU整数forwardを使うため、かなり遅くなります。普段はGPU近似、候補を絞り込むときだけ exact にするのが使いやすいです。

このページでは、保存済みの `nn.bin` を直接調べる2つのコマンドを説明します。こちらは整数化された `nn.bin` を読むため、より実機に近い確認に使います。

| コマンド | 用途 |
|---|---|
| `quantized-test` | 量子化後のaccuracy / lossを測る |
| `calibrate-nn-bin` | 出力scaleを調べ、最終biasにoffsetを畳み込む |

## 量子化後のaccuracy / lossを測る

```powershell
.\target\release\examples\bulletou.exe quantized-test `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv
```

`--test-positions` を省略すると、検証ファイルの全局面を使います。局面数を指定する場合は、`--test-sample sequential` / `random` と `--test-seed` でサンプリング方法を選べます。

出力される `accuracy` は、引き分けを除外した勝ち負けの符号一致率です。やねうら王の `test eval_accuracy` と同じ見方です。

## 出力scaleとoffsetを確認する

`nn.bin` ごとに、整数NNUEの最終raw outputの大きさは少し変わります。やねうら王は最終的に次の形で評価値へ戻します。

```text
engine_score = raw / FV_SCALE
```

そのため、同じ `FV_SCALE` を使っても、`nn.bin` によって評価値の振れ幅が変わることがあります。

`calibrate-nn-bin` は、検証局面に対して量子化後forwardを行い、`FV_SCALE` とoffsetを調べます。

| 項目 | 意味 |
|---|---|
| `estimated_fv_scale` | raw output と教師評価値を線形に合わせた診断値 |
| `selected_fv_scale` | 検証lossが最も小さくなる `FV_SCALE` |
| `selected_offset` | 選ばれた `FV_SCALE` のもとでlossが最も小さくなる評価値offset |

例:

```powershell
.\target\release\examples\bulletou.exe calibrate-nn-bin `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --output checkpoints\...\0002\nn2.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv `
  --fv-scale auto
```

`--fv-scale` を省略した場合、BulletOu は `FV_SCALE=24` として計測します。これは量子化後検証の初期候補です。デフォルトの WRM loss では、`FV_SCALE` は学習lossの式には入りません。

`--fv-scale auto` を指定すると、デフォルトで `16..=40` の整数 `FV_SCALE` を探索します。範囲を変える場合は `--fv-scale-min`、`--fv-scale-max`、`--fv-scale-step` を指定します。

`--fv-scale 24` のように整数を指定した場合は、その `FV_SCALE` に固定してoffsetだけを探します。

offset の選び方は `--objective` で指定できます。

| 指定 | 意味 |
|---|---|
| `--objective loss` | 検証lossが最も小さいoffsetを選ぶ。デフォルト |
| `--objective accuracy` | 符号一致accuracyが最も高いoffsetを選ぶ |

棋力計測に出す候補を作る場合は、`loss` 版と `accuracy` 版を両方作って対局で比べるのが安全です。offset は全LayerStack共通の1パラメータだけなので、重い再学習なしに試せます。

出力例:

```text
searched_fv_scales= 25
searched_offsets  = 257
searched_candidates= 6,425
selected_fv_scale = 16
estimated_fv_scale= 2.390  score ~= raw/2.390 +200.311
scale_fit         = samples 921,060  rmse 2271.179  r2 0.27811  current_fv_offset +27.783
selected_offset   = +26 Value
folded_raw_delta  = +416 l3b
before            = acc 63.2031%  loss_engine 0.07208891
after             = acc 62.8638%  loss_engine 0.07186714
```

`estimated_fv_scale` は、`raw` と教師評価値の関係を次の形で最小二乗fitした診断値です。

```text
teacher_score ~= raw / FV_SCALE + offset
```

これは検証lossを最小にする `FV_SCALE` そのものではありません。実際に使う候補は `selected_fv_scale` を見てください。

`selected_offset` は、`selected_fv_scale` のもとでlossを下げるための補正値です。この補正は `--output` の `nn.bin` に書き込まれます。具体的には、各LayerStackの最終biasに `selected_offset * selected_fv_scale` を加えます。

`FV_SCALE` 自体は、このコマンドでは `nn.bin` に書き込みません。やねうら王で使うときは、表示された `selected_fv_scale` をエンジン側の `FV_SCALE` に設定してください。

前へ: [応用編トップ](README.md)
