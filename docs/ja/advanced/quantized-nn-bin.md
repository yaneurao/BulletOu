# 量子化後 `nn.bin` の検証

<a href="../../en/advanced/quantized-nn-bin.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習中の検証は、基本的に f32 の重みで測ります。
一方、やねうら王で実際に使うのは、保存時に整数化された `nn.bin` です。

このページでは、保存済みの `nn.bin` を直接評価する2つのコマンドを扱います。

| コマンド | 使う場面 |
| --- | --- |
| `quantized-test` | 量子化後の accuracy / loss を測る |
| `calibrate-nn-bin` | `nn.bin` の出力 scale を確認し、offset を補正する |

## 量子化後の accuracy / loss を測る

```powershell
.\target\release\examples\bulletou.exe quantized-test `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv
```

`--test-positions` を省略すると検証ファイルの全局面を使います。`--test-positions N` を指定した場合は、`--test-sample sequential` / `random` と `--test-seed` でサンプル方法を選べます。

出力される `accuracy` は、やねうら王の `test eval_accuracy` と同じく、引き分けを除外した勝ち負けの符号一致率です。

## 出力 scale と offset を確認する

`nn.bin` ごとに、整数 NNUE の最終 raw output の大きさは少し変わります。やねうら王は最終的に

```text
engine_score = raw / FV_SCALE
```

として評価値に戻すため、同じ `FV_SCALE` でも `nn.bin` によって評価値の振れ幅が変わることがあります。

`calibrate-nn-bin` は、検証局面に対して量子化後の forward を行い、`FV_SCALE` と offset を調べます。

| 項目 | 意味 |
| --- | --- |
| `estimated_fv_scale` | 教師評価値に raw output を線形に合わせたときの診断値 |
| `selected_fv_scale` | loss が一番小さくなる `FV_SCALE` |
| `selected_offset` | 選ばれた `FV_SCALE` のもとで loss が一番小さくなる評価値 offset |

例:

```powershell
.\target\release\examples\bulletou.exe calibrate-nn-bin `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --nn-bin checkpoints\...\0002\nn.bin `
  --output checkpoints\...\0002\nn2.bin `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.psv `
  --fv-scale auto
```

`--fv-scale auto` は、デフォルトで `16..=40` の整数 `FV_SCALE` を探索します。範囲を変える場合は `--fv-scale-min` / `--fv-scale-max` / `--fv-scale-step` を指定します。

`--fv-scale 28` のように整数を指定した場合は、その `FV_SCALE` に固定して offset だけを探します。

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

`estimated_fv_scale` は、`raw` と教師評価値の関係を

```text
teacher_score ~= raw / FV_SCALE + offset
```

として最小二乗で合わせた診断値です。これは loss を最小にする `FV_SCALE` そのものではありません。WRM loss は非線形なので、実際に使う候補は `selected_fv_scale` を見てください。

`selected_offset` は、`selected_fv_scale` のもとで loss を下げるための補正値です。この補正は `--output` の `nn.bin` に書き込まれます。具体的には、全 LayerStack の最終 bias に `selected_offset * selected_fv_scale` を加えます。

`FV_SCALE` 自体は、このコマンドでは `nn.bin` に書き込みません。やねうら王で使うときは、表示された `selected_fv_scale` をエンジン側の `FV_SCALE` に設定してください。

前へ: [応用編トップ](README.md)
