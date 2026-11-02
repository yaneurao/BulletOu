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
| `export-nn16` | [FP32 checkpointからL1/L2/L3を16bitで書き出す](nn16.md)（対応エンジンが必要） |

## 量子化後のaccuracy / lossを測る

### 学習済みcheckpointにL2/L3 GPTQを試す

`export-gptq` はFP32 checkpointを変更せず、校正用の教師局面における層入力の相関を使ってL2/L3の重みを整数化します。FTとL1、biasは通常の丸めです。非BN・FT factorizer OFF・L2/L3 factorizerなし、count gateなし、factorizer係数1のcheckpointに対応します。QATの有無にかかわらず実行できますが、QATを使わない実験では学習時に各QAT設定をOFFにしてください。

```powershell
.\target\release\examples\bulletou.exe export-gptq `
  --arch SFNN_halfkahm2_2304_15_64_k9k9_progress8 `
  --state-bin D:\...\0020\state.bin `
  --settings-file D:\...\0020\bulletou-settings.json `
  --epoch 5 `
  --calibration-teacher D:\...\training.bin `
  --calibration-positions 1000000 `
  --output D:\...\gptq-e5\nn.bin
```

`--epoch` は保存checkpointのepochで、設定のepoch別factorizerを解決するために使います。校正には検証用とは別の代表的な学習局面を使ってください。現在は指定ファイルの先頭から取得します。`--min-bucket-positions`（デフォルト128）未満のbucketや校正行列を計算できないbucketは通常の丸めに戻し、適用bucket数を表示します。`--damping`（デフォルト0.01）は相関行列の安定化係数です。校正で層出力の二乗誤差が通常丸めより悪い行も通常丸めに戻します。これは検証精度や棋力の改善を保証しません。

出力先の既存ファイルは上書きしません。出力先にはcheckpointの `progress.bin` も保存します。通常の `nn.bin` とGPTQ版を別々に `quantized-test --mode cpu-exact` で比較してください。学習中のqvalidは通常の丸めによる近似で、GPTQを反映しません。

設定に `"sfnn_gptq": true` を加えると、番号付きcheckpointの通常保存後に `nn-gptq.bin` を自動生成します。省略時や `false` では生成しません。学習中に作られる `nn.bin` と `state.bin` は従来どおりです。校正はデフォルトで学習用teacherのファイル先頭262,144局面を使い、teacherがディレクトリなら対応ファイルを名前順に並べた最初のファイルを使います。`sfnn_gptq_calibration_teacher` と `sfnn_gptq_calibration_positions` で変更できます。校正用は検証とは別の代表的な学習データを選んでください。

自動生成の対象・制約は上記のL2/L3 GPTQと同じです。checkpointの保存後にGPUで校正するため、保存のたびに時間とGPUメモリが必要です。GPTQが失敗した場合は警告を出し、通常checkpointを保持して学習を続けます。

自動生成後、`test_teacher` で `nn-gptq.bin` をCPUの整数演算で評価し、checkpoint内の `gptq-validation.json` に記録します。grid searchの `grid_summary.csv` ではepoch末尾のcheckpointに対する値を `gptq_qacc`、`gptq_qloss`、`gptq_mode`、`gptq_positions` 列に出します。評価値は通常のqvalidと同じ学習尺度（最終FV_SCALE除算前）です。評価できなかった場合やepoch末尾にcheckpointがない場合は空欄です。通常の `quantized_value_accuracy/loss` がGPU近似の場合、GPTQ列は整数演算なので値の差には評価方法の差も含まれます。設定した検証局面数とlossの条件は通常の検証に従います。GPTQの整数評価にもcheckpointごとの時間がかかります。

### L1のQATをON/OFFで比較する

QAT（Quantization-Aware Training、量子化を考慮した学習）は、丸めた重みでの出力を改善するように学習する方法です。
SFNNのdense・grouped/common-shard L1では、学習コマンドに `--sfnn-qat-l1` を追加して試せます。**デフォルトはOFF**です。
JSON設定なら、既存の `bulletou-settings.json` のトップレベルに次の項目を追加します。

```json
"sfnn_qat_l1": true
```

`false` または省略で従来の学習です。通常学習・worker・profilingの各経路で有効です。
grouped/common-shard L1でも接続構造を保ち、共有項をfoldしてから量子化します。

L1だけを有効にした場合は **L1のweightとbiasだけ**が対象で、活性値の整数演算は模擬しません。L1・L2・L3をすべて有効にした場合は、下記のdense QATになります。
shared / axis / pair / count gateを使っている場合は、それらを足し合わせた実効重み `W` に対して、次の計算をします。

```text
W = gate * residual + alpha_shared * shared + Σ(alpha_axis * confidence_axis * axis)
Q(W) = clamp(round(64 * W), -128, 127) / 64
Q(b) = clamp(round(8128 * b), INT32_MIN, INT32_MAX) / 8128
L1出力 = Q(W) * 入力 + Q(b)
```

丸めは最近傍（ちょうど中間ならゼロから遠い側）で、GPU量子化検証と共通の処理を使います。
逆伝播では、入力への勾配にも `Q(W)` を使います。重みへの勾配は、丸め・clampの微分を1と近似する **identity STE** でFP32の元の重みに流します。
clamp範囲外でもSTE自体は勾配を止めません。既存のoptimizer weight clipやsaturation penaltyの設定は変更しません。
factorizerの係数・count gateに対する通常のchain ruleも維持します。

重み・optimizer stateはFP32のまま `state.bin` に保存し、`nn.bin` の形式は変更しません。
既存のcheckpointから `--initial-state` / `--initial-dataloader-pos` で追加学習でき、`--resume` 時にQATを切り替えることもできます。
QATのON/OFFは起動ログと保存設定に記録されます。再開時にも使用したい値を指定してください。

`test_value_accuracy/loss` は従来どおり **丸めないFP32モデル**、`quantized_value_accuracy/loss` は従来どおり **全層を量子化したモデル**の指標です。
QATをONにしても指標の定義や教師のscale、loss target、LRは変わりません。
FP32と量子化後の差が縮まるだけでなく、量子化後の精度そのものが改善するか確認してください。

A/Bでは、同一の `state.bin` と `dataloader_pos.txt` から、教師局面・LR・bpu・学習量を揃え、QATだけをON/OFFにします。
保存先は別tagにしてください。改善や棋力向上は保証されないため、最後は実際の `nn.bin` で棋力も比較します。

追加処理はGPU内だけです。丸めたL1を重み更新まで再利用し、復元やfactorizer設定変更でも再計算します。
追加VRAMは `4 * stacks * L1出力数 * (FT幅 + 1)` byteです。`1024_8_64_progress8` なら約0.25 MiBですが、大量bucketのarchでは比例して増えます。
OFFではこの作業領域を確保しません。学習速度への影響は未測定で、ON/OFFの同条件比較が必要です。

### FT・L2・L3の層別QAT（BNなし）

CUDA SFNNでは次を個別に指定できます。すべてデフォルトOFFです。

```json
"sfnn_qat_ft": true,
"sfnn_qat_l1": true,
"sfnn_qat_l2": true,
"sfnn_qat_l3": true
```

- FT: weight/biasを127倍してint16範囲で丸めます。FT factorizer使用時はfold後の実効重みが対象です。
- L2/L3: weightを64倍してint8範囲、biasを8128倍してint32範囲で丸めます。
- L1・L2・L3のいずれかがOFFの場合、選択した層の重みだけを丸め、活性値はf32で計算します。
- **L1・L2・L3をすべてON**にすると、tatara方式のdense QATになります。FTの出力を1/127刻みに切り捨て、L1〜L3を整数積和、整数活性化、L1 skipの整数加算で計算します。逆伝播は量子化後の重みと丸め前の連続活性値を使うSTEです。FT演算自体は浮動小数点で、エンジンの最後のFV_SCALE除算は含みません。
- FTのQAT指定は独立です。tataraと同様にFTを浮動小数点のままにするなら `sfnn_qat_ft: false` とし、L1・L2・L3を `true` にしてください。
- 通常学習・worker・profiling・batches_per_updateに対応します。checkpointはFP32重みとoptimizer stateを保持し、既存checkpointからの再開時にも切り替えられます。
- `"sfnn_qat_l2": {"epoch1": false, "epoch6": true}` のようなepoch別指定にも対応します。
- BNとは併用しません。BNが有効なら警告を出してこれら3項目を無効化します。BN用は `sfnn_bn_qat` を使ってください。
- GPU上に指定層のFP32コピーを追加します。FTは大きく、HalfKA2/FT1024では約512 MiB増えます（factorizerありの場合）。L2/L3のみならFTコピーは作りません。丸め処理の速度負担もあります。
- bpuによる勾配蓄積中は、重みが更新されるまで丸め済みのコピーを再利用します。更新・復元・層別QAT設定変更時は再計算します。
- 旧形式のL2/L3 factorizerテンソルを残した状態は対象外です。

grid searchでは、例えば `--grid sfnn_qat_ft false true --grid sfnn_qat_l2 false true --grid sfnn_qat_l3 false true` で8条件を比較できます。全層を一度にONにするなら各値を `true` だけにします。既存のL1指定は独立です。

### 保存済みのnn.binを計測する

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
