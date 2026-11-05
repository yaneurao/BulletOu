# BCE with Logitsで学習する

[English](../../en/advanced/bce-with-logits.md)

通常の `bulletou.exe`（cuda-cpp）、worker、検証、GPU/CPUの量子化後検証で使用できます。デフォルトは従来のWRM確率空間の二乗誤差のままです。

## 指定方法

学習用JSONに次を追加します。既存の他の学習条件はそのまま使えます。

```json
{
  "loss_bce_with_logits": true,
  "wrm_in_offset": 0
}
```

CLIでは `--loss-bce-with-logits --wrm-in-offset 0` です。
`--win-rate-model` と `--loss-sigmoid-mse` は同時指定できません。JSONでは削除するかfalseにしてください。

予測側のoffsetは通常のBCE logitには適用しません。黙って無視せず、非0ならエラーにします（通常のWRMの既定offsetは270なので、0の指定が必要です）。`loss_pow_exp` はBCEでは使用せず、既定値2以外ならエラーです。

## 計算式

学習スケールのネットワーク出力を `y`、教師評価値を `s` とします。sigmoidは `σ(x)=1/(1+exp(-x))` です。

```text
z = y * wrm_nnue2score / wrm_in_scaling
p = σ(z)
F(s) = (1 + σ((s - wrm_target_offset)/wrm_target_scaling)
          - σ((-s - wrm_target_offset)/wrm_target_scaling)) / 2
t_score = ε + (1 - 2ε) * F(s)       ε = wrm_target_epsilon
t = lambda * t_score + (1-lambda) * game_result
L = -t*log(p) - (1-t)*log(1-p)
  = max(z,0) - t*z + log(1+exp(-abs(z)))
dL/dy = (p-t) * wrm_nnue2score / wrm_in_scaling
```

`game_result` は勝ち1、引き分け0.5、負け0です。教師側のoffset、scaling、epsilon、lambdaは従来どおり有効です。実装はlogitから安定な式で直接計算するため、大きい出力でも `log(0)` を計算しません。学習時は既存の局面weightを掛け、batch内で平均します。

標準のloss/qlossはこのBCEになり、Norm loss等の正則化項は含みません。acc/qaccの定義は変更しません。qvalidのGPU近似/CPU整数推論という違いも従来どおりです。`quantized-test` にも同じBCEとWRMパラメーターを指定してください。train-scale qlossではraw/8128に上の式を適用し、FV_SCALEは使いません。CPUモードの別表示engine-scale lossはraw/FV_SCALEを入力とし、nnue2score=1として計算します。GPUモードは従来どおりtrain-scaleのみを計算し、engine-scale欄にも同じ値を表示します。

**BCEのloss値と二乗誤差のloss値は直接比較できません。** BCEは飽和を自動的に抑制する機能ではありません。特に教師確率0/1では有限の最適logitがないため、必要なら教師側のepsilonなどを別途比較してください。損失の勾配スケールが変わるので、同じLRが最適とは限りません。

## Grid search

共通JSONを `wrm_in_offset: 0`、`win_rate_model: false`、`loss_sigmoid_mse: false` として、次で二乗誤差とBCEを比較できます。

```powershell
python .\scripts\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-bce `
  --grid loss_bce_with_logits false true
```

falseは従来のWRM二乗誤差、trueはBCEです。`grid_summary.csv` に `loss_bce_with_logits` が記録されます。lossの定義が変わるため、値の大小で方式間の優劣を決めないでください。既存モデルから試す場合も、条件ごとに同じcheckpoint/教師位置から始めてください。

## 誤差が大きい局面を重視するBCE

学習引数 `--bce-error-weight-k K`（JSON: `"bce_error_weight_k": K`）で、予測勝率と教師勝率の差が大きい局面への重み付けを試せます。デフォルトは **0（通常のBCE）**。有限の0以上だけ指定でき、0以外は `loss_bce_with_logits: true` が必要です。教師epsilonを有効にする必要はありません。

```text
e_i = abs(p_i - t_i)
w_i = 1 + K * e_i
mean_w = sum(entry_weight_i * w_i) / sum(entry_weight_i)
a_i = stop_gradient(w_i / mean_w)
training_loss = mean(entry_weight_i * a_i * BCE(z_i, t_i))
gradient_i = entry_weight_i * a_i * (p_i - t_i) * nnue2score / in_scaling / batch_size
```

`entry_weight` は既存の局面weightです。すべて1なら通常のbatch平均で正規化し、0の除外局面は正規化に含めません。全局面が除外なら勾配とlossは0です。分子・分母の**両方に微分を通しません**。正規化は1 mini-batch単位で、bpuでまとめた全batch単位ではありません。Kを大きくすると誤差が大きい局面を相対的に重視しますが、教師の誤りも強調する可能性があります。

**検証のloss/qlossはKによらず通常のBCEです。** train loss readbackを有効にした場合の学習lossには上記の重みが掛かりますが、検証には掛けません。これにより同じ検証条件のK違いを共通の基準で比較できます。acc/qaccの定義も変わりません。`quantized-test` にはKを指定する必要はありません。

共通JSONでBCEを有効にし、`wrm_in_offset: 0`、`win_rate_model: false`、`loss_sigmoid_mse: false` として実行します。

```powershell
python .\scripts\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-bce-error-weight `
  --grid bce_error_weight_k 0 1 2
```

`grid_summary.csv`には `bce_error_weight_k` 列を出します。K=0では既存のBCE経路をそのまま使います。K>0ではGPUカーネルが2回追加されますが、既存の小さなloss作業領域を再利用し、追加VRAM確保やCPU読み戻しはありません。実学習の速度への影響は未測定です。
