# NNUE / SFNNのBatch Normalization（実験用）

## 通常NNUE

CUDA C++バックエンドの通常NNUE（HalfKP / KP / KA2 / HalfKPE9 / HalfKP_vm）にも対応しています。
FT・L1・L2のactivation前へ個別にBNを入れます。FTの2視点は従来どおり**連結**し、SFNNの積や二乗枝へ変更しません。

| JSONキー | デフォルト | 意味 |
|---|---:|---|
| `nnue_bn_ft` | `false` | FT加算後、CReLU前。両視点で統計・γ・βを共有 |
| `nnue_bn_l1` | `false` | 第1全結合隠れ層のCReLU前 |
| `nnue_bn_l2` | `false` | 第2全結合隠れ層のCReLU前 |
| `nnue_bn_gamma` | `0.25` | γの初期値（以降は学習） |
| `nnue_bn_beta` | `0.5` | βの初期値（以降は学習） |
| `nnue_bn_momentum` | `0.1` | running統計のEMAで新しいbatch側の係数。`(0,1]` |
| `nnue_bn_epsilon` | `0.00001` | 分散へ加える正の定数 |

L1/L2はbucketなしでunitごとに統計を持ちます。計算式・統計更新・γ/βの扱いは下記SFNNと同じです。
通常NNUEは既存の制限どおり `batches_per_update=1` のみです。
通常NNUE BNはstandalone/grid search用で、plateau、worker、epoch別BN切り替え、BN QATは未対応です。
`sfnn_bn_*` は通常NNUEには使用しません。

grid searchの例（`settings.json` のarchには通常NNUEを指定）：

```powershell
python .\grid_search.py `
  --settings-file .\settings.json `
  --output-folder D:\BulletOu-snapshots\grid-nnue-bn `
  --grid nnue-bn-ft false true `
  --grid nnue-bn-l1 true `
  --grid nnue-bn-l2 true
```

validationはrunning統計をfoldした重みで計算します。`nn.bin`にもfoldしてから従来どおり量子化するため、
やねうら王側にBN層や追加ファイルは不要です。ただし量子化誤差やclipがなくなるわけではありません。
通常NNUEの既存の検証項目を変更・追加する機能ではありません。
`state.bin` / 完全状態の`weights.bin`には未fold重み、BN統計、γ/βとそのoptimizer stateを保存します。
resume時は同じBNオプション・設定値を指定してください。保存済みBNをOFFにしてのresumeはエラーにし、黙って破棄しません。

## SFNN

FT・L1・L2の線形出力に、activation前のBatchNorm（BN）を個別に追加できます。
デフォルトはすべてOFFです。中心化とは別の処理で、学習中のforwardも変わります。

## オプション

JSONでは `_`、CLIでは `-` を使います。grid searchではどちらも使えます。

| JSONキー | デフォルト | 意味 |
|---|---:|---|
| `sfnn_bn_ft` | `false` | FT加算後、clamp・前半後半の積より前にBN |
| `sfnn_bn_l1` | `false` | L1線形出力をBNしてから通常枝・二乗枝へ分岐。skip出力は対象外 |
| `sfnn_bn_l2` | `false` | L2線形出力をBNしてからclamp |
| `sfnn_bn_gamma` | `0.25` | 有効なBN各層のγの**初期値**。γはその後学習される |
| `sfnn_bn_beta` | `0.5` | 有効なBN各層のβの**初期値**。βはその後学習される |
| `sfnn_bn_momentum` | `0.1` | 推論用統計のEMAで、**新しいbatch側**に掛ける係数。範囲 `(0,1]` |
| `sfnn_bn_epsilon` | `0.00001` | 分散の分母へ足す正の定数 |

γ=0.25、β=0.5は、標準偏差1のまま上限1のclampへ入れることを避けるための実験用初期値です。
最適値が確認されたわけではありません。標準的なγ=1、β=0も指定できます。
これらのオプションはepoch別切り替えには対応していません。

## 計算と統計の単位

\[
 y=\gamma\frac{z-\mu_B}{\sqrt{v_B+\varepsilon}}+\beta
\]

FTは両視点を合わせ、同じunitの統計・γ・βを共有します。
L1/L2はbucket・unitごとです。学習batchの全レコードを統計に使用し、lossのentry weightでは重み付けしません。
学習時の分散は母分散、推論用の移動分散には不偏分散を使います。
初めて2件以上出現したgroupは、そのbatchで推論用統計を初期化し、以降EMAで更新します。
0件なら統計は変更せず、1件なら推論用統計を使い統計は更新しません。
未初期化の統計は平均0・分散1です。

BNのbackwardは平均・分散の微分を含みます。STEではありません。
γ・βは本体と同じRanger更新タイミング・学習率で学習しますが、weight decayとweight clipの対象にはしません。
`batches_per_update`が2以上の場合、各mini-batchでBNを計算し、γ・βの勾配も本体と同様に蓄積します。
蓄積した全batchを一括してBNする方式ではありません。

## 推論・保存・resume

validationは推論用の移動平均・分散を使用します。qvalidとnn.binは、その同じ統計を次の式でfoldしてから量子化します。

\[
r=\frac{\gamma}{\sqrt{v_{\rm running}+\varepsilon}},\quad
W'=rW,\quad b'=r(b-\mu_{\rm running})+\beta
\]

L1のshared成分も合算してからbucketごとにfoldします。nn.binの形式や、やねうら王の推論処理を変更する必要はありません。
fold後の重みが量子化範囲に収まる保証はないため、acc/qacc・飽和率も確認してください。

`state.bin`にはfold前の重みとBNのγ・β、移動統計、γ・βのoptimizer stateを保存します。
BN付きcheckpointのresumeには同じBN設定が必要です。保存されたBNを無視してOFFで読み込むことはエラーにします。
BNなしのcheckpointに新たにBNを追加すると、forwardが変わります。単なる等価変換ではありません。

## 対応範囲と負荷

- cuda-cppの通常学習と`grid_search.py`。dense SFNN、factorizer `none` / `shared`。
- 現時点ではworker、plateau、compact/grouped L1、bucket-count gates、層のfreeze／個別LR倍率は未対応。
- 従来のL1単独QATはBNと併用できません。`sfnn_bn_qat=false`で`sfnn_qat_l1=true`なら黄色のWARNINGを出し、L1単独QATだけを無効化します。BN用QATは下記の別オプションです。
- `sfnn_l1_effective_weight_clip`、FT/weight saturation penaltyは併用不可で、引き続きエラーにします。
- L1／L2・L3の中心化とは併用できます（中心化側の制約も適用）。
- 各BN層は `bucket数 × unit数 <= 65536`。FTは1group。
- validationのbatch sizeは学習batch size以下にしてください。
- `average-sfnn-state`と`compare-sfnn-quantization`はBN付きstate.bin未対応です。BNを無視した結果を返さずエラーにします。学習中のvalidation/qvalidとnn.bin出力は対応しています。

BNは追加のGPU集計とbackwardが必要です。FT幅1024・batch65536の場合、両視点の正規化済み値だけで約512 MiBのVRAMが追加されます。
GPU版qvalidはGPU上でBNをfold・量子化し、重みのCPU readback・再uploadは行いません。CPU exact版は従来どおりCPUで計算します。
BN有効時のGPU量子化はnn.bin書き出しと同じく倍率との乗算を倍精度で行い、量子化境界の丸めを合わせます。
学習の集計は1024局面単位に分割し、幅が32の倍数なら隣接32unit、それ以外は8unitをまとめて読みます。倍精度の集計・二段階の分散計算・EMAの定義は変えていません。
FTの学習時はBN適用・clamp・前半と後半の積を一つのGPU処理に統合し、中間結果の読み書きを削減します。追加のVRAMは不要です。
L1のBN適用と通常枝・二乗枝の生成、L2のBN適用とclampも学習時に統合します。
FT backwardはBN勾配適用とbias勾配集計を統合し、そのbatchに出現しない特徴への0加算を省きます。
少数bucketのdense L1では連続入力をまとめて読む専用の重み勾配集計を使い、8/9出力のshared L1では入力勾配にも専用処理を使います。対応外の形状は既存処理へ戻ります。
これらはBNの定義・統計更新頻度・勾配蓄積方式を変更せず、追加VRAMも要求しません。並列加算順序の違いにより学習結果のbit一致は保証しません。
通常validationでは移動統計を直接適用し、batch統計を集計しません。
集計用の追加scratchはFT1024/L1=8/L2=64、8bucket、batch65536で合計約1.6 MiBです（上記activation保存領域とは別）。
BN OFFならこれらの追加領域・経路は使いません。加算順序変更による微小な丸め差はあり得ます。棋力改善は未保証です。

## Grid searchの例

新しい出力フォルダで、FT/L1を独立にON/OFFする4条件です。
既存設定にQAT等があれば、比較全条件で明示的にOFFにします。

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260923-progress8-bceloss.json `
  --output-folder D:\BulletOu-snapshots\20260925\grid-progress8-bn `
  --grid sfnn-bn-ft false true `
  --grid sfnn-bn-l1 false true `
  --grid sfnn-bn-l2 false `
  --grid sfnn-qat-l1 false `
  --grid sfnn-l1-effective-weight-clip false `
  --grid sfnn-ft-saturation-penalty 0 `
  --grid sfnn-saturation-penalty 0
```

L2も比較するなら `--grid sfnn-bn-l2 false true` に変更すると8条件になります。
BN以外の学習条件は共通JSONから引き継ぎます。BNの各条件は`grid_summary.csv`にも出力されます。
新しいBN機能を使用する前にBulletOuを再ビルドしてください。学習中の実行ファイルは置き換えないでください。

## BN用QAT（追加学習）

`--sfnn-bn-qat` / JSONの `"sfnn_bn_qat": true` はデフォルトOFFです。
BN学習済みの **state.bin（またはfull-state weights.bin）** を読み込んで使います。
新規初期化直後の未較正BNではエラーにします。まず通常BN学習でcheckpointを保存してください。
未出現bucketは保存時の初期統計を保持します。各有効BN層に少なくとも一つの較正済みchannelが必要です。

- running平均・分散は固定。γ/βと元の重み・biasは学習します。
- FT factorizer、L1 shared、BNをfoldしたFT/L1/L2/L3の重み・biasをnn.binと同じ倍率・丸め・clippingで疑似量子化します。BN OFFの層も対象です。
- 元のFP32重み・optimizer stateを保持し、量子化コピーでforward/backwardします。
- activationは既存の浮動小数点学習経路です。整数推論の全演算を再現する機能ではなく、weight/bias QATです。
- 丸めとclippingの両方にidentity STEを使い、foldの微分で元の重み・γ/βへ勾配を戻します。範囲外でも勾配をゼロにはしません。
- BPUの勾配は蓄積し、optimizer更新時に一度だけ元の座標へ戻します。
- acc/lossは量子化前のBNモデル、qacc/qlossは量子化後のままです。

固定統計で \(r=\gamma/\sqrt{v+\varepsilon}\) とすると、\(W'=rW,\ b'=r(b-\mu)+\beta\) です。
量子化コピーに対する勾配を \(G_W,G_b\) として、STE後は

\[
\frac{\partial L}{\partial W}=rG_W,\quad
\frac{\partial L}{\partial b}=rG_b,\quad
\frac{\partial L}{\partial\beta}=G_b,\quad
\frac{\partial L}{\partial\gamma}
=\frac{\langle G_W,W\rangle+G_b(b-\mu)}{\sqrt{v+\varepsilon}}.
\]

shared/factorizerにはさらに合成のchain ruleを適用します。γで割らないのでγ=0にも対応します。

既存のBNオプションを維持してONにしてください。`sfnn_qat_l1=true`が残っていてもBN用QATが置き換えます。
中心化、effective-weight clipping、saturation penaltiesとは併用不可。workerも未対応です。
通常BNからのresume時にON/OFFを変更できます。epochスケジュール切替は未対応です。設定ファイルは自動書換えしません。

BN checkpointを`initial_state`に指定した共通設定から、新しいgrid rootで比較できます。

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bn-finetune.json `
  --output-folder D:\BulletOu-snapshots\grid-bn-qat `
  --grid sfnn-bn-qat false true
```

ONは「量子化＋統計固定」、OFFは通常BNです。量子化だけの単独効果の比較ではありません。
量子化コピーのVRAMが追加されます（HalfKA2 FT1024で約522 MiB、起動ログに表示）。
保存形式は変更せず、やねうら王の変更も不要です。精度・棋力改善は保証しません。
