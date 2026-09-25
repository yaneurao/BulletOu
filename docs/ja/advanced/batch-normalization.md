# NNUE / SFNNのBatch Normalization（実験用）

## 復元時のL2定数unit再利用

`--sfnn-l2-revive`（JSON: `"sfnn_l2_revive": true`、デフォルトfalse）は、学習済みL2 BN checkpointを復元した直後、学習開始前に一度だけ校正・再初期化します。各epoch開始時には実行しません。
`sfnn_bn_l2`、`sfnn_bn_qat`、`sfnn_bn_qat_freeze_stats`をtrueにしてください。直接学習とgrid_searchに対応し、worker、通常NNUE、scratch開始、compact L1、axis/pair・residual count gate・旧L2/L3 factorizerは未対応で明示エラーになります。

```json
"sfnn_bn_l2": true,
"sfnn_bn_qat": true,
"sfnn_bn_qat_freeze_stats": true,
"sfnn_l2_revive": true
```

gridのA/B比較は `--grid sfnn-l2-revive false true`。共通設定の`initial_state`に同じcheckpointを指定してください。

- 復元した教師位置から16batchを別途読み、量子化推論で校正します。校正時のshuffleは0。学習側の位置・shuffle設定は進めたり変更したりしません。検証データ・正解評価値は校正に使用しません。
- bucket内で1,024局面以上観測され、その全てで出力1のL2 unitだけが対象です。少数・未出現bucketは処理しません。「サンプル全件」であり、未知の全局面で定数という証明ではありません。
- 定数寄与をL3 biasへ移し、入力をGlorot一様分布（seed=20260926）で再初期化。対象のBN倍率を1にし、校正入力平均で出力が約0.5になるようbetaを設定します。L3接続を元と同符号の±1/64から再学習し、その平均寄与もbias補償します。
- 対象のmomentとLookahead slowを整合させます。他unitとFT/L1はリセットしません。上限に達しないunitや出力0のunitは自動リセットしません。
- 校正・処理済み情報をL2 BNのstateレコード（version 2）に保存します。後続checkpointからのresumeでは、フラグがtrueでも再実行しません。対象0件でも校正済みになります。nn.bin形式は変わりません。旧stateは読み込めますが、処理済みstateを旧実行ファイルでは読めません。
- 出力先の`l2-revive.csv`に各bucket/unitの局面数・上限/下限回数・対象判定を保存します。既存ファイルは上書きせず連番にします。校正後、checkpoint保存前に中断した場合は、元checkpointから再校正します。

これは完全な等価変換ではありません。定数寄与の移動後に新しい局面依存出力を作り、量子化丸めも変わります。精度・棋力の向上は保証しません。校正時のみ推論用GPU重み/workspaceとCPU入力サンプルを一時保持し、学習前に解放します。

## BN-QAT追加学習時のL2有効重み制限

`--sfnn-bn-l2-effective-weight-clip`（JSON: `sfnn_bn_l2_effective_weight_clip`、デフォルトfalse）は、BN fold後のL2重みを `[-2, 127/64]` に制限します。`sfnn_bn_l2=true`、`sfnn_bn_qat=true`、`sfnn_bn_qat_freeze_stats=true` が必要です。学習済みBN checkpointからの追加学習用で、通常NNUEには適用しません。

```json
"sfnn_bn_l2": true,
"sfnn_bn_qat": true,
"sfnn_bn_qat_freeze_stats": true,
"sfnn_bn_l2_effective_weight_clip": true,
"lr": 0.00005,
"lr_min": 0.00005
```

開始時と各optimizer更新後（BNのγ更新・Ranger Lookaheadの後）に、`r=gamma/sqrt(running_variance+epsilon)` として `W <- clip(r*W,-2,127/64)/r` をGPUで計算します。Rangerのslow weightも同じ制限を適用し、momentum/velocityは保持します。γ=0では変更しません。BPU>1では各microbatchではなく更新時に実行します。追加VRAMは不要です。

対象は **L2 weightのみ**。bias・FT・L1・L3は変更しません。通常の`optimizer_weight_clip`とは別で、BN倍率を含めた制限です。平均・分散は固定しますが、重みとγ/βは学習を続けます。再開時にON/OFFを変更できます。設定をOFFに戻しても、既に補正した重みは元に戻りません。nn.bin/state.bin形式は変わりません。

古いL2 shared重みが有効な構成には未対応で、明示的にエラーにします。現在のL1のみのsharedは対応しています。

grid searchでは `--grid sfnn-bn-l2-effective-weight-clip false true` で比較できます。共通設定には必要なBN/QAT/統計固定を指定してください。途中epochのON/OFFスケジュールはこのオプションでは未対応です。

この制限は量子化前後の乖離を抑えるためのもので、acc・qacc・棋力の向上を保証するものではありません。

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
resume時は同じBNオプション・設定値を指定してください。ただし `nnue_bn_momentum` は変更できます。保存済みBNをOFFにしてのresumeはエラーにし、黙って破棄しません。

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

### 統計の取り込み率を下げて追加学習する

既存runの設定JSONで、次を指定して再開できます。

```json
"sfnn_bn_momentum": 0.01
```

更新式は `新しい移動統計 = (1 - momentum) × 以前の移動統計 + momentum × 今回のbatch統計` です。
既定値は引き続き0.1。0.01では各batchの取り込み率が1%になり、0.001では0.1%になります。
毎mini-batch更新する頻度は変わりません。SBごとの更新や16sb分の厳密な平均を意味しません。
履歴の半減時間は0.1で約6.6、0.01で約69、0.001で約693回の統計更新です。
bucketがそのbatchに十分出現しない場合は更新されないため、これは必ずしも経過batch数と一致しません。

resumeおよび `--initial-state` では、保存済みの平均・分散、学習済みγ/β、optimizer状態を保持し、
**再開後の取り込み率だけ**を指定値へ変更します。起動時に旧値→新値を表示し、次のcheckpointにも新値を保存します。
epsilon・初期γ/βなど、その他のBN設定の変更を許可するものではありません。
`sfnn_bn_qat_freeze_stats=true` のときは統計固定なので、取り込み率を変えても統計更新は起きません。
小さい値は変動を平滑化する一方、重み変化への追従を遅らせます。最適値・棋力改善は未確認です。

新規の条件比較はgrid searchでも指定できます。

```powershell
python .\grid_search.py --settings-file .\settings.json --output-folder D:\BulletOu-snapshots\grid-bn-ema --grid sfnn-bn-momentum 0.1 0.01 0.001
```

**既存trialを続けたい場合はgrid軸を追加せず、共通設定JSONの値を変更し、元のgrid指定に `--resume` を付けます。**
既存のBNが0.1で保存されていても再開できます。完了済みepochから1epoch追加する場合は `max_epochs` も合計epoch数へ増やしてください。

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
BN付きcheckpointのresumeには、取り込み率momentumを除き同じBN設定が必要です。保存されたBNを無視してOFFで読み込むことはエラーにします。
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

## BN用QAT（ゼロからの学習・追加学習）

`--sfnn-bn-qat` / JSONの `"sfnn_bn_qat": true` はデフォルトOFFです。
**ゼロからONで学習できます。BN checkpointの事前作成は不要です。**
`sfnn_bn_qat_freeze_stats` は既定 `false` で、BN統計をmini-batchごとに更新します。

### 通常モード：BN統計も学習

BN付きの層では、そのbatch開始時のrunning平均・分散とγから、推論時の量子化スケールを決めます。
shared等を合成した有効重みを \(W\)、nn.binの丸め・clipを \(Q\) として、

\[
r=\frac{\gamma}{\sqrt{v_{running}+\varepsilon}},\qquad
\widetilde W=\frac{Q(rW)}{r},\qquad
y=\mathrm{BN}_{batch}(\widetilde W x+b).
\]

- \(r\) は量子化用の**微分しない較正値**です。重みへは丸め・clipのidentity STE、BNへはbatch平均・分散を含む通常の微分を使います。γ/βも学習します。
- 未初期化channelは最初にデータが来たbatchでは重みの疑似量子化をせず、BN統計を初期化します。以降のbatchから反映します。全体を固定sb数だけwarmupする方式ではありません。
- γ=0のchannelも除算を避けて元の重みでBNを計算し、γが再び学習できるようにします。
- BN前biasはbatch平均で相殺されるため、このモードの学習中は疑似量子化しません。丸めたfold済みbiasを微小γで割ることによる数値崩壊も避けます。BNなしの層（L3等）は重み・biasとも疑似量子化します。
- running統計は疑似量子化重みを用いたBN入力から更新します。BPUでも各mini-batchで更新・再量子化し、勾配のみ蓄積します。
- activationは浮動小数点です。整数推論の完全再現ではありません。validation/exportは元のFP32重みとrunning統計を使い、nn.binではbiasも従来どおり量子化します。

### 統計固定モード（従来の追加学習方式）

`"sfnn_bn_qat_freeze_stats": true`（CLI: `--sfnn-bn-qat-freeze-stats`）を追加すると以下の方式になります。
この場合だけ、BN学習済み **state.bin / full-state weights.bin** が必要です。
各有効BN層に少なくとも一つの較正済みchannelが必要で、未出現bucketは保存済み初期統計を保持します。

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
通常BNからのresume時にON/OFFを変更できます。単体SFNN学習とgrid searchでは、次のepochスケジュールも使えます。

```json
"sfnn_bn_qat": {"epoch1": false, "epoch6": true}
```

epoch 1〜5はBNのみ、epoch 6開始時からBN＋QATになります。重み・optimizer・学習済みγ/β・BN統計は初期化しません。
resumeは再開epochの値を使用します。warmupのepoch 0はepoch1の値を使用し、将来のQAT指定を先に適用しません。
`sfnn_bn_ft/l1/l2`は必要な層を最初から有効にしてください。`sfnn_bn_qat_freeze_stats`は既定のfalseのまま使います。
切替時は`[BN QAT] epoch=6 enabled=true`と表示します。true→falseも対応します。設定ファイルは自動書換えしません。

ゼロからの共通設定（BN層ON、`initial_state`なし）から、新しいgrid rootで比較できます。

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bn-training.json `
  --output-folder D:\BulletOu-snapshots\grid-bn-qat `
  --grid sfnn-bn-qat false true
```

既定ではON/OFFとも統計を更新します。固定モードを選ぶ場合だけ、量子化に加えて統計固定の影響も入ります。
量子化コピーのVRAMが追加されます（HalfKA2 FT1024で約522 MiB、起動ログに表示）。
保存形式は変更せず、やねうら王の変更も不要です。精度・棋力改善は保証しません。

BN QATでは、FTの量子化と学習座標への変換を一つのGPU処理にまとめ、
重複するFT全体の読み書きと、統計更新型で不要な恒等勾配変換を省いています。
量子化・BN統計更新は従来どおり毎mini-batchです。丸め処理も従来と同じです。
高速化用の追加指定は不要です。
