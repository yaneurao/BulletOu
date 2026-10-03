# L1・L2の低貢献unitをepoch開始時に再初期化する

```json
"sfnn_l1_revive": true,
"sfnn_l2_revive": true,
"sfnn_l1_revive_contribution_threshold": 0.01,
"sfnn_l2_revive_contribution_threshold": 0.01
```

有効化の既定はfalse、閾値の既定は0.01です。閾値だけでは有効になりません。
閾値は有限の `(0, 1]`。同じ層・bucket内の最大貢献度を1とし、**閾値未満**を選びます。
0.01なら1%未満であり、1%ちょうどは対象外です。飽和率・棋力への貢献割合ではありません。

## 指標

$$U_i=\mathbb E[|h_i-\bar h_i|]\sum_j|w_{ji}|,\qquad R_i=U_i/\max_j U_j$$

教師サンプルの実測平均と平均絶対偏差（MAD）を用います。EMAや分散の近似ではありません。
L1は通常枝と二乗枝それぞれのUを合算します。枝間の打ち消しは考慮せず、skipは除外します。
L2はL3への接続重みを使用します。出力と後段重みはいずれも量子化GPU proxyで測ります。
全unitのUが0のbucketではRを全て0とします。そのbucketに十分な局面があれば全unitが対象です。
通常枝0・二乗枝1のような混合定数、出力一定0、後段の量子化重み0も検出できます。

## 校正・タイミング

trueの各epochの開始時（warmup epoch0を含む）に、その時点の教師位置から16batchを読みます。
shuffleなし、sample weightが0の局面は除外。bucket内1,024局面未満なら処理しません。
検証セットや正解評価値は使わず、学習の読み出し位置も進めません。
MAD用の出力をCPU RAMに一時保持します。64unit×約100万局面なら約256MiBです。追加VRAMは不要です。
L1とL2を両方指定するとL1を先に処理し、変更後のネットワークでL2を測定します。
有限サンプルでの推定であり、全局面での定数性・低貢献を保証しません。

```json
"sfnn_l1_revive": {"epoch3": true, "epoch4": false},
"sfnn_l2_revive": {"epoch3": true, "epoch4": false},
"sfnn_l1_revive_contribution_threshold": {"epoch1": 0.01, "epoch9": 0.02}
```

有効化は最初の指定までfalse、閾値のepoch指定はepoch1必須です。
処理済み情報をcheckpointに保存し、epoch途中のresumeでは再処理しません。次の有効epochでは再判定します。
校正後のcheckpoint保存前に中断すると、保存元から再実行されます。

```powershell
python .\grid_search.py --settings-file settings.json --output-folder results `
  --grid sfnn_l1_revive true --grid sfnn_l1_revive_contribution_threshold 0.005 0.01 0.02
```

## リセットと平均補償

旧unitの**各枝の実測平均×元の後段重み**を後段biasへ移します。0または1への決め打ちはしません。
入力重みをGlorot一様初期化し、校正入力で平均preactivationが0.5となるようbiasを調整します。
L1 sharedは保持し、個別重みで差し引きます。後段接続は元の符号を持つ±1/64から再学習します。
新しい接続による平均寄与を後段biasから差し引きます。L1では同じ教師16batchを再推論して新しい各枝の平均を測ります。
L2では保持した校正入力から新出力の平均を求めます。Lookahead slowも別の接続重みで補償し、対象のmomentをリセットします。
これは推定した平均の補償であり、局面ごとの等価性・精度・棋力維持を保証しません。

`l1-revive.csv` / `l2-revive.csv`に局面数・上限/ゼロ回数・U・相対貢献度R・選択結果・閾値を保存します。
Rと閾値のCSV値は0～1、stdoutは%です。既存監査ファイルは上書きせず連番にします。

## 対応範囲・移行

cuda-cppのdense SFNN、L1 factorizer none/shared、通常学習または層別QAT、standalone/grid searchに対応。
L1はBNなしのみ。L2でBNを使用する場合は校正済みL2 BN・BN QAT・統計固定が必要です。
worker、通常NNUE、compact L1、axis/pair、residual count gate、旧L2/L3 factorizerは未対応です。
FTの対応は下記を参照してください。nn.bin/checkpoint形式は変更しません。

`sfnn_l1_revive_zero` / `sfnn_l2_revive_zero`と、旧`revive_threshold` / `revive_zero_threshold`（各層）は廃止です。
指定が残っていると移行エラーになります。旧0.99を新閾値へ転記せず、冒頭の0.01へ変更してください。
古いcheckpointは読めますが、起動設定の旧項目は削除が必要です。

## FTの積ペアをリセットする

```json
"sfnn_ft_revive": {
  "epoch5": true,
  "epoch6": false,
  "epoch9": true,
  "epoch13": false
},
"sfnn_ft_revive_contribution_threshold": 0.01
```

epoch5とepoch9〜12の開始時に再判定します。epoch9だけならepoch10をfalseにしてください。
最初の指定まではfalse。既定は無効、閾値の既定は0.01、範囲は有限の `(0,1]` です。
閾値もepoch指定できます（epoch1必須）。epoch途中の再開では実行せず、次の有効epoch開始時に実行します。

FT単体ではなく `i × (i + FT幅/2)` の積を対象とします。FT幅1024なら512組です。
両視点の積について、bucket別にMAD×L1接続重み絶対値和を求めて合算します。
FTからの接続にはL1 skipも含みます。bucket内の最大を1として正規化し、
**すべてのbucketで閾値未満**のペアだけをリセットします。全bucketの局面数加重平均は判定に使いません。
全組の貢献度が0なら、そのbucketの相対値は0です。

教師の現在位置から同じ16batchを2回読み、平均とMADを求めます。sample weight=0は除外。
**1,024局面未満のbucketが一つでもあれば、警告を出してFTリセット全体を見送ります。**
FTは全bucket共通なので、観測不足のbucketを無視してリセットしません。
GPU量子化proxyを使用し、検証セットは使用せず、学習cursorは進めません。
MADのために全局面のFT出力をRAMに保持せず、bucket×FT幅の集計だけを保持します。
既存revive同様にGPU proxy/workspaceと重み・optimizerのCPU readbackが必要です。

対象ペアの両FT列を再初期化します。実特徴の重みは
`max(sqrt(6/(実入力数+FT幅)), 1/127)` を幅とする一様乱数を1/127刻みに丸め、
biasは0.5、FT factorizerの仮想行は対象列だけ0にします。これは通常の初期学習の初期化とは別の再生用初期化です。
L1の両視点への接続は元の符号の±1/64とし、L1 shared分は個別重み側で相殺します。
旧積の実測平均寄与をL1 biasへ加算し、同じ16batchをもう一度推論して新しい平均寄与を差し引きます。
対象のmomentum/velocityは0、Lookaheadのslow側も別の接続重みで補償します。
**平均の近似補償であり、局面ごとの出力・棋力が維持される保証はありません。**

処理順はFT→L1→L2です。監査は`ft-revive.csv`（既存時は連番）に保存されます。
貢献度と閾値はCSVで0〜1、stdoutで%です。以前のepoch9検証セットの測定とは局面・GPU proxyが異なるため、選択個数は一致するとは限りません。
対応はnon-BN dense SFNN、L1 none/shared、FT factorizer on/off、層別QAT、standalone/grid searchです。
BN、compact L1、axis/pair、count gate、workerは未対応です。

```powershell
python .\grid_search.py --settings-file settings.json --output-folder results `
  --grid sfnn_ft_revive true --grid sfnn_ft_revive_contribution_threshold 0.005 0.01
```
