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
FTの貢献度リセットは今回追加していません。nn.bin/checkpoint形式は変更しません。

`sfnn_l1_revive_zero` / `sfnn_l2_revive_zero`と、旧`revive_threshold` / `revive_zero_threshold`（各層）は廃止です。
指定が残っていると移行エラーになります。旧0.99を新閾値へ転記せず、冒頭の0.01へ変更してください。
古いcheckpointは読めますが、起動設定の旧項目は削除が必要です。
