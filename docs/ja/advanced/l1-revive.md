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



## 初期学習のL1学習率抑制

`"sfnn_l1_initial_lr_suppress_sb": 32` とすると、epoch 1の最初の32sbだけ、L1全体の学習率を `sfnn_l1_lr_mult` のさらに0.1倍にします。reviveの有無とは独立し、初期値の再設定は行いません。既定値は0（抑制なし）。epoch 1途中のcheckpointからresumeしても、sb番号に基づき残り期間を抑制します。独立warmupのepoch 0には適用しません。revive後の抑制期間と重なっても0.1倍は一度だけ適用します。cuda-cpp SFNNのproduction scheduleで使用でき、plateauとdirect stepsには対応しません。

## revive後のL1学習率抑制

`"sfnn_l1_revive_lr_suppress_sb": 16` とすると、L1 reviveを有効にしたepochの開始後16sbの間、L1全体の学習率を `sfnn_l1_lr_mult` のさらに0.1倍にします。既定値は0（抑制なし）です。実際にresetしたunitがない場合は新たな抑制期間を開始しません。期間はepochをまたいでも継続し、複数の期間が重なった場合は最も遅い終了時点まで抑制します。途中のcheckpointからのresumeでも復元します。resume時はrevive・抑制期間・epochのsb数の設定を維持してください。epoch別指定も可能で、期間はrevive開始時の値を使います。cuda-cpp SFNNの通常のproduction scheduleで使用でき、plateauとdirect stepsには対応しません。

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
python .\scripts\grid_search.py --settings-file settings.json --output-folder results `
  --grid sfnn_l1_revive true --grid sfnn_l1_revive_contribution_threshold 0.005 0.01 0.02
```

## リセットと平均補償

旧unitの**各枝の実測平均×元の後段重み**を後段biasへ移します。0または1への決め打ちはしません。
入力重みをGlorot一様初期化し、校正入力で平均preactivationが0.5となるようbiasを調整します。
L1 sharedは保持し、個別重みで差し引きます。後段接続は元の符号を持つ±1/64から再学習します。
新しい接続による平均寄与を後段biasから差し引きます。L1では同じ教師16batchを再推論して新しい各枝の平均を測ります。
L2では保持した校正入力から新出力の平均を求めます。Lookahead slowも別の接続重みで補償し、対象のmomentをリセットします。
これは推定した平均の補償であり、局面ごとの等価性・精度・棋力維持を保証しません。

### revive乱数と再現性

FT/L1/L2はrunner内のrevive専用乱数状態を共有します。新規学習の開始時に一度だけ
固定seedで初期化し、対象unit・層・epochごとには初期化し直しません。
同じunitを再度リセットすると、前回の乱数列を再利用せず続きから引きます。
対象が0個の処理では乱数を消費しません。層や対象数・順序が変われば、その後の乱数割当も変わります。

乱数状態は学習checkpoint（`state.bin`等の完全状態）とメモリsnapshotに保存します。
resumeは保存位置から系列を継続します。同じ保存状態・処理順序なら同じ乱数を再現できます。
`nn.bin`には保存しません。旧checkpointに乱数状態がない場合は固定seedから開始し、以後の保存で状態を記録します。
保存前に中断して古いcheckpointへ戻った場合は、重みと同様、乱数もその保存時点へ戻ります。

reviveのGlorot幅やbias補償、後段の±1/64は従来どおりで、scratch用の
`sfnn_init_l1_glorot`等の設定によってreviveの初期化方式は変わりません。
量子化で後段接続がゼロにならないようにしていますが、非ゼロ勾配を常に保証するものではありません。

FT/L1/L2の判定記録は、学習出力フォルダの`revive.csv`にまとめて追記します。
先頭列は`epoch,run,layer`で、`run`は同epoch内の実行番号（1から開始）です。
同じepoch開始処理で行うFT→L1→L2には同じ番号が付き、同epochを再実行すると次の番号になります。
続く列は`bucket,unit,pair,positions,upper_hits,zero_hits,contribution,relative_contribution,selected,contribution_threshold`です。
FTは`pair`、L1/L2は`unit`を使用し、該当しない項目は空欄です。
対象が0個でも判定結果を記録します。`selected`はリセット候補の選択結果で、完了保証ではありません（重み変更前に記録します）。
revive無効でもthresholdを明示した層は判定結果を記録し、`selected=true`は有効なら対象になることを示します。
既存行は上書き・削除しません。旧`ft-revive*.csv` / `l1-revive*.csv` / `l2-revive*.csv`はそのまま残し、自動統合はしません。
Rと閾値のCSV値は0～1、stdoutは%です。

### epochごとの集計

SFNNの学習起動時に、reviveやthresholdの指定にかかわらずヘッダーだけのファイルを作成します。既存の行は保持します。

epoch開始時に判定する層の処理がすべて正常完了すると、学習出力フォルダの
`revive-summary.csv`へ1行追記します。`epoch,run`は詳細CSVと共通で、再実行でも過去の行は消しません。
各層の`revive_contribution_threshold`をCLI・設定JSON・epoch別設定で明示指定すると、
その層の`revive=false`でも候補を判定します。デフォルト値と同じ`0.01`の明示指定も対象です。
threshold未指定かつrevive無効の層は判定せず、全層が未判定なら行は出力しません。
revive有効時はthreshold未指定でも従来どおりデフォルト`0.01`で判定・リセットします。

各層の列は`ft_`、`l1_`、`l2_`の順で以下を出力します。

| 接尾辞 | 意味 |
| --- | --- |
| `eligible` | サンプル不足を除いた判定対象数 |
| `reset_candidates` | リセット条件に該当した候補数（実際のresetの有無によらない） |
| `reset_candidate_rate` | `reset_candidates / eligible` |
| `mean_relative_contribution` | リセット前の判定対象の平均相対貢献度 |
| `contribution_threshold` | 使用した閾値 |

末尾3列は`ft_revived,l1_revived,l2_revived`で、リセットと平均補償まで正常完了した数です。
判定のみなら0、未判定なら空欄です。各層の判定結果は実際のresetの有無によって変わりません
（ただし、有効な上流層のresetが後段の判定に与える影響はあります）。
旧形式の集計CSVは過去の行を保持して並べ替えます。実reset数は末尾へ移し、
記録済みの候補数は各層の列へ移して候補率を再計算します。過去の候補数が不明なら候補数・率は空欄にします。

FT の飽和率は相対貢献度と異なるため、この旧形式集計の FT 欄は空欄です。unit 別の結果は `ft-revive-saturation.csv` に記録します。
L1/L2は1,024局面以上のbucketのunit数で、平均は局面数で重み付けしないunit単純平均です。
L1の通常枝・二乗枝は合わせて1unit。L1はFT処理後、L2はL1処理後の判定結果です。
未判定の層は空欄。対象0なら候補数を含む数は0、率と平均は空欄。
率・平均・閾値は0～1の小数10桁。平均は最大unitとの相対値であり、棋力への貢献割合ではありません。
集計は校正結果を再利用します。無効な層の判定には追加で教師16batchの校正が必要です
FT は検証局面を使います。判定のみなら重み・optimizer・revive乱数・学習cursorは変更しません。
判定の対応範囲はreviveと同じです。FT→L1→L2の順で、その時点の重みを使うため、
無効な上流層も有効にしてリセットした場合の後段候補数とは異なることがあります。
処理途中の失敗は完了行にしません。
集計CSVの書き込み失敗は黄色のWARNINGを出して学習を継続します。
旧詳細CSVからの自動補完はしません（`selected`だけでは完了を保証できないため）。

## 対応範囲・移行

cuda-cppのSFNN（dense・grouped/common-shard L1）、L1 factorizer none/shared/axis/pair、通常学習または層別QAT、standalone/grid searchに対応。
L1はBNなしのみ。L2でBNを使用する場合は校正済みL2 BN・BN QAT・統計固定が必要です。
worker、通常NNUE、residual count gate、旧L2/L3 factorizerは未対応です。
axisではshared・axis重みとそのoptimizer状態は保持し、対象bucketの個別重み・biasで共有分を差し引きます。
alphaとaxis confidenceを含めてmaster/Lookaheadを別々に補償します。別bucketを共有因子のリセットに巻き込みません。pairもalphaとcount補正を合算して個別重みで補償し、共有成分の重み・optimizer状態は変更しません。
L2のリセットはL1のaxis/pair重みを変更しません。shared/axis/pairのepoch切り替えとも併用できます。
FTの対応は下記を参照してください。nn.bin/checkpoint形式は変更しません。

`sfnn_l1_revive_zero` / `sfnn_l2_revive_zero`と、旧`revive_threshold` / `revive_zero_threshold`（各層）は廃止です。
指定が残っていると移行エラーになります。旧0.99を新閾値へ転記せず、冒頭の0.01へ変更してください。
古いcheckpointは読めますが、起動設定の旧項目は削除が必要です。

## FT unit の上限飽和による revive

```json
"sfnn_ft_revive": {"epoch2": true, "epoch4": false},
"sfnn_ft_revive_saturation_threshold": 0.99,
"sfnn_ft_revive_batches": 16
```

`test_teacher` の検証局面について、量子化 proxy の FT 上限到達率を両視点で unit ごとに集計します。閾値以上の unit のみ再初期化し、積の相方の FT unit は維持します。候補判定に bucket の最低局面数は使いません。

バッチ数は `test_batch_size` 単位の上限です。既定16で、256も指定できます。再初期化の前後に検証局面の平均を測り、後段 bias を補償します。未観測 bucket には全局面の平均を代用するため、補償は近似です。

結果は `ft-revive-saturation.csv` に記録します。旧 `sfnn_ft_revive_contribution_threshold` は意味が異なるため無視し、警告を出します。
