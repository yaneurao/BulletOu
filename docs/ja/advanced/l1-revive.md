# L1の定数unitをepoch開始時に再初期化する

BNなしのcuda-cpp SFNNで使用できます。両方ともデフォルトはfalseです。

```json
"sfnn_l1_revive": true,
"sfnn_l1_revive_zero": true
```

- `sfnn_l1_revive`: 通常枝・二乗枝の**両方が常に1**のunitを対象にします。
- `sfnn_l1_revive_zero`: 両枝が**常に0**のunitを対象にします。上限到達率0%とは異なります。
- L1 skip出力は対象外です。二乗枝だけが飽和しているunitも対象外です。

CLIは `--sfnn-l1-revive` / `--sfnn-l1-revive-zero`。grid searchでは
`--grid sfnn_l1_revive false true` / `--grid sfnn_l1_revive_zero false true` です。
同じ`initial_state`を共通設定に指定して比較できます。scratch開始にも対応します。

## 判定と適用タイミング

trueのepochの最初のbatchを学習する前に、その時点の教師読み出し位置から16batchを推論します。
検証セットではなく教師を使用し、学習の読み出し位置は進めません。校正時のshuffleは無効です。
学習のsample weightが0の局面は判定から除外します。
量子化proxyの両枝を調べ、bucket内に1,024局面以上あり、その全局面で条件を満たした場合のみ対象にします。
これは有限サンプルでの判定であり、全局面での定数性の証明ではありません。

各種類の処理済み情報はstate.bin/weights.binに保存します。対象0件でも処理済みとなります。
次のepoch開始時には改めて判定します。epoch途中のcheckpointからresumeすると、そのepochでは再実行しません。
4つのrevive設定はepoch別指定にも対応します。指定開始前はfalse、指定後は次の指定まで値を引き継ぎます。
`"sfnn_l1_revive": {"epoch3": true, "epoch4": false}`ならepoch3だけです。
単一のtrueなら全epoch（warmup epoch0を含む）で判定します。詳しくは[epoch設定](epoch-settings.md)。
処理後のcheckpoint保存前に中断した場合は、元checkpointから再判定します。

判定内容は学習出力フォルダの`l1-revive.csv`に記録します。既存ファイルは上書きせず連番にします。

## 何を変更するか

1. 上限側では両枝からL2への定数寄与をL2のbiasに足します。ゼロ側では足しません。
2. 対象unitのL1有効重みをGlorot一様分布で再初期化し、校正入力での平均preactivationが0.5になるようbiasを設定します。
3. 対象の両枝からL2への重みを0にします。これらは学習可能です。最初はL1への勾配が0ですが、L2接続が更新されるとL1も学習を再開します。
4. 変更箇所のoptimizer momentをリセットし、Lookahead slow側も独立に補償します。

L1 shared自体は変更せず、対象bucketの個別重みで差し引きます。他bucket/unit、FT、L3はリセットしません。
L2 reviveの微小な非ゼロ接続とは異なり、L1では新unitによる即時の出力変化を避けるため接続を0にします。
元の枝が定数である局面ではbias移動は代数的に等価ですが、量子化丸めと未観測局面も含む完全一致を保証するものではありません。
再初期化後に再び定数化しない保証や、棋力向上の保証もありません。A/B比較用の機能です。

対応範囲: dense L1、L1 factorizer none/shared、BNなし、通常学習または層別QAT、standalone/grid search。
BN、axis/pair、residual count gate、compact L1、旧L2/L3 factorizer、worker trialは未対応でエラーにします。
nn.binの推論形式は変更しません。既存checkpointに処理済み情報がなければ未処理として読み込みます。

[L2の上限・ゼロrevive](batch-normalization.md)もBNなしに対応しています。4項目を同時にtrueにした場合は、L1を処理した後のネットワークでL2を校正・処理します。
