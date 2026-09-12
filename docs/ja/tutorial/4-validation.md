# 4. 検証を有効にする

<a href="../../en/tutorial/4-validation.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習中に accuracy / loss を見たい場合は、学習用の教師データとは別に、検証用局面を指定します。

## 4.1 何を指定するか

検証に関係する基本設定は2つです。

| JSON key / CLI option | 役割 | 省略時 |
| --- | --- | --- |
| `test_teacher` / `--test-teacher` | 検証用局面ファイルを指定する。これを指定しないと `test_value_accuracy` / `test_value_loss` は出ません | 検証しない |
| `validation_rate` / `--validation-rate` | 何 sb ごとに検証するかを指定する。`0` ならepoch末尾だけ、`-1` なら無効 | `save_rate` と同じ |

つまり、検証を有効にする最低条件は `test_teacher` です。

毎 sb で accuracy / loss を見たい場合は、`validation_rate` を `1` にします。epoch末尾だけでよい場合は `0`、検証を一時的に止めたい場合は `-1` にします。

## 4.2 設定例

```json
{
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "test_teacher": "C:/shogi/teacher/test/test.hcpe",
  "validation_rate": 1,
  "positions_per_superbatch": 1000000,
  "superbatches": 1,
  "max_epochs": 1,
  "tag": "first-halfkp"
}
```

実行はこうです。

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

この例では、`teachers` で学習し、`C:\shogi\teacher\test\test.hcpe` で検証します。

検証用局面は、学習に使う教師データとは別のファイルを使ってください。学習データそのもので accuracy / loss を測ると、未知局面に対する性能を見誤ります。

## 4.3 検証に使う局面数

全件検証を明示する場合は、`--test-positions all`、またはJSONに次のように書きます。

```json
{
  "test_positions": "all"
}
```

省略（JSONでは `null` も可）でも同じ全件検証です。`all` は大きな件数で代用するのではなく全件読込の経路を使います。この場合、`test_sample` / `test_seed` は使いません。resume判定と検証cacheも、省略時と同じ扱いです。通常学習・worker・`quantized-test` で共通です。

短時間で動作確認したい場合だけ、次のように局面数を制限します。

```json
{
  "test_positions": 300000
}
```

本格的に比較する場合は、同じ検証ファイル、同じ `test_positions`、同じ `test_sample` を使ってください。

## 4.4 画面に出るもの

検証が有効な場合、sb の区切りで次のような行が出ます。

```text
[train]  epoch 1  sb 1/36  this-sb=... pos  wall=...s  train=...s  pos/s=...
[valid]  epoch 1  sb 1     test_value_accuracy=0.6123456  test_value_loss=0.12345678  elapsed=0.123s
```

`test_value_accuracy` は、検証局面で評価値の符号が勝敗と合っている割合です。

`test_value_loss` は、検証局面での loss です。通常はこちらも下がっているか見ます。

## 4.5 保存頻度とは別に考える

`save_rate` / `--save-rate` は checkpoint を保存する頻度（単位: sb）です。正の整数ならその間隔で保存し、`0` または `"none"` なら途中の定期保存を行いません。省略時は `20` です。

`validation_rate` / `--validation-rate` は accuracy / loss を測る頻度です。

`summary-learn.csv` 自体は1sbごとに1行書かれますが、検証しないsbの
`test_value_accuracy` / `test_value_loss` は `-` になります。

たとえば、保存は epoch 末だけでよく、検証は毎 sb 見たい場合は次のようにします。

```powershell
--save-rate 0 `
--validation-rate 1
```

`bulletou-settings.json` に書くならこうです。

```json
{
  "save_rate": 0,
  "validation_rate": 1
}
```

JSON の `"save_rate": "none"`、CLI の `--save-rate none` も `0` と同じ意味です。JSONでは `none` を文字列として引用符で囲んでください。`null` は省略扱いで、`0` とは異なります。

`--save-epoch-end` はデフォルトで有効なので、`save_rate: 0` でも**各epochの最後のsb**は保存されます。全学習の最終epochだけ、という意味ではありません。

epoch末の暗黙保存も無効にするには、次のように指定します。

```json
{
  "save_rate": 0,
  "no_save_epoch_end": true
}
```

CLIでは `--save-rate none --no-save-epoch-end` です。これで学習中の番号付きcheckpointを保存しなくなります。正常終了時の `cuda-cpp-direct/` への最終出力は別処理で、この指定では無効になりません。途中で停止した場合、未保存の重みからはresumeできません。

正の `save_rate` を指定したままの場合、`no_save_epoch_end` は定期保存を止めません。例えば `superbatches: 324, save_rate: 81` なら、sb 324も定期保存の対象です。

保存頻度を変えても、明示した検証頻度は変わりません。`save_rate: 0` で検証頻度を省略すると、通常のvalidationはepoch末のみ、量子化validationは保存時のみです。`lr_schedule: "plateau"` は従来どおり `save_rate: 1` が必要です。

## 4.6 量子化後の検証

学習中の `test_value_accuracy` / `test_value_loss` は、基本的にはメモリ上の f32 重みで測ります。

保存された `nn.bin` と同じように量子化した後の accuracy / loss も見たい場合は、`quantized_validation_rate` / `--quantized-validation-rate` を使います。`0` ならepoch末尾だけ、`-1` なら量子化 validation を行いません。

```json
{
  "quantized_validation_rate": 1
}
```

量子化後検証をしないsbでは、`summary-learn.csv` の
`quantized_value_accuracy` / `quantized_value_loss` は `-` になります。

量子化後検証は少し重いので、最初は `--test-teacher` と `--validation-rate` だけで十分です。詳しくは [応用編: 量子化後の `nn.bin` を検証する](../advanced/quantized-nn-bin.md) を参照してください。

---

## 4.7 epochごとの結果をCSVで見る

学習フォルダの `summary-epoch-last.csv` に、各epochの最終sbの結果と、epoch内で記録された各指標のベスト値が自動で集計されます。追加のオプションやスクリプト実行は不要です。

```csv
metric,epoch 1,epoch 2
acc,0.620000,0.630000
loss,0.130000,0.120000
qacc,0.619000,0.629000
qloss,0.131000,0.121000
max-acc,0.622000,0.632000
min-loss,0.129000,0.119000
max-qacc,0.621000,0.631000
min-qloss,0.130000,0.120000
lr,0.000875,0.000500
lr-min,0.000030,0.000020
sb,324,324
bpu,1,4
```

新規学習では、開始時にヘッダ行 `metric` だけのファイルを作ります。epochが完了すると、右にそのepochの列が追加されます。行は `acc`、`loss`、`qacc`、`qloss`、`max-acc`、`min-loss`、`max-qacc`、`min-qloss`、`lr`、`lr-min`、`sb`、`bpu` の順です。accuracyは百分率ではなく0～1の値です。

`lr` はそのepochのsb 1開始時のlr、`lr-min` は最終sb終了時に実際に使ったlrです。後者は設定上の下限値とは限りません。`sb` は最終sb番号（そのepochのsb数）、`bpu` は最終sbで使った `batches_per_update` です。epoch途中でbpuを変えた場合も、終了時の値を記録します。

再開時は、既存の `summary-epoch-last.csv` の記録済みセルを優先します。`summary-learn.csv` と保存されている学習設定から再集計するのは、新しいepoch・追加行・空欄を補完するためです。手入力した `bpu` も保持し、元ログの値や空欄で上書きしません。checkpointフォルダや元ログを削除しても、このCSVに記録済みの値は消しません。CSV自体を削除した場合、元ログにない手入力値は復元できないので、このCSVは残してください。

学習途中のepochは追加しません。学習再開で明示的に巻き戻すepochの除去は別扱いです。既存CSVにない最新epochの完了を確認できる記録がない場合、そのepochは追加しません。

先頭の `acc`、`loss`、`qacc`、`qloss` は最終sbの値です。そのsbで検証していない項目は空欄です。

`max-acc`・`max-qacc` はそのepoch内の最大値、`min-loss`・`min-qloss` は最小値です。4項目はそれぞれ独立に集計するので、ベスト値を記録したsbが同じとは限りません。集計対象は `summary-learn.csv` に記録された有限の検証値だけです。検証間隔が8sbならその間隔で得た値のベストであり、未計測sbの値は推定しません。そのepochで一度も有効な値が記録されていない項目は空欄です。

epoch完了処理や量子化後の検証値の追記で新しい値が得られた場合、ベスト値の4行は記録済みのベストより改善したときだけ更新します。元ログが一部消えても、記録済みのベストを悪い値や空欄に戻しません。

sb 1の記録がないepochの `lr` や、bpuの記録がない過去epochの `bpu` も空欄にし、現在の設定からは推定しません。

`batches_per_update` は `summary-learn.csv` にも記録されるため、このCSVを消しても再開時に再集計できます。集計のための追加推論はありません。

## 4.8 qvalid時の飽和率と出力の大きさ

SFNNのGPU量子化後検証では、qacc/qlossと同じforwardの中間値を使って、以下を自動集計します。追加の引数は不要です。`--quantized-validation-rate 1` なら毎sb、保存時のqvalidでも記録します。

| `summary-learn.csv` の列 | 意味 |
|---|---|
| `quantized_ft_upper_ratio` | FTの上限到達率。先後両視点を含む |
| `quantized_l1_upper_ratio` | L1通常枝の上限到達率。skip出力は含めない |
| `quantized_l1_square_upper_ratio` | L1二乗枝の上限到達率。二乗と127/128倍を適用した後の値で判定 |
| `quantized_l2_upper_ratio` | L2の上限到達率 |
| `quantized_output_raw_rms` | 最終出力のRMS。`sqrt(mean(output²)) × 8128`。FV_SCALEで割る前のraw単位 |

上限到達率は、検証した全局面で使用した要素のうち活性化上限1に到達した要素の割合です。重み自体のclipping率ではありません。CSVでは0〜1の割合（0.08456なら8.456%）、stdoutの `[qstats] mode=gpu` 行では百分率で表示します。最終batchが小さい場合も局面数・要素数で加重して集計します。

列は既存のacc/loss/qacc/qlossの順序を変えず、`batches_per_update` と末尾の `checkpoint` の間に追加します。qvalid未実施sb、過去の未計測行、対象外arch、CPU厳密検証時の新しい診断列は空欄です。既存CSVは次回書き込み時に列名で移行し、既存値を保持します。過去の値を現在のモデルで埋め直しません。

これらは**GPU簡易検証の値**です。CPU整数推論の丸めを全層で再現するものではなく、別途CPUで測定した飽和率・raw RMSとは差が生じ得ます。比較時は計算経路をそろえてください。

追加処理はGPU内の集計と小さな集計結果の転送で、再forwardや中間テンソル全体のCPU転送は行いません。集計用追加VRAMは約5KiBです。処理時間はゼロではなく、qvalidのelapsedに含まれます。学習のforward・loss・勾配・重みは変更しません。

---

次へ: [5. 中断・再開](5-resume.md)

詳しい検証指標: [仕様: Validation Metrics](../../spec/06-validation-metrics.md)

前へ: [3. 学習を走らせる](3-train.md)
