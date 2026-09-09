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

`test_positions` / `--test-positions` を省略すると、検証用ファイルの全局面を使います。

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

`save_rate` / `--save-rate` は checkpoint を保存する頻度です。

`validation_rate` / `--validation-rate` は accuracy / loss を測る頻度です。

`summary-learn.csv` 自体は1sbごとに1行書かれますが、検証しないsbの
`test_value_accuracy` / `test_value_loss` は `-` になります。

たとえば、保存は epoch 末だけでよく、検証は毎 sb 見たい場合は次のようにします。

```powershell
--save-rate 9999 `
--validation-rate 1
```

`bulletou-settings.json` に書くならこうです。

```json
{
  "save_rate": 9999,
  "validation_rate": 1
}
```

`--save-epoch-end` はデフォルトで有効なので、`--save-rate` を大きくしても epoch 末の checkpoint は保存されます。

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

`batches_per_update` は `summary-learn.csv` のcheckpoint列の直前にも記録されるため、このCSVを消しても再開時に再集計できます。集計のための追加推論はありません。

---

次へ: [5. 中断・再開](5-resume.md)

詳しい検証指標: [仕様: Validation Metrics](../../spec/06-validation-metrics.md)

前へ: [3. 学習を走らせる](3-train.md)
