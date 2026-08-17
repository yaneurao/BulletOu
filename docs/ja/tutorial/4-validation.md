# 4. 検証を有効にする

<a href="../../en/tutorial/4-validation.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習中に accuracy / loss を見たい場合は、学習用の教師データとは別に、検証用局面を指定します。

## 4.1 何を指定するか

検証に関係する基本オプションは2つです。

| オプション | 役割 | 省略時 |
| --- | --- | --- |
| `--test-teacher` | 検証用局面ファイルを指定する。これを指定しないと `test_value_accuracy` / `test_value_loss` は出ません | 検証しない |
| `--validation-rate` | 何 sb ごとに検証するかを指定する | `--save-rate` と同じ |

つまり、検証を有効にする最低条件は `--test-teacher` です。

毎 sb で accuracy / loss を見たい場合は、`--validation-rate 1` も指定します。

## 4.2 コマンド例

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --test-teacher C:\shogi\teacher\test\test.hcpe `
  --validation-rate 1 `
  --positions-per-superbatch 1000000 `
  --superbatches 1 `
  --max-epochs 1 `
  --tag first-halfkp
```

この例では、`teachers` で学習し、`C:\shogi\teacher\test\test.hcpe` で検証します。

検証用局面は、学習に使う教師データとは別のファイルを使ってください。学習データそのもので accuracy / loss を測ると、未知局面に対する性能を見誤ります。

## 4.3 検証に使う局面数

`--test-positions` を省略すると、検証用ファイルの全局面を使います。

短時間で動作確認したい場合だけ、次のように局面数を制限します。

```powershell
--test-positions 300000
```

本格的に比較する場合は、同じ検証ファイル、同じ `--test-positions`、同じ `--test-sample` を使ってください。

## 4.4 画面に出るもの

検証が有効な場合、sb の区切りで次のような行が出ます。

```text
[train]  epoch 1  sb 1/36  this-sb=... pos  wall=...s  train=...s  pos/s=...
[valid]  epoch 1  sb 1     test_value_accuracy=0.6123456  test_value_loss=0.12345678  elapsed=0.123s
```

`test_value_accuracy` は、検証局面で評価値の符号が勝敗と合っている割合です。

`test_value_loss` は、検証局面での loss です。通常はこちらも下がっているか見ます。

## 4.5 保存頻度とは別に考える

`--save-rate` は checkpoint を保存する頻度です。

`--validation-rate` は accuracy / loss を測る頻度です。

たとえば、保存は epoch 末だけでよく、検証は毎 sb 見たい場合は次のようにします。

```powershell
--save-rate 9999 `
--validation-rate 1
```

`--save-epoch-end` はデフォルトで有効なので、`--save-rate` を大きくしても epoch 末の checkpoint は保存されます。

## 4.6 量子化後の検証

学習中の `test_value_accuracy` / `test_value_loss` は、基本的にはメモリ上の f32 重みで測ります。

保存された `nn.bin` と同じように量子化した後の accuracy / loss も見たい場合は、`--quantized-validation-rate` を使います。

```powershell
--quantized-validation-rate 1
```

量子化後検証は少し重いので、最初は `--test-teacher` と `--validation-rate` だけで十分です。詳しくは [応用編: 量子化後の `nn.bin` を検証する](../advanced/quantized-nn-bin.md) を参照してください。

---

次へ: [5. 中断・再開](5-resume.md)

詳しい検証指標: [仕様: Validation Metrics](../../spec/06-validation-metrics.md)

前へ: [3. 学習を走らせる](3-train.md)
