# 6. 結果を確認する

<a href="../../en/tutorial/6-result.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習後に見る場所は2つです。

| 見るもの | 用途 |
| --- | --- |
| `000N/nn.bin` | エンジンに読み込ませる評価関数 |
| `summary-learn.csv` | accuracy / loss の推移を見るログ |

## 6.1 出力ファイル

NNUE / SFNN の出力例です。

```text
checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
  summary-learn.csv
  0001/
    nn.bin
    state.bin
    learn.log
  0002/
    nn.bin
    state.bin
    learn.log
```

エンジンに渡すのは、使いたい checkpoint の `nn.bin` です。

`state.bin` は BulletOu が再開するためのファイルです。エンジンには渡しません。

KPPT 系では `nn.bin` の代わりに、次の3ファイルが出ます。

```text
KK_synthesized.bin
KKP_synthesized.bin
KPP_synthesized.bin
```

## 6.2 ログ

`summary-learn.csv` はCSV形式です。VSCodeのCSV用拡張機能や表計算ソフトで開けます。
学習フォルダに `summary-learn.log` があれば、学習開始時に内容を保ったまま `summary-learn.csv` へ名前を変更します。両方ある場合は上書きせずエラーにするので、残すファイルを確認して片方を別の場所へ移してください。実行中の学習が使っているファイルは手で改名しないでください。

`summary-learn.csv` には、superbatch ごとに1行ずつ進捗が入ります。
検証を行っていないsbでは、`test_value_accuracy` / `test_value_loss` は `-` になります。
量子化後検証を行っていないsbでは、`quantized_value_accuracy` / `quantized_value_loss` は `-` になります。

4つの指標列は **acc → loss → qacc → qloss** の順に並びます。保存フォルダ内の `learn.log` も同じ順序です。`train_value_loss` 列は出力しません。

よく見る列は次の通りです。

| 列 | 意味 |
| --- | --- |
| `epoch` | 何 epoch 目か |
| `superbatch` | epoch 内の何 sb 目か |
| `test_value_accuracy` | 検証局面での符号一致率 |
| `test_value_loss` | 検証局面での loss |
| `quantized_value_accuracy` | 量子化後の検証局面での符号一致率（qacc） |
| `quantized_value_loss` | 量子化後の検証局面でのloss（qloss） |
| `positions` | 累積で処理した局面数 |

詳しい読み方やプロットは [応用編](../advanced/) を参照してください。

---

次へ: [7. エンジンに組み込む](7-engine.md)

詳しい検証: [応用編](../advanced/)

前へ: [5. 中断・再開](5-resume.md)
