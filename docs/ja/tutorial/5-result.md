# 5. 結果を確認する

<a href="../../en/tutorial/5-result.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習後に見る場所は2つです。

| 見るもの | 用途 |
| --- | --- |
| `000N/nn.bin` | エンジンに読み込ませる評価関数 |
| `summary-learn.log` | accuracy / loss の推移を見るログ |

## 5.1 出力ファイル

NNUE / SFNN の出力例です。

```text
checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
  summary-learn.log
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

## 5.2 ログ

`summary-learn.log` には、superbatch ごとの検証結果が入ります。

よく見る列は次の通りです。

| 列 | 意味 |
| --- | --- |
| `epoch` | 何 epoch 目か |
| `superbatch` | epoch 内の何 sb 目か |
| `test_value_accuracy` | 検証局面での符号一致率 |
| `test_value_loss` | 検証局面での loss |
| `train_value_loss` | 予約列。現在の cuda-cpp 学習では `-` |
| `positions` | 累積で処理した局面数 |

詳しい読み方やプロットは [応用編](../advanced/) を参照してください。

---

次へ: [6. エンジンに組み込む](6-engine.md)

詳しい検証: [応用編](../advanced/)

前へ: [4. 中断・再開](4-resume.md)
