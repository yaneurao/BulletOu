# BulletOu 応用編

<a href="../../en/advanced/README.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

この章は、チュートリアルを一通り終えて「もっと良くしたい」「比較実験したい」「出力を詳しく検証したい」ときに読むページです。

最初に学習を1回動かすだけなら、まず [チュートリアル](../tutorial/) を読んでください。

| ページ | 内容 |
| --- | --- |
| [学習設定を調整する](tuning.md) | 学習率、保存頻度、検証頻度、loss、SFNN factorizer |
| [`--scale` と `--fv-scale`](scale-and-fv-scale.md) | 教師評価値を勝率へ戻す係数と、NNUE/SFNN の出力レンジの関係 |
| [追加学習](additional-training.md) | 完走後にさらに epoch を足す、教師や学習率を変えて続ける |
| [LayerStack](layerstack.md) | SFNN の hand / king / progress bucket を使う |
| [量子化後 `nn.bin` の検証](quantized-nn-bin.md) | `quantized-test` と `calibrate-nn-bin` |
| [`bulletou_lib` をコードから使う](bullet-lib.md) | 独自 example や外部 crate からの利用 |

仕様レベルの詳細は [リファレンス](../) を参照してください。
