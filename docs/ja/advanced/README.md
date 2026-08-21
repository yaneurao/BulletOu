# BulletOu 応用編

<a href="../../en/advanced/README.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

この章は、チュートリアルを一通り終えたあとに読むページです。学習設定の調整、追加学習、LayerStack の実験、書き出した `nn.bin` の検証などを扱います。

最初に学習を1回動かすだけなら、まず [チュートリアル](../tutorial/) を読んでください。

| ページ | 内容 |
| --- | --- |
| [学習設定を調整する](tuning.md) | 学習率、保存頻度、検証頻度、loss |
| [ESによる自動調整](auto-tuning.md) | `parameters.json` を使った factorizer / count confidence の自動調整 |
| [loss の scale と `FV_SCALE`](scale-and-fv-scale.md) | WRM loss、単純 sigmoid loss、量子化後の出力 scale |
| [追加学習](additional-training.md) | 完走後にさらに epoch を足す、教師や学習率を変えて続ける |
| [LayerStack](layerstack.md) | SFNN の hand / king / progress bucket |
| [SFNN factorizer](sfnn-factorizer.md) | bucket 間の共有成分、axis/pair、alpha の仕組み |
| [量子化後 `nn.bin` の検証](quantized-nn-bin.md) | `quantized-test` と `calibrate-nn-bin` |
| [`bulletou_lib` をコードから使う](bullet-lib.md) | 独自 example や外部 crate からの利用 |

ファイル形式や実装レベルの詳細は [リファレンス](../reference/) を参照してください。
