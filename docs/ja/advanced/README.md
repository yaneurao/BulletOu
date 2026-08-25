# BulletOu 応用編

<a href="../../en/advanced/README.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

この章は、基本チュートリアルを一通り終えたあとに読むページです。学習条件の調整、追加学習、LayerStack の実験、書き出した `nn.bin` の検証などを扱います。

最初に軽く学習を回したいだけなら、まず [チュートリアル](../tutorial/) を読んでください。

| ページ | 内容 |
| --- | --- |
| [学習設定を調整する](tuning.md) | learning rate、保存頻度、検証頻度、loss |
| [population search による自動調整](auto-tuning.md) | `tuning-settings.json` と `bulletou-settings.json` を使った factorizer / count confidence の自動調整 |
| [loss の scale と `FV_SCALE`](scale-and-fv-scale.md) | WRM loss、sigmoid loss、量子化後の出力 scale |
| [追加学習](additional-training.md) | 完了済み checkpoint からさらに学習する方法 |
| [LayerStack](layerstack.md) | SFNN の hand / king / progress bucket |
| [SFNN factorizer](sfnn-factorizer.md) | shared / axis / pair factorizer と alpha |
| [量子化 `nn.bin` の確認](quantized-nn-bin.md) | `quantized-test` と `calibrate-nn-bin` |
| [`bulletou_lib` をコードから使う](bullet-lib.md) | 独自 example や外部 crate からの利用 |

ファイル形式や実装寄りの詳細は [リファレンス](../reference/) を参照してください。
