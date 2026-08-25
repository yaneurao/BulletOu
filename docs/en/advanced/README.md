# BulletOu Advanced Guide

<a href="../../ja/advanced/README.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

This section is for practical experiments after you have completed the basic tutorial: tuning, continued training, LayerStack variants, and checking exported `nn.bin` files.

If you only want to run your first training job, start with the [Tutorial](../tutorial/).

| Page | Topic |
| --- | --- |
| [Adjust training settings](tuning.md) | Learning rate, save frequency, validation frequency, and loss |
| [Automatic ES tuning](auto-tuning.md) | Automatic factorizer / count-confidence tuning with `es-settings.json` and `bulletou-settings.json` |
| [Optuna-style fixed trial search](optuna-style-tuning.md) | Short fixed-parameter trials for `lr`, `lr_min`, factorizer, and count confidence |
| [Loss scale and `FV_SCALE`](scale-and-fv-scale.md) | WRM loss, plain sigmoid loss, and quantized output scale |
| [Continued training](additional-training.md) | Add epochs after a finished run, or continue with a new teacher or LR |
| [LayerStack](layerstack.md) | SFNN hand / king / progress buckets |
| [SFNN factorizer](sfnn-factorizer.md) | Shared components between buckets, axis/pair, and alpha |
| [Quantized `nn.bin` checks](quantized-nn-bin.md) | `quantized-test` and `calibrate-nn-bin` |
| [Using `bulletou_lib` from code](bullet-lib.md) | Custom examples and external crate usage |

For file formats and implementation-level details, see the [Reference docs](../reference/).
