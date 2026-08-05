# BulletOu Advanced Guide

<a href="../../ja/advanced/README.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

This section is for practical experiments after you have completed the basic tutorial: tuning, continued training, LayerStack variants, and checking exported `nn.bin` files.

If you only want to run your first training job, start with the [Tutorial](../tutorial/).

| Page | Topic |
| --- | --- |
| [Adjust training settings](tuning.md) | Learning rate, save frequency, validation frequency, loss, and SFNN factorizer |
| [`--scale` and `--fv-scale`](scale-and-fv-scale.md) | How teacher win-rate scale and NNUE/SFNN output range fit together |
| [Continued training](additional-training.md) | Add epochs after a finished run, or continue with a new teacher or LR |
| [LayerStack](layerstack.md) | SFNN hand / king / progress buckets |
| [Quantized `nn.bin` checks](quantized-nn-bin.md) | `quantized-test` and `calibrate-nn-bin` |
| [Using `bulletou_lib` from code](bullet-lib.md) | Custom examples and external crate usage |

For file formats and implementation-level details, see the [Reference docs](../).
