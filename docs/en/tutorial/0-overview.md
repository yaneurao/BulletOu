# 0. Overview

<a href="../../ja/tutorial/0-overview.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

BulletOu trains evaluation functions for shogi engines.
It reads a large teacher-position set and writes files such as `nn.bin` that YaneuraOu can load.

BulletOu does not play shogi. Its job is only to learn an evaluation function from teacher data.

## Start with HalfKP NNUE

This tutorial starts with:

```text
NNUE_halfkp_256x2_32_32
```

It is small, easy to train, and good for checking that the whole pipeline works.

BulletOu can also train NNUE K-P, NNUE K-A2, SFNN, and KPPT-family evals. See the [Reference docs](../) and the [Advanced guide](../advanced/) for details.

## Overall flow

```text
prepare teacher data
        ↓
train with BulletOu
        ↓
get nn.bin under checkpoints/
        ↓
load nn.bin in YaneuraOu
```

## What this tutorial covers

| # | Page | Topic |
| --- | --- | --- |
| 1 | [Quick Start](1-quickstart.md) | Build and run a smoke test |
| 2 | [Prepare training data](2-data.md) | Prepare `--arch` and `--teacher` |
| 3 | [Run the training](3-train.md) | Run the minimal command |
| 4 | [Stop and resume](4-resume.md) | Continue after interruption |
| 5 | [Inspect the result](5-result.md) | Check output files and logs |
| 6 | [Load into an engine](6-engine.md) | Verify in YaneuraOu |

Next: [1. Quick Start](1-quickstart.md)
