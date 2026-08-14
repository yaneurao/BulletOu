# 2. Prepare training data

<a href="../../ja/tutorial/2-data.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: choose one evaluation-function architecture and prepare teacher data for BulletOu.

## 2.1 First `--arch`

Start with:

```text
NNUE_halfkp_256x2_32_32
```

This is a small HalfKP NNUE. It is easy to train, easy to debug, and easy to load in YaneuraOu.

Try other evaluation functions after your first training run works.
This tutorial uses only this `--arch`.

## 2.2 Teacher data

BulletOu reads these formats:

| Extension | Meaning |
| --- | --- |
| `.psv` | YaneuraOu-style fixed-position data |
| `.bin` | Treated as the same format as `.psv` |
| `.hcpe` | Apery / dlshogi-style fixed-position data |
| `.hcpe3` | dlshogi-style game data |
| `.pack` | Output from YaneuraOu's gensfen scripts |

In this tutorial, put teacher files under a `teachers/` directory:

```text
teachers/
  teacher.psv
```

Passing `--teacher teachers` makes BulletOu read the teacher files in that directory.

## 2.3 Shuffle

Training is more stable when teacher positions are well mixed.

BulletOu shuffles during training by default, so you do not need a shuffle option for the first run.
If you later need to tune memory usage or the shuffle window, see the [Advanced guide](../advanced/).

## 2.4 When teacher score scales differ

Even when two teacher datasets were both produced by DL-based re-scoring, their score magnitudes may differ if the DL win rate was converted back to eval scores with different coefficients.

You can ignore this for your first training run. When you start mixing multiple teacher datasets, see [Loss scale and `FV_SCALE`](../advanced/scale-and-fv-scale.md) in the advanced guide.

---

Next: [3. Run the training](3-train.md)

Previous: [1. Quick Start](1-quickstart.md)
