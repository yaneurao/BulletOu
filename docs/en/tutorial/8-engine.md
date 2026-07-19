# 8. Load into an engine — verify in YaneuraOu

<a href="../../ja/tutorial/8-engine.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

A minimum walkthrough for verifying the trained weights in a YaneuraOu engine.

## 8.1 For NNUE evals (`nn.bin`)

Put the latest `000N/nn.bin` where the engine looks for its eval file. With YaneuraOu the path is set via the `EvalDir` USI option:

```
# After the engine starts, in the USI command line:
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/0005
isready
bench
```

Alternatively, place `000N/nn.bin` as `eval/nn.bin` if your engine expects that relative path.

`isready` succeeding means the engine loaded the file. `bench` prints the hash of the loaded `nn.bin`, so a different number on each re-trained model confirms you're really using different weights.

## 8.2 For KPPT-family evals (three-file set)

Point `EvalDir` at the latest `000N/` directory directly (it must contain all three files):

```
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/KPPT/0005
isready
bench
```

The engine refuses to load if any of the three files is missing.

## 8.3 If the result is weak

The first training run uses a small teacher and few superbatches, so don't expect competitive strength. To get something usable in real play:
- Increase teacher size (100M → 1B+ positions)
- Run several epochs (e.g. `--max-epochs 3`)
- Increase `--save-rate` (e.g. 10) and only use the later saves; epoch-end saves are kept by default

Per-eval-type hyperparameter advice lives in the reference docs ([halfkp.md](../shogi/halfkp.md) / [kp.md](../shogi/kp.md) / [halfkpe9.md](../shogi/halfkpe9.md) / [kppt.md](../shogi/kppt.md)).

---

Previous: [7. Inspect the result](7-result.md)
