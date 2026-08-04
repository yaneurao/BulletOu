# 5. Inspect the result

<a href="../../ja/tutorial/5-result.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

After training, check two things.

| Item | Use |
| --- | --- |
| `000N/nn.bin` | Evaluation file loaded by the engine |
| `summary-learn.log` | Accuracy / loss history |

## 5.1 Output files

NNUE / SFNN output looks like this:

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

Pass the `nn.bin` from the checkpoint you want to test to the engine.

`state.bin` is only for BulletOu resume. The engine does not use it.

For KPPT-family evals, the output is three files instead of `nn.bin`:

```text
KK_synthesized.bin
KKP_synthesized.bin
KPP_synthesized.bin
```

## 5.2 Log

`summary-learn.log` contains one row per validated superbatch.

The main columns are:

| Column | Meaning |
| --- | --- |
| `epoch` | Current epoch |
| `superbatch` | Current sb inside the epoch |
| `test_value_accuracy` | Validation sign accuracy |
| `test_value_loss` | Validation loss |
| `train_value_loss` | Training-batch loss |
| `positions` | Total processed positions |

For deeper log analysis and plotting, see the [Advanced guide](../advanced/).

---

Next: [6. Load into an engine](6-engine.md)

Detailed checks: [Advanced guide](../advanced/)

Previous: [4. Stop and resume](4-resume.md)
