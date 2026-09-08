# 6. Inspect the result

<a href="../../ja/tutorial/6-result.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

After training, check two things.

| Item | Use |
| --- | --- |
| `000N/nn.bin` | Evaluation file loaded by the engine |
| `summary-learn.csv` | Accuracy / loss history |

## 6.1 Output files

NNUE / SFNN output looks like this:

```text
checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
  summary-learn.csv
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

## 6.2 Log

`summary-learn.csv` is a CSV file, usable with VSCode CSV extensions and spreadsheet software.
If the training folder contains `summary-learn.log`, training startup renames it to `summary-learn.csv` without changing its contents. If both files exist, startup reports an error instead of overwriting either one. Check which history to keep and move the other file elsewhere. Do not manually rename a file while a training process is writing to it.

`summary-learn.csv` contains one row per superbatch.
If that sb did not run ordinary validation, `test_value_accuracy` / `test_value_loss` are `-`.
If that sb did not run quantized validation, `quantized_value_accuracy` / `quantized_value_loss` are `-`.

The four metric columns are adjacent, in **acc → loss → qacc → qloss** order. Checkpoint-local `learn.log` uses the same order. There is no `train_value_loss` column.

The main columns are:

| Column | Meaning |
| --- | --- |
| `epoch` | Current epoch |
| `superbatch` | Current sb inside the epoch |
| `test_value_accuracy` | Validation sign accuracy |
| `test_value_loss` | Validation loss |
| `quantized_value_accuracy` | Post-quantization validation sign accuracy (qacc) |
| `quantized_value_loss` | Post-quantization validation loss (qloss) |
| `positions` | Total processed positions |

For deeper log analysis and plotting, see the [Advanced guide](../advanced/).

---

Next: [7. Load into an engine](7-engine.md)

Detailed checks: [Advanced guide](../advanced/)

Previous: [5. Stop and resume](5-resume.md)
