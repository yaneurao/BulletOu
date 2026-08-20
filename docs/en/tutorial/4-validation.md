# 4. Enable validation

<a href="../../ja/tutorial/4-validation.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

To watch accuracy / loss during training, point BulletOu at a validation set separate from the training teacher.

## 4.1 Required options

Two options control ordinary validation:

| Option | Meaning | If omitted |
| --- | --- | --- |
| `--test-teacher` | Validation-position file. Without this, `test_value_accuracy` / `test_value_loss` are not computed | no validation |
| `--validation-rate` | Validate every N sb | same as `--save-rate` |

So the minimum option needed to enable validation is `--test-teacher`.

If you want accuracy / loss every sb, also pass `--validation-rate 1`.

## 4.2 Example command

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --test-teacher C:\shogi\teacher\test\test.hcpe `
  --validation-rate 1 `
  --positions-per-superbatch 1000000 `
  --superbatches 1 `
  --max-epochs 1 `
  --tag first-halfkp
```

This trains on `teachers` and validates on `C:\shogi\teacher\test\test.hcpe`.

Use a validation file that is separate from the training teacher. Measuring accuracy / loss on the same data used for training can make the model look better than it is on unseen positions.

## 4.3 Number of validation positions

If `--test-positions` is omitted, BulletOu uses every position in the validation file.

For a quick check, limit the count:

```powershell
--test-positions 300000
```

For comparisons, keep the validation file, `--test-positions`, and `--test-sample` fixed.

## 4.4 Output

When validation is enabled, BulletOu prints lines like:

```text
[train]  epoch 1  sb 1/36  this-sb=... pos  wall=...s  train=...s  pos/s=...
[valid]  epoch 1  sb 1     test_value_accuracy=0.6123456  test_value_loss=0.12345678  elapsed=0.123s
```

`test_value_accuracy` is sign agreement on the validation positions.

`test_value_loss` is the validation loss. In practice, watch both accuracy and loss.

## 4.5 Save frequency is separate

`--save-rate` controls checkpoint saves.

`--validation-rate` controls accuracy / loss measurement.

`summary-learn.log` still gets one row per sb. For sb where ordinary
validation is not run, `test_value_accuracy` / `test_value_loss` are `-`.

For example, to save only at epoch end but validate every sb:

```powershell
--save-rate 9999 `
--validation-rate 1
```

`--save-epoch-end` is enabled by default, so epoch-end checkpoints are still written even when `--save-rate` is large.

## 4.6 Quantized validation

Ordinary `test_value_accuracy` / `test_value_loss` are measured with the in-memory f32 weights.

To also watch accuracy / loss after quantizing like `nn.bin`, use `--quantized-validation-rate`:

```powershell
--quantized-validation-rate 1
```

For sb where quantized validation is not run, `summary-learn.log` writes
`quantized_value_accuracy` / `quantized_value_loss` as `-`.

Quantized validation is heavier, so start with only `--test-teacher` and `--validation-rate`. For details, see [Advanced: Validate a quantized `nn.bin`](../advanced/quantized-nn-bin.md).

---

Next: [5. Stop and resume](5-resume.md)

Metric details: [Spec: Validation Metrics](../../spec/06-validation-metrics.md)

Previous: [3. Run the training](3-train.md)
