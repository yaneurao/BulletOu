# 4. Enable validation

<a href="../../ja/tutorial/4-validation.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

To watch accuracy / loss during training, point BulletOu at a validation set separate from the training teacher.

## 4.1 Required settings

Two settings control ordinary validation:

| JSON key / CLI option | Meaning | If omitted |
| --- | --- | --- |
| `test_teacher` / `--test-teacher` | Validation-position file. Without this, `test_value_accuracy` / `test_value_loss` are not computed | no validation |
| `validation_rate` / `--validation-rate` | Validate every N sb. `0` means epoch-end only; `-1` disables it | same as `save_rate` |

So the minimum setting needed to enable validation is `test_teacher`.

If you want accuracy / loss every sb, set `validation_rate` to `1`. Use `0` for epoch-end-only validation, or `-1` to disable validation temporarily.

## 4.2 Example settings

```json
{
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "test_teacher": "C:/shogi/teacher/test/test.hcpe",
  "validation_rate": 1,
  "positions_per_superbatch": 1000000,
  "superbatches": 1,
  "max_epochs": 1,
  "tag": "first-halfkp"
}
```

Run it with:

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

This trains on `teachers` and validates on `C:\shogi\teacher\test\test.hcpe`.

Use a validation file that is separate from the training teacher. Measuring accuracy / loss on the same data used for training can make the model look better than it is on unseen positions.

## 4.3 Number of validation positions

If `test_positions` / `--test-positions` is omitted, BulletOu uses every position in the validation file.

For a quick check, limit the count:

```json
{
  "test_positions": 300000
}
```

For comparisons, keep the validation file, `test_positions`, and `test_sample` fixed.

## 4.4 Output

When validation is enabled, BulletOu prints lines like:

```text
[train]  epoch 1  sb 1/36  this-sb=... pos  wall=...s  train=...s  pos/s=...
[valid]  epoch 1  sb 1     test_value_accuracy=0.6123456  test_value_loss=0.12345678  elapsed=0.123s
```

`test_value_accuracy` is sign agreement on the validation positions.

`test_value_loss` is the validation loss. In practice, watch both accuracy and loss.

## 4.5 Save frequency is separate

`save_rate` / `--save-rate` controls checkpoint saves.

`validation_rate` / `--validation-rate` controls accuracy / loss measurement.

`summary-learn.log` still gets one row per sb. For sb where ordinary
validation is not run, `test_value_accuracy` / `test_value_loss` are `-`.

For example, to save only at epoch end but validate every sb:

```powershell
--save-rate 9999 `
--validation-rate 1
```

or in `bulletou-settings.json`:

```json
{
  "save_rate": 9999,
  "validation_rate": 1
}
```

`--save-epoch-end` is enabled by default, so epoch-end checkpoints are still written even when `--save-rate` is large.

## 4.6 Quantized validation

Ordinary `test_value_accuracy` / `test_value_loss` are measured with the in-memory f32 weights.

To also watch accuracy / loss after quantizing like `nn.bin`, use `--quantized-validation-rate`. Use `0` for epoch-end-only quantized validation, or `-1` to disable it:

```json
{
  "quantized_validation_rate": 1
}
```

For sb where quantized validation is not run, `summary-learn.log` writes
`quantized_value_accuracy` / `quantized_value_loss` as `-`.

Quantized validation is heavier, so start with only `--test-teacher` and `--validation-rate`. For details, see [Advanced: Validate a quantized `nn.bin`](../advanced/quantized-nn-bin.md).

---

Next: [5. Stop and resume](5-resume.md)

Metric details: [Spec: Validation Metrics](../../spec/06-validation-metrics.md)

Previous: [3. Run the training](3-train.md)
