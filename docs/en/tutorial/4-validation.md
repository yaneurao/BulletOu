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

## 4.7 Compare epoch results in CSV

BulletOu automatically writes `summary-epoch-last.csv` in the training directory. No extra option or script is needed. Each column contains the final sb of a completed epoch:

```csv
metric,epoch 1,epoch 2
acc,0.620000,0.630000
loss,0.130000,0.120000
qacc,0.619000,0.629000
qloss,0.131000,0.121000
lr,0.000875,0.000500
lr-min,0.000030,0.000020
sb,324,324
bpu,1,4
```

A new run creates the file at startup with just the header line `metric`. Each completed epoch adds a column on the right. Rows are ordered as `acc`, `loss`, `qacc`, `qloss`, `lr`, `lr-min`, `sb`, and `bpu`. Accuracies are ratios from 0 to 1, not percentages.

`lr` is the LR at the start of sb 1; `lr-min` is the LR actually used at the end of the final sb, not necessarily the configured lower bound. `sb` is the final sb number (the number of sb in that epoch). `bpu` is the `batches_per_update` used in the final sb, including when this setting changed during the epoch.

If this CSV is missing when resuming, it is reconstructed from `summary-learn.log` and the recorded training settings. Incomplete epochs and results rolled back on resume are excluded. If the latest epoch cannot be confirmed complete, it is omitted.

The four validation metrics come from the final sb, not the best or average metrics of that epoch. Unmeasured metrics are blank. `lr` is also blank if sb 1 is missing, and historical `bpu` values without a record are blank. Neither is inferred from the current settings.

`batches_per_update` is also recorded immediately before `checkpoint` in `summary-learn.log`, allowing this CSV to be reconstructed after deletion. Exporting the table performs no additional inference.

---

Next: [5. Stop and resume](5-resume.md)

Metric details: [Spec: Validation Metrics](../../spec/06-validation-metrics.md)

Previous: [3. Run the training](3-train.md)
