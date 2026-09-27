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

To explicitly validate every position, pass `--test-positions all`, or use:

```json
{
  "test_positions": "all"
}
```

Omission (or JSON `null`) means the same thing. `all` uses the full-file reader, not an artificially large sample count. `test_sample` / `test_seed` are ignored in this case. Resume signatures and validation cache keys are identical to omission. This applies to ordinary training, worker mode, and `quantized-test`.

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

`save_rate` / `--save-rate` controls checkpoint saves in sb units. A positive integer saves at that interval; `0` or `"none"` disables periodic intermediate saves. Omission defaults to `20`.

`validation_rate` / `--validation-rate` controls accuracy / loss measurement.

`summary-learn.csv` still gets one row per sb. For sb where ordinary
validation is not run, `test_value_accuracy` / `test_value_loss` are `-`.

For example, to save only at epoch end but validate every sb:

```powershell
--save-rate 0 `
--validation-rate 1
```

or in `bulletou-settings.json`:

```json
{
  "save_rate": 0,
  "validation_rate": 1
}
```

JSON `"save_rate": "none"` and CLI `--save-rate none` mean exactly the same as `0`. Quote `none` as a string in JSON. JSON `null` means omission, not `0`.

`--save-epoch-end` is enabled by default, so `save_rate: 0` still saves **the final sb of each epoch**, not just the final epoch of the entire run.

To disable implicit epoch-end saves as well:

```json
{
  "save_rate": 0,
  "save_epoch_end": false
}
```

The CLI equivalent is `--save-rate none --save-epoch-end false`. This disables numbered checkpoints during training. The final output to `cuda-cpp-direct/` on normal completion is a separate operation and is not disabled by these settings. Unsaved weights cannot be resumed after interruption.

With a positive `save_rate`, `save_epoch_end: false` does not disable periodic saves. For example, `superbatches: 324, save_rate: 81` still saves at sb 324 because it is a periodic boundary.

Explicit validation rates remain independent of saves. With `save_rate: 0` and omitted validation rates, ordinary validation runs at epoch end, while quantized validation runs only on saves. `lr_schedule: "plateau"` still requires `save_rate: 1`.

## 4.6 Quantized validation

Ordinary `test_value_accuracy` / `test_value_loss` are measured with the in-memory f32 weights.

To also watch accuracy / loss after quantizing like `nn.bin`, use `--quantized-validation-rate`. Use `0` for epoch-end-only quantized validation, or `-1` to disable it:

```json
{
  "quantized_validation_rate": 1
}
```

For sb where quantized validation is not run, `summary-learn.csv` writes
`quantized_value_accuracy` / `quantized_value_loss` as `-`.

Quantized validation is heavier, so start with only `--test-teacher` and `--validation-rate`. For details, see [Advanced: Validate a quantized `nn.bin`](../advanced/quantized-nn-bin.md).

---

## 4.7 Training results in CSV

Training results are recorded in `summary-learn.csv`. For per-epoch grid-search results, use `grid_summary.csv`. The retired `summary-epoch-last.csv` is no longer generated or updated; existing files are left untouched.

## 4.8 Quantized activation saturation and output magnitude

SFNN GPU quantized validation automatically aggregates the existing qacc/qloss forward buffers. No new option is needed. With `--quantized-validation-rate 1`, diagnostics are measured every sb; save-time GPU qvalid also records them.

| `summary-learn.csv` column | Meaning |
|---|---|
| `quantized_ft_upper_ratio` | FT upper-saturation ratio, including both perspectives |
| `quantized_l1_upper_ratio` | L1 normal-branch upper ratio, excluding the skip output |
| `quantized_l1_square_upper_ratio` | L1 squared-branch upper ratio, after squaring and multiplying by 127/128 |
| `quantized_l2_upper_ratio` | L2 upper-saturation ratio |
| `quantized_output_raw_rms` | Final output RMS: `sqrt(mean(output²)) × 8128`, before FV_SCALE division |

Upper ratios count activation elements reaching the upper bound of 1, not clipped weights. They cover the elements actually used by all validation positions. CSV ratios are 0–1 (0.08456 means 8.456%); stdout `[qstats] mode=gpu` displays percentages. Aggregation weights batches by their actual element counts, including a short final batch.

The `[qstats]` / `[qstats-unit]` console lines require `--verbose` (JSON: `"verbose": true`). `grid_search.py` also accepts `--verbose`. Without it, measurement and existing CSV recording continue unchanged.

With verbose output, GPU qvalid also prints `[qstats-unit]`: **maximum per-unit upper saturation rates** for
`ft_unit_upper_max`, `l1_unit_upper_max`, `l1_square_unit_upper_max`, and `l2_unit_upper_max`.
FT combines both perspectives (denominator `2 × positions`). L1/L2 use each bucket's
position count as denominator. L1's linear skip output is excluded. Counts are summed
over the entire validation set before taking the maximum, not maximized per batch.

Example: `l1_unit_upper_max=100.0000%(bucket=2,unit=6,hits=34,n=34)`.
Bucket/unit IDs are zero-based; `hits` is the upper-hit count and `n` the denominator.
FT reports `bucket=shared`. Unseen buckets are excluded; rare buckets are not.
Ties select the first bucket/unit. A rate of 100% means constant 1 on the observed
validation positions for that bucket, not proof of a constant over every possible position.
L3 is linear without an activation clamp, so it reports `l3_unit_upper_max=n/a(no-clamp)`.
These diagnostics apply to quantized GPU forward, not ordinary f32 validation or CPU qvalid.
Existing averages and CSV columns remain unchanged, as do weights and optimizer updates.

The five columns follow `batches_per_update` and precede the final `checkpoint` column. The acc/loss/qacc/qloss order is unchanged. New diagnostic cells remain empty for unmeasured sb, historical rows, unsupported architectures and CPU-exact validation. Existing CSVs migrate by column name at the next write, preserving existing values; historical diagnostics are not inferred from current weights.

These are **GPU approximate-validation diagnostics**, not exact integer-engine measurements. Layer rounding differs from CPU-exact inference; compare results from the same path.

The extra operation reduces existing GPU buffers and downloads only small partial sums. It performs no second forward or full activation readback and adds about 5 KiB of VRAM. Its nonzero overhead is included in qvalid elapsed time. Training forward, loss, gradients and weights are unchanged.

---

Next: [5. Stop and resume](5-resume.md)

Metric details: [Spec: Validation Metrics](../../spec/06-validation-metrics.md)

Previous: [3. Run the training](3-train.md)
