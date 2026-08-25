# 3. Run the training

<a href="../../ja/tutorial/3-train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: create an evaluation-function file that YaneuraOu can load.

This page continues from [2. Prepare training data](2-data.md).

## 3.1 Build once

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
```

On Windows, the executable is:

```text
.\target\release\examples\bulletou.exe
```

## 3.2 Minimal settings file

For a first real run, HalfKP NNUE is the easiest target.

Create `bulletou-settings.json` in the BulletOu directory:

```json
{
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "tag": "first-halfkp"
}
```

Then run:

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

`arch` selects the evaluation-function shape. `NNUE_halfkp_256x2_32_32` is a small, easy-to-test NNUE.

`teacher` points to a teacher file or a directory of teacher files.

`tag` names the experiment. It is optional, but useful once you run more than one experiment.

You can still override a JSON value on the command line:

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --tag another-test
```

## 3.3 Output

Training writes checkpoints under `checkpoints/`. For NNUE / SFNN targets, each saved checkpoint contains an `nn.bin`.

Example:

```text
checkpoints/
  NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
    0001/
      nn.bin
      state.bin
```

`nn.bin` is the file you load into YaneuraOu.

## 3.4 Short smoke run

Before launching a long run on huge teacher data, you can force a small run by adding these fields:

```json
{
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "positions_per_superbatch": 1000000,
  "superbatches": 1,
  "max_epochs": 1,
  "tag": "smoke-halfkp"
}
```

Use this to check that loading, training, and saving all work.

## 3.5 What you should see

Training prints lines like:

```text
[train] epoch 1  sb 1/1  this-sb=... pos  wall=...s  train=...s  pos/s=...
```

`pos/s` is the training-speed indicator. Save and validation time are excluded from the training speed.

To watch accuracy / loss during training, configure a validation set on the next page.

## 3.6 Training SFNNs with many buckets

For SFNN architectures with many buckets, such as `hand1024_k3k3_progress4`, rarely seen buckets can learn unstable bucket-specific residuals. You can pre-count bucket occurrences into a `count.bin` file and pass it during training:

```powershell
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin
```

When `--sfnn-bucket-counts` is set and an SFNN factorizer is active, BulletOu enables the residual count gate by default. Low-count buckets lean more on shared factorizer terms; well-observed buckets keep more of their bucket-specific residual.

Disable this gate explicitly if you only want to load the count file for statistics or for other count-confidence options:

```powershell
--sfnn-residual-count-gate-confidence 0
```

You can also apply count-based confidence to axis and pair factorizer terms:

```powershell
--sfnn-axis-count-confidence 1.0 `
--sfnn-pair-count-confidence 1.0
```

If needed, split them by factorizer family, for example `--sfnn-king-axis-count-confidence`, `--sfnn-hand-axis-count-confidence`, `--sfnn-progress-axis-count-confidence`, `--sfnn-king-hand-pair-count-confidence`, `--sfnn-king-progress-pair-count-confidence`, and `--sfnn-hand-progress-pair-count-confidence`.

The same `count.bin` file is used for residual, axis, and pair confidence. For the count command and the exact formulas, see [Advanced: SFNN factorizer](../advanced/sfnn-factorizer.md).

When the architecture contains `progressN`, such as `progress4`, the progress calculation parameters are trained and saved into the Progress section of `nn.bin`. The count command also needs an `nn.bin` for the same architecture so it can use the same progress bucket assignment.

For a `progressN` architecture, BulletOu updates the progress parameters by default. That training path uses neighboring progress buckets during training, so it is much slower than an ordinary hard-bucket path.

Once you no longer want to move the progress parameters, resume with:

```powershell
--sfnn-freeze-progress
```

This freezes the progress parameters and trains with the same hard bucket assignment that is exported to `nn.bin`. It also makes validation caching cheaper. Do not pass this option at the start if you still want BulletOu to learn the progress parameters.

When you build `count.bin` from a very large teacher folder, `bucket-count` reads fixed-size `.psv` / `.bin` files in large chunks while counting. If read speed fluctuates on a drive such as `D:`, see the Advanced guide for `--buffer-mb` and `--read-buffers`.

## 3.7 Use population search-tuned values for normal training

Sometimes you want to keep using the `parameters.current` values from `tuning-settings.json`, but stop running population search candidate search. Set `tuning.enabled` to `false`:

```json
"tuning": {
  "enabled": false
}
```

Then launch the runner:

```powershell
python .\bulletou_tuner.py `
  --tuning-settings-file D:\BulletOu-snapshots\settings\tuning-settings-20260821.json `
  --resume
```

In this mode, the runner launches `bulletou.exe` once. It does not create candidates, worker caches, or snapshots. It only converts `parameters.current` into `--sfnn-factorizer-alpha` and count-confidence options, so memory overhead is roughly the same as running `bulletou.exe` directly.

The runner fills `superbatches` from `trial_sbs` and `max_epochs` from `generations`. It uses `validation_rate` and `quantized_validation_rate` from the `tuning` section of `tuning-settings.json`. Put ordinary training settings such as `lr` and `save_rate` in `bulletou-settings.json`.

For `recommended-parameters.json` and the recommendation formula, see [Advanced: Fixed-length trial parameter tuning](../advanced/parameter-tuning.md).

---

Next: [4. Enable validation](4-validation.md)

For tuning and comparison experiments, see the [Advanced guide](../advanced/).

Previous: [2. Prepare training data](2-data.md)
