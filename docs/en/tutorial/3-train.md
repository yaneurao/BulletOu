# 3. Run the training

<a href="../../ja/tutorial/3-train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: create an evaluation-function file that YaneuraOu can load.

This page continues from [2. Prepare training data](2-data.md).

To experiment with quantization-aware training, SFNN dense L1 supports `--sfnn-qat-l1`
(JSON: `"sfnn_qat_l1": true`, **off by default**). See [L1 QAT](../advanced/quantized-nn-bin.md#compare-l1-qat-onoff)
for scope, metric definitions and a same-checkpoint A/B comparison.

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

When the architecture contains `progressN`, such as `progress4`, the progress calculation parameters decide the progress bucket. BulletOu saves those parameters to `progress.bin` next to each checkpoint. Use that file when you build `count.bin`:

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\sojoteam_datasets `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --progress-bin D:\path\to\checkpoint\progress.bin `
  --output D:\BulletOu-snapshots\counts\count.bin
```

If you only have an existing `state.bin` or `nn.bin`, extract `progress.bin` first:

```powershell
.\target\release\examples\bulletou.exe export-progress-bin `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --state-bin D:\path\to\checkpoint\state.bin `
  --output D:\path\to\checkpoint\progress.bin
```

Normal evaluation-network training always keeps the progress classifier fixed and uses the hard bucket assignment exported to `nn.bin`. Validation caches can be reused. Train the classifier separately with [`progress-train`](../advanced/progress-training.md); evaluation loss never updates it.

For count-aware fine-tuning, usually pass the same `progress.bin` during training:

```powershell
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin `
--sfnn-progress-bin D:\path\to\checkpoint\progress.bin
```

If `--sfnn-progress-bin` is omitted when resuming, BulletOu keeps the classifier from `state.bin`. For a fresh run, supply a trained `progress.bin`: otherwise the untrained initial classifier stays fixed. `count.bin` and `progress.bin` are not strictly paired, so you can intentionally mix provisional files during experiments.

To train only the progress classifier from complete `.pack` games, with the first position mapped to 0 and the last to 255, see [Advanced: Training a progress classifier](../advanced/progress-training.md).

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
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json `
  --resume
```

In this mode, the runner launches `bulletou.exe` once. It does not create candidates, worker caches, or snapshots. It only converts `parameters.current` into `--sfnn-factorizer-alpha` and count-confidence options, so memory overhead is roughly the same as running `bulletou.exe` directly.

The runner fills `superbatches` from `trial_sbs` and `max_epochs` from `generations`. It uses `validation_rate` and `quantized_validation_rate` from the `tuning` section of `tuning-settings.json`. Put ordinary training settings such as `lr` and `save_rate` in `bulletou-settings.json`.

For `recommended-parameters.json` and the recommendation formula, see [Advanced: Fixed-length trial parameter tuning](../advanced/parameter-tuning.md).

---

Next: [4. Enable validation](4-validation.md)

HalfKA2 / HalfKP FT (first-layer) weight sharing is enabled by default. Disable it with `"no_ft_factorize": true` in JSON or `--no-ft-factorize` on the CLI. This is separate from `"sfnn_factorizer": "none"`, which disables L1 sharing. L2/L3 do not use sharing. Resume with the same FT ON/OFF setting as the saved checkpoint.

For `SFNN_halfka2`, `"ft_factorizer_alpha": 0.5` also controls FT sharing strength (default 1.0). L1 shared strength is set separately with `"sfnn_factorizer_alpha": "shared=0.5"`. When resuming with changed alpha from a checkpoint that records its coefficients, rebase preserves the immediate effective weights. See [settings and rebase caveats](../advanced/sfnn-factorizer.md).

SFNN enables tatara-style weight clipping by default: L1/L2 weights and biases, plus L3 weights, are limited to ±1.984375; FT and the output bias are unbounded. Disable it with `--optimizer-weight-clip 0` (JSON: `"optimizer_weight_clip": 0`). See [what this controls and its limitations](../advanced/tuning.md#weight-clipping-during-training).

For tuning and comparison experiments, see the [Advanced guide](../advanced/).

Previous: [2. Prepare training data](2-data.md)
