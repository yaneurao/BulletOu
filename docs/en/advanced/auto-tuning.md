# Automatic ES tuning

<a href="../../ja/advanced/auto-tuning.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

Factorizer alpha values and count-confidence values have many useful combinations. If you already have a good checkpoint, use `es_local_runner.py` to run a beam-search-style ES (evolution strategy) around it.

The runner treats `parameters.json` as the current hyperparameter state. Each generation creates multiple candidates, trains each candidate for the configured number of superbatches, prunes weaker candidates at beam stages, and then promotes the final survivor. The survivor's NN weights and hyperparameters become the next current state.

The important rules are:

- The survivor's hyperparameter values become the next generation's values directly.
- There is no separate partial parameter update after selection.
- Each candidate is sampled independently.
- `parameters.json` is written back after every accepted generation. To manually edit values, stop the runner, edit the file, then resume.

## `parameters.json`

`parameters.json` contains the ES runner settings and current values for tuned parameters.

```json
{
  "version": 1,
  "es": {
    "enabled": true,
    "generations": 100,
    "population": 16,
    "beam": [
      { "after_sbs": 8, "keep": 8 },
      { "after_sbs": 16, "keep": 4 },
      { "after_sbs": 24, "keep": 2 },
      { "after_sbs": 32, "keep": 1 }
    ],
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "seed": 1,
    "save_rate": 1,
    "candidate_validation_rate": 1,
    "candidate_quantized_validation_rate": 1
  },
  "parameters": {
    "shared": { "current": 1.0, "tune": false, "step": 0.0, "min": 0.0, "max": 10.0 },
    "king_axis": { "current": 1.0, "tune": true, "step": 0.03, "min": 0.0, "max": 10.0 },
    "hand_axis": { "current": 1.0, "tune": true, "step": 0.03, "min": 0.0, "max": 10.0 },
    "progress_axis": { "current": 1.0, "tune": true, "step": 0.03, "min": 0.0, "max": 10.0 },
    "pair": { "current": 0.3, "tune": true, "step": 0.02, "min": 0.0, "max": 10.0 },
    "residual_count": { "current": 1.0, "tune": true, "step": 0.25, "min": 0.0, "max": 20.0 },
    "king_axis_count": { "current": 4.0, "tune": true, "step": 0.5, "min": 0.0, "max": 100.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "step": 0.5, "min": 0.0, "max": 100.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "step": 0.5, "min": 0.0, "max": 100.0 },
    "king_hand_pair_count": { "current": 10.0, "tune": true, "step": 1.0, "min": 0.0, "max": 200.0 },
    "king_progress_pair_count": { "current": 10.0, "tune": true, "step": 1.0, "min": 0.0, "max": 200.0 },
    "hand_progress_pair_count": { "current": 10.0, "tune": true, "step": 1.0, "min": 0.0, "max": 200.0 }
  }
}
```

`es.enabled` tells BulletOu how to use this JSON file.

| Value | Meaning |
| --- | --- |
| `true` | Run ES with `es_local_runner.py` |
| `false` | Use only `parameters.current` values in normal `bulletou.exe` training |

`step` is the random sampling width. If `pair.current = 0.3` and `pair.step = 0.02`, candidates sample `pair` between `0.28` and `0.32`. Parameters with `tune: false` stay fixed.

`beam` means when to prune candidates. The example starts with 16 candidates, keeps 8 after 8 sb, 4 after 16 sb, 2 after 24 sb, and 1 after 32 sb. The final survivor is accepted.

## Run example

```powershell
$base = "C:\shogi\YaneuraOuWorks\BulletOu\checkpoints\...\0256"

python .\es_local_runner.py `
  --exe C:\shogi\YaneuraOuWorks\BulletOu\target\release\examples\bulletou.exe `
  --parameters-file .\parameters.json `
  --base-checkpoint $base `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --output-folder D:\BulletOu-snapshots\20260820 `
  --temp-folder C:\BulletOu-es-temp `
  --tag-prefix pair2-qloss `
  --factorizer pair `
  --positions-per-superbatch 40000000 `
  -- `
  --lr 0.000030 `
  --lr-min 0.000010 `
  --wrm-in-offset 0 `
  --wrm-target-offset 0 `
  --lr-schedule step `
  --optimizer ranger `
  --optimizer-weight-decay 0.0 `
  --batches-per-update 1 `
  --sfnn-dirty-bucket-update `
  --sfnn-saturation-penalty 1e-7
```

The standalone `--` line is the delimiter. Everything after it is passed to `bulletou.exe`, not to the runner. Put common candidate options such as `--lr` and `--optimizer` there.

Do not put `--resume`, `--parameters-file`, `--superbatches`, `--max-epochs`, `--save-rate`, `--validation-rate`, `--quantized-validation-rate`, `--tag`, `--output-folder`, `--initial-state`, `--initial-dataloader-pos`, `--sfnn-factorizer-alpha`, or count-confidence options after the delimiter. The runner owns those options for each candidate.

## Use `parameters.json` without ES

If you want to train with the tuned values fixed, set `es.enabled` to `false` in `parameters.json`. Then pass the same file to `bulletou.exe`.

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --sfnn-factorizer pair `
  --parameters-file .\parameters.json `
  --sfnn-bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --positions-per-superbatch 40000000 `
  --superbatches 32 `
  --max-epochs 1 `
  --lr 0.000030 `
  --lr-min 0.000010
```

In this mode, `bulletou.exe` reads only `parameters.*.current`. Fields such as `step`, `tune`, and `beam` are runner settings and are ignored by normal training.

Do not combine `--parameters-file` with `--sfnn-factorizer-alpha` or count-confidence options. There should be only one source for those values.

## Output layout

The runner root is `--output-folder\es-<tag-prefix>`.

| Path | Purpose |
| --- | --- |
| `current/` | Latest accepted checkpoint. Runner `--resume` continues from here |
| `accepted-checkpoints/sbXXXXXXXX/` | Public checkpoint saved every `save_rate` accepted generations |
| `summary-learn.log` | All candidate/stage results |
| `accepted-summary-learn.log` | Final survivor results only |
| `parameters-history.jsonl` | Accepted parameter history |
| `runner-state.json` | Resume state |

Use `--temp-folder` to place temporary candidate checkpoints on a fast SSD such as `C:\BulletOu-es-temp`. Pruned candidates are deleted automatically. Use `--keep-temp` only when you want to inspect those directories.

## Resume

To resume, use the same `--output-folder` and `--tag-prefix`, then add `--resume`. The current hyperparameters are read from `parameters.json`, so you do not need to paste checkpoint-time values into the command line.

```powershell
python .\es_local_runner.py `
  --resume `
  --exe C:\shogi\YaneuraOuWorks\BulletOu\target\release\examples\bulletou.exe `
  --parameters-file .\parameters.json `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --output-folder D:\BulletOu-snapshots\20260820 `
  --temp-folder C:\BulletOu-es-temp `
  --tag-prefix pair2-qloss `
  --factorizer pair `
  --positions-per-superbatch 40000000 `
  -- --lr 0.000030 --lr-min 0.000010 --wrm-in-offset 0 --wrm-target-offset 0 --lr-schedule step --optimizer ranger --optimizer-weight-decay 0.0 --batches-per-update 1 --sfnn-dirty-bucket-update --sfnn-saturation-penalty 1e-7
```

The console prints colored milestone lines: `[GEN START]`, `[CAND 001 START]`, `[CAND 001 END]`, `[BEAM]`, `[ACCEPT]`, and `[SAVE]`. Stop after `[SAVE]` or `[SAFE TO STOP]` if you want a public checkpoint. The `current/` checkpoint is updated after every accept, so runner `--resume` can also continue from accepted states that have not been copied to `accepted-checkpoints/`.
