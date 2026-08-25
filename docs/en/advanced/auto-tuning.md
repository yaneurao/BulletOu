# Automatic tuning

<a href="../../ja/advanced/auto-tuning.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

This page explains how to tune SFNN factorizer alpha values and count-confidence values with `bulletou_tuner.py`.

Population search means that the runner trains several short candidate runs and uses the best candidate as the next starting point. Each generation creates `population` candidates with slightly different hyperparameters, trains every candidate for a fixed trial length, and selects the candidate that ranks best by the configured metric. It does not estimate a gradient and then apply a separate small update. The selected candidate's NN weights and hyperparameter values directly become the next generation's starting point.

## JSON files

The runner uses two JSON files.

| File | Purpose |
| --- | --- |
| `tuning-settings.json` | population search generations, population, trial length, tunable parameters, and current values |
| `bulletou-settings.json` | Normal `bulletou.exe` training options |

`tuning-settings.json` points to `bulletou-settings.json`. You normally pass only `--tuning-settings-file` to the runner.

```powershell
python .\bulletou_tuner.py --tuning-settings-file .\tuning-settings.json
```

To continue the same runner, add `--resume`.

```powershell
python .\bulletou_tuner.py --tuning-settings-file .\tuning-settings.json --resume
```

## `tuning-settings.json` example

```json
{
  "version": 1,
  "tuning": {
    "enabled": true,
    "generations": 100,
    "population": 100,
    "trial_sbs": 4,
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "use_worker": true,
    "seed": 1,
    "save_rate": 1,
    "validation_rate": 1,
    "quantized_validation_rate": 1
  },
  "run": {
    "exe": "C:/shogi/YaneuraOuWorks/BulletOu/target/release/examples/bulletou.exe",
    "bulletou_settings_file": "./bulletou-settings.json",
    "base_checkpoint": "C:/shogi/YaneuraOuWorks/BulletOu/checkpoints/.../0256",
    "output_folder": "D:/BulletOu-snapshots/20260820",
    "temp_folder": "C:/BulletOu-tuning-temp",
    "tag_prefix": "pair2-qloss"
  },
  "parameters": {
    "shared": { "current": 1.0, "tune": false, "step": 0.0, "min": 0.0, "max": 100.0 },
    "king_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "progress_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_hand_pair": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_progress_pair": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_progress_pair": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "residual_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_axis_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_hand_pair_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_progress_pair_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_progress_pair_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "lr": { "current": 0.000030, "tune": false, "step": 0.005, "min": 0.000001, "max": 0.001 },
    "lr_min": { "current": 0.000010, "tune": false, "step": 0.005, "min": 0.000001, "max": 0.001 }
  }
}
```

## `bulletou-settings.json` example

`bulletou-settings.json` contains the training options you would normally pass to `bulletou.exe`. JSON keys are CLI option names without `--`, written with underscores instead of hyphens. For example, `--lr-min` becomes `lr_min`.

```json
{
  "backend": "cuda-cpp",
  "teacher": "D:/sojoteam_datasets",
  "test_teacher": "C:/shogi/teacher/test/test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe",
  "test_positions": "all",
  "arch": "SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4",
  "sfnn_factorizer": "pair",
  "sfnn_bucket_counts": "D:/sojo_counts/SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin",
  "positions_per_superbatch": 40000000,
  "lr_schedule": "step",
  "optimizer": "ranger",
  "optimizer_weight_decay": 0.0,
  "batches_per_update": 1,
  "wrm_in_offset": 0,
  "wrm_target_offset": 0,
  "sfnn_dirty_bucket_update": true,
  "sfnn_saturation_penalty": 1e-7
}
```

During population search, the runner owns these candidate-specific values, so do not put them in `bulletou-settings.json`.

| Runner-controlled setting | Why |
| --- | --- |
| `initial_state`, `initial_dataloader_pos` | Each candidate starts from a runner-selected checkpoint |
| `output`, `output_folder`, `tag` | Each candidate needs a separate output directory |
| `superbatches`, `max_epochs` | The runner sets the candidate trial length |
| `save_rate`, `validation_rate`, `quantized_validation_rate` | The runner sets candidate evaluation cadence |
| `sfnn_factorizer_alpha` | Built from `parameters` for each candidate |
| `sfnn_*_count_confidence` | Built from `parameters` for each candidate |
| `lr`, `lr_min` | Optional. If these are written in `tuning-settings.json` `parameters`, the runner passes them as `--lr` / `--lr-min` and writes accepted values back to `current` |

## `tuning` fields

| Field | Meaning |
| --- | --- |
| `enabled` | `true` runs population search. `false` runs one normal `bulletou.exe` job using `parameters.current` values |
| `generations` | Number of generations. One candidate is accepted per generation |
| `population` | Number of candidates tested in one generation. The example tests 100 candidates |
| `trial_sbs` | Training length per candidate. The example uses 1 trial = 4 sb |
| `metric` | Metric used to rank candidates |
| `lower_is_better` | `true` if smaller metric values are better. Ignored for `borda_count` because lower rank sum is always better |
| `use_worker` | Use a long-lived `bulletou worker` process. The default is `true` when omitted |
| `seed` | Random seed for candidate generation |
| `save_rate` | Copy a public checkpoint to `accepted-checkpoints/` every N accepted generations |
| `validation_rate` | f32 validation cadence. With population search enabled, it applies to each candidate; with `enabled: false`, it applies to the ordinary training run. `0` measures only at the end of each trial. `-1` disables it |
| `quantized_validation_rate` | Quantized validation cadence. With population search enabled, it applies to each candidate; with `enabled: false`, it applies to the ordinary training run. `0` measures only at the end of each trial. `-1` disables it |

You cannot disable a validation that the selected `metric` needs. For example, `metric: "borda_count"` uses all f32/quantized accuracy/loss values, so both validation rates must be enabled. Use `0` when you want trial-end-only validation.

`trial_sbs` is the training length for one candidate. With `trial_sbs: 4`, every candidate trains for 4 sb, and the runner ranks candidates after the full `population` has finished.

Supported metrics:

| Value | Meaning | Suggested `lower_is_better` |
| --- | --- | --- |
| `quantized_value_loss` | Validation loss after quantization | `true` |
| `quantized_value_accuracy` | Validation accuracy after quantization | `false` |
| `test_value_loss` | Validation loss with f32 weights | `true` |
| `test_value_accuracy` | Validation accuracy with f32 weights | `false` |
| `borda_count` | Rank candidates by all four metrics, then choose the smallest rank sum | `true` |

For engine-strength-oriented tuning, `quantized_value_loss` is usually the first metric to try.
If qloss and qacc disagree too often, `borda_count` is a safer compromise:

1. rank candidates by higher `test_value_accuracy`;
2. rank candidates by lower `test_value_loss`;
3. rank candidates by higher `quantized_value_accuracy`;
4. rank candidates by lower `quantized_value_loss`;
5. add the four ranks and keep the candidate with the smallest sum.

Tied values receive the average rank of the tied range. For example, if two candidates tie for 2nd and 3rd place, both receive rank 2.5 for that metric.

When `borda_count` is used with worker mode, the runner stores candidate state caches on disk under `temp_folder`. After all candidates are ranked, the survivor is restored from its disk cache into the worker. This avoids holding `population` copies of the large optimizer state in host RAM and does not require retraining the survivor. The tradeoff is temporary `state.bin` write/read I/O per candidate.

If a candidate is worse than an already evaluated candidate in all four metrics, it cannot become the best Borda candidate. In that case, the worker skips cache writing immediately after the trial. If a new candidate is better than a previously cached candidate in all four metrics, the runner drops the older cache immediately.

## `run` fields

| Field | Meaning |
| --- | --- |
| `exe` | Path to `bulletou.exe` |
| `bulletou_settings_file` | JSON file containing normal training options |
| `base_checkpoint` | Initial checkpoint directory. It must contain `state.bin` and `dataloader_pos.txt` |
| `output_folder` | Parent directory for the runner root |
| `temp_folder` | Temporary candidate checkpoint directory. A fast SSD is recommended |
| `tag_prefix` | Name used in the runner root |

The runner root is `output_folder/tuning-<tag_prefix>`.

`run.base_checkpoint` is only the initial starting point. After each generation, the accepted checkpoint is recorded in `runner-state.json` as `current_checkpoint`. With `--resume`, the runner resumes from `current_checkpoint`, not from `run.base_checkpoint`. The runner does not rewrite `run.base_checkpoint` in `tuning-settings.json`.

## Common `parameters` fields

When `shared` is fixed, the main tuning set has 13 parameters: three axis alpha values, three pair alpha values, three axis count-confidence values, three pair count-confidence values, and one residual count-confidence value. If you want the runner to own learning-rate values too, you can optionally add `lr` / `lr_min`.

Each parameter is written like this:

```json
"king_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 }
```

| Field | Meaning |
| --- | --- |
| `current` | Current value. Candidates are sampled around it |
| `tune` | If `true`, population search may change it. If `false`, it is fixed |
| `step` | Multiplicative sampling radius. Candidate values are drawn as `current * exp(random(-step, step))` |
| `min` | Lower bound |
| `max` | Upper bound |

`step` is not an additive width. `step = 0.02` samples roughly within ?2% of `current`; `step = 0.10` samples roughly from 0.90x to 1.11x. To widen the search, edit the `step` value of the parameters you want to move. Parameters with `tune = true` must have `current > 0`.

When a candidate is accepted, its parameter values are written back to `current`. You do not need to paste long values such as `king_axis=...` by hand when resuming.

`lr` and `lr_min` can also be written in `parameters` when you want the runner to own the current learning-rate values:

```json
"lr": { "current": 0.000030, "tune": false, "step": 0.005, "min": 0.000001, "max": 0.001 },
"lr_min": { "current": 0.000010, "tune": false, "step": 0.005, "min": 0.000001, "max": 0.001 }
```

If you put them in `parameters`, the runner appends `--lr` and `--lr-min` to every `bulletou.exe` run, so those values override any learning rates in `bulletou-settings.json`. If they are omitted from `parameters`, `bulletou-settings.json` owns the learning rates.

When tuning both learning-rate values, choose `current`/`min`/`max` so that every sampled candidate satisfies `lr_min <= lr`. The runner reports an error instead of silently clamping this relationship.

## Alpha parameters

Alpha values are multipliers for factorizer terms.

| Key | Meaning |
| --- | --- |
| `shared` | Strength of the term shared by all buckets |
| `king_axis` | Strength of the king-bucket axis term |
| `hand_axis` | Strength of the hand-bucket axis term |
| `progress_axis` | Strength of the progress-bucket axis term |
| `king_hand_pair` | Strength of the king-hand pair term |
| `king_progress_pair` | Strength of the king-progress pair term |
| `hand_progress_pair` | Strength of the hand-progress pair term |

Changing `shared` moves the global base term, so keeping `shared = 1.0` fixed is often easier to interpret.

In `bulletou.exe`, `--sfnn-factorizer-alpha pair=...` is a shortcut that sets all three pair alpha values to the same value. In population search settings, tune `king_hand_pair`, `king_progress_pair`, and `hand_progress_pair` separately.

## Count-confidence parameters

Count confidence uses a `count.bin` file created by `bucket-count`. It weakens terms that do not have enough observed positions.

| Key | BulletOu option | Meaning |
| --- | --- | --- |
| `residual_count` | `--sfnn-residual-count-gate-confidence` | Count gate confidence for bucket-specific residuals |
| `king_axis_count` | `--sfnn-king-axis-count-confidence` | King-axis override |
| `hand_axis_count` | `--sfnn-hand-axis-count-confidence` | Hand-axis override |
| `progress_axis_count` | `--sfnn-progress-axis-count-confidence` | Progress-axis override |
| `king_hand_pair_count` | `--sfnn-king-hand-pair-count-confidence` | King-hand pair override |
| `king_progress_pair_count` | `--sfnn-king-progress-pair-count-confidence` | King-progress pair override |
| `hand_progress_pair_count` | `--sfnn-hand-progress-pair-count-confidence` | Hand-progress pair override |

`0.0` disables that confidence. Larger values require more observations before the corresponding term is trusted strongly.

Plain `bulletou.exe` still has shared options for all axis terms or all pair terms. population search `parameters` intentionally use only the explicit fields listed above, so it is always clear which component is being tuned.

## Use current values without population search

If you want to use the tuned `parameters.current` values without running population search, set `tuning.enabled` to `false`.

```json
"tuning": {
  "enabled": false
}
```

Then launch the runner once:

```powershell
python .\bulletou_tuner.py --tuning-settings-file .\tuning-settings.json
```

With this path, you do not manually copy the 13 `parameters.current` values into `bulletou-settings.json`. The runner reads `parameters.current`, converts them to `--sfnn-factorizer-alpha` and count-confidence options, and passes them to `bulletou.exe`.

In this mode, the runner fills `superbatches` from `trial_sbs` and `max_epochs` from `generations`. It uses `validation_rate` and `quantized_validation_rate` from the `tuning` section of `tuning-settings.json`. If `lr` / `lr_min` are written in `parameters`, the runner passes those current values; otherwise it uses the learning rates from `bulletou-settings.json`. `save_rate` controls how often public checkpoints are copied under `accepted-checkpoints/`.

With `enabled: false`, the runner does not create population search candidates, worker caches, or snapshots. It launches `bulletou.exe` once and only converts `parameters.current` into CLI arguments. stdout is written to `output_folder/tuning-<tag_prefix>/logs/bulletou-settings-run.stdout.log`.

The ordinary training output is written under `output_folder/tuning-<tag_prefix>/bulletou-run/`. The runner imports that `summary-learn.log` into `output_folder/tuning-<tag_prefix>/summary-learn.log` and `accepted-summary-learn.log`. Saved checkpoints are copied to `accepted-checkpoints/sbXXXXXXXX/`, and the latest checkpoint is copied to `current/`.

After that, you can set `enabled` back to `true` and run with `--resume`. population search resumes from the updated `current/`. The runner also writes the ordinary-training progress into `runner-state.json`, so `accepted_sbs`, `generation`, and public checkpoint numbering do not roll back.

## `bulletou.exe --settings-file`

`bulletou.exe` can also read a settings JSON directly.

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

Explicit CLI arguments override values from the settings file.

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --lr 0.000050
```

## Output layout

The runner root is `output_folder/tuning-<tag_prefix>`.

| Path | Purpose |
| --- | --- |
| `current/` | Latest accepted checkpoint. `--resume` continues from here |
| `accepted-checkpoints/sbXXXXXXXX/` | Public checkpoints saved every `save_rate` accepted generations |
| `summary-learn.log` | Results for all candidates |
| `accepted-summary-learn.log` | Results for accepted survivors only |
| `parameters-history.jsonl` | Accepted parameter history |
| `runner-state.json` | Resume state |
| `logs/` | Per-candidate stdout logs |
| `temp/` | Temporary candidate checkpoints, unless `temp_folder` is set |

population search writes metric columns in `summary-learn.log` and `accepted-summary-learn.log` in the same order as normal BulletOu summaries: `test_value_accuracy`, `test_value_loss`, `quantized_value_accuracy`, `quantized_value_loss`.

The runner copies the current `tuning-settings.json` and `bulletou-settings.json` into `current/` and `accepted-checkpoints/sbXXXXXXXX/`. This makes it possible to inspect the exact settings used for a checkpoint later.

If you want to stop the runner manually, the safe point is immediately after `[SAFE TO STOP]` appears. `[GEN RANK]` only means candidate evaluation and ranking for the generation finished; it does not mean a public checkpoint has been fully saved.

During population search, the runner uses one long-lived `bulletou worker` process by default. This avoids rebuilding the CUDA context, validation cache, qvalid cache, and worker warmup for every candidate.

If `use_worker` is `false`, candidates are run as short `bulletou.exe` child jobs. In that mode, `[epoch] start epoch 1/1` belongs to the child job, not to the whole population search run. The runner prefixes streamed child output with labels such as `[G0002 S0032 C001]`, meaning generation 2, 32sb trial, candidate 1.

When normal training uses `bulletou.exe --settings-file`, each BulletOu checkpoint gets a copy of `bulletou-settings.json`.
