# Fixed-length trial parameter tuning

`tuning_parameters.py` runs many short fixed-parameter BulletOu trials to search over `lr`, `lr_min`, factorizer, and count-confidence settings.

It does not depend on external Python packages. It is a small BulletOu-specific TPE-style sampler: random trials first, then it compares the distribution of good trials against the distribution of bad trials and samples promising regions more often.

## What this runner does

- Each trial starts from scratch or from the same fixed checkpoint.
- The length of one trial is `tuning.trial_sbs`. It can be a number or an array.
- Parameters stay fixed during a trial.
- For every generation, the runner evaluates `population` trials and records the metric.
- `recommended-parameters.json` contains inferred parameters from the top trials.

For short trials, the single best observed trial can be noisy. Check both `best_observed` and `recommended` before using the values in a longer run.

## What `log` means

Parameter entries may include `log: true`.

`log: true` means logarithmic sampling. This is useful for learning rates, where values such as `0.000001`, `0.00001`, `0.0001`, and `0.001` differ by ratio rather than by absolute distance.

`log: true` cannot be used with `min=0`, because `log(0)` is undefined.

If you want to allow `0` for factorizer alpha or count confidence, omit `log` or set `log: false`. When `min` is non-positive and `log` is omitted, the runner defaults to linear sampling.

## Generations and the TPE sampler

`tuning_parameters.py` creates candidates generation by generation. Trial results from the current generation are not used to create more candidates in the same generation. They are used starting from the next generation.

```json
"tuning": {
  "generations": 3,
  "population": [100, 50],
  "trial_sbs": [4, 8],
  "sampler": "tpe"
}
```

This means:

- generation 1: 100 trials, 4 sb per trial
- generation 2 and later: 50 trials, 8 sb per trial

If `generations` is omitted, the runner infers it from the `population` / `trial_sbs` array length. If a `population` or `trial_sbs` array is shorter than `generations`, the last value is reused.

The TPE-style sampler uses results from previous generations. It sorts completed trials by the configured metric, treats the top fraction as good trials and the rest as bad trials, builds per-parameter distributions, and prefers values that are likely under the good distribution and unlikely under the bad distribution.

## Sampler fields

These are sampler settings, not NNUE training parameters. They control how the runner chooses the next trial parameters.

| Field | Meaning | Default |
| --- | --- | --- |
| `sampler` | `"tpe"` or `"random"`. Usually use `"tpe"`. | `"tpe"` |
| `tpe_startup_trials` | Number of completed trials required before TPE starts. Until then, the runner samples from the whole search range. | `16` |
| `tpe_good_fraction` | Fraction of completed trials treated as good candidates. `0.25` means the top 25% are used. | `0.25` |
| `tpe_bandwidth` | Lower bound for the TPE KDE width. Larger values spread candidates more broadly; smaller values concentrate them closer to observed good trials. | `0.15` |

## Settings example

```json
{
  "version": 1,
  "tuning": {
    "generations": 3,
    "population": [100, 50],
    "trial_sbs": [4, 8],
    "sampler": "tpe",
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "seed": 20260825,
    "tpe_startup_trials": 16,
    "tpe_good_fraction": 0.25,
    "tpe_bandwidth": 0.15,
    "validation_rate": 0,
    "quantized_validation_rate": 0,
    "keep_all_trials": false
  },
  "run": {
    "exe": "C:/shogi/YaneuraOuWorks/BulletOu/target/release/examples/bulletou.exe",
    "bulletou_settings_file": "./bulletou-settings.json",
    "base_checkpoint": null,
    "output_folder": "D:/BulletOu-snapshots/20260825",
    "temp_folder": "D:/BulletOu-snapshots/20260825",
    "tag_prefix": "tuning-scratch-4sb"
  },
  "parameters": {
    "lr": { "current": 0.0003, "tune": true, "min": 0.000001, "max": 0.001, "log": true },
    "lr_min": { "current": 0.0001, "tune": true, "min": 0.000001, "max": 0.001, "log": true },

    "shared": 1.0,

    "king_axis": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_axis": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "progress_axis": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },

    "king_hand_pair": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_progress_pair": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_progress_pair": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },

    "residual_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_axis_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_hand_pair_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_progress_pair_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_progress_pair_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 }
  }
}
```

When both `lr` and `lr_min` are tuned, the runner samples `lr_min` with the sampled `lr` as its upper bound, so generated trials satisfy `lr_min <= lr`.

## Keeping teacher data in RAM

`bulletou.exe worker` can keep PSV-compatible teacher records (`.psv` / `.bin`) in the worker process RAM.
When many trials run inside the same worker process, this avoids rereading the same teacher window from a USB HDD or other slow storage.

Add this to `bulletou-settings.json`:

```json
{
  "teacher_memory_cache_sbs": 4
}
```

The value means "keep this many superbatches in RAM". If one superbatch is `610 * 65536` positions, four superbatches are about 160M records, or roughly 6 GiB of RAM.

Notes:

- The cache lives only inside the worker process. It disappears when the worker exits.
- It currently supports only `.psv` / `.bin` teachers.
- If a trial runs for 4sb, set `teacher_memory_cache_sbs` to at least 4. Smaller values fail with an error.
- `tuning_parameters.py` uses worker mode by default. If you set `tuning.use_worker: false`, it starts a fresh `bulletou.exe` process for every trial, so this cache cannot help.
- When the cache is enabled, startup prints `[CACHE] teacher_memory_cache_sbs=...`, and the worker log prints `worker teacher memory cache = loading/ready`.

## Run

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json
```

Resume:

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json `
  --resume
```

## Outputs

The runner root is:

```text
<output_folder>/tuning-<tag_prefix>/
```

| path | Meaning |
| --- | --- |
| `summary-learn.log` | Every trial result |
| `best-checkpoint/` | Checkpoint from the best observed trial |
| `recommended-parameters.json` | Inferred parameters from top trials |
| `runner-state.json` | Resume state |
| `logs/` | stdout for each trial |

## Checkpoint retention

`keep_all_trials` controls how many trial checkpoints are kept.

```json
"metric": "quantized_value_loss",
"lower_is_better": true,
"keep_all_trials": false
```

With this setting, lower `quantized_value_loss` is better. The runner keeps only the current best trial under `best-checkpoint/`. Trial output directories and checkpoints that do not become the best are deleted after the trial finishes.

Even when a non-best trial checkpoint is deleted, `summary-learn.log` and `logs/trialXXXX.stdout.log` remain, so you can still inspect the metric values and stdout later.

To keep every trial checkpoint, use one of these:

- `keep_all_trials: true`
- `--keep-temp` at runtime

For normal tuning, `keep_all_trials: false` is safer because it avoids rapid storage growth.

`recommended-parameters.json` contains `best_observed` and `recommended`. `best_observed` is the single best trial. `recommended` is a rank-weighted estimate from top trials. Parameters with `log: true` are averaged in log space.
