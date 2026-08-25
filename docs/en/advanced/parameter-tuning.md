# Fixed-length trial parameter tuning

`tuning_parameters.py` runs many short fixed-parameter BulletOu trials to search over `lr`, `lr_min`, factorizer, and count-confidence settings.

It does not depend on external Python packages. It is a small BulletOu-specific sampler: random trials first, then samples near the best completed trials.

## What this runner does

- Each trial starts from scratch or from the same fixed checkpoint.
- The length of one trial is `tuning.trial_sbs`.
- Parameters stay fixed during a trial.
- The runner evaluates `population` trials and records the metric.
- `recommended-parameters.json` contains inferred parameters from the top trials.

For short trials, the single best observed trial can be noisy. Check both `best_observed` and `recommended` before using the values in a longer run.

## What `log` means

Parameter entries may include `log: true`.

`log: true` means logarithmic sampling. This is useful for learning rates, where values such as `0.000001`, `0.00001`, `0.0001`, and `0.001` differ by ratio rather than by absolute distance.

`log: true` cannot be used with `min=0`, because `log(0)` is undefined.

If you want to allow `0` for factorizer alpha or count confidence, omit `log` or set `log: false`. When `min` is non-positive and `log` is omitted, the runner defaults to linear sampling.

## `startup_trials` / `elite_fraction` / `elite_sigma`

These are sampler settings, not NNUE training parameters. They control how the runner chooses the next trial parameters.

| Field | Meaning | Default |
| --- | --- | --- |
| `startup_trials` | Number of fully random trials at the beginning. Until this many trials complete, the runner samples from the whole search range instead of sampling near good trials. | `16` |
| `elite_fraction` | Fraction of completed trials treated as good candidates after startup. `0.25` means the top 25% are used. | `0.25` |
| `elite_sigma` | Spread used when sampling near a good candidate. For linear parameters this is `(max - min) * elite_sigma`; for `log: true` parameters it is the same fraction in log space. | `0.15` |

The flow is:

```text
Run startup_trials random trials.
Then sample near the top elite_fraction trials with spread elite_sigma.
```

## Settings example

```json
{
  "version": 1,
  "tuning": {
    "population": 100,
    "trial_sbs": 4,
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "seed": 20260825,
    "startup_trials": 16,
    "elite_fraction": 0.25,
    "elite_sigma": 0.15,
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
