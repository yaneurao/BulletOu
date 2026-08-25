# Optuna-style fixed-length trial search

`optuna_style_runner.py` runs many short fixed-parameter BulletOu trials to search over `lr`, `lr_min`, factorizer, and count-confidence settings.

It does not depend on the Optuna Python package. It is a small BulletOu-specific sampler: random trials first, then samples near the best completed trials.

## Difference from the ES runner

The ES runner continues from the accepted checkpoint. If you change factorizer or count-confidence values in the middle of training, the result mixes two effects: the parameter value itself, and the transient damage from changing the training dynamics.

The Optuna-style runner starts every trial from scratch or from the same fixed base checkpoint. Parameters stay fixed during the trial.

Use it when you want to:

- compare fixed parameters from early training,
- include `lr` and `lr_min` in the search,
- run many short trials such as 16sb each,
- get inferred recommended parameters, not just the single best observed trial.

## Settings example

```json
{
  "version": 1,
  "study": {
    "trials": 64,
    "trial_sbs": 16,
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
    "tag_prefix": "optuna-scratch-16sb"
  },
  "parameters": {
    "lr": { "tune": true, "low": 0.00003, "high": 0.001, "log": true },
    "lr_min_ratio": { "tune": true, "low": 0.03, "high": 1.0, "log": true },

    "shared": 1.0,
    "king_axis": 1.0,
    "hand_axis": 1.0,
    "progress_axis": 1.0,
    "king_hand_pair": 1.0,
    "king_progress_pair": 1.0,
    "hand_progress_pair": 1.0,

    "residual_count": { "tune": true, "low": 0.3, "high": 3.0, "log": true },
    "king_axis_count": { "tune": true, "low": 0.3, "high": 3.0, "log": true },
    "hand_axis_count": { "tune": true, "low": 0.3, "high": 3.0, "log": true },
    "progress_axis_count": { "tune": true, "low": 0.3, "high": 3.0, "log": true },
    "king_hand_pair_count": { "tune": true, "low": 0.3, "high": 3.0, "log": true },
    "king_progress_pair_count": { "tune": true, "low": 0.3, "high": 3.0, "log": true },
    "hand_progress_pair_count": { "tune": true, "low": 0.3, "high": 3.0, "log": true }
  }
}
```

`lr_min_ratio` is converted to:

```text
lr_min = lr * lr_min_ratio
```

This avoids invalid samples such as `lr_min > lr`.

## Run

```powershell
python .\optuna_style_runner.py `
  --settings-file D:\BulletOu-snapshots\settings\optuna-style-settings.json
```

Resume:

```powershell
python .\optuna_style_runner.py `
  --settings-file D:\BulletOu-snapshots\settings\optuna-style-settings.json `
  --resume
```

## Outputs

The runner root is:

```text
<output_folder>/optuna-<tag_prefix>/
```

| path | Meaning |
| --- | --- |
| `summary-learn.log` | Every trial result |
| `best-checkpoint/` | Checkpoint from the best observed trial |
| `recommended-parameters.json` | Inferred parameters from top trials |
| `runner-state.json` | Resume state |
| `logs/` | stdout for each trial |

## Best observed vs recommended

`best_observed` is the single trial that produced the best metric.

`recommended` is an inferred parameter set from the top trials. It uses a rank-weighted mean. Log-sampled parameters are averaged in log space, so values such as `lr` and count-confidence use a geometric-mean-like estimate.

For short trials, the single best trial can be noisy. If you want fixed parameters for a longer confirmation run, `recommended.parameters` is usually the more stable value to inspect.

