# Fixed-length trial parameter tuning

`tuning_parameters.py` runs many short fixed-parameter BulletOu trials to search over `lr`, `lr_min`, factorizer, and count-confidence settings while training moves forward generation by generation.

It does not depend on external Python packages. It is a small BulletOu-specific TPE-style sampler: random trials first, then it compares the distribution of good trials against the distribution of bad trials and samples promising regions more often.

## What this runner does

- Trials in the same generation start from the same checkpoint.
- The length of one trial is `tuning.trial_sbs`. It can be a number or an array.
- Parameters stay fixed during a trial.
- For every generation, the runner evaluates `population` trials and records the metric.
- At the end of a generation, the runner performs one commit run with the selected parameters and saves that checkpoint as `current-checkpoint/`.
- Depending on `tuning.save_rate`, committed generation checkpoints are also saved to stable paths such as `generation-checkpoints/gen0001/`.
- `recommended-parameters.json` contains inferred parameters from the top trials.

For short trials, the single best observed trial can be noisy. Check both `best_observed` and `recommended` before using the values in a longer run.

## What `log` means

Parameter entries may include `log: true`.

`log: true` means logarithmic sampling. This is useful for learning rates, where values such as `0.000001`, `0.00001`, `0.0001`, and `0.001` differ by ratio rather than by absolute distance.

`log: true` cannot be used with `min=0`, because `log(0)` is undefined.

If you want to allow `0` for factorizer alpha or count confidence, omit `log` or set `log: false`. When `min` is non-positive and `log` is omitted, the runner defaults to linear sampling.

## Generations and the TPE sampler

`tuning_parameters.py` creates candidates generation by generation. Completed trials from the current generation are used immediately when creating later candidates in that same generation. For example, with `tpe_startup_trials: 16`, the first 16 trials explore broadly, and trial 17 starts using current-generation observations for TPE.

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

Set `schedule_repeat: true` to repeat the arrays cyclically instead of reusing the last value. For example, `population: [100, 0]` and `trial_sbs: [4, 128]` means odd generations do short search trials and even generations do long stabilization training.

The TPE-style sampler uses trials that have already completed in the same generation. It sorts completed trials by the configured metric, treats the top fraction as good trials and the rest as bad trials, builds per-parameter distributions, and prefers values that are likely under the good distribution and unlikely under the bad distribution.

At the start of each generation there are not enough current-generation observations yet. In generation 1, the runner samples from the full configured range until `tpe_startup_trials` trials are complete. In generation 2 and later, a checkpoint already exists, so before enough observations exist the runner samples with Gaussian noise around the currently accepted parameter values.

For example, with `tpe_startup_trials: 16`, generation 1 trials 1 through 16 are random, and trials 17 through 100 use TPE based on completed generation 1 trials. In a later search generation that starts from a checkpoint, the first 16 trials sample around the current values, and trial 17 starts using TPE based on that generation's completed trials.

The runner does not mix all generations for TPE, because each generation starts from a different checkpoint. Mixing metrics from different training stages would make older generations unfairly worse and distort the sampler.

`max_parameter_change_ratio` limits how far a candidate may move away from the currently accepted value. For example, `2.0` samples a parameter with current value `1.0` from the range `0.5` through `2.0`. The runner applies this limit before sampling rather than clipping sampled values afterward, so values do not artificially pile up at exactly `0.5` or `2.0`. Generation 1 from scratch has no accepted checkpoint yet, so it still explores the full `min` to `max` range.

If the current value is `0`, the ratio limit keeps it at `0`. This is intentional: `0` means the component is disabled. For factorizer alpha and count-confidence parameters, prefer `min: 0.1` unless you explicitly want to test disabled components.

## Search generations and commit-only generations

`population` is the number of candidate trials in that generation. A generation with `population=0` does not create candidate trials. It keeps the currently accepted parameters, trains for `trial_sbs`, and updates `current-checkpoint/`.

Use this when you want to find promising parameters with short trials, train longer with those parameters, and then search again from the improved checkpoint.

```json
"tuning": {
  "generations": 5,
  "schedule_repeat": true,
  "population": [100, 0],
  "trial_sbs": [4, 128],
  "tpe_startup_trials": 16,
  "sampler": "tpe"
}
```

This runs:

- generation 1: train and compare 100 candidates for 4 sb each
- generation 2: keep the accepted generation-1 parameters and train for 128 sb
- generation 3: search 100 candidates for 4 sb from the generation-2 checkpoint
- generation 4: keep the accepted generation-3 parameters and train for 128 sb
- generation 5: run another short search generation

Commit-only generations do not use TPE, so `tpe_startup_trials` is ignored for those generations.

## Sampler fields

These are sampler settings, not NNUE training parameters. They control how the runner chooses the next trial parameters.

| Field | Meaning | Default |
| --- | --- | --- |
| `schedule_repeat` | If `true`, array-valued `population` / `trial_sbs` / `tpe_startup_trials` are repeated by generation. If `false`, the last array value is reused. | `false` |
| `sampler` | `"tpe"` or `"random"`. Usually use `"tpe"`. | `"tpe"` |
| `tpe_startup_trials` | Number of completed trials from the same generation required for TPE density estimation. This can be a number or an array, interpreted per generation like `population` and `trial_sbs`. In generation 1, trials are sampled from the full range until enough observations exist. In generation 2 and later, candidates are sampled around the currently accepted parameter values until enough current-generation observations exist. | `16` |
| `tpe_good_fraction` | Fraction of completed trials treated as good candidates. `0.25` means the top 25% are used. | `0.25` |
| `tpe_bandwidth` | Lower bound for the TPE KDE width. Larger values spread candidates more broadly; smaller values concentrate them closer to observed good trials. | `0.15` |
| `max_parameter_change_ratio` | Limits candidate values to a ratio around the currently accepted value. `2.0` means the sampling range is `current/2` to `current*2`. `null` or omission disables this limit. | none |
| `commit_source` | Parameters used for the generation-end commit run. `"best"` uses the best measured trial; `"recommended"` uses the inferred value from the top trials. | `"best"` |
| `save_rate` | How often to save committed generation checkpoints under `generation-checkpoints/genXXXX/`. `1` saves every generation; `0` disables generation checkpoint retention. | `1` |

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
    "tpe_startup_trials": [16, 8],
    "tpe_good_fraction": 0.25,
    "tpe_bandwidth": 0.15,
    "max_parameter_change_ratio": 2.0,
    "commit_source": "best",
    "save_rate": 1,
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

    "king_axis": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_axis": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "progress_axis": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },

    "king_hand_pair": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_progress_pair": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_progress_pair": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },

    "residual_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_axis_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_hand_pair_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_progress_pair_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_progress_pair_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 }
  }
}
```

## Choosing `metric`

`metric` is the criterion used to compare candidate trials. Common values are:

| metric | Meaning | `lower_is_better` |
| --- | --- | --- |
| `quantized_value_loss` | Choose the candidate with lower quantized loss | `true` |
| `quantized_value_accuracy` | Choose the candidate with higher quantized accuracy | `false` |
| `test_value_loss` | Choose the candidate with lower fp32 validation loss | `true` |
| `test_value_accuracy` | Choose the candidate with higher fp32 validation accuracy | `false` |
| `borda_count` | Rank candidates by all four metrics and choose the smallest rank sum | `true` |

`quantized_value_loss` is a useful first choice because it directly checks the quantized network. If qloss disagrees with qacc or with the fp32 metrics too often, `borda_count` is the safer compromise.

`borda_count` adds these four ranks:

1. higher `test_value_accuracy`;
2. lower `test_value_loss`;
3. higher `quantized_value_accuracy`;
4. lower `quantized_value_loss`.

If two candidates have the same rank sum, the runner prefers the one with lower `quantized_value_loss`.

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
- If a trial runs for 4sb, set `teacher_memory_cache_sbs` to at least 4 to use the cache. If the cache is too small, BulletOu prints a warning and falls back to normal streaming for that trial.
- A generation with `population: 0` is a single longer commit-only training run, not candidate evaluation, so the runner disables teacher memory cache automatically.
- `tuning_parameters.py` uses worker mode by default. If you set `tuning.use_worker: false`, it starts a fresh `bulletou.exe` process for every trial, so this cache cannot help.
- When the cache is enabled, startup prints `[CACHE] teacher_memory_cache_sbs=...`, and the worker log prints `worker teacher memory cache = loading/ready`.

## Lightweight factorizer rebase

In worker mode, when a trial or commit run changes factorizer alpha or axis/pair count-confidence parameters from the previous state, BulletOu lightly rebases axis/pair factorizer tensors in place.

When switching from `alpha_old * K_old` to `alpha_new * K_new`, BulletOu scales the stored factorizer tensor so that the effective contribution at the start of the run is approximately preserved:

```text
scale = (alpha_old * K_old) / (alpha_new * K_new)
```

Here `K` is the axis/pair multiplier produced by count confidence. This reduces loss jumps caused only by parameter rescaling and makes the trial measure whether learning proceeds under the new setting.

The rebase scales existing GPU tensors in place and does not allocate a large extra VRAM buffer. Ranger slow params, momentum, and velocity are transformed consistently.

The rebase currently applies to axis/pair factorizer tensors only. It does not rebase the shared factorizer or residual count gate.

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
| `current-checkpoint/` | Checkpoint accepted at the latest completed generation; the next generation starts from here |
| `pending-commit-checkpoint/` | Temporary checkpoint after the commit run and before it is moved to `current-checkpoint/`; normally it does not remain |
| `generation-checkpoints/genXXXX/` | Stable per-generation checkpoint saved according to `tuning.save_rate` |
| `recommended-parameters.json` | Inferred parameters from top trials |
| `generation-best-parameters.csv` | Wide CSV summary of the best trial parameters for each generation. If a generation checkpoint is saved, the `checkpoint` column contains `generation-checkpoints/genXXXX/` |
| `runner-state.json` | Resume state |
| `logs/` | stdout for each trial |

`generation-best-parameters.csv` is updated once after each generation-end commit run.
If the same generation is resumed, the row for that generation is replaced instead of duplicated.

With the usual `commit_source: "best"` setting, the parameter columns in this CSV are the values passed to the next generation.
With `commit_source: "recommended"`, the CSV still records the best measured trial parameters, while the commit-run result is written separately in the `commit_*` columns.
In that case, the best trial parameters themselves may not have a saved checkpoint.

`current-checkpoint/` is a mutable runner directory. It is not a fixed per-generation snapshot, so `generation-best-parameters.csv` does not write it as a historical checkpoint. Use `generation-checkpoints/genXXXX/` for fixed retained checkpoints.

## How to read `recommended-parameters.json`

`recommended-parameters.json` contains two different kinds of parameter values.

| Field | Meaning |
| --- | --- |
| `best_observed` | The single completed trial with the best configured metric |
| `recommended.parameters` | Inferred values from the top completed trials |

`best_observed` is the best trial that actually ran. With short trials, it can be noisy.
`recommended.parameters` averages the top trials from the latest generation and is often the more useful value to inspect before a longer run.

The recommendation is computed as follows:

1. Sort completed trials from the latest generation by the configured metric.
2. Keep the top fraction specified by `tpe_good_fraction`.
3. Average those top trials with rank weights.

For example:

```json
"tpe_startup_trials": [16, 8],
"tpe_good_fraction": 0.25
```

After 16 completed trials, the runner uses the top `ceil(16 * 0.25) = 4` trials.
The rank weights are `4, 3, 2, 1` from best to fourth-best.

For parameters with `log: false`, or parameters that default to linear handling, the recommendation is a weighted arithmetic mean:

```text
recommended = (4 * p1 + 3 * p2 + 2 * p3 + 1 * p4) / (4 + 3 + 2 + 1)
```

For parameters with `log: true`, the average is taken in log space. This is a weighted geometric mean:

```text
recommended = exp((4 * log(p1) + 3 * log(p2) + 2 * log(p3) + 1 * log(p4)) / (4 + 3 + 2 + 1))
```

Learning rates such as `lr` and `lr_min` are good candidates for `log: true`.
Factorizer alpha and count confidence values often allow `min: 0`, so they usually use linear averaging because `log(0)` is undefined.

This recommendation calculation is separate from the TPE sampler that creates the next trial.
The TPE sampler compares the distributions of good and bad trials to generate candidates.
`recommended.parameters` is a compact human-readable estimate from completed trials.

## Checkpoint retention

`save_rate` controls how many generation checkpoints are kept.

```json
"save_rate": 1
```

`1` saves every committed generation as `generation-checkpoints/gen0001/`, `generation-checkpoints/gen0002/`, and so on. `0` disables generation checkpoint retention and only updates `current-checkpoint/` for resume.

`keep_all_trials` controls how many trial checkpoints are kept.

```json
"metric": "quantized_value_loss",
"lower_is_better": true,
"keep_all_trials": false
```

With this setting, lower `quantized_value_loss` is better. By default, trial checkpoints are not saved. The runner records each trial's metrics and parameters in `summary-learn.log`, then runs one generation-end commit run using the parameters selected by `commit_source`. That commit run becomes `current-checkpoint/`. If `save_rate` is `1`, the same checkpoint is also saved under `generation-checkpoints/genXXXX/`.

With `commit_source: "best"`, the commit run uses the parameters from the best measured trial in that generation. With `commit_source: "recommended"`, it uses the same inferred parameters written to `recommended-parameters.json` for the latest generation. `recommended` is an unevaluated estimate, so the safer default is `"best"`.

## Changing population or trial_sbs mid-run

`population` and `trial_sbs` define the generation boundary. If you change them in the middle of a generation, already-finished trials and new trials would have different meanings and should not be compared directly.

To change these settings safely, reset the generation you want to restart:

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json `
  --resume `
  --reset-generation 17
```

This discards trial results from generation 17 onward and restarts generation 17 from the generation 16 checkpoint. Training continues immediately.

If you only want to rewind the state and logs without starting training, add `--reset-only`:

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json `
  --resume `
  --reset-generation 17 `
  --reset-only
```

The runner records each generation's `population`, `trial_sbs`, and trial-number range in `runner-state.json`. Finished generations are therefore not reinterpreted when you edit the settings file. New settings apply to generations that have not started yet, or to a generation explicitly rewound with `--reset-generation`.

Even when a non-best trial checkpoint is deleted, `summary-learn.log` and `logs/trialXXXX.stdout.log` remain, so you can still inspect the metric values and stdout later.

To keep every trial checkpoint, use one of these:

- `keep_all_trials: true`
- `--keep-temp` at runtime

For normal tuning, `keep_all_trials: false` is safer because it avoids rapid storage growth.
