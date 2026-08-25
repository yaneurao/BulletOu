# Fixed-length trial parameter tuning

`tuning_parameters.py` runs many short fixed-parameter BulletOu trials to search over `lr`, `lr_min`, factorizer, and count-confidence settings while training moves forward generation by generation.

It does not depend on external Python packages. It is a small BulletOu-specific TPE-style sampler: random trials first, then it compares the distribution of good trials against the distribution of bad trials and samples promising regions more often.

## What this runner does

- Trials in the same generation start from the same checkpoint.
- The length of one trial is `tuning.trial_sbs`. It can be a number or an array.
- Parameters stay fixed during a trial.
- For every generation, the runner evaluates `population` trials and records the metric.
- At the end of a generation, the runner performs one commit run with the selected parameters and saves that checkpoint as `current-checkpoint/`.
- `best-checkpoint/` stores the best checkpoint among completed commit runs. It is separate from the next generation's starting point.
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

Set `schedule_repeat: true` to repeat the arrays cyclically instead of reusing the last value. For example, `population: [100, 0]` and `trial_sbs: [4, 128]` means odd generations do short search trials and even generations do long stabilization training.

The TPE-style sampler uses results from the immediately previous generation. It sorts completed trials by the configured metric, treats the top fraction as good trials and the rest as bad trials, builds per-parameter distributions, and prefers values that are likely under the good distribution and unlikely under the bad distribution.

If the previous generation has fewer than that generation's `tpe_startup_trials` completed trials, or if it has only one trial and cannot form a bad distribution, the runner does not fall back to fully uniform random sampling. Instead, it samples with Gaussian noise around the previous generation's `recommended` parameters. In other words, generation 1 explores broadly, and generation 2 and later explore around what the previous generation learned.

For example, generation 2 trials continue from the commit checkpoint produced at the end of generation 1, and their candidate parameters are sampled from generation 1 results. Generation 3 then continues from the commit checkpoint produced at the end of generation 2 and samples from generation 2 results.

The runner does not mix all generations for TPE, because each generation starts from a different checkpoint. Mixing metrics from different training stages would make older generations unfairly worse and distort the sampler.

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
| `tpe_startup_trials` | Number of completed trials required for TPE density estimation. This can be a number or an array. Arrays are interpreted per generation, like `population` and `trial_sbs`. With `schedule_repeat: false`, the last value is reused when the array is shorter than `generations`; with `true`, the array is repeated cyclically. In generation 1, trials are sampled from the full range until there are enough observations. In generation 2 and later, if the previous generation has too few trials, candidates are sampled around the previous generation's `recommended` parameters instead. | `16` |
| `tpe_good_fraction` | Fraction of completed trials treated as good candidates. `0.25` means the top 25% are used. | `0.25` |
| `tpe_bandwidth` | Lower bound for the TPE KDE width. Larger values spread candidates more broadly; smaller values concentrate them closer to observed good trials. | `0.15` |
| `commit_source` | Parameters used for the generation-end commit run. `"best"` uses the best measured trial; `"recommended"` uses the inferred value from the top trials. | `"best"` |

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
    "commit_source": "best",
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
| `current-checkpoint/` | Checkpoint accepted at the latest completed generation; the next generation starts from here |
| `pending-commit-checkpoint/` | Temporary checkpoint after the commit run and before it is moved to `current-checkpoint/`; normally it does not remain |
| `best-checkpoint/` | Best checkpoint among completed commit runs |
| `recommended-parameters.json` | Inferred parameters from top trials |
| `runner-state.json` | Resume state |
| `logs/` | stdout for each trial |

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

`keep_all_trials` controls how many trial checkpoints are kept.

```json
"metric": "quantized_value_loss",
"lower_is_better": true,
"keep_all_trials": false
```

With this setting, lower `quantized_value_loss` is better. By default, trial checkpoints are not saved. The runner records each trial's metrics and parameters in `summary-learn.log`, then runs one generation-end commit run using the parameters selected by `commit_source`. That commit run becomes `current-checkpoint/`.

With `commit_source: "best"`, the commit run uses the parameters from the best measured trial in that generation. With `commit_source: "recommended"`, it uses the same inferred parameters written to `recommended-parameters.json` for the latest generation. `recommended` is an unevaluated estimate, so the safer default is `"best"`.

Even when a non-best trial checkpoint is deleted, `summary-learn.log` and `logs/trialXXXX.stdout.log` remain, so you can still inspect the metric values and stdout later.

To keep every trial checkpoint, use one of these:

- `keep_all_trials: true`
- `--keep-temp` at runtime

For normal tuning, `keep_all_trials: false` is safer because it avoids rapid storage growth.
