# Automatic ES tuning

<a href="../../ja/advanced/auto-tuning.md"><img alt="Read in Japanese" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

Factorizer alpha values and count-confidence values have many useful combinations. If you already have a good checkpoint, use `es_local_runner.py` to run a beam-search-style ES (evolution strategy) around it.

This page explains the `parameters.json` file, every parameter key, the runner arguments, and how to reuse the same JSON file for normal training without running ES.

## What the runner does

`es_local_runner.py` creates multiple candidates, trains them for short runs, and keeps the better candidates.

One generation works like this:

1. Read current values from `parameters.json`
2. Randomly perturb parameters with `tune: true`
3. Create `population` candidates
4. Train each candidate for the superbatches described by `beam`
5. Rank candidates by `metric` and keep only the requested count
6. Promote the final survivor's NN weights and hyperparameters
7. Write the survivor's values back to `parameters.json`

There is no separate partial parameter update after selection. The final survivor's parameter values directly become the next generation's starting point.

## Full `parameters.json` shape

The same `parameters.json` file is used by both the ES runner and normal `bulletou.exe` training.

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

`es.enabled` selects how the JSON file is used.

| Value | Meaning |
| --- | --- |
| `true` | Run ES with `es_local_runner.py` |
| `false` | Use only `parameters.*.current` values in normal `bulletou.exe` training |

## `es` fields

| Field | Meaning |
| --- | --- |
| `enabled` | `true` for `es_local_runner.py`; `false` for `bulletou.exe --parameters-file` |
| `generations` | Number of ES generations. One survivor is accepted per generation |
| `population` | Number of candidates created at the start of each generation |
| `beam` | When to prune candidates and how many to keep |
| `metric` | Metric used to rank candidates |
| `lower_is_better` | `true` if smaller metric values are better, `false` if larger values are better |
| `seed` | Random seed for candidate generation |
| `save_rate` | Copy a public checkpoint to `accepted-checkpoints/` every N accepted generations |
| `candidate_validation_rate` | How often each candidate prints test accuracy/loss |
| `candidate_quantized_validation_rate` | How often each candidate prints qacc/qloss |

`beam` is written like this:

```json
"beam": [
  { "after_sbs": 8, "keep": 8 },
  { "after_sbs": 16, "keep": 4 },
  { "after_sbs": 24, "keep": 2 },
  { "after_sbs": 32, "keep": 1 }
]
```

This example starts 16 candidates, keeps 8 after 8 sb, 4 after 16 sb, 2 after 24 sb, and 1 after 32 sb. The final `keep` must be `1`.

Supported `metric` values are:

| Value | Meaning | `lower_is_better` |
| --- | --- | --- |
| `quantized_value_loss` | Loss after quantized validation. This is usually the most useful tuning target | `true` |
| `quantized_value_accuracy` | Accuracy after quantized validation | `false` |
| `test_value_loss` | Validation loss with f32 weights | `true` |
| `test_value_accuracy` | Validation accuracy with f32 weights | `false` |

A practical default is `metric = "quantized_value_loss"` and `lower_is_better = true`.

## Common fields under `parameters`

Each entry under `parameters` has this shape:

```json
"pair": { "current": 0.3, "tune": true, "step": 0.02, "min": 0.0, "max": 10.0 }
```

| Field | Meaning in ES runner | Meaning in normal training |
| --- | --- | --- |
| `current` | Current value. Candidates are sampled around this value | The only value passed to `bulletou.exe` |
| `tune` | If `true`, this parameter is randomized for candidates. If `false`, it is fixed | Ignored |
| `step` | Candidate values are sampled in `current ± step` | Ignored |
| `min` | Lower bound for candidate values | Ignored |
| `max` | Upper bound for candidate values | Ignored |

For example, if `pair.current = 0.3` and `pair.step = 0.02`, candidate `pair` values are sampled between `0.28` and `0.32`. Entries with `tune: false` do not move.

## Alpha parameters

Alpha values control how strongly factorizer terms contribute to forward/backward. `1.0` means unchanged, `0.5` means half strength, and `2.0` means double strength.

| JSON key | Meaning | Used when |
| --- | --- | --- |
| `shared` | Strength of the shared factorizer term | Shared factorizer is active |
| `king_axis` | Strength of the king-axis factorizer term | The arch has king buckets such as `k3k3`, `k9k9`, `k21k21`, or `k29k29` |
| `hand_axis` | Strength of the hand-axis factorizer term | The arch has hand buckets such as `hand4`, `hand16`, `hand64`, or `hand1024` |
| `progress_axis` | Strength of the progress-axis factorizer term | The arch has `progress4`, `progress8`, etc. |
| `pair` | Strength of pair factorizer terms | `--sfnn-factorizer pair` enables pair terms |

`pair` is one alpha value for king-hand, king-progress, and hand-progress pair terms. Per-pair alpha values are not separate at the moment; use count-confidence keys if you need to treat pair kinds differently.

The effective weight can be thought of as:

```text
W_effective =
    W_residual
  + shared_alpha * W_shared
  + axis_alpha   * axis_confidence * W_axis
  + pair_alpha   * pair_confidence * W_pair
```

Lowering `shared` changes the global base term, so keeping `shared = 1.0` is usually the safer starting point. Tuning axis and pair values changes how bucket-specific structure is used.

## Count-confidence parameters

Count confidence uses a `count.bin` file produced by `bucket-count`. It weakens terms whose bucket or factorizer row appears too rarely. A value of `0.0` disables that confidence. Larger values require more observed positions before the term is trusted strongly.

If any count-confidence value is non-zero, pass `--bucket-counts <count.bin>` to the runner, or `--sfnn-bucket-counts <count.bin>` to normal training.

| JSON key | CLI equivalent | Meaning |
| --- | --- | --- |
| `residual_count` | `--sfnn-residual-count-confidence` | Dampens bucket-specific residual terms by count |
| `axis_count` | `--sfnn-axis-count-confidence` | Shared value for king / hand / progress axis terms |
| `king_axis_count` | `--sfnn-king-axis-count-confidence` | Override for king-axis rows |
| `hand_axis_count` | `--sfnn-hand-axis-count-confidence` | Override for hand-axis rows |
| `progress_axis_count` | `--sfnn-progress-axis-count-confidence` | Override for progress-axis rows |
| `pair_count` | `--sfnn-pair-count-confidence` | Shared value for king-hand / king-progress / hand-progress pair terms |
| `king_hand_pair_count` | `--sfnn-king-hand-pair-count-confidence` | Override for king-hand pair rows |
| `king_progress_pair_count` | `--sfnn-king-progress-pair-count-confidence` | Override for king-progress pair rows |
| `hand_progress_pair_count` | `--sfnn-hand-progress-pair-count-confidence` | Override for hand-progress pair rows |

If a specific key is omitted, the group key is used. For example, if `axis_count = 1.0` and `king_axis_count` is omitted, king-axis rows use `1.0`. If a specific key is explicitly set to `0.0`, that axis or pair kind is disabled.

Axis and pair factorizer terms are multiplied by:

```text
confidence = count_term / (count_term + term_params * option_value)
```

`count_term` is the summed count for the buckets that touch that factorizer row. `term_params` is the number of parameters owned by that row. Larger `option_value` means rare rows are weakened more.

Residual confidence is different: it applies decay to bucket-specific residual weights. See [SFNN factorizer](sfnn-factorizer.md) for the detailed formula.

## ES runner example

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

## Runner arguments

| Argument | Meaning |
| --- | --- |
| `--exe` | Path to `bulletou.exe` |
| `--parameters-file` | JSON file containing ES settings and parameter values. `es.enabled` must be `true` |
| `--base-checkpoint` | Initial checkpoint directory. It must contain `state.bin` and `dataloader_pos.txt` |
| `--teacher` | Training teacher data |
| `--test-teacher` | Validation teacher data |
| `--arch` | Architecture to train |
| `--bucket-counts` | `count.bin` used by count-confidence parameters |
| `--output-folder` | Parent directory for the runner root |
| `--temp-folder` | Directory for temporary candidate checkpoints. A fast SSD is recommended |
| `--tag-prefix` | Name used in the runner root |
| `--factorizer` | Value passed as `--sfnn-factorizer` to candidate runs |
| `--positions-per-superbatch` | Positions per sb |
| `--generations` | Temporarily overrides `es.generations` |
| `--save-rate` | Temporarily overrides `es.save_rate` |
| `--metric` | Temporarily overrides `es.metric` |
| `--resume` | Resume from `runner-state.json` and `current/` |
| `--keep-temp` | Keep pruned candidate directories |
| `--dry-run` | Print commands without training |
| `--no-stream-child-output` | Do not mirror child `bulletou.exe` stdout to the console. Logs are still written |
| `--color` | Controls colored output: `auto`, `always`, or `never` |

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

In this mode, `bulletou.exe` reads only `parameters.*.current`. `step`, `tune`, and `beam` are not used by normal training.

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
| `logs/` | Per-candidate stdout logs |
| `temp/` | Candidate temporary checkpoints. If `--temp-folder` is set, they are created there instead |

Use `--temp-folder` to place temporary candidate checkpoints on a fast SSD such as `C:\BulletOu-es-temp`. Pruned candidates are deleted automatically. Use `--keep-temp` only when you want to inspect those directories.

## Resume

To resume, use the same `--output-folder` and `--tag-prefix`, then add `--resume`. Current hyperparameters are read from `parameters.json`, so you do not need to paste checkpoint-time values into the command line.

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
