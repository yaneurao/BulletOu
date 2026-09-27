# Compare training conditions with grid search

For independent FT/L1/L2 BatchNorm switches, defaults, limitations and grid examples, see [Batch normalization](batch-normalization.md).

## Short experiments without editing common settings

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-short `
  --max-epochs 1 --superbatches 32 `
  --lrs 0.0001 0.0002
```

Each condition trains for one epoch of 32 SB. The source JSON is not modified. Precedence is common JSON < CLI overrides < explicit `--grid superbatches ...`. With both `--max-epochs` and `--epochs`, the former sets the training endpoint and the latter selects report epochs; report epochs beyond the endpoint are rejected. Without `--max-epochs`, `--epochs` retains its existing train-through-maximum behavior. Overrides also apply on `--resume`, but cannot reduce the total epoch budget. Repeat the arguments on resume to retain the overrides.

## Verbose saturation diagnostics

Training hides `[qstats]` / `[qstats-unit]` console lines by default. Use `python .\grid_search.py ... --verbose` to pass `--verbose` to BulletOu. Direct training also accepts `--verbose`, or `"verbose": true` in its settings JSON.

This only changes display: diagnostics computation, existing CSV fields, and acc/loss/qacc/qloss output are unchanged. The grid-level flag does not change trial identity and can be toggled on resume. A common JSON setting of `verbose: true` also enables detailed output.

## Initial LR warmup

Warmup is an independent **epoch 0**, unrestricted by `superbatches`. For example, `warmup_sb: 1024, superbatches: 64, max_epochs: 1` runs 1024 warmup SB plus 64 regular SB (1088 total), not just a one-SB prefix.

Set `"warmup_sb": 1` in JSON or `--warmup-sb 1` on the CLI for a one-SB epoch 0. LR rises linearly from near zero to `lr`. This is a nonnegative integer, default `0` (disabled).

- Supported for cuda-cpp production step/geometric/cos schedules, including workers. Plateau and direct-step smoke mode reject nonzero warmup.
- For optimizer update u=1..W during warmup, LR is `lr * u/W`. BPU changes do not change the requested SB duration.
- Epoch 0 is additional to `max_epochs`; regular epoch lengths and decay schedules are unchanged. Warmup uses epoch 1 settings, including LR and BPU.
- Regular decay begins at epoch 1. Explicit `lr_step_gamma: 1.0` keeps LR constant during regular epochs.
- Resume continues at the saved epoch/SB, without repeating warmup. A completed epoch 0 resumes at epoch 1. Separate initial-state runs and independent grid/worker trials start a new epoch 0.
- Epoch 0 follows the normal save/validation intervals and epoch-end save controls. It is recorded in `summary-learn.csv` and `grid_summary.csv`. This differs from the old within-epoch-1 warmup semantics: start comparisons in a new output folder.
- `warmup_sb` is run-wide, not an epoch-mapped setting. It is recorded in checkpoint metadata and grid CSV.

With common settings configured for 16 SB and one epoch, compare three conditions:

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260922-progress8-bceloss-16sb.json `
  --output-folder D:\BulletOu-snapshots\20260922\grid-progress8-warmup `
  --grid loss_bce_with_logits true `
  --grid lr 0.000400 `
  --grid lr_step_gamma 1.0 `
  --grid warmup_sb 0 1 4
```

This changes only warmup length.

The common JSON supports [per-epoch settings](epoch-settings.md), such as `"lr": {"epoch1": 0.0004, "epoch11": 0.0002}`, including boolean transitions for QAT. The CSV records effective values for each epoch.

`grid_summary.csv` updates **after each completed epoch during training**, as well as at trial start, exit, and interruption. The runner checks the trainer's `summary-learn.csv` approximately once per second, without waiting for more stdout or the whole trial to finish. In a ten-epoch trial, epoch 1 results therefore appear as soon as they are complete. Unfinished epochs retain blank metric cells; `--epochs` still controls which epochs are reported. Partial source records and temporarily locked output files produce a warning and are retried without stopping training.

Use `--grid loss_bce_with_logits false true` to compare squared error with BCE. Set the common `wrm_in_offset` to0 and disable explicit `win_rate_model` / `loss_sigmoid_mse` flags. See [BCE formulas and caveats](bce-with-logits.md).

With common JSON `loss_bce_with_logits: true`, use `--grid bce_error_weight_k 0 1 2` to compare error-weighted BCE training. Zero is ordinary BCE. Validation loss/qloss remain ordinary BCE for all K values; the coefficient is recorded in `grid_summary.csv`. See [formula and normalization](bce-with-logits.md#error-weighted-bce-training).

Use `--grid sfnn_norm_loss_strength 0 0.000001 0.00001 0.0001` to compare SFNN norm regularization. See [Norm loss: scope, formula and examples](norm-loss.md).

[日本語](../../ja/advanced/grid-search.md)

## Compare WRM teacher-probability compression

`--wrm-target-epsilon` (JSON: `wrm_target_epsilon`) transforms the existing WRM teacher probability `t` into `epsilon + (1 - 2*epsilon) * t`. Default **0 preserves the original calculation**; valid values are finite `0 <= epsilon < 0.5`. For epsilon=0.01, 0 maps to 0.01, 0.5 stays 0.5, and 1 maps to 0.99. This compresses the entire range towards 0.5 rather than clipping only the tails.

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260911-k3k3-b65536-sb40m.json `
  --output-folder D:\BulletOu-snapshots\grid-target-epsilon `
  --wrm-target-epsilons 0 0.005 0.01
```

This runs three conditions sequentially. Additional grid axes form a Cartesian product. `grid_summary.csv` includes `wrm_target_epsilon`. For standalone training, use `--wrm-target-epsilon 0.01` or `"wrm_target_epsilon": 0.01` in the settings JSON.

- Applied only to the teacher, before existing result/lambda blending; blending may therefore produce final targets outside `[epsilon, 1-epsilon]`. Prediction WRM, nn.bin format and accuracy definitions are unchanged.
- Training, regular validation, GPU quantized validation and exact CPU quantized validation share the target transform. Pass the same epsilon to standalone quantized validation.
- Unlike `wrm_target_offset`, whose limiting probabilities remain 0 and 1, epsilon changes those limits. It does not directly constrain network weights or outputs.
- Nonzero epsilon with `--loss-sigmoid-mse` is rejected.
- **Loss/qloss across different epsilon values are different objectives and cannot be ranked directly as a common loss.** Grid search warns about this; also compare accuracy and playing strength.

`grid_search.py` follows the workflow of YOSC's `trainer/grid_search.py`: enumerate the Cartesian product of explicit values, train each condition sequentially, and aggregate CSV results. It uses only the Python 3.10+ standard library and supports BulletOu's cuda-cpp production schedule.

This is separate from the TPE-based `tuning_parameters.py`. Trials are independent, never warm-started from another trial's winner or pruned. Fresh runs use the normal deterministic scratch initialization; checkpoint runs share the same initial state and dataloader position. GPU numerical reproducibility is not guaranteed.

## Preview, then run

Use an ordinary **bulletou-settings.json**, not tuning-settings.json. The [sample settings](../../examples/grid-search-bulletou-settings.json) use k3k3, FT factorization and L1 shared, 324 superbatches per epoch, 40M requested positions per superbatch, one epoch, and saves every 81 sb. Adjust paths and conditions for your machine. Adding these files does not start training.

From the BulletOu directory:

Use `"test_positions": "all"` in the JSON to validate all positions. Omission or `null` means the same thing. To limit the sample, use a positive integer such as `"test_positions": 300000`. Rebuild BulletOu if an older executable rejects `all`.

```powershell
python .\grid_search.py `
  --settings-file .\docs\examples\grid-search-bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\20260911\grid-target-scale `
  --lrs 0.000875 0.000400 `
  --wrm-target-scalings 600 1200 `
  --dry-run
```

This is 2 LRs × 2 target scalings = **4 conditions**. Dry-run checks supported options using the executable's `--help`, prints the plan, and writes no files or training results. Remove `--dry-run` to train. No Rust rebuild is needed for the runner.

Lists form a Cartesian product, not zipped pairs. Invalid combinations such as `lr_min > lr` fail before training; they are neither silently dropped nor corrected.

## Arguments

| Argument | Meaning |
| --- | --- |
| `--settings-file PATH` | Common BulletOu JSON; unnecessary for summary-only |
| `--output-folder DIR` | Required, dedicated grid root containing `grid_summary.csv` and each `trial0001-…/` directly |
| `--exe PATH` | Defaults to `target/release/examples/bulletou.exe` next to the script. Supply the appropriate executable on Ubuntu |
| `--checkpoint DIR` | Common starting checkpoint directory with nonempty `state.bin` and `dataloader_pos.txt` |
| `--epochs 1 2 5` | Train each condition once through epoch 5 and report epochs 1, 2, 5. Omitted: use JSON `max_epochs`, reporting every epoch |
| `--max-epochs N` | Override common JSON `max_epochs`; total epoch count, not additional epochs |
| `--superbatches N` | Override common JSON `superbatches` (SB per epoch) |
| `--summary-only` | Rebuild aggregate CSV from the manifest and logs; no training, executable, or original settings needed |
| `--summary-csv PATH` | Defaults to `<grid root>/grid_summary.csv`. Must be a `.csv` outside all trial directories (including legacy `trials/`) to protect source logs |
| `--dry-run` | Plan only; no writes or training |
| `--resume` | Resume unfinished conditions from their latest checkpoints; archive unsaved attempts and restart from the original initial state if no checkpoint exists |
| `--continue-on-error` | Continue remaining conditions after a child training failure. Runner still exits nonzero if any failed |

The runner controls `output`, `output_folder`, `tag`, `resume` per trial. Other common settings are preserved except explicit grid overrides and `--epochs`. The original JSON is never edited. Relative input paths use the **invocation working directory**, just as in standalone BulletOu. Resume from the same directory with the same command.

Without `--checkpoint`, existing `initial_state`/`initial_dataloader_pos` in the common JSON are honored; without either, training starts from scratch. Optimizer loading and factorizer-migration reset rules remain those of BulletOu. The runner adds no optimizer resets.

### Grid axes

| Argument | BulletOu setting |
| --- | --- |
| `--lrs` | `lr` |
| `--lr-mins` | `lr_min` |
| `--wrm-target-scalings` | `wrm_target_scaling` |
| `--wrm-in-scalings` | `wrm_in_scaling` |
| `--wrm-nnue2scores` | `wrm_nnue2score` |
| `--batch-sizes` | `batch_size` |
| `--batches-per-updates` | `batches_per_update` |
| `--factorizers` | `sfnn_factorizer` |
| `--loss-pow-exps` | `loss_pow_exp` |

Other scalar options use repeatable `--grid NAME VALUE1 VALUE2 ...`. Names are BulletOu option names without leading `--`; hyphens and underscores are accepted.

```powershell
python .\grid_search.py `
  --settings-file .\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\20260911\grid-shared `
  --checkpoint C:\path\to\0033 `
  --grid sfnn_factorizer_alpha "shared=0.5" "shared=1.0" `
  --grid ft_factorizer false true `
  --epochs 1 2
```

This runs four conditions. The positive boolean options `ft_factorizer` (alias `sfnn_ft_factorizer`), `save_epoch_end`, and `sfnn_factorized` explicitly pass both `true` and `false`. Other flags retain the JSON convention: `true` passes a flag and `false` omits it, so false does not necessarily disable a default-enabled feature. Numbers and strings are supported. Lifecycle/initial-state options, including output paths and resume, cannot be grid axes.

Use `--grid max-epochs 1 2` (`max_epochs` also works) to train independent one-epoch and two-epoch conditions. Combining it with `--grid superbatches 16 32` produces four conditions. The grid axis overrides common JSON and `--max-epochs`. Reports stop at each condition's endpoint; `--epochs` optionally selects reported epochs within that range. On resume, a different epoch-budget grid value is a separate condition, not an extension of an existing condition.

## Outputs

```text
grid-target-scale/
  grid-manifest.json
  grid_summary.csv
  grid.lock
  trial0001-lr=...-<hash>/
    bulletou-settings.json
    grid-state.json
    stdout.log
    summary-learn.csv
    0001/state.bin, nn.bin, dataloader_pos.txt, ...
  trial0002-.../
    ...
```

New trials are created directly under the grid root. Existing manifests retain their nested `trials/` paths on `--resume`; existing folders are not moved. Newly added conditions use the flat layout.

The short hash covers the complete condition; full settings are kept in the per-trial JSON and manifest. Resuming also writes `bulletou-resume-settings.json`, omitting the common initial-state arguments so BulletOu resumes the trial itself.

Like YOSC, the aggregate CSV includes condition rows before training starts, with settings, target epoch, output directory and status. It is rebuilt at trial start/end/interruption. **Measurements are populated per epoch when its final sb is recorded**, even if the overall trial remains unfinished. Only unfinished epochs have blank metrics, last-eight-SB averages, measured sb and checkpoint. Inspect `summary-learn.csv` or `stdout.log` for intermediate measurements. Aggregation never modifies source logs.

Example: after extending a completed epoch 1 to max_epochs=3, while epoch 2 is running, epoch 1 retains its measurements and epochs 2–3 show conditions only. Completed epochs remain visible even if the trial is interrupted or fails. Existing aggregate CSVs adopt these rules on the next run or with `--summary-only`.

One row represents **one condition × one reported epoch**:

| Columns | Meaning |
| --- | --- |
| `trial`, `epoch`, `superbatch` | Condition ID, epoch, last recorded sb |
| `test_value_accuracy`, `test_value_loss`, `quantized_value_accuracy`, `quantized_value_loss` | **acc, loss, qacc, qloss**, in that order, from the last row in the epoch. Accuracy is a 0–1 fraction |
| `last8_acc`, `last8_loss`, `last8_qacc`, `last8_qloss` | Arithmetic means of finite validation values within the final eight SB of each epoch, formatted with exactly ten decimal places (e.g. `0.6349000000`), immediately after the final-SB metrics. Replace the eight extrema/value-location columns |
| `positions` | Cumulative position count from the last native row. Final-sb `lr_start` / `lr_end` are omitted; configured `lr` / `lr_min` remain in the condition columns |
| `lr`, `lr_min`, `wrm_target_scaling`, etc. | Settings, including individual grid-axis columns. Unspecified executable defaults are not guessed |
| `status` | `done` for a completed epoch; otherwise `pending`, `running`, `interrupted`, `failed` or `incomplete`. This may differ from the overall `trial_status` |
| `trial_status` | Whole-condition state. `elapsed_seconds` is not included in the aggregate CSV |
| `output_dir` | Condition directory |
| `checkpoint` | **Last column**: saved checkpoint corresponding to the last row, blank if unsaved or removed |

Missing/non-finite metrics remain blank. Older measured values are never substituted into an unmeasured final row. `last8_*` uses SB numbers N−7 through N, where N is the final SB, **not the last eight validation events**. Epochs shorter than eight SB use the whole epoch; the window never crosses epochs (warmup epoch 0 is independent). Missing/non-finite values are excluded separately for each metric; no valid values means blank. Averages for unfinished epochs remain blank. At completion, stdout lists the best epoch-end condition independently for all four metrics; no overall score/winner is silently chosen.

On the next launch or `--summary-only`, existing aggregate CSVs are rebuilt from source logs with the new columns. No executable rebuild is needed; an already running runner is unaffected. Native `summary-learn.csv` is not modified.

Native save frequency and epoch-end saves are unchanged. **The runner neither deletes checkpoints nor makes best-checkpoint copies.** Budget disk space for all conditions' saves.

For saves only at each epoch end, set `"save_rate": 0` or `"save_rate": "none"` in the common settings. Explicit `validation_rate: 1` / `quantized_validation_rate: 1` still measure every sb independently. See the [validation tutorial](../tutorial/4-validation.md#45-save-frequency-is-separate) for disabling epoch-end saves and the associated caveats.

## Resume and summary-only

Repeating a command skips completed conditions. Add `--resume` for unfinished conditions. If a checkpoint exists, resume from it; unsaved progress rolls back according to native BulletOu resume rules.

**If no resumable checkpoint exists, the trial restarts from its original initial state within the same grid.** An interruption before the first save in epoch 1 therefore restarts at the beginning of epoch 1. The original common initial checkpoint and dataloader position are honored if configured; otherwise initialization is from scratch. The interrupted sb's unsaved state cannot be recovered.

The previous trial directory is moved intact to `<grid root>/interrupted-runs/<trial-name>-<timestamp>-<identifier>/`, and the trial restarts under its original ID and output directory. Old logs and incomplete saves are preserved but excluded from the new run's aggregate. Stdout prints `[RESTART]` and `[ARCHIVE]`. Completed and unselected conditions are not restarted. No manual deletion or new grid root is necessary.

Ordinary reruns reject a changed plan. With `--resume`, the same grid condition resumes its own checkpoint while accepting changes to the common JSON, including LR, batch size, bpu, and save/validation rates. `[SETTINGS CHANGED]` prints the old and new values. Changes apply at the next invocation; the runner does not hot-reload JSON during training.

Explicit grid arguments identify conditions and override common JSON values. For example, `--lrs 0.0001 0.0002` overrides JSON `lr`. A different grid value creates a new condition rather than reusing another condition's checkpoint. Changes to `arch`, `backend`, or `ft_factorizer` are rejected for checkpoint compatibility. Other settings remain subject to native BulletOu argument and checkpoint validation.

Changing settings alone does not restart completed trials: increase `max_epochs` to extend them. Results after common-setting changes are not equivalent to training with one constant configuration from the beginning. Rebuilding the executable at the same path is allowed, but implementation changes and input-file content changes also affect comparisons.

Each trial's `grid-settings-history.json` records launch settings, timestamps, and the last epoch/sb present in the log before launch. This log position is not the actual checkpoint resume position. Original `bulletou-settings.json` files remain immutable; actual launch settings go into `bulletou-resume-settings.json` (checkpoint resume) or `bulletou-run-settings.json` (fresh run). The common JSON is never written back.

Settings for previously completed epochs are preserved in the manifest, so their LR, bpu and sb columns in `grid_summary.csv` are not relabeled with new settings. An epoch interrupted by a settings change displays the settings used at completion, not a claim that those settings applied throughout the epoch; consult launch history for transitions. If no checkpoint exists, the archived attempt is restarted using the newly requested initial settings.

Changed, added or removed settings automatically become columns in `grid_summary.csv`, even outside the predefined summary fields. For example, enabling QAT during training adds its setting column with historical values for completed epochs and new values afterward. Columns remain after reverting or removing a setting. Unspecified values are blank; native BulletOu defaults are not guessed. Additional columns follow existing condition columns and precede status/administrative columns; checkpoint remains last. Changes recorded in older manifests' initial and per-epoch settings are also recovered.

### Extend completed conditions

Normal runs and `--resume` follow **the current command-line condition order** for execution and CSV display, like normal YOSC runs. Values are not numerically sorted; existing and new conditions share the same ordering. Unselected existing conditions follow at the end in their previous relative order. IDs and output paths remain unchanged, so trial IDs need not be ascending in the CSV. `--summary-only` retains the last order stored in the manifest.

For an existing 600/1200/1800 grid planned for five epochs each, extend only 600/1200 by five epochs, **to ten epochs total**, using the same output root:

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260911-k3k3-b65536-sb40m.json `
  --output-folder D:\BulletOu-snapshots\20260911\grid-k3k3-b65536-sb40m `
  --wrm-target-scalings 600 1200 `
  --epochs 10 `
  --resume
```

- `--epochs 10` is the total endpoint, not ten additional epochs. It overrides the common JSON's `max_epochs`, so that file can remain at five.
- Grid arguments select conditions; no separate selection flag is needed. Keep all original axis names, but value lists may be narrowed or expanded. With `--resume`, unknown conditions are appended after the highest existing trial ID. Existing IDs, folders and results stay unchanged. New conditions train from the common initial state and teacher position; completed existing conditions are skipped.
- Example: after `--grid wrm_target_offset 135 270 540 0`, use `--grid wrm_target_offset 70 35 100 135 170 200 235 270 540 0 --resume` with the same output folder to add six conditions. Unselected existing conditions remain in the manifest and aggregate. Completed new conditions join the same `grid_summary.csv`. Common JSON changes apply to selected conditions. Add `--dry-run` to inspect additions and changes without writing files or starting training.
- Each selected condition resumes its own saved weights, optimizer state and teacher position via native `--resume`. A completed epoch-five checkpoint starts at epoch six; interrupted unsaved progress rolls back to the last checkpoint.
- IDs and directory names, including their original hashes, stay unchanged. Original `bulletou-settings.json` files stay intact; updated launch settings go into `bulletou-resume-settings.json` and the manifest. The common JSON is never written back.
- Existing report epochs are preserved and every newly added epoch is included in the same `grid_summary.csv`. With the original epochs 1–5, `--epochs 10` reports 1–10 on extension.
- Unselected 1800 retains its original budget and results. It is not launched and does not get fictitious rows for epochs 6–10.
- If an extended condition has no usable checkpoint, its previous result is archived and training restarts from the original initial state through the target epoch. Extending five to ten without a checkpoint therefore trains epochs 1–10 again.
- Repeat the same `--resume --epochs 10` command after interruption. Once complete, it skips completed conditions instead of adding more epochs. Reducing the target is rejected.
- Add `--dry-run` to inspect mapped trials, existing folders and target epochs without writing files or starting training.

Stop any runner already using this grid before executing the extension. The output root remains protected by an exclusive file lock.

```powershell
python .\grid_search.py `
  --output-folder D:\BulletOu-snapshots\20260911\grid-target-scale `
  --summary-only
```

Add `--epochs 1 2` to regenerate only those epochs. Source logs are untouched. The aggregate CSV is generated output: manual edits are replaced on aggregation. An OS file lock prevents concurrent runners/summary writers for the same root; `grid.lock` remaining after exit is normal.

## Comparison caveats

- Losses with different target scaling, loss kind or exponent are not numerically comparable as the same objective. The printed minimum-loss result does not correct for that difference.
- Keep validation teacher, sample size, seed and quantized validation mode identical. qacc alone does not establish playing strength.
- Changing batch size/bpu never automatically changes the position budget. Native logs describe rounding and update counts.
- Each condition uses a **separate trainer process**, not worker mode. Teacher RAM and validation caches are not shared across conditions; startup overhead matters for very short trials.
- Validation rate `0` means epoch-end only; `-1` disables periodic validation. Quantized validation on saves follows the native rules.
- This runner does not implement YOSC's `--value-loss-min-weight` in BulletOu. Unsupported executable options are rejected before training.
