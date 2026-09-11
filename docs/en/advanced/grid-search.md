# Compare training conditions with grid search

[日本語](../../ja/advanced/grid-search.md)

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
| `--output-folder DIR` | Required, dedicated grid root for `grid_summary.csv` and `trials/` |
| `--exe PATH` | Defaults to `target/release/examples/bulletou.exe` next to the script. Supply the appropriate executable on Ubuntu |
| `--checkpoint DIR` | Common starting checkpoint directory with nonempty `state.bin` and `dataloader_pos.txt` |
| `--epochs 1 2 5` | Train each condition once through epoch 5 and report epochs 1, 2, 5. Omitted: use JSON `max_epochs`, reporting every epoch |
| `--summary-only` | Rebuild aggregate CSV from the manifest and logs; no training, executable, or original settings needed |
| `--summary-csv PATH` | Defaults to `<grid root>/grid_summary.csv`. Must be a `.csv` outside `trials/` to protect source logs |
| `--dry-run` | Plan only; no writes or training |
| `--resume` | Resume unfinished conditions from their latest checkpoints; archive unsaved attempts and restart from the original initial state if no checkpoint exists |
| `--continue-on-error` | Continue remaining conditions after a child training failure. Runner still exits nonzero if any failed |

The runner controls `output`, `output_folder`, `tag`, `resume`, and `no_resume` per trial. Other common settings are preserved except explicit grid overrides and `--epochs`. The original JSON is never edited. Relative input paths use the **invocation working directory**, just as in standalone BulletOu. Resume from the same directory with the same command.

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
  --grid no_ft_factorize false true `
  --epochs 1 2
```

This runs four conditions. As in BulletOu JSON, `true` passes a flag and `false` omits it; **false does not necessarily disable the underlying feature**. Numbers and strings are supported. Lifecycle/initial-state options, including output paths, resume, and max_epochs, cannot be grid axes.

## Outputs

```text
grid-target-scale/
  grid-manifest.json
  grid_summary.csv
  grid.lock
  trials/
    trial0001-lr=...-<hash>/
      bulletou-settings.json
      grid-state.json
      stdout.log
      summary-learn.csv
      summary-epoch-last.csv
      0001/state.bin, nn.bin, dataloader_pos.txt, ...
    trial0002-.../
      ...
```

The short hash covers the complete condition; full settings are kept in the per-trial JSON and manifest. Resuming also writes `bulletou-resume-settings.json`, omitting the common initial-state arguments so BulletOu resumes the trial itself.

The aggregate CSV exists before the first training run, initially containing the header and empty result rows. It is updated at trial start/end/interruption. During a trial, inspect its ordinary `summary-learn.csv` and live prefixed stdout; the child output is also appended to `stdout.log`.

One row represents **one condition × one reported epoch**:

| Columns | Meaning |
| --- | --- |
| `trial`, `epoch`, `superbatch` | Condition ID, epoch, last recorded sb |
| `test_value_accuracy`, `test_value_loss`, `quantized_value_accuracy`, `quantized_value_loss` | **acc, loss, qacc, qloss**, in that order, from the last row in the epoch. Accuracy is a 0–1 fraction |
| `max_acc`, `min_loss`, `max_qacc`, `min_qloss` | Independently measured epoch extrema, possibly from different sb |
| Corresponding `*_sb` columns | Location of each extremum; first sb on ties |
| `positions`, `lr_start`, `lr_end` | Last native row's position count and LR interval. `lr_start` need not be the epoch-start LR |
| `lr`, `lr_min`, `wrm_target_scaling`, etc. | Settings, including individual grid-axis columns. Unspecified executable defaults are not guessed |
| `status` | Epoch progress: `done` only when its final sb is present |
| `trial_status`, `elapsed_seconds` | Whole-condition state and total elapsed seconds including startup; repeated across its epoch rows |
| `output_dir` | Condition directory |
| `checkpoint` | **Last column**: saved checkpoint corresponding to the last row, blank if unsaved or removed |

Missing/non-finite metrics remain blank. Older measured values are never substituted into an unmeasured final row. No checkpoint path is invented for an unsaved peak. At completion, stdout lists the best epoch-end condition independently for all four metrics; no overall score/winner is silently chosen.

Native save frequency and epoch-end saves are unchanged. **The runner neither deletes checkpoints nor makes best-checkpoint copies.** Budget disk space for all conditions' saves.

For saves only at each epoch end, set `"save_rate": 0` or `"save_rate": "none"` in the common settings. Explicit `validation_rate: 1` / `quantized_validation_rate: 1` still measure every sb independently. See the [validation tutorial](../tutorial/4-validation.md#45-save-frequency-is-separate) for disabling epoch-end saves and the associated caveats.

## Resume and summary-only

Repeating a command skips completed conditions. Add `--resume` for unfinished conditions. If a checkpoint exists, resume from it; unsaved progress rolls back according to native BulletOu resume rules.

**If no resumable checkpoint exists, the trial restarts from its original initial state within the same grid.** An interruption before the first save in epoch 1 therefore restarts at the beginning of epoch 1. The original common initial checkpoint and dataloader position are honored if configured; otherwise initialization is from scratch. The interrupted sb's unsaved state cannot be recovered.

The previous trial directory is moved intact to `<grid root>/interrupted-runs/<trial-name>-<timestamp>-<identifier>/`, and the trial restarts under its original ID and output directory. Old logs and incomplete saves are preserved but excluded from the new run's aggregate. Stdout prints `[RESTART]` and `[ARCHIVE]`. Completed and unselected conditions are not restarted. No manual deletion or new grid root is necessary.

Ordinary reruns reject a changed plan. With `--resume`, you may select a subset of existing grid conditions and increase their total epoch budget. Other training settings (LR, teacher, batch size, etc.) must remain unchanged. Rebuilding the executable at the same path is allowed, but comparisons across implementations require care. Keep input files, initial state, progress.bin and count.bin contents fixed as well.

### Extend completed conditions

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
- Existing grid arguments select conditions; no separate selection flag is needed. Keep all original axis names and narrow their value lists. Unknown conditions are rejected, not added.
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
