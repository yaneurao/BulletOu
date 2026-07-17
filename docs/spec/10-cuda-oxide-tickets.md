# 10. cuda-oxide implementation tickets

This is the active ticket queue for turning the current cuda-oxide smoke/bridge
work into a production BulletOu training backend. Work the tickets in order and
commit each completed slice.

| ticket | status | scope | exit criteria |
|---|---|---|---|
| BO-CUDA-001 | done | cuda-oxide resume from root `state.bin` | `--nnue-teacher-train` can restore weights + Ranger optimizer state from root-format `state.bin`, not only `state.boung`; smoke verifies the same next-step result as `state.boung` resume |
| BO-CUDA-002 | done | promote direct cuda-oxide loop into end-user BulletOu CLI | `examples/bulletou.rs` exposes an opt-in cuda-oxide NNUE HalfKP training path that writes the normal numbered checkpoint layout |
| BO-CUDA-003 | doing | production schedule integration | cuda-oxide path honors `--superbatches`, epoch boundaries, LR schedule, `--save-rate`, positions carry-over, and plateau control in the same user-facing sense as the Bullet backend |
| BO-CUDA-004 | done | validation metrics integration | cuda-oxide checkpoints write production-compatible `learn.log` / `summary-learn.log` columns including `test_value_accuracy`, `test_value_loss`, and `train_value_loss` |
| BO-CUDA-005 | todo | dataloader resume generalisation | HCPE3, shogipack, multi-teacher specs, and teacher changes have explicit resume behavior and smoke coverage |
| BO-CUDA-006 | todo | async input/readback rings | input upload and loss readback are pipelined without changing fp32 baseline results |
| BO-CUDA-007 | todo | speed benchmark | same teacher / seed / schedule benchmark compares Bullet backend vs cuda-oxide positions/sec |
| BO-CUDA-008 | todo | SFNN training integration | SFNN cuda-oxide training path can stream real teacher batches and write compatible checkpoints |

## Notes

- `BO-CUDA-001` is first because all later production integration work becomes
  safer once cuda-oxide can resume from the same root-format `state.bin` that
  the normal BulletOu CLI uses.
- `BO-CUDA-002` should remain opt-in until the schedule, validation, and async
  tickets have caught up.
- Update each row's status as work lands: `todo` -> `doing` -> `done`.

## Completed notes

### BO-CUDA-001

- Added `--nnue-train-state-bin <PATH>` to `bulletou-cuda-train --nnue-teacher-train`.
- Root-format `state.bin` restore loads `nnue/{weights,momentum,velocity,slow,step_ranger}/*`.
- `--output` auto-resume still prefers `state.boung`, but falls back to `state.bin` if the exact cuda-oxide fixture is absent.
- Validation: resuming from `0001/state.bin` and from the matching `0001/state.boung` produced the same step2 loss (`0.031774815`) and identical final fixture SHA-256 `07424BCDCA1802127E16AF14DF3887A26AC72A5FB7FA5D176704497C17E27396`.
- Validation: temporarily hiding `0001/state.boung` made `--output` auto-resume select `0001/state.bin`, restore `resume_hcpe byte_offset=76`, run step2, and write `0002`.

### BO-CUDA-002

- Added `--backend cuda-oxide` as an opt-in BulletOu CLI path for direct NNUE HalfKP teacher training.
- Added temporary cuda-oxide bridge flags: `--cuda-oxide-train-steps`, `--cuda-oxide-cargo-dir`, `--cuda-oxide-release`, `--cuda-oxide-ptx`, `--cuda-oxide-weights-bin`, `--cuda-oxide-device`, and `--cuda-oxide-debug-readback`.
- The wrapper launches `cargo run -p bulletou-cuda-train --features cuda,root-loader -- --nnue-teacher-train ...` in the nested `cuda-oxide` workspace and forwards teacher/output/batch/loader/loss/save/checkpoint settings.
- Unsupported production semantics (`--superbatches`, `--max-epochs`, LR schedule, validation metrics, SFNN/KPPT families, and `--no-resume`) fail fast and remain assigned to later tickets.
- Validation: `cargo test --example bulletou cuda_oxide_backend` passed.
- Validation: WSL CUDA smoke through `examples/bulletou --backend cuda-oxide` consumed one real HCPE batch and wrote `0001/nn.bin`, `state.boung`, `state.bin`, `dataloader_pos.txt`, `learn.log`, `summary-learn.log`, and `tag.txt`.

### BO-CUDA-004

- Added cuda-oxide `--test-teacher`, `--test-positions`, `--test-batch-size`, and `--test-seed`.
- Checkpoint-time validation reads trained weights back, runs CPU fast HalfKP NNUE forward on sampled HCPE positions, and computes `test_value_accuracy` / `test_value_loss` with the same helper used by the Bullet backend.
- cuda-oxide `learn.log` now uses the production per-save CSV schema: `eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher`.
- cuda-oxide `summary-learn.log` now uses the production top-level CSV schema without `curr_batch`.
- Validation: WSL CUDA smoke with 8 sampled test positions wrote `test_value_accuracy=0.625000`, `test_value_loss=0.051297`, and `train_value_loss=0.020710541`.

## In-progress notes

### BO-CUDA-003

- Added bounded production schedule mode for `--backend cuda-oxide`: `--superbatches N --max-epochs N` maps to direct cuda-oxide train steps.
- Added cuda-oxide trainer flags for `--batches-per-superbatch`, `--lr-schedule fixed|step|geometric|cos`, LR period/step parameters, and Ranger hyperparameters.
- `--save-rate` is interpreted as superbatch units when BulletOu passes `--batches-per-superbatch`; direct smoke mode keeps the old batch-unit behavior via `batches_per_superbatch=1`.
- Plateau remains fail-fast until BO-CUDA-004 validation metrics are available.
- Validation: WSL CUDA smoke with `--superbatches 2 --positions-per-superbatch 2 --save-rate 2 --lr-schedule cos` ran two real HCPE batches, reported `lr_start=0.01`, `lr_last=0.0055`, and wrote one checkpoint.
