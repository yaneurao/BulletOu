# 10. cuda-oxide implementation tickets

This is the active ticket queue for turning the current cuda-oxide smoke/bridge
work into a production BulletOu training backend. Work the tickets in order and
commit each completed slice.

| ticket | status | scope | exit criteria |
|---|---|---|---|
| BO-CUDA-001 | done | cuda-oxide resume from root `state.bin` | `--nnue-teacher-train` can restore weights + Ranger optimizer state from root-format `state.bin`, not only `state.boung`; smoke verifies the same next-step result as `state.boung` resume |
| BO-CUDA-002 | todo | promote direct cuda-oxide loop into end-user BulletOu CLI | `examples/bulletou.rs` exposes an opt-in cuda-oxide NNUE HalfKP training path that writes the normal numbered checkpoint layout |
| BO-CUDA-003 | todo | production schedule integration | cuda-oxide path honors `--superbatches`, epoch boundaries, LR schedule, `--save-rate`, positions carry-over, and plateau control in the same user-facing sense as the Bullet backend |
| BO-CUDA-004 | todo | validation metrics integration | cuda-oxide checkpoints write production-compatible `learn.log` / `summary-learn.log` columns including `test_value_accuracy`, `test_value_loss`, and `train_value_loss` |
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
