# 10. cuda-oxide implementation tickets

This is the active ticket queue for turning the current cuda-oxide smoke/bridge
work into a production BulletOu training backend. Work the tickets in order and
commit each completed slice.

| ticket | status | scope | exit criteria |
|---|---|---|---|
| BO-CUDA-001 | done | cuda-oxide resume from root `state.bin` | `--nnue-teacher-train` can restore weights + Ranger optimizer state from root-format `state.bin`, not only `state.boung`; smoke verifies the same next-step result as `state.boung` resume |
| BO-CUDA-002 | done | promote direct cuda-oxide loop into end-user BulletOu CLI | `examples/bulletou.rs` exposes an opt-in cuda-oxide NNUE HalfKP training path that writes the normal numbered checkpoint layout |
| BO-CUDA-003 | done | production schedule integration | cuda-oxide path honors `--superbatches`, epoch boundaries, LR schedule, `--save-rate`, positions carry-over, and plateau control in the same user-facing sense as the Bullet backend |
| BO-CUDA-004 | done | validation metrics integration | cuda-oxide checkpoints write production-compatible `learn.log` / `summary-learn.log` columns including `test_value_accuracy`, `test_value_loss`, and `train_value_loss` |
| BO-CUDA-005 | done | dataloader resume generalisation | HCPE3, shogipack, multi-teacher specs, and teacher changes have explicit resume behavior and smoke coverage |
| BO-CUDA-006 | done | async input/readback rings | input upload and loss readback are pipelined without changing fp32 baseline results |
| BO-CUDA-007 | done | speed benchmark | same teacher / seed / schedule benchmark compares Bullet backend vs cuda-oxide positions/sec |
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

### BO-CUDA-003

- Added bounded production schedule mode for `--backend cuda-oxide`: `--superbatches N --max-epochs N` maps to direct cuda-oxide train steps.
- Added cuda-oxide trainer flags for `--batches-per-superbatch`, `--lr-schedule fixed|step|geometric|cos`, LR period/step parameters, and Ranger hyperparameters.
- `--save-rate` is interpreted as superbatch units when BulletOu passes `--batches-per-superbatch`; direct smoke mode keeps the old batch-unit behavior via `batches_per_superbatch=1`.
- Added BulletOu-side plateau orchestration for cuda-oxide: one fixed-LR child run per superbatch, validation-metric monitoring, rejected checkpoint removal, summary-log row trimming, and same-superbatch retry at the lowered LR.
- Validation: WSL CUDA smoke with `--superbatches 2 --positions-per-superbatch 2 --save-rate 2 --lr-schedule cos` ran two real HCPE batches, reported `lr_start=0.01`, `lr_last=0.0055`, and wrote one checkpoint.
- Validation: WSL CUDA plateau reject smoke forced a no-improvement second superbatch, removed rejected `0002`, retried the same HCPE byte offset at `lr_min=0.005`, trimmed `summary-learn.log` back to the accepted row, and wrote `0001/plateau_epoch_done.txt`.

### BO-CUDA-004

- Added cuda-oxide `--test-teacher`, `--test-positions`, `--test-batch-size`, and `--test-seed`.
- Checkpoint-time validation reads trained weights back, runs CPU fast HalfKP NNUE forward on sampled HCPE positions, and computes `test_value_accuracy` / `test_value_loss` with the same helper used by the Bullet backend.
- cuda-oxide `learn.log` now uses the production per-save CSV schema: `eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher`.
- cuda-oxide `summary-learn.log` now uses the production top-level CSV schema without `curr_batch`.
- Validation: WSL CUDA smoke with 8 sampled test positions wrote `test_value_accuracy=0.625000`, `test_value_loss=0.051297`, and `train_value_loss=0.020710541`.

### BO-CUDA-005

- Replaced the HCPE-only resume offset with `TeacherDataloaderPos { byte_offset, plies }`, carried through `HalfkpTeacherBatch` and cuda-oxide bridge checkpoints.
- `teacher.txt` is now written next to `dataloader_pos.txt`; auto-resume uses loader positions only when the stored teacher spec matches the current teacher. If the teacher changes, weights/optimizer state still resume but the teacher stream starts at batch 0.
- HCPE resume keeps exact fixed-record offsets (`batch_size * 38` per consumed batch), while HCPE3 and shogipack carry `(byte_offset, plies)` from their loaders.
- Shogipack buffering now attaches the resume position to each expanded PSV, so small batch boundaries save the correct `plies` rather than the end of the whole game.
- Validation:
  - `cargo test -p bulletou_lib teacher_batch -- --nocapture`
  - `cargo check --example export_nnue_forward_fixture`
  - `cargo check -p bulletou-cuda-train`
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`
  - WSL: `cargo test -p bulletou-cuda-train --features cuda,root-loader dataloader_pos -- --nocapture`
  - WSL CUDA HCPE smoke: `shuffled-001.hcpe` wrote `0001/dataloader_pos.txt = 76,0`; same-teacher resume printed `resume_data byte_offset=76, plies=0` and wrote `152,0`.
  - WSL CUDA teacher-change smoke: switching to `shuffled-002.hcpe` resumed `state.boung` but printed `teacher changed; starting teacher stream at batch 0` and wrote the new `teacher.txt`.
  - WSL CUDA HCPE3 smoke: `arch000073330000.hcpe3` wrote `0,2`; same-teacher resume printed `resume_data byte_offset=0, plies=2` and wrote `0,4`.
  - WSL CUDA shogipack smoke: synthetic `bo005-tiny.pack` wrote `0,2`; same-teacher resume printed `resume_data byte_offset=0, plies=2` and wrote `0,4`. No non-git `.pack` teacher file was present under the searched local teacher/work directories.

### BO-CUDA-006

- Reworked the NNUE loss/Ranger step runner around a two-slot ring. Each slot owns its device batch/workspaces, pinned host upload buffers, pinned host loss readback buffers, and CUDA events for upload, compute, and readback lifetime tracking.
- Added an upload stream and readback stream alongside the compute stream. Async steps copy batch inputs from pinned host buffers on the upload stream, make compute wait on the upload event, then copy loss readbacks to pinned host buffers on the readback stream after the compute stream records the loss-ready event.
- The default final-only teacher train path (`--save-rate 0`) uses the async pipeline and drains the final pending readback before writing outputs. Periodic checkpoint mode remains synchronized so checkpoint state, logs, and save boundaries stay aligned.
- Validation:
  - `cargo check -p bulletou-cuda-train`
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`
  - WSL: `cargo test -p bulletou-cuda-train --features cuda,root-loader dataloader_pos -- --nocapture`
  - WSL CUDA async final-output smoke on `shuffled-001.hcpe` with `--train-steps 2 --batch-size 2` wrote `0001/{nn.bin,state.boung,state.bin,teacher.txt,dataloader_pos.txt,learn.log,trained-forward.nnuef}`, `summary-learn.log`, and `dataloader_pos.txt = 152,0`.
  - WSL CUDA async-vs-sync baseline smoke on the same teacher produced identical losses: step1 `weighted_sum=0.4999314 mean=0.2499657`, step2 `weighted_sum=0.013783315 mean=0.0068916576`.

### BO-CUDA-007

- Added a cuda-oxide NNUE teacher-train throughput line: `throughput : positions=... time=...s pos/sec=...`.
- The timer starts when the first prepared training batch is submitted, so it excludes initial teacher prefill in the same spirit as the Bullet backend superbatch timer, and includes the final async loss drain.
- Validation:
  - `cargo check -p bulletou-cuda-train`
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`
  - WSL CUDA throughput smoke on `shuffled-001.hcpe`: `--train-steps 8 --batches-per-superbatch 8 --batch-size 64` printed `throughput : positions=512 time=1.449s pos/sec=353`.
  - Same-teacher/same-schedule debug benchmark on RTX 4090, fixed 1-superbatch step schedule (`lr=0.01`, `lr_min=0.01`, `lr_step_gamma=1`, Ranger `weight_decay=0.01 beta1=0.99 beta2=0.999 epsilon=1e-8`, `loader_threads=1`, `threads=1`):
    - 512 positions (`batch_size=64`, 8 batches): Bullet backend `2130 pos/sec`; cuda-oxide `353 pos/sec`.
    - 8192 positions (`batch_size=1024`, 8 batches): Bullet backend `18109 pos/sec`; cuda-oxide `1037 pos/sec`.
  - These are debug-build smoke numbers for regression tracking, not final tuned production throughput.
