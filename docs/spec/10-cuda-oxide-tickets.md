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
| BO-CUDA-008 | done | SFNN training integration | SFNN cuda-oxide training path can stream real teacher batches and write compatible checkpoints |
| BO-CUDA-009 | done | expose SFNN cuda-oxide through BulletOu CLI | `examples/bulletou --backend cuda-oxide --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3` launches the SFNN cuda-oxide child trainer and writes the normal output layout |
| BO-CUDA-010 | done | SFNN validation metrics | `--sfnn-teacher-train --test-teacher` and the BulletOu SFNN cuda-oxide wrapper write `test_value_accuracy` / `test_value_loss` into the normal logs |
| BO-CUDA-011 | done | SFNN production schedule / periodic checkpoints | SFNN cuda-oxide child honors `--save-rate` and the BulletOu wrapper accepts bounded `--superbatches` / `--max-epochs` direct production schedule |
| BO-CUDA-012 | done | SFNN resume from root `state.bin` | `--sfnn-teacher-train` can auto-resume weights, Ranger state, completed step count, and teacher dataloader position from SFNN bridge checkpoints |
| BO-CUDA-013 | done | SFNN plateau schedule | BulletOu `--backend cuda-oxide --eval-type SFNN_HALFKA2 --lr-schedule plateau` runs through the generic plateau orchestrator using SFNN validation metrics and auto-resume |
| BO-CUDA-014 | done | SFNN factorized L1 forward foundation | cuda-oxide SFNN forward/runtime/fixture paths can carry optional shared `l1f` weights and match CPU golden output before training-backward integration |
| BO-CUDA-015 | done | SFNN factorized L1 backward/Ranger smoke | cuda-oxide SFNN backward/runtime/optimizer paths can compute and update optional shared `l1f` weights, with `factorized-tiny` backward and Ranger-step smokes matching CPU golden |
| BO-CUDA-016 | done | SFNN factorized L1 production integration | `--sfnn-teacher-train --sfnn-factorized-l1` and BulletOu `--backend cuda-oxide --sfnn-factorized-l1` can train, checkpoint, resume, validate, and save folded `nn.bin` with shared `l1f` state preserved in root `state.bin` |
| BO-CUDA-017 | done | tatara parity data bridge | BulletOu can export HCPE/HCPE3/pack/PSV teachers to flat PSV for tatara, and BulletOu validation accepts PSV held-out data so both trainers can consume the same positions |
| BO-CUDA-018 | done | tatara parity benchmark harness | run standard NNUE HalfKP BulletOu cuda-oxide and tatara on the same exported PSV teacher/test slices, collect comparable train throughput and held-out accuracy/loss, and record any remaining loss/schedule mismatches |
| BO-CUDA-019 | done | NNUE L0 sparse backward scatter optimization | larger same-PSV benchmark exposed the dense gather L0 backward bottleneck; BulletOu now zeroes L0 gradients and atomic-scatters active feature gradients, preserving correctness while greatly improving standard NNUE throughput |
| BO-CUDA-020 | todo | remaining tatara speed/accuracy parity | continue standard NNUE HalfKP profiling/tuning from the post-BO-CUDA-019 baseline, close the remaining speed gap to tatara, and verify held-out accuracy over a longer same-PSV run |

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

### BO-CUDA-008

- Added a host-side SFNN loss/Ranger train-step runner for cuda-oxide.
- Added `bulletou-cuda-train --sfnn-teacher-train` for the fixed HalfKA2 / `SFNN_halfka2_1024_7_64_k3k3` path.
- The SFNN teacher path streams real `ShogiHalfKa2` + `ShogiLayerStackBucket9::KingRank9` batches from `bulletou_lib`, supports deterministic initial weights or `--weights-bin`, and feeds batches directly to the SFNN forward/loss/backward/Ranger kernels.
- `--output` writes numbered bridge checkpoints with YaneuraOu-compatible `nn.bin`, root-format `state.bin` under the usual `nnue/*` component records, `teacher.txt`, `dataloader_pos.txt`, `learn.log`, and `summary-learn.log`.
- Validation:
  - `cargo check -p bulletou-cuda-train` from the nested `cuda-oxide` workspace.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - `cargo test -p bulletou_lib teacher_batch -- --nocapture`.
  - WSL: `cargo test -p bulletou-cuda-train --features cuda,root-loader dataloader_pos -- --nocapture`.
  - WSL CUDA smoke on `shuffled-001.hcpe`: `--sfnn-teacher-train --train-steps 1 --batch-size 1 --output /tmp/bo008-sfnn-output-smoke` streamed one real HCPE batch, printed `step1_loss weighted_sum=0.25801346 mean=0.25801346`, and wrote `0001/nn.bin` (129 MiB), `0001/state.bin` (2.1 GiB), `teacher.txt`, `dataloader_pos.txt = 38,0`, `learn.log`, and top-level `summary-learn.log`.
  - The smoke `nn.bin` advertised `ModelType=SFNNWithoutPsqt;Features=HalfKA2(Friend)[131949->1024x2],Network=SFNN-1024{LayerStack=9}`, and `state.bin` contained `nnue/weights/l0w` plus `nnue/step_ranger/l3w` records.

### BO-CUDA-009

- Extended `examples/bulletou --backend cuda-oxide` to accept `--eval-type SFNN_HALFKA2` with the fixed cuda-oxide-supported architecture `--arch SFNN_halfka2_1024_7_64_k3k3`.
- The BulletOu wrapper dispatches SFNN runs to the nested `bulletou-cuda-train --sfnn-teacher-train` child.
- Other SFNN families and the default `SFNN_halfka2_1536_15_32_k3k3` architecture remain fail-fast until the corresponding SFNN child features are implemented.
- Validation:
  - `cargo test --example bulletou cuda_oxide_backend -- --nocapture`.
  - `cargo check --example bulletou`.
  - `cargo check -p bulletou_lib --example bulletou`.
  - WSL CUDA smoke through `examples/bulletou --backend cuda-oxide --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3` on `shuffled-001.hcpe` streamed one real HCPE batch, launched the nested child with `--sfnn-teacher-train --save-rate 0`, wrote `0001/nn.bin`, `0001/state.bin`, `teacher.txt`, `dataloader_pos.txt = 38,0`, `learn.log`, and top-level `summary-learn.log`, and reported `step1_loss weighted_sum=0.25801346 mean=0.25801346`.

### BO-CUDA-010

- Added SFNN checkpoint-time validation for `bulletou-cuda-train --sfnn-teacher-train --test-teacher`.
- Validation reads the trained SFNN state back, runs CPU fast HalfKA2 + `ShogiLayerStackBucket9::KingRank9` forward on sampled HCPE positions, and computes `test_value_accuracy` / `test_value_loss` with the same sign/loss helper used by the NNUE bridge path.
- The BulletOu `--backend cuda-oxide --eval-type SFNN_HALFKA2` wrapper now allows `--test-teacher` in direct SFNN mode and forwards the validation flags to the nested child trainer.
- Validation:
  - `cargo test --example bulletou cuda_oxide_backend -- --nocapture`.
  - `cargo check -p bulletou-cuda-train` from the nested `cuda-oxide` workspace.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - `cargo check --example bulletou`.
  - WSL CUDA smoke on `shuffled-001.hcpe`: `--sfnn-teacher-train --test-teacher shuffled-001.hcpe --test-positions 8 --test-batch-size 4 --train-steps 1 --batch-size 1` wrote `test_value_accuracy=0.375000`, `test_value_loss=0.111235`, and `train_value_loss=0.258013457` to both `0001/learn.log` and top-level `summary-learn.log`.

### BO-CUDA-011

- Removed the SFNN child trainer's final-checkpoint-only restriction. `--sfnn-teacher-train` now uses the same `--save-rate * --batches-per-superbatch` interval rule as the NNUE bridge path.
- Periodic SFNN checkpoint writes include `nn.bin`, root `state.bin`, `teacher.txt`, `dataloader_pos.txt`, checkpoint-local `learn.log`, and top-level `summary-learn.log`; validation metrics are computed per saved checkpoint when `--test-teacher` is present.
- The BulletOu wrapper now forwards SFNN `--save-rate` unchanged and accepts bounded non-plateau production schedule mode (`--superbatches N --max-epochs N`). Plateau scheduling is enabled by BO-CUDA-013.
- Validation:
  - `cargo test --example bulletou cuda_oxide_backend -- --nocapture`.
  - `cargo check -p bulletou-cuda-train` from the nested `cuda-oxide` workspace.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - `cargo check --example bulletou`.
  - WSL CUDA periodic smoke on `shuffled-001.hcpe`: `--sfnn-teacher-train --train-steps 2 --batches-per-superbatch 1 --batch-size 1 --save-rate 1 --test-positions 4` wrote both `0001` and `0002`, with `dataloader_pos.txt = 38,0` then `76,0`; `summary-learn.log` had two SFNN rows with train losses `0.258013457` and `0.241245091`.
  - WSL CUDA wrapper smoke through `examples/bulletou --backend cuda-oxide --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3 --superbatches 1 --max-epochs 1 --positions-per-superbatch 1 --batch-size 1 --save-rate 1` launched the nested child in production schedule mode with `--superbatches-per-epoch 1` and wrote checkpoint `0001`.

### BO-CUDA-012

- Added SFNN root `state.bin` restore for the same `nnue/{weights,momentum,velocity,slow,step_ranger}/...` component layout written by SFNN bridge checkpoints.
- Added SFNN runner construction from restored weights plus Ranger momentum/velocity/slow state, so later runs continue optimizer state rather than only loading weights.
- `--sfnn-teacher-train --output` now auto-resumes the latest numbered SFNN checkpoint; when `teacher.txt` matches, it also resumes `dataloader_pos.txt`. Explicit root-state restore is available through the existing `--nnue-train-state-bin <PATH>` option.
- Validation:
  - `cargo check -p bulletou-cuda-oxide-runtime` from the nested `cuda-oxide` workspace.
  - `cargo check -p bulletou-cuda-train` from the nested `cuda-oxide` workspace.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - WSL CUDA resume smoke on `shuffled-001.hcpe`: first `--sfnn-teacher-train --train-steps 1 --save-rate 1 --output /tmp/bo012-sfnn-resume-smoke` wrote `0001` with `dataloader_pos.txt = 38,0`; the second identical run printed `resume_state : .../0001/state.bin`, `resume_data : byte_offset=38, plies=0`, `start_step : 2`, consumed teacher batch 1, wrote `0002/dataloader_pos.txt = 76,0`, and produced `step2_loss weighted_sum=0.24124509 mean=0.24124509`.

### BO-CUDA-013

- Removed the SFNN plateau fail-fast in `examples/bulletou --backend cuda-oxide`.
- The existing cuda-oxide plateau orchestrator now works for `SFNN_HALFKA2`: each plateau superbatch launches the SFNN child trainer with fixed LR, `--save-rate 1`, validation flags, and the SFNN trainer's checkpoint auto-resume from BO-CUDA-012.
- Validation:
  - `cargo test --example bulletou cuda_oxide_backend -- --nocapture`.
  - `cargo check --example bulletou`.
  - WSL CUDA plateau smoke through `examples/bulletou --backend cuda-oxide --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3 --lr-schedule plateau --superbatches 1 --max-epochs 1 --positions-per-superbatch 1 --batch-size 1 --save-rate 1 --test-teacher shuffled-001.hcpe --test-positions 4 --test-batch-size 4` launched the nested SFNN child with `--lr-schedule fixed --learning-rate 0.000875`, wrote `0001/dataloader_pos.txt = 38,0`, wrote validation metrics `test_value_accuracy=0.666667` and `test_value_loss=0.138324`, and the plateau orchestrator printed `initial validation metrics = loss=0.138324, accuracy=0.666667`.

### BO-CUDA-014

- Extended SFNN forward weight layout / host-device weight ownership with optional shared factorized L1 weights (`l1fw`, `l1fb`).
- Added CUDA kernel `sfnn_shared_l1_add`, launched after stacked L1 and before pairwise L2 input assembly when shared L1 weights are present.
- Added synthetic `factorized-tiny` SFNN forward case and fixture read/write support for optional shared L1 payloads while preserving old fixtures without that trailer.
- Backward and Ranger smoke paths now reject factorized L1 cases explicitly until training-backward integration is wired.
- Validation:
  - `cargo check -p bulletou-cuda-oxide-runtime`.
  - `cargo check -p bulletou-cuda-train`.
  - `cargo test -p bulletou-cuda-oxide-runtime sfnn`.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - WSL: `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release`.
  - WSL CUDA smoke used a local cubin produced from `bulletou_cuda_train.ll` via `llvm-link-20` + libdevice, `llc-20`, and `ptxas`, avoiding the known WSL `nvJitLinkCreate` symbol issue on the `.ll` runtime load path.
  - WSL CUDA `--sfnn-forward-smoke --sfnn-forward-case factorized-tiny --debug-readback` matched CPU golden: output max_abs diff `0.000000029802322`; `l1` max_abs diff `0.000000007450581`; compare `ok`.
  - WSL CUDA regression `--sfnn-forward-smoke --sfnn-forward-case tiny --debug-readback` matched CPU golden with all reported max_abs diffs `0`.
  - WSL CUDA fixture round-trip of `factorized-tiny` via `--write-sfnn-forward-fixture` then `--sfnn-forward-fixture` matched CPU golden with output max_abs diff `0.000000029802322`.

### BO-CUDA-015

- Added runtime layout and CUDA kernel `sfnn_shared_l1_backward`.
- The shared L1 backward pass computes `l1fw` / `l1fb` gradients and adds the shared-L1 contribution into `combined_gradients` before pairwise/L0 backward.
- Extended SFNN backward workspace with `l1fw_gradients` / `l1fb_gradients`.
- Extended SFNN Ranger optimizer state/update paths with optional `l1fw` / `l1fb` parameter groups, present only when the forward weights carry shared L1 weights.
- Enabled `factorized-tiny` for SFNN dense-backward and Ranger-step smokes.
- Validation:
  - `cargo check -p bulletou-cuda-train`.
  - `cargo test -p bulletou-cuda-oxide-runtime sfnn`.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - WSL: `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release`.
  - WSL CUDA smoke used a local cubin produced from `bulletou_cuda_train.ll` via `llvm-link-20` + libdevice, `llc-20`, and `ptxas`.
  - WSL CUDA `--sfnn-dense-backward-smoke --sfnn-forward-case factorized-tiny` matched CPU golden: `l1fw_grad` max_abs diff `0.000000014901161`; `l1fb_grad` max_abs diff `0.000000059604645`; compare `ok`.
  - WSL CUDA `--sfnn-ranger-step-smoke --sfnn-forward-case factorized-tiny` matched CPU golden including `l1fw_*` and `l1fb_*` weight/momentum/velocity/slow buffers; compare `ok`.
  - WSL CUDA regressions `--sfnn-dense-backward-smoke --sfnn-forward-case tiny` and `--sfnn-ranger-step-smoke --sfnn-forward-case tiny` both matched CPU golden; compare `ok`.

### BO-CUDA-016

- Added `--sfnn-factorized-l1` to `bulletou-cuda-train --sfnn-teacher-train`.
- New SFNN cuda-oxide teacher runs can zero-initialize optional shared L1 weights (`l1fw`, `l1fb`) and optimizer state.
- Root `state.bin` write/read now preserves optional `nnue/{weights,momentum,velocity,slow,step_ranger}/l1fw` and `l1fb` records; resuming a factorized state keeps the shared L1 path even if the new invocation omits the initialization flag.
- SFNN validation folds `l1fw/l1fb` into the per-bucket `l1w/l1b` CPU fast-forward view, matching the `nn.bin` save semantics.
- SFNN `nn.bin` saving now passes `factorized_l1=true` when shared L1 state is present, so the saved YaneuraOu-compatible weights fold the shared term into every bucket.
- The BulletOu wrapper no longer rejects `--backend cuda-oxide --eval-type SFNN_HALFKA2 --sfnn-factorized-l1`; it forwards `--sfnn-factorized-l1` to the nested child trainer.
- Validation:
  - `cargo check -p bulletou-cuda-train`.
  - `cargo test --example bulletou cuda_oxide_backend`.
  - `cargo test -p bulletou-cuda-oxide-runtime sfnn`.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - WSL CUDA `bulletou-cuda-train --sfnn-teacher-train --sfnn-factorized-l1` on `shuffled-001.hcpe` ran one real HCPE batch, wrote `0001/nn.bin` and `0001/state.bin`, printed `l1_factor : enabled`, ran SFNN validation on 4 held-out positions, and `state.bin` contained `l1fw/l1fb` records for weights, momentum, velocity, slow, and step.
  - WSL CUDA resume smoke from that checkpoint, without passing `--sfnn-factorized-l1`, restored `0001/state.bin`, resumed data at `byte_offset=38, plies=0`, printed `l1_factor : enabled`, wrote `0002`, and preserved `l1fw/l1fb` records.
  - WSL CUDA wrapper smoke through `examples/bulletou --backend cuda-oxide --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3 --sfnn-factorized-l1` ran one real HCPE batch, launched the nested child with `--sfnn-factorized-l1`, wrote `0001/nn.bin` and `0001/state.bin`, and preserved `l1fw/l1fb` records.

### BO-CUDA-017

- Added `export_teacher_psv`, a BulletOu example utility that resolves the normal teacher spec syntax (`.hcpe` / `.hcpe3` / `.pack` / `.psv`, directory, or comma-separated list) and writes the same decoded `PackedSfenValue` stream as flat 40-byte `.psv`.
- The exporter supports `--positions` and `--start-position` caps for smoke/parity slices, plus HCPE decode controls (`--buffer-mb`, `--loader-threads`). Existing HCPE/HCPE3/pack loaders are reused, so tatara can train on the same decoded positions without modifying tatara.
- Added `read_random_teacher_positions` for held-out validation. It accepts fixed-record `.hcpe` and `.psv` teacher specs and keeps the old `read_random_hcpe_positions` wrapper for compatibility.
- BulletOu CPU validation and the cuda-oxide NNUE/SFNN validation cache now use `read_random_teacher_positions`, so `--test-teacher` can be the exported PSV file used by tatara.
- Validation:
  - `cargo check -p bulletou_lib --example export_teacher_psv`.
  - `cargo test -p bulletou_lib validate::tests`.
  - `cargo check --example bulletou`.
  - `cargo check -p bulletou-cuda-train`.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - Export smoke: `shuffled-001.hcpe` -> `target/tatara-parity/teacher-128.psv` and `yamaoka-floodgate.hcpe` -> `target/tatara-parity/yamaoka-128.psv`, both `128 * 40 = 5120` bytes.
  - WSL CUDA NNUE smoke on the exported PSV teacher/test slice loaded `32` PSV validation positions and completed one training batch (`step1_loss mean=0.09178529`).
  - WSL CUDA NNUE checkpoint smoke with `--save-rate 1` wrote `summary-learn.log` from the PSV validation slice with `test_value_accuracy=0.468750` and `test_value_loss=0.118148`.

### BO-CUDA-018

- Added `scripts/tatara_parity_smoke.ps1`, a repeatable Windows/WSL harness that:
  - exports configurable train/test PSV slices from `C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe` and `C:\shogi\teacher\test\yamaoka-floodgate.hcpe`;
  - runs tatara `nnue-train` on the exported PSV with `simple --arch 256x2-32-32`, `--feature-set halfkp`, and a WRM profile matching BulletOu's current `--loss-kind wrm` constants (`scale=600`, prediction `offset/scaling=270/340`, target `offset/scaling=270/380`, `nnue2score=600`, `loss-pow-exp=2.5`);
  - runs BulletOu cuda-oxide standard NNUE HalfKP on the same train/test PSV slice;
  - separates the BulletOu speed smoke from the checkpoint/validation smoke, so checkpoint serialization overhead does not pollute the primary throughput line.
- The harness can build tatara's PTX via `-BuildTataraKernel` using the current WSL CUDA/LLVM-20 route when `tatara\nnue_train.ptx` is absent.
- Validation command:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\tatara_parity_smoke.ps1 -TrainPositions 128 -TestPositions 128 -BatchSize 64 -BatchesPerSuperbatch 1 -Superbatches 1`.
- Validation result from `target\tatara-parity\parity-20260718-062118`:
  - PSV export wrote `teacher-128.psv` and `test-128.psv`, both `128 * 40 = 5120` bytes.
  - tatara: train `loss=0.091041`, toy throughput line `1825 pos/s`, held-out `test_loss=0.117308`, `test_acc=0.5000`.
  - BulletOu speed smoke: train `step1_loss mean=0.09178529`, throughput line `556 pos/s`.
  - BulletOu metrics smoke: held-out `accuracy=50.0000%`, `loss=0.116964`; the printed `pos/sec=1` is expected checkpoint/write overhead and is intentionally not used as the speed comparison.

### BO-CUDA-019

- Fixed `scripts/tatara_parity_smoke.ps1` so BulletOu receives `--train-steps Superbatches*BatchesPerSuperbatch`. The first larger run revealed the harness was only advancing one BulletOu batch when `BatchesPerSuperbatch > 1`.
- Added `-BuildBulletKernel` to the parity harness. The atomic L0 scatter kernel needs the current WSL `sm_100` NVVM IR route (`cargo-oxide build --emit-nvvm-ir --arch sm_100`, then `llvm-link-20` + `opt-20` + `llc-20 --mcpu=sm_89` + `ptxas`) because direct legacy `cargo-oxide build --arch sm_89` cannot lower cuda-oxide atomic RMW yet.
- Replaced NNUE `nnue_l0_sparse_backward`'s dense gather algorithm:
  - old shape: one thread per `input_size*l1` weight, scanning every batch sample and sparse slot to find matching active features;
  - new shape: zero `l0w/l0b` gradients, then launch `batch*max_active*l1` scatter threads that `DeviceAtomicF32::fetch_add` active STM/NSTM feature gradients into `l0w`, plus atomic bias accumulation into `l0b`.
- Validation:
  - `rustfmt` on the touched cuda-oxide files.
  - `cargo test -p bulletou-cuda-oxide-runtime backward::tests::nnue_l0_sparse_layout_counts_buffers`.
  - `cargo check -p bulletou-cuda-train`.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - WSL: `cargo-oxide build --emit-nvvm-ir --arch sm_100 --features cuda -- --package bulletou-cuda-train --release`.
  - WSL: rebuilt `target/cuda-oxide-artifacts/bulletou_cuda_train_bo015.cubin` via `llvm-link-20` + `opt-20` + `llc-20 --mcpu=sm_89` + `ptxas`; local SHA-256 `344e1ad4c6e48c94020163726e054c6dc068ce2886cce310701e9d9bd1619a04`.
  - WSL CUDA `--nnue-dense-backward-smoke --nnue-forward-case tiny` matched CPU golden (`l0w_grad` max_abs `0.0000000037252903`, compare `ok`).
  - WSL CUDA `--nnue-ranger-step-smoke --nnue-forward-case tiny` matched CPU golden (compare `ok`).
- Same-PSV parity measurement (`TrainPositions=65536`, `TestPositions=8192`, `BatchSize=8192`, `BatchesPerSuperbatch=8`, `Superbatches=1`):
  - pre-fix BulletOu speed, after correcting harness but before scatter L0: `65536` positions in `92.171s`, `711 pos/s`;
  - post-fix BulletOu speed: `65536` positions in `2.971s`, `22058 pos/s`;
  - tatara speed on the same slice: `104696 pos/s`;
  - tatara held-out: `test_loss=0.070236`, `test_acc=0.5065`;
  - BulletOu held-out: `loss=0.069942`, `accuracy=49.3530%`; the metrics run still includes checkpoint/write overhead (`1209 pos/s`) and is not used for speed comparison.

### BO-CUDA-020

- Remaining parity work:
  - profile the post-scatter baseline (`~22k pos/s`) against tatara (`~105k pos/s`) and identify the next bottleneck;
  - likely candidates are remaining dense backward kernels, full-parameter Ranger update over `l0w`, atomic contention in L0 scatter, and host/batch transfer overlap;
  - run a longer same-PSV accuracy comparison after the next speed pass, since the one-superbatch held-out losses are close but accuracy differs by about `1.3pp`.
