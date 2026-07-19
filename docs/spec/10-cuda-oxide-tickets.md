# 10. cuda-oxide implementation tickets

This started as the active ticket queue for turning the cuda-oxide smoke/bridge
work into a production BulletOu training backend. As of BO-CUDA-031 the production
fast-backend direction is pivoting to a Windows-native C++/CUDA backend while
keeping cuda-oxide available as a reference/experimental implementation. Work
the tickets in order and commit each completed slice.

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
| BO-CUDA-020 | done | NNUE train-step profiling and WRM loss reduction | parity harness now runs host binaries in release mode by default, `--profile-train-step` reports NNUE stage timings, and WRM loss no longer recomputes every sample serially in thread 0 |
| BO-CUDA-021 | done | NNUE teacher prepare/GPU pipeline | `--profile-train-step` now exposes CPU batch materialisation time, standard NNUE teacher training overlaps CPU prepare with GPU work via a bounded producer queue, and the tatara parity harness can run realistic multi-threaded prepare |
| BO-CUDA-022 | done | remaining tatara accuracy parity | validation now reports prediction-sign distribution; the one-superbatch accuracy complement was a short-run all-one-sign prediction artifact, and a 4-superbatch same-PSV run matches tatara accuracy with close WRM loss |
| BO-CUDA-023 | done | YaneuraOu quantized eval cross-check | YaneuraOu `test eval_accuracy` on the cuda-oxide checkpoint `nn.bin` matches BulletOu's f32 checkpoint-time validation on the same held-out PSV |
| BO-CUDA-024 | done | retire stale legacy cuda-oxide TODO rows | synced the older `09-cuda-oxide-todo.md` CO-008..CO-013 summary rows with the completed BO-CUDA implementation slices |
| BO-CUDA-025 | done | final folder-teacher parity audit | reran the standard-NNUE parity harness using the requested teacher directory and validation HCPE, confirming same-PSV tatara/BulletOu accuracy parity and BulletOu speed parity |
| BO-CUDA-026 | done | longer standard NNUE speed benchmark | reran the folder-teacher parity harness on 4M positions and exposed that the remaining gap was not the loader, but tatara's default HalfKP FT factorizer path |
| BO-CUDA-027 | done | tatara-style HalfKP FT factorizer | BulletOu cuda-oxide standard NNUE training now uses tatara's piece-input factorizer for scratch HalfKP runs, folds it when writing `nn.bin`, and matches tatara's 4M same-PSV loss/accuracy/speed envelope |
| BO-CUDA-028 | done | beat tatara on 4M speed and accuracy | reduce cuda-oxide host readback overhead and tune the 4M same-PSV recipe until BulletOu exceeds the latest tatara 4M reference in both held-out accuracy and train throughput |
| BO-CUDA-029 | done | NNUE idle recompare after external GPU load stopped | remeasure tatara/BulletOu on the same 4M PSV slice with the GPU idle and record a BulletOu recipe that exceeds tatara in speed and accuracy |
| BO-CUDA-030 | done | SFNN full-teacher tatara parity | using the full shuffled SFNN teacher for training and only `C:\shogi\teacher\test\yamaoka-floodgate.psv` for validation, the Windows-native C++/CUDA direct path now exceeds tatara in train throughput, held-out loss, and held-out accuracy |
| BO-CUDA-031 | done | Windows-native C++/CUDA backend foundation | add a `bulletou-cuda-cpp` crate that compiles `.cu` with Windows `nvcc`, exposes Rust FFI, runs without WSL, and has a real CUDA smoke plus a Ranger/RAdam update kernel smoke |
| BO-CUDA-032 | done | persistent C++/CUDA device runtime | replace host-copy smoke calls with persistent device buffers, streams, events, async upload slots, and CUDA Graph capture/replay hooks suitable for NNUE/SFNN train steps |
| BO-CUDA-033 | done | port fixed-layout NNUE trainer to C++/CUDA | Windows-native C++/CUDA HalfKP direct training streams real teachers, writes/resumes numbered checkpoints, validates only against the held-out yamaoka PSV, and beats the BO-CUDA-029 tatara idle 4M reference in speed and held-out quality |
| BO-CUDA-034 | done | port fixed-layout SFNN trainer to C++/CUDA | port the SFNN HalfKA2/factorized-L1 train step to C++/CUDA, use only `C:\shogi\teacher\test\yamaoka-floodgate.psv` for validation, and resume the full-teacher tatara comparison from BO-CUDA-030 |
| BO-CUDA-035 | done | cuda-cpp production schedule parity | Windows-native C++/CUDA direct mode accepts bounded `--superbatches` / `--max-epochs`, writes `--save-rate` numbered checkpoints, resumes epoch/superbatch/LR state, and supports step/geometric/cos/plateau schedules without requiring manual `--cuda-cpp-train-steps` sizing |
| BO-CUDA-036 | todo | cuda-cpp HalfKP post-parity optimisation | partial: first-step warmup, direct benchmark timing, CPU/GPU teacher-prepare overlap/profiling, pinned staged upload, sparse L0 zero-gradient atomic skip, HalfKP direct teacher preparation, WRM score-target lookup, teacher single-bucket cleanup, and short/16M speed-quality probes are in; remaining work is further sparse L0/update/feed hot spots plus longer multi-file confirmation against the previous cuda-oxide throughput ceiling |

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

- Updated `scripts/tatara_parity_smoke.ps1` so host binaries run under `cargo run --release` by default. `-DebugHost` restores the old debug-host behavior when needed.
  - This supersedes the BO-CUDA-019 short-run speed line (`~22k pos/s`), which was contaminated by debug host execution.
- Added `--profile-train-step` to `bulletou-cuda-train --nnue-teacher-train`.
  - Profiling disables the async ring for that run and synchronizes after each NNUE stage.
  - Reported stages: `forward`, `loss`, `out_bwd`, `l2_bwd`, `l1_bwd`, `l0_crelu`, `l0_sparse`, `optimizer`.
- Reworked scalar loss kernels:
  - old WRM behavior: one thread per sample wrote per-sample/gradient, but thread 0 recomputed the whole batch loss serially for `weighted_sum` / `mean`;
  - new behavior: one thread per sample writes `per_sample` and `mean_output_gradients`, then `loss_finalize_from_per_sample` sums the already-computed per-sample values in deterministic order.
- Validation:
  - `rustfmt --edition 2024` on touched cuda-oxide files.
  - `cargo test -p bulletou-cuda-oxide-runtime loss::tests`.
  - `cargo check -p bulletou-cuda-train`.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - WSL: `cargo-oxide build --emit-nvvm-ir --arch sm_100 --features cuda -- --package bulletou-cuda-train --release`.
  - WSL: rebuilt `target/cuda-oxide-artifacts/bulletou_cuda_train_bo015.cubin` via `llvm-link-20` + `opt-20` + `llc-20 --mcpu=sm_89` + `ptxas`; local SHA-256 `2a4795651ed46270c12e942d9ffaa7e79056ce2b56fccce54e696f645698814b`.
  - WSL CUDA `--loss-smoke --loss-kind wrm --loss-case weighted` matched CPU golden (compare `ok`).
  - WSL CUDA `--nnue-ranger-step-smoke --nnue-forward-case tiny` matched CPU golden (compare `ok`).
- Stage profile on a PSV `batch_size=8192` teacher batch:
  - before loss split: `loss` was `6.938 ms`;
  - after loss split: `loss` is `0.685 ms`;
  - remaining large stages after this pass: `optimizer` `5.587 ms`, `l1_bwd` `3.967 ms`, `l2_bwd` `2.937 ms`, `forward` `2.343 ms`, `l0_sparse` `1.983 ms`.
- Same-PSV release-host parity measurements (`TrainPositions=65536`, `TestPositions=8192`, `BatchSize=8192`, `BatchesPerSuperbatch=8`, `Superbatches=1`, speed smoke only):
  - pre-loss-split BulletOu release speed: `65536` positions in `0.275s`, `238720 pos/s`;
  - post-loss-split BulletOu release speed: `65536` positions in `0.248s`, `264119 pos/s`;
  - tatara reported `698269 pos/s` on the pre-loss-split run and `468829 pos/s` on the post-loss-split run; these one-superbatch lines are short and noisy, but BulletOu still trails tatara materially.

### BO-CUDA-021

- Extended `HalfkpTeacherBatchConfig` with `profile_prepare`.
  - `bulletou-cuda-train --nnue-teacher-train --profile-train-step` now prints `profile_teacher : ... prepare ... ms` before the GPU stage profile.
  - On the same PSV `batch_size=8192` slice, CPU HalfKP materialisation was the real post-BO-CUDA-020 bottleneck: with `--threads 1`, prepare was about `31-35 ms/batch`, while the GPU stages summed to roughly `7-8 ms/batch`.
  - With `--threads 8`, prepare dropped to about `7-9 ms/batch`.
- Reworked standard NNUE teacher training so non-profile runs use a bounded producer/consumer queue:
  - producer thread streams and materialises teacher batches;
  - consumer thread keeps the existing cuda-oxide async upload/readback ring and GPU step logic;
  - `--profile-train-step` keeps the old serial path so CPU/GPU timing remains easy to read.
- Updated `scripts/tatara_parity_smoke.ps1`:
  - added `-Threads <N>` (default `8`) and passes it to both tatara and BulletOu;
  - passes `--optimizer-weight-decay 0` to BulletOu so the parity run matches tatara's default `weight_decay=0.0`.
- Validation:
  - `cargo check -p bulletou_lib --example export_nnue_forward_fixture`.
  - `cargo check -p bulletou-cuda-train`.
  - WSL: `cargo check -p bulletou-cuda-train --features cuda,root-loader`.
  - WSL release host build: `cargo build --release -p bulletou-cuda-train --features cuda,root-loader`.
- Same-PSV speed measurements (`TrainPositions=65536`, `TestPositions=8192`, `BatchSize=8192`, `BatchesPerSuperbatch=8`, `Superbatches=1`):
  - direct `target/release` BulletOu, `--threads 1`, producer/consumer enabled: `65536` positions in `0.186s`, `352672 pos/s`;
  - parity harness, `-Threads 8`: tatara `520491 pos/s`, BulletOu `489083 pos/s`;
  - warm direct `target/release` BulletOu, `--threads 8`: `65536` positions in `0.085s`, `771281 pos/s`.
  - These one-superbatch speed lines are short and noisy, but BulletOu is now in tatara's observed speed range on this slice.
- Same-PSV BulletOu metrics run with checkpoint/validation overhead excluded from speed comparison:
  - held-out `loss=0.069941`, `accuracy=49.3530%`;
  - tatara on the matching harness run reported `test_loss=0.070236`, `test_acc=0.5065`.

### BO-CUDA-022

- Added validation prediction-sign diagnostics:
  - `AccuracyReport` now tracks decisive-position `pred>=0`, `pred<0`, and exact `zero` counts.
  - NNUE/SFNN cuda-oxide validation logs print those counts next to accuracy/loss. The counters do not change the metric; they make short-run majority-class artifacts visible.
- Root cause of the apparent one-superbatch mismatch:
  - The 8192-position held-out PSV has `Win=4043`, `Loss=4149`, `Draw=0`.
  - The earlier one-superbatch BulletOu metrics run reported `accuracy=49.3530% (4043/8192)`. With diagnostics enabled, the same run showed `pred>=0 8192`, `pred<0 0`, `zero 0`: the model predicted the non-negative side for every held-out position, so the accuracy was exactly the held-out Win ratio.
  - tatara's `test_acc=0.5065` was the complementary Loss ratio in that short run. This was not a PSV result-sign bug; it was a tiny early-training sign bias while WRM loss was already matching.
- Longer same-PSV comparison:
  - Command shape: `TrainPositions=262144`, `TestPositions=8192`, `BatchSize=8192`, `BatchesPerSuperbatch=8`, `Superbatches=4`, `Threads=8`.
  - tatara final report: `test_loss=0.070322`, `test_acc=0.5065`.
  - BulletOu metrics report: `accuracy=50.6470% (4149/8192; pred>=0 0 pred<0 8192 zero 0)`, `loss=0.070590`.
  - BulletOu speed smoke on the same run: `262144` positions in `0.423s`, `619152 pos/s`.

### BO-CUDA-023

- Built YaneuraOu NNUE engine under WSL from `C:\shogi\YaneuraOuWorks\YaneuraOu\source` with the default Makefile settings (`YANEURAOU_ENGINE_NNUE`, `TARGET_CPU=AVX2`, `clang++`).
- Cross-checked the BO-CUDA-022 4-superbatch cuda-oxide checkpoint:
  - `EvalDir=/mnt/c/shogi/YaneuraOuWorks/BulletOu/target/tatara-parity/parity-20260718-070939/bulletou-metrics/0001`
  - test PSV: `/mnt/c/shogi/YaneuraOuWorks/BulletOu/target/tatara-parity/parity-20260718-070939/test-8192.psv`
  - command: `./YaneuraOu-by-gcc EvalDir <checkpoint-dir> , test eval_accuracy <test.psv> , quit`
- Result:
  - YaneuraOu quantized `nn.bin`: `accuracy=50.6470% (4149/8192)`, `drawn=0`, `skipped=0`.
  - BulletOu f32 checkpoint-time validation for the same checkpoint/test set: `accuracy=50.6470% (4149/8192; pred>=0 0 pred<0 8192 zero 0)`.
  - No quantization-induced sign flip was observed on this smoke set.

### BO-CUDA-024

- `docs/spec/09-cuda-oxide-todo.md` was the original low-level cuda-oxide bring-up tracker. After BO-CUDA-001..023, its top summary table still showed CO-008..CO-013 as `todo` even though the corresponding implementation work had landed.
- Updated those legacy rows to `done` and pointed each row at the BO-CUDA slice that completed it:
  - CO-008 loss kernel -> BO-CUDA-020/022;
  - CO-009 backward kernels -> BO-CUDA-019/015/016;
  - CO-010 optimizer kernel/state -> BO-CUDA-012/015/016;
  - CO-011 async rings/pipeline -> BO-CUDA-006/021;
  - CO-012 checkpoint compatibility -> BO-CUDA-004/012/023;
  - CO-013 speed benchmark -> BO-CUDA-018/020/021/022.

### BO-CUDA-025

- Final audit run used the requested source paths directly:
  - teacher: `C:\shogi\teacher\yane-distill-hcpe-20260508shuffled` (directory);
  - validation: `C:\shogi\teacher\test\yamaoka-floodgate.hcpe`;
  - command shape: `scripts\tatara_parity_smoke.ps1 -Teacher <teacher-dir> -TestTeacher <test-hcpe> -TrainPositions 262144 -TestPositions 8192 -BatchSize 8192 -BatchesPerSuperbatch 8 -Superbatches 4 -Threads 8`.
- Export evidence:
  - train export read `input_files=65`, `input_positions=649263458`, wrote `262144` PSV positions;
  - validation export read `input_files=1`, `input_positions=856923`, wrote `8192` PSV positions.
- Same-PSV tatara run:
  - final `test_loss=0.070322`, `test_acc=0.5065`;
  - reported superbatch speed ranged from `621035` to `2633446 pos/s` on the short 4-superbatch run.
- Same-PSV BulletOu cuda-oxide run:
  - speed smoke: `262144` positions in `0.234s`, `1120522 pos/s`;
  - checkpoint-time validation: `accuracy=50.6470% (4149/8192; pred>=0 0 pred<0 8192 zero 0)`, `loss=0.070590`;
  - `summary-learn.log`: `test_value_accuracy=0.506470`, `test_value_loss=0.070590`, `train_value_loss=0.100704312`.
- YaneuraOu quantized cross-check on the same BulletOu checkpoint:
  - `./YaneuraOu-by-gcc EvalDir <bulletou-metrics/0001> , test eval_accuracy <test-8192.psv> , quit`;
  - result: `accuracy=50.6470% (4149/8192)`, `drawn=0`, `skipped=0`.

### BO-CUDA-026

- Re-ran the folder-teacher parity benchmark with a longer training slice because the BO-CUDA-025 speed smoke (`0.234s`) was too short/noisy for a useful speed comparison.
- Harness run:
  - run directory: `target\tatara-parity\parity-20260718-073904`;
  - command shape: `scripts\tatara_parity_smoke.ps1 -Teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled -TestTeacher C:\shogi\teacher\test\yamaoka-floodgate.hcpe -TrainPositions 4194304 -TestPositions 8192 -BatchSize 8192 -BatchesPerSuperbatch 8 -Superbatches 64 -Threads 8`.
- Export evidence:
  - train export read `input_files=65`, `input_positions=649263458`, wrote `4194304` PSV positions (`167772160` bytes);
  - validation export read `input_files=1`, `input_positions=856923`, wrote `8192` PSV positions (`327680` bytes).
- Same-PSV tatara validation run:
  - final report: train `loss=0.054048`, `test_loss=0.054207`, `test_acc=0.6630`;
  - per-superbatch reported training speed over 64 superbatches: min `531384`, median `1021295`, mean `1190192`, max `2214292` pos/s;
  - `done in 12s` includes the external validation pass after every superbatch, so it is not directly comparable with BulletOu's no-output speed smoke.
- Clean speed-only comparison on the already-exported `teacher-4194304.psv`:
  - tatara with held-out validation disabled still wrote its default checkpoints, so its wall-clock `done in 34s` is checkpoint-I/O contaminated; the per-superbatch training-speed statistics were min `508949`, median `1028940.5`, mean `1116150.3`, max `1874347` pos/s;
  - BulletOu cuda-oxide with validation/output disabled processed `4194304` positions in `7.083s`, `592156` pos/s, final `step512_loss mean=0.061553404`;
  - on this longer clean run, BulletOu is roughly `53%` of tatara's mean per-superbatch training throughput (`592156 / 1116150.3`), so standard NNUE cuda-oxide training still trails tatara materially after the async loader and loss-kernel fixes.
- Same-PSV BulletOu checkpoint/validation run:
  - checkpoint-time validation: `accuracy=65.0635% (5330/8192; pred>=0 4139 pred<0 4053 zero 0)`, `loss=0.055417`;
  - `summary-learn.log`: `test_value_accuracy=0.650635`, `test_value_loss=0.055417`, `train_value_loss=0.061433263`;
  - metrics-run throughput (`82659` pos/s) is intentionally not used for speed comparison because it includes checkpoint serialization and validation overhead.

### BO-CUDA-027

- Root cause of the BO-CUDA-026 gap:
  - tatara's default `simple --feature-set halfkp` trains with the HalfKP FT factorizer enabled;
  - BulletOu's cuda-oxide scratch path was training the non-factorized HalfKP input, so the comparison was unintentionally "tatara factorized" vs "BulletOu non-factorized".
- Added a tatara-style `ShogiHalfKPPieceFactorizer`:
  - virtual factor rows are the 1,548 `bona_piece` rows;
  - factorized scratch shape is `input=126936` (`125388 + 1548`), `l1=256`, `l2=32`, `l3=32`;
  - virtual rows are laid out before base HalfKP rows, matching `Factorised::merge_factoriser` folding semantics;
  - cuda-oxide checkpoint validation uses the factorized input when the train state shape is factorized;
  - `nn.bin` export folds the virtual piece rows back into ordinary HalfKP rows.
- Also aligned the standard NNUE Ranger update clamp policy with tatara: the FT layer stays unclamped in fp32 training state, while the later quantized layers use their signed-int8-compatible clamp ranges.
- Controlled 4M same-PSV comparison on `target\tatara-parity\parity-20260718-073904\teacher-4194304.psv`:
  - tatara factorized, `threads=1`: superbatch64 train `loss=0.053733`, held-out `test_loss=0.054000`, `test_acc=0.6655`;
  - tatara non-factorized, `threads=1 --no-ft-factorize`: superbatch64 train `loss=0.056900`, held-out `test_loss=0.055529`, `test_acc=0.6503`;
  - BulletOu non-factorized control: superbatch64 average train `loss=0.056960131`, matching tatara's non-factorized result;
  - BulletOu factorized speed run: `4194304` positions in `3.797s`, `1104772` pos/s, superbatch64 average train `loss=0.053683987`, `step512_loss mean=0.053936914`;
  - BulletOu factorized validation run: `4194304` positions in `3.832s`, `1094486` pos/s, `step512_loss mean=0.053869173`, held-out `test_loss=0.052451`, `test_acc=0.664429`.
- Result: on the 4M same-PSV benchmark, BulletOu is no longer materially behind tatara in either training loss/accuracy or throughput. The previous "BulletOu is roughly 53% of tatara throughput" note applies to the old non-factorized BulletOu run and should not be used as the current status.
- Validation:
  - `cargo fmt -p bulletou_lib`;
  - `cargo fmt --package bulletou-cuda-train`;
  - `cargo check -p bulletou_lib`;
  - `cargo check -p bulletou-cuda-train`;
  - WSL `cargo check -p bulletou-cuda-train --features cuda,root-loader`;
  - WSL 4M factorized speed and checkpoint-validation runs listed above.

### BO-CUDA-028

- Latest tatara reference rerun on the same 4M PSV teacher/test slice:
  - command shape: tatara `simple --arch 256x2-32-32 --feature-set halfkp`, `TrainPositions=4194304`, `TestPositions=8192`, `BatchSize=8192`, `BatchesPerSuperbatch=8`, `Superbatches=64`, `Threads=8`, constant LR `0.01`;
  - log: `target\tatara-parity\parity-20260718-073904\tatara-current-t8.log`;
  - final line: train `loss=0.053627`, `1010364` pos/s, held-out `test_loss=0.054091`, `test_acc=0.6656`;
  - per-superbatch speed: min `736620`, median `1022982.5`, mean `1136996.6`, max `2139663` pos/s.
- Accuracy sweep:
  - BulletOu factorized, `BatchSize=8192`, `Threads=4`, LR `0.012`: held-out `test_acc=0.6677246`, `test_loss=0.053633444`;
  - BulletOu factorized, `BatchSize=8192`, `Threads=8`, LR `0.012`: held-out `test_acc=0.6665039` in a clean final-only readback run;
  - both exceed the tatara reference `test_acc=0.6656`.
- Added `--test-teacher` final validation without requiring `--output`; this avoids writing `state.bin`/`state.boung` just to measure held-out accuracy during sweep runs.
- Added async NNUE loss-readback thinning:
  - no-output async runs now read train loss at superbatch boundaries by default instead of every batch;
  - `--loss-readback-interval <N>` can force a wider interval, e.g. one final loss readback for speed probes;
  - checkpoint/output runs keep the previous all-loss behavior so production logs are not silently weakened.
- Increased the NNUE teacher producer queue depth from the old fixed `2` batches to a configurable `--teacher-queue-depth` defaulting to `8`, so CPU batch materialisation jitter is less likely to starve the GPU during speed probes.
- Reused CPU prepare workers through a Rayon thread pool for HalfKP teacher batch materialisation instead of spawning scoped OS threads for every batch. In the contaminated-GPU 8-step profile, prepare mean improved from `9.620ms` to `8.872ms`; final clean speed still needs to be remeasured after the external GPU job exits.
- Speed measurements before an external GPU load appeared:
  - `BatchSize=8192`, `Threads=8`, LR `0.012`, final-only loss readback: `4194304` positions in `4.143s`, `1012349` pos/s, `test_acc=0.6665039`;
  - this slightly beats tatara's latest final-line speed (`1010364` pos/s) and accuracy (`0.6656`), but not tatara's per-superbatch mean speed (`1136996.6` pos/s).
- Later speed measurements became invalid for final comparison because a separate Windows-side `dlshogi.train ... --gpu 0` Python process was using the RTX 4090 at roughly `86-91%` GPU util / `~320W`, while WSL `nvidia-smi pmon` did not attribute the load to the BulletOu run. Do not use those contaminated `~0.60-0.62M` pos/s BulletOu logs as trainer-regression evidence.
- Current status:
  - accuracy target is met by the LR/thread sweep;
  - speed target is met only against tatara's final-line speed in the clean pre-contention measurement, and still needs a clean rerun after the external dlshogi GPU job exits to prove whether BulletOu also beats the stronger tatara mean-speed criterion.

### BO-CUDA-029

- Re-ran the 4M same-PSV comparison after the external Windows-side GPU worker was stopped. `nvidia-smi` showed no compute process on the RTX 4090 before the reruns.
- Same input slice:
  - train: `target\tatara-parity\parity-20260718-073904\teacher-4194304.psv`;
  - held-out: `target\tatara-parity\parity-20260718-073904\test-8192.psv`.
- Latest tatara idle references, all `BatchSize=8192`, `BatchesPerSuperbatch=8`, `Superbatches=64`, `Threads=8`, constant LR `0.01`:
  - `tatara-recompare-idle-t8.log`: mean speed `2468171` pos/s, final `test_loss=0.053455`, `test_acc=0.6659`;
  - `tatara-recompare-idle-t8-b.log`: mean speed `2576967` pos/s, final `test_loss=0.052763`, `test_acc=0.6635`;
  - `tatara-recompare-idle-t8-c.log`: mean speed `2440299` pos/s, final `test_loss=0.053477`, `test_acc=0.6613`;
  - 3-run mean: speed `2495146` pos/s, `test_acc=0.663567`, `test_loss=0.053232`.
- BulletOu changes validated in this ticket:
  - HalfKP feature extraction now scans the board once instead of iterating piece-type/color buckets repeatedly;
  - NNUE dense hidden-layer backward can use cuBLAS GEMM for the CReLU backward weight-gradient and input-gradient phases;
  - final validation can be run without checkpoint output, and async loss readback can be thinned for speed probes.
- Hyper-parameter result for the short 4M comparison:
  - `BatchSize=16384`, `TrainSteps=256`, `BatchesPerSuperbatch=4`, `Threads=10`, `optimizer_weight_decay=0`, `optimizer_beta1=0.975`, fixed LR `0.024`;
  - `bulletou-16k-final-beta0975-wd0-lr0024-t10-a.log`: `4194304` positions in `1.397s`, `3001693` pos/s, `test_loss=0.05288611`, `test_acc=0.66760254`;
  - `bulletou-16k-final-beta0975-wd0-lr0024-t10-b.log`: `4194304` positions in `1.423s`, `2947515` pos/s, `test_loss=0.052554403`, `test_acc=0.6727295`;
  - `bulletou-16k-final-beta0975-wd0-lr0024-t10-c.log`: `4194304` positions in `1.405s`, `2985953` pos/s, `test_loss=0.053361595`, `test_acc=0.6680908`;
  - 3-run mean: speed `2978387` pos/s, `test_acc=0.669474`, `test_loss=0.052934`.
- Result:
  - BulletOu is `1.19x` faster than the latest tatara idle 3-run mean (`2978387 / 2495146`);
  - BulletOu also beats tatara's best observed idle accuracy in this set (`0.6727295` max, 3-run mean `0.669474`, versus tatara max `0.6659`);
  - the stronger BO-CUDA-028 speed criterion is now met on the 4,194,304-position same-PSV benchmark.
- Recommended reproducible BulletOu speed/accuracy probe:
  - `--train-steps 256 --batch-size 16384 --batches-per-superbatch 4 --threads 10 --loss-kind nnue-pytorch-wrm --optimizer-weight-decay 0 --optimizer-beta1 0.975 --lr 0.024 --lr-min 0 --lr-schedule fixed --loss-readback-interval 64`.

### BO-CUDA-030

- Current SFNN target:
  - train: full exported PSV at `target\full-epoch-sfnn-20260718\teacher-all.psv`;
  - validation must use only `C:\shogi\teacher\test\yamaoka-floodgate.psv` (`C:\shogi\teacher\test\yamaoka-floodgate.hcpe` may be converted to that PSV in-place if needed);
  - do not use the shuffled training teacher as held-out validation data.
- Tatara reference for `SFNN_HALFKA2-SFNN_halfka2_1024_7_64_k3k3`, 1 epoch over the full teacher:
  - `BatchSize=16384`, `Superbatches=4`, `BatchesPerSuperbatch=9907`, default Ranger schedule;
  - final superbatch: about `1.29M` pos/s, `test_loss=0.064964`, `test_acc=0.663361` on the yamaoka PSV validation set.
- BulletOu pre-fold observations:
  - `BatchSize=262144`, LR `0.056`, 1 epoch: about `1.32M` pos/s, but final `test_loss=0.070951`, `test_acc=0.652969`;
  - large batches can beat tatara speed, but not yet the held-out loss/accuracy.
- Added a HalfKA2 factorized-L0 forward folding path for cuda-oxide SFNN training:
  - before each train step, a CUDA kernel materializes `base_feature_row + virtual_piece_row` into a base-shaped L0 forward buffer;
  - the backward path still accumulates gradients in the factorized train-state layout and reduces virtual gradients as before;
  - WSL smoke with nonzero virtual rows: `--sfnn-forward-smoke --sfnn-forward-case halfka2-factorized-nonzero-virtual` passed with max output diff about `7.3e-11`;
  - WSL train-step smoke with nonzero virtual rows: `--sfnn-ranger-step-smoke --sfnn-forward-case halfka2-factorized-nonzero-virtual` passed.
- Short real-data probe with yamaoka validation:
  - command shape: `--sfnn-teacher-train --sfnn-factorized-l1 --teacher teacher-all.psv --test-teacher /mnt/c/shogi/teacher/test/yamaoka-floodgate.psv --test-positions 65536 --test-sample sequential --score-drop-abs 0 --train-steps 192 --batch-size 262144 --teacher-batch-size 262144 --lr 0.056 --lr-schedule fixed`;
  - result: `50331648` positions in `35.303s`, `1425699` pos/s, `test_loss=0.073977835`, `test_acc=0.63442993`;
  - this improves the speed side but quality still needs LR/schedule/batch-size work before the full-epoch tatara target is met.
- Added a training-only fused pairwise/L0 sparse backward path:
  - the smoke/debug path still keeps the separate `sfnn_pairwise_backward` and `sfnn_l0_sparse_backward` kernels so internal `stm_l0_pre`/`nstm_l0_pre` buffers remain testable;
  - real SFNN training now computes pairwise gradients and sparse L0 gradient accumulation in one kernel and skips unused pre-gradient writes;
  - WSL train-step smoke with nonzero virtual rows passed against the CPU golden;
  - bs65k profile: separate pairwise+L0 was about `19.1ms`; fused pairwise/L0 is about `16.4ms`;
  - bs65k/lr0.014 50M yamaoka probe improved throughput from about `1.23M` to `1.31M` pos/s, with `test_loss=0.07200843`, `test_acc=0.6422272`.
- Full-epoch bs65k/lr0.014 before the fused pairwise/L0 change:
  - `649265152` positions in `520.834s`, `1246587` pos/s;
  - yamaoka validation `test_loss=0.072359465`, `test_acc=0.6563568`;
  - quality remained well behind the tatara full-epoch target even though accuracy improved with more data.
- Added small-batch SFNN training refinements:
  - training-only pairwise/L0 sparse backward now skips atomic adds when the CReLU pre-gradient is exactly zero;
  - HalfKA2 factorized-L0 forward folding is now used only for batches at least `65536`, because bs16k profiling showed no-fold forward (`~3.0ms`) was faster than fold+forward (`~3.8ms`);
  - factorized L1 forward/backward can use fused kernels that combine bucket-specific and shared L1 terms in one pass; the factorized-tiny forward and Ranger-step smokes passed against CPU goldens.
- bs16k/50M yamaoka probe after these changes:
  - command shape: `--sfnn-teacher-train --sfnn-factorized-l1 --teacher teacher-all.psv --test-teacher /mnt/c/shogi/teacher/test/yamaoka-floodgate.psv --test-positions 65536 --test-sample sequential --score-drop-abs 0 --batch-size 16384 --teacher-batch-size 131072 --train-steps 3072 --batches-per-superbatch 3072 --optimizer-weight-decay 0 --learning-rate 0.000875 --lr-schedule step`;
  - result: `50331648` positions in `53.940s`, `933104` pos/s, `test_loss=0.073467635`, `test_acc=0.62983704`;
  - this is a modest speed improvement over the earlier bs16k full-superbatch speed line, but still far below the tatara full-epoch speed target; next bottlenecks are the dense RAdam update, sparse L0 backward atomics, and upload/compute pipeline overlap.
- Added a fused SFNN forward L0/pairwise path:
  - the new kernel computes both perspective L0 CReLU row-pairs and writes the pairwise-concat buffer in one launch, replacing the former `stm_l0`, `nstm_l0`, and `pairwise_concat` launches;
  - `--sfnn-forward-smoke --sfnn-forward-case halfka2-factorized-nonzero-virtual` and `--sfnn-ranger-step-smoke --sfnn-forward-case factorized-tiny` passed against CPU goldens with `bulletou_cuda_train_bo042.cubin`;
  - bo042 bs65k/50M yamaoka probe with `beta1=0.9`, `wd=0`, LR `0.014`: `50331648` positions in `35.046s`, `1436154` pos/s, `test_loss=0.07165658`, `test_acc=0.63778687`.
- Full-epoch bs65k yamaoka probes:
  - bo039, `wd=0`, `beta1=0.99`, LR `0.014`: `649265152` positions in `447.735s`, `1450109` pos/s, `test_loss=0.067095526`, `test_acc=0.652298`;
  - bo039, `wd=0`, `beta1=0.9`, LR `0.014`: `649265152` positions in `451.525s`, `1437937` pos/s, `test_loss=0.06694752`, `test_acc=0.6571045`;
  - speed now clears the tatara target, but held-out loss/accuracy are still short of tatara's `0.064964` / `0.663361`.
- Full-epoch bs49k yamaoka probes with bo042:
  - `wd=0`, `beta1=0.9`, LR `0.0105`: `649297920` positions in `480.784s`, `1350499` pos/s, `test_loss=0.06689378`, `test_acc=0.6546936`;
  - `wd=0`, `beta1=0.9`, LR `0.014`: `649297920` positions in `467.789s`, `1388014` pos/s, `test_loss=0.06720718`, `test_acc=0.6560669`;
  - bs49k also clears the speed target, but did not close the quality gap; bs65k/beta1=0.9/LR0.014 remains the best full-epoch comparable BulletOu line so far.
- Full-epoch bs57k yamaoka probe with bo042:
  - `wd=0`, `beta1=0.9`, LR `0.01225`: `649248768` positions in `475.050s`, `1366695` pos/s, `test_loss=0.06690939`, `test_acc=0.65657043`;
  - this also clears the speed target, but quality remains in the same band as bs49k/bs65k and still misses tatara's `0.064964` / `0.663361`.
- Added SFNN train-step upload pipelining:
  - the runner now owns two device-batch/upload slots and a separate upload stream, so the next batch's sparse indices, buckets, targets, and weights can be copied while the current compute stream is still executing;
  - weights, optimizer state, folded L0 buffer, and forward/backward workspaces remain single-stream and are reused in compute-stream order, keeping memory growth small;
  - `cargo check --features cuda,root-loader --package bulletou-cuda-train`, release build, and `--sfnn-ranger-step-smoke --sfnn-forward-case factorized-tiny` pass.
- Upload-pipelined bo043 yamaoka probes:
  - bs40k/50M, `teacher_batch=327680`, `wd=0`, `beta1=0.9`, LR `0.014`: `50339840` positions in `36.600s`, `1375396` pos/s, `test_loss=0.07042214`, `test_acc=0.6411133`;
  - bs40k/full epoch, `teacher_batch=327680`, `wd=0`, `beta1=0.9`, LR `0.014`: `649256960` positions in `436.040s`, `1488985` pos/s, `test_loss=0.06752966`, `test_acc=0.6534729`;
  - bs32k/50M, `teacher_batch=262144`, `wd=0`, `beta1=0.9`, LR `0.014`: `50331648` positions in `39.003s`, `1290471` pos/s, `test_loss=0.07143991`, `test_acc=0.64115906`;
  - bs32k/50M, `teacher_batch=262144`, `wd=0`, `beta1=0.9`, LR `0.020`: `50331648` positions in `38.581s`, `1304582` pos/s, `test_loss=0.07097649`, `test_acc=0.6410217`;
  - bs16k/50M, `teacher_batch=131072`, `wd=0`, `beta1=0.99`, LR `0.000875`: `50331648` positions in `50.208s`, `1002472` pos/s, `test_loss=0.07336029`, `test_acc=0.6298981`;
  - the pipelined bs40k line now has ample speed headroom over tatara, but quality is still short; bs32k is the current boundary candidate for recovering quality while keeping speed close to the tatara line.
- Later full-epoch yamaoka probes before the C++/CUDA pivot:
  - bs28k, `teacher_batch=458752`, `wd=0`, `beta1=0.975`, LR `0.0020`, `wdl=0.01`, `test_wdl=0`: `649277440` positions in `499.748s`, `1299210` pos/s, `test_loss=0.06488502`, `test_acc=0.6633301`;
  - this beat tatara on loss and speed and missed tatara accuracy by only 2 decisive positions on the 65,536-position yamaoka validation set;
  - nearby WDL/LR/beta probes did not reliably recover the remaining accuracy while keeping loss/speed, so BO-CUDA-030 stayed open at the cuda-oxide stage.
- C++/CUDA full-epoch yamaoka result after the BO-CUDA-034 final-validation wiring:
  - command shape: `--backend cuda-cpp --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3 --teacher target\full-epoch-sfnn-20260718\teacher-all.psv --cuda-cpp-train-steps 4953 --batch-size 131072 --threads 10 --sfnn-factorized-l1 --cuda-cpp-loss-readback-interval 0 --test-teacher C:\shogi\teacher\test\yamaoka-floodgate.psv --test-positions 65536 --test-sample sequential --test-batch-size 8192`;
  - result: `649199616` positions in `489.182s`, `1327112` pos/s, final train loss `0.05046421`, yamaoka `test_loss=0.05579023`, `test_acc=0.6717072`;
  - this beats the tracked tatara reference on all three target metrics: speed (`1327112` > `~1290000` pos/s), held-out loss (`0.05579023` < `0.064964`), and held-out accuracy (`0.6717072` > `0.663361`).
  - The quality target is therefore met by the Windows-native C++/CUDA direct path; BO-CUDA-034 still tracks production checkpoint/log/resume integration for this backend.
- Important validation rule:
  - validation for this target is fixed to `C:\shogi\teacher\test\yamaoka-floodgate.psv`;
  - if that PSV is missing, convert `C:\shogi\teacher\test\yamaoka-floodgate.hcpe` to PSV in the same folder;
  - do not use the training teacher folder or `teacher-all.psv` as validation data.
- Decision:
  - cuda-oxide remains a useful correctness/reference implementation, but it is experimental and Linux-only;
  - production fast-backend work moves to a Windows-native C++/CUDA backend starting with BO-CUDA-031.

### BO-CUDA-031

- Added a new workspace crate `crates/cuda_cpp` (`bulletou-cuda-cpp`):
  - builds `cpp/bulletou_cuda_backend.cu` with `nvcc` through Cargo on Windows;
  - links against the CUDA Toolkit from `CUDA_PATH`;
  - exposes a small safe Rust wrapper around C ABI entry points.
- Added Windows-native CUDA smoke coverage:
  - `bulletou-cuda-cpp-smoke` queries device 0, runs an AXPY kernel, and runs a host-copy Ranger/RAdam update kernel smoke;
  - this validates the native Windows CUDA toolchain without WSL/Ubuntu.
- Added the first C++/CUDA training-relevant kernel:
  - `radam_update_reset_gradients_kernel` mirrors the cuda-oxide RAdam update/reset kernel;
  - `ranger_lookahead_kernel` mirrors the cuda-oxide lookahead phase;
  - the current wrapper copies host arrays in/out for smoke correctness only. BO-CUDA-032 must replace this with persistent device state for training throughput.
- Added BulletOu CLI plumbing:
  - `--backend cuda-cpp` is recognized;
  - `--cuda-cpp-device <N>` selects the CUDA device;
  - `--cuda-cpp-smoke` runs the C++/CUDA backend smoke through `examples/bulletou`;
  - actual NNUE/SFNN training intentionally fails fast until BO-CUDA-033/034 connect the trainer.
- Validation on Windows, CUDA Toolkit `v13.1`, RTX 4090:
  - `cargo test -p bulletou-cuda-cpp --lib` passed;
  - `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed and reported `NVIDIA GeForce RTX 4090`;
  - `cargo check --example bulletou` passed;
  - `cargo check --features cuda-cpp-backend --example bulletou` passed;
  - `cargo run --features cuda-cpp-backend --example bulletou -- --backend cuda-cpp --cuda-cpp-smoke --eval-type NNUE_HALFKP --teacher dummy --cuda-cpp-device 0` passed;
  - `cargo test --example bulletou` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou` passed.

### BO-CUDA-032

- Added persistent C++/CUDA runtime primitives on top of the BO-CUDA-031 smoke:
  - `bulletou_cuda_cpp_context_create/destroy/synchronize` owns a Windows-native CUDA stream per context;
  - `bulletou_cuda_cpp_event_create/destroy/record/wait/synchronize/elapsed_ms` exposes cross-stream dependency points and timing;
  - `bulletou_cuda_cpp_f32_buffer_create/destroy/upload/download/fill` owns reusable device allocations;
  - `bulletou_cuda_cpp_axpy_device` launches on persistent buffers rather than doing a host-copy one-shot;
  - `bulletou_cuda_cpp_ranger_update_device` runs the RAdam reset-gradient phase plus optional Ranger lookahead on persistent device buffers;
  - `bulletou_cuda_cpp_graph_begin_capture/end_capture/launch/destroy` provides generic CUDA Graph capture/replay hooks for fixed train-step launch sequences.
- Added Rust RAII wrappers:
  - `Context`;
  - `Event`;
  - `GraphExec`;
  - `F32Buffer`;
  - `F32UploadSlot`;
  - `axpy_device`;
  - `ranger_update_device`;
  - `RangerDeviceStateMut`.
- Updated both C++/CUDA smokes to verify the persistent path:
  - standalone `bulletou-cuda-cpp-smoke` now checks host-copy AXPY/Ranger, persistent-device AXPY/Ranger, CUDA event timing, CUDA Graph AXPY replay, and upload-context-to-compute-context event handoff;
  - BulletOu CLI `--backend cuda-cpp --cuda-cpp-smoke` also checks the persistent, event, graph, and upload-slot paths.
- Validation on Windows, CUDA Toolkit `v13.1`, RTX 4090:
  - `cargo check -p bulletou-cuda-cpp` passed;
  - `cargo check --features cuda-cpp-backend --example bulletou` passed;
  - `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed and printed matching `axpy_d`, `graph`, `upload`, and `ranger_d` results;
  - `cargo run --features cuda-cpp-backend --example bulletou -- --backend cuda-cpp --cuda-cpp-smoke --eval-type NNUE_HALFKP --teacher dummy --cuda-cpp-device 0` passed;
  - `cargo test -p bulletou-cuda-cpp --lib` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou` passed.
- Result:
  - BO-CUDA-032 is complete as a backend-runtime foundation. BO-CUDA-033 can now allocate persistent NNUE train state, upload teacher batches through `F32UploadSlot` rings, and graph-capture fixed kernel sequences without depending on WSL or cuda-oxide.

### BO-CUDA-033

- Started the Windows-native NNUE port by moving the fixed-layout NNUE forward sequence into `crates/cuda_cpp`:
  - added reusable `I32Buffer` device allocations for sparse feature indices;
  - added C++/CUDA kernels for sparse L0 + CReLU, perspective concat, dense L1 + CReLU, dense L2 + CReLU, and scalar output;
  - added Rust-side `NnueForwardShape`, host/device batch and weight wrappers, `NnueForwardWorkspace`, `nnue_forward_host`, and `nnue_forward_device`;
  - extended `bulletou-cuda-cpp-smoke` to run a two-position tiny NNUE through both one-shot host and persistent-device paths.
- Added BulletOu root-side correctness coverage:
  - `value::fast_nnue::tests::cuda_cpp_tiny_forward_matches_scalar_reference` compares the C++/CUDA tiny NNUE output against the existing CPU scalar golden behind `--features cuda-cpp-backend`;
  - the test is `#[ignore]` so normal CPU-only test runs do not require an NVIDIA GPU.
- Added scalar loss kernels and wrappers:
  - C++/CUDA now implements sigmoid-MSE and nnue-pytorch-WRM per-sample loss, mean output gradients, deterministic weighted-sum finalize, and persistent-device workspace readback;
  - `bulletou-cuda-cpp-smoke` checks both host convenience and persistent-device scalar-loss paths;
  - `value::fast_loss::tests::cuda_cpp_scalar_loss_matches_cpu_reference` compares the C++/CUDA loss output against the existing CPU scalar loss golden behind `--features cuda-cpp-backend`.
- Added a correctness-first NNUE backward path:
  - C++/CUDA now implements dense output backward, dense CReLU backward, L0 CReLU split backward, L0 gradient zeroing, and sparse L0 gradient scatter with `atomicAdd`;
  - Rust now exposes `NnueBackwardWorkspaceLayout`, `NnueBackwardWorkspace`, gradient readback, and `nnue_backward_device`;
  - `bulletou-cuda-cpp-smoke` runs tiny NNUE forward -> sigmoid loss -> backward and compares every gradient buffer against an in-smoke CPU backprop reference.
- Added a first Rust-side C++/CUDA NNUE train-step runner:
  - `NnueTrainStepRunner` owns persistent sparse batch buffers, targets, entry weights, device weights, Ranger optimizer state, forward/loss/backward workspaces, and runs forward -> scalar loss -> backward -> Ranger update;
  - optimizer slow weights initialise from the initial host weights, matching Ranger semantics;
  - the standalone smoke checks one tiny train step by comparing the runner's updated weights against applying the existing host Ranger update to the CPU reference gradients.
- Connected the runner to the BulletOu root CLI for a Windows-native direct NNUE HalfKP path:
  - `examples/bulletou --backend cuda-cpp --cuda-cpp-train-steps N --eval-type NNUE_HALFKP` now streams real teacher batches through the shared fixed-layout `HalfkpTeacherBatchConfig`;
  - the direct path is intentionally limited to Ranger and constant-LR direct steps; production schedule flags remain follow-up work under BO-CUDA-035;
  - initial weights are generated host-side with the same affine default scale as the Bullet builder (`Normal(0, sqrt(2/fan_in))`, zero biases), so `cuda-cpp-backend` no longer needs Bullet's `device-cuda` runtime just to create the model.
- Reduced direct-step synchronization overhead:
  - `NnueTrainStepRunner::step_no_readback` runs upload -> forward -> loss -> backward -> Ranger update without downloading the loss every batch;
  - the compatibility `step` method remains readback-producing, while the direct CLI now samples loss only at step 1, every 10 steps, and the final step.
- Moved the direct C++/CUDA HalfKP scratch path onto the tatara/cuda-oxide factorized FT layout:
  - the input shape is now `HALFKP_PIECE_INPUTS + ShogiHalfKP` (virtual piece rows first, normal HalfKP rows offset by 1548);
  - C++/CUDA sparse L0 forward/backward now expands each normal HalfKP feature into both its offset base row and its virtual piece row when the factorized input size is selected;
  - `HalfkpTeacherBatchConfig::ft_factorize = false` and `max_active = ShogiHalfKP.max_active()`, so the host streams normal HalfKP indices and avoids duplicating factorized rows in the teacher batch;
  - initial weights match the cuda-oxide tatara-simple recipe: virtual L0 rows zero, base L0/bias/dense/output tensors initialized by deterministic `TataraXorShift` uniform values in `[-0.01, 0.01]`.
- Added direct-output writing:
  - after `--cuda-cpp-train-steps`, the runner reads trained weights and Ranger optimizer buffers back and writes `<output>/cuda-cpp-direct/nn.bin` plus `<output>/cuda-cpp-direct/weights.bin`;
  - `nn.bin` folds factorized HalfKP virtual rows back into normal HalfKP L0 rows before quantization;
  - `weights.bin` stores raw f32 `nnue/weights/*`, `nnue/momentum/*`, `nnue/velocity/*`, `nnue/slow/*`, and `nnue/step_ranger/*` records with a `cuda-cpp` backend marker, matching the root state.bin component namespace.
- Added `--cuda-cpp-weights-bin <PATH>` for direct-trainer initial weights/state:
  - it accepts root-format/unprefixed `l0w`..`outb` records or `nnue/weights/*` component records;
  - if `nnue/{momentum,velocity,slow}/*` records are present, it restores Ranger optimizer buffers too;
  - if `nnue/step_ranger/*` records are present, the direct trainer continues RAdam's step counter from that value;
  - older weights-only files still load, but their optimizer buffers are reinitialized.
  - this is explicit direct-mode state replay; normal numbered-checkpoint continuation now uses `--resume`/auto-resume, so `--cuda-cpp-weights-bin` is rejected when combined with `--resume` or `--no-resume`.
- Added direct-step CUDA event profiling:
  - `--cuda-cpp-profile-steps N` profiles only the first N direct train steps, leaving normal unprofiled throughput unaffected afterward;
  - each profiled step prints upload / forward / loss / backward / Ranger update / total GPU time.
- Reduced train-step L0 backward overhead:
  - the public `nnue_backward_device` path keeps the previous fresh-gradient semantics and zeroes L0 gradients every call;
  - `NnueTrainStepRunner` now initialises L0 gradient buffers once and then reuses Ranger's per-step gradient reset, skipping the large per-step L0 zero kernel in the direct training path.
- Reduced Ranger update overhead:
  - C++/CUDA RAdam/Ranger update now uses a `float4` kernel whenever the parameter group length is divisible by four;
  - scalar update remains as the fallback for groups such as `outb`.
- Reduced L0 bias backward contention:
  - L0 weight gradients still use the correctness-first sparse `atomicAdd` scatter;
  - L0 bias gradients now use a row-wise reduction kernel instead of contended atomics into 256 bias entries.
- Moved dense hidden-layer CReLU backward onto cuBLAS:
  - the C++/CUDA context owns a cuBLAS handle on the same stream and links `cublas` from the CUDA toolkit;
  - L1/L2 dense input-gradient and weight-gradient phases now use `cublasSgemm`, with small custom kernels only for CReLU pre-gradient and bias-gradient reduction;
  - this cuts the profiled HalfKP backward hot section roughly in half on RTX 4090 after the one-time cuBLAS warmup.
- Validation on Windows, CUDA Toolkit `v13.1`, RTX 4090:
  - `cargo check -p bulletou-cuda-cpp` passed;
  - `cargo check -p bulletou_lib --features cuda-cpp-backend` passed;
  - `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed and printed `nnue_h: [1.208, 1.1194999]` / `nnue_d: [1.208, 1.1194999]`, matching `loss_h` / `loss_d`, `bwd_d : outb_grad=[0.09245913] l0b_grad=[0.05391511, 0.05685694]`, and `train : loss_mean=0.13517515 outb=[0.04907541]`;
  - `cargo test -p bulletou-cuda-cpp --lib` passed;
  - `cargo test -p bulletou_lib --features cuda-cpp-backend cuda_cpp_tiny_forward_matches_scalar_reference -- --ignored` passed;
  - `cargo test -p bulletou_lib --features cuda-cpp-backend cuda_cpp_scalar_loss_matches_cpu_reference -- --ignored` passed;
  - `cargo check --features cuda-cpp-backend --example bulletou` passed;
  - `cargo check --example bulletou` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture` passed, including direct full-state writer/loader tests;
  - `cargo run --features cuda-cpp-backend --example bulletou -- --eval-type NNUE_HALFKP --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --backend cuda-cpp --cuda-cpp-train-steps 2 --batch-size 1024 --buffer-mb 64 --threads 4` passed without WSL and reported two real HCPE train steps;
  - `cargo run --release --features cuda-cpp-backend --example bulletou -- --eval-type NNUE_HALFKP --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --backend cuda-cpp --cuda-cpp-train-steps 100 --batch-size 4096 --buffer-mb 128 --threads 8` passed and reported `throughput=867894 pos/s` for the short direct-step probe after readback sampling.
  - after switching to the factorized FT layout, `cargo run --release --features cuda-cpp-backend --example bulletou -- --eval-type NNUE_HALFKP --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --backend cuda-cpp --cuda-cpp-train-steps 50 --batch-size 4096 --buffer-mb 128 --threads 8` passed and reported `throughput=802739 pos/s`.
  - direct output smoke with `--output target\cuda-cpp-direct-smoke` wrote `cuda-cpp-direct/nn.bin` (64,217,077 bytes) and `cuda-cpp-direct/weights.bin` (130,054,016 bytes).
  - direct weights reload smoke with `--cuda-cpp-weights-bin target\cuda-cpp-direct-smoke\cuda-cpp-direct\weights.bin --output target\cuda-cpp-direct-resume-smoke` passed and wrote a fresh `cuda-cpp-direct/{nn.bin,weights.bin}`.
  - full-state direct output smoke with `--output target\cuda-cpp-fullstate-smoke --cuda-cpp-train-steps 1 --batch-size 256` wrote `cuda-cpp-direct/nn.bin` (64,217,077 bytes) and full-state `cuda-cpp-direct/weights.bin` (520,215,138 bytes).
  - full-state reload smoke with `--cuda-cpp-weights-bin target\cuda-cpp-fullstate-smoke\cuda-cpp-direct\weights.bin --output target\cuda-cpp-fullstate-resume-smoke --cuda-cpp-train-steps 1 --batch-size 256` restored `weights + Ranger optimizer state`, printed `initial completed optimizer steps = 1`, and ran the next update with `optimizer_step=2`.
  - profile smoke with `--cuda-cpp-train-steps 20 --cuda-cpp-profile-steps 3 --batch-size 4096 --buffer-mb 128 --threads 8 --output target\cuda-cpp-profile-smoke` passed and reported average profiled GPU time: upload `0.455ms`, forward `0.377ms`, loss `0.165ms`, backward `1.862ms`, Ranger update `1.379ms`, total `4.237ms`.
  - after skipping the direct-path L0 zero kernel, `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed, `cargo test -p bulletou-cuda-cpp --lib persistent_device_api_smoke -- --ignored --nocapture` passed, and the same 20-step profile reported backward `1.758ms`, total `4.117ms`.
  - unprofiled 100-step release smoke after the skip-zero change reported `throughput=868393 pos/s` for `--cuda-cpp-train-steps 100 --batch-size 4096 --buffer-mb 128 --threads 8 --output target\cuda-cpp-skipzero-100`.
  - after vectorising the C++/CUDA Ranger update, `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed and `cargo test -p bulletou-cuda-cpp --lib persistent_device_api_smoke -- --ignored --nocapture` passed.
  - vectorized update profile smoke reported average profiled GPU time: upload `0.427ms`, forward `0.332ms`, loss `0.143ms`, backward `1.703ms`, Ranger update `1.215ms`, total `3.821ms`.
  - unprofiled 100-step release smoke after vectorized update reported `throughput=897088 pos/s` for `--cuda-cpp-train-steps 100 --batch-size 4096 --buffer-mb 128 --threads 8 --output target\cuda-cpp-vec4-100`.
  - after moving L0 bias backward off atomics, `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed, `cargo test -p bulletou-cuda-cpp --lib persistent_device_api_smoke -- --ignored --nocapture` passed, and the unprofiled 100-step release smoke reported `throughput=910234 pos/s`.
  - Added direct-mode numbered checkpoint/log emission alongside the temporary `cuda-cpp-direct` folder:
    `--backend cuda-cpp --eval-type NNUE_HALFKP --cuda-cpp-train-steps 1 --batch-size 64 --output target\cuda-cpp-numbered-halfkp-smoke` wrote `0001/{nn.bin,state.bin,teacher.txt,dataloader_pos.txt,learn.log}`, top-level `summary-learn.log`, and `tag.txt`; `learn.log` / `summary-learn.log` used the production CSV schemas and `dataloader_pos.txt` was `2432,0`.
- Added direct-mode auto-resume from numbered checkpoints:
  - `--backend cuda-cpp` now participates in the normal `resume-config.txt` compatibility check, accepts `--resume` / `--no-resume`, and auto-loads the latest numbered `state.bin` when the output directory is compatible;
  - the direct path resumes both weights and Ranger optimizer state, restores the completed optimizer-step counter, and passes the latest `dataloader_pos.txt` into the shared teacher batch loader;
  - if `--resume` is forced while the teacher spec differs from the latest checkpoint, weights/optimizer state still resume but the dataloader starts from the new teacher's beginning, matching the normal BulletOu resume rule;
  - fixed-record PSV teacher batches now map `TeacherDataloaderPos.byte_offset` back to a batch index for both HalfKP and SFNN, while rejecting nonzero plies, record-misaligned offsets, and batch-misaligned offsets;
  - direct `learn.log` / `summary-learn.log` rows now keep `positions` cumulative across resumed direct runs.
- Added HalfKP final validation for the Windows-native direct path:
  - root `bulletou` now accepts `--test-teacher` for `--backend cuda-cpp --eval-type NNUE_HALFKP`;
  - validation folds factorized L0 virtual rows into normal HalfKP rows, runs the existing CPU fast NNUE validator, and writes `test_value_accuracy` / `test_value_loss` into numbered `learn.log` and `summary-learn.log`;
  - benchmark and smoke validation use only `C:\shogi\teacher\test\yamaoka-floodgate.psv` as the held-out set.
- Validation for direct auto-resume:
  - `cargo check --features cuda-cpp-backend --example bulletou` passed;
  - `cargo test -p bulletou_lib psv_resume_offset_maps_to_batch_index -- --nocapture` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture` passed (26 cuda-cpp tests);
  - HalfKP auto-resume smoke on `target\cuda-cpp-numbered-halfkp-smoke` loaded `0002/state.bin`, printed `initial completed optimizer steps = 2`, ran `optimizer_step=3`, wrote `0003/dataloader_pos.txt = 7296,0`, and the corrected `0003/learn.log` row used cumulative `positions=128`.
- Validation after the final HalfKP C++/CUDA rewrite:
  - `cargo test -p bulletou-cuda-cpp --lib` passed;
  - `cargo check --features cuda-cpp-backend --example bulletou` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture` passed (27 cuda-cpp tests);
  - `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed and reported `result : ok` on `NVIDIA GeForce RTX 4090`;
  - one-step HalfKP validation smoke using `--test-teacher C:\shogi\teacher\test\yamaoka-floodgate.psv --test-positions 128` wrote held-out metrics (`test_value_accuracy=0.5000000`, `test_value_loss=0.17730953`) to the numbered logs;
  - profiled HalfKP bs16k after implicit factorized rows and cuBLAS: post-warmup steps reported upload `0.682-0.987ms`, forward `0.720-0.742ms`, loss `0.306-0.309ms`, backward `3.098-3.108ms`, update `1.146-1.155ms`, total `5.955-6.299ms`; the first measured step includes one-time cuBLAS warmup and is not representative of steady state.
- BO-CUDA-033 4M comparison result:
  - command shape: `--backend cuda-cpp --eval-type NNUE_HALFKP --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --cuda-cpp-train-steps 256 --batch-size 16384 --threads 10 --cuda-cpp-loss-readback-interval 0 --test-teacher C:\shogi\teacher\test\yamaoka-floodgate.psv --test-positions 65536 --test-sample sequential --test-batch-size 8192`;
  - three clean bs16k/lr0.024 runs reported `2,651,604`, `2,645,789`, and `2,664,118` positions/sec, with held-out accuracy/loss `0.6733398/0.05194952`, `0.6632080/0.05239367`, and `0.6640625/0.05355105`;
  - mean result: `2,653,837` positions/sec, `test_value_accuracy=0.6668701`, `test_value_loss=0.0526314`;
  - BO-CUDA-029 tatara idle reference mean was `2,495,146` positions/sec, `test_value_accuracy=0.663567`, `test_value_loss=0.053232`, so the Windows-native C++/CUDA HalfKP path clears the tracked tatara speed/quality target without WSL.
- Result:
  - BO-CUDA-033 is complete for the requested Windows-native HalfKP C++/CUDA direct trainer and tatara-beating 4M comparison.
  - Remaining ergonomics belong to BO-CUDA-035, and deeper HalfKP micro-optimisation against the previous cuda-oxide throughput ceiling belongs to BO-CUDA-036.

### BO-CUDA-034

- Started the Windows-native SFNN port by adding a fixed-layout SFNN forward path to `crates/cuda_cpp`:
  - Rust now exposes `SfnnForwardShape`, host/device batch wrappers, host/device weight wrappers, `SfnnForwardWorkspace`, and `sfnn_forward_device`;
  - the current layout supports stacked `l1/l2/l3` weights, per-sample bucket selection, PSQT shortcut output, and optional shared/factorized L1 weights;
  - C++/CUDA now runs sparse L0 + CReLU, pairwise concat, stacked L1, L2 input transform, stacked L2 + CReLU, and stacked L3 output kernels;
  - the sparse L0 kernel also supports HalfKA2-style virtual piece rows when `input_size == 131949 + 1629`, so normal features can add their factorized piece row without expanding teacher indices on the host.
- Added correctness smoke coverage:
  - `tests::sfnn_workspace_layout_counts_forward_activations` verifies the Rust workspace sizing;
  - ignored GPU test `tests::sfnn_tiny_forward_gpu_smoke` compares the C++/CUDA SFNN forward result against an in-test CPU scalar reference;
  - standalone `bulletou-cuda-cpp-smoke` now includes the same SFNN tiny forward check alongside NNUE/loss/backward/Ranger smokes.
- Added the correctness-first SFNN backward path:
  - Rust now exposes `SfnnBackwardWorkspaceLayout`, `SfnnBackwardWorkspace`, gradient readback, and `sfnn_backward_device`;
  - C++/CUDA now runs stacked L3 backward, stacked L2 CReLU backward, L2-input transform backward, stacked/factorized L1 backward, pairwise backward, and sparse L0 CReLU backward;
  - parameter-gradient buffers are zeroed at the start of each public backward call because the stacked SFNN gradients are accumulated with atomics;
  - the L0 sparse backward also adds gradients to HalfKA2 virtual piece rows when the factorized input shape is used.
- Added the first persistent SFNN train-step runner:
  - `SfnnTrainStepRunner` owns reusable sparse batches, buckets, targets, entry weights, SFNN weights, optional L1f optimizer state, forward/loss/backward workspaces, and Ranger optimizer buffers;
  - `SfnnRangerOptimizerStates` mirrors the NNUE optimizer bundle and keeps `l1fw/l1fb` optional as a matched pair;
  - one runner step now executes upload -> SFNN forward -> existing scalar loss -> SFNN backward -> Ranger update without leaving the C++/CUDA backend.
- Wired the runner into the root BulletOu CLI for direct Windows-native smoke/training:
  - `examples/bulletou --backend cuda-cpp --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3 --cuda-cpp-train-steps N` now streams real `SfnnTeacherBatchConfig` batches through `SfnnTrainStepRunner`;
  - `crates/bulletou_lib::value` now publicly re-exports the SFNN fixed-layout teacher batch helpers needed by the root CLI;
  - SFNN cuda-cpp direct mode accepts optional `--sfnn-factorized-l1`, with `l1fw/l1fb` zero-initialised to match the existing Bullet CLI semantics;
  - SFNN initial weights use a deterministic nnue-pytorch-style scratch layout: HalfKA2 base rows plus zero virtual piece rows for the factorized FT, and bucket0-copied stacked L1/L2/L3 weights.
- Added SFNN direct output and explicit resume support:
  - after direct training, the runner reads trained SFNN weights and Ranger optimizer buffers back and writes `<output>/cuda-cpp-direct/nn.bin` plus full-state `<output>/cuda-cpp-direct/weights.bin`;
  - `nn.bin` is written in YaneuraOu SFNN HalfKA2 format, folds the HalfKA2 FT virtual piece rows into base feature rows, and folds optional `l1fw/l1fb` into each stack's fc0 weights/biases;
  - `weights.bin` uses the root `nnue/{weights,momentum,velocity,slow,step_ranger}/*` namespace with a `cuda-cpp` backend marker, including optional `l1fw/l1fb` records when factorized L1 is enabled;
  - `--cuda-cpp-weights-bin <PATH>` can now restore SFNN direct weights, Ranger state, optional factorized L1 state, and the completed optimizer-step counter.
- Added SFNN per-stage direct-step profiling:
  - `SfnnTrainStepRunner::step_profiled_no_readback` mirrors the NNUE profiled runner and measures upload, forward, scalar loss, backward, Ranger update, and total CUDA time with events;
  - `examples/bulletou --backend cuda-cpp --eval-type SFNN_HALFKA2 --cuda-cpp-profile-steps N` now prints per-step and average SFNN profile lines.
- Added the first SFNN training-only backward speedup:
  - the public/debug backward entry remains split so `stm_l0_gradients`, `nstm_l0_gradients`, and pre-gradient buffers stay inspectable for correctness tests;
  - `SfnnTrainStepRunner` now calls a dedicated `bulletou_cuda_cpp_sfnn_backward_train_device` entry that fuses pairwise backward with sparse L0 CReLU backward, avoiding the intermediate pairwise-gradient write/read in the hot train path;
  - the existing ignored SFNN backward smoke still covers the split path, and the train-step runner smoke covers the fused path against the CPU golden.
- Reduced SFNN backward atomic overhead further:
  - the fused pairwise/L0 sparse backward kernel now works per pair instead of per row, so each thread handles both halves of a pair and scans the sparse feature list once instead of twice;
  - stacked L3/L2/L1 backward kernels skip zero-gradient or zero-activation weight/bias `atomicAdd`s, preserving the same gradients while avoiding a large fraction of atomics after CReLU/pairwise sparsification.
- Added double-buffered SFNN upload pipelining for the root direct trainer:
  - `I32UploadSlot` mirrors the existing `F32UploadSlot`, so sparse feature indices can be uploaded on a separate CUDA stream;
  - `SfnnTrainStepRunner` owns two `SfnnTrainStepUploadSlot`s containing sparse indices, buckets, targets, entry weights, upload-ready events, and compute-done events;
  - `step_pipelined_no_readback` uploads the next batch on `upload_ctx`, makes the compute stream wait on the upload-ready event, then records compute-done after Ranger update so slot reuse cannot overwrite buffers still read by kernels;
  - `examples/bulletou --backend cuda-cpp --eval-type SFNN_HALFKA2` uses the pipeline for non-profiled direct steps while keeping profiled steps serial for clean stage timings.
- Added SFNN teacher CPU/GPU overlap parity:
  - `SfnnTeacherBatchConfig` now has `queue_depth`, matching the HalfKP helper;
  - when `queue_depth > 1` and `profile_prepare=false`, a producer thread materializes `FastBatchHost` SFNN batches into a bounded queue while the caller continues enqueueing GPU work;
  - root `--backend cuda-cpp --eval-type SFNN_HALFKA2` passes the same auto-tuned `--batch-queue-size` path as HalfKP, while the cuda-oxide caller keeps `queue_depth=1` to preserve its existing behavior.
- Reduced direct-loop loss readback synchronization:
  - `--cuda-cpp-loss-readback-interval N` controls how often the direct C++/CUDA trainer synchronizes the compute stream to read/report loss;
  - the default `10` preserves the previous step1/every10/final reporting cadence, while `0` keeps only the final readback for throughput probes.
- Reduced no-readback SFNN train-step work:
  - `SfnnTrainStepRunner` now clears parameter-gradient buffers once when the runner is created, then the hot train-only backward entry skips the per-step parameter-gradient zero because the Ranger update path consumes and resets those buffers;
  - scalar loss now has an internal `finalize_loss=false` path so SFNN direct steps that will not read/report loss still compute output gradients but skip the final weighted-sum/mean reduction kernel.
- Added final held-out validation for the SFNN C++ direct path:
  - root `bulletou` now accepts `--test-teacher` for `--backend cuda-cpp --eval-type SFNN_HALFKA2`; HalfKP validation support is covered by BO-CUDA-033;
  - added root `--test-sample random|sequential` so the C++ direct path can use the same sequential yamaoka PSV subset as the tatara/cuda-oxide SFNN parity probes;
  - the final C++ direct validation reads the trained weights back, folds optional `l1fw/l1fb` into the stacked L1 weights/biases for the CPU fast SFNN forward, evaluates the cached yamaoka positions, and prints `test_value_accuracy` / `test_value_loss`;
  - root `--backend cuda-oxide` now forwards `--test-sample` to the child trainer so the same CLI spelling works across both experimental backends.
- Added the same direct-mode numbered checkpoint/log emission to SFNN C++ direct:
  - after training, the backend still writes the temporary `<output>/cuda-cpp-direct/{nn.bin,weights.bin}` compatibility folder, and now also writes the production-shaped `<output>/<NNNN>/{nn.bin,state.bin,teacher.txt,dataloader_pos.txt,learn.log}` plus top-level `summary-learn.log`;
  - direct `learn.log` is a one-row production CSV for the completed direct run, with final validation metrics populated when `--test-teacher` is present.
- Added SFNN direct auto-resume through the same numbered-checkpoint path as HalfKP:
  - compatible `--backend cuda-cpp --eval-type SFNN_HALFKA2` runs now load the latest numbered `state.bin` automatically, restore optional factorized-L1 weights and Ranger state, and continue the optimizer-step counter;
  - same-teacher resume passes the stored dataloader position to `SfnnTeacherBatchConfig`, including PSV fixed-record offsets.
- Validation on Windows, CUDA Toolkit `v13.1`, RTX 4090:
  - `cargo check -p bulletou-cuda-cpp` passed;
  - `cargo test -p bulletou-cuda-cpp --lib` passed;
  - `cargo test -p bulletou-cuda-cpp --lib sfnn_tiny_forward_gpu_smoke -- --ignored --nocapture` passed;
  - `cargo test -p bulletou-cuda-cpp --lib sfnn_tiny_backward_gpu_smoke -- --ignored --nocapture` passed;
  - `cargo test -p bulletou-cuda-cpp --lib sfnn_tiny_train_step_runner_smoke -- --ignored --nocapture` passed;
  - `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed and printed `sfnn_d: [0.06838137, 0.0869026]`;
  - `cargo check --features cuda-cpp-backend --example bulletou` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture` passed (26 cuda-cpp tests);
  - `cargo run --features cuda-cpp-backend --example bulletou -- --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3 --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --backend cuda-cpp --cuda-cpp-train-steps 2 --batch-size 256 --buffer-mb 64 --threads 4 --sfnn-factorized-l1` passed on Windows and ran two real HCPE SFNN train steps.
  - Direct output smoke with `--output target\cuda-cpp-sfnn-output-smoke --cuda-cpp-train-steps 1 --batch-size 256 --sfnn-factorized-l1 --cuda-cpp-loss-readback-interval 0` wrote `cuda-cpp-direct/nn.bin` (135,212,356 bytes) and full-state `weights.bin` (2,190,019,306 bytes).
  - Direct resume smoke with `--cuda-cpp-weights-bin target\cuda-cpp-sfnn-output-smoke\cuda-cpp-direct\weights.bin --output target\cuda-cpp-sfnn-resume-smoke --cuda-cpp-train-steps 1 --batch-size 256 --cuda-cpp-loss-readback-interval 0` restored `weights + Ranger optimizer state`, printed `initial completed optimizer steps = 1`, and ran the next update with `optimizer_step=2`.
  - Profile smoke with `--cuda-cpp-train-steps 3 --cuda-cpp-profile-steps 2 --batch-size 256 --buffer-mb 64 --threads 4 --sfnn-factorized-l1` passed and reported average profiled GPU time: upload `0.193ms`, forward `2.232ms`, loss `0.176ms`, backward `4.283ms`, Ranger update `5.870ms`, total `12.753ms`.
  - Training-only fused pairwise/L0 backward smokes passed:
    `cargo test -p bulletou-cuda-cpp --lib sfnn_tiny_backward_gpu_smoke -- --ignored --nocapture`,
    `cargo test -p bulletou-cuda-cpp --lib sfnn_tiny_train_step_runner_smoke -- --ignored --nocapture`,
    `cargo check --features cuda-cpp-backend --example bulletou`, and
    `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture`.
  - Release bs16k profile before fused backward: `throughput=719579 pos/s`, profile avg upload `0.773ms`, forward `2.887ms`, loss `0.394ms`, backward `11.336ms`, update `4.901ms`, total `20.291ms`.
  - Release bs16k profile after fused backward: `throughput=760770 pos/s`, profile avg upload `0.664ms`, forward `2.820ms`, loss `0.353ms`, backward `10.036ms`, update `4.903ms`, total `18.776ms`.
  - Release bs65k short probe after fused backward: `throughput=921239 pos/s`, profile avg upload `3.091ms`, forward `10.234ms`, loss `1.167ms`, backward `38.634ms`, update `4.919ms`, total `58.045ms`.
  - Upload-pipelined smokes passed:
    `cargo test -p bulletou-cuda-cpp --lib persistent_device_api_smoke -- --ignored --nocapture` (including `I32UploadSlot`),
    `cargo test -p bulletou-cuda-cpp --lib sfnn_tiny_train_step_runner_smoke -- --ignored --nocapture`,
    `cargo check --features cuda-cpp-backend --example bulletou`, and
    `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture`.
  - Release bs65k no-profile direct smoke after upload pipelining:
    `--cuda-cpp-train-steps 50 --batch-size 65536 --buffer-mb 512 --threads 10 --sfnn-factorized-l1` reported `throughput=1064822 pos/s`;
    a shorter post-display-change bs65k/20-step smoke reported `throughput=1059357 pos/s`.
  - With final-only loss readback, the same bs65k/50-step release smoke plus `--cuda-cpp-loss-readback-interval 0` reported `throughput=1120866 pos/s`.
  - After pairwise-L0 and zero-atomic skipping, bs65k/50-step final-only release smoke reported `throughput=1190496 pos/s`.
  - Backward profile moved from the post-fused baseline `38.249ms` through pairwise-L0 `37.705ms` to zero-skip `33.576ms` on the bs65k/3-profile-step probe; the rejected L1-small-output experiment regressed to `35.039ms` and was not kept.
  - After one-time SFNN gradient zeroing, bs65k/50-step final-only release smoke reported `throughput=1205993 pos/s`.
  - After skipping scalar-loss finalization on non-reported SFNN steps, bs131k/50-step final-only release smoke reported `throughput=1301058 pos/s`, and bs262k/20-step final-only release smoke reported `throughput=1320817 pos/s`.
  - Final-validation wiring checks passed:
    `cargo check --features cuda-cpp-backend --example bulletou`,
    `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture` (26 cuda-cpp tests), and
    `cargo test --example bulletou cuda_oxide_backend_accepts_sfnn_halfka2_direct_steps -- --nocapture`.
  - Real-data validation smoke on Windows used training `C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe` and held-out `C:\shogi\teacher\test\yamaoka-floodgate.psv` with `--test-positions 128 --test-sample sequential`; it printed `test_value_accuracy=0.5000000`, `test_value_loss=0.17757529`.
  - Full exported-teacher yamaoka comparison on Windows used `target\full-epoch-sfnn-20260718\teacher-all.psv` for training and only `C:\shogi\teacher\test\yamaoka-floodgate.psv` for validation:
    `--cuda-cpp-train-steps 4953 --batch-size 131072 --threads 10 --sfnn-factorized-l1 --cuda-cpp-loss-readback-interval 0 --test-positions 65536 --test-sample sequential --test-batch-size 8192` reported `649199616` positions in `489.182s`, `throughput=1327112 pos/s`, `test_value_loss=0.05579023`, `test_value_accuracy=0.6717072`.
  - Auto-resume smoke for SFNN direct wrote `target\cuda-cpp-sfnn-numbered-resume-smoke\0001`, resumed from `0001/state.bin` with `dataloader resume = byte_offset 608, plies 0`, then wrote `0002`; a follow-up resume from `0002/state.bin` printed `initial completed optimizer steps = 2`, ran `optimizer_step=3`, wrote `0003/dataloader_pos.txt = 1824,0`, and the corrected `0003/learn.log` row used cumulative `positions=32`.
  - After the cuda-cpp teacher CPU auto-tune and SFNN producer queue, a bs131k/128-step HCPE probe with final-only WRM loss readback reported `16777216` positions in `9.180s`, `throughput=1827661 pos/s`.
  - The same queue-enabled shape with held-out validation fixed to `C:\shogi\teacher\test\yamaoka-floodgate.psv`, `--test-positions 65536 --test-sample sequential --test-batch-size 8192`, reported `throughput=1816050 pos/s`, `test_value_loss=0.03449630`, `test_value_accuracy=0.6285858`.
  - A serial-profile control run on the same 128-step shape reported yamaoka `test_value_loss=0.03461872`, `test_value_accuracy=0.6276855`, confirming the queue path preserves the expected quality band.
- Rejected SFNN upload experiment:
  - adding pinned staged host buffers to `SfnnTrainStepUploadSlot` did not improve the bs131k/128-step speed materially and corrupted the yamaoka check (`test_value_accuracy=0.6073761`, `test_value_loss=0.05824073`), while the serial upload control stayed healthy, so the pinned SFNN upload change was reverted.
- Rejected SFNN scatter micro-optimisation:
  - moving input-value loads behind `grad != 0.0f` checks in the stacked L1/L2/L3 weight-gradient scatter kernels passed the tiny GPU backward/train smokes, but regressed the bs131k/6-step WRM profile (`backward=69.416ms` vs `65.129ms` baseline), so it was reverted.
- Rechecked the SFNN pairwise-L0 train fuse:
  - temporarily switching the train entry back to the older non-fused pairwise + L0 sparse backward path worsened the bs131k/6-step WRM profile (`backward=74.532ms`), confirming the fused `sfnn_pairwise_l0_sparse_backward_kernel` remains the faster path.
- BO-CUDA-034 is complete for the tracked tatara speed/quality target and Windows-native auto-resume. Follow-up optimisation and production-schedule ergonomics continue under BO-CUDA-033/035.

### BO-CUDA-035

- Started production-schedule parity for the Windows-native C++/CUDA direct backend:
  - `--backend cuda-cpp` still accepts explicit `--cuda-cpp-train-steps N` for short direct-step smoke/profiling runs;
  - it now also accepts bounded production mode with `--superbatches N --max-epochs N` for both `NNUE_HALFKP` and `SFNN_HALFKA2`;
  - direct-step mode and production mode are mutually exclusive, so `--cuda-cpp-train-steps` cannot be mixed with `--superbatches`;
  - bounded production mode expands `superbatches * max_epochs * effective_batches_per_superbatch` into direct C++/CUDA train steps and chunks checkpoint saves by `--save-rate`;
  - each save chunk writes a normal numbered checkpoint and summary row, so `--save-rate 1` produces one checkpoint per superbatch and larger values save at the end of each save-rate chunk;
  - step/geometric/cos LR schedules are applied to the actual C++/CUDA Ranger update per batch using the same positions-based LR formulas as the normal trainer, with epoch-local warm restarts.
- Production mode is intentionally bounded and requires `--max-epochs` to avoid accidental infinite direct runs, matching the cuda-oxide production wrapper's safety guard.
- Added production-mode resume scheduling:
  - mid-epoch resume detects the latest saved superbatch and resumes from `last_sb + 1` while keeping the displayed epoch number stable;
  - cleanly completed epochs continue as additional epochs with `epoch = previous_max_epoch + 1` and `superbatch = 1`;
  - LR computation now uses `prior_positions + current_run_positions`, so mid-epoch resumes continue inside the same LR cycle and clean epoch continuations warm-restart when the previous epoch size is complete;
  - the existing direct C++ resume path continues to restore weights, Ranger optimizer state, completed optimizer-step counters, and the teacher dataloader position.
- Added C++/CUDA production plateau orchestration:
  - `--backend cuda-cpp --lr-schedule plateau` is now accepted in bounded production mode for both `NNUE_HALFKP` and `SFNN_HALFKA2`;
  - plateau still requires `--test-teacher` and `--save-rate 1`, matching the existing per-superbatch validation decision model;
  - each attempted superbatch snapshots C++/CUDA weights and Ranger optimizer state in memory, trains at the current plateau LR, validates, then either writes an accepted numbered checkpoint or restores the snapshot and retries/rejects according to the shared `PlateauLrState`;
  - rejected final-min runs do not write a new checkpoint; the latest accepted checkpoint is marked with `plateau_epoch_done.txt`.
- Validation:
  - `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp_run_schedule_ -- --nocapture` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture` passed (37 cuda-cpp tests);
  - HalfKP production smoke on Windows:
    `--backend cuda-cpp --eval-type NNUE_HALFKP --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --superbatches 2 --max-epochs 1 --positions-per-superbatch 64 --batch-size 64 --save-rate 1 --cuda-cpp-loss-readback-interval 0 --test-teacher C:\shogi\teacher\test\yamaoka-floodgate.psv --test-positions 32 --test-sample sequential`
    wrote `0001` and `0002`, logged `positions=64` then `128`, and wrote `dataloader_pos.txt` as `2432,0` then `4864,0`.
  - Re-running the same HalfKP production smoke auto-resumed from `0002/state.bin`, restored `optimizer_step=3`, resumed the teacher at `4864,0`, wrote `0003` and `0004` as `epoch=2, superbatch=1..2`, logged `positions=192` then `256`, and advanced `dataloader_pos.txt` to `7296,0` then `9728,0`.
  - HalfKP plateau accept smoke with yamaoka PSV validation wrote `0001`, logged `lr_start=lr_end=0.001000`, and advanced `dataloader_pos.txt` to `2432,0`.
  - HalfKP plateau reject smoke using `--lr-plateau-monitor accuracy --lr 0.001 --lr-min 0.001` accepted `0001`, rejected the second attempted superbatch, left no `0002`, and wrote `0001/plateau_epoch_done.txt`.
  - SFNN HalfKA2/factorized-L1 plateau smoke on Windows:
    `--backend cuda-cpp --eval-type SFNN_HALFKA2 --arch SFNN_halfka2_1024_7_64_k3k3 --sfnn-factorized-l1 --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --superbatches 1 --max-epochs 1 --positions-per-superbatch 8 --batch-size 8 --save-rate 1 --lr-schedule plateau --test-teacher C:\shogi\teacher\test\yamaoka-floodgate.psv --test-positions 8 --test-sample sequential`
    wrote `0001/{nn.bin,state.bin,learn.log,dataloader_pos.txt}`, logged `test_value_accuracy=0.625000`, `test_value_loss=0.239121`, `positions=8`, `lr_start=lr_end=0.001000`, and wrote `dataloader_pos.txt = 304,0`.
- Result:
  - BO-CUDA-035 is complete for bounded production-schedule parity on the Windows-native C++/CUDA direct backend. Remaining work now moves to BO-CUDA-036's post-parity HalfKP optimisation.

### BO-CUDA-036

- Adopted first-step warmup for the Windows-native C++/CUDA HalfKP direct trainer:
  - `Context` creation now performs a tiny CUDA/cuBLAS warmup, and `NnueTrainStepRunner::warmup` additionally launches the actual HalfKP dense-backward GEMM shapes once against scratch workspaces before the training timer starts;
  - the warmup does not update trainable weights or optimizer state, and it avoids reading sparse feature indices, so it is safe before the first real teacher batch upload.
- Adopted direct-mode benchmark timing cleanup:
  - explicit `--cuda-cpp-train-steps` direct mode now defers its final numbered checkpoint and final validation until after `cuda-cpp direct train = ok` has measured elapsed training time;
  - numbered checkpoints, `summary-learn.log`, validation metrics, and the compatibility `cuda-cpp-direct/{nn.bin,weights.bin}` folder are still written.
- Adopted HalfKP CPU/GPU teacher-prepare overlap:
  - `HalfkpTeacherBatchConfig` now has `queue_depth`;
  - when `queue_depth > 1` and `profile_prepare=false`, a producer thread materializes `FastBatchHost` batches into a bounded queue while the caller consumes the previous batch on the GPU;
  - `examples/bulletou --backend cuda-cpp` passes the existing `--batch-queue-size` to this queue;
  - cuda-oxide and fixture-export callers use `queue_depth=1`, preserving their previous serial behavior.
- HalfKP bs16k profile on RTX 4090 after warmup:
  - command shape: `--backend cuda-cpp --eval-type NNUE_HALFKP --teacher C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe --cuda-cpp-train-steps 4 --batch-size 16384 --threads 10 --cuda-cpp-profile-steps 4 --cuda-cpp-loss-readback-interval 0`;
  - before runner warmup, step1 backward was about `49.1ms`; after the tiny context warmup it was about `16.4ms`;
  - after the dense-backward runner warmup, step1 backward dropped to `3.36-3.52ms`, with steady later steps about `3.10ms`;
  - direct-mode checkpoint deferral moved the final checkpoint write after the measured `elapsed`.
- HalfKP 4M speed probes on the same HCPE teacher, final-only WRM loss readback, `wd=0`, `beta1=0.975`, `lr=0.024`, `threads=10`:
  - before CPU/GPU teacher overlap, bs16k reported about `1.95M` pos/s after checkpoint timing cleanup;
  - after overlap, bs16k improved to `2.21M` then `2.30M` pos/s on repeat runs;
  - bs32k reported `2.39M` pos/s;
  - bs65k reported `2.51M` pos/s;
  - bs131k reported `2.55M` pos/s.
- Held-out validation check used only `C:\shogi\teacher\test\yamaoka-floodgate.psv`:
  - bs16k/4M with the same WRM/optimizer settings and `--test-positions 8192 --test-sample sequential --test-batch-size 1024` reported training `throughput=2196263 pos/s`, then post-timer validation `test_value_loss=0.05183934`, `test_value_accuracy=0.6716309`;
  - the final numbered checkpoint and validation summary were written after the measured training line, confirming the direct-mode timing cleanup.
- Adopted pinned staged upload for the HalfKP C++ direct path:
  - added C++/CUDA pinned host buffers backed by `cudaMallocHost` and staged `f32` / `i32` upload entry points;
  - `NnueTrainStepRunner` now owns two upload slots, each with device batch buffers, pinned host staging buffers, upload-ready events, and compute-done events;
  - non-profiled HalfKP direct/prod steps upload the next batch through a separate upload context while the compute stream finishes the previous step; profiled steps keep the serial path for clean stage timing.
- HalfKP 4M speed after pinned staged upload on the same HCPE teacher and WRM settings:
  - bs16k reported `2311626` pos/s, similar to the previous overlap-only best;
  - bs65k improved from the overlap-only `~2.51M` pos/s line to `2752872` pos/s, and a validation run reported `2715894` pos/s with yamaoka `test_value_loss=0.05523509`, `test_value_accuracy=0.6442871`;
  - bs131k reported `2670155` pos/s, so the best speed point in this short 4M probe moved to bs65k;
  - bs16k with `--threads 16` reported `2389868` pos/s and yamaoka `test_value_loss=0.05368488`, `test_value_accuracy=0.6662598`, but the bs16k quality probe still needs a larger validation/sample comparison before changing the recommended quality recipe.
- Short 4M speed/quality recipe sweep with validation fixed to `C:\shogi\teacher\test\yamaoka-floodgate.psv`, `--test-positions 65536 --test-sample sequential --test-batch-size 4096`:
  - bs16k, `beta1=0.975`, LR `0.024`: `2315240` pos/s, `test_loss=0.03574366`, `test_acc=0.6266327`;
  - bs32k, `beta1=0.975`, LR `0.024`: `2478123` pos/s, `test_loss=0.03513663`, `test_acc=0.6267548`;
  - bs49k, `beta1=0.975`, LR `0.024`: `2627500` pos/s, `test_loss=0.03489432`, `test_acc=0.6256714`;
  - bs65k, `beta1=0.975`, LR `0.024`: `2741555` pos/s, `test_loss=0.03517838`, `test_acc=0.6232605`;
  - bs32k LR probes: LR `0.032` regressed to `test_loss=0.03571033`, `test_acc=0.6262054`; LR `0.018` regressed to `test_loss=0.03567243`, `test_acc=0.6249542`;
  - bs32k beta probes at LR `0.024`: `beta1=0.99` gave `test_loss=0.03541317`, `test_acc=0.6266327`; `beta1=0.9` gave the best short-run accuracy, `test_loss=0.03685767`, `test_acc=0.6276093`.
- Current short 4M guidance after the sweep:
  - speed-only probe: bs65k / `beta1=0.975` / LR `0.024`;
  - balanced loss+speed probe: bs32k / `beta1=0.975` / LR `0.024`;
  - accuracy-biased short probe: bs32k / `beta1=0.9` / LR `0.024`, with the caveat that its held-out loss is worse and needs a longer/full-teacher check before becoming the production recommendation.
- Adopted C++/CUDA teacher CPU default auto-tuning:
  - the root CLI keeps the generic `--threads`, `--loader-threads`, and `--batch-queue-size` flags, but the Windows-native `--backend cuda-cpp` path now maps their historical defaults to GPU-feeding defaults before constructing the HalfKP/SFNN teacher batch configs;
  - default `--threads 4` becomes `available_parallelism() * 2`, clamped to `4..=24`; default `--loader-threads 0` uses the same value; default `--batch-queue-size 32` becomes `4` for HalfKP;
  - explicit non-default values are preserved, so manual A/B probes such as `--threads 10` or `--batch-queue-size 8` still do exactly what they say.
- HalfKP 4M CPU-feed sweep after the auto-tune change, same HCPE teacher and WRM settings (`bs65k`, `beta1=0.975`, LR `0.024`, `wd=0`, final-only loss readback):
  - `threads=10`, loader auto (`12`), queue `2`: `2499720` pos/s;
  - `threads=16`, `loader_threads=16`, queue `4`: `2589646` pos/s;
  - `threads=20`, `loader_threads=20`, queue `4`: `2617199` pos/s;
  - `threads=24`, `loader_threads=24`, queue `4`: best short run `2875248` pos/s;
  - `threads=32`, `loader_threads=32`, queue `4`: regressed to `2621696` pos/s;
  - `threads=24`, `loader_threads=24`, queue `8`: `2810461` pos/s; queue `32`: `2767411` pos/s.
- Default auto-tune validation:
  - with no explicit `--threads`, `--loader-threads`, or `--batch-queue-size`, the CLI printed `cuda-cpp teacher CPU = prepare_threads=24, loader_threads=24, batch_queue_size=4`;
  - repeat 4M speed-only run reported `2840476` pos/s;
  - held-out yamaoka check used only `C:\shogi\teacher\test\yamaoka-floodgate.psv` with `--test-positions 65536 --test-sample sequential --test-batch-size 4096` and reported `2627038` pos/s, `test_loss=0.03545475`, `test_acc=0.6218414`.
- Adopted HalfKP sparse L0 zero-gradient atomic skip:
  - the training-only `nnue_l0_sparse_backward_kernel` now checks each STM/NSTM CReLU pre-gradient before issuing the `atomicAdd` into `l0w_gradients`;
  - exact zero and `-0.0` gradients skip the atomic, while `NaN != 0.0f` still takes the atomic path, preserving the previous non-finite propagation behavior;
  - bs65k/6-step profile before the skip, same HCPE teacher and WRM settings: upload `3.261ms`, forward `3.026ms`, loss `1.232ms`, backward `12.744ms`, update `1.297ms`, total `21.561ms`;
  - after the skip: upload `3.047ms`, forward `2.954ms`, loss `1.177ms`, backward `12.209ms`, update `1.297ms`, total `20.685ms`, a backward-stage improvement of about `4.2%`.
- HalfKP post-skip short/longer probes, all with validation fixed to `C:\shogi\teacher\test\yamaoka-floodgate.psv` when validation was enabled:
  - 4M yamaoka run (`--test-positions 65536 --test-sample sequential --test-batch-size 4096`) reported `2576342` pos/s, `test_loss=0.03539697`, `test_acc=0.6243286`;
  - separate 4M speed-only repeat reported `2788019` pos/s;
  - 16M speed-only run reported `16777216` positions in `5.855s`, `2865226` pos/s;
  - 16M yamaoka validation summary logged `test_loss=0.034298`, `test_acc=0.631088`, `train_loss=0.046177`.
- Status against the previous cuda-oxide 4M ceiling:
  - BO-CUDA-028's clean cuda-oxide 4M recipe mean remains `2978387` pos/s with `test_acc=0.669474`, `test_loss=0.052934`;
  - the Windows-native cuda-cpp HalfKP path is already above the BO-CUDA-029 tatara idle mean, but this post-skip increment still does not clear that older cuda-oxide short-run speed ceiling on the current bs65k recipe.
- Adopted a WRM score-target lookup table in the shared teacher prepare path:
  - when `use_win_rate_model && !wdl`, `DefaultDataLoader` now warms a static 65536-entry i16 score table before training-time batch preparation;
  - this removes the two per-position `exp()` calls from the WRM score target calculation while preserving the same f32 formula and the same CPU-prepared targets;
  - HalfKP bs65k after warmup reported `2795413` pos/s on a 4M speed-only probe and `2805348` pos/s on a 16M speed-only probe;
  - the yamaoka-fixed 4M validation run (`--test-positions 65536 --test-sample sequential --test-batch-size 4096`) reported `2819074` pos/s, `test_loss=0.03546835`, `test_acc=0.6233521`;
  - this is a useful feed-side cleanup, but by itself still does not clear the old BO-CUDA-028 cuda-oxide 4M mean.
- Removed redundant teacher tail-fill writes in `PreparedData::new_with_pool`:
  - `stm` and `nstm` sparse buffers are already allocated with `-1` in every slot, so the per-position tail loops that re-wrote unused slots to `-1` were unnecessary;
  - HalfKP bs65k probes remained speed-neutral within run-to-run variance: 4M speed-only `2794386` pos/s and 16M speed-only `2788785` pos/s;
  - the yamaoka-fixed 4M validation run reported `2792950` pos/s, `test_loss=0.03538862`, `test_acc=0.6237640`.
- Skipped `out.bucket(pos)` calls for single-bucket teacher preparation:
  - when `OutputBuckets::BUCKETS == 1`, the preallocated `buckets` vector is already all zeros, so the per-position bucket call/write is unnecessary;
  - HalfKP bs65k probes again stayed speed-neutral: 4M speed-only `2802036` pos/s and 16M speed-only `2783633` pos/s;
  - the yamaoka-fixed 4M validation run reported `2794712` pos/s, `test_loss=0.03534004`, `test_acc=0.6218414`.
- Exposed teacher prepare profiling for the Windows-native backend:
  - new CLI flag: `--cuda-cpp-profile-teacher-prepare`;
  - it passes the existing `profile_prepare` path into HalfKP/SFNN teacher batch configs and disables the prepared-batch producer queue for clearer per-batch CPU timings;
  - HalfKP bs65k/3-step smoke on `shuffled-001.hcpe` printed `profile_teacher` prepare times of `15.400ms`, `18.732ms`, and `14.520ms` per batch.
- Adopted a non-factorized HalfKP direct teacher preparation path:
  - `ShogiHalfKP` now exposes `fill_halfkp_feature_indices`, and `HalfkpTeacherBatchConfig::ft_factorize=false` materializes `FastBatchHost` directly instead of routing through generic `PreparedData`;
  - the factorized HalfKP path still uses the generic `Factorised<ShogiHalfKP, ShogiHalfKPPieceFactorizer>` materializer;
  - HalfKP bs65k/3-step teacher-prepare profile after the direct path printed `16.119ms`, `14.930ms`, and `14.184ms`;
  - speed-only probes reported 4M `2824693` pos/s and 16M `2828516` pos/s;
  - the yamaoka-fixed 4M validation run reported `2693410` pos/s, `test_loss=0.03538197`, `test_acc=0.6238556`.
- Adopted direct PackedSfen-to-HalfKP sparse-index mapping:
  - `ShogiHalfKP::map_features` and the non-factorized `fill_halfkp_feature_indices` path now decode the `PackedSfenValue` bitstream directly instead of constructing a temporary `ShogiBoard`;
  - the old board-based mapper remains test-only, and a synthetic PackedSfen unit test covers board pieces, promoted pieces, hands, and both STM colours to assert exact sparse-index order parity;
  - HalfKP bs65k/3-step teacher-prepare profile on `shuffled-001.hcpe` printed `14.056ms`, `15.303ms`, and `18.178ms`, so the short prepare-only profile is mostly noise-neutral;
  - WRM/tatara-style bs65k speed-only probes (`--nnue-pytorch-wrm-loss --optimizer-weight-decay 0 --optimizer-beta1 0.975 --lr 0.024`, final-only loss readback) reported 4M repeats `2758937` and `2834683` pos/s, and a longer 16M run reported `2920370` pos/s;
  - a yamaoka-fixed 4M validation run using only `C:\shogi\teacher\test\yamaoka-floodgate.psv` reported `2655504` pos/s, `test_loss=0.03627902`, `test_acc=0.6215057`; this short quality point is slightly worse than the preceding best and needs a longer/seed-matched check before treating the mapper change as a quality improvement.
- Added SFNN C++/CUDA backward stage profiling for the direct train-step runner:
  - `SfnnTrainStepRunner::step_profiled_no_readback` now returns an SFNN-specific profile that includes C++/CUDA event timings for zero, L3, L2, L2-input, L1, pairwise/L0, and total backward stages;
  - the normal training FFI entry point remains unchanged, and the profiling-only FFI wrapper records CUDA events only when `--cuda-cpp-profile-steps` uses the profiled SFNN path;
  - real SFNN WRM profile on RTX 4090, `shuffled-001.hcpe`, `SFNN_halfka2_1024_7_64_k3k3`, factorized L1, bs131k, 6 profiled steps reported `throughput=1050585` pos/s including profile/event synchronization overhead;
  - average top-level profile: upload `6.747ms`, forward `20.518ms`, loss `2.217ms`, backward `72.677ms`, update `5.360ms`, total `107.520ms`;
  - average C++/CUDA backward stage profile: zero `0.575ms`, L3 `0.625ms`, L2 `5.305ms`, L2-input `0.041ms`, L1 `7.738ms`, pairwise/L0 `53.644ms`, total `67.928ms`;
  - the SFNN optimisation target is therefore the fused pairwise/L0 sparse backward kernel first, not the L1 factorized scatter.
- Rejected experiments / cautions:
  - an entry-per-sparse-feature HalfKP L0 scatter kernel passed correctness but did not improve steady backward (`~3.10ms` remained unchanged), so it was not kept;
  - a fused HalfKP L0 CReLU+sparse-backward kernel reduced thread count on paper but regressed bs65k profiled backward from about `12.36ms` to `14.19ms`, so it was reverted;
  - a block-shared `l1=256` HalfKP L0 sparse-backward mapping experiment was not kept: the first attempt incorrectly launched only `ceil(entries/256)` blocks and visibly worsened 16M loss, and the corrected launch was slower than the zero-skip kernel (`12.636ms` vs `12.209ms` backward on the bs65k/6-step profile);
  - a HalfKP-factorized-only L0 scatter kernel that removed the generic factorized-feature branch passed the real-data smoke but did not improve the profiled backward stage (`12.351ms` vs the generic kernel's `12.208ms` on the bs65k/6-step WRM profile), so it was reverted;
  - a HalfKP upload-slot pipeline passed build/smoke but regressed the 4M run to about `1.30M` pos/s, so it was reverted;
  - `cargo test -p bulletou_lib teacher_batch -- --nocapture` passed, but an existing pack-loader background thread can still print a post-test panic after the harness reports success; this appears unrelated to the HCPE HalfKP C++ direct path.
- Validation for this partial BO-CUDA-036 increment:
  - `cargo check --features cuda-cpp-backend --example bulletou` passed;
  - `cargo test -p bulletou-cuda-cpp --lib` passed;
  - `cargo test -p bulletou-cuda-cpp --lib persistent_device_api_smoke -- --ignored --nocapture` passed;
  - `cargo test -p bulletou_lib shogi_halfkp -- --nocapture` passed;
  - `cargo test -p bulletou_lib value::loader -- --nocapture` passed;
  - `cargo run -p bulletou-cuda-cpp --bin bulletou-cuda-cpp-smoke` passed;
  - `cargo test --features cuda-cpp-backend --example bulletou cuda_cpp -- --nocapture` passed (39 cuda-cpp tests);
  - `cargo test -p bulletou_lib teacher_batch -- --nocapture` passed.
- Remaining BO-CUDA-036 work:
  - further sparse L0 backward/update optimisation if the previous cuda-oxide 4M ceiling remains the target;
  - longer multi-file confirmation of the auto-tuned CPU defaults, because 4M HCPE runs still show noticeable run-to-run variance.
