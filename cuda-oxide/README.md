# BulletOu cuda-oxide Workspace

This nested workspace is reserved for the NNUE / SFNN fast backend.

It is intentionally separate from the root BulletOu workspace:

- root workspace remains the existing generic Bullet backend
- cuda-oxide workspace may require nightly Rust, `cargo-oxide`, LLVM 21+, and Linux / WSL2
- cuda-oxide dependencies must not be added to the root workspace dependency set

Initial scope:

1. load an already-generated PTX file
2. launch a smoke kernel
3. add fixed-layout NNUE / SFNN kernels one by one

Current smoke command:

```bash
cargo run -p bulletou-cuda-train --features cuda
```

This loads `smoke/noop.ptx`, resolves the `noop` kernel symbol, launches it,
and checks a host-device-host buffer round trip. A custom PTX module can be
tested with:

```bash
cargo run -p bulletou-cuda-train --features cuda -- --ptx /path/to/module.ptx --kernel noop
```

CO-006 NNUE forward smoke compares a fixed NNUE batch against a CPU scalar
golden. The default case is the small `tiny` shape:

```bash
cargo run -p bulletou-cuda-train --features cuda -- --nnue-forward-smoke --ptx /path/to/bulletou-cuda-train.ptx
```

Use `--nnue-forward-case halfkp` to exercise the full
`NNUE_HALFKP_256x2_32_32` layout with deterministic synthetic weights and
sparse indices:

```bash
cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-forward-smoke --nnue-forward-case halfkp --ptx /path/to/bulletou-cuda-train.ptx
```

`--write-nnue-forward-fixture <PATH>` writes the selected smoke case as a simple
little-endian fixture. `--nnue-forward-fixture <PATH>` reads the same format.
This is the bridge point for root BulletOu code to export a `FastBatchHost` plus
`NnueForwardOwnedWeights` without making this nested workspace depend on the
root workspace:

```bash
cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-forward-smoke --nnue-forward-case halfkp \
  --write-nnue-forward-fixture /tmp/bulletou-halfkp.nnuef \
  --ptx ./bulletou_cuda_train.ptx

cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-forward-smoke --nnue-forward-fixture /tmp/bulletou-halfkp.nnuef \
  --ptx ./bulletou_cuda_train.ptx --debug-readback
```

Add `--debug-readback` to compare L0 / concat / hidden buffers as well as the
final output.

CO-010 Ranger step smokes run the full fixed-layout forward/backward path and
then update every parameter group through the grouped Ranger launcher:

```bash
cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-ranger-step-smoke --nnue-forward-case halfkp

cargo run -p bulletou-cuda-train --features cuda -- \
  --sfnn-ranger-step-smoke --sfnn-forward-case halfka2
```

For the current Windows + WSL2 development setup, the root workspace can export
a real HCPE teacher batch and immediately validate the cuda-oxide NNUE
forward -> loss -> backward -> Ranger update path with:

```powershell
powershell -ExecutionPolicy Bypass -File ..\scripts\cuda_oxide_nnue_teacher_smoke.ps1 `
  -Teacher C:\path\to\teacher.hcpe
```

If `-Teacher` is omitted, the script uses the first `.hcpe` file under
`C:\shogi\teacher\yane-distill-hcpe-20260508shuffled` when that directory
exists. The script runs the root exporter with `--train-fixture`, writes an
ignored `BOUNTRN1` train fixture under `target/cuda-oxide-fixtures/`, then runs
`bulletou-cuda-train --nnue-loss-ranger-step-smoke --nnue-train-fixture` in
WSL2. Add `-LossKind sigmoid-mse` or `-LossKind wrm` to select the loss smoke,
`-TrainSteps 2` to export an initial `BOUNTRN1` fixture plus subsequent
batch-only `BOUNBCH1` fixtures and run multiple optimizer steps, and
`-SkipCudaBuild` to reuse an existing `cargo oxide build` artifact.

The fixture-backed trainer loop is available as `--nnue-fixture-train`. It can
write a trained forward fixture with `--write-nnue-trained-forward-fixture` and
a cuda-oxide Ranger checkpoint fixture with `--write-nnue-train-state-fixture`.
The latter is restored with `--nnue-train-state-fixture`, after which supplied
later batch fixtures are applied from `completed_steps + 1`.
The helper script exposes the restore path as `-ResumeTrainStateFixture`: it
reads `completed_steps` from `BOUNRNG1`, exports only later teacher batches, and
runs the fixture-backed resume loop.

For the first direct loader bridge, build with `--features cuda,root-loader` and
run `--nnue-teacher-train`. This makes the nested cuda-oxide binary depend on
root `bulletou_lib` only on demand, reads real teacher batches directly into
`FastBatchHost`, and feeds them to the NNUE loss/Ranger runner without writing
intermediate `BOUNTRN1`/`BOUNBCH1` files. Multiple batches are streamed through
one loader pass:

```bash
cargo run -p bulletou-cuda-train --features cuda,root-loader --release -- \
  --nnue-teacher-train \
  --teacher /mnt/c/shogi/teacher/yane-distill-hcpe-20260508shuffled/shuffled-001.hcpe \
  --weights-bin /mnt/c/path/to/checkpoint/state.bin \
  --output /mnt/c/path/to/cuda-oxide-checkpoints \
  --train-steps 2 --batch-size 2 --buffer-mb 1 --loader-threads 1 --threads 1
```

Pass `--nnue-train-state-fixture <BOUNRNG1>` to resume direct teacher training.
In this mode `--train-steps` is the number of additional batches to run, and
the loader starts at `completed_steps` from the state fixture.
Pass `--output <DIR>` to write a numbered cuda-oxide bridge checkpoint:
`<DIR>/0001/nn.bin`, `<DIR>/0001/trained-forward.nnuef`,
`<DIR>/0001/state.boung`, `<DIR>/0001/state.bin`, and
`<DIR>/0001/learn.log`. `nn.bin` is the YaneuraOu/Stockfish-style quantized
HalfKP network; `state.boung` is the cuda-oxide exact resume artifact;
`state.bin` is the root BulletOu record stream for NNUE weights, momentum,
velocity, Ranger slow weights, and step counters. HCPE runs also write
`dataloader_pos.txt` using the same fixed-record offset convention as the
current direct resume path, and append one row to top-level
`summary-learn.log`. If `<DIR>` already contains numbered bridge checkpoints,
the direct trainer automatically restores the latest `state.boung` and writes
the next number. If that checkpoint has `dataloader_pos.txt`, HCPE input resumes
from its byte offset through the loader's exact resume path. In that case omit
`--weights-bin`, because the restored train state already carries weights and
optimizer state.

The PowerShell helper can run this extra path with `-RunDirectTeacherTrain`.
Use `-DirectTrainedForwardFixture <PATH>` or `-DirectTrainStateFixture <PATH>`
to save its readback fixtures. If `-ResumeTrainStateFixture` is also supplied,
the direct path resumes from the same BOUNRNG1 state and runs the remaining
batches up to `-TrainSteps`. Use `-WeightsBin <PATH>` to initialise fresh
fixture/direct runs from root weights. Use `-Output <DIR>` to pass the bridge
checkpoint output directory to the direct path.

WSL2 Ubuntu 24.04 validation example:

```bash
export CUDA_HOME=/usr
export CUDA_PATH=/usr
export CUDA_TOOLKIT_PATH=/usr
export CUDA_OXIDE_LIBDEVICE=/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc
export LIBCLANG_PATH=/usr/lib/llvm-20/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CARGO_TARGET_DIR=/tmp/bulletou-cuda-target

cargo oxide setup
cargo oxide build --arch sm_89 --features cuda \
  --cargo-target-dir "$CARGO_TARGET_DIR" -- --package bulletou-cuda-train
cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-forward-smoke --nnue-forward-case halfkp --ptx ./bulletou_cuda_train.ptx --debug-readback
```

`cargo oxide build` writes `bulletou_cuda_train.{ll,opt.ll,ptx}` into this
workspace root. These are generated artifacts and are intentionally ignored.
Large forward/train fixtures are also ignored.

The default build intentionally does not enable CUDA:

```bash
cargo check
```

Use the CUDA feature only on a machine with a CUDA Toolkit install root visible
through `CUDA_HOME`, `CUDA_PATH`, or `CUDA_TOOLKIT_PATH`.
