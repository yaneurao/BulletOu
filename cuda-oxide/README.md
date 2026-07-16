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

CO-006 NNUE forward smoke compares a tiny fixed NNUE batch against a CPU scalar
golden:

```bash
cargo run -p bulletou-cuda-train --features cuda -- --nnue-forward-smoke --ptx /path/to/bulletou-cuda-train.ptx
```

Add `--debug-readback` to compare L0 / concat / hidden buffers as well as the
final output.

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
  --nnue-forward-smoke --ptx ./bulletou_cuda_train.ptx --debug-readback
```

`cargo oxide build` writes `bulletou_cuda_train.{ll,opt.ll,ptx}` into this
workspace root. These are generated artifacts and are intentionally ignored.

The default build intentionally does not enable CUDA:

```bash
cargo check
```

Use the CUDA feature only on a machine with a CUDA Toolkit install root visible
through `CUDA_HOME`, `CUDA_PATH`, or `CUDA_TOOLKIT_PATH`.
