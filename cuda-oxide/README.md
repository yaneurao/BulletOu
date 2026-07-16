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

Until the runtime is connected, `bins/bulletou-cuda-train` is a fail-fast
placeholder.
