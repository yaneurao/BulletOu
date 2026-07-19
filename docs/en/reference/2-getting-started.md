# Getting Started

<a href="../../ja/reference/2-getting-started.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

### Installing Rust

Install Rust via [rustup](https://www.rust-lang.org/tools/install) (this is the official way to install rust).

### General Usage

You can use `bullet` as a crate:
```toml
bullet = { git = "https://github.com/jw1912/bullet", package = "bullet_lib" }
```
or by editing and running one of the [examples](../../examples):
```
cargo r -r --example <example name>
```

A basic inference example is included in [examples/simple](../../examples/simple.rs), and if you've never
trained an NNUE before it is recommended to start with an architecture and training schedule similar to it.

### Utilities

You can build `bullet-utils` with `cargo b -r --package bullet-utils`, to do the following:
- Convert between data formats
- Interleave multiple data files
- Shuffle data files
- Validate data files

Use `./target/release/bullet-utils[.exe] help` to see specific usage.

This does **not** require CUDA.

### Backend

BulletOu's maintained trainer backend is `cuda-cpp` for NVIDIA GPUs:

```bash
cargo build --release --features cuda-cpp-backend --example bulletou
```

- Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
- The `CUDA_PATH` environment variable must be set to the CUDA install location.
- ROCm/HIP support and the old `bullet-gpu` feature backends have been retired from the maintained build.
