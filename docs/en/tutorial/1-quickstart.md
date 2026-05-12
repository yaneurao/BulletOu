# 1. Quick Start — Build BulletOu and Run a Tiny Training

<a href="../../ja/tutorial/1-quickstart.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal of this page: from a fresh checkout, build BulletOu and run a short training session that produces a checkpoint file. If you reach the end, your toolchain is healthy.

We are **not** trying to train a strong NNUE here — that comes in the next page. This is a smoke test.

## 1.1 Prerequisites

You need:

- **A modern NVIDIA or AMD GPU.** CPU-only training is not supported (the GPU runtime is built into the design).
- **Rust toolchain** (stable, 1.87 or later). See §1.1.1 below for OS-specific install steps.
- **CUDA Toolkit 12.x** (for NVIDIA GPUs) or **HIP SDK / ROCm** (for AMD GPUs).
- About **10 GB of free disk space** for the build and test data.

If you are on Windows and using NVIDIA, you also need the matching versions of cuDNN (and optionally TensorRT). See the [reference doc on Windows GPU setup](https://github.com/yaneurao/BulletOu) when in doubt.

> **CPU-only?** There is a `mock` GPU backend in the source tree, but it is a type-checking stub — it cannot actually train. If you have no GPU, this tutorial is not going to work. Consider using a cloud GPU instance (e.g. Vast.ai, Lambda Labs, Paperspace, or Google Colab).

### 1.1.1 Installing the Rust toolchain

#### Windows

1. Download **rustup-init.exe** from <https://rustup.rs/> (the "DOWNLOAD RUSTUP-INIT.EXE (64-BIT)" button) and run it.
2. Accept the defaults:
   - `Default host triple: x86_64-pc-windows-msvc` (the `msvc` target works best with CUDA EP)
   - `Default toolchain: stable`
   - `Profile: default`
3. If rustup reports that MSVC C++ Build Tools are missing, follow its link to install **Visual Studio Build Tools** and check the **"Desktop development with C++"** workload.
4. **Open a new PowerShell or cmd window** so the PATH update is picked up.

One-line install via PowerShell (alternative):

```powershell
Invoke-WebRequest -Uri https://win.rustup.rs/x86_64 -OutFile rustup-init.exe
.\rustup-init.exe -y --default-host x86_64-pc-windows-msvc --default-toolchain stable
```

#### Linux / macOS / WSL

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"   # or just open a new shell
```

#### Verify

```bash
cargo --version
rustc --version
```

Both commands should print a version like `cargo 1.x.x ...`.

## 1.2 Get the source

```bash
git clone https://github.com/yaneurao/BulletOu.git
cd BulletOu
```

## 1.3 Build

Choose **one** of the following depending on your GPU:

```bash
# NVIDIA GPU (CUDA)
CUDA_PATH=/usr/local/cuda cargo build --release --features device-cuda
```

```bash
# AMD GPU (ROCm)
HIP_PATH=/opt/rocm cargo build --release --features device-rocm
```

(On Windows, set the env vars accordingly: `set CUDA_PATH=...` or use PowerShell `$env:CUDA_PATH=...`.)

The first build will take several minutes. If it succeeds without errors, you're ready.

### Common build problems

- **`CUDA_PATH is not defined`** — set the environment variable to your CUDA install path (e.g. `/usr/local/cuda`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`).
- **Linker error mentioning `cublas` / `nvrtc`** — the CUDA version may be too old. Use 12.x or newer.
- **Linker error mentioning `hipblas` / `hiprtc`** — install the HIP SDK, set `HIP_PATH`, and possibly set `GCN_ARCH_NAME` (find it with `rocminfo` on Linux or `hipinfo` on Windows).

## 1.4 Run a smoke-test training

The `simple` example trains a tiny chess (yes, chess) NNUE. It does not need any external data and only takes a few minutes. It exercises the full pipeline end-to-end, which is what we want at this stage.

```bash
# NVIDIA
cargo run --release --features device-cuda --example simple

# AMD
cargo run --release --features device-rocm --example simple
```

If everything is working, you will see output like:

```
... starting training ...
superbatch 1 ... loss = ...
superbatch 2 ...
...
```

and a `checkpoints/` directory containing the trained output is created.

> The `simple` example is **chess**, not shogi. It exists upstream and we keep it because it is the smallest end-to-end example. The shogi examples come next.

## 1.5 What just happened

You built BulletOu and ran a complete training session. The pipeline that ran is:

1. Built a tiny NNUE in `simple.rs` (chess `Chess768` input feature → 1 small hidden layer → 1 scalar output)
2. Loaded data from a small bundled file (in chess `bulletformat`)
3. Trained for a few superbatches
4. Wrote checkpoints

In the next page, we replace the chess input feature with a **shogi** one and use a real `.pack` dataset. Same pipeline; different feature set and data loader.

## 1.6 Cleanup

When you're done, you can delete the `checkpoints/` directory and the `target/` build artifacts:

```bash
rm -rf checkpoints target
```

The `target/` directory will be rebuilt next time you `cargo build`.

---

Next: [2. Running a training](2-nnue-tutorial.md) — train a real evaluation function on actual data and load it into an engine.

---

<details>
<summary>1.7 Developer notes (optional)</summary>

Material beyond the smoke test, for when you start adapting BulletOu to your own training. **Safe to skip on first read.**

### Editing an existing example

The usual workflow: clone the repo and edit one of the files under [examples/](/examples) to your taste. `shogi_simple.rs` or `bulletou.rs` are common starting points.

### Registering a custom example

Just placing a new file under `examples/` is not enough — `cargo build --example xxx` will not find it until you register it in `bullet_lib`'s `Cargo.toml`:

```toml
# Append to crates/bullet_lib/Cargo.toml
[[example]]
name = "my_example"
path = "../../examples/my_example.rs"
```

Once registered, the example survives `git pull` from upstream more easily, which helps when maintaining long-running custom experiments.

### Importing `bullet_lib` from another project

You can also depend on `bullet_lib` as a crate from a separate project:

```toml
[dependencies]
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bullet_lib" }
```

### API documentation

Detailed API documentation lives in Rust's docstrings. To generate and open it locally:

```bash
cargo doc --open
```

</details>
