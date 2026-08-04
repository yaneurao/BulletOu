# 1. Quick Start — Build BulletOu and Run a Tiny Training

<a href="../../ja/tutorial/1-quickstart.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal of this page: from a fresh checkout, build BulletOu and run a short training session that produces a checkpoint file. If you reach the end, your toolchain is healthy.

We are **not** trying to train a strong NNUE here — that comes later. This is a smoke test.

## 1.1 Prerequisites

You need:

- **A modern NVIDIA GPU.** CPU-only training is not supported (the maintained trainer backend uses CUDA).
- **Rust toolchain** (stable, 1.87 or later). See §1.1.1 below for OS-specific install steps.
- **CUDA Toolkit 12.x**.
- About **10 GB of free disk space** for the build and test data.

On Windows, you also need the MSVC C++ build tools visible to the shell that runs Cargo.

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

Build the maintained BulletOu trainer backend:

```bash
# NVIDIA GPU (CUDA 12.x)
cargo build --release --features cuda-cpp-backend --example bulletou
```

On Windows, make sure the CUDA Toolkit and a matching Visual Studio C++ toolchain are available in the build environment.

The first build will take several minutes. If it succeeds without errors, you're ready.

### Common build problems

- **`CUDA_PATH is not defined`** — set the environment variable to your CUDA install path (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x` or `/usr/local/cuda`).
- **`nvcc` / MSVC build errors on Windows** — install the CUDA Toolkit and Visual Studio C++ build tools, then run from a shell where both are visible.
- **Linker error mentioning CUDA runtime libraries** — use CUDA Toolkit 12.x or newer.

## 1.4 Run a smoke-test training

The CUDA C++ smoke test does not need teacher data. It checks that the backend can initialize CUDA, launch kernels, and run a tiny Ranger update.

```bash
cargo run --release --features cuda-cpp-backend --example bulletou -- --cuda-cpp-smoke
```

If everything is working, you will see output like:

```
... starting training ...
superbatch 1 ... loss = ...
superbatch 2 ...
...
```

and a `checkpoints/` directory containing the trained output is created.

> The `simple` example is **chess**, not shogi. It exists upstream and we keep it because it is the smallest end-to-end example. The shogi examples come later.

## 1.5 What just happened

You built BulletOu and ran a complete training session. The pipeline that ran is:

1. Built a tiny NNUE in `simple.rs` (chess `Chess768` input feature → 1 small hidden layer → 1 scalar output)
2. Loaded data from a small bundled file (in chess `bulletformat`)
3. Trained for a few superbatches
4. Wrote checkpoints

Later we'll swap the chess input feature for a **shogi** one and feed it a real `.pack` dataset. Same pipeline; different feature set and data loader.

## 1.6 Cleanup

When you're done, you can delete the `checkpoints/` directory and the `target/` build artifacts:

```bash
rm -rf checkpoints target
```

The `target/` directory will be rebuilt next time you `cargo build`.

---

Next:
- [2. Using `bulletou_lib` from your own code](2-bullet-lib.md) — developer notes (optional)
- Or jump straight to [3. Prepare training data](3-data.md)
