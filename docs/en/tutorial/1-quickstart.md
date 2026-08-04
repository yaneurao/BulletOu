# 1. Quick Start

<a href="../../ja/tutorial/1-quickstart.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: build BulletOu and verify that the CUDA backend runs.

## 1.1 Requirements

- NVIDIA GPU
- CUDA Toolkit 12.x
- Rust stable
- Visual Studio C++ Build Tools on Windows

CPU-only training is not supported.

Install Rust from <https://rustup.rs/>. After installation, open a new PowerShell and check:

```powershell
cargo --version
rustc --version
```

## 1.2 Get the source

```powershell
git clone https://github.com/yaneurao/BulletOu.git
cd BulletOu
```

Skip this if you already have a checkout.

## 1.3 Build

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
```

On Windows, the executable is:

```text
.\target\release\examples\bulletou.exe
```

## 1.4 CUDA smoke test

You can check CUDA initialization and a tiny training kernel without teacher data.

```powershell
cargo run --release --features cuda-cpp-backend --example bulletou -- --cuda-cpp-smoke
```

If it exits without errors, your build environment is basically working.

## 1.5 Common build errors

| Error | Check |
| --- | --- |
| `CUDA_PATH is not defined` | CUDA Toolkit install path is in the environment |
| `nvcc` not found | Reopen PowerShell after installing CUDA Toolkit |
| MSVC-related errors | Visual Studio C++ Build Tools are installed |

---

Next: [2. Prepare training data](2-data.md)
