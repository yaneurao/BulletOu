<div align="center">

# BulletOu

<a href="README-ja.md"><img alt="日本語で読む" src="https://img.shields.io/badge/README-日本語-DC2626?style=flat-square"></a>

</div>

A domain-specific ML library for training NNUE-style value networks for **shogi (将棋)** engines, in particular for use with [YaneuraOu](https://github.com/yaneurao/YaneuraOu).

BulletOu is yaneurao's fork of [bullet-shogi](https://github.com/SH11235/bullet-shogi), which is in turn a shogi-oriented fork of [bullet](https://github.com/jw1912/bullet) by jw1912. The original `bullet` is a Rust-based NNUE trainer with best-in-class GPU performance, widely used by top chess engines.

### Lineage / Upstream

- **Original**: [jw1912/bullet](https://github.com/jw1912/bullet) — general-purpose NNUE trainer (chess-focused)
- **Upstream**: [SH11235/bullet-shogi](https://github.com/SH11235/bullet-shogi) — shogi support, PackedSfenValue loader, HalfKA / HalfKP / Threat / HandThreat features, Layer Stack with KP-Absolute progress buckets
- **This repository**: [yaneurao/BulletOu](https://github.com/yaneurao/BulletOu) — yaneurao's adaptation for YaneuraOu

Upstream changes can be pulled in with:

```bash
git remote add upstream https://github.com/SH11235/bullet-shogi.git
git fetch upstream
git merge upstream/shogi-support
```

### Usage for NNUE / Value Network Training

Before using, read the documentation at [docs/en/0-contents.md](docs/en/0-contents.md), which covers building, managing training data, and the network output format.

Most users clone the repo and edit one of the [examples](/examples) to their taste. If you want a custom example file that survives upstream pulls, register it in [`bullet_lib`'s `Cargo.toml`](crates/bullet_lib/Cargo.toml).

Alternatively, import the `bullet_lib` crate with:

```toml
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bullet_lib" }
```

API documentation is covered by Rust's docstrings. Generate local documentation with `cargo doc`.

### Building

```bash
# NVIDIA GPU (CUDA 12.x + cuDNN 9.x)
CUDA_PATH=/usr/local/cuda cargo build --release --features cuda

# AMD GPU (ROCm)
HIP_PATH=/opt/rocm cargo build --release --features rocm
```

CPU-only execution is not supported (the mock GPU runtime is for type-checking only).

### Documentation

- English: [docs/en/](docs/en/)
- 日本語: [docs/ja/](docs/ja/)

### License

MIT, inherited from upstream. Original copyrights are preserved in `LICENSE`.

### Help / Feedback

- File issues at <https://github.com/yaneurao/BulletOu/issues>
- For general bullet-related discussion (upstream / chess), see the `#bullet` channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) Discord server.
