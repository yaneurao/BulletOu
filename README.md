<div align="center">

# BulletOu

<a href="README-ja.md"><img alt="日本語で読む" src="https://img.shields.io/badge/README-日本語-DC2626?style=flat-square"></a>

</div>

A Rust-based domain-specific ML library for training shogi AI evaluation function parameters. Designed for training evaluation function parameters used by [YaneuraOu](https://github.com/yaneurao/YaneuraOu).

Target evaluation functions:

- NNUE halfKP
- NNUE KP
- NNUE halfka1 / halfka2
- SFNN + layerstack9 (NNUEwoSQPT1536)
- KPPT
- KPP_KPPT

### Usage

**First-time users** should start with the [tutorial (docs/en/tutorial/)](docs/en/tutorial/), which walks through installation, building, and running a first training session step by step.

For specification-level details such as training-data formats and output formats, see [docs/en/](docs/en/).

### Building

```bash
# NVIDIA GPU (CUDA 12.x + cuDNN 9.x)
CUDA_PATH=/usr/local/cuda cargo build --release --features cuda

# AMD GPU (ROCm)
HIP_PATH=/opt/rocm cargo build --release --features rocm
```

CPU-only training is not supported (the mock GPU runtime is a type-checking stub only).

### Documentation

See [docs/en/](docs/en/).


### Lineage / Upstream

- **Original**: [jw1912/bullet](https://github.com/jw1912/bullet) — general-purpose NNUE trainer (chess-focused)
- **Upstream**: [SH11235/bullet-shogi](https://github.com/SH11235/bullet-shogi) — shogi fork. `.pack` loader (decodes YaneuraOu `gensfen`'s per-game variable-length file into a `PackedSfenValue` stream internally), HalfKA / HalfKP / Threat / HandThreat features, Layer Stack with KP-Absolute progress buckets
- **This repository**: [yaneurao/BulletOu](https://github.com/yaneurao/BulletOu) — yaneurao's adaptation for YaneuraOu


### License

MIT (inherited from upstream). Original copyright notices are preserved in `LICENSE`.

### Help / Feedback

- File issues at <https://github.com/yaneurao/BulletOu/issues>
- For general bullet-related discussion (upstream / chess), see the `#bullet` channel in the [Engine Programming](https://discord.com/invite/F6W6mMsTGN) Discord server.
