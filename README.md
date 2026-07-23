<p align="center">
  <img src="docs/images/bulletou-logo-mascot-s0.png" alt="BulletOu UltraFast Shogi AI ML" width="480px">
</p>

<div align="center">

  <h1>BulletOu</h1>

  A Rust-based domain-specific ML library for training shogi AI evaluation function parameters. Designed for training evaluation function parameters used by [YaneuraOu](https://github.com/yaneurao/YaneuraOu).

<a href="README-ja.md"><img alt="日本語で読む" src="https://img.shields.io/badge/README-日本語-DC2626?style=flat-square"></a>

</div>


Target evaluation functions:

- KPPT
- KPP_KKPT
- NNUE_HALFKP
- NNUE_KP
- NNUE_KA2
- NNUE_HALFKPE9
- NNUE_HALFKPVM
- SFNN_HALFKA1HM
- SFNN_HALFKA2HM
- SFNN_HALFKA2
- SFNN_KA2

Target LayerStacks:

- k3k3(king3-by-king3)
- k9k9(king9-by-king9)
- hand64
- hand64_k3k3
- hand64_k9k9


### Usage

- [Tutorial](docs/en/tutorial/): walks through installation, building, and running a first training session step by step.
- [Documentation](docs/en/): specification-level details such as training-data formats and output formats.


### Building

```bash
# NVIDIA GPU (CUDA 12.x; Windows-native CUDA C++ backend)
cargo build --release --features cuda-cpp-backend --example bulletou
```

CPU-only training is not supported. The maintained BulletOu trainer backend is `cuda-cpp`.


### Lineage / Upstream

- **Original**: [jw1912/bullet](https://github.com/jw1912/bullet) — general-purpose NNUE trainer (chess-focused)
- **Upstream**: [SH11235/bullet-shogi](https://github.com/SH11235/bullet-shogi) — shogi fork. `.pack` loader (decodes YaneuraOu `gensfen`'s per-game variable-length file into a `PackedSfenValue` stream internally), HalfKA / HalfKP / Threat / HandThreat features, Layer Stack with KP-Absolute progress buckets
- **This repository**: [yaneurao/BulletOu](https://github.com/yaneurao/BulletOu) — yaneurao's adaptation for YaneuraOu


### License

MIT (inherited from upstream). Original copyright notices are preserved in `LICENSE`.
