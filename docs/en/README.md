# BulletOu Documentation

<a href="../ja/"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Documentation is split into two layers:

- **Tutorial** — step-by-step guides for getting started. Read these first if you are new to BulletOu.
- **Reference** — specifications and design details. Read these when you need to understand or modify specific behaviour.

---

## Tutorial (start here)

See [`tutorial/`](tutorial/) for the full table of contents.

1. [Overview — what BulletOu trains, supported evaluation function families](tutorial/0-overview.md)
2. [Quick Start — install, build, and run your first training session](tutorial/1-quickstart.md)
3. [NNUE Tutorial — a deeper walkthrough of training a shogi NNUE](tutorial/2-nnue-tutorial.md)
4. [KPPT / KPP_KPPT Roadmap — current state and what's planned for legacy evaluation functions](tutorial/3-kppt-roadmap.md)

## Reference

Specification-level documents on the training pipeline. These assume you already know what you are doing.

NNUE training internals:

1. [NNUE Basics](1-basics.md) — input/hidden/output layers, perspective networks, common pitfalls
2. [Getting Started](2-getting-started.md) — rustup, examples, bullet-utils, backend setup
3. [Training Data](3-data.md) — workflow, builtin data loaders, ChessBoard / binpack formats
4. [Saved Networks](4-saved-networks.md) — checkpoint layout, SavedFormat, quantisation, transformation chains

Shogi-specific:

- [shogi/kp-absolute-progress.md](shogi/kp-absolute-progress.md) — game-progress estimation using KP-Absolute features
- [shogi/shogi_progress_kpabs_train.md](shogi/shogi_progress_kpabs_train.md) — CLI spec of the `shogi_progress_kpabs_train` tool

KPPT / KPP_KPPT:

- Currently not implemented. See [tutorial/3-kppt-roadmap.md](tutorial/3-kppt-roadmap.md) for the design plan.

## Examples

Practical examples to copy-edit are in the [`examples/`](../../examples/) directory, including the progression series ([`examples/progression/`](../../examples/progression/)) and the shogi-specific examples (`shogi_simple`, `shogi_layerstack`, etc.). The tutorial walks through several of these.
