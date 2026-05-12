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

## Reference

Specification-level documents on the training pipeline. These assume you already know what you are doing.

- [NNUE Basics](1-basics.md) — input/hidden/output layers, perspective networks
- [Saved Networks](4-saved-networks.md) — checkpoint layout, SavedFormat, quantisation, transformation chains

Shogi-specific:

- [shogi/kppt.md](shogi/kppt.md) — KPPT / KPP_KKPT evaluation function training
- [shogi/kp-absolute-progress.md](shogi/kp-absolute-progress.md) — game-progress estimation using KP-Absolute features
- [shogi/shogi_progress_kpabs_train.md](shogi/shogi_progress_kpabs_train.md) — CLI spec of the `shogi_progress_kpabs_train` tool
