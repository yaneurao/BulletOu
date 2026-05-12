# BulletOu Reference

<a href="../ja/"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Specification-level documents on the training pipeline. These assume you already know what you are doing.

If you are new to BulletOu, start with the [tutorial](tutorial/).

- [NNUE Basics](reference/1-basics.md) — input/hidden/output layers, perspective networks
- [Getting started with BulletOu](reference/2-getting-started.md) — high-level training pipeline overview (upstream-derived)
- [Training data formats](reference/3-data.md) — bulletformat / .pack / .hcpe / .hcpe3 / .psv (upstream + shogi extensions)
- [Saved Networks](reference/4-saved-networks.md) — checkpoint layout, SavedFormat, quantisation, transformation chains

Shogi-specific:

- [shogi/halfkp.md](shogi/halfkp.md) — NNUE HalfKP evaluation function training
- [shogi/halfkpe9.md](shogi/halfkpe9.md) — NNUE HalfKPE9 evaluation function training (HalfKP plus per-square attacker-count buckets)
- [shogi/kp.md](shogi/kp.md) — NNUE K-P evaluation function training (same network as HalfKP with a different input)
- [shogi/kppt.md](shogi/kppt.md) — KPPT / KPP_KKPT evaluation function training
- [shogi/kp-absolute-progress.md](shogi/kp-absolute-progress.md) — game-progress estimation using KP-Absolute features
- [shogi/shogi_progress_kpabs_train.md](shogi/shogi_progress_kpabs_train.md) — CLI spec of the `shogi_progress_kpabs_train` tool
