# 0. Overview — What BulletOu Trains

<a href="../../ja/tutorial/0-overview.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

This page answers three questions before you write any code:

1. What is BulletOu for?
2. Which evaluation functions can it train?
3. How does the overall flow look?

## What is BulletOu for?

BulletOu is a **trainer for shogi evaluation functions**. Given a large set of training positions (each labelled with a score and a game result), it produces a binary file that a shogi engine — primarily [YaneuraOu](https://github.com/yaneurao/YaneuraOu) — can load and use as its evaluation function.

BulletOu does **not** play shogi itself. It is the part of the pipeline that **learns** an evaluation function from training data; the resulting file is then used by an engine at play time.

## Supported evaluation function families

BulletOu inherits its core from `bullet` (jw1912) and `bullet-shogi` (SH11235). It targets both **shogi NNUE-style value networks** and **YaneuraOu's legacy KPPT-family evaluation functions**.

| Family | Notes |
|---|---|
| **NNUE (HalfKP / HalfKA / Layer Stack)** | Inherited from bullet-shogi. Layer Stack with KP-Absolute progress buckets is the typical configuration for the strongest results. |
| **NNUE with Threat / HandThreat / HandCount features** | 7 input feature variants are available. |
| **KPPT** | `bulletou --eval-type kppt` trains KK / KKP / KPP in one run and produces the three-file set (elmo(WCSC27)-compatible). See [KPPT / KPP_KKPT Training](../shogi/kppt.md). |
| **KPP_KKPT (factorised variant)** | KK and KKP files are identical to KPPT; KPP is written without the turn channel (`--eval-type kpp-kkpt-kpp`). |
| KK-only / KKP-only minimal variants | Run the corresponding `bulletou --eval-type kppt-kk` / `kppt-kkp` standalone. |

## Where the data comes from

BulletOu reads training data in one of the following formats:

- **`.pack`** — produced by the `gensfen` script in [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection).
- **`.hcpe`** / **`.hcpe3`** — dlshogi-style formats.

Training data is not bundled with BulletOu. Generate your own or use a shared dataset.

## Where the output goes

When training finishes (and at every checkpoint along the way), BulletOu writes the binary file(s) appropriate to the evaluation function family being trained. For NNUE-style targets this is **`nn.bin`** (the parameter file loaded by a YaneuraOu engine at play time). For KPPT-style targets it is the three-file set `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin`.

How the engine consumes the file depends on the engine; see its documentation.

## Overall flow

```
[ Generate / obtain training data ]
        │
        │  YaneuraOu-ScriptCollection's gensfen script → *.pack
        ▼
[ BulletOu training ]            ← this tutorial walks you through this part
        │
        │  cargo run --release --example ... -- --data ... --output ...
        ▼
[ Output ]                       ← consumed by the engine at play time
        nn.bin (NNUE family)
        or KK_synthesized.bin / KKP_synthesized.bin / KPP_synthesized.bin (KPPT family)
```

The rest of the tutorial:

- [1. Quick Start](1-quickstart.md) — get the toolchain working and run a smoke-test training
- [2. NNUE Tutorial](2-nnue-tutorial.md) — a deeper walkthrough that trains a real NNUE
- [KPPT / KPP_KKPT Training](../shogi/kppt.md) — how to train legacy YaneuraOu evals (reference)
