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

| Family | Notes |
|---|---|
Currently `bulletou` can train these four targets:

| `--eval-type` | What it is | Output |
|---|---|---|
| **`NNUE_HALFKP`** ★ start here | Classic HalfKP NNUE (Nasu-san's 2018 PR #75). See [NNUE HalfKP Training](../shogi/halfkp.md). | `nn.bin` |
| `NNUE_KP` | Same 4-layer ClippedReLU network as HalfKP but with the K and P features kept separate — lighter input. See [NNUE K-P Training](../shogi/kp.md). | `nn.bin` |
| `KPPT` | Legacy three-file evaluation (elmo(WCSC27)-compatible). See [KPPT / KPP_KKPT Training](../shogi/kppt.md). | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` |
| `KPP_KKPT` | KPPT's factorised variant — only the KPP file changes (no turn channel, ~half size) | Same three files, KPP in a different layout |

Coming later (the input-feature Rust code exists but isn't wired into `bulletou` yet): HalfKA / HalfKA_hm / Threat / HandThreat / HandThreatDefensive / HandCount / SFNN + ls9 (NNUEwoSQPT1536), etc.

## Where the data comes from

BulletOu reads training data in one of the following formats:

| Format | Producible by `gensfen` script | Producible by dlshogi self-play | Description |
|---|---|---|---|
| `.pack` | ☑ | □ | YaneuraOu's `gensfen` script output |
| `.psv` | ☑ | □ | YaneuraOu's traditional teacher format |
| `.hcpe` | ☑ | ☑ | Apery's teacher format |
| `.hcpe3` | ☑ | ☑ | An extension of `.hcpe` by dlshogi's author |

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
        │  cargo run --release --example bulletou -- --eval-type ... --teacher ... --output ...
        ▼
[ Output ]                       ← consumed by the engine at play time
        nn.bin (NNUE family)
        or KK_synthesized.bin / KKP_synthesized.bin / KPP_synthesized.bin (KPPT family)
```

The rest of the tutorial:

- [1. Quick Start](1-quickstart.md) — get the toolchain working and run a smoke-test training
- [2. Running a training](2-nnue-tutorial.md) — train a real evaluation function on actual data and load it into an engine
- [KPPT / KPP_KKPT Training](../shogi/kppt.md) — how to train legacy YaneuraOu evals (reference)
