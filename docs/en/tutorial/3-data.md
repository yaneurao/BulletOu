# 3. Prepare training data — choosing the eval type and pre-processing

<a href="../../ja/tutorial/3-data.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: decide what to train and prepare the teacher data you'll feed to `bulletou`.

This page assumes you have already completed [1. Quick Start](1-quickstart.md) — your toolchain works and a smoke-test training succeeded.

We use **NNUE HalfKP as the running example** in this tutorial, but the same command shape applies to the other targets (NNUE K-P / NNUE HalfKPE9 / KPPT / KPP_KKPT) by switching `--eval-type`.

## 3.1 Choosing what to train

`bulletou --eval-type <X>` selects which evaluation function to train. The currently public choices:

| `--eval-type` | What it trains | Output (per save) | `--arch` used? |
|---|---|---|---|
| **`NNUE_HALFKP`** ★ start here | Classic HalfKP NNUE — YaneuraOu's longest-standing evaluation function family. See [NNUE HalfKP Training](../shogi/halfkp.md). | `nn.bin` | yes |
| `NNUE_KP` | Same network as HalfKP, but the input keeps K and P as independent features. See [NNUE K-P Training](../shogi/kp.md). | `nn.bin` | yes |
| `NNUE_HALFKPE9` | HalfKP augmented with per-square attacker-count info (own/opp 0/1/2, 9 combos). See [NNUE HalfKPE9 Training](../shogi/halfkpe9.md). | `nn.bin` | yes |
| `KPPT` | Legacy three-file evaluation (elmo(WCSC27)-compatible). See [KPPT / KPP_KKPT Training](../shogi/kppt.md). | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` | no |
| `KPP_KKPT` | KPPT's factorised variant — only KPP changes (no turn channel, ~half size) | Same three files, only KPP layout differs | no |

Coming later: HalfKA, SFNN + ls9 (NNUEwoSQPT1536), and other variants.

## 3.2 Get training data

You need a `.pack`, `.hcpe`, `.hcpe3`, or `.psv` file.

- **Generate your own** — `.pack` is produced by the `gensfen` script in [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection); `.hcpe` / `.hcpe3` come from dlshogi-style generators. For this tutorial, 10–100 million positions is enough.
- **Use a shared dataset** — the shogi community shares files in all formats.

For this walkthrough we'll put teacher files under a `teachers/` directory next to the working directory:

```
teachers/
    teacher.pack
```

(`.hcpe` / `.hcpe3` / `.psv` work the same way. Format is inferred from the extension. You may also point `--teacher` at a directory, in which case all files of the same extension inside are concatenated.)

### Pre-shuffle the teacher file

> ⚠️ **Important**: shuffle the teacher file **before** handing it to BulletOu.

Bullet's loader uses an **in-memory shuffle buffer (default 256 MB ≒ 6.7M positions for HCPE)** and Fisher-Yates shuffles its contents before slicing into batches. The shuffle is **intra-buffer only** — successive buffers contain disjoint sequential regions of the file, so the loader is really just doing local shuffles over 6.7M-position windows.

`gensfen` and dlshogi-style generators emit positions **grouped by game** (positions from one game are contiguous), so training on an un-shuffled file produces **periodic loss spikes at every buffer boundary** (≈ every 410 batches with `--batch-size 16384`) as the distribution shifts.

How to shuffle:
- **`.hcpe` / `.hcpe3`**: easiest is dlshogi's shuffle script. HCPE records are fixed-length (38 bytes), so a byte-level random permutation is sufficient. Look in dlshogi's `utils/` directory.
- **`.pack`**: enable the shuffle option in `gensfen` at generation time, or convert to PSV and shuffle that.
- **Workaround if you only have an un-shuffled file**: raise `--buffer-mb` so the whole file fits in a single buffer. Example: a 1.94 GB `.hcpe` (≈ 51M positions) fits with `--buffer-mb 2048` and stops crossing buffer boundaries. This costs **host RAM** (not GPU memory), so it works as long as you have the headroom.

### Trying with a small subset first

Before running on a huge dataset, you can try a smaller subset by generating a smaller file from `gensfen`, or by limiting `--batches-per-superbatch` so each superbatch consumes less data (see [§6.1 Training schedule](6-tune.md#61-training-schedule)).

---

Next: [4. Run the training](4-train.md) — train a real evaluation function on actual data.

Previous: [1. Quick Start](1-quickstart.md)
