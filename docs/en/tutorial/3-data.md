# 3. Prepare training data — choosing the architecture and pre-processing

<a href="../../ja/tutorial/3-data.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: decide what to train and prepare the teacher data you'll feed to `bulletou`.

This page assumes you have already completed [1. Quick Start](1-quickstart.md) — your toolchain works and a smoke-test training succeeded.

We use **NNUE HalfKP as the running example** in this tutorial, but the same command shape applies to the other targets (NNUE K-P / NNUE HalfKPE9 / KPPT / KPP_KKPT) by switching `--arch`.

## 3.1 Choosing what to train

`bulletou --arch <X>` selects which evaluation function to train. For KPPT-family evals, `<X>` is `KPPT` or `KPP_KKPT`; for NNUE / SFNN evals, `<X>` is the YaneuraOu architecture name without the `YANEURAOU_ENGINE_` prefix. Common choices:

| `--arch` value | What it trains | Output (per save) |
|---|---|---|
| **`NNUE_halfkp_256x2_32_32`** — start here | Classic HalfKP NNUE — YaneuraOu's longest-standing evaluation function family. See [NNUE HalfKP Training](../shogi/halfkp.md). | `nn.bin` |
| `NNUE_kp_256x2_32_32` | Same network as HalfKP, but the input keeps K and P as independent features. See [NNUE K-P Training](../shogi/kp.md). | `nn.bin` |
| `NNUE_ka2_256x2_32_32` | Same network as K-P, but with K+A2 input. See [NNUE K-A2 Training](../shogi/ka2.md). | `nn.bin` |
| `NNUE_halfkpe9_256x2_32_32` | HalfKP augmented with per-square attacker-count info (own/opp 0/1/2, 9 combos). See [NNUE HalfKPE9 Training](../shogi/halfkpe9.md). | `nn.bin` |
| `NNUE_halfkpvm_256x2_32_32` | HalfKP with king-position file-mirror folding (files 6-9 mirrored to 1-4). Input dim is ~half of HalfKP. | `nn.bin` |
| `SFNN_halfkahm2_1536_15_32_k3k3` | LayerStacks evaluation for YaneuraOu's `YANEURAOU_ENGINE_SFNN1536` build. Usage in [§9 LayerStack](9-layerstack.md); full spec in the [SFNN-1536 reference](../shogi/sfnn-1536.md). | `nn.bin` |
| `SFNN_halfkahm1_1536_15_32_k3k3` | v1 ablation of the above. | `nn.bin` |
| `SFNN_ka2_1536_15_32_k3k3` | Same SFNN-1536 topology, but with lightweight K+A2 input. | `nn.bin` |
| `KPPT` | Legacy three-file evaluation (elmo(WCSC27)-compatible). See [KPPT / KPP_KKPT Training](../shogi/kppt.md). | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` |
| `KPP_KKPT` | KPPT's factorised variant — only KPP changes (no turn channel, ~half size) | Same three files, only KPP layout differs |

Architecture size variants such as `NNUE_halfkp_1024x2_8_64` or `SFNN_ka2_8192_7_64_c0_s1024x8_k3k3` are accepted for experiments when the matching YaneuraOu architecture exists.

## 3.2 Get training data

You need a `.pack`, `.hcpe`, `.hcpe3`, `.psv`, or PSV-compatible `.bin` file.

- **Generate your own** — `.pack` is produced by the `gensfen` script in [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection); `.hcpe` / `.hcpe3` come from dlshogi-style generators. For this tutorial, 10–100 million positions is enough.
- **Use a shared dataset** — the shogi community shares files in all formats.

For this walkthrough we'll put teacher files under a `teachers/` directory next to the working directory:

```
teachers/
    teacher.pack
```

(`.hcpe` / `.hcpe3` / `.psv` / `.bin` work the same way. Format is inferred from the extension. You may also point `--teacher` at a directory, in which case all matching files inside are concatenated. `.bin` is treated as the same 40-byte `PackedSfenValue` format as `.psv`, so `.psv` and `.bin` may be mixed.)

### Shuffle teacher positions

> ⚠️ **Important**: shuffle teacher positions either before training, or during training with `--teacher-shuffle-buffer-batches`.

`--buffer-mb` controls the loader read buffer size; it is not a shuffle option. To shuffle during training, use `--teacher-shuffle-buffer-batches N`. BulletOu accumulates `batch_size × N` decoded positions on CPU, Fisher-Yates shuffles that window, and then emits mini-batches. `N` must divide the effective `batches_per_superbatch`.

`gensfen` and dlshogi-style generators usually emit positions **grouped by game** (positions from one game are contiguous). If you train on such files directly, nearby positions from the same game dominate consecutive mini-batches, and loss / plateau decisions become sensitive to local teacher bias.

How to shuffle:
- **`.hcpe` / `.psv` / `.bin`**: use `teacher/shuffle_split_teacher_external.py` from [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection). It bucket-distributes huge teacher folders and writes shuffled split files without loading everything into memory.
- **`.hcpe3` / `.pack`**: these are variable-length game formats, so record-level shuffling is not straightforward. Shuffle at generation time, or convert to a fixed-position format such as `.psv` / `.hcpe` and shuffle that.
- **In-trainer shuffle**: specify something like `--teacher-shuffle-buffer-batches 61`. For example, with `batch_size=65536` and `positions-per-superbatch=40000000`, the effective `batches_per_superbatch` is `610`, so `61` gives 10 shuffle windows per superbatch and keeps checkpoint/resume boundaries aligned.

Example: shuffle/split an HCPE or PSV folder into 10M-position files:

```bash
python /path/to/YaneuraOu-ScriptCollection/teacher/shuffle_split_teacher_external.py \
    src_teacher_folder \
    dst_teacher_folder \
    --positions 10000000
```

Outputs are named like `shuffled-00001.hcpe`, `shuffled-00002.hcpe`, ... . For more than 99,999 output files, increase the width with `--digits 6`.

### Trying with a small subset first

Before running on a huge dataset, you can try a smaller subset by generating a smaller file from `gensfen`, or by limiting `--positions-per-superbatch` so each superbatch consumes less data (see [§6.1 Training schedule](6-tune.md#61-training-schedule)).

---

Next: [4. Run the training](4-train.md) — train a real evaluation function on actual data.

Previous: [1. Quick Start](1-quickstart.md)
