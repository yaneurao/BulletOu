# 4. Run the training  Einvoking `bulletou`

<a href="../../ja/tutorial/4-train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本誁EDC2626?style=flat-square"></a>

Goal: produce a YaneuraOu-loadable evaluation function from the teacher data you prepared.

This page assumes you have already completed [3. Prepare training data](3-data.md)  Ethe teacher file (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) is ready and pre-shuffled.

## 4.1 Build (one-off)

Build `bulletou` first. You only need this once, until the source changes:

```bash
cargo build --release --features cuda-cpp-backend --example bulletou
```

On Windows the binary is at `.\target\release\examples\bulletou.exe`; the run commands below use Unix-style paths  Etranslate as needed.

## 4.2 Minimal command (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_256x2_32_32 \
    --teacher teachers/
```

That's it  Eno further target flag is needed. With `--output` omitted, checkpoints land under `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/` (auto-derived from `--arch`). Pass `--output checkpoints/my-halfkp` (or any other path) to override.

## 4.3 Specifying `--arch`

Pass the training target through `--arch`. For KPPT-family evals, use `KPPT` or `KPP_KKPT`. For NNUE / SFNN evals, use the YaneuraOu Makefile edition name after removing the `YANEURAOU_ENGINE_` prefix. For example, HalfKP 256x2-32-32 is `NNUE_halfkp_256x2_32_32`, K-P 256x2-32-32 is `NNUE_kp_256x2_32_32`, and SFNN looks like `SFNN_halfka2_1024_7_64_k3k3`. The old shorthand `256x2-32-32` is not accepted.

For NNUE names, the size part is `<L1>x2_<L2>_<L3>`. `L1` (the per-perspective accumulator size) must be a positive multiple of 32 (FT SIMD-padding requirement); `L2` and `L3` (hidden-layer sizes) must be positive integers. Common sizes:

| Size suffix | L1 (accumulator) | L2 | L3 | Notes |
|---|---|---|---|---|
| `256x2-32-32` (default) | 256 | 32 | 32 | Classic small NNUE; fast to train, good for sanity checks |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | Medium |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | Larger (inference cost grows) |
| `1024x2-8-64` | 1024 | 8 | 64 | Larger |
| `SFNN_halfkahm2_1536_15_32_k3k3` | 1536 | 15 | 32 | SFNN-1536 with k3k3(king3-by-king3) LayerStacks |
| `SFNN_halfka2_1024_7_64` | 1024 | 7 | 64 | SFNN single stack (`LayerStacks = 1`, no bucket suffix) |
| `SFNN_halfka2_4096_3_64_c0_s1024x4_k3k3` | 4096 | 3 | 64 | Grouped SFNN L1: 4096 is split into 4 groups, so each group maps 1024 -> 1 |
| `SFNN_halfka2_8192_3_64_c0_s2048x4_k3k3` | 8192 | 3 | 64 | Grouped SFNN L1: 8192 is split into 4 groups, so each group maps 2048 -> 1 |
| `SFNN_halfka2_4096_7_64_c0_s1024x4_k3k3` | 4096 | 7 | 64 | Grouped SFNN L1: 4096 is split into 4 groups |
| `SFNN_halfka2_1024_7_64_hand64` | 1024 | 7 | 64 | SFNN with YaneuraOu hand64 LayerStack buckets (64 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k3k3` | 1024 | 7 | 64 | hand64 ÁEk3k3 LayerStack buckets (576 stacks; much larger) |
| `SFNN_halfka2_1024_7_64_k9k9` | 1024 | 7 | 64 | king9-by-king9 LayerStack buckets (81 stacks) |
| `SFNN_halfka2_1024_7_64_k21k21` | 1024 | 7 | 64 | king21-by-king21 LayerStack buckets (441 stacks) |
| `SFNN_halfka2_1024_7_64_k29k29` | 1024 | 7 | 64 | king29-by-king29 LayerStack buckets (841 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k9k9` | 1024 | 7 | 64 | hand64 ÁEk9k9 LayerStack buckets (5184 stacks; very large) |
| `SFNN_halfka2_1024_7_64_hand64_k21k21` | 1024 | 7 | 64 | hand64 x k21k21 LayerStack buckets (28224 stacks; huge) |
| `SFNN_halfka2_1024_7_64_hand64_k29k29` | 1024 | 7 | 64 | hand64 ÁEk29k29 LayerStack buckets (53824 stacks; huge) |
| `SFNN_halfka2_1024_7_64_hand256` | 1024 | 7 | 64 | hand256 hand-presence LayerStack buckets (256 stacks) |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 1024 | 7 | 64 | hand256 ÁEk3k3 LayerStack buckets (2304 stacks; very large) |
| `SFNN_halfka2_1024_7_64_hand1024` | 1024 | 7 | 64 | hand1024 hand-presence LayerStack buckets (1024 stacks) |
| `SFNN_halfka2_1024_7_64_hand1024_k3k3` | 1024 | 7 | 64 | hand1024 ÁEk3k3 LayerStack buckets (9216 stacks; huge) |
| `SFNN_halfka2_1024_7_64_progress8` | 1024 | 7 | 64 | progress8 LayerStack buckets (8 stacks; progress axis only) |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 1024 | 7 | 64 | k3k3 x progress8 LayerStack buckets (72 stacks) |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 1024 | 7 | 64 | hand256 x k3k3 x progress16 LayerStack buckets (36864 stacks) |
| `SFNN_ka2_4096_15_64_c0_s256x16_k3k3` | 4096 | 15 | 64 | Same grouped SFNN form, but with lightweight KA2 input |
| `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3` | 8192 | 7 | 64 | Common+shard notation for pure grouped L1: 0 common + 1024 x 8 shards |
| `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3` | 3072 | 7 | 64 | Common+shard SFNN L1: 1024 common channels plus 8 shards of 256 channels |

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_1024x2_8_64 \
    --teacher teachers/
```

`--arch` is required for training because it is now the single source of truth for both the architecture and the internal target family. Sizes outside the table above are accepted for experimentation, but the resulting `nn.bin` is only loadable by a YaneuraOu build whose architecture header matches the same architecture  Egenerate it by passing the matching edition name to `make` (see [§8 Engine](8-engine.md)).

Grouped SFNN experiments can be written with `_c0_sMxG` before the optional LayerStack suffix. For example, `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3` means `FT=8192`, `L1 hidden=7`, `L2=64`, and L1 is split into 8 shards of 1024 channels. Non-zero common+shard L1 uses the same form; for example `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3` means 1024 common FT channels plus 8 shard blocks of 256 channels. If the suffix is omitted, the model uses a single stack (`LayerStacks = 1`). Otherwise the suffix can combine the independent `hand64/hand256/hand1024`, `k3k3/k9k9/k21k21/k29k29`, and `progress2/3/4/8/16/32` axes, e.g. `hand256_k3k3_progress16`. The parser accepts these tokens in any order and canonicalizes them to `hand`, `king`, `progress`. The `ka2` / `halfka2` feature token in the architecture name selects the internal target automatically.

## 4.4 Training SFNN-1536 (YaneuraOu NNUEwoSQPT1536)

If you need an evaluation function that loads in YaneuraOu's **`YANEURAOU_ENGINE_SFNN1536` build**, pass its architecture name:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

The difference from the rest of the NNUE family is that the network uses 9 sub-networks selected per position. The `k3k3` suffix in `--arch` selects the YaneuraOu-compatible LayerStack scheme; see [§9 LayerStack](9-layerstack.md). The full architecture, quantisation, and `nn.bin` layout spec lives in [the SFNN-1536 reference](../shogi/sfnn-1536.md).

## 4.5 Training a KPPT eval

For KPPT-family evals, use the fixed target name as `--arch`:

```bash
./target/release/examples/bulletou \
    --arch KPPT \
    --teacher teachers/
```

The default output dir is `checkpoints/KPPT/`. To get the factorised variant, swap to `--arch KPP_KKPT`.

## 4.6 Passing teacher data

`--teacher` accepts:
- a single file (e.g. `teachers/teacher.pack`),
- a directory (above; all matching files inside are concatenated; `.bin` is treated like `.psv`),
- or a comma-separated combination of the two.

## 4.7 How long does training run

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader reaches EOF). For multi-epoch runs, set the epoch length with `--superbatches` and then pass `--max-epochs N`. `step` / `geometric` / `cos` restart to `--lr` at epoch boundaries.

If you know the teacher size you can pin "1 epoch = N sb" explicitly with `--superbatches N` (see [§6.1 Training schedule](6-tune.md#61-training-schedule)). The `--count-teacher` flag tells you the total position count instantly:

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
# ↁE"Total: 461373440 positions, suggested --superbatches 4"
```

This matters especially for `--lr-schedule cos`: pick `--superbatches` so one cosine cycle equals one epoch, and `lr_min` lands at each epoch's last batch with a clean warm restart back to `lr_max` at the next epoch. In this mode, the teacher itself does not rewind at epoch boundaries. It is treated as a cyclic stream and rewinds only when it reaches EOF.

## 4.8 What you should see

While it runs:

```
=== bulletou: running NNUE_HALFKP (256x2-32-32 ClippedReLU, dual-perspective) ===
Training Preamble
Net Name               : shogi_nnue_halfkp
Batch Size             : 65536
Batches / Superbatch   : 1525
Positions / Superbatch : 99942400
...
  cuda-cpp loss progress log = checkpoints/.../cuda-cpp-progress.log (step 1, every 10 step(s), checkpoint, final)
  cuda-cpp SFNN checkpoint: epoch=1 sb=1/36 batch=2543/2543 positions=41664512 pos/s=... dir=checkpoints/.../0001
  cuda-cpp SFNN validation summary: epoch=1, superbatch=1, test_value_accuracy=..., test_value_loss=...
  cuda-cpp SFNN direct train = ok: steps=..., positions=..., train_elapsed=...s, elapsed=...s, throughput=... pos/s, ...
```

For cuda-cpp, stdout `pos/s` excludes checkpoint file saving, validation, loss readback, and progress-log writes. Per-batch loss is written to `<output>/cuda-cpp-progress.log` instead of being streamed to stdout.

`pos/s` (positions per second) is the rough training-speed indicator. On a single RTX 4090 expect tens of millions of pos/s; on slower GPUs proportionally less.

---

Next:
- To stop and resume training, see [5. Stop and resume](5-resume.md)
- To tune the training schedule or teacher target, see [6. Tune the training](6-tune.md)
- If you already have a trained model, jump to [7. Inspect the result](7-result.md)

Previous: [3. Prepare training data](3-data.md)
