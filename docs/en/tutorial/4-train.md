# 4. Run the training — invoking `bulletou`

<a href="../../ja/tutorial/4-train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: produce a YaneuraOu-loadable evaluation function from the teacher data you prepared.

This page assumes you have already completed [3. Prepare training data](3-data.md) — the teacher file (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) is ready and pre-shuffled.

## 4.1 Build (one-off)

Build `bulletou` first. You only need this once, until the source changes:

```bash
cargo build --release --features cuda-cpp-backend --example bulletou
```

On Windows the binary is at `.\target\release\examples\bulletou.exe`; the run commands below use Unix-style paths — translate as needed.

## 4.2 Minimal command (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

That's it — no further flags needed. With `--output` omitted, checkpoints land under `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/` (auto-derived from `--eval-type` and `--arch`). Pass `--output checkpoints/my-halfkp` (or any other path) to override.

## 4.3 Specifying `--arch`

For NNUE / SFNN eval types, pass the YaneuraOu Makefile edition name after removing the `YANEURAOU_ENGINE_` prefix. For example, HalfKP 256x2-32-32 is `NNUE_halfkp_256x2_32_32`, K-P 256x2-32-32 is `NNUE_kp_256x2_32_32`, and SFNN looks like `SFNN_halfka2_1024_7_64_k3k3`. The old shorthand `256x2-32-32` is not accepted.

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

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --arch NNUE_halfkp_1024x2_8_64 \
    --teacher teachers/
```

Omitting `--arch` uses the per-eval-type default. For example, `NNUE_HALFKP` defaults to `NNUE_halfkp_256x2_32_32`, and `NNUE_KP` defaults to `NNUE_kp_256x2_32_32`. Sizes outside the table above are accepted for experimentation, but the resulting `nn.bin` is only loadable by a YaneuraOu build whose architecture header matches the same architecture — generate it by passing the matching edition name to `make` (see [§8 Engine](8-engine.md)).

## 4.4 Training SFNN-1536 (YaneuraOu NNUEwoSQPT1536)

If you need an evaluation function that loads in YaneuraOu's **`YANEURAOU_ENGINE_SFNN1536` build**, use its dedicated `--eval-type`:

```bash
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

The difference from the rest of the NNUE family is that the network uses 9 sub-networks selected per position. The `k3k3` suffix in `--arch` selects the YaneuraOu-compatible LayerStack scheme; see [§9 LayerStack](9-layerstack.md). The full architecture, quantisation, and `nn.bin` layout spec lives in [the SFNN-1536 reference](../shogi/sfnn-1536.md).

## 4.5 Training a KPPT eval

For KPPT-family eval types the architecture is fixed and `--arch` is ignored:

```bash
./target/release/examples/bulletou \
    --eval-type KPPT \
    --teacher teachers/
```

The default output dir is `checkpoints/KPPT/`. To get the factorised variant, swap to `--eval-type KPP_KKPT`.

## 4.6 Passing teacher data

`--teacher` accepts:
- a single file (e.g. `teachers/teacher.pack`),
- a directory (above; all same-extension files inside are concatenated),
- or a comma-separated combination of the two.

## 4.7 How long does training run

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader reaches EOF). For multi-epoch runs, set the epoch length with `--superbatches` and then pass `--max-epochs N`. `step` / `geometric` / `cos` restart to `--lr` at epoch boundaries.

If you know the teacher size you can pin "1 epoch = N sb" explicitly with `--superbatches N` (see [§6.1 Training schedule](6-tune.md#61-training-schedule)). The `--count-teacher` flag tells you the total position count instantly:

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
# → "Total: 461373440 positions, suggested --superbatches 4"
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
superbatch 1   pos = ... pos/s = ...   loss = ...
superbatch 2   ...
```

`pos/s` (positions per second) is the rough training-speed indicator. On a single RTX 4090 expect tens of millions of pos/s; on slower GPUs proportionally less.

---

Next:
- To stop and resume training, see [5. Stop and resume](5-resume.md)
- To tune the training schedule or teacher target, see [6. Tune the training](6-tune.md)
- If you already have a trained model, jump to [7. Inspect the result](7-result.md)

Previous: [3. Prepare training data](3-data.md)
