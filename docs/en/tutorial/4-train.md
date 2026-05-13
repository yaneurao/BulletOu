# 4. Run the training — invoking `bulletou`

<a href="../../ja/tutorial/4-train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: produce a YaneuraOu-loadable evaluation function from the teacher data you prepared.

This page assumes you have already completed [3. Prepare training data](3-data.md) — the teacher file (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) is ready, ideally pre-shuffled.

## 4.1 Build (one-off)

Build `bulletou` first. You only need this once, until the source changes:

```bash
cargo build --release --features device-cuda --example bulletou
```

(Use `--features device-rocm` instead of `cuda` for AMD GPUs. On Windows the binary is at `.\target\release\examples\bulletou.exe`; the run commands below use Unix-style paths — translate as needed.)

## 4.2 Minimal command (NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/
```

That's it — no further flags needed. With `--output` omitted, checkpoints land under `checkpoints/NNUE_HALFKP-256x2-32-32/` (auto-derived from `--eval-type` and `--arch`). Pass `--output checkpoints/my-halfkp` (or any other path) to override.

## 4.3 Specifying `--arch`

For NNUE eval types, the layer sizes are selected with `--arch <L1>x2-<L2>-<L3>`. The set mirrors the per-arch directories under YaneuraOu's NNUE engine binary distribution (`NNUE_halfkp_*`):

| `--arch` | L1 (accumulator) | L2 | L3 | Notes |
|---|---|---|---|---|
| `256x2-32-32` (default) | 256 | 32 | 32 | Classic small NNUE; fast to train, good for sanity checks |
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | Medium |
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | Larger (inference cost grows) |
| `1024x2-8-64` | 1024 | 8 | 64 | Larger |

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --arch 1024x2-8-64 \
    --teacher teachers/
```

Omitting `--arch` falls back to `256x2-32-32`. `NNUE_KP` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` accept the same preset list.

## 4.4 Training SFNN-1536 (YaneuraOu NNUEwoSQPT1536)

If you need an evaluation function that loads in YaneuraOu's **`YANEURAOU_ENGINE_NNUE_SFNNwoP1536` build**, use its dedicated `--eval-type`:

```bash
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch 1536x2-15-32 \
    --layerstack king3-by-king3 \
    --teacher teachers/
```

The only thing that's different from the rest of the NNUE family is the `--layerstack` flag (the network uses 9 sub-networks selected per position — see [§9 LayerStack](9-layerstack.md) for an explanation). The full architecture, quantisation, and `nn.bin` layout spec lives in [the SFNN-1536 reference](../shogi/sfnn-1536.md).

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

Without `--superbatches` or `--max-epochs`, training runs through the teacher data once (until the dataloader reaches EOF). To run multiple passes, pass `--max-epochs N` — the LR scheduler restarts at the beginning of each epoch.

## 4.8 What you should see

While it runs:

```
=== bulletou: running NNUE_HALFKP (256x2-32-32 ClippedReLU, dual-perspective) ===
Training Preamble
Net Name               : shogi_nnue_halfkp
Batch Size             : 16384
Batches / Superbatch   : 6104
Positions / Superbatch : 100007936
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
