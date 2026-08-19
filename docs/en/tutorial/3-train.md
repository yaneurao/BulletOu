# 3. Run the training

<a href="../../ja/tutorial/3-train.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Goal: create an evaluation-function file that YaneuraOu can load.

This page continues from [2. Prepare training data](2-data.md).

## 3.1 Build once

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
```

On Windows, the executable is:

```text
.\target\release\examples\bulletou.exe
```

## 3.2 Minimal command

For a first real run, HalfKP NNUE is the easiest target.

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --tag first-halfkp
```

`--arch` selects the evaluation-function shape. `NNUE_halfkp_256x2_32_32` is a small, easy-to-test NNUE.

`--teacher` points to a teacher file or a directory of teacher files.

`--tag` names the experiment. It is optional, but useful once you run more than one experiment.

## 3.3 Output

Training writes checkpoints under `checkpoints/`. For NNUE / SFNN targets, each saved checkpoint contains an `nn.bin`.

Example:

```text
checkpoints/
  NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
    0001/
      nn.bin
      state.bin
```

`nn.bin` is the file you load into YaneuraOu.

## 3.4 Short smoke run

Before launching a long run on huge teacher data, you can force a small run:

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --positions-per-superbatch 1000000 `
  --superbatches 1 `
  --max-epochs 1 `
  --tag smoke-halfkp
```

Use this to check that loading, training, and saving all work.

## 3.5 What you should see

Training prints lines like:

```text
[train] epoch 1  sb 1/1  this-sb=... pos  wall=...s  train=...s  pos/s=...
```

`pos/s` is the training-speed indicator. Save and validation time are excluded from the training speed.

To watch accuracy / loss during training, configure a validation set on the next page.

## 3.6 Training SFNNs with many buckets

For SFNN architectures with many buckets, such as `hand1024_k3k3_progress4`, rarely seen buckets can learn unstable bucket-specific residuals. You can pre-count bucket occurrences into a `count.bin` file and pass it during training:

```powershell
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin `
--sfnn-residual-count-confidence 1.0
```

`--sfnn-residual-count-confidence 1.0` means: do not strongly trust a bucket-specific residual until that bucket has appeared about as many times as its own parameter count.

The count-based controls are off unless you specify them. You can also apply count-based confidence to factorizer terms:

```powershell
--sfnn-axis-count-confidence 1.0 `
--sfnn-pair-count-confidence 1.0
```

The same `count.bin` file is used for residual, axis, and pair confidence. For the count command and the exact formula, see [Advanced: SFNN factorizer](../advanced/sfnn-factorizer.md).

When you build `count.bin` from a very large teacher folder, `bucket-count` reads fixed-size `.psv` / `.bin` files in large chunks while counting. If read speed fluctuates on a drive such as `D:`, see the Advanced guide for `--buffer-mb` and `--read-buffers`.

---

Next: [4. Enable validation](4-validation.md)

For tuning and comparison experiments, see the [Advanced guide](../advanced/).

Previous: [2. Prepare training data](2-data.md)
