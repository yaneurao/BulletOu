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
[valid] epoch 1  sb 1    test_value_accuracy=...  test_value_loss=...
```

`pos/s` is the training-speed indicator. Save and validation time are excluded from the training speed.

---

Next: [4. Stop and resume](4-resume.md)

For tuning and comparison experiments, see the [Advanced guide](../advanced/).

Previous: [2. Prepare training data](2-data.md)
