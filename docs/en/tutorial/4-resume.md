# 4. Stop and resume

<a href="../../ja/tutorial/4-resume.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

If training stops, rerun the same command with the same settings. BulletOu continues from the latest checkpoint.

## 4.1 Basic rule

Use the same `--tag` or the same `--output`, and rerun the command. If you used `--output-folder` to put checkpoints on another drive, pass the same folder again.

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --tag first-halfkp
```

When BulletOu finds `checkpoints/.../000N/state.bin`, it loads it automatically.

To put checkpoints on the D: drive, specify only the parent folder with `--output-folder`. `--tag` still works.

```powershell
.\target\release\examples\bulletou.exe `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --teacher D:\sojoteam_datasets `
  --output-folder D:\checkpoints `
  --tag sfnn-test
```

This writes to:

```text
D:\checkpoints\SFNN_HALFKA2-SFNN_halfka2_1024_7_64_k3k3-sfnn-test
```

```text
checkpoints/.../
  0001/
  0002/
  0003/   ← latest saved checkpoint
  0004/   ← resumed run writes from here
```

## 4.2 Changing settings

If you change settings such as `--lr`, `--batch-size`, or `--superbatches`, automatic resume stops.

If you intentionally want to continue from the same checkpoint with changed settings, pass `--resume`.

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --tag first-halfkp `
  --resume `
  --lr 0.0001
```

For a new experiment, use a new `--tag`.

---

Next: [5. Inspect the result](5-result.md)

Detailed notes: [Advanced guide](../advanced/)

Previous: [3. Run the training](3-train.md)
