# 5. Stop and resume

<a href="../../ja/tutorial/5-resume.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

If training stops, rerun the same command with the same settings. BulletOu continues from the latest checkpoint.

## 5.1 Basic rule

Use the same `tag` or the same `output`, and rerun with the same `bulletou-settings.json`. If you used `output_folder` to put checkpoints on another drive, keep that value in the settings file too.

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

When BulletOu finds `checkpoints/.../000N/state.bin`, it loads it automatically.

To put checkpoints on the D: drive, specify only the parent folder with `output_folder`. `tag` still works.

```json
{
  "arch": "SFNN_halfka2_1024_7_64_k3k3",
  "teacher": "D:/sojoteam_datasets",
  "output_folder": "D:/checkpoints",
  "tag": "sfnn-test"
}
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

## 5.2 Cleaning old checkpoints

Checkpoints can become large. You may delete old save points as long as the latest checkpoint you want to resume from still has these files:

```text
checkpoints/.../
  resume-config.txt
  0074/
    state.bin
    learn.log
    dataloader_pos.txt
```

| File / folder | Needed for resume? | Meaning |
| --- | --- | --- |
| `0074/state.bin` | yes | Weights and optimizer state |
| `0074/dataloader_pos.txt` | yes | Where the teacher loader should continue |
| `0074/learn.log` | yes | Metadata used to treat the checkpoint as fully saved |
| `resume-config.txt` | yes | Training-control signature used by auto-resume |
| `0074/nn.bin` | no | Quantized network for the engine. Resume does not use it |
| `summary-learn.log` | no | Cumulative validation log. Useful to keep, but not required for resume |
| old `0001/` ... `0073/` | no | Safe to delete if you only need to resume from `0074` |

For example, if you only need to resume from `0074` and no longer need old `nn.bin` files, you can delete `0001` through `0073`.

```powershell
$exp = "C:\shogi\YaneuraOuWorks\BulletOu\checkpoints\experiment-folder-name"

Get-ChildItem $exp -Directory |
  Where-Object { $_.Name -match '^\d{4}$' -and [int]$_.Name -lt 74 } |
  Remove-Item -Recurse -Force -WhatIf
```

Run it with `-WhatIf` first to confirm what will be removed. If it looks right, remove `-WhatIf`.

```powershell
Get-ChildItem $exp -Directory |
  Where-Object { $_.Name -match '^\d{4}$' -and [int]$_.Name -lt 74 } |
  Remove-Item -Recurse -Force
```

If training is currently running, do not delete the checkpoint directory that BulletOu is writing. When unsure, keep the latest two checkpoints.

## 5.3 Changing settings

If you change settings such as `lr`, `batch_size`, or `superbatches`, automatic resume stops.

If you intentionally want to continue from the same checkpoint with changed settings, pass `--resume`.

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --resume
```

For a new experiment, use a new `tag`.

---

Next: [6. Inspect the result](6-result.md)

Detailed notes: [Advanced guide](../advanced/)

Previous: [4. Enable validation](4-validation.md)
