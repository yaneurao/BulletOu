# 5. Stop and resume

<a href="../../ja/tutorial/5-resume.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Stop mid-training (Ctrl+C, machine reboot, whatever) and **re-run the exact same command with the same `--output` — `bulletou` automatically resumes from the latest `000N/state.bin`**.

```
checkpoints/.../
├── 0001/             ← first saved checkpoint
├── 0002/
├── 0003/             ← latest save when training was interrupted
├── 0004/             ← the resumed run writes from here
└── 0005/
```

How it works:
- On startup, `bulletou` looks under `--output` for numbered dirs containing `state.bin`.
- The highest-numbered `state.bin` is loaded, restoring weights and Ranger optimizer state.
- New saves continue numbering from one past the existing maximum (`0004/` here).
- `summary-learn.log` keeps appending CSV rows after resume. The superbatch counter resets to 1, but the `positions` column keeps increasing from the saved maximum. For `step` / `geometric` / `cos`, LR restarts from `--lr` at epoch boundaries.

This works the same way for KPPT / KPP_KKPT / NNUE / SFNN. To start fresh, point `--output` at a different directory or delete the output directory you no longer need.

---

Next:
- [5.5 Continued training](5b-additional-training.md) — add more epochs to a finished run, or change settings mid-stream
- [6. Adjust training settings](6-tune.md) — adjust `--lambda`, `--lr`, `--superbatches`, etc. (optional)
- If you already have a trained model, jump to [7. Inspect the result](7-result.md)

Previous: [4. Run the training](4-train.md)
