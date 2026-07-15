# 5. Stop and resume

<a href="../../ja/tutorial/5-resume.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Stop mid-training (Ctrl+C, machine reboot, whatever) and **re-run the exact same command with the same `--output` — `bulletou` automatically resumes from the latest `000N/state.bin`**.

```
checkpoints/.../
├── 0001/             ← from the previous run
├── 0002/
├── 0003/             ← latest save when training was interrupted
├── 0004/             ← the resumed run writes from here
└── 0005/
```

How it works:
- On startup, `bulletou` looks under `--output` for numbered dirs containing `state.bin`.
- The highest-numbered `state.bin` is loaded, restoring weights and Adam moments.
- New saves continue numbering from one past the existing maximum (`0004/` here).
- The cumulative `summary-learn.log` keeps appending CSV rows for the resumed run. The superbatch counter resets to 1, but the `positions` column continues from the previous run's max (read off the existing `summary-learn.log` at startup). LR behaviour is schedule-dependent: `step_gamma` continues from cumulative positions, while `step` / `cos` start a new cycle for the new run.

This behaviour is identical for every eval-type (KPPT / KPP_KKPT / NNUE_HALFKP / NNUE_KP / NNUE_HALFKPE9 — all share the same mechanism). To start fresh, point `--output` at a different directory or delete the existing one.

---

Next:
- [5.5 Continued training](5b-additional-training.md) — add more epochs to a finished run, or change settings mid-stream
- [6. Tune the training](6-tune.md) — adjust `--lambda`, `--lr`, `--superbatches`, etc. (optional)
- If you already have a trained model, jump to [7. Inspect the result](7-result.md)

Previous: [4. Run the training](4-train.md)
