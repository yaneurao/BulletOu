# BulletOu Tutorial

<a href="../../ja/tutorial/README.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

The shortest path for new BulletOu users.
Read these pages in order to run one training job and load the result in an engine.

Detailed tuning, comparison experiments, and deeper output-file checks are in the [Advanced guide](../advanced/).

| # | Page | What you'll do |
|---|---|---|
| 0 | [Overview](0-overview.md) | Understand what BulletOu trains and which evaluation function families are supported |
| 1 | [Quick Start](1-quickstart.md) | Install prerequisites, build BulletOu, and run a tiny training session to verify everything works |
| 2 | [Prepare training data](2-data.md) | Choose an architecture and point BulletOu at teacher data |
| 3 | [Run the training](3-train.md) | Run the minimal `bulletou` command |
| 4 | [Enable validation](4-validation.md) | Use `--test-teacher` and `--validation-rate` to watch accuracy / loss |
| 5 | [Stop and resume](5-resume.md) | Continue from an interrupted run |
| 6 | [Inspect the result](6-result.md) | Check output files and the training log |
| 7 | [Load into an engine](7-engine.md) | Verify the trained weights in YaneuraOu |

After this tutorial, go to the [Advanced guide](../advanced/) for practical experiments, or the [Reference docs](../) for specifications.
