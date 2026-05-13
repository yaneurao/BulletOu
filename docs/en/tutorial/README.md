# BulletOu Tutorial

<a href="../../ja/tutorial/README.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

Step-by-step guide for users who are new to BulletOu. Read these pages in order.

| # | Page | What you'll do |
|---|---|---|
| 0 | [Overview](0-overview.md) | Understand what BulletOu trains and which evaluation function families are supported |
| 1 | [Quick Start](1-quickstart.md) | Install prerequisites, build BulletOu, and run a tiny training session to verify everything works |
| 2 | [Using `bulletou_lib` from your own code](2-bullet-lib.md) | Developer notes (registering custom examples, importing as a crate). Optional |
| 3 | [Prepare training data](3-data.md) | Choose the eval type and pre-process (shuffle) the teacher file |
| 4 | [Run the training](4-train.md) | Invoke `bulletou` (eval-type / arch / teacher) |
| 5 | [Stop and resume](5-resume.md) | Auto-resume by re-running with the same `--output` |
| 6 | [Tune the training](6-tune.md) | Adjust the schedule (`--lr`, `--superbatches`) and `--lambda` (optional) |
| 7 | [Inspect the result](7-result.md) | Output layout and reading `learn.log` |
| 8 | [Load into an engine](8-engine.md) | Verify the trained weights in YaneuraOu |
| 9 | [Training SFNN-1536 (NNUEwoSQPT1536)](9-sfnn-1536.md) | Training for YaneuraOu's SFNNwoP LayerStacks=9 build |

After finishing the tutorial, the [Reference docs](../) cover specifications and design details (data formats, network output formats, etc.).
