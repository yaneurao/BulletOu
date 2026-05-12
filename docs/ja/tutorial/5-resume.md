# 5. 中断・再開

<a href="../../en/tutorial/5-resume.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習途中で `Ctrl+C` で止めたり、マシンの再起動などで中断しても、**同じ `--output` で同じコマンドをもう一度実行するだけで、自動的に最新 `000N/state.bin` から学習が続行される**。

```
checkpoints/.../
├── 0001/             ← 前回の最初の save
├── 0002/
├── 0003/             ← 中断時点で最新だった save
├── 0004/             ← 再開後ここから書かれる
└── 0005/
```

仕組み:
- `bulletou` 起動時、`--output` 配下に番号付き dir + `state.bin` があれば検出
- 最大番号の `state.bin` から重みと Adam moments を復元
- 新 save は既存最大番号の次から書く (前例で `0003/` まであれば `0004/` から)
- `learn.log` (累積版) には新 run の CSV 行がそのまま追記される。LR scheduler は run ごとに reset されるため superbatch カウンタは 1 から再開するが、`positions` 列は累積される (新 run 開始時に既存 `learn.log` の最大 positions を読み取って続きから書く)

この挙動は eval-type 横断 (KPPT / KPP_KKPT / NNUE_HALFKP / NNUE_KP / NNUE_HALFKPE9 すべて同じ仕組み)。新規学習にしたい場合は `--output` を別の dir にするか、既存 dir を削除する。

---

次へ:
- [6. 学習をチューニング](6-tune.md) — `--lambda`、`--lr`、`--superbatches` 等で学習を調整する (任意)
- 学習結果がもう手元にあるなら [7. 結果を確認](7-result.md) へ

前へ: [4. 学習を走らせる](4-train.md)
