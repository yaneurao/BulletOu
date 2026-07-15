# 5. 中断・再開

<a href="../../en/tutorial/5-resume.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習途中で `Ctrl+C` で止めたり、マシンの再起動などで中断しても、**同じ `--output` で同じ学習設定のコマンドをもう一度実行するだけで、自動的に最新 `000N/state.bin` から学習が続行される**。

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
- `<output>/resume-config.txt` と現在の学習設定を比較し、一致した場合だけ auto resume する
- 最大番号の `state.bin` から重みと Adam moments を復元
- 新 save は既存最大番号の次から書く (前例で `0003/` まであれば `0004/` から)
- `summary-learn.log` (累積版) には新 run の CSV 行がそのまま追記される。superbatch カウンタは 1 から再開するが、`positions` 列は累積される (新 run 開始時に既存 `summary-learn.log` の最大 positions を読み取って続きから書く)。`step` / `geometric` / `cos` の LR cycle は epoch 境界で `--lr` から再開する

`--superbatches`、`--lr-schedule`、`--lr`、`--lr-min`、`--batch-size` などを変えると、既存 checkpoint は同じ `--tag` でも自動復元されない。これは設定変更に気づかず古い実験を引き継ぐ事故を避けるため。

意図して古い checkpoint を引き継ぎたい場合だけ `--resume` を付ける。逆に、checkpoint がある出力先を誤って使っていないか確認したい場合は `--no-resume` を付けると、既存 checkpoint がある時点で停止する。

この挙動は eval-type 横断 (KPPT / KPP_KKPT / NNUE_HALFKP / NNUE_KP / NNUE_HALFKPE9 すべて同じ仕組み)。新規学習にしたい場合は `--tag` / `--output` を別の dir にするか、既存 dir を削除する。

---

次へ:
- [5.5 追加学習の仕方](5b-additional-training.md) — 完走後にもっと回したい / 設定を変えて続行したい
- [6. 学習をチューニング](6-tune.md) — `--lambda`、`--lr`、`--superbatches` 等で学習を調整する (任意)
- 学習結果がもう手元にあるなら [7. 結果を確認](7-result.md) へ

前へ: [4. 学習を走らせる](4-train.md)
