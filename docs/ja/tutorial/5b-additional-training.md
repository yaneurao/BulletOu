# 5.5 追加学習の仕方

<a href="../../en/tutorial/5b-additional-training.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[§5 中断・再開](5-resume.md) は「学習が途中で死んだ → 続き」のシナリオでした。本章は **正常に完走した学習に、さらに epoch / 異なる設定で学習を積み上げる** シナリオを扱います。

例:
- 3 epoch 学習 → 結果見て「もう 3 epoch 追加で回したい」
- 16384 batch_size で学習 → 4090 が VRAM 余ってるので 32768 に増やしたい
- 一度学習した nn.bin に **別の教師** で fine-tune したい
- 同じ重みから **LR を小さく** して微調整したい

## 5.5.1 基本: 同じ `--tag` で再実行

追加学習は **`--tag` と学習設定が同じなら自動 resume**, **`--tag` が違うと新規学習** というルール。

```powershell
# 1 回目: 3 epoch 学習
.\bulletou.exe --teacher c:\shogi\teacher\... `
    --arch NNUE_kp_256x2_32_32 `
    --tag round1 --max-epochs 3 --superbatches 6 `
    --lr-schedule step --lr-min 0.00001

# 2 回目: 追加で 3 epoch
.\bulletou.exe --teacher c:\shogi\teacher\... `
    --arch NNUE_kp_256x2_32_32 `
    --tag round1 --max-epochs 3 --superbatches 6 `
    --lr-schedule step --lr-min 0.00001
```

2 回目起動時:
- 出力 dir (`checkpoints/NNUE_KP-NNUE_kp_256x2_32_32-round1/`) を発見
- 最大番号の `0018/state.bin` (= 前回の最終 checkpoint) を load
- 続きの `0019/` から save 開始
- `summary-learn.log` も追記される (= 累積)

**累積 epoch 数** は前回 3 + 今回 3 = 6 epoch 相当の学習に。

⚠️ `--max-epochs` は **「この invocation で何 epoch」** であって「合計目標」ではありません。

`--superbatches` や LR 関係の設定を変えた場合、同じ `--tag` でも auto resume は停止する。設定変更込みで続きを学習したい場合は、意図を明示するために `--resume` を付ける。

## 5.5.2 変えていいフラグ / 変えてはいけないフラグ

### ✅ 変えても OK。ただし `--resume` を明示する

| フラグ | 注意点 |
|---|---|
| `--batch-size` | state.bin は batch_size 非依存。Ranger optimizer state も per-parameter なので互換 |
| `--positions-per-superbatch` | 1 superbatch の局面数が変わる。実効値は `batch_size` の倍数へ切り捨て |
| `--lr` | `step` では StepLR の開始値、`geometric` / `cos` では lr_max |
| `--lr-min` | LR の下限 |
| `--lr-schedule` (`step` / `geometric` / `cos` / `plateau`) | LR の動きが変わる。bullet-shogi 寄せの既定値は `step` |
| `--max-epochs` | この invocation の epoch 数 |
| `--superbatches` | `geometric` / `cos` では LR cycle 長、`step` では epoch 内の処理上限 |
| `--save-rate` | checkpoint 保存頻度だけが変わる。既存 checkpoint を引き継ぐなら `--resume` を明示する |
| `--validation-rate` | `--test-teacher` の accuracy/loss 計測頻度だけが変わる。既存 checkpoint を引き継ぐなら `--resume` を明示する |
| `--lambda` | 教師ターゲットの混合比 |
| `--test-teacher` | 検証セットの差し替え |
| `--sfnn-factorizer` | SFNN residual factorizer の有効項を変える。`shared` / `none` / `axis`、または `king=axis,hand=shared` のような混合指定が可能。`axis` は arch に存在する bucket axis をまとめて有効化する shorthand で、`hand1024_k3k3` なら `king=axis,hand=axis` 相当。既存 checkpoint を引き継ぐなら `--resume` を明示する。`state.bin` は利用可能なfactorizer tensorを保持し、validation と `nn.bin` export では現在の invocation で有効な項だけをfoldする |

これらはモデル構造とは無関係なので、重みと Ranger optimizer state 自体は継続できる。ただし学習制御が変わるため、BulletOu は勝手には auto resume しない。続けるなら `--resume` を付ける。

`--teacher` だけは例外で、同じ設定のまま教師だけ変える fine-tune は auto resume できる。教師変更検出が働き、dataloader は新ファイルの先頭から読む ([§5.5.4](#554-教師を変えて-fine-tune))。

### ❌ 変えてはいけない (モデル構造に関わる)

| フラグ | 理由 |
|---|---|
| `--arch` | target family や NN topology (feature set / FT / L1 / L2 dims) が変わる = state.bin の tensor shape が合わない |
| `--arch` の LayerStack suffix (SFNN 系) | LayerStack 数が変わると最終層の dim が変わる |
| `--sfnn-factorized` / `--no-sfnn-factorized` | 互換用alias。新しいコマンドでは `--sfnn-factorizer shared` / `--sfnn-factorizer none` を推奨 |
| `--tag` | これを変えると別 dir = 新規学習に分岐 (= 別実験を作る目的でのみ使う) |

これらを変えるなら **`--tag` を変えて別 run として起動** してください。`--resume` を付けても tensor shape が合わないので復元できない。

## 5.5.3 例: batch_size を 16384 → 32768 に増やす

```powershell
# 続きの 3 epoch を 32768 で
.\bulletou.exe --teacher c:\shogi\teacher\... `
    --arch NNUE_kp_256x2_32_32 `
    --tag round1 --max-epochs 3 --superbatches 6 `
    --batch-size 32768 `
    --resume `
    --lr-schedule step --lr-min 0.00001
```

### `positions-per-superbatch` と `batch-size`

`--positions-per-superbatch` は目標局面数で、実際の `sb_size` は `floor(positions_per_superbatch / batch_size) * batch_size` になる。

| batch_size | positions_per_superbatch | 実効 sb_size |
|---|---|---|
| 16384 | 100,000,000 | 99,991,552 |
| 32768 | 100,000,000 | 99,975,168 |
| 65536 | 100,000,000 | 99,942,400 |

`--batch-size` を変えると切り捨て後の実効 sb_size は少し変わる。完全に同じ LR cycle 長へ揃えたい場合は、`--positions-per-superbatch` も明示して調整する。

### Optimizer state の意味の差

batch_size を 2 倍にすると gradient の noise が ~√2 倍小さくなります。Adam の second moment は前 run の noise レベルで計算されているので、最初の 1-2 sb は微妙に過大/過小評価される可能性があります。実用上は気にならないレベルですが、loss が一瞬不自然になっても狼狽しないでください。

## 5.5.4 教師を変えて fine-tune

別の教師ファイル (= 別の corpus、別の生成方法、別の質) で続きを学習する典型例:

```powershell
# 大量の弱教師で 3 epoch 学習
.\bulletou.exe --teacher c:\shogi\teacher\bulk\ `
    --arch NNUE_kp_256x2_32_32 `
    --tag distill `
    --max-epochs 3 --superbatches 6 `
    --lr-schedule step --lr-min 0.00001

# 小規模・高品質教師で fine-tune (= LR を小さめに)
.\bulletou.exe --teacher c:\shogi\teacher\strong\ `
    --arch NNUE_kp_256x2_32_32 `
    --tag distill `
    --max-epochs 2 --superbatches 4 `
    --resume `
    --lr 0.0001 --lr-min 0.000001 `
    --lr-schedule step
```

教師変更時の挙動:
- bulletou が `summary-learn.log` の最終行を見て **`teacher` 列が変わっている** ことを検出
- 警告メッセージを出して dataloader を新教師の先頭から読み直し
- `dataloader_pos.txt` をリセット
- sb counter は連続表示するため `cb_ctx.sb_offset` を内部調整

`step` / `geometric` / `cos` は、いずれも epoch 境界で LR cycle が `--lr` から始まります。教師を変えて fine-tune する場合は、必要に応じて `--lr` / `--lr-min` を明示し、別実験として分けたいときは `--tag` を変えてください。

## 5.5.5 LR を小さくして微調整

完走後の **「もうほとんど収束してるけど、もう少しだけ動かしたい」** 時は LR を 1 桁下げて短く回します:

```powershell
# 初回: 3 epoch 通常学習
.\bulletou.exe --teacher ... --tag main `
    --max-epochs 3 --superbatches 6 `
    --lr 0.001 --lr-min 0.00001 `
    --lr-schedule step ...

# 仕上げ: 1 epoch だけ LR を 1 桁小さく
.\bulletou.exe --teacher ... --tag main `
    --max-epochs 1 --superbatches 6 `
    --resume `
    --lr 0.0001 --lr-min 0.000001 `
    --lr-schedule step ...
```

これは "learning rate annealing for fine-tuning" の典型パターン。最後の 1 epoch で大きな揺れを起こさず微調整できます。

## 5.5.6 `--max-epochs` 累積 vs 一気

3 epoch + 3 epoch を 2 invocation で回すのと、6 epoch を 1 invocation で回すのは **学習動作的にはほぼ等価** です:

| 観点 | 2 invocation | 1 invocation |
|---|---|---|
| 重み更新の総量 | 同じ | 同じ |
| LR cycle | `step` / `geometric` / `cos` は各 epoch で `--lr` から再開 | 同じ |
| CUDA JIT compile | 2 回起動 = 2 回発生 (= 初回だけ重い) | 1 回 |
| 中間 checkpoint | 同じ (= 各 sb で save) | 同じ |
| 中断耐性 | 高い (= 1 invocation 終わるごとに完了) | 1 起動が長くなる |

実用上は **2-3 epoch ごとに区切って回す** のが、CUDA cache の温まり以降は同等の速度で、中断/設定変更がしやすくおすすめです。

---

次へ: [6. 学習をチューニング](6-tune.md) — `--lr` / `--superbatches` / `--lambda` の意味

前へ: [5. 中断・再開](5-resume.md)
