# 5.5 追加学習の仕方

<a href="../../en/tutorial/5b-additional-training.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[§5 中断・再開](5-resume.md) は「学習が途中で止まったので続きを回す」話でした。本章は **完走した学習に、さらに学習を足す** 場合を扱います。

例:
- 3 epoch 学習 → 結果見て「もう 3 epoch 追加で回したい」
- `--batch-size 16384` で学習 → RTX 4090 の VRAM が余っているので `32768` に増やしたい
- 一度学習した重みに **別の教師** で追加学習したい
- 同じ重みから **学習率を小さく** して微調整したい

## 5.5.1 基本: 同じ `--tag` で再実行

追加学習の基本ルールは単純です。

- `--tag` と学習設定が同じなら、自動で続きから始まる
- `--tag` を変えると、別の学習として最初から始まる

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
- 出力ディレクトリ (`checkpoints/NNUE_KP-NNUE_kp_256x2_32_32-round1/`) を見つける
- 最大番号の `0018/state.bin`、つまり前回最後の保存状態を読む
- 続きの `0019/` から保存する
- `summary-learn.log` にも追記する

**累積 epoch 数** は前回 3 + 今回 3 = 6 epoch 相当の学習に。

⚠️ `--max-epochs` は **「今回の起動で何 epoch 回すか」** です。「合計で何 epoch まで」という意味ではありません。

`--superbatches` や学習率関係の設定を変えた場合、同じ `--tag` でも自動再開は止まります。設定変更込みで続きを学習したい場合は、意図を明示するために `--resume` を付けます。

## 5.5.2 変えていいフラグ / 変えてはいけないフラグ

### ✅ 変えても OK。ただし `--resume` を明示する

| フラグ | 注意点 |
|---|---|
| `--batch-size` | 重みの形は変わらないので継続できる |
| `--positions-per-superbatch` | 1 superbatch の局面数が変わる。実効値は `batch_size` の倍数へ切り捨て |
| `--lr` | epoch 先頭の学習率が変わる |
| `--lr-min` | 学習率の下限が変わる |
| `--lr-schedule` (`step` / `geometric` / `cos` / `plateau`) | 学習率の下げ方が変わる |
| `--max-epochs` | 今回の起動で回す epoch 数 |
| `--superbatches` | 1 epoch の長さが変わる |
| `--save-rate` | 保存頻度だけが変わる。保存済みデータを引き継ぐなら `--resume` を明示する |
| `--validation-rate` | `--test-teacher` の accuracy/loss 計測頻度だけが変わる。保存済みデータを引き継ぐなら `--resume` を明示する |
| `--lambda` | 教師評価値と勝敗結果の混ぜ方が変わる |
| `--test-teacher` | 検証用ファイルを差し替える |
| `--sfnn-factorizer` | SFNN の bucket 間で共通成分をどう持つかを変える。保存済みデータを引き継ぐなら `--resume` を明示する |

これらは重みの形を変えないので、保存済みの重みから続けられます。ただし学習の条件は変わるため、BulletOu は勝手には自動再開しません。続けるなら `--resume` を付けます。

`--teacher` だけは例外で、同じ設定のまま教師だけ変える追加学習は自動再開できます。教師が変わったことを検出し、新しい教師ファイルの先頭から読みます ([§5.5.4](#554-教師を変えて追加学習する))。

### ❌ 変えてはいけない (モデル構造に関わる)

| フラグ | 理由 |
|---|---|
| `--arch` | 評価関数の種類や層サイズが変わるため、保存済み重みと合わない |
| `--arch` の `k3k3` / `hand1024` など | SFNN の分岐数が変わるため、保存済み重みと合わない |
| `--sfnn-factorized` / `--no-sfnn-factorized` | `--sfnn-factorizer shared` / `--sfnn-factorizer none` の短縮指定。基本形を使う |
| `--tag` | これを変えると別ディレクトリになり、新規学習として扱われる |

これらを変えるなら **`--tag` を変えて別の学習として起動** してください。`--resume` を付けても保存済み重みの形が合わないので復元できません。

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

### optimizer 状態について

`--batch-size` を変えても重みは引き継げます。ただし optimizer は、ここまでの勾配の大きさや揺れ方を内部に持っています。batch size を変えた直後の 1〜2 sb だけ loss が少し不自然に見えることがあります。長く続くのでなければ、まずは様子を見てください。

## 5.5.4 教師を変えて追加学習する

別の教師ファイルで続きを学習する典型例:

```powershell
# 大量の弱教師で 3 epoch 学習
.\bulletou.exe --teacher c:\shogi\teacher\bulk\ `
    --arch NNUE_kp_256x2_32_32 `
    --tag distill `
    --max-epochs 3 --superbatches 6 `
    --lr-schedule step --lr-min 0.00001

# 小規模・高品質教師で追加学習 (= 学習率を小さめに)
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
- 警告メッセージを出して、新しい教師の先頭から読み直す
- `dataloader_pos.txt` をリセット
- 表示上の sb 番号は続きに見えるように調整される

`step` / `geometric` / `cos` は、いずれも epoch 境界で学習率が `--lr` から始まります。教師を変えて追加学習する場合は、必要に応じて `--lr` / `--lr-min` を明示し、別実験として分けたいときは `--tag` を変えてください。

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

これは最後に学習率を小さくして微調整する典型パターンです。最後の 1 epoch で大きな揺れを起こさずに少しだけ重みを動かせます。

## 5.5.6 `--max-epochs` 累積 vs 一気

3 epoch + 3 epoch を 2 回に分けて回すのと、6 epoch を 1 回で回すのは **学習動作としてはほぼ同じ** です:

| 観点 | 2 回に分ける | 1 回で回す |
|---|---|---|
| 重み更新の総量 | 同じ | 同じ |
| 学習率 | `step` / `geometric` / `cos` は各 epoch で `--lr` から再開 | 同じ |
| CUDA の初期化 | 2 回起動するぶん発生 | 1 回 |
| 中間保存 | 同じ (= 各 sb で保存) | 同じ |
| 中断耐性 | 高い (= 1 回終わるごとに保存済み) | 1 起動が長くなる |

実用上は **2-3 epoch ごとに区切って回す** のが、CUDA cache の温まり以降は同等の速度で、中断/設定変更がしやすくおすすめです。

---

次へ: [6. 学習設定を調整する](6-tune.md) — `--lr` / `--superbatches` / `--lambda` の意味

前へ: [5. 中断・再開](5-resume.md)
