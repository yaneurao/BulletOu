# 6. 学習をチューニング — スケジュールと教師ターゲット

<a href="../../en/tutorial/6-tune.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[4. 学習を走らせる](4-train.md) のデフォルト設定で動くことを確認したら、この章で **学習を調整するためのフラグ群** を紹介する。**最初の学習ではすべてデフォルトのままで問題ない**。チューニングが必要になったときに戻ってくる位置付け。

## 6.1 学習スケジュール

ログに出てくる `superbatch` は **checkpoint や学習率を更新するためのまとまり**で、デフォルトで約 1 億局面ぶん。

主要なフラグ:

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--batch-size` | 1 gradient step あたりの局面数 | 16384 |
| `--batches-per-superbatch` | 1 superbatch を構成する mini-batch 数 | `ceil(100M / batch-size)` (≒ 1 superbatch ≒ 1 億局面) |
| `--superbatches` | epoch あたりの superbatch 数の上限 | 上限なし (= EOF まで) |
| `--max-epochs` | 教師データを何周するか | 1 |
| `--save-rate` | N superbatch ごとに checkpoint を保存 | 1 |
| `--lr` | 初期学習率 (lr_max) | 0.001 |
| `--lr-schedule` | `step` (= 指数減衰) または `cos` (= cosine annealing + warm restart) | `step` |
| `--lr-gamma` / `--lr-step-positions` | (step only) `lr-step-positions` 局面ごとに `lr-gamma` 倍 | 0.9 / 100000000 |
| `--lr-cosine-period` / `--lr-min` | (cos only) `lr-cosine-period` 局面を 1 cycle として `--lr` → `--lr-min` を滑らかに往復 | 500000000 / 0.0 |
| `--lambda` | 教師 eval と対局結果 (WDL) のブレンド比 ([§6.2](#62-教師ターゲット-lambda) 参照) | 1.0 (= 純 eval) |

実行例 (1 億局面 × 40 superbatch = 計 40 億局面):

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --superbatches 40
```

教師ファイルが 1 superbatch 未満 (≒ 1 億局面未満) しか無い場合は `--batches-per-superbatch` を小さくする (例: `1024` で 1 superbatch ≒ 1670 万局面) と、何回も save が走るようになる。

### 学習率の動き — `--lr-schedule step` (デフォルト)

`--lr 0.001 --lr-gamma 0.9 --lr-step-positions 100000000` (デフォルト) の場合、**累積学習局面数** が 100M を超えるごとに lr を 0.9 倍する:

| 累積局面 | lr |
|---|---|
| 0 〜 100M | 0.001 |
| 100M 〜 200M | 0.000900 |
| 200M 〜 300M | 0.000810 |
| 500M | 0.000591 |
| 1G | 0.000349 |
| 2.2G | 0.0001 (≒ 初期値の 1/10) |

`--lr-gamma 0.1` のような攻撃的な値を指定すると 100M ごとに 10× drop。長く回すなら `0.9` 系の緩い設定が普通。

学習が走った後で実際の lr 推移を確認するには、`learn.log` の `lr` 列を見れば良い ([§7.2 学習ログの読み方](7-result.md#72-学習ログ-learnlog-の読み方))。

### 学習率の動き — `--lr-schedule cos` (cosine annealing)

`--lr-schedule cos` を指定すると、stepwise の代わりに **cosine annealing + warm restart** (SGDR) スケジュールになる:

```
t  = (累積局面 mod cosine_period) / cosine_period
lr = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π · t))
```

`--lr-cosine-period 500000000 --lr-min 0.00001` の場合の挙動:

| cycle 内位置 | t | lr |
|---|---|---|
| 0M (cycle 開始) | 0.0 | 0.001 (= `--lr`、lr_max) |
| 125M | 0.25 | 0.000856 |
| 250M | 0.5 | 0.000505 (midpoint) |
| 375M | 0.75 | 0.000155 |
| 500M (cycle 末) | 1.0 | 0.00001 (= `--lr-min`、lr_min) |
| 500M + 1 | 0.0 (次 cycle) | **0.001** ← warm restart |

500M 局面 = 1 epoch ぶんに `--lr-cosine-period` を合わせれば、各 epoch がきれいに 1 cycle = lr_max → lr_min を 1 回スイープしてリセット、を繰り返す。

#### `step` vs `cos` を比較したい

同じ教師・同じ arch で 2 回 run して `learn.log` を並べると、どちらが効くかすぐ分かる。`--tag` で出力先を分けるのがコツ:

```bash
# stepwise
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --tag 5G-step \
    --lr-schedule step --lr-step-positions 100000000 --lr-gamma 0.9

# cosine (epoch ごとに 1 cycle)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --tag 5G-cos \
    --lr-schedule cos --lr-cosine-period 500000000 --lr-min 0.00001
```

出力先がそれぞれ `checkpoints/NNUE_KP-256x2-32-32-5G-step/` と `-5G-cos/` に分かれる。`learn.log` の `test_value_accuracy` / `test_value_loss` 列を pandas / Excel で重ねれば比較完了。

### 複数 epoch 回す

`--max-epochs N` を指定すると教師データを N 周する。各 epoch 開始時に:
- LR scheduler が reset される (superbatch 1 から再開、`lr = --lr` に戻る — `step` でも `cos` でも同じ)
- データローダーが先頭にシークし直す

つまり N 回学習し直すに近い挙動。各 epoch ごとに lr が再下降するので、長時間学習で局所最適から脱出させたいときに使う。`cos` schedule で `--lr-cosine-period = epoch_size` を指定するのが典型的な SGDR-style 用法。

## 6.2 教師ターゲット (`--lambda`)

教師局面ファイル (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) には、各局面ごとに **2 種類のラベル** が記録されている:

1. **教師 eval** — その局面に対する教師エンジンの評価値 (sigmoid 後)
2. **対局結果** — その対局が最終的にどう終わったか (W/D/L = 1.0 / 0.5 / 0.0、side-to-move 視点)

`--lambda <λ>` で、loss target をこの 2 つでどう混ぜるかを指定する (やねうら王内蔵学習器の `lambda` と同じ慣例):

```
target = λ × 教師eval + (1 − λ) × 対局結果
```

| `--lambda` 値 | 意味 |
|---|---|
| `1.0` (デフォルト) | 100% 教師 eval、対局結果は無視 |
| `0.5` | eval 50% + 対局結果 50% (elmo 式の典型値) |
| `0.0` | 100% 対局結果、教師 eval は無視 |
| `0.7` 等 | 中間値も自由に指定可能 |

デフォルトの `1.0` (純 eval) が安全な初期値。教師エンジンの評価値そのものを真似に行く動作になる。

対局結果 (W/D/L = Win / Draw / Loss) も混ぜたいときに `--lambda` を下げる。完全結果ベース (`--lambda 0.0`) は教師エンジンの強さに依存しないが、勾配が疎で収束が遅い傾向。実用は `0.5 〜 0.8` あたりの混合が多い。

```bash
# elmo 式の 50:50 ブレンドで KPPT 学習
./target/release/examples/bulletou \
    --eval-type KPPT \
    --teacher teachers/ \
    --lambda 0.5
```

(`WDL` は Win/Draw/Loss の略。)

---

次へ: [7. 結果を確認](7-result.md) — 学習結果の確認、ログの読み方

前へ: [4. 学習を走らせる](4-train.md)
