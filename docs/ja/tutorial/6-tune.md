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
| `--lr-min` | (cos only) cycle 末で到達する最小 lr。cycle 長は `--superbatches` / 教師サイズから自動算出 | 0.0 |
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

**`cosine_period` は自動算出**: 別途 `--lr-cosine-period` を指定する必要はない (= 削除されたフラグ)。決定ルールは:

| 状況 | period |
|---|---|
| `--superbatches N` 指定あり | `N × sb_size` (= 1 epoch ぶん。**最推奨**) |
| `--superbatches` 未指定 AND HCPE / PSV 教師 | 教師全体の局面数 (file size から自動計算) |
| `--superbatches` 未指定 AND HCPE3 / pack 教師 | エラー (= 可変長レコードなので教師サイズ不明、明示が必要) |

たとえば `--superbatches 4 --lr-schedule cos --lr-min 0.00001` (1 epoch = 4 sb ≒ 400M 局面) の場合の lr 推移:

| cycle 内位置 | t | lr |
|---|---|---|
| 0M (sb 1 頭) | 0.0 | 0.001 (= `--lr`、lr_max) |
| 100M (sb 2 頭) | 0.25 | 0.000856 |
| 200M (sb 3 頭) | 0.5 | 0.000505 (midpoint) |
| 300M (sb 4 頭) | 0.75 | 0.000155 |
| 400M (sb 4 末) | 1.0 | 0.00001 (= `--lr-min`、lr_min) |
| 次 epoch sb 1 頭 | 0.0 | **0.001** ← warm restart |

epoch 跨ぎで cycle がきれいに重なるよう `--superbatches` を選ぶのがコツ ([§6.1.x 教師を数えて --superbatches を決める](#教師を数えて---superbatches-を決める) 参照)。

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
    --max-epochs 10 --tag 5G-cos --superbatches 4 \
    --lr-schedule cos --lr-min 0.00001
```

出力先がそれぞれ `checkpoints/NNUE_KP-256x2-32-32-5G-step/` と `-5G-cos/` に分かれる。`summary-learn.log` の `test_value_accuracy` / `test_value_loss` 列を pandas / Excel で重ねれば比較完了。

### 教師を数えて `--superbatches` を決める

`--lr-schedule cos` で cycle を 1 epoch ぴったりに揃えるには、まず **教師の総局面数** を知る必要がある。BulletOu には専用フラグ `--count-teacher` があり、`std::fs::metadata` でファイルサイズを読むだけなので **数百 GB の教師でも一瞬で完了** する (中身は read しない):

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
```

出力例:
```
Counting Hcpe teacher files (38 byte/record)...
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0001.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0002.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0003.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0004.hcpe
        92274688 positions  ( 3340.66 MB)  teachers/yane-distill-0005.hcpe
---
Total: 461373440 positions  (16.71 GB)  across 5 file(s)
Per-default-sb (= 100M positions): 4 full sb + 0.61 partial sb
Suggested `--superbatches`: 4 (= use 4 full sb per epoch; ~61M positions leftover ...)
```

この場合、`--superbatches 4` を指定すれば:
- 1 epoch = 4 sb = 400M 局面
- cos period = 400M (= 1 epoch ぴったり)
- sb 4 末尾で lr_min 着地、次 epoch の sb 1 頭で warm restart して lr_max に戻る

教師末尾の余り 61M は使われない (= 各 epoch 同じ先頭 400M を re-shuffle)。多少の無駄は許容して cycle を綺麗に揃える方が学習結果は読みやすい。

#### 対応フォーマット

| フォーマット | レコードサイズ | `--count-teacher` |
|---|---|---|
| HCPE | 38 byte 固定 | ✅ 即計算 |
| PSV  | 40 byte 固定 | ✅ 即計算 |
| HCPE3 | 可変長 (棋譜単位) | ❌ 未対応 (全 game header を walk する必要あり) |
| pack | 可変長 (棋譜単位) | ❌ 同上 |

HCPE3 / pack を使うなら、同じ corpus を HCPE / PSV に事前変換するか、`--superbatches` を手動指定してください。

### 複数 epoch 回す

`--max-epochs N` を指定すると教師データを N 周する。各 epoch 開始時に:
- LR scheduler が reset される (superbatch 1 から再開、`lr = --lr` に戻る — `step` でも `cos` でも同じ)
- データローダーが先頭にシークし直す

つまり N 回学習し直すに近い挙動。各 epoch ごとに lr が再下降するので、長時間学習で局所最適から脱出させたいときに使う。`cos` schedule で `--superbatches N` を指定すれば cycle = epoch で自動的に揃う (= 典型的な SGDR-style 用法)。

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
