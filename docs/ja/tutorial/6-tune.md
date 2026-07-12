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
| `--superbatches` | epoch あたりの superbatch 数の上限。`plateau` では通常不要 | 上限なし (= 非 plateau は EOF まで、plateau は `lr_min` 到達まで) |
| `--max-epochs` | epoch を最大何回実行するか。`step` / `cos` では基本的に教師を何周するか、`plateau` では `lr_min` 到達までの試行を何回行うか | 1 |
| `--save-rate` | N superbatch ごとに checkpoint を保存 | 1 |
| `--lr` | 初期学習率 (lr_max。1 cycle の頭の値) | 0.001 |
| `--lr-schedule` | `step` (= geometric/対数線形)、`cos` (= cosine annealing)、`plateau` (= validation loss が改善しないときだけ LR を下げる) | `step` |
| `--lr-min` | 最小 lr。`step` / `cos` では cycle 末で到達する値、`plateau` では最終 lr | 0.00001 |
| `--lr-plateau-factor` | `plateau` で loss が改善しなかったときに LR へ掛ける係数 | 0.5 |
| `--lr-plateau-min-delta` | `plateau` で改善とみなす最小 loss 差 | 0.0 |
| `--lambda` | 教師 eval と対局結果 (WDL) のブレンド比 ([§6.2](#62-教師ターゲット-lambda) 参照) | 1.0 (= 純 eval) |

実行例 (1 億局面 × 40 superbatch = 計 40 億局面):

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --superbatches 40
```

教師ファイルが 1 superbatch 未満 (≒ 1 億局面未満) しか無い場合は `--batches-per-superbatch` を小さくする (例: `1024` で 1 superbatch ≒ 1670 万局面) と、何回も save が走るようになる。

### 学習率の動き

`step` と `cos` の **両方** が、1 epoch をかけて `--lr` (lr_max) から `--lr-min` (lr_min) へ滑らかに減衰、epoch 境界で warm restart して lr_max に戻る形になります。違いは曲線の形だけ:

| schedule | 式 | 形 |
|---|---|---|
| `step` (default) | `lr(t) = lr_max × (lr_min/lr_max)^t` (geometric) | 対数線形 — batch ごとに一定倍率で下がる |
| `cos` | `lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(πt))` | 先頭と末尾は緩やか、中盤で最も急 |

`t = (累積局面 mod period) / period`、`period = 1 epoch ぶんの局面数` (= 自動算出)。

**`step` / `cos` の period 決定ルール**:

| 状況 | period |
|---|---|
| `--superbatches N` 指定あり | `N × sb_size` (= 1 epoch ぶん。`step` / `cos` では推奨) |
| `--superbatches` 未指定 AND HCPE / PSV 教師 | 教師全体の局面数 (file size から自動計算) |
| `--superbatches` 未指定 AND HCPE3 / pack 教師 | エラー (= 可変長レコードなので教師サイズ不明、明示が必要) |

たとえば `--superbatches 4 --lr 0.001 --lr-min 0.00001` (1 epoch = 4 sb ≒ 400M 局面) の lr 推移:

| cycle 内位置 | t | step (geometric) | cos (cosine) |
|---|---|---|---|
| 0M (sb 1 頭) | 0.0 | 0.001 | 0.001 |
| 100M (sb 2 頭) | 0.25 | 0.000316 | 0.000856 |
| 200M (sb 3 頭) | 0.5 | 0.000100 | 0.000505 (midpoint) |
| 300M (sb 4 頭) | 0.75 | 0.0000316 | 0.000155 |
| 400M (sb 4 末) | 1.0 | 0.00001 | 0.00001 |
| 次 epoch sb 1 頭 | 0.0 | **0.001** ← warm restart | **0.001** ← warm restart |

step は **対数線形**: 各 batch で `(lr_min/lr_max)^(1/batches_per_epoch)` ≒ `0.99987` 倍ずつ下がる、超滑らかな指数減衰です。

⚠️ **`--lr-min` は step では必ず `> 0`**: geometric の式 `lr_max × (lr_min/lr_max)^t` が `lr_min = 0` だと t > 0 で即 0 になり破綻するため、CLI 起動時にエラーになります。`1e-5` 〜 `1e-6` あたりが典型。cos は 0 でも数学的に動きますが、警告は出ます。

実際の lr 推移は `<NNNN>/learn.log` の `lr` 列で確認できる ([§7.2 学習ログの読み方](7-result.md#72-学習ログ-learnlog-の読み方))。**bullet stdout の `LR dropped to X` は sb 開始時のみ表示** されるので、batch ごとの変化を見たいときは per-dir log を見てください。

#### ReduceLROnPlateau を使う

教師データ量が限られていて、epoch 長に合わせて `cos` を1周期回すよりも、validation loss を見ながら LR を下げたい場合は `--lr-schedule plateau` を使う。

`plateau` は各 superbatch の保存後に `--test-teacher` の `test_value_loss` を計測し、過去 best より下がっていなければ LR に `--lr-plateau-factor` を掛け、同じ superbatch の教師区間を下げた LR でもう一度学習する。このとき、棄却した更新は採用せず、model weight と optimizer state (Adam の momentum / variance) の両方をその superbatch 開始前に戻す。教師データが尽きた場合は同じ epoch のまま教師の先頭に戻り、`lr_min` に到達するまで継続する。次の LR が `--lr-min` を下回る段階になったら、最後に **ちょうど `--lr-min`** で同じ教師区間を1 superbatchだけ学習して、その epoch を終了する。次の epoch は、また `--lr` から plateau 判定を開始する。

`plateau` では 1 epoch の superbatch 数は固定しない。教師ファイルを読み切っても epoch は終わらず、教師先頭へ戻って続行する。`--superbatches` は「この数を超えたら打ち切る」という安全上限であり、通常は指定しない。superbatch の大きさだけを `--batches-per-superbatch` で決める。

validation loss が改善しなかった attempt は正式な checkpoint (`000N/`) には残さない。checkpoint と `summary-learn.log` に残るのは、採用された更新と最後の `lr_min` run だけ。

既存の checkpoint がある `--tag` で `--superbatches` を付けたり外したりすると、BulletOu は設定変更として扱い、auto resume を拒否する。古い checkpoint を意図して引き継ぐ場合だけ `--resume` を付ける。新しい実験として始めたい場合は `--tag` / `--output` を変える。

制約:

- `--test-teacher` 必須。validation loss がないと判定できない。
- `--save-rate 1` 必須。1 superbatchごとに検証してLRを更新するため。
- 現状は NNUE/SFNN 系の学習で使用する。

例:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_HALFKP \
    --max-epochs 10 \
    --lr 0.001 --lr-min 0.00001 \
    --lr-schedule plateau \
    --lr-plateau-factor 0.5
```

factor を緩めたい場合は `--lr-plateau-factor 0.7` のようにする。`--lr-plateau-min-delta 0.000001` のように指定すると、それ未満の微小な改善は「改善なし」とみなす。

#### `step` vs `cos` を比較したい

同じ教師・同じ arch で 2 回 run して `summary-learn.log` を並べると、どちらが効くかすぐ分かる。両方とも同じ `--lr-min` 値を共有できるので apples-to-apples 比較しやすい:

```bash
# stepwise (geometric decay)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --superbatches 4 --tag 5G-step \
    --lr-schedule step --lr-min 0.00001

# cosine (epoch ごとに 1 cycle)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --eval-type NNUE_KP --arch 256x2-32-32 \
    --max-epochs 10 --tag 5G-cos --superbatches 4 \
    --lr-schedule cos --lr-min 0.00001
```

出力先がそれぞれ `checkpoints/NNUE_KP-256x2-32-32-5G-step/` と `-5G-cos/` に分かれる。`summary-learn.log` の `test_value_accuracy` / `test_value_loss` 列を pandas / Excel で重ねれば比較完了。

### 教師を数えて `--superbatches` を決める

`step` / `cos` どちらでも cycle を 1 epoch ぴったりに揃えるには、まず **教師の総局面数** を知る必要がある。BulletOu には専用フラグ `--count-teacher` があり、`std::fs::metadata` でファイルサイズを読むだけなので **数百 GB の教師でも一瞬で完了** する (中身は read しない):

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

`step` / `cos` で HCPE3 / pack を使うなら、同じ corpus を HCPE / PSV に事前変換するか、`--superbatches` を手動指定してください。`plateau` は period を使わないので、この制約はない。

### 複数 epoch 回す

`--max-epochs N` を指定すると epoch を N 回実行する。`step` / `cos` では通常「教師データを N 周する」の意味になる。`plateau` では教師を複数周しながら `lr_min` 到達まで同じ epoch を続けるので、「plateau 学習を最大 N 回繰り返す」の意味になる。各 epoch 開始時に:
- LR scheduler が reset される (superbatch 1 から再開、`lr = --lr` に戻る — `step` でも `cos` でも同じ)
- データローダーが先頭にシークし直す

つまり N 回学習し直すに近い挙動。各 epoch ごとに lr が再下降するので、長時間学習で局所最適から脱出させたいときに使う。`cos` schedule で `--superbatches N` を指定すれば cycle = epoch で自動的に揃う (= 典型的な SGDR-style 用法)。`plateau` では `--superbatches` で cycle を揃える必要はない。

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
