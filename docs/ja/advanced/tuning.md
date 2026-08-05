# 学習設定を調整する

<a href="../../en/advanced/tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[チュートリアル 3: 学習を走らせる](../tutorial/3-train.md) のコマンドが動いたあとに読むページです。
最初はデフォルトのままで構いません。速度、保存頻度、検証頻度、学習率、loss、SFNN factorizer を変えたいときに参照してください。

## 1. ログに出てくる単位

BulletOu の学習ログでは `batch`、`superbatch`、`epoch` を分けて扱います。

| 名前 | 意味 |
|---|---|
| batch | GPU で 1 回の重み更新に使う局面数。デフォルトは `--batch-size 65536` |
| superbatch / sb | 進捗表示、検証、保存の単位。サイズは `--positions-per-superbatch` で決まる |
| epoch | `--superbatches` 個の sb をまとめた区切り。epoch 開始時に学習率が `--lr` に戻る |
| checkpoint | 再開用の `state.bin` と、エンジン用の `nn.bin` を保存したもの |
| validation / 検証 | `--test-teacher` の局面で accuracy と loss を測ること |

例:

```text
--batch-size 65536
--positions-per-superbatch 40000000
--superbatches 36
```

この場合、1 sb は `65536 x 610 = 39,976,960` 局面です。1 epoch は 36 sb なので約 14.4 億局面です。

## 2. よく変更するオプション

| 目的 | オプション | 例 |
|---|---|---|
| 1 epoch を何 sb にするか決める | `--superbatches` | `--superbatches 36` |
| 1 sb の局面数を決める | `--positions-per-superbatch` | `--positions-per-superbatch 40000000` |
| 何 epoch 学習するか決める | `--max-epochs` | `--max-epochs 3` |
| checkpoint 保存を減らす | `--save-rate` | `--save-rate 9999` なら epoch 末だけ保存しやすい |
| accuracy/loss を毎 sb 見る | `--validation-rate` | `--validation-rate 1` |
| tatara 風の StepLR に寄せる | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
| WRM loss を試す | `--win-rate-model` | `--win-rate-model --loss-pow-exp 2.5` |
| sigmoid loss の指数を変える | `--loss-pow-exp` | `--loss-pow-exp 1.5` |
| SFNN の factorizer を変える | `--sfnn-factorizer` | `--sfnn-factorizer none` |

詳しい一覧です。

| フラグ | 何を変えるか | デフォルト |
|---|---|---|
| `--backend` | 学習 backend。通常は `cuda-cpp` | `cuda-cpp` |
| `--batch-size` | 1 回の重み更新に使う局面数 | 65536 |
| `--positions-per-superbatch` | 1 sb の目標局面数。実際には `batch-size` の倍数に丸められる | 100000000 |
| `--teacher-shuffle-buffer-sbs` | 教師局面を何 sb 分まとめて shuffle するか。`4` なら 4 sb 分の buffer を 2 本使う | 1 |
| `--teacher-shuffle-buffer-batches` | shuffle buffer を batch 数で指定する。通常は `--teacher-shuffle-buffer-sbs` を使う | 省略 |
| `--teacher-shuffle-seed` | 教師 shuffle の seed | 0 |
| `--threads` | 局面変換に使う CPU worker 数 | auto |
| `--loader-threads` | 教師ファイル読み込み・decode 側の CPU worker 数 | auto |
| `--cuda-cpp-diagnostics-rate` | 速度調査用の診断ログを出す頻度 | 1 |
| `--superbatches` | 1 epoch の sb 数 | 省略 |
| `--max-epochs` | 最大 epoch 数 | 省略 |
| `--save-rate` | 何 sb ごとに checkpoint を保存するか | 20 |
| `--validation-rate` | 何 sb ごとに検証するか。保存頻度とは独立 | `--save-rate` と同じ |
| `--test-positions` | 検証に使う局面数。省略すると検証ファイルの全局面を使う | 全件 |
| `--test-batch-size` | 検証時の GPU batch size。VRAM が足りないときだけ下げる | 65536 |
| `--save-epoch-end` / `--no-save-epoch-end` | epoch 末に保存するか | on |
| `--lr` | epoch 開始時の学習率 | 0.000875 |
| `--lr-min` | 学習率の下限 | 0.00001 |
| `--lr-schedule` | 学習率 schedule。まずは `step` でよい | `step` |
| `--lr-step-gamma` | `step` で学習率に掛ける係数 | auto / 0.992 |
| `--lr-step-positions` | 何局面ごとに学習率を下げるか。省略時は 1 sb ごと | 省略 |
| `--lambda` | 教師評価値と勝敗結果を混ぜる比率 | 1.0 |
| `--scale` | `sigmoid(score / scale)` の scale。省略時は教師データから推定 | 省略 |
| `--scale-calibration-positions` | `--scale` 推定に使う教師先頭局面数。`0` なら内蔵値を使う | 100000 |
| `--win-rate-model` | prediction 側に WRM 曲線を使う | off |
| `--loss-pow-exp` | `|prediction - target|^p` の `p`。`2.0` は二乗誤差 | 2.0 |
| `--wrm-nnue2score` | WRM で network output を評価値 scale に戻す係数 | 600 |
| `--wrm-target-calibration-positions` | WRM target 曲線の推定に使う教師先頭局面数 | 100000 |
| `--wrm-target-offset` / `--wrm-target-scaling` | WRM target 曲線を手で指定する | 省略 |
| `--sfnn-factorizer` | SFNN の bucket 間で共通成分を共有する方法 | `shared` |
| `--optimizer` | optimizer | `ranger` |
| `--optimizer-weight-decay` | weight decay | 0.0 |
| `--optimizer-epsilon` / `--optimizer-beta1` / `--optimizer-beta2` | optimizer の詳細パラメータ | 省略 |

## 3. 学習率 schedule

`--lr-schedule step` では、一定間隔で

```text
lr = lr * gamma
```

のように学習率を下げます。`--lr-step-positions` を省略すると 1 sb ごとに下がります。次の epoch の sb 1 では `--lr` に戻ります。

tatara 風に `gamma=0.992` を指定する例です。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --lr 0.000875 \
    --lr-min 0.00001 \
    --lr-schedule step \
    --lr-step-gamma 0.992 \
    --tag step-gamma-0992
```

`--lr-step-gamma` を省略して `--superbatches` を指定すると、1 epoch の中で `--lr` から `--lr-min` へ届くように BulletOu が gamma を計算します。

## 4. 教師データから学習ラベルを作る

教師データには主に次の 2 つがあります。

| 情報 | 意味 |
|---|---|
| 教師評価値 | 教師エンジンがその局面を何点と見たか |
| 勝敗結果 | その対局が最終的に勝ち・引き分け・負けのどれになったか |

`--lambda` は、この 2 つをどう混ぜるかを決めます。

```text
training label = lambda * label_from_teacher_score + (1 - lambda) * label_from_game_result
```

| `--lambda` | 意味 |
|---|---|
| `1.0` | 教師評価値だけを見る。デフォルト |
| `0.5` | 教師評価値と勝敗結果を半分ずつ混ぜる |
| `0.0` | 勝敗結果だけを見る |

## 5. loss と score scale

何も指定しない場合、BulletOu は sigmoid probability loss を使います。

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid(network_output)
loss       = |prediction - target|^p
```

`p` は `--loss-pow-exp` で指定します。デフォルトは `2.0` なので、このときは sigmoid-MSE です。

```bash
# sigmoid-MSE
--loss-pow-exp 2.0

# 誤差の小さい/大きい局面への重み付けを変える実験
--loss-pow-exp 1.5
--loss-pow-exp 2.5
```

`scale` は教師評価値を 0〜1 のラベルへ写すための係数です。`--scale` を省略すると、BulletOu は教師データの先頭局面から推定します。推定では勝ち・負けが付いている局面だけを使い、引き分け局面は使いません。

```bash
# scale を自動推定する
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --tag sigmoid-auto-scale
```

推定に使う局面数を変える場合:

```bash
--scale-calibration-positions 300000
```

比較実験で scale を固定したい場合:

```bash
--scale 600
```

## 6. WRM loss を試す

WRM は win-rate model の略です。loss は同じ 0〜1 の空間で計算しますが、prediction 側の勝率変換に WRM 曲線を使います。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --win-rate-model \
    --loss-pow-exp 2.5 \
    --tag wrm-pow25
```

WRM でも loss は次の形です。

```text
loss = |prediction - target|^p
```

`--loss-pow-exp` は sigmoid loss と WRM loss の両方に効きます。

`--wrm-nnue2score` は network output を評価値 scale に戻す係数です。デフォルトは `600` です。

## 7. 教師データ上の score → winrate を確認する

教師評価値と勝敗結果の関係は、教師データによって変わります。次の診断コマンドで、単純 sigmoid と WRM のどちらが実データに合っているかを確認できます。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --analyze-score-winrate \
    --fit-positions 100000 \
    --analyze-positions 1000000 \
    --bin-size 50 \
    --score-winrate-csv score-winrate.csv
```

このコマンドは学習しません。先頭 `--fit-positions` 局面で曲線を推定し、その後の `--analyze-positions` 局面で BCE / Brier score と score bucket ごとの実測勝率を出します。推定と BCE / Brier score では、勝ち・負けが付いている局面だけを使います。

| 出力 | 意味 |
|---|---|
| `sigmoid(score/s)` | `score` を 1 つの scale で 0〜1 に変換する形 |
| `WRM(offset,scale)` | offset と scale を使う形 |
| `heldout_bce` | 小さいほど勝敗統計に合っている |
| `heldout_brier` | 小さいほど確率予測として合っている |
| `empirical` | その score bucket での `wins / (wins + losses)` |

## 8. SFNN factorizer

SFNN では `k3k3` や `hand1024` のように bucket を増やせます。bucket が多いほど表現力は上がりますが、1 bucket あたりの教師局面は減ります。

factorizer は、bucket ごとに完全に別の重みを持つのではなく、bucket 間で共通成分を共有する仕組みです。教師密度が足りないときの過学習を抑えたり、学習を安定させたりする目的で使います。

| 指定 | 意味 |
|---|---|
| `--sfnn-factorizer shared` | bucket 全体で共通成分を持つ。デフォルト |
| `--sfnn-factorizer none` | factorizer を使わない |
| `--sfnn-factorizer axis` | arch に存在する軸をまとめて有効化する。例: `hand1024_k3k3` なら king と hand の両方 |
| `--sfnn-factorizer king=axis,hand=axis` | 軸ごとに明示する |
| `--sfnn-factorizer king=axis,hand=shared` | king は軸方向、hand は共通成分だけにする |

例:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k29k29 \
    --sfnn-factorizer king=axis \
    --tag k29-axis
```

## 9. 保存と検証の頻度

保存と検証は別々に指定できます。

```bash
--save-rate 20 --validation-rate 1
```

これは「checkpoint は 20 sb ごとに保存し、accuracy/loss は毎 sb 測る」という意味です。

epoch 末の保存はデフォルトで有効です。epoch 末だけ保存したい場合は、1 epoch 内で到達しない大きな `--save-rate` を指定します。

```bash
--save-rate 9999
```

## 10. 速度を見るときのログ

速度を見るときは stdout の `[train]` 行を見ます。

```text
[train]  epoch 1  sb 12/36  this-sb=39,976,960 pos (...)  wall=...s  train=...s  pos/s=...
```

| 表示 | 意味 |
|---|---|
| `wall` | その sb の実時間。検証や保存も含む |
| `train` | 学習処理そのものの時間。検証・保存は含まない |
| `pos/s` | `train` から計算した学習速度 |

GPU が空いているのに `pos/s` が低い場合は、教師局面の読み込み、decode、shuffle が詰まっている可能性があります。`cuda-cpp-diagnostics.log` の teacher queue wait も見てください。
