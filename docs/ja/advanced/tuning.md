# 学習設定を調整する

<a href="../../en/advanced/tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[チュートリアル 3: 学習を走らせる](../tutorial/3-train.md) のコマンドが動いたあとに読むページです。最初の学習ではデフォルト値のままで十分です。速度、保存頻度、検証頻度、学習率、loss、SFNN factorizer を調整したくなったら、このページを参照してください。

## 1. ログに出てくる単位

BulletOu の学習ログでは `batch`、`superbatch`、`epoch` を使います。

| 名前 | 意味 |
|---|---|
| batch | GPUで1回の重み更新に使う局面数。デフォルトは `--batch-size 65536` |
| superbatch / sb | 進捗表示、検証、保存の単位。サイズは `--positions-per-superbatch` で指定 |
| epoch | `--superbatches` 個の sb をまとめた区切り。epoch開始時に学習率は `--lr` に戻る |
| checkpoint | 再開用の `state.bin` と、エンジン用の `nn.bin` |
| validation / 検証 | `--test-teacher` の局面で accuracy と loss を測ること |

例:

```text
--batch-size 65536
--positions-per-superbatch 40000000
--superbatches 36
```

この場合、1 sb は `65536 x 610 = 39,976,960` 局面です。1 epoch は 36 sb なので、約 14.4 億局面です。

## 2. よく変更するオプション

| 目的 | オプション | 例 |
|---|---|---|
| 1 epoch を何 sb にするか決める | `--superbatches` | `--superbatches 36` |
| 1 sb の局面数を決める | `--positions-per-superbatch` | `--positions-per-superbatch 40000000` |
| 何 epoch 学習するか決める | `--max-epochs` | `--max-epochs 3` |
| checkpoint保存を減らす | `--save-rate` | `--save-rate 9999` ならepoch末保存だけにしやすい |
| accuracy/loss を毎 sb 測る | `--validation-rate` | `--validation-rate 1` |
| StepLRの減衰率を指定する | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
| sigmoid loss の指数を変える | `--loss-pow-exp` | `--loss-pow-exp 1.5` |
| SFNN の factorizer を変える | `--sfnn-factorizer` | `--sfnn-factorizer none` |

主なオプション一覧:

| フラグ | 何を変えるか | デフォルト |
|---|---|---|
| `--backend` | 学習backend。通常は `cuda-cpp` | `cuda-cpp` |
| `--batch-size` | 1回の重み更新に使う局面数 | 65536 |
| `--positions-per-superbatch` | 1 sb の目標局面数。実際には `batch-size` の倍数に丸められる | 100000000 |
| `--teacher-shuffle-buffer-sbs` | 何 sb 分の教師局面をRAM上でshuffleするか。`4`なら4 sb分のbufferを2本使う | 1 |
| `--teacher-shuffle-buffer-batches` | shuffle bufferをbatch数で指定する。通常は `--teacher-shuffle-buffer-sbs` を使う | 省略 |
| `--teacher-shuffle-seed` | 学習中shuffleのseed | 0 |
| `--threads` | 局面変換に使うCPU worker数 | auto |
| `--loader-threads` | 教師ファイル読み込み/decode側のCPU worker数 | auto |
| `--cuda-cpp-diagnostics-rate` | 診断ログを何 sb ごとに書くか | 1 |
| `--superbatches` | 1 epoch の sb 数 | 省略 |
| `--max-epochs` | 最大 epoch 数 | 省略 |
| `--save-rate` | 何 sb ごとにcheckpoint保存するか | 20 |
| `--validation-rate` | 何 sb ごとに検証するか。保存頻度とは独立 | `--save-rate` と同じ |
| `--test-positions` | 検証に使う局面数。省略すると検証ファイルの全局面を使う | 全件 |
| `--test-batch-size` | 検証時のGPU batch size。VRAM不足のときだけ下げる | 65536 |
| `--save-epoch-end` / `--no-save-epoch-end` | epoch末に保存するか | on |
| `--lr` | epoch開始時の学習率 | 0.000875 |
| `--lr-min` | 学習率の下限 | 0.00001 |
| `--lr-schedule` | 学習率schedule。まずは `step` でよい | `step` |
| `--lr-step-gamma` | `step` で学習率に掛ける係数 | auto / 0.992 |
| `--lr-step-positions` | 何局面ごとに学習率を下げるか。省略時は1 sbごと | 省略 |
| `--lambda` | 教師評価値と勝敗結果を混ぜる比率 | 1.0 |
| `--scale` | `sigmoid(score / scale)` の scale。省略時は固定値 290 | 省略 |
| `--loss-pow-exp` | `|prediction - target|^p` の `p`。`2.0` は二乗誤差 | 2.0 |
| `--sfnn-factorizer` | SFNNのbucket間で共通成分を共有する方法 | `shared` |
| `--optimizer` | optimizer | `ranger` |
| `--optimizer-weight-decay` | weight decay | 0.0 |
| `--optimizer-epsilon` / `--optimizer-beta1` / `--optimizer-beta2` | optimizerの詳細パラメータ | 省略 |

## 3. 学習率schedule

`--lr-schedule step` では、一定間隔で次のように学習率を下げます。

```text
lr = lr * gamma
```

`--lr-step-positions` を省略すると、1 sbごとに学習率が下がります。次のepochのsb 1では `--lr` に戻ります。

`gamma=0.992` を明示する例:

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

`--lr-step-gamma` を省略して `--superbatches` を指定すると、1 epoch の中で `--lr` から `--lr-min` へ近づくように BulletOu が gamma を計算します。

## 4. 教師データから学習ラベルを作る

教師データには主に次の2つが入っています。

| 情報 | 意味 |
|---|---|
| 教師評価値 | 教師エンジンがその局面を何点と見たか |
| 勝敗結果 | その対局が最終的に勝ち・引き分け・負けのどれになったか |

`--lambda` は、この2つをどう混ぜるかを決めます。

```text
training label = lambda * label_from_teacher_score + (1 - lambda) * label_from_game_result
```

| `--lambda` | 意味 |
|---|---|
| `1.0` | 教師評価値だけを見る。デフォルト |
| `0.5` | 教師評価値と勝敗結果を半分ずつ混ぜる |
| `0.0` | 勝敗結果だけを見る |

## 5. loss と score scale

BulletOu の学習lossは sigmoid probability loss です。

```text
target     = sigmoid(teacher_score / scale)
prediction = sigmoid(network_output)
loss       = |prediction - target|^p
```

`p` は `--loss-pow-exp` で指定します。デフォルトは `2.0` なので、このときはsigmoid空間での二乗誤差です。

```bash
# 二乗誤差
--loss-pow-exp 2.0

# 誤差の大きさに対する重み付けを変える実験
--loss-pow-exp 1.5
--loss-pow-exp 2.5
```

`scale` は教師評価値を 0〜1 のラベルへ写すための係数です。`--scale` を省略すると、BulletOu は固定値 `290` を使います。

教師データに含まれる勝敗結果は、常に教師評価値の較正に使えるとは限りません。たとえば、弱い対局者の棋譜を別エンジンでre-scoreした教師では、勝敗結果と教師評価値の関係が学習したい評価関数の性質を表していない場合があります。そのため、BulletOu は学習時に勝敗結果から `scale` を自動推定しません。

```bash
# 固定scale 290で学習する
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --tag sigmoid-scale290
```

比較実験でscaleを固定したい場合:

```bash
--scale 600
```

## 6. 教師評価値と勝敗結果の関係を確認する

学習には使いませんが、教師データの性質を調べたいときは、次の診断コマンドを使えます。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --analyze-score-winrate \
    --fit-positions 100000 \
    --analyze-positions 1000000 \
    --bin-size 50 \
    --score-winrate-csv score-winrate.csv
```

このコマンドは学習しません。先頭 `--fit-positions` 局面で `sigmoid(score / scale)` のscaleを推定し、その後の `--analyze-positions` 局面で BCE / Brier score とscore bucketごとの実測勝率を出力します。推定とBCE / Brier scoreには、勝ち・負けが付いている局面だけを使います。

| 出力 | 意味 |
|---|---|
| `sigmoid(score/s)` | `score` を1つのscaleで0〜1へ変換する形 |
| `heldout_bce` | 小さいほど勝敗統計に合っている |
| `heldout_brier` | 小さいほど確率予測として合っている |
| `empirical` | そのscore bucketでの `wins / (wins + losses)` |

## 7. SFNN factorizer

SFNNでは、`k3k3` や `hand1024` のようにbucketを増やせます。bucketが多いほど表現力は上がりますが、bucketあたりの教師局面は減ります。

factorizerは、bucketごとに完全に別の重みを持つのではなく、bucket間で共通成分を共有する仕組みです。教師密度が足りないときの過学習を抑えたり、学習を安定させたりする目的で使います。

| 指定 | 意味 |
|---|---|
| `--sfnn-factorizer shared` | bucket全体で共通成分を持つ。デフォルト |
| `--sfnn-factorizer none` | factorizerを使わない |
| `--sfnn-factorizer axis` | archに存在する軸をまとめて有効化する。例: `hand1024_k3k3` なら king と hand |
| `--sfnn-factorizer king=axis,hand=axis` | 軸ごとに明示する |
| `--sfnn-factorizer king=axis,hand=shared` | kingは軸方向、handは共通成分だけにする |

例:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k29k29 \
    --sfnn-factorizer king=axis \
    --tag k29-axis
```

## 8. 保存と検証の頻度

保存と検証は別々に指定できます。

```bash
--save-rate 20 --validation-rate 1
```

これは「checkpointは20 sbごとに保存し、accuracy/lossは毎 sb 測る」という意味です。

epoch末保存はデフォルトで有効です。epoch末だけ保存したい場合は、epoch内で到達しない大きな `--save-rate` を指定します。

```bash
--save-rate 9999
```

## 9. 速度を見るときのログ

速度を見るときは stdout の `[train]` 行を見ます。

```text
[train]  epoch 1  sb 12/36  this-sb=39,976,960 pos (...)  wall=...s  train=...s  pos/s=...
```

| 表示 | 意味 |
|---|---|
| `wall` | その sb の実時間。検証や保存も含む |
| `train` | 学習処理そのものの時間。検証・保存は含まない |
| `pos/s` | `train` から計算した学習速度 |

GPUが空いているのに `pos/s` が低い場合は、教師局面の読み込み、decode、shuffleが詰まっている可能性があります。`cuda-cpp-diagnostics.log` の teacher queue wait を見てください。
