# 6. 学習設定を調整する

<a href="../../en/tutorial/6-tune.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[4. 学習を走らせる](4-train.md) のコマンドで動くことを確認したあとに読む章です。最初はデフォルトのままで構いません。速度が出ない、保存頻度を変えたい、tatara と条件を揃えたい、loss を変えて試したい、という段階になってから触ってください。

## 6.1 まず覚える単位

BulletOu のログには `batch`、`superbatch`、`epoch` が出てきます。ここが混ざると設定を読み違えます。

| 名前 | 意味 |
|---|---|
| batch | GPU で 1 回重みを更新する局面数。デフォルトは `--batch-size 65536` |
| superbatch / sb | 進捗表示、検証、保存の単位。`--positions-per-superbatch` で決まる |
| epoch | `--superbatches` 個の superbatch をまとめた学習の区切り。学習率は epoch の先頭で `--lr` に戻る |
| checkpoint | 再開用の `state.bin` と、エンジンで使う `nn.bin` を保存したもの |
| validation / 検証 | `--test-teacher` の局面で accuracy と loss を測ること。学習用教師とは別ファイルを使う |

例:

```text
--batch-size 65536
--positions-per-superbatch 40000000
--superbatches 36
```

この場合、1 sb は `65536 × 610 = 39,976,960` 局面です。1 epoch は 36 sb なので、約 14.4 億局面です。

`--superbatches` を指定した場合、epoch は「教師データを 1 周した」という意味ではありません。学習率を戻す区切り、保存する区切り、検証結果を比較する区切りです。教師データは、EOF に到達したときだけ先頭へ戻ります。

## 6.2 よく変更するオプション

まず触ることが多いものだけを先にまとめます。

| 目的 | 指定するもの | 例 |
|---|---|---|
| 1 epoch を何 sb にするか決める | `--superbatches` | `--superbatches 36` |
| 1 sb の局面数を決める | `--positions-per-superbatch` | `--positions-per-superbatch 40000000` |
| 何 epoch 学習するか決める | `--max-epochs` | `--max-epochs 3` |
| checkpoint 保存を減らす | `--save-rate` | `--save-rate 9999` なら epoch 末だけ保存しやすい |
| accuracy/loss は毎 sb 見る | `--validation-rate` | `--validation-rate 1` |
| tatara と同じように LR を落とす | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
| WRM loss の指数を変える | `--loss-pow-exp` | `--loss-pow-exp 2.5` |
| SFNN の factorizer を切る | `--sfnn-factorizer` | `--sfnn-factorizer none` |

詳細な一覧です。

| フラグ | 何を変えるか | デフォルト |
|---|---|---|
| `--backend` | 学習処理。通常は `cuda-cpp` のままでよい | `cuda-cpp` |
| `--batch-size` | 1 回の重み更新に使う局面数。大きいほど 1 回の勾配は安定するが VRAM を使う | 65536 |
| `--positions-per-superbatch` | 1 sb の目標局面数。実際には `batch-size` の倍数に切り捨てられる | 100000000 |
| `--teacher-shuffle-buffer-sbs` | 学習時に教師局面を何 sb 分まとめてシャッフルするか。`4` なら 4 sb 分の読み込み領域を 2 個使う。`0` で無効 | 1 |
| `--teacher-shuffle-buffer-batches` | シャッフル領域を batch 数で細かく指定する。通常は `--teacher-shuffle-buffer-sbs` を使えばよい | 省略 |
| `--teacher-shuffle-seed` | 学習時シャッフルの seed | 0 |
| `--threads` | 教師局面の変換に使う CPU worker 数。CPU が詰まるなら明示する | auto |
| `--loader-threads` | 教師ファイル読み込み・decode 側の CPU worker 数。GPU への供給が遅いときに調整する | auto |
| `--cuda-cpp-diagnostics-rate` | 速度低下の原因調査用ログを出す頻度。通常はそのままでよい | 1 |
| `--superbatches` | 1 epoch を何 sb にするか | 省略時は教師EOFなどで決まる |
| `--max-epochs` | 最大何 epoch 学習するか | 省略時は自動停止まで |
| `--save-rate` | 何 sb ごとに checkpoint を保存するか | 20 |
| `--validation-rate` | 何 sb ごとに検証するか。保存頻度とは独立 | `--save-rate` と同じ |
| `--test-positions` | 検証に使う局面数。省略時は検証ファイルの全局面 | 全件 |
| `--test-batch-size` | 検証時の GPU batch size。VRAM 不足時だけ下げる | 65536 |
| `--save-epoch-end` / `--no-save-epoch-end` | epoch 末を保存するか | on |
| `--lr` | epoch 先頭の学習率 | 0.000875 |
| `--lr-min` | 学習率の下限 | 0.00001 |
| `--lr-schedule` | 学習率の下げ方。通常は `step` | `step` |
| `--lr-step-gamma` | `step` で学習率に掛ける倍率 | 自動 / 0.992 |
| `--lr-step-positions` | 何局面ごとに学習率を下げるか。省略時は 1 sb ごと | 省略 |
| `--lambda` | 教師評価値と勝敗結果を混ぜる比率 | 1.0 |
| `--win-rate-model` | 教師評価値を勝率に直して loss を計算する。デフォルトで有効 | on |
| `--loss-sigmoid-mse` | WRM ではなく `sigmoid(model_output)` の MSE で学習する比較用設定 | off |
| `--loss-pow-exp` | WRM loss の指数。`2.0` は二乗誤差、`2.5` も試験候補 | 2.0 |
| `--wrm-nnue2score` | network output を評価値スケールへ戻す倍率 | 600 |
| `--wrm-target-calibration-positions` | 教師評価値→勝率ラベルの係数を、教師先頭何局面から推定するか。`0` なら推定せず既定係数を使う | 100000 |
| `--wrm-target-offset` / `--wrm-target-scaling` | 教師評価値→勝率ラベルの係数を手で指定する。通常は指定しない | 省略 |
| `--sfnn-factorizer` | SFNN の bucket 間で共通成分を共有する方法。通常は `shared` | `shared` |
| `--optimizer` | optimizer。通常は `ranger` のままでよい | `ranger` |
| `--optimizer-weight-decay` | weight decay | 0.0 |
| `--optimizer-epsilon` / `--optimizer-beta1` / `--optimizer-beta2` | optimizer の細かい係数。比較実験用 | 省略 |

## 6.3 学習率をどう下げるか

`--lr-schedule step` がデフォルトです。`step` は、一定間隔で

```text
lr = lr * gamma
```

と学習率を下げます。`--lr-step-positions` を省略すると、1 sb ごとに下がります。epoch の先頭では `--lr` に戻ります。

tatara と同じように `gamma=0.992` を明示するなら、こう書きます。

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

`--lr-step-gamma` を書かずに `--superbatches` を指定すると、1 epoch の中で `--lr` から `--lr-min` へ届くように BulletOu が `gamma` を計算します。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --positions-per-superbatch 40000000 \
    --superbatches 36 \
    --max-epochs 3 \
    --lr 0.000875 \
    --lr-min 0.00001 \
    --lr-schedule step \
    --tag step-auto-gamma
```

この例では、各 epoch の 36 sb で `--lr` から `--lr-min` へ下がり、次の epoch の sb 1 で `--lr` に戻ります。

`geometric` と `cos` は、1 epoch をかけて滑らかに下げる方式です。まずは `step` だけで十分です。

| 値 | 動き |
|---|---|
| `step` | sb ごと、または指定局面数ごとに階段状に下げる |
| `geometric` | 毎 batch 少しずつ一定倍率で下げる |
| `cos` | cosine カーブで滑らかに下げる |
| `plateau` | 検証 loss / accuracy が改善しないときだけ学習率を下げて同じ区間をやり直す |

## 6.4 教師評価値を学習用ラベルにする

教師データには、主に次の 2 種類の情報があります。

| 情報 | 意味 |
|---|---|
| 教師評価値 | 教師エンジンがその局面を何点と見たか |
| 勝敗結果 | その対局が最終的に勝ち・負け・引き分けのどれになったか |

`--lambda` は、この 2 つをどれだけ混ぜるかを決めます。

```text
学習ラベル = λ × 教師評価値由来のラベル + (1 - λ) × 勝敗結果由来のラベル
```

| `--lambda` | 意味 |
|---|---|
| `1.0` | 教師評価値だけを見る。デフォルト |
| `0.5` | 教師評価値と勝敗結果を半分ずつ混ぜる |
| `0.0` | 勝敗結果だけを見る |

通常は `1.0` から始めます。勝敗結果も混ぜたい実験だけ `0.5` や `0.7` などを試してください。

## 6.5 WRM loss

WRM は win-rate-model の略です。BulletOu では、教師評価値をそのまま loss に入れるのではなく、まず「勝率っぽい 0〜1 の値」に変換してから学習します。これがデフォルトです。

たとえば `+300` の教師評価値を「だいたい勝ちやすい局面」として 0〜1 の値に直し、network output も同じ 0〜1 の値に直して、その差を loss にします。

WRM で変わるのは次の 3 つです。

| 対象 | 何をするか |
|---|---|
| 教師評価値 | 勝率ラベルに変換する |
| network output | 勝率予測に変換する |
| loss | `abs(勝率ラベル - 勝率予測)^p` を使う |

この `p` が `--loss-pow-exp` です。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --loss-pow-exp 2.5 \
    --tag wrm-pow25
```

### 教師評価値→勝率ラベルの係数

教師評価値を勝率ラベルに変換するには、評価値スケールに合った係数が必要です。BulletOu はデフォルトで、教師データ先頭 100,000 局面を見て、この係数を推定します。

```text
教師評価値と実際の勝敗結果を見て、
「この教師では評価値 +何点がどれくらい勝ちに対応するか」
を起動時に推定する
```

この推定に使う局面数は `--wrm-target-calibration-positions` で変えられます。

```bash
# 先頭 300,000 局面で係数を推定する
--wrm-target-calibration-positions 300000
```

推定を使わず、あらかじめ決めた係数を使いたい場合だけ `0` を指定します。

```bash
--wrm-target-calibration-positions 0
```

この場合は `offset=270`, `scaling=380` を使います。実験上どうしても値を手で固定したい場合だけ、次のように明示できます。

```bash
--wrm-target-offset 270 --wrm-target-scaling 380
```

通常の学習では、これらを指定しないでください。デフォルトの自動推定を使うのが基本です。

### `--wrm-nnue2score`

`--wrm-nnue2score` は、network output を評価値スケールに戻す倍率です。デフォルトは `600` です。tatara と条件を揃えるときや、明確に比較実験をするとき以外は変更しません。

### WRM ではない loss で比較したい場合

WRM を使わず、`sigmoid(model_output)` に対する MSE で比較したい場合は、次を指定します。

```bash
--loss-sigmoid-mse
```

WRM と sigmoid-MSE では loss の式が違うので、数値をそのまま横比較しないでください。同じ loss 設定同士で比較します。

## 6.6 SFNN factorizer

SFNN では、`k3k3` や `hand1024` のように bucket を増やすと、bucket ごとに別の後段ネットワークを持ちます。bucket が多いほど表現力は増えますが、各 bucket に届く教師局面は減ります。

factorizer は、bucket ごとに完全に別々の重みを持つのではなく、bucket 間で共通しやすい成分を別枠で持つ仕組みです。過学習を抑えたり、少ない教師量でも学習を安定させたりする目的で使います。

| 指定 | 意味 |
|---|---|
| `--sfnn-factorizer shared` | bucket 全体で共通成分を持つ。デフォルト |
| `--sfnn-factorizer none` | factorizer を使わない |
| `--sfnn-factorizer axis` | arch に存在する軸をまとめて有効化する。例: `hand1024_k3k3` なら king と hand の両方 |
| `--sfnn-factorizer king=axis,hand=axis` | 軸ごとに明示する |
| `--sfnn-factorizer king=axis,hand=shared` | king は軸方向、hand は共有成分だけにする |

例:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k29k29 \
    --sfnn-factorizer king=axis \
    --tag k29-axis
```

途中で factorizer 設定を変えて再開する場合、必ず `--resume` を明示してください。BulletOu は設定変更を検出して、意図しない再開を止めます。

## 6.7 保存と検証の頻度

保存と検証は別々に指定できます。

```bash
--save-rate 20 --validation-rate 1
```

これは「checkpoint は 20 sb ごとに保存するが、accuracy/loss は毎 sb 測る」という意味です。

epoch 末の保存はデフォルトで有効です。1 epoch が 36 sb で、epoch 末だけ保存したい場合は、次のように大きな `--save-rate` を指定します。

```bash
--save-rate 9999
```

この場合、36 sb の中では `save-rate` に到達しないので、epoch 末保存だけが残ります。

## 6.8 速度が遅いときに見るところ

速度を見るときは stdout の `[train]` 行を見ます。

```text
[train]  epoch 1  sb 12/36  this-sb=39,976,960 pos (...)  wall=...s  train=...s  pos/s=...
```

| 表示 | 意味 |
|---|---|
| `wall` | その sb の実時間。検証や保存も含む |
| `train` | 学習処理そのものの時間。検証・保存は含まない |
| `pos/s` | `train` から計算した学習速度 |

GPU が空いているのに `pos/s` が低い場合は、教師局面の読み込み・decode・シャッフルが詰まっている可能性があります。まず次を確認します。

- 教師データが遅いストレージにないか
- `--teacher-shuffle-buffer-sbs` が大きすぎないか
- `--threads` / `--loader-threads` が CPU を使い切っていないか
- `cuda-cpp-diagnostics.log` で teacher queue wait が大きくないか

## 6.9 optimizer

通常は `--optimizer ranger` のままで構いません。`--optimizer-weight-decay` もデフォルト `0.0` のままで始めます。

optimizer の細かい係数を変える場合は、他の条件を動かさず 1 つずつ比較してください。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ \
    --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --optimizer ranger \
    --optimizer-weight-decay 0.0 \
    --optimizer-beta1 0.9 \
    --optimizer-beta2 0.999 \
    --optimizer-epsilon 0.0000001 \
    --tag optimizer-test
```

## 6.10 迷ったときの基本形

SFNN で 1 epoch = 36 sb、毎 sb 検証、epoch 末だけ保存する例です。

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher C:\shogi\teacher\sojo `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_7_64_k3k3 `
  --positions-per-superbatch 40000000 `
  --superbatches 36 `
  --max-epochs 1 `
  --lr 0.000875 `
  --lr-min 0.000030 `
  --lr-schedule step `
  --optimizer ranger `
  --optimizer-weight-decay 0.0 `
  --save-rate 9999 `
  --validation-rate 1 `
  --tag sfnn-sojo-36sb
```

次へ: [7. 結果を確認する](7-result.md) — accuracy / loss / ログの読み方

前へ: [4. 学習を走らせる](4-train.md)
