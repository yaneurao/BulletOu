# 6. 学習をチューニング — スケジュールと教師ターゲット

<a href="../../en/tutorial/6-tune.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[4. 学習を走らせる](4-train.md) のデフォルト設定で動くことを確認したら、この章で **学習を調整するためのフラグ群** を紹介する。**最初の学習ではすべてデフォルトのままで問題ない**。チューニングが必要になったときに戻ってくる位置付け。

## 6.1 学習スケジュール

ログに出てくる `superbatch` は **checkpoint や学習率を更新するためのまとまり**で、デフォルトで約 1 億局面ぶん。

主要なフラグ:

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--backend` | 学習 backend。対応 backend は Windows-native `cuda-cpp`。公開 eval type はすべて `cuda-cpp` で学習可能 | `cuda-cpp` |
| `--batch-size` | 1 gradient step あたりの局面数。省略時は tatara に合わせて 65536 | 65536 |
| `--positions-per-superbatch` | 1 superbatch あたりの目標局面数。実際には `--batch-size` の倍数へ切り捨て | 100000000 |
| `--teacher-shuffle-buffer-sbs` | 学習時 teacher shuffle window を superbatch 単位で指定する。`4` なら 4 superbatch 分の CPU window を 2 個確保する。`0` で無効。通常はこちらを使う | 1 |
| `--teacher-shuffle-buffer-batches` | 学習時 teacher shuffle window を mini-batch 単位で細かく指定する。`--teacher-shuffle-buffer-sbs` とは同時指定不可 | 省略 |
| `--teacher-shuffle-seed` | 学習時 teacher shuffle の base seed | 0 |
| `--threads` | teacher batch preparation の CPU worker 数。省略または `0` なら OS の論理スレッド数 (`available_parallelism()`) を使う。明示値はそのまま使う | auto (= logical threads) |
| `--loader-threads` | HCPE decode の CPU worker 数。省略または `0` なら `--threads` と同じ auto 値。decode が GPU upload/main thread を圧迫する場合は明示的に下げる | auto (= logical threads) |
| `--cuda-cpp-diagnostics-rate` | SFNN の superbatch 診断ログ。`cuda-cpp-diagnostics.log` に teacher queue wait / load / prepare と CUDA 代表 step の stage 時間を書く。`1` なら毎 sb、`N` なら N sb ごとに CUDA stage を profile、`0` で無効 | 1 |
| `--superbatches` | 1 epoch を何 superbatch にするか。`geometric` / `cos` では LR cycle 長そのもの。`step` では epoch 内の処理上限、`plateau` では安全上限 | 上限なし (= 非 plateau は教師EOFまで、plateau は `lr_min` 到達まで) |
| `--max-epochs` | epoch を最大何回実行するか。`--max-epoch` も alias として使える。`step` / `geometric` / `cos` では LR cycle を最大何回繰り返すか、`plateau` では plateau epoch を最大何回繰り返すか。`--test-teacher` があれば epoch 末の loss/accuracy がどちらも改善しない時点で上限前でも停止 | 省略時は epoch 上限なし |
| `--save-rate` | N superbatch ごとに checkpoint を保存。デフォルトでは save-rate 境界でなくても各 epoch の最後の superbatch も保存する。`plateau` では引き続き `--save-rate 1` が必要 | 20 |
| `--validation-rate` | `--test-teacher` の accuracy/loss 検証を N superbatch ごとに実行する。checkpoint 保存とは独立。省略時は `--save-rate` と同じ。`--validation-rate 1 --save-rate 20` なら毎 sb 検証しつつ save は 20 sb ごと | `--save-rate` と同じ |
| `--test-positions` | `--test-teacher` から検証に使う局面数。省略時は、固定長形式の検証ファイル/フォルダに含まれる全局面を使う。意図的に小さい random/sequential subset で比較したいときだけ指定する | 全件 |
| `--test-batch-size` | `--test-teacher` 検証時の GPU batch size。大きいほど validation の分割 overhead が減るが VRAM を使う。cudaMalloc になる場合だけ下げる | 65536 |
| `--save-epoch-end` / `--no-save-epoch-end` | 各 epoch 最後の superbatch を暗黙に保存するかどうか | on |
| `--lr` | 初期学習率 (lr_max。1 cycle の頭の値) | 0.000875 |
| `--optimizer` | optimizer。対応 optimizer は `ranger`。`ranger` は BulletOu の RAdam+Lookahead 実装で、Ranger21 と同一実装ではない | `ranger` |
| `--lr-schedule` | `step` (= 階段状 StepLR)、`geometric` (= 対数線形)、`cos` (= cosine annealing)、`plateau` (= validation 指標が改善しないときだけ LR を下げる) | `step` |
| `--lr-min` | 最小 lr。`step` / `plateau` では下限、`geometric` / `cos` では cycle 末で到達する値 | 0.00001 |
| `--lr-step-gamma` | `step` で LR に掛ける係数。省略時、`--superbatches` があれば 1 epoch 内で `--lr` から `--lr-min` へ届く値を自動計算する。epoch 長が決まらない場合は `0.992` | 自動 / 0.992 |
| `--lr-step-positions` | `step` で何局面ごとに LR を落とすか。省略時は 1 superbatch | 省略 |
| `--lr-plateau-factor` | `plateau` で監視指標が改善しなかったときに LR へ掛ける係数 | 0.5 |
| `--lr-plateau-min-delta` | `plateau` で改善とみなす最小 loss 差 | 0.0 |
| `--lr-plateau-monitor` | `plateau` で採用判定に使う指標。`loss` / `accuracy` / `loss_or_accuracy` | `loss_or_accuracy` |
| `--lambda` | 教師 eval と対局結果 (WDL) のブレンド比 ([§6.2](#62-教師ターゲット-lambda) 参照) | 1.0 (= 純 eval) |
| `--scale` | sigmoid-MSE target で使う eval-to-score sigmoid scale | 290 |
| `--win-rate-model` | WRM (win-rate-model) の target 変換と loss を使う ([§6.2](#wrm-win-rate-model-loss) 参照)。KPPT / KPP_KKPT / NNUE / SFNN などの scalar value network でデフォルト有効 | on |
| `--loss-sigmoid-mse` | WRM ではなく `sigmoid(model_output)` MSE loss を使う | off |
| `--loss-pow-exp` | WRM の誤差項 `|prediction - target|^p` の指数。デフォルト WRM loss で使われる | 2.0 |
| `--wrm-nnue2score` | WRM で network output を score 空間へ戻す倍率。`prediction = wrm(model_output × wrm_nnue2score)` の `wrm_nnue2score` | 600 |
| `--wrm-target-calibration-positions` | 教師データ先頭 N 局面の `(teacher_score, game_result)` から、WRM target 側の score 変換係数を推定する。`0` なら固定値 (`offset=270`, `out_scaling=380`) を使う | 100000 |
| `--wrm-target-offset` / `--wrm-target-scaling` | WRM target 側の固定係数を明示指定する。両方指定すると自動推定を使わない | 省略 |
| `--sfnn-factorizer` | SFNN の residual factorizer を選ぶ。`shared` は stack 共有 factorizer、`none` は全factorizer無効、`axis` は共有項に加えて、その arch に存在する bucket axis factorizer をすべて有効化する。たとえば `hand1024_k3k3` なら `king=axis,hand=axis` 相当、`k3k3` だけなら `king=axis` 相当。`king=axis,hand=shared` のような混合指定も可能 | `shared` |
| `--sfnn-factorized` / `--no-sfnn-factorized` | 短縮alias。基本形は `--sfnn-factorizer shared` / `--sfnn-factorizer none` | on |
| `--optimizer-weight-decay` | 選択中 optimizer の weight decay | 0.0 |
| `--optimizer-epsilon` | 選択中 optimizer の epsilon を上書き。省略時は optimizer 固有の既定値 | 省略 |
| `--optimizer-beta1` | 選択中 optimizer の beta1 を上書き。省略時は optimizer 固有の既定値 | 省略 |
| `--optimizer-beta2` | 選択中 optimizer の beta2 を上書き。省略時は optimizer 固有の既定値 | 省略 |

実行例 (1 億局面 × 40 superbatch = 計 40 億局面):

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_256x2_32_32 \
    --teacher teachers/ \
    --superbatches 40
```

教師ファイルが 1 superbatch 未満 (≒ 1 億局面未満) しか無い場合は `--positions-per-superbatch 10000000` のように小さくすると、何回も save が走るようになる。実効値は `floor(positions / batch_size) * batch_size` に丸められる。

### 学習率の動き

デフォルトの `step` は、1 epoch 内で階段状に LR を下げる StepLR 系です。`--lr-step-positions` を省略すると、1 superbatch ごとに `lr *= gamma` し、`--lr-min` を下限としてそれ以上は下がりません。epoch 境界では `--lr` に戻ります。

`--lr-step-gamma` を省略し、かつ `--superbatches` を指定した場合は、1 epoch 内で `--lr` から `--lr-min` へ到達する `gamma` を自動計算します。たとえば `--superbatches 15` で `--lr-step-positions` を省略すると、15 step で `lr_min` に届く値になります。`--lr-step-positions` を明示した場合は、1 epoch の局面数をその間隔で割った step 数から計算します。epoch 長が決まらない場合は `gamma=0.992` を使います。

`geometric` と `cos` を明示した場合は、1 epoch をかけて `--lr` (lr_max) から `--lr-min` (lr_min) へ滑らかに減衰、epoch 境界で warm restart して lr_max に戻る形になります。違いは曲線の形だけ:

| schedule | 式 | 形 |
|---|---|---|
| `geometric` | `lr(t) = lr_max × (lr_min/lr_max)^t` | 対数線形 — batch ごとに一定倍率で下がる |
| `cos` | `lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(πt))` | 先頭と末尾は緩やか、中盤で最も急 |

`t = (累積局面 mod period) / period`、`period = 1 epoch ぶんの局面数` (= 自動算出)。

**`geometric` / `cos` の period 決定ルール**:

| 状況 | period |
|---|---|
| `--superbatches N` 指定あり | `N × sb_size` (= 1 epoch ぶん。`geometric` / `cos` では推奨) |
| `--superbatches` 未指定 AND HCPE / PSV 教師 | 教師全体の局面数 (file size から自動計算) |
| `--superbatches` 未指定 AND HCPE3 / pack 教師 | エラー (= 可変長レコードなので教師サイズ不明、明示が必要) |

`--superbatches N` を指定した場合、**epoch は教師1周ではなく validation / LR 制御の周期**になる。教師が epoch の途中で EOF した場合は、同じ epoch のまま教師先頭へ戻って N superbatch まで継続する。逆に、epoch 末になっても教師は先頭へ戻さない。次 epoch は、前 epoch の最後に読んだ教師位置の続きから始まる。つまり教師は cyclic stream として流れ続ける。LR は `step` / `geometric` / `cos` では epoch 境界で `--lr` に戻る。

たとえば `--superbatches 4 --lr 0.001 --lr-min 0.00001` (1 epoch = 4 sb ≒ 400M 局面) の lr 推移:

| cycle 内位置 | t | geometric | cos (cosine) |
|---|---|---|---|
| 0M (sb 1 頭) | 0.0 | 0.001 | 0.001 |
| 100M (sb 2 頭) | 0.25 | 0.000316 | 0.000856 |
| 200M (sb 3 頭) | 0.5 | 0.000100 | 0.000505 (midpoint) |
| 300M (sb 4 頭) | 0.75 | 0.0000316 | 0.000155 |
| 400M (sb 4 末) | 1.0 | 0.00001 | 0.00001 |
| 次 epoch sb 1 頭 | 0.0 | **0.001** ← warm restart | **0.001** ← warm restart |

`geometric` は **対数線形**: 各 batch で `(lr_min/lr_max)^(1/batches_per_epoch)` ≒ `0.99987` 倍ずつ下がる、超滑らかな指数減衰です。

⚠️ **`--lr-min` は `step` / `geometric` では必ず `> 0`**: `geometric` は式 `lr_max × (lr_min/lr_max)^t` が `lr_min = 0` だと破綻し、`step` では decay の下限として `lr_min` を使うため、どちらも CLI 起動時に正値を要求します。`1e-5` 〜 `1e-6` あたりが典型。cos は 0 でも数学的に動きますが、警告は出ます。

実際の lr 推移は `<NNNN>/learn.log` の `lr_start` / `lr_end` 列で確認できる ([§7.2 学習ログの読み方](7-result.md#72-学習ログ-learnlog-の読み方))。**bullet stdout の `LR dropped to X` は sb 開始時のみ表示** されるので、batch ごとの変化を見たいときは per-dir log を見てください。

#### tatara / bullet-shogi / nnue-pytorch の StepLR 条件

`--lr-schedule step` は、指定局面数ごとに `lr *= gamma` する階段状のスケジューラです。滑らかな指数減衰を使いたい場合は `geometric` を指定します。`step` は 1 epoch ごとに `--lr` へ戻ります。

tatara / bullet-shogi と同じ固定 `gamma=0.992` 条件を明示するなら、次のように書けます:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --lr 0.000875 \
    --lr-schedule step \
    --lr-step-gamma 0.992 \
    --lr-min 0.00001 \
    --tag step-ablation
```

`--lr-step-positions` を省略すると 1 superbatch ごとに LR を落とします。これは tatara の `lr_step=1` および `bullet-shogi` の `StepLR { gamma=0.992, step=1 }` に対応します。局面数で固定したい比較実験では `--lr-step-positions 100000000` のように明示できます。

一方で「1 epoch の中で `--lr` から `--lr-min` まで落としたい」のように、epoch 長から `gamma` を決めたい場合は `--lr-step-gamma` を書かないでください:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --positions-per-superbatch 40000000 \
    --superbatches 15 \
    --max-epochs 3 \
    --lr 0.000875 \
    --lr-schedule step \
    --lr-min 0.00001 \
    --tag step-auto-gamma
```

この例では、`--lr-step-positions` を省略しているので 1 superbatch ごとに減衰し、各 epoch で `gamma = (lr_min / lr)^(1 / 15)` 相当が内部で使われます。epoch 2、epoch 3 の開始時には LR が再び `--lr` に戻ります。

#### ReduceLROnPlateau を使う

教師データ量が限られていて、epoch 長に合わせて `cos` を1周期回すよりも、validation 指標を見ながら LR を下げたい場合は `--lr-schedule plateau` を使う。

`plateau` は各 superbatch の保存後に `--test-teacher` の `test_value_loss` / `test_value_accuracy` を計測し、`--lr-plateau-monitor` で指定した指標が改善していれば採用する。改善していなければ LR に `--lr-plateau-factor` を掛け、同じ superbatch の教師区間を下げた LR でもう一度学習する。このとき、棄却した更新は採用せず、model weight と optimizer state の両方をその superbatch 開始前に戻す。教師データが尽きた場合は同じ epoch のまま教師の先頭に戻り、`lr_min` に到達するまで継続する。次の LR が `--lr-min` を下回る段階になったら、最後に **ちょうど `--lr-min`** で同じ教師区間を1 superbatchだけ学習する。この最後の試行も監視指標が改善した場合だけ採用し、改善しなければ破棄して、その epoch を終了する。次の epoch は、また `--lr` から plateau 判定を開始する。

`--lr-plateau-monitor` は次の3種類:

| 値 | 採用条件 |
|---|---|
| `loss` | `test_value_loss` が下がった場合だけ採用する。loss 監視型の ReduceLROnPlateau |
| `accuracy` | `test_value_accuracy` が上がった場合だけ採用する |
| `loss_or_accuracy` | loss が下がるか、accuracy が上がった場合に採用する。デフォルト |

`--lr-plateau-min-delta` は loss 側だけに効く。accuracy 側は「厳密に上がったか」だけを見る。

`plateau` で `--max-epochs` を省略すると、epoch 数の固定上限は置かない。各 epoch の最後に `summary-learn.log` へ残った最終 validation 指標を前 epoch の最終指標と比較し、`test_value_loss` が下がらず、かつ `test_value_accuracy` も上がっていなければそこで学習を停止する。`--lr-plateau-monitor` と `--lr-plateau-min-delta` は epoch 内の superbatch 採用判定だけに効き、epoch 間の停止判定は常に tolerance なしの loss-or-accuracy 改善で見る。

`plateau` では 1 epoch の superbatch 数は固定しない。教師ファイルを読み切っても epoch は終わらず、教師先頭へ戻って続行する。`--superbatches` は「この数を超えたら打ち切る」という安全上限であり、通常は指定しない。superbatch の大きさだけを `--positions-per-superbatch` で決める。

監視指標が改善しなかった attempt は正式な checkpoint (`000N/`) には残さない。最後の `lr_min` run も例外ではなく、改善しなければ破棄される。checkpoint と `summary-learn.log` に残るのは、採用された更新だけ。

checkpoint がある `--tag` で `--superbatches` を付けたり外したりすると、BulletOu は設定変更として扱い、auto resume を拒否する。その checkpoint を意図して引き継ぐ場合だけ `--resume` を付ける。別実験として始めたい場合は `--tag` / `--output` を変える。

制約:

- `--test-teacher` 必須。validation 指標がないと判定できない。
- `--save-rate 1` 必須。1 superbatchごとに検証してLRを更新するため。
- `--validation-rate 1` も必須。1 superbatch ごとに validation 指標が必要なため。
- 現状は NNUE/SFNN 系の学習で使用する。

例:

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch NNUE_halfkp_256x2_32_32 \
    --lr 0.001 --lr-min 0.00001 \
    --lr-schedule plateau \
    --lr-plateau-factor 0.5 \
    --lr-plateau-monitor loss_or_accuracy
```

factor を緩めたい場合は `--lr-plateau-factor 0.7` のようにする。`--lr-plateau-min-delta 0.000001` のように指定すると、epoch 内の superbatch 判定でそれ未満の微小な loss 改善は「改善なし」とみなす。epoch 間の停止判定は `--lr-plateau-monitor` ではなく、常に `test_value_loss` が下がるか `test_value_accuracy` が上がるかで見る。

#### `geometric` vs `cos` を比較したい

同じ教師・同じ arch で 2 回 run して `summary-learn.log` を並べると、どちらが効くかすぐ分かる。両方とも同じ `--lr-min` 値を共有できるので apples-to-apples 比較しやすい:

```bash
# geometric decay
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch NNUE_kp_256x2_32_32 \
    --max-epochs 10 --superbatches 4 --tag 5G-geometric \
    --lr-schedule geometric --lr-min 0.00001

# cosine (epoch ごとに 1 cycle)
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch NNUE_kp_256x2_32_32 \
    --max-epochs 10 --tag 5G-cos --superbatches 4 \
    --lr-schedule cos --lr-min 0.00001
```

出力先がそれぞれ `checkpoints/NNUE_KP-NNUE_kp_256x2_32_32-5G-geometric/` と `-5G-cos/` に分かれる。`summary-learn.log` の `test_value_accuracy` / `test_value_loss` 列を pandas / Excel で重ねれば比較完了。

### 教師を数えて `--superbatches` を決める

`geometric` / `cos` で cycle を 1 epoch ぴったりに揃えるには、まず **教師の総局面数** を知る必要がある。BulletOu には専用フラグ `--count-teacher` があり、`std::fs::metadata` でファイルサイズを読むだけなので **数百 GB の教師でも一瞬で完了** する (中身は read しない):

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

教師末尾の余り 61M は捨てない。1 epoch 目は先頭から 400M を読み、2 epoch 目は残り 61M の続きから始まり、EOF したら先頭へ戻って残りを読む。`--superbatches` は「どこで LR cycle / validation epoch を切るか」を決めるもので、「教師を毎 epoch 先頭から読み直す」指定ではない。

#### 対応フォーマット

| フォーマット | レコードサイズ | `--count-teacher` |
|---|---|---|
| HCPE | 38 byte 固定 | ✅ 即計算 |
| PSV  | 40 byte 固定 | ✅ 即計算 |
| HCPE3 | 可変長 (棋譜単位) | ❌ 未対応 (全 game header を walk する必要あり) |
| pack | 可変長 (棋譜単位) | ❌ 同上 |

`geometric` / `cos` で HCPE3 / pack を使うなら、同じ corpus を HCPE / PSV に事前変換するか、`--superbatches` を手動指定してください。`plateau` は period を使わないので、この制約はない。

### 複数 epoch 回す

`--max-epochs N` を指定すると epoch を最大 N 回実行する。`step` / `geometric` / `cos` では「LR cycle を N 回繰り返す」の意味になる。`plateau` では教師を複数周しながら `lr_min` 到達まで同じ epoch を続けるので、「plateau 学習を最大 N 回繰り返す」の意味になる。省略した場合は、どの schedule でも epoch 数の固定上限は置かない。`--test-teacher` があれば前 epoch より最終 validation 指標が改善しなくなるまで続ける。`--test-teacher` がなければ、非 plateau schedule は中断されるまで epoch を繰り返す。

各 epoch 開始時に superbatch 表示と LR cycle はリセットされるが、`--superbatches` 指定時の教師位置はリセットされない。教師は cyclic stream として継続し、EOF した時点でだけ先頭へ戻る。`--superbatches` 未指定の非 plateau では「教師EOF = epoch終了」として扱われ、次 epoch は教師先頭から始まる。

各 epoch ごとに lr が再下降するので、長時間学習で局所最適から脱出させたいときに使う。`cos` schedule で `--superbatches N` を指定すれば cycle = epoch で自動的に揃う (= 典型的な SGDR-style 用法)。`--test-teacher` が指定されている場合、どの schedule でも epoch 末の validation 指標を前 epoch 末と比較し、`test_value_loss` が下がらず、かつ `test_value_accuracy` も上がらなければ、`--max-epochs` に到達していなくてもそこで停止する。`plateau` では `--superbatches` で cycle を揃える必要はない。

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

### WRM (win-rate-model) loss

BulletOu は、KPPT / KPP_KKPT / NNUE / SFNN などの scalar value network で WRM 形式の target 変換と loss をデフォルトで使う。

`sigmoid(model_output)` に対する MSE で比較したい場合は、`--loss-sigmoid-mse` を指定する。

起動時に、教師データ先頭 `--wrm-target-calibration-positions` 局面（デフォルト 100,000）を使い、教師 score と実際の game_result の対応から WRM target 側の `offset` / `scaling` を推定する。推定した target 変換は、学習 target と `test_value_loss` の両方で使われる。

固定 target 変換（`offset=270`, `out_scaling=380`）を使いたい場合は `--wrm-target-calibration-positions 0` を指定する。値を明示したい実験では、`--wrm-target-offset` と `--wrm-target-scaling` を両方指定できる。

変わるものは次の通り。

- 教師 eval を、推定済みの target 側 WRM 係数による win-rate target に変換する
- network output を `nnue2score=600`, `offset=270`, `in_scaling=340` の win-rate prediction に変換する
- loss を `abs(target - prediction)^p` にする。この `p` が `--loss-pow-exp`
- `test_value_loss` と `plateau` 判定も同じ WRM loss に切り替える

`--loss-pow-exp` と `--wrm-nnue2score` は tatara と同じ意味のオプション。`--loss-pow-exp` のデフォルトは `2.0`、つまり WRM 空間での二乗誤差。報告のあった nnue-pytorch 風の設定を試すなら `2.5` を指定する。`--wrm-nnue2score` のデフォルトは `600`。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-wrm-test \
    --loss-pow-exp 2.5 \
    --wrm-nnue2score 600
```

`--loss-pow-exp` と `--wrm-nnue2score` は WRM loss に効く。WRM run の `test_value_loss` は sigmoid-MSE とは式が違うので、loss 定義が違う run の数値とそのまま横比較しない。同じ WRM 設定同士で比較するか、accuracy / 実戦棋力で見る。

### Optimizer の選択

`--optimizer` は `ranger` を受け付ける。デフォルトは `bullet-shogi` の将棋用 example に合わせて `ranger`。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger \
    --optimizer ranger
```

`ranger` は BulletOu の RAdam+Lookahead 実装で、nodchip 版 nnue-pytorch の Ranger21 と同一実装ではない。Ranger21との差を調べるための ablation として使う。nnue-pytorch 条件に寄せるなら、まず次のようにする。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger-decay0 \
    --optimizer ranger \
    --optimizer-weight-decay 0.0 \
    --optimizer-beta1 0.9 \
    --optimizer-beta2 0.999 \
    --optimizer-epsilon 0.0000001
```

`--optimizer-beta1` / `--optimizer-beta2` / `--optimizer-epsilon` を省略した場合は、選択した optimizer の既定値を使う。特に `ranger` の beta1 既定値は `bullet-shogi` と同じ `0.99` で、AdamW の `0.9` ではない。

### Optimizer weight decay

BulletOu の標準設定は、tatara の SFNN-1536 reference run に寄せて `--optimizer-weight-decay 0.0` にしている。weight decay の有無を比較したい場合は、optimizer を同じまま `--optimizer-weight-decay 0.01` などを明示して試す。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger-decay001 \
    --optimizer-weight-decay 0.01
```

これは loss 定義を変えないので、`test_value_loss` は同じ loss 設定の run と直接比較できる。デフォルトの WRM loss とは独立した実験として、単独で ON/OFF 比較する。

### Optimizer epsilon

BulletOu の optimizer epsilon は省略時に選択中 optimizer の既定値を使う。nodchip 版 nnue-pytorch の Ranger21 は `eps=1e-7` なので、epsilon だけ寄せる場合は `--optimizer-epsilon 0.0000001` を使う。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-ranger-eps1e-7 \
    --optimizer-epsilon 0.0000001
```

これも optimizer 条件の差分調査用フラグなので、まず単独で比較する。

### Optimizer beta

optimizer の `beta1` / `beta2` も CLI から変更できる。省略時は `ranger` の既定値 (`beta1=0.99`, `beta2=0.999`) を使う。

optimizer の momentum 条件だけを動かして切り分けたい場合は、`--optimizer-beta1` / `--optimizer-beta2` を指定する。

```bash
./target/release/examples/bulletou \
    --teacher teachers/ --test-teacher test.hcpe \
    --arch SFNN_halfka2_1536_15_32_k3k3 \
    --tag sfnn-optimizer-beta-test \
    --optimizer-beta1 0.85 \
    --optimizer-beta2 0.995
```

これは内部時定数だけを見る ablation。weight decay や epsilon と混ぜず、まず単独で比較する。

```bash
# elmo 式の 50:50 ブレンドで KPPT 学習
./target/release/examples/bulletou \
    --arch KPPT \
    --teacher teachers/ \
    --lambda 0.5
```

(`WDL` は Win/Draw/Loss の略。)

---

次へ: [7. 結果を確認](7-result.md) — 学習結果の確認、ログの読み方

前へ: [4. 学習を走らせる](4-train.md)
