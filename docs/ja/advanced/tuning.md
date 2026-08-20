# 学習設定を調整する

<a href="../../en/advanced/tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[チュートリアル 3: 学習を走らせる](../tutorial/3-train.md) のコマンドが動いたあとに読むページです。

最初の学習では、まずデフォルト値のままで十分です。速度、保存頻度、検証頻度、学習率、loss、SFNN factorizer などを調整したくなったら、このページを見てください。

## 1. ログに出てくる単位

| 名前 | 意味 |
| --- | --- |
| batch | GPUで1回処理する局面数。デフォルトは `--batch-size 65536` |
| superbatch / sb | 進捗表示、検証、保存の単位。大きさは `--positions-per-superbatch` で決まる |
| epoch | `--superbatches` 個の sb をまとめた単位 |
| checkpoint | 再開用の `state.bin` と、エンジン用の `nn.bin` |
| validation / 検証 | `--test-teacher` の局面で accuracy と loss を測ること |

例:

```text
--batch-size 65536
--positions-per-superbatch 40000000
--superbatches 36
```

この場合、1 sb は `65536 x 610 = 39,976,960` 局面です。1 epoch は36 sbなので、約14.4億局面です。

## 2. よく調整するオプション

| 目的 | オプション | 例 |
| --- | --- | --- |
| 1 epoch を何 sb にするか決める | `--superbatches` | `--superbatches 36` |
| 1 sb の局面数を決める | `--positions-per-superbatch` | `--positions-per-superbatch 40000000` |
| 何 epoch 学習するか決める | `--max-epochs` | `--max-epochs 3` |
| checkpoint 保存を減らす | `--save-rate` | `--save-rate 9999` なら epoch 末保存だけにしやすい |
| checkpointを別ドライブに置く | `--output-folder` | `--output-folder D:\checkpoints` |
| accuracy/loss を毎 sb 測る | `--validation-rate` | `--validation-rate 1` |
| StepLR の減衰率を指定する | `--lr-step-gamma` | `--lr-step-gamma 0.992` |
| loss の指数を変える | `--loss-pow-exp` | `--loss-pow-exp 2.5` |
| SFNN の factorizer を変える | `--sfnn-factorizer` | `--sfnn-factorizer none` |
| SFNN の factorizer の効き具合を変える | `--sfnn-factorizer-alpha` | `--sfnn-factorizer-alpha king=0.90` |
| SFNN の量子化飽和を抑える | `--sfnn-saturation-penalty` | `--sfnn-saturation-penalty 1e-7` |

主なオプション一覧:

| フラグ | 何を変えるか | デフォルト |
| --- | --- | --- |
| `--backend` | 学習 backend。通常は `cuda-cpp` | `cuda-cpp` |
| `--output-folder` | checkpointの親フォルダ。自動ディレクトリ名と `--tag` はそのまま使う | `checkpoints` |
| `--output` | checkpoint保存先を完全指定する。`--tag` は使わない | 省略 |
| `--batch-size` | 1回のmini-batchで処理する局面数 | 65536 |
| `--batches-per-update` | N mini-batchぶんの勾配を足してから1回だけoptimizer更新する | 1 |
| `--positions-per-superbatch` | 1 sb の目標局面数。実際には `batch-size` の倍数に丸められる | 100000000 |
| `--teacher-shuffle-buffer-sbs` | 何 sb 分の教師局面をRAM上でshuffleするか | 1 |
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
| `--lr` | 学習率の開始値 | 0.000875 |
| `--lr-min` | 学習率の下限 | 0.00001 |
| `--lr-schedule` | 学習率schedule。まずは `step` でよい | `step` |
| `--lr-step-gamma` | `step` で学習率に掛ける係数 | auto / 0.992 |
| `--lr-step-positions` | 何局面ごとに学習率を下げるか。省略時は1 sbごと | 省略 |
| `--lambda` | 教師評価値と勝敗項を混ぜる比率 | 1.0 |
| `--loss-pow-exp` | `|prediction - target|^p` の `p` | 2.0 |
| `--wrm-nnue2score` | WRM lossで `network_output` をscoreへ戻す係数 | 600 |
| `--wrm-in-offset` / `--wrm-in-scaling` | WRM lossのprediction側カーブ | 270 / 340 |
| `--wrm-target-offset` / `--wrm-target-scaling` | WRM lossのteacher側カーブ | 270 / 380 |
| `--loss-sigmoid-mse` | WRMではなく単純なsigmoid lossを使う | off |
| `--scale` | `--loss-sigmoid-mse` の target scale | 600 |
| `--fv-scale` | `nn.bin` の量子化検証・補正で想定する `FV_SCALE` | 24 |
| `--quantized-validation-rate` | 学習中に量子化後のaccuracy/lossを何sbごとに見るか。デフォルトはGPU近似 | 保存時のみ |
| `--quantized-validation-exact` | 量子化後検証をCPU整数forwardで正確に測る。遅いので必要なときだけ使う | off |
| `--sfnn-factorizer` | SFNNのbucket間で共通成分を共有する方法 | `shared` |
| `--sfnn-factorizer-alpha` | factorizer成分をどれだけ効かせるか | 1.0 |
| `--sfnn-saturation-penalty` | fold後のL1/L2/L3重みがi8の端に張り付くのを抑える追加ペナルティ | 0.0 |
| `--sfnn-saturation-threshold` | 飽和ペナルティをかけ始めるi8量子化値 | 127.0 |
| `--optimizer` | optimizer | `ranger` |
| `--optimizer-weight-decay` | weight decay | 0.0 |
| `--optimizer-epsilon` / `--optimizer-beta1` / `--optimizer-beta2` | optimizerの詳細パラメータ | 省略 |

## 3. 学習率schedule

`--lr-schedule step` では、一定間隔で次のように学習率を下げます。

```text
lr = lr * gamma
```

`--lr-step-positions` を省略すると、1 sbごとに学習率が下がります。`--lr-step-gamma` を省略して `--superbatches` を指定すると、BulletOu は epoch 内で `--lr` から `--lr-min` へ近づくように gamma を計算します。

明示的に `gamma=0.992` を指定する例:

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

## 4. 勾配accumulation

VRAMの都合で `--batch-size` を小さくする必要があるが、optimizerには大きなbatch相当の勾配を渡したい場合は `--batches-per-update N` を使います。

例:

```bash
--batch-size 16384
--batches-per-update 4
```

この指定では、16,384局面のmini-batchを4回流し、その4回分の勾配を足してから、Rangerの更新を1回だけ行います。optimizerから見ると、仮想batch sizeは次のようになります。

```text
16384 x 4 = 65536 局面
```

これはCUDAのforward/backward自体を65,536局面で1回だけ実行するのと完全に同じ速度にはなりません。ただし、optimizer updateの回数を減らし、勾配のノイズを小さくできます。

`--positions-per-superbatch` は `--batch-size` の倍数に丸められます。たとえば、

```text
--positions-per-superbatch 40000000
--batch-size 65536
```

`--batches-per-update 1` なら、実際には `610 * 65,536 = 39,976,960` 局面を1 sbとして扱います。

`--batches-per-update` が2以上の場合は、さらにmini-batch数を `--batches-per-update` の倍数へ丸めます。たとえば `--batches-per-update 4` なら、610 batchではなく608 batchを1 sbとして扱います。

```text
608 * 65,536 = 39,845,888 局面
608 / 4 = 152 optimizer updates
```

つまり、ユーザーは `--positions-per-superbatch 40000000` のような丸い値を指定すればよく、`39,845,888` のような端数を手で計算して指定する必要はありません。

## 5. loss

デフォルトは WRM loss です。明示しなくても使われます。

```text
score_net  = network_output * wrm_nnue2score
prediction = wrm(score_net;     wrm_in_offset,     wrm_in_scaling)
target     = wrm(teacher_score; wrm_target_offset, wrm_target_scaling)
loss       = |prediction - target|^loss_pow_exp
```

デフォルト値:

```bash
--wrm-nnue2score 600
--wrm-in-offset 270
--wrm-in-scaling 340
--wrm-target-offset 270
--wrm-target-scaling 380
--loss-pow-exp 2.0
```

offsetなしWRMと比較したい場合:

```bash
--wrm-in-offset 0
--wrm-target-offset 0
```

単純なsigmoid lossを使う場合:

```bash
--loss-sigmoid-mse
--scale 600
--fv-scale 24
```

lossの式と `FV_SCALE` の関係は [loss の scale と `FV_SCALE`](scale-and-fv-scale.md) を参照してください。

## 6. SFNN factorizer

SFNNでは、`k3k3` や `hand1024` のようにbucketを増やせます。bucketが多いほど表現力は上がりますが、bucketあたりの教師局面は減ります。

factorizerは、bucket間で共通成分を共有する仕組みです。教師密度が足りないときの過学習を抑えたり、学習を安定させたりする目的で使います。

hand axis や `king-hand` / `hand-progress` のような2軸factorizerの詳しい式は [SFNN factorizer](sfnn-factorizer.md) を参照してください。このページでは、学習コマンドでよく使う指定だけをまとめます。

学習中の有効な重みは、概念的には次の形になります。

```text
W_effective = W_base + W_shared + W_axis + W_pair
```

`W_base` は各bucketが個別に持つ重みです。`W_shared` は全bucketで共有する成分、`W_axis` は king bucket、hand bucket、progress bucket のような単独軸で共有する成分です。`W_pair` は `king-hand`、`king-progress`、`hand-progress` のような2軸の組み合わせで共有する成分です。

| 指定 | 意味 |
| --- | --- |
| `--sfnn-factorizer shared` | bucket全体で共通成分を持つ。デフォルト |
| `--sfnn-factorizer none` | factorizerを使わない |
| `--sfnn-factorizer axis` | archに存在する軸をまとめて有効化する |
| `--sfnn-factorizer pair` | `axis` に加えて、使える2軸factorizerをまとめて有効化する |
| `--sfnn-factorizer king=axis,hand=axis,progress=axis` | 軸ごとに明示する |
| `--sfnn-factorizer king-hand,king-progress,hand-progress` | 2軸factorizerを個別に指定する |
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

factorizerの効き具合を変えたい場合は `--sfnn-factorizer-alpha` を使います。

```text
W_effective = W_base
            + alpha_shared * W_shared
            + alpha_king   * W_king_axis
            + alpha_hand   * W_hand_axis
            + alpha_progress * W_progress_axis
            + alpha_pair   * W_pair
```

たとえば、king bucket の axis factorizer だけを90%の強さで使う場合:

```bash
--sfnn-factorizer king=axis
--sfnn-factorizer-alpha king=0.90
```

king と hand を別々に弱める場合:

```bash
--sfnn-factorizer king=axis,hand=axis
--sfnn-factorizer-alpha king=0.90,hand=0.80
```

axis成分全体と2軸成分全体を強める場合:

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha axis=4.0,pair=4.0
```

progress axis だけを強める場合:

```bash
--sfnn-factorizer progress=axis
--sfnn-factorizer-alpha progress=4.0
```

全factorizer成分を同じ強さにする場合:

```bash
--sfnn-factorizer axis
--sfnn-factorizer-alpha 0.90
```

同じ意味を、明示的に `all=` で書くこともできます。実験メモとして残すなら、こちらのほうが意図が読み取りやすいです。

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha all=3.0
```

`all=` のあとに個別指定を書くと、その成分だけ上書きされます。

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha all=3.0,pair=4.0
```

この例では `shared` と `axis` は3.0、`pair` は4.0になります。

`hand1024` と `progress8` のように複数のbucket軸を組み合わせる場合は、2軸factorizerも試せます。

```bash
--arch SFNN_halfka2_1024_7_64_k3k3_hand1024_progress8
--sfnn-factorizer pair
```

この指定は、利用できる範囲で `shared`、`king-axis`、`hand-axis`、`progress-axis`、`king-hand`、`king-progress`、`hand-progress` を有効化します。archに存在しない軸は自動的に無視されます。

`alpha=1.0` が通常の状態です。`alpha=0.0` にすると、そのfactorizer成分はforwardに足されず、その成分への勾配も0になります。これは「保存済みのfactorizer tensorをbase weightへ畳み込む」操作ではありません。factorizerを完全に外した状態で追加学習したい場合は `--sfnn-factorizer none` を使います。

`alpha` は `1.0` より大きくすることもできます。指定可能範囲は `0.0` から `10.0` です。たとえば `king=2.0` は king-axis 成分を forward で2倍して足し、同時に king-axis tensor への勾配も2倍します。大きすぎる値は学習を不安定にする可能性があるため、実験用途として扱ってください。

`nn.bin` を書き出すときは、上の `W_effective` の形に畳み込まれます。そのため `--sfnn-factorizer-alpha king=0.90` で保存した `nn.bin` には、king axis成分が90%で反映されます。

### 6.1 量子化飽和を抑える

SFNNのL1/L2/L3重みは、`nn.bin` に保存するときにi8へ量子化されます。factorizerを強くしたりbucket数を増やしたりすると、fold後の有効重みがi8の上限付近に張り付き、量子化後のlossやaccuracyが悪くなることがあります。

その場合は、実験用に飽和ペナルティを足せます。

```bash
--sfnn-saturation-penalty 1e-7
```

このペナルティはデフォルトでは無効です。指定したときだけ、更新直前の勾配に次の追加項を足します。

```text
q = W_effective * QB
penalty_loss_per_weight = lambda * max(0, |q| - threshold)^2
```

`QB` はL1/L2/L3重みの量子化倍率で、通常は64です。`threshold=127` なら、i8の端に到達する重みだけを抑えます。少し手前から抑えたい場合は、たとえば次のようにします。

```bash
--sfnn-saturation-penalty 1e-7
--sfnn-saturation-threshold 120
```

このペナルティは報告される `test_value_loss` の定義を変えません。lossの表示は通常どおり比較できます。

### 6.2 rare bucket を count で抑える

bucket 数が多い arch では、出現回数の少ない stack の個別成分だけが暴れることがあります。その場合は、教師データから stack ごとの出現回数を数えた `count.bin` を作り、count に応じて base stack residual を弱く正則化できます。

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\sojoteam_datasets `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --positions 500000000 `
  --output D:\BulletOu-snapshots\counts\count.bin
```

`--positions` を省略すると、teacher path 内の全ファイルを1回だけ読んで count します。

学習では次のように指定します。

```powershell
--sfnn-factorizer pair `
--sfnn-factorizer-alpha all=1.0 `
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin `
--sfnn-residual-count-confidence 1.0
```

`--sfnn-residual-count-confidence 1.0` は、「bucket 固有 residual のパラメーター数と同じぐらいの出現回数があるまでは、その bucket 固有成分をまだ強く信用しない」という意味です。この指定で residual count decay が有効になり、最大減衰量はデフォルトで `1e-7` になります。

count decay は factorizer 成分ではなく、bucket 固有の residual にだけかかります。count が多い bucket は residual を大きく学習しやすく、count が少ない bucket は factorizer の共有成分に寄ります。

axis 行・pair 行そのものを count に応じて弱めたい場合は、`--sfnn-axis-count-confidence` と `--sfnn-pair-count-confidence` を使います。特定の種類だけ強さを変えたい場合は、`--sfnn-king-axis-count-confidence` や `--sfnn-hand-progress-pair-count-confidence` のような個別指定を使います。詳しい式と考え方は [SFNN factorizer](sfnn-factorizer.md) を参照してください。

### 6.3 qloss を見ながら自動調整する

factorizer の alpha や count confidence は、組み合わせが多く、手で総当たりすると時間がかかります。既に良い checkpoint がある場合は、`spsa_local_runner.py` で近傍探索できます。

この runner は、同じ checkpoint から2本の短い probe を走らせます。2本は、ハイパーパラメーター空間でランダムに選んだ方向 `v` に対して、片方を `+v`、もう片方を `-v` へ動かしたものです。量子化後 loss (`quantized_value_loss`) を比較し、その差から「この近傍ではどちらへ動かすと qloss が下がりやすいか」を推定します。accuracy は値が荒いので、判断には qloss を使います。

開始時には、継続元 checkpoint の `nn.bin` を使って `bulletou.exe quantized-test --mode gpu` を実行し、基準 qacc/qloss を自分で測り直します。その結果は `[base]` 行として stdout に表示されます。summary から読みたい場合だけ `--base-metric-source summary` を指定します。

両方の trial が基準より悪い場合は、すぐには採用せず変化幅を小さくして retry します。retry 中に出た trial のうち、qloss が一番良かったものだけを `retry-best/` に退避します。`--max-retries` に達しても基準を超えられない場合は、最後の2本ではなく、retry 全体の best qloss を強制採用します。

```text
1 iteration:
  probe A: base checkpoint から 8 sb 学習
  probe B: base checkpoint から 8 sb 学習
  qloss が小さい probe の checkpoint を次の継続元にする
  パラメーターは SPSA の推定方向へ、探索幅の一部だけ動かす
```

`--sb-per-trial 8` は「1本の trial が8 sb」という意味です。1 iteration では2本走るので、GPUで実行する量は16 sb、採用経路として進む量は8 sbです。

retry は同じ iteration のまま実行されます。たとえば iteration 12 の1回目が悪かった場合、次は iteration 13 ではなく iteration 12 / retry 2 になります。`accepted-summary-learn.log` には採用経路が進んだ行だけが出るので、iteration 番号は飛びません。すべての trial を1行ずつ見たい場合は `summary-learn.log` を見ます。

保存も retry の回数ではなく、accept された回数で決まります。runner の `--save-rate 1` なら accept ごとに保存、`--save-rate 4` なら4回 accept するごとに `accepted-checkpoints/` へ保存します。`--save-rate 0` なら accepted checkpoint の自動保存を無効にします。

パラメーターの探索幅は倍率で指定します。デフォルトは `--step-scale 1.03` です。たとえば `pair=0.3` なら、ある probe ではおおむね `0.309`、反対側の probe では `0.291` になります。ただし、デフォルトの `--update-mode spsa` では、良かった probe の値へそのままジャンプしません。`--spsa-move-ratio 0.1` により、探索幅の10%だけ動きます。つまりこの例では `0.309` に飛ぶのではなく、だいたい `0.3009` 付近へ動きます。両方悪化した場合は `--step-shrink 0.70` で探索幅を縮め、改善した場合は `--step-grow 1.01` で少しだけ広げます。探索幅の範囲はデフォルトで `--min-step-scale 1.005` から `--max-step-scale 1.10` です。

qloss が小さかった probe のパラメーター値へそのまま移動する動作を試したい場合だけ、`--update-mode winner` を指定します。

デフォルトでは、probe の trial フォルダは採否判定後に削除されます。採用された state は runner 内部の `current/` に移動され、採用経路が32 sb進むたびに `accepted-checkpoints/0001`, `0002`, ... として外向け checkpoint が保存されます。

trial 内の checkpoint 保存は trial 末だけです。runner は BulletOu本体へ、trial中に到達しない大きな `--save-rate` を渡し、デフォルト有効の epoch末保存でtrial末checkpointだけを作ります。通常の validation と量子化後 validation はデフォルトで毎 sb 実行されます。そのため stdout には各 sb の `test_value_loss` と `quantized_value_loss` が表示されます。変えたい場合は `--trial-validation-rate-sbs` と `--trial-quantized-validation-rate-sbs` を使います。

`--use-worker` を付けると、runner は `bulletou.exe worker` を1回だけ起動し、GPU上に学習sessionを開いたまま trial を実行します。probe の測定では、worker がGPU上の重みとoptimizer stateをsnapshot/restoreするため、trialごとのprocess起動、CUDA warmup、checkpoint保存を避けられます。採用するときだけ、qloss が良かった probe と同じ条件で学習を進め、その状態をGPU上に残します。

`--use-worker` の保存も runner の `--save-rate` に従います。たとえば `--sb-per-trial 8` で `--save-rate 4` なら、4回 accept した時点、つまり採用経路が32 sb進むたびに `accepted-checkpoints/0001`, `0002`, ... へ保存します。保存されていない採用状態はworker processのメモリ上にだけあります。途中で止めた場合の再開地点は、最後に保存された accepted checkpoint です。こまかく中断再開したい実験では、runner の `--save-rate` を小さくしてください。

SPSA runner では、trial の qloss 差からハイパーパラメーターの良し悪しを判断します。そのため、trial 側の学習率が高すぎると、ハイパーパラメーターの効果ではなく短期的な学習の振動を拾いやすくなります。本命のチューニングでは、通常の追加学習より小さめの `--lr` / `--lr-min` を使ってください。高い学習率を使う場合は、`--sb-per-trial` を16や32に増やして、8sbだけの判定に頼りすぎないほうが安全です。

例:

```powershell
$base = "C:\shogi\YaneuraOuWorks\BulletOu\checkpoints\SFNN_HALFKA2-SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-sfnn-sojo2tb-32sb-pair2-4.0\0256"

python .\spsa_local_runner.py `
  --exe C:\shogi\YaneuraOuWorks\BulletOu\target\release\examples\bulletou.exe `
  --base-checkpoint $base `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --output-folder D:\BulletOu-snapshots\20260820 `
  --tag-prefix spsa-pair2-qloss `
  --factorizer pair `
  --iterations 20 `
  --sb-per-trial 8 `
  --epoch-sbs 32 `
  --save-rate 4 `
  --positions-per-superbatch 40000000 `
  --metric quantized_value_loss `
  --use-worker `
  --theta "shared=1.0,axis=1.0,pair=0.3,residual_count=1.0,axis_count=1.0,pair_count=10.0,king_axis_count=4.0" `
  --tune axis `
  --tune pair `
  --tune count `
  --fixed shared `
  -- `
  --lr 0.000100 `
  --lr-min 0.000100 `
  --wrm-in-offset 0 `
  --wrm-target-offset 0 `
  --lr-schedule step `
  --optimizer ranger `
  --optimizer-weight-decay 0.0 `
  --batches-per-update 1 `
  --sfnn-dirty-bucket-update `
  --sfnn-saturation-penalty 1e-7
```

`--tune` は「runner が動かすパラメーター」を指定します。

| 指定 | 動くもの |
| --- | --- |
| `--tune alpha` | `shared_alpha`, `king_axis_alpha`, `hand_axis_alpha`, `progress_axis_alpha`, `pair_alpha` |
| `--tune axis` | `king_axis_alpha`, `hand_axis_alpha`, `progress_axis_alpha` |
| `--tune pair` | `pair_alpha` だけ。`king-hand` などの3種類を個別には動かさない |
| `--tune count` | residual count、axis count、pair count の全体 |
| `--tune axis_count` | `king_axis_count`, `hand_axis_count`, `progress_axis_count` |
| `--tune pair_count` | `king_hand_pair_count`, `king_progress_pair_count`, `hand_progress_pair_count` |

shared 以外を全部動かしたい場合は、次のようにします。

```powershell
--tune alpha `
--tune count `
--fixed shared
```

この場合、`--tune alpha` には `shared_alpha` も含まれるので、`--fixed shared` を必ず付けます。

`--` だけの行は区切りです。そこから後ろは runner ではなく `bulletou.exe` へ渡されます。`--lr` や `--optimizer` のような、各 trial で共通に使う学習条件を書きます。

runner が自動で指定するので、`--` より後ろ側には `--resume`、`--superbatches`、`--max-epochs`、`--save-rate`、`--validation-rate`、`--quantized-validation-rate`、`--tag`、`--output-folder`、`--initial-state`、`--initial-dataloader-pos` は書かないでください。SPSA runner 自体の保存頻度を変える `--save-rate` は、`--` より前に書きます。

runner は `bulletou.exe` の stdout をコンソールへそのまま表示し、同時に `logs/*.stdout.log` にも保存します。画面出力を止めてログファイルだけにしたい場合は `--no-stream-child-output` を付けます。

runner root には3種類のCSVログが出ます。

| ファイル | 内容 |
| --- | --- |
| `summary-learn.log` | すべての trial を1行ずつ記録する。採用しなかった trial も残る |
| `accepted-summary-learn.log` | 採用経路だけを記録する。棋力計測候補や停止地点の確認に使う |
| `history.csv` | 1回のA/B判定を1行にまとめた内部寄りの履歴 |

`summary-learn.log` には `result`, `quantized_value_accuracy`, `quantized_value_loss`, `test_value_accuracy`, `test_value_loss`, `saved_checkpoint`, `theta_change`, `theta_json` が出ます。`result` は `accepted`, `forced_accepted`, `retry_best`, `discarded` のような値です。`saved_checkpoint` は `accepted-checkpoints/` に保存されたフォルダ名です。保存境界でない行では空欄になります。`accepted-summary-learn.log` も `quantized_value_accuracy`, `quantized_value_loss`, `test_value_accuracy`, `test_value_loss`, `saved_checkpoint` が先頭側に来る列順です。`reason` や `theta_json` のような長い列は右側に寄せています。見るべきなのは、probe の名前ではなく、qloss が改善したか、どこが保存済みか、そして各ハイパーパラメーターがどれだけ動いたかです。stdout ログをファイルで見る場合は runner root の `logs/*.stdout.log` を見てください。`trials/` の中は削除対象なので、そこにあるファイルをエディタで開いたままにしないでください。trial フォルダを削除せず調査用に残したい場合は `--keep-trials` を付けます。元の checkpoint は上書きされません。

画面上では、節目の行だけ色付きで出ます。`BASE` は開始時の基準、`TARGET` はその iteration での採用閾値です。通常は `accept_if qloss < ...` と表示され、この値より trial の qloss が小さければ採用されます。各 probe の開始時には `TRIAL trial 1 start` / `TRIAL trial 2 start` が出ます。各 probe が終わると `done` 行の `qloss` / `qacc` が色付きで出ます。基準より良ければ緑、悪ければ黄色です。続く `TRIAL` 行には `final_qloss`, `start_qloss`, `delta` が出ます。`final_qloss < start_qloss` なら緑、超えていなければ黄色です。2本の probe のあとに `DECISION` 行が出ます。ここに `best_probe_qloss`, `start_qloss`, `delta`, `decision=accept best probe` または `decision=retry ...` が出ます。`DECISION` が緑なら採用、黄色ならretryまたは強制採用候補です。`--use-worker` では、採用probeをGPU上に残すために `ACCEPT RUN` / `ACCEPT DONE` が出ることがあります。これは新しい探索点ではなく、採用済みprobeをkeepするための内部再実行です。retry上限に達して強制採用する場合は黄色の `FORCE` 行になります。`ACCEPT` は改善した trial の採用、`SAVE` は checkpoint 保存、`SAFE TO STOP` はその時点で停止しても保存済みの地点です。`--use-worker` で `ACCEPT` だけ出て `SAVE` がまだ出ていない場合、その採用状態はGPU上にだけあります。その場合は黄色の `WAIT FOR SAVE` が出るので、停止するなら次の `SAVE` / `SAFE TO STOP` まで待ってください。色を消したい場合は `--color never` を指定します。

中断した runner を再開する場合は、同じ `--output-folder` と `--tag-prefix` を指定して `--resume` を付けます。runner root は `--output-folder\spsa-<tag-prefix>` で決まるので、通常は `--runner-dir` を書く必要はありません。`state.json` に保存された `current/` の checkpoint、theta、step scale、retry状態から再開します。

```powershell
python .\spsa_local_runner.py `
  --resume `
  --exe C:\shogi\YaneuraOuWorks\BulletOu\target\release\examples\bulletou.exe `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --output-folder D:\BulletOu-snapshots\20260820 `
  --tag-prefix spsa-pair2-qloss `
  --factorizer pair `
  --iterations 1000 `
  --sb-per-trial 8 `
  --positions-per-superbatch 40000000 `
  --metric quantized_value_loss `
  --tune alpha `
  --tune count `
  --fixed shared `
  -- --lr 0.000100 --lr-min 0.000100 --wrm-in-offset 0 --wrm-target-offset 0 --lr-schedule step --optimizer ranger --optimizer-weight-decay 0.0 --batches-per-update 1 --sfnn-dirty-bucket-update --sfnn-saturation-penalty 1e-7
```

`--iterations` は「runner全体で何iterationまで進めるか」です。たとえば `state.json` が `next_iteration=37` なら、`--iterations 1000` で37回目から1000回目まで続きます。

## 7. 保存と検証の頻度

保存と検証は別々に指定できます。

```bash
--save-rate 20 --validation-rate 1
```

これは「checkpointは20 sbごとに保存し、accuracy/lossは毎 sb 測る」という意味です。

epoch末保存はデフォルトで有効です。epoch末だけ保存したい場合は、epoch内で到達しない大きな `--save-rate` を指定します。

```bash
--save-rate 9999
```

## 8. 速度を見るログ

速度を見るときは stdout の `[train]` 行を見ます。

```text
[train]  epoch 1  sb 12/36  this-sb=39,976,960 pos (...)  wall=...s  train=...s  pos/s=...
```

| 表示 | 意味 |
| --- | --- |
| `wall` | その sb の実時間。検証や保存も含む |
| `train` | 学習処理そのものの時間。検証・保存は含まない |
| `pos/s` | `train` から計算した学習速度 |

GPUが空いているのに `pos/s` が低い場合は、教師局面の読み込み、decode、shuffle が詰まっている可能性があります。`cuda-cpp-diagnostics.log` の teacher queue wait を見てください。
