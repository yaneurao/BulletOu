# SPSA 風ローカルチューニング runner

## 目的

SFNN LayerStack の factorizer まわりは、`shared/axis/pair` の強さ、count-aware confidence、saturation penalty などのつまみが多い。これを人間が 1 つずつ試すと時間がかかりすぎる。

そこで、既存 checkpoint から短い追加学習を 2 本だけ分岐させ、量子化後 loss (`quantized_value_loss`) が良いほうを採用する runner を用意する。

この runner は Optuna のように大きく全域探索するものではない。いま良さそうな checkpoint の近くで、少しずつパラメーターを動かしながら qloss が下がる方向を探すためのもの。

## 基本ルール

1 iteration は次の流れで進む。

1. 現在の採用 checkpoint を基準にする。
2. 調整対象パラメーターにランダムな `+/-` 方向を作る。
3. `plus` trial と `minus` trial を、同じ checkpoint から同じ局面位置で開始する。
4. 各 trial は短い superbatch 数だけ学習し、最後に qloss を測る。
5. qloss が下がった trial があれば、その checkpoint とパラメーターを採用する。
6. 両方悪化した場合は、変化幅を小さくして retry する。
7. retry が続きすぎた場合は、悪化が小さいほうを採用して先へ進める。

デフォルトの評価指標は `quantized_value_loss`。`quantized_value_accuracy` は目的関数として粗すぎるので、採否判定には使わない。

## ファイル

- runner: `spsa_local_runner.py`
- 実行設定: `config.json`
- 現在状態: `state.json`
- 採否履歴: `history.csv`
- 採用 checkpoint だけの要約: `accepted-summary-learn.log`
- 各 trial の標準出力: `logs/*.stdout.log`
- 一時 trial 出力: `trials/`
- 採用経路の現在状態: `current/`
- 外向けに残す採用 checkpoint: `accepted-checkpoints/`

runner は既存の checkpoint を上書きしない。デフォルトでは plus/minus の trial フォルダは採否判定後に削除する。採用された state は `current/` に移動し、採用経路が `--accepted-save-rate-sbs` ぶん進んだときだけ `accepted-checkpoints/` にコピーする。

デフォルトは次の通り。

```text
--epoch-sbs 32
--accepted-save-rate-sbs 32
```

つまり、採用経路が32 sb進むたびに `accepted-checkpoints/0001`, `0002`, ... ができる。8 sb trialなら4 iterationごとに1回保存される。trial フォルダを調査用に残したい場合だけ `--keep-trials` を指定する。

## 今の実験から継続する例

`--base-checkpoint` は、継続元にしたい checkpoint フォルダを明示する。保存時刻では選ばない。

例では `0256` から始めるが、実際には使いたい checkpoint 番号に置き換える。

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
  --accepted-save-rate-sbs 32 `
  --positions-per-superbatch 40000000 `
  --metric quantized_value_loss `
  --theta "shared=1.0,axis=1.0,pair=0.3,residual_count=1.0,axis_count=1.0,pair_count=10.0,king_axis_count=4.0" `
  --tune axis `
  --tune pair `
  --tune count `
  --fixed shared `
  -- --lr 0.000100 --lr-min 0.000100 --wrm-in-offset 0 --wrm-target-offset 0 --lr-schedule step --optimizer ranger --optimizer-weight-decay 0.0 --batches-per-update 1 --sfnn-dirty-bucket-update --sfnn-saturation-penalty 1e-7
```

`--` より後ろは、そのまま `bulletou.exe` に渡す。runner 側が自動で指定するので、ここには `--resume`、`--superbatches`、`--max-epochs`、`--save-rate`、`--validation-rate`、`--quantized-validation-rate` は書かない。

## 初期パラメーターの書き方

`--theta` はカンマ区切りで書ける。

```text
shared=1.0,axis=1.0,pair=0.3,residual_count=1.0,axis_count=1.0,pair_count=10.0,king_axis_count=4.0
```

この例の意味:

- `shared=1.0`: shared factorizer の係数
- `axis=1.0`: king/hand/progress axis の係数
- `pair=0.3`: king-hand / king-progress / hand-progress pair の係数
- `residual_count=1.0`: stack residual の count confidence
- `axis_count=1.0`: king/hand/progress axis の count confidence
- `pair_count=10.0`: 3 種類の pair の count confidence
- `king_axis_count=4.0`: king axis だけを上書き

## 調整対象

`--tune` は複数指定できる。

よく使う指定:

- `--tune axis`
- `--tune pair`
- `--tune count`
- `--tune axis_count`
- `--tune pair_count`

`--fixed shared` を付けると、shared は固定される。shared は全体の土台になりやすいので、最初の実験では固定するほうが結果を読みやすい。

## qloss が悪化したときの扱い

両方の trial が悪化した場合、runner はすぐには採用せず、変化幅を小さくして retry する。

`--max-retries 5` の場合、5 回連続で改善しないときだけ、悪化が小さいほうを採用して進める。これは「どの方向でも短期 qloss は下がらないが、局所的に動かないと先へ進めない」状態を避けるため。

## 注意点

- 1 iteration で `plus` と `minus` の 2 本を走らせるので、`--sb-per-trial 8` なら 16sb 分の学習時間を使う。
- trial は同じ checkpoint と同じ dataloader position から始まる。
- `--base-checkpoint` の親フォルダにある `summary-learn.log` から基準 qloss を読む。そこに qloss がない場合は `--base-score` で手入力する。
- count confidence を 1 つでも非ゼロにする場合は `--bucket-counts` が必要。
- 採用されなかった trial と、採用後に `current/` へ移動済みの trial は削除される。残したい場合は `--keep-trials` を使う。
