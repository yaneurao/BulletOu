# Grid searchで学習条件を比較する

[English](../../en/advanced/grid-search.md)

`grid_search.py` は、YOSC の `trainer/grid_search.py` と同様に、指定した値の**全組み合わせを1本ずつ学習し、比較結果をCSVへ集計する**スクリプトです。Python 3.10以降の標準ライブラリだけで動きます。対象はBulletOuのcuda-cpp production学習です。

TPEの `tuning_parameters.py` とは別です。勝者から次の条件へ追加学習することも、途中で条件を間引くこともありません。新規学習なら各条件が同じ決定的初期化から、追加学習なら全条件が同じ開始checkpointと教師位置から始まります。GPUの浮動小数点演算による再現誤差までなくすものではありません。

## まず実行計画だけ確認する

共通設定には普通の `bulletou-settings.json` を使います。**tuning-settings.jsonではありません。** [設定例](../../examples/grid-search-bulletou-settings.json) はk3k3、FT factorizer有効・L1 shared、1epoch=324sb、1sb=40M指定、81sbごと保存です。パスや条件は自分の環境に合わせてください。このファイルを追加しただけで学習が開始されることはありません。

BulletOuフォルダで実行する例です。

```powershell
python .\grid_search.py `
  --settings-file .\docs\examples\grid-search-bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\20260911\grid-target-scale `
  --lrs 0.000875 0.000400 `
  --wrm-target-scalings 600 1200 `
  --dry-run
```

LR 2通り × 教師側scaling 2通り = **4条件**です。`--dry-run` は実行ファイルの `--help` で対応オプションを確認して計画を表示するだけで、学習もファイル作成も行いません。実行するには同じコマンドから `--dry-run` を外します。学習本体を再ビルドする必要はありません。

値の列は直積であり、対応する位置同士の組ではありません。全組み合わせを列挙してから順番に実行します。たとえば `--lrs` と `--lr-mins` の組み合わせに `lr_min > lr` が含まれると、実行前にエラーにします。勝手に除外したり、数値を修正したりしません。

## 引数

| 引数 | 意味 |
| --- | --- |
| `--settings-file PATH` | 共通のBulletOu設定JSON。集計だけの場合は不要 |
| `--output-folder DIR` | **このgrid専用**の保存先。直下に `grid_summary.csv`、`trials/` を作る。必須 |
| `--exe PATH` | 学習実行ファイル。既定はスクリプト隣の `target/release/examples/bulletou.exe`。Ubuntuでは対応する実行ファイルを指定 |
| `--checkpoint DIR` | 全条件の共通開始checkpoint。非空の `state.bin` と `dataloader_pos.txt` が必要 |
| `--epochs 1 2 5` | 集計したいepoch番号。各条件を一度だけ最大値5まで学習し、1・2・5の結果を出す。省略時はJSONの `max_epochs` まで学習し全epochを集計 |
| `--summary-only` | 保存済みmanifestと各条件のログからCSVだけ再生成する。学習・実行ファイル・共通設定JSONは不要 |
| `--summary-csv PATH` | 集計CSVの出力先。既定はgrid rootの `grid_summary.csv`。元ログ保護のため `trials/` 内は指定不可 |
| `--dry-run` | 計画表示のみ。学習・ファイル作成なし |
| `--resume` | 未完了条件を、それぞれ自身の最新保存checkpointから再開する |
| `--continue-on-error` | ある条件が学習エラーになっても残りを実行する。エラーがあればrunnerの終了コードは非0 |

共通設定の `output` / `output_folder` / `tag` / `resume` / `no_resume` はrunnerが条件ごとに管理します。それ以外はgridで明示した項目と `--epochs` による上書きを除き保持します。元の設定JSONには書き戻しません。入力データ等の相対パスは、BulletOu単体と同じく**コマンド実行時の作業フォルダ**基準です。再開時も同じ作業フォルダ・同じコマンドを使ってください。

`--checkpoint` を省略した場合、共通JSONの `initial_state` / `initial_dataloader_pos` があればそれを使い、なければscratchから始めます。optimizer stateも通常のBulletOu仕様で読み込みます。factorizer構造変更などによる本体側のreset規則は変えません。runner独自のresetや、前の条件の重みの流用はありません。

### 複数値を指定できる項目

| Grid引数 | 上書きするBulletOu設定 |
| --- | --- |
| `--lrs` | `lr` |
| `--lr-mins` | `lr_min` |
| `--wrm-target-scalings` | `wrm_target_scaling` |
| `--wrm-in-scalings` | `wrm_in_scaling` |
| `--wrm-nnue2scores` | `wrm_nnue2score` |
| `--batch-sizes` | `batch_size` |
| `--batches-per-updates` | `batches_per_update` |
| `--factorizers` | `sfnn_factorizer` |
| `--loss-pow-exps` | `loss_pow_exp` |

その他のオプションは `--grid 名前 値1 値2 ...` を繰り返して指定します。名前はBulletOuのオプションから `--` を取ったものです。ハイフンでもアンダースコアでも書けます。

```powershell
python .\grid_search.py `
  --settings-file .\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\20260911\grid-shared `
  --checkpoint C:\path\to\0033 `
  --grid sfnn_factorizer_alpha "shared=0.5" "shared=1.0" `
  --grid no_ft_factorize false true `
  --epochs 1 2
```

これも2×2の4条件です。JSONと同様に、`true` はフラグを指定、`false` はフラグを省略します（**falseが必ず「機能OFF」を意味するわけではありません**）。数値や文字列も指定できます。出力先・開始state・resume・max_epochsなどの実行管理項目はgrid軸にはできません。

## 出力と集計

```text
grid-target-scale/
  grid-manifest.json
  grid_summary.csv
  grid.lock
  trials/
    trial0001-lr=...-<hash>/
      bulletou-settings.json
      grid-state.json
      stdout.log
      summary-learn.csv
      summary-epoch-last.csv
      0001/state.bin, nn.bin, dataloader_pos.txt, ...
    trial0002-.../
      ...
```

短いhashは条件全体から作ります。フォルダ名では省略された条件も `bulletou-settings.json` とmanifestに残ります。再開時だけ、開始state指定を外した `bulletou-resume-settings.json` も作ります。

`grid_summary.csv` は学習開始前からヘッダと空の結果行を作り、条件開始・終了・中断時に更新します。各条件の学習中は本体の `summary-learn.csv` と、接頭辞付きで流れるstdoutを確認してください。stdoutは条件ごとの `stdout.log` にも追記保存します。

CSVは **1行＝1条件×1集計epoch** です。主な列は次の通りです。

| 列 | 内容 |
| --- | --- |
| `trial`, `epoch`, `superbatch` | 条件番号、epoch、実際に記録された最後のsb |
| `test_value_accuracy`, `test_value_loss`, `quantized_value_accuracy`, `quantized_value_loss` | **acc → loss → qacc → qloss**。そのepochの最後の行の値。accuracyは0～1 |
| `max_acc`, `min_loss`, `max_qacc`, `min_qloss` | そのepochで実際に計測された値の最大／最小。4指標は別々のsbで達成していてもよい |
| `max_acc_sb`, `min_loss_sb`, `max_qacc_sb`, `min_qloss_sb` | 各最大／最小のsb。同値なら最初のsb |
| `positions`, `lr_start`, `lr_end` | 本体の最後の行の局面数とLR。`lr_start` はepoch先頭とは限らず、その行の区間の先頭 |
| `lr`, `lr_min`, `wrm_target_scaling` 等 | 指定した学習条件。grid軸は個別列になる。JSONにない既定値を推測で埋めない |
| `status` | そのepochの進捗。末尾sbまであれば `done`。途中の値を完了結果として扱わない |
| `trial_status`, `elapsed_seconds` | 条件全体の実行状態と、起動等も含む総経過秒数。複数epochの行で同じ値になる |
| `output_dir` | 条件の保存先 |
| `checkpoint` | **末尾列**。その最後の行に対応する保存checkpointのフォルダ。未保存・削除済みなら空欄 |

未計測、`nan`、`inf` は空欄です。最後のsbで未計測なら、以前のsbの値で埋めません。最大／最小を達成したsbが未保存なら、そのnn.binがあるかのようなpathは作りません。各指標のepoch末best条件も完了時にstdoutへ表示します。単一の総合点や勝者は勝手に決めません。

本体の `save_rate`、epoch末保存の挙動はそのままです。**runnerはcheckpointを削除せず、bestの自動コピーも作りません。** 各条件の全保存分のディスク容量を見込んでください。

## 再開・集計だけの実行

同じコマンドを再実行すると完了済みの条件をスキップします。未完了条件に保存checkpointがある場合は `--resume` を付けてください。保存時点より後の未保存分は本体の通常resumeと同じく巻き戻ります。保存前に中断してcheckpointが一つもない場合は、記録済みの進捗を勝手に消さずエラーにします。その結果を残したまま、新しいgrid専用出力先でやり直してください。

開始後のgrid内容・epoch数・共通設定がmanifestと違う場合はエラーです。異なる条件を同じ試行として混ぜません。実行ファイルは同じpathで再ビルドして構いませんが、実装変更前後の比較には注意してください。データ・開始state・progress.bin・count.bin等の入力ファイルも内容を固定してください。

```powershell
python .\grid_search.py `
  --output-folder D:\BulletOu-snapshots\20260911\grid-target-scale `
  --summary-only
```

`--epochs 1 2` を付けると、そのepochだけのCSVを再生成できます。条件の元ログは変更しません。集計CSVは生成物なので、手編集した内容は再集計で置き換わります。実行中の同じgridへ別runner／集計処理が書き込むのを防ぐため、OSのファイルロックを使います。`grid.lock` は終了後も残って正常です。

## 比較の注意点

- `wrm_target_scaling` やlossの種類・指数を変えた条件のloss/qlossは、目的関数が異なるため数値をそのまま横比較できません。stdoutの最小loss表示も、この違いを補正していません。
- 比較するtest教師・サンプル数・seed・量子化検証modeを揃えてください。qaccだけで棋力の優劣が確定するわけではありません。
- batch sizeやbpuを変えても、教師局面数をrunnerが自動で増減することはありません。丸めと更新回数は本体のログを確認してください。
- 各条件は本体の**別プロセス**です。workerを使わず、教師RAM cacheやvalidation cacheを条件間で共有しません。短すぎるtrialでは起動コストが目立ちます。
- `validation_rate=0` はepoch末だけ、`-1` は定期検証無効。量子化検証は保存時にも行われる本体の規則に従います。
- YOSCの `--value-loss-min-weight` に相当する機能は、このスクリプトだけでは追加されません。実行ファイルに存在しない引数は実行前にエラーにします。
