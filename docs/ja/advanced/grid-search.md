# Grid searchで学習条件を比較する

### 再開時にgrid軸を外す

`--resume` 時に、たとえば `--grid lr 0.0004` を外してsettings.jsonのlr（epoch別スケジュールも可）へ切り替えられます。残したgrid条件で既存trialが一意に決まる場合に限り、同じフォルダ・checkpointから継続します。完了済みepochの設定・集計値は保持します。trialの元の条件ID・parametersには旧lrが残りますが、実際に次回使う値は`[SETTINGS CHANGED]`およびepoch別のCSVに反映されます。

以前lrを複数値で試していて、外すと複数trialに一致する場合は、勝手に選ばずエラーにします。明示して残したgrid軸の値は従来どおり条件の識別に使います。`--resume --dry-run`で、ファイル変更・学習起動なしに確認できます。

FT・L1・L2のBNは `--grid sfnn-bn-ft false true`、`--grid sfnn-bn-l1 false true`、
`--grid sfnn-bn-l2 false true` で比較できます。全指定で8条件です。
初期値・併用制限・保存仕様は [Batch Normalization](batch-normalization.md) を参照してください。

## 共通設定を編集せずに短い実験を行う

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-short `
  --max-epochs 1 --superbatches 32 `
  --lrs 0.0001 0.0002
```

各条件を1epoch・32SBで学習します。元のJSONは変更しません。優先順位は共通JSON < 上記のCLI上書き < 明示的な `--grid superbatches ...` です。`--max-epochs` と `--epochs` の併用時、前者が学習終了epoch、後者が集計対象です（終了epochを超える指定はエラー）。`--max-epochs` 未指定時の `--epochs` は従来通り最大値まで学習します。`--resume` にも適用されますが、通算epoch数を減らすことはできません。再開時の上書きを維持するには同じ引数を指定してください。

## 飽和率の詳細表示

通常は学習中の `[qstats]` / `[qstats-unit]` をコンソールに表示しません。表示したいときは `python .\grid_search.py ... --verbose` を指定してください。子プロセスのBulletOuに `--verbose` を渡します。本体を直接起動するときも `--verbose`、学習JSONでは `"verbose": true` を指定できます。

表示の有無だけを変更し、飽和率の計測・既存CSV列への記録、acc/loss/qacc/qlossの表示は変えません。gridの `--verbose` は実験条件・trial識別には含めず、resume時にも変更できます。共通JSON側で `verbose: true` とした場合も詳細を表示します。

## 学習開始時のLR warmup

warmupは独立した**epoch 0**です。`warmup_sb`は`superbatches`より大きくても構いません。例えば`warmup_sb: 1024, superbatches: 64, max_epochs: 1`はwarmupを1024sb、通常学習を64sb、合計1088sb実行します。最初の1sbだけで打ち切る指定ではありません。

JSONの `"warmup_sb": 1`（CLI: `--warmup-sb 1`）で、epoch 0を1sb実行し、ほぼ0から指定の`lr`まで線形に増加させます。非負整数で、デフォルト`0`は無効です。

- cuda-cppの通常学習とworkerの`step` / `geometric` / `cos`に対応。`plateau`とdirect-step smokeモードは未対応で、指定時はエラーです。
- optimizer更新ごとに変更します。warmup中の更新総数をW、更新番号をu（1～W）とすると `lr × u/W`。bpuが異なっても指定sbの末尾で`lr`に達します。
- `max_epochs`にepoch 0は含みません。通常epochのsb数や減衰スケジュールは変えません。epoch 0にはepoch 1の設定（LR、bpuなど）を使います。
- epoch 1から通常の減衰を行います。明示的な`lr_step_gamma: 1.0`なら通常epochは固定LRです。
- resumeは保存済みepoch/sbから続行し、warmupをやり直しません。epoch 0完了checkpointならepoch 1から再開します。別runへの`initial_state`による追加学習と独立したgrid/worker trialでは、新たにepoch 0から適用します。
- epoch 0も通常の保存・検証間隔とepoch末尾保存の設定に従います。`summary-learn.csv`、`grid_summary.csv`にepoch 0として記録します。旧仕様のepoch 1内warmupとは意味が異なるため、比較実験は新しい出力フォルダで開始してください。
- epoch別の値変更には対応せず、run全体で一つの整数です。`warmup_sb`は保存情報とgrid CSVに記録します。

比較例（共通設定で`superbatches: 16`、`max_epochs: 1`を確認してください）：

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260922-progress8-bceloss-16sb.json `
  --output-folder D:\BulletOu-snapshots\20260922\grid-progress8-warmup `
  --grid loss_bce_with_logits true `
  --grid lr 0.000400 `
  --grid lr_step_gamma 1.0 `
  --grid warmup_sb 0 1 4
```

warmupだけを変える3条件の比較です。

共通JSONでは `"lr": {"epoch1": 0.0004, "epoch11": 0.0002}` のような[epoch別設定](epoch-settings.md)も使えます。QATなどのtrue/false切替にも対応し、CSVには各epochの有効値を記録します。

`grid_summary.csv` はtrial開始時・終了時・中断時に加え、**学習中もepoch完了ごとに更新**します。本体の `summary-learn.csv` を約1秒間隔で確認するため、stdoutの次の出力やtrial全体の終了を待ちません。10epochのtrialなら、1epoch完了時点からその結果を確認できます。未完了epochの測定値は空欄のままです（`--epochs` で掲載epochを限定した場合はその指定に従います）。書き込み途中の元CSVや、一時的な出力ファイルのロックは警告して再試行し、学習は停止しません。

BCEとの比較は `--grid loss_bce_with_logits false true` で指定できます。共通設定の `wrm_in_offset` は0にし、明示的な `win_rate_model` / `loss_sigmoid_mse` は無効にしてください。[BCEの計算式・注意点](bce-with-logits.md)を参照してください。

BCEの誤差重み付けは、共通JSONを `loss_bce_with_logits: true` にして `--grid bce_error_weight_k 0 1 2` で比較できます。0が通常のBCEです。学習時だけ重み付けし、検証loss/qlossは全条件で通常のBCEを使います。係数は `grid_summary.csv` に記録します。[計算式と正規化](bce-with-logits.md#誤差が大きい局面を重視するbce)も参照してください。

SFNNのノルム正則化も `--grid sfnn_norm_loss_strength 0 0.000001 0.00001 0.0001` で比較できます。[Norm lossの対象・計算式・実行例](norm-loss.md)を参照してください。

[English](../../en/advanced/grid-search.md)

## WRM教師勝率の圧縮を比較する

`--wrm-target-epsilon`（JSON: `wrm_target_epsilon`）は、従来のWRM変換後の教師勝率 `t` を `ε + (1 - 2ε) * t` に変換します。既定値は **0（無効）**、範囲は有限の `0 <= ε < 0.5` です。例えばε=0.01なら、0→0.01、0.5→0.5、1→0.99です。端だけをclipするのではなく、全域を0.5へ寄せます。

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260911-k3k3-b65536-sb40m.json `
  --output-folder D:\BulletOu-snapshots\grid-target-epsilon `
  --wrm-target-epsilons 0 0.005 0.01
```

この例は3条件を順次実行します。他のgrid軸も指定すると直積になります。`grid_summary.csv` の `wrm_target_epsilon` 列に各値を記録します。単独学習では `--wrm-target-epsilon 0.01` または設定JSONの `"wrm_target_epsilon": 0.01` を使います。

- 教師側だけに適用し、予測側WRM・nn.bin形式・acc/qaccの定義は変更しません。学習と通常/GPU量子化/CPU量子化検証のlossで共通の変換です。単独の量子化検証にも同じεを指定してください。
- 既存のresult/lambda混合より前に適用します。対局結果との混合を使う場合、最終教師値の範囲が必ず `[ε, 1-ε]` になるとは限りません。
- `wrm_target_offset` は非線形な勝率変換を変えますが、極限は0/1のままです。εは極限自体をε/1−εにします。重みや出力を直接制限する機能ではありません。
- ε=0は従来どおりです。`--loss-sigmoid-mse` と非ゼロεの併用はエラーです。
- **εが異なるloss/qlossは目的関数が異なり、そのまま優劣を比較できません。** acc/qaccや棋力も確認してください。gridはこの点を警告します。

`grid_search.py` は、YOSC の `trainer/grid_search.py` と同様に、指定した値の**全組み合わせを1本ずつ学習し、比較結果をCSVへ集計する**スクリプトです。Python 3.10以降の標準ライブラリだけで動きます。対象はBulletOuのcuda-cpp production学習です。

TPEの `tuning_parameters.py` とは別です。勝者から次の条件へ追加学習することも、途中で条件を間引くこともありません。新規学習なら各条件が同じ決定的初期化から、追加学習なら全条件が同じ開始checkpointと教師位置から始まります。GPUの浮動小数点演算による再現誤差までなくすものではありません。

## まず実行計画だけ確認する

共通設定には普通の `bulletou-settings.json` を使います。**tuning-settings.jsonではありません。** [設定例](../../examples/grid-search-bulletou-settings.json) はk3k3、FT factorizer有効・L1 shared、1epoch=324sb、1sb=40M指定、81sbごと保存です。パスや条件は自分の環境に合わせてください。このファイルを追加しただけで学習が開始されることはありません。

BulletOuフォルダで実行する例です。

検証局面を全件使う場合は、JSONに `"test_positions": "all"` と書けます。省略・`null` でも同じ全件検証です。件数を制限する場合は `"test_positions": 300000` のように正の整数を指定します。古い実行ファイルで `all` がエラーになる場合は、本体を再ビルドしてください。

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
| `--output-folder DIR` | **このgrid専用**の保存先。直下に `grid_summary.csv` と各 `trial0001-…/` を作る。必須 |
| `--exe PATH` | 学習実行ファイル。既定はスクリプト隣の `target/release/examples/bulletou.exe`。Ubuntuでは対応する実行ファイルを指定 |
| `--checkpoint DIR` | 全条件の共通開始checkpoint。非空の `state.bin` と `dataloader_pos.txt` が必要 |
| `--epochs 1 2 5` | 集計したいepoch番号。各条件を一度だけ最大値5まで学習し、1・2・5の結果を出す。省略時はJSONの `max_epochs` まで学習し全epochを集計 |
| `--max-epochs N` | 共通JSONの `max_epochs` を上書き。追加数ではなく学習の通算epoch数。未指定なら従来通り |
| `--superbatches N` | 共通JSONの `superbatches`（1epochのSB数）を上書き |
| `--summary-only` | 保存済みmanifestと各条件のログからCSVだけ再生成する。学習・実行ファイル・共通設定JSONは不要 |
| `--summary-csv PATH` | 集計CSVの出力先。既定はgrid rootの `grid_summary.csv`。元ログ保護のため各trialフォルダ内（旧 `trials/` 内も含む）は指定不可 |
| `--dry-run` | 計画表示のみ。学習・ファイル作成なし |
| `--resume` | 未完了条件を最新保存checkpointから再開する。checkpointがなければ途中ログを退避して元の初期状態から再実行 |
| `--continue-on-error` | ある条件が学習エラーになっても残りを実行する。エラーがあればrunnerの終了コードは非0 |

共通設定の `output` / `output_folder` / `tag` / `resume` はrunnerが条件ごとに管理します。それ以外はgridで明示した項目と `--epochs` による上書きを除き保持します。元の設定JSONには書き戻しません。入力データ等の相対パスは、BulletOu単体と同じく**コマンド実行時の作業フォルダ**基準です。再開時も同じ作業フォルダ・同じコマンドを使ってください。

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
  --grid sfnn_l1_factorizer_alpha "shared=0.5" "shared=1.0" `
  --grid ft_factorizer false true `
  --epochs 1 2
```

これも2×2の4条件です。`ft_factorizer`（alias: `sfnn_ft_factorizer`）、`save_epoch_end`、`sfnn_factorized` は `true` / `false` の両方を明示的に渡します。それ以外のフラグはJSONと同様に、`true` で指定、`false` で省略するため、falseが必ず既定で有効な機能をOFFにするとは限りません。数値や文字列も指定できます。出力先・開始state・resumeなどの実行管理項目はgrid軸にはできません。

`--grid max-epochs 1 2`（`max_epochs`表記も可）で、1epochと2epochを独立した条件として学習できます。`--grid superbatches 16 32`と併用すれば4条件です。grid指定は共通JSONや `--max-epochs` より優先します。集計は各条件の終了epochまでで、`--epochs`を併用するとその範囲内の指定epochだけを表示します。再開時もepoch数が異なるgrid値は別条件です。既存条件の延長ではありません。

## 出力と集計

新規trialはgrid root直下に配置します。既存manifestの `trials/` 配下にあるtrialは移動せず、`--resume` で従来の場所から再開します。追加条件の新規trialは直下に作ります。

```text
grid-target-scale/
  grid-manifest.json
  grid_summary.csv
  grid.lock
  trial0001-lr=...-<hash>/
    bulletou-settings.json
    grid-state.json
    stdout.log
    summary-learn.csv
    0001/state.bin, nn.bin, dataloader_pos.txt, ...
  trial0002-.../
    ...
```

短いhashは条件全体から作ります。フォルダ名では省略された条件も `bulletou-settings.json` とmanifestに残ります。再開時だけ、開始state指定を外した `bulletou-resume-settings.json` も作ります。

`grid_summary.csv` は、YOSCと同様に学習開始前から全条件の行を作ります。各行には実験条件・対象epoch・保存先・状態を記録し、条件開始・終了・中断時に再集計します。**測定結果はepoch単位で、末尾sbまで記録されたepochに記入します。** trial全体が未完了でも、完了済みepochの結果は表示します。未完了epochのacc/loss/qacc/qloss・末尾8SB平均・実測sb・checkpointだけを空欄にします。途中経過は本体の `summary-learn.csv` や条件ごとの `stdout.log` で確認してください。元ログは変更しません。

例：epoch 1完了後にmax_epochsを3へ延長し、epoch 2の途中なら、epoch 1の測定結果は表示し、epoch 2・3は条件だけ表示します。trialが中断・失敗しても完了済みepochは掲載します。既存CSVも `--summary-only` または次回起動で、この規則に従って再集計されます。

CSVは **1行＝1条件×1集計epoch** です。主な列は次の通りです。

| 列 | 内容 |
| --- | --- |
| `trial`, `epoch`, `superbatch` | 条件番号、epoch、実際に記録された最後のsb |
| `test_value_accuracy`, `test_value_loss`, `quantized_value_accuracy`, `quantized_value_loss` | **acc → loss → qacc → qloss**。そのepochの最後の行の値。accuracyは0～1 |
| `last8_acc`, `last8_loss`, `last8_qacc`, `last8_qloss` | 各epoch末尾8SB内の有限な検証値の算術平均。小数点以下10桁固定（例：`0.6349000000`）。最終SBの4指標の直後に出力。最大／最小値とそのSB番号の8列は廃止 |
| `positions` | 本体の最後の行の累積局面数。最終sbの `lr_start` / `lr_end` は集計に出しません。設定値の `lr` / `lr_min` は条件列に残します |
| `lr`, `lr_min`, `wrm_target_scaling` 等 | 指定した学習条件。grid軸は個別列になる。JSONにない既定値を推測で埋めない |
| `status` | epoch完了は `done`。未完了は `pending` / `running` / `interrupted` / `failed` / `incomplete`。`trial_status`はtrial全体の状態なので異なる場合があります |
| `trial_status` | 条件全体の実行状態。経過時間 `elapsed_seconds` は集計CSVには出力しない |
| `output_dir` | 条件の保存先 |
| `checkpoint` | **末尾列**。その最後の行に対応する保存checkpointのフォルダ。未保存・削除済みなら空欄 |

未計測、`nan`、`inf` は空欄です。最後のsbで未計測なら、最終SBの指標を以前のsbの値で埋めません。`last8_*` は「最後の8回の検証」ではなく、末尾SB番号をNとしてN−7〜NのSB内だけを対象にします。8SB未満のepochはそのepoch全体を対象とし、epochをまたぎません（warmupのepoch 0も独立）。未計測・非有限値は指標ごとに除外し、有効値がなければ空欄です。未完了epochの平均も空欄です。各指標のepoch末best条件は完了時にstdoutへ表示します。単一の総合点や勝者は勝手に決めません。

既存のCSVは次回起動または `--summary-only` で元ログから新しい列へ再集計されます。本体の再ビルドは不要です。起動中のrunnerには反映されません。元の `summary-learn.csv` は変更しません。

本体の `save_rate`、epoch末保存の挙動はそのままです。**runnerはcheckpointを削除せず、bestの自動コピーも作りません。** 各条件の全保存分のディスク容量を見込んでください。

各epoch末だけ保存するなら、共通設定に `"save_rate": 0` または `"save_rate": "none"` を指定します。`validation_rate: 1` / `quantized_validation_rate: 1` とは独立なので、毎sbの計測は継続できます。epoch末保存も止める指定と注意点は[検証チュートリアル](../tutorial/4-validation.md#45-保存頻度とは別に考える)を参照してください。

## 再開・集計だけの実行

同じコマンドを再実行すると完了済みの条件をスキップします。未完了条件には `--resume` を付けてください。保存checkpointがあれば、そこから再開します。保存時点より後の未保存分は本体の通常resumeと同じく巻き戻ります。

**checkpointが一つもなければ、同じgrid内でそのtrialの元の初期状態から再実行します。** 例えばepoch 1の途中で保存前に停止した場合、epoch 1の先頭からやり直します。共通開始checkpointを指定していた場合はその重み・教師位置を使い、指定していなければscratchからです。中断したsbの状態は復元できません。

以前のtrialフォルダは丸ごと `<grid root>/interrupted-runs/<trial名>-<日時>-<識別子>/` に退避し、元のtrial番号・保存先で再実行します。途中ログや不完全な保存ファイルも消さずに残し、新しい学習の集計には混ぜません。stdoutに `[RESTART]` と `[ARCHIVE]` を表示します。完了済み条件・選択していない条件は再実行しません。手動でフォルダを削除したり、新しいgrid rootを作る必要はありません。

通常の再実行ではmanifestと異なる計画を拒否します。ただし `--resume` では、同じgrid条件のcheckpointを引き継ぎつつ、共通JSONのLR・batch size・bpu・保存／検証頻度などを変更できます。stdoutの `[SETTINGS CHANGED]` に変更前後を表示します。実行中のJSON変更を自動反映する機能ではなく、次の起動から反映します。

条件の識別には明示的なgrid引数を使います。例えば `--lrs 0.0001 0.0002` を指定していれば、JSONの `lr` よりその値を優先します。gridの値を変更した条件には、別条件のcheckpointを流用せず、新規trialを作ります。`arch`・`backend`・`ft_factorizer` の変更は既存checkpointとの互換性のため拒否します。それ以外もBulletOu本体の読み込み・引数検証は行われます。

完了済みtrialは設定変更だけでは再実行しません。延長するには `max_epochs` を増やしてください。共通設定を途中変更した結果は、最初から同一条件で学習した結果とは区別して比較してください。実行ファイルは同じpathで再ビルドできますが、入力ファイルの内容変更や実装変更も比較に影響します。

各trialの `grid-settings-history.json` に起動ごとの設定・時刻・起動前ログの最終epoch/sbを記録します。このログ位置は実際の再開checkpoint位置ではありません。元の `bulletou-settings.json` は保持し、実行設定は `bulletou-resume-settings.json`（checkpoint再開）または `bulletou-run-settings.json`（新規実行）へ書きます。共通JSONには書き戻しません。

変更前の完了epochには当時の設定をmanifestに記録し、`grid_summary.csv` のLR・bpu・sb数などを新設定で上書きしません。途中epochで設定を変更した場合は、完了時の設定をそのepochの行に表示します。epoch全体がその設定だったことを意味しないため、途中変更の詳細は起動履歴を確認してください。checkpointがなく再実行する場合は、新しい要求設定の初期stateから開始し、古い結果は退避します。

途中で変更・追加・削除した設定項目も、既定の集計項目以外なら `grid_summary.csv` に自動で列を追加します。例えば途中からQATを有効化した場合、変更前の完了epochには当時の値、変更後には新しい値を表示します。元の値に戻したり設定から項目を削除しても、一度追加した列は残します。そのepochで未指定だった値は空欄で、BulletOuのデフォルト値を推測して埋めません。列は既存の条件列の後、status等の管理列の前に追加し、checkpointは末尾を維持します。旧manifestに残っている初期設定・epoch別設定からも変更列を復元します。

### 完了した条件を延長する

通常実行・`--resume` の実行順とCSV表示順は、YOSCの通常実行と同じく**今回の引数で指定した条件の順**です。数値順にはソートしません。既存条件と追加条件を区別せず並べ、今回指定しなかった既存条件は元の相対順で末尾に残します。trial番号・保存先は不変なので、CSVのtrial番号は昇順とは限りません。`--summary-only` はmanifestに保存された直近の並び順を使います。

例えば600／1200／1800を各5epochで計画した既存gridで、600／1200だけをさらに5epoch、**通算10epochまで**学習するには、同じ出力先を指定します。共通JSONの `max_epochs: 5` は書き換えなくて構いません。

```powershell
python .\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings-20260911-k3k3-b65536-sb40m.json `
  --output-folder D:\BulletOu-snapshots\20260911\grid-k3k3-b65536-sb40m `
  --wrm-target-scalings 600 1200 `
  --epochs 10 `
  --resume
```

- `--epochs 10` は「追加10epoch」ではなく、通算の終了epochです。`--epochs 1 2 3 4 5 6 7 8 9 10` と列挙しても構いません。
- 通常のgrid引数に書いた条件だけ実行します。専用のtrial選択オプションはありません。元のgrid軸名はすべて指定してください。値のリストは絞ることも追加することもできます。`--resume` で未登録の条件を指定すると、既存の最大trial番号に続けて新規登録します。既存の番号・保存先・結果は保持し、新条件は共通の初期state・教師位置から学習します。完了済みの既存条件はスキップします。
- 例：最初に `--grid wrm_target_offset 135 270 540 0` を実行した同じoutput-folderで、`--grid wrm_target_offset 70 35 100 135 170 200 235 270 540 0 --resume` とすれば、新しい6条件を追加できます。指定しなかった既存条件もmanifestと集計に残り、新条件の完了後は同じ `grid_summary.csv` で比較できます。共通JSONの変更は選択した条件に適用します。`--dry-run` なら追加・変更計画だけ確認でき、ファイルも学習も変更しません。
- 600／1200それぞれの保存済みcheckpointから、本体の `--resume` で重み・optimizer・教師位置を継続します。完了した5epoch目の保存があればepoch 6から再開します。中断中なら最後の保存点から再開し、未保存分は巻き戻ります。
- trial番号・フォルダ名は変えません。名前末尾のhashも作成時のものを維持します。
- 元の `bulletou-settings.json` は保持し、延長後の設定は `bulletou-resume-settings.json` とmanifestに記録します。共通JSONへの書き戻しはしません。
- 既存の集計epochを保持し、延長区間の各epochを同じ `grid_summary.csv` に追加します。元の集計が1～5なら、`--epochs 10` だけでも1～10を集計します。
- 指定しなかった1800は学習・設定を変更せず、既存の1～5epochの結果だけCSVに残します。1800の6～10epochという空行は作りません。
- 延長対象に使用可能なcheckpointがない場合も、以前の結果を退避して元の初期状態から目標epochまで再実行します。例えば5→10への延長でも、checkpointがなければepoch 1から10までの再学習になります。
- 延長中に止めた場合も、同じ `--resume --epochs 10` のコマンドで再開できます。10epoch完了後の再実行はスキップし、さらに5epochを勝手に追加しません。目標epochの縮小は拒否します。
- `--dry-run` を追加すると、対応付けられた既存trial・フォルダ・終了epochを表示するだけです。ファイルを書き換えず、学習も開始しません。

同じgridを実行中のrunnerがある場合は、先にそちらを停止してください。同じ出力先への実行・更新はファイルロックで排他します。

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
