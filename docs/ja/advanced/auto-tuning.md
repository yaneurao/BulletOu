# ESによる自動調整

<a href="../../en/advanced/auto-tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

factorizer の alpha や count confidence は、組み合わせが多く、手で総当たりすると時間がかかります。すでに良い checkpoint がある場合は、`es_local_runner.py` で beam search 風の ES (evolution strategy) を回せます。

このページでは、`parameters.json` の書き方、各パラメーターの意味、runner の引数、ESせずに同じパラメーターだけを通常学習へ渡す方法を説明します。

## 何をする仕組みか

`es_local_runner.py` は、複数の candidate を作って短く学習させ、成績の良い candidate だけを残していく runner です。

1 generation の流れは次の通りです。

1. `parameters.json` の `current` を中心にして、`tune: true` のパラメーターをランダムに動かす
2. `population` 本の candidate を作る
3. 各 candidate を `beam` に書いた sb 数だけ学習する
4. `metric` で成績を比べ、`keep` 本だけ残す
5. 最後に残った candidate の NN 重みとパラメーター値を採用する
6. 採用された値を `parameters.json` に書き戻す

採用後にパラメーターだけを少し動かす処理はありません。最後に残った candidate のパラメーター値そのものが、次の generation の開始値になります。

## `parameters.json` の全体形

`parameters.json` は、ES runner と通常学習の両方で使います。

```json
{
  "version": 1,
  "es": {
    "enabled": true,
    "generations": 100,
    "population": 16,
    "beam": [
      { "after_sbs": 8, "keep": 8 },
      { "after_sbs": 16, "keep": 4 },
      { "after_sbs": 24, "keep": 2 },
      { "after_sbs": 32, "keep": 1 }
    ],
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "seed": 1,
    "save_rate": 1,
    "candidate_validation_rate": 1,
    "candidate_quantized_validation_rate": 1
  },
  "parameters": {
    "shared": { "current": 1.0, "tune": false, "step": 0.0, "min": 0.0, "max": 10.0 },
    "king_axis": { "current": 1.0, "tune": true, "step": 0.03, "min": 0.0, "max": 10.0 },
    "hand_axis": { "current": 1.0, "tune": true, "step": 0.03, "min": 0.0, "max": 10.0 },
    "progress_axis": { "current": 1.0, "tune": true, "step": 0.03, "min": 0.0, "max": 10.0 },
    "pair": { "current": 0.3, "tune": true, "step": 0.02, "min": 0.0, "max": 10.0 },
    "residual_count": { "current": 1.0, "tune": true, "step": 0.25, "min": 0.0, "max": 20.0 },
    "king_axis_count": { "current": 4.0, "tune": true, "step": 0.5, "min": 0.0, "max": 100.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "step": 0.5, "min": 0.0, "max": 100.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "step": 0.5, "min": 0.0, "max": 100.0 },
    "king_hand_pair_count": { "current": 10.0, "tune": true, "step": 1.0, "min": 0.0, "max": 200.0 },
    "king_progress_pair_count": { "current": 10.0, "tune": true, "step": 1.0, "min": 0.0, "max": 200.0 },
    "hand_progress_pair_count": { "current": 10.0, "tune": true, "step": 1.0, "min": 0.0, "max": 200.0 }
  }
}
```

`es.enabled` は、この JSON をどう使うかを表します。

| 値 | 使い方 |
| --- | --- |
| `true` | `es_local_runner.py` で ES を回す |
| `false` | `bulletou.exe` の通常学習で、`parameters.*.current` の値だけ使う |

## `es` の各項目

| 項目 | 意味 |
| --- | --- |
| `enabled` | `true` なら `es_local_runner.py` 用。`false` なら `bulletou.exe --parameters-file` 用 |
| `generations` | ESを何世代回すか。1世代ごとに最後の survivor が採用される |
| `population` | 1世代の最初に作る candidate 数 |
| `beam` | 何 sb の時点で何本残すか |
| `metric` | candidate を比較する指標 |
| `lower_is_better` | `metric` が小さいほど良いなら `true`、大きいほど良いなら `false` |
| `seed` | candidate のランダム生成に使うseed |
| `save_rate` | 何回 accept するごとに `accepted-checkpoints/` へ外向け保存するか |
| `candidate_validation_rate` | candidate 学習中に test accuracy/loss を何 sb ごとに出すか |
| `candidate_quantized_validation_rate` | candidate 学習中に qacc/qloss を何 sb ごとに出すか |

`beam` は、次のように書きます。

```json
"beam": [
  { "after_sbs": 8, "keep": 8 },
  { "after_sbs": 16, "keep": 4 },
  { "after_sbs": 24, "keep": 2 },
  { "after_sbs": 32, "keep": 1 }
]
```

この例では、16本で開始し、8 sb で8本、16 sb で4本、24 sb で2本、32 sb で1本に絞ります。最後の `keep` は必ず `1` にしてください。

`metric` は次の値を指定できます。

| 値 | 意味 | `lower_is_better` |
| --- | --- | --- |
| `quantized_value_loss` | 量子化後の loss。棋力計測に近い候補を探す目的では主にこれを見る | `true` |
| `quantized_value_accuracy` | 量子化後の accuracy | `false` |
| `test_value_loss` | f32 weight の validation loss | `true` |
| `test_value_accuracy` | f32 weight の validation accuracy | `false` |

通常は `metric = "quantized_value_loss"`、`lower_is_better = true` で始めるのが扱いやすいです。

## `parameters` の共通フィールド

`parameters` の各項目は、次の形で書きます。

```json
"pair": { "current": 0.3, "tune": true, "step": 0.02, "min": 0.0, "max": 10.0 }
```

| フィールド | ES runner での意味 | 通常学習での意味 |
| --- | --- | --- |
| `current` | 現在値。candidate はこの値を中心に作られる | この値だけが `bulletou.exe` に渡される |
| `tune` | `true` なら candidate 生成時に動かす。`false` なら固定する | 無視される |
| `step` | `current ± step` の範囲で candidate を作る | 無視される |
| `min` | candidate 値の下限 | 無視される |
| `max` | candidate 値の上限 | 無視される |

例えば `pair.current = 0.3`, `pair.step = 0.02` なら、candidate の `pair` は `0.28` から `0.32` の範囲で作られます。`tune: false` にした項目は動きません。

## alpha 系パラメーター

alpha は、factorizer 成分をどれだけ forward/backward に効かせるかを決める係数です。`1.0` ならそのまま、`0.5` なら半分、`2.0` なら2倍です。

| JSONキー | 対応する意味 | 使われる条件 |
| --- | --- | --- |
| `shared` | 全 bucket 共通の shared factorizer の強さ | `--sfnn-factorizer shared` または `pair` などで shared が有効なとき |
| `king_axis` | king bucket 軸の axis factorizer の強さ | arch に `k3k3` / `k9k9` / `k21k21` / `k29k29` などの king bucket があるとき |
| `hand_axis` | hand bucket 軸の axis factorizer の強さ | arch に `hand4` / `hand16` / `hand64` / `hand1024` などの hand bucket があるとき |
| `progress_axis` | progress bucket 軸の axis factorizer の強さ | arch に `progress4` / `progress8` などがあるとき |
| `pair` | 2軸組み合わせの pair factorizer の強さ | `--sfnn-factorizer pair` で pair が有効なとき |

`pair` は、king-hand / king-progress / hand-progress の pair factorizer をまとめて指定する alpha です。pair の種類ごとに変えたい場合は、現時点では alpha ではなく count confidence 側で調整します。

実効重みは、おおまかには次のように考えます。

```text
W_effective =
    W_residual
  + shared_alpha * W_shared
  + axis_alpha   * axis_confidence * W_axis
  + pair_alpha   * pair_confidence * W_pair
```

`shared` を下げると全体の土台まで弱くなるので、まずは `shared = 1.0` 固定にするのが安全です。`axis` や `pair` を動かすほうが、bucket ごとの差分の効かせ方を調整しやすいです。

## count confidence 系パラメーター

count confidence は、`bucket-count` で作った `count.bin` を使い、出現回数が少ない成分を弱めるための係数です。これらの値を `0.0` にすると無効です。値を大きくすると、より多くの出現回数がないと、その成分を強く使わなくなります。

count confidence を1つでも使う場合は、runner には `--bucket-counts <count.bin>`、通常学習には `--sfnn-bucket-counts <count.bin>` が必要です。

| JSONキー | 対応するCLI | 意味 |
| --- | --- | --- |
| `residual_count` | `--sfnn-residual-count-confidence` | bucket 固有 residual を count に応じて抑える |
| `axis_count` | `--sfnn-axis-count-confidence` | king / hand / progress axis の共通値 |
| `king_axis_count` | `--sfnn-king-axis-count-confidence` | king axis だけの値 |
| `hand_axis_count` | `--sfnn-hand-axis-count-confidence` | hand axis だけの値 |
| `progress_axis_count` | `--sfnn-progress-axis-count-confidence` | progress axis だけの値 |
| `pair_count` | `--sfnn-pair-count-confidence` | king-hand / king-progress / hand-progress pair の共通値 |
| `king_hand_pair_count` | `--sfnn-king-hand-pair-count-confidence` | king-hand pair だけの値 |
| `king_progress_pair_count` | `--sfnn-king-progress-pair-count-confidence` | king-progress pair だけの値 |
| `hand_progress_pair_count` | `--sfnn-hand-progress-pair-count-confidence` | hand-progress pair だけの値 |

個別キーを省略すると、共通キーを使います。たとえば `axis_count = 1.0` だけを指定し、`king_axis_count` を省略すると、king axis も `1.0` として扱われます。個別キーに `0.0` を明示すると、その軸だけ count confidence を無効にできます。

axis / pair factorizer の足し込み量には、次の係数が掛かります。

```text
confidence = count_term / (count_term + term_params * option_value)
```

`count_term` は、その factorizer 行が対応する bucket の出現回数合計です。`term_params` は、その factorizer 行が持つパラメーター数です。`option_value` が大きいほど、出現回数が少ない行は弱くなります。

residual 側は少し違い、bucket 固有 residual に decay を入れます。詳しい数式は [SFNN factorizer](sfnn-factorizer.md) を参照してください。

## ES runner の実行例

```powershell
$base = "C:\shogi\YaneuraOuWorks\BulletOu\checkpoints\...\0256"

python .\es_local_runner.py `
  --exe C:\shogi\YaneuraOuWorks\BulletOu\target\release\examples\bulletou.exe `
  --parameters-file .\parameters.json `
  --base-checkpoint $base `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --output-folder D:\BulletOu-snapshots\20260820 `
  --temp-folder C:\BulletOu-es-temp `
  --tag-prefix pair2-qloss `
  --factorizer pair `
  --positions-per-superbatch 40000000 `
  -- `
  --lr 0.000030 `
  --lr-min 0.000010 `
  --wrm-in-offset 0 `
  --wrm-target-offset 0 `
  --lr-schedule step `
  --optimizer ranger `
  --optimizer-weight-decay 0.0 `
  --batches-per-update 1 `
  --sfnn-dirty-bucket-update `
  --sfnn-saturation-penalty 1e-7
```

`--` だけの行は区切りです。そこから後ろは runner ではなく `bulletou.exe` へ渡されます。`--lr` や `--optimizer` のような、candidate 間で共通に使う学習条件を書きます。

runner が自動で指定するので、`--` より後ろ側には `--resume`、`--parameters-file`、`--superbatches`、`--max-epochs`、`--save-rate`、`--validation-rate`、`--quantized-validation-rate`、`--tag`、`--output-folder`、`--initial-state`、`--initial-dataloader-pos`、`--sfnn-factorizer-alpha`、count confidence 系オプションは書かないでください。

## ES runner の引数

| 引数 | 意味 |
| --- | --- |
| `--exe` | 実行する `bulletou.exe` のパス |
| `--parameters-file` | ES設定とパラメーター現在値を書いた JSON。`es.enabled` は `true` にする |
| `--base-checkpoint` | 最初に読み込む checkpoint フォルダ。`state.bin` と `dataloader_pos.txt` が必要 |
| `--teacher` | 学習用教師データ |
| `--test-teacher` | validation 用教師データ |
| `--arch` | 学習する architecture |
| `--bucket-counts` | count confidence を使うための `count.bin` |
| `--output-folder` | runner root を作る親フォルダ |
| `--temp-folder` | candidate の一時 checkpoint を置くフォルダ。速いSSDを推奨 |
| `--tag-prefix` | runner root 名に使う識別名 |
| `--factorizer` | candidate 学習時に渡す `--sfnn-factorizer` の値 |
| `--positions-per-superbatch` | 1 sb の局面数 |
| `--generations` | JSON の `es.generations` を一時的に上書きする |
| `--save-rate` | JSON の `es.save_rate` を一時的に上書きする |
| `--metric` | JSON の `es.metric` を一時的に上書きする |
| `--resume` | `runner-state.json` と `current/` から再開する |
| `--keep-temp` | 落選 candidate の一時フォルダを残す |
| `--dry-run` | コマンド表示だけ行う |
| `--no-stream-child-output` | `bulletou.exe` の stdout を画面へ流さない。ログファイルには残る |
| `--color` | 色付き出力の制御。`auto` / `always` / `never` |

## ESせずに `parameters.json` の値だけ使う

ESで見つけた値を固定して学習したい場合は、`parameters.json` の `es.enabled` を `false` にします。その状態で `bulletou.exe` に同じ `--parameters-file` を渡します。

```powershell
.\target\release\examples\bulletou.exe `
  --backend cuda-cpp `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --sfnn-factorizer pair `
  --parameters-file .\parameters.json `
  --sfnn-bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --positions-per-superbatch 40000000 `
  --superbatches 32 `
  --max-epochs 1 `
  --lr 0.000030 `
  --lr-min 0.000010
```

この使い方では、`parameters.*.current` だけが学習に反映されます。`step`、`tune`、`beam` は通常学習では使いません。

`--parameters-file` と `--sfnn-factorizer-alpha`、count confidence 系オプションを同時に指定することはできません。値の入口が2つあると、どちらが有効かわからなくなるためです。

## 出力フォルダ

runner root は `--output-folder\es-<tag-prefix>` です。ここにログと現在 checkpoint が置かれます。

| パス | 内容 |
| --- | --- |
| `current/` | 常に最新の採用 checkpoint。`--resume` はここから再開する |
| `accepted-checkpoints/sbXXXXXXXX/` | `save_rate` 世代ごとの外向け保存 checkpoint |
| `summary-learn.log` | すべての candidate / stage の結果 |
| `accepted-summary-learn.log` | 採用された survivor だけの結果 |
| `parameters-history.jsonl` | 採用時点のパラメーター履歴 |
| `runner-state.json` | 再開用の内部状態 |
| `logs/` | candidate ごとの stdout log |
| `temp/` | candidate の一時 checkpoint。`--temp-folder` 指定時はそちらに作る |

`--temp-folder` を指定すると、candidate の一時 checkpoint をそこへ置きます。SSD の `C:\BulletOu-es-temp` などを指定すると、`D:` に大量の一時フォルダを作らずに済みます。落選 candidate の一時フォルダは自動削除されます。調査用に残す場合だけ `--keep-temp` を指定します。

## 再開

再開するときは同じ `--output-folder` と `--tag-prefix` を指定して `--resume` を付けます。現在のハイパーパラメーターは `parameters.json` から読むので、checkpoint 時点の値をコマンドラインへ手で書き直す必要はありません。

```powershell
python .\es_local_runner.py `
  --resume `
  --exe C:\shogi\YaneuraOuWorks\BulletOu\target\release\examples\bulletou.exe `
  --parameters-file .\parameters.json `
  --teacher D:\sojoteam_datasets `
  --test-teacher C:\shogi\teacher\test\test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --bucket-counts D:\sojo_counts\SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin `
  --output-folder D:\BulletOu-snapshots\20260820 `
  --temp-folder C:\BulletOu-es-temp `
  --tag-prefix pair2-qloss `
  --factorizer pair `
  --positions-per-superbatch 40000000 `
  -- --lr 0.000030 --lr-min 0.000010 --wrm-in-offset 0 --wrm-target-offset 0 --lr-schedule step --optimizer ranger --optimizer-weight-decay 0.0 --batches-per-update 1 --sfnn-dirty-bucket-update --sfnn-saturation-penalty 1e-7
```

画面上では、`[GEN START]`、`[CAND 001 START]`、`[CAND 001 END]`、`[BEAM]`、`[ACCEPT]`、`[SAVE]` のように節目が色付きで出ます。止めるなら `[SAVE]` または `[SAFE TO STOP]` の直後が安全です。`current/` は accept ごとに更新されるので、`accepted-checkpoints/` に保存されていない地点からも runner の `--resume` は可能です。
