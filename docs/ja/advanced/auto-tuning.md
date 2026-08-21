# ESによる自動調整

<a href="../../en/advanced/auto-tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

factorizer の alpha や count confidence は、組み合わせが多く、手で総当たりすると時間がかかります。すでに良い checkpoint がある場合は、`es_local_runner.py` で beam search 風の ES (evolution strategy) を回せます。

この runner は `parameters.json` を現在値として扱います。各 generation で複数 candidate を作り、candidate ごとに指定 sb だけ学習します。途中段階で成績の悪い candidate を捨て、最後に残った candidate の NN 重みとハイパーパラメーターを次の現在値にします。

重要な点は次の通りです。

- 採用された candidate のパラメーター値を、そのまま次の generation の開始値にします。
- 採用後にパラメーターだけを少し動かす処理はありません。
- 各 candidate は独立にランダム生成されます。
- `parameters.json` は accept ごとに書き戻されます。手で値を直したい場合は、停止してこのファイルを編集してから `--resume` します。

## `parameters.json`

`parameters.json` には、ES runner の設定とチューニング対象の現在値を書きます。

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
| `false` | `bulletou.exe` の通常学習で、`parameters.current` の値だけ使う |

`step` は candidate を作るときのランダム幅です。たとえば `pair.current = 0.3`, `pair.step = 0.02` なら、candidate の `pair` は `0.28` から `0.32` の範囲で作られます。`tune: false` の項目は固定されます。

`beam` は「何 sb 走らせた時点で何本残すか」です。上の例では、16本を開始し、8 sbで8本、16 sbで4本、24 sbで2本、32 sbで1本に絞ります。最後に残った1本だけが採用されます。

## 実行例

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

この使い方では、`parameters.*.current` だけが学習に反映されます。`step` や `tune` や `beam` は ES runner 用の設定なので、`bulletou.exe` は読みません。

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
