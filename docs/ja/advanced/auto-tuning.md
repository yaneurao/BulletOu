# ES による自動調整

<a href="../../en/advanced/auto-tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-2563EB?style=flat-square"></a>

このページでは、`es_local_runner.py` を使って SFNN の factorizer alpha と count confidence を自動調整する方法を説明します。

ここでいう ES は evolution strategy です。いくつかの候補を少し違うパラメーターで学習させ、設定した指標で良い候補を残します。勾配を推定して別の小さな更新を行う方式ではありません。最後に残った候補の NN 重みとパラメーター値を、そのまま次の世代の開始点にします。

## 使う JSON ファイル

runner は 2 つの JSON ファイルを使います。

| ファイル | 役割 |
| --- | --- |
| `es-settings.json` | ES の世代数、population、beam、調整するパラメーター、現在値を書く |
| `bulletou-settings.json` | 通常の `bulletou.exe` 学習オプションを書く |

`es-settings.json` の中に `bulletou-settings.json` の path を書きます。普段は runner に `--es-settings-file` だけを渡します。

```powershell
python .\es_local_runner.py --es-settings-file .\es-settings.json
```

同じ runner を再開するときは `--resume` を付けます。

```powershell
python .\es_local_runner.py --es-settings-file .\es-settings.json --resume
```

## `es-settings.json` の例

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
    "metric": "borda_count",
    "lower_is_better": true,
    "use_worker": true,
    "seed": 1,
    "save_rate": 1,
    "validation_rate": 1,
    "quantized_validation_rate": 1
  },
  "run": {
    "exe": "C:/shogi/YaneuraOuWorks/BulletOu/target/release/examples/bulletou.exe",
    "bulletou_settings_file": "./bulletou-settings.json",
    "base_checkpoint": "C:/shogi/YaneuraOuWorks/BulletOu/checkpoints/.../0256",
    "output_folder": "D:/BulletOu-snapshots/20260820",
    "temp_folder": "C:/BulletOu-es-temp",
    "tag_prefix": "pair2-es"
  },
  "parameters": {
    "shared": { "current": 1.0, "tune": false, "step": 0.0, "min": 0.0, "max": 100.0 },
    "king_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "progress_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_hand_pair": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_progress_pair": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_progress_pair": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "residual_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_axis_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_hand_pair_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "king_progress_pair_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 },
    "hand_progress_pair_count": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 }
  }
}
```

## `bulletou-settings.json` の例

`bulletou-settings.json` には、通常 `bulletou.exe` に渡す学習オプションを書きます。JSON のキーは CLI オプション名から `--` を外し、ハイフンをアンダースコアにした名前です。たとえば `--lr-min` は `lr_min` です。

```json
{
  "backend": "cuda-cpp",
  "teacher": "D:/sojoteam_datasets",
  "test_teacher": "C:/shogi/teacher/test/test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe",
  "test_positions": "all",
  "arch": "SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4",
  "sfnn_factorizer": "pair",
  "sfnn_bucket_counts": "D:/sojo_counts/SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4-count-all.bin",
  "positions_per_superbatch": 40000000,
  "lr": 0.000030,
  "lr_min": 0.000010,
  "lr_schedule": "step",
  "optimizer": "ranger",
  "optimizer_weight_decay": 0.0,
  "batches_per_update": 1,
  "wrm_in_offset": 0,
  "wrm_target_offset": 0,
  "sfnn_dirty_bucket_update": true,
  "sfnn_freeze_progress": true,
  "sfnn_saturation_penalty": 1e-7
}
```

ES 実行時は runner が候補ごとに次の値を決めます。そのため、これらは `bulletou-settings.json` には書かないでください。

| runner が決める項目 | 理由 |
| --- | --- |
| `initial_state`, `initial_dataloader_pos` | 候補ごとに開始 checkpoint が変わる |
| `output`, `output_folder`, `tag` | 候補ごとに出力先を分ける |
| `superbatches`, `max_epochs` | beam の stage ごとに学習 sb 数が変わる |
| `save_rate`, `validation_rate`, `quantized_validation_rate` | 候補評価用に runner が指定する |
| `sfnn_factorizer_alpha` | ES の `parameters` から候補ごとに作る |
| `sfnn_*_count_confidence` | ES の `parameters` から候補ごとに作る |

`progress` 付き arch で進行度パラメーターを固定したい場合は、`bulletou-settings.json` に `sfnn_freeze_progress: true` を入れます。進行度を固定すると validation cache を使いやすくなり、qvalid も軽くなります。

## `es` の項目

| 項目 | 意味 |
| --- | --- |
| `enabled` | `true` なら ES を実行する。`false` なら `parameters.current` だけを使って通常学習を 1 回起動する |
| `generations` | 世代数。1 世代ごとに 1 つの候補を採用する |
| `population` | 世代開始時に作る候補数 |
| `beam` | 何 sb 学習した時点で、候補をいくつ残すか |
| `metric` | 候補を比較する指標 |
| `lower_is_better` | 小さいほど良い指標なら `true`。`borda_count` では常に順位和が小さいほど良いので、この値は使われない |
| `use_worker` | 長寿命の `bulletou worker` を使う。省略時は `true` |
| `seed` | 候補生成用の乱数 seed |
| `save_rate` | 何回採用するごとに `accepted-checkpoints/` へ公開 checkpoint を保存するか |
| `validation_rate` | f32 validation 間隔。ES 有効時は候補ごと、`enabled: false` では通常学習に使う。`0` なら各 stage の末尾だけで測る。`-1` なら無効 |
| `quantized_validation_rate` | 量子化 validation 間隔。ES 有効時は候補ごと、`enabled: false` では通常学習に使う。`0` なら各 stage の末尾だけで測る。`-1` なら無効 |

`metric` が必要とする validation を `-1` にすることはできません。例えば `metric: "borda_count"` は f32/量子化の accuracy/loss をすべて使うので、両方の validation rate を有効にしてください。stage 末尾だけでよい場合は `0` を指定します。

`beam` は次のように読みます。

```json
"beam": [
  { "after_sbs": 8, "keep": 8 },
  { "after_sbs": 16, "keep": 4 },
  { "after_sbs": 24, "keep": 2 },
  { "after_sbs": 32, "keep": 1 }
]
```

この例では、16 候補で開始し、8 sb 後に 8 候補、16 sb 後に 4 候補、24 sb 後に 2 候補、32 sb 後に 1 候補へ絞ります。最後の `keep` は必ず `1` にしてください。

## `metric`

`metric` は主に次のどれかを使います。

| 値 | 意味 | 推奨する `lower_is_better` |
| --- | --- | --- |
| `quantized_value_loss` | 量子化後の validation loss | `true` |
| `quantized_value_accuracy` | 量子化後の validation accuracy | `false` |
| `test_value_loss` | f32 weight の validation loss | `true` |
| `test_value_accuracy` | f32 weight の validation accuracy | `false` |
| `borda_count` | 4 指標それぞれで順位を付け、順位和が最小の候補を選ぶ | `true` |

`quantized_value_loss` だけを見ると、`quantized_value_accuracy` や f32 側の指標と食い違うことがあります。その場合は `borda_count` が使えます。

`borda_count` は次の手順で候補を比較します。

1. `test_value_accuracy` が大きい順に順位を付ける。
2. `test_value_loss` が小さい順に順位を付ける。
3. `quantized_value_accuracy` が大きい順に順位を付ける。
4. `quantized_value_loss` が小さい順に順位を付ける。
5. 4 つの順位を合計し、合計がもっとも小さい候補を残す。

同点の場合は、その順位範囲の平均順位を使います。たとえば 2 位と 3 位が同点なら、どちらも 2.5 位として計算します。

`borda_count` を worker mode で使う場合、runner は候補ごとの state cache を `temp_folder` 側のdiskに保存します。全候補の順位和を計算したあと、残す候補だけをdisk cacheからworkerへ読み戻します。これにより、`population` 個ぶんの巨大な optimizer state をメインメモリに保持せず、採用候補を再学習する必要もありません。代わりに、候補ごとの一時 `state.bin` をSSDへ書き出すI/Oが発生します。

ある候補が、すでに評価済みの候補に4指標すべてで負けている場合、その候補が Borda で最良になることはありません。この場合は trial 終了時点でcache保存をスキップします。逆に、新しい候補が既存候補に4指標すべてで勝っている場合は、負けた既存候補のcacheをすぐ削除します。

## `run` の項目

| 項目 | 意味 |
| --- | --- |
| `exe` | 実行する `bulletou.exe` |
| `bulletou_settings_file` | 通常学習設定を書いた JSON |
| `base_checkpoint` | 最初に読む checkpoint。`state.bin` と `dataloader_pos.txt` が必要 |
| `output_folder` | runner root を作る親フォルダ |
| `temp_folder` | 候補の一時 checkpoint を作る場所。高速な SSD 推奨 |
| `tag_prefix` | runner root 名に使う名前 |

runner root は `output_folder/es-<tag_prefix>` です。

## `parameters` の書き方

`shared` を固定する場合、主な調整対象は 13 個です。axis alpha が 3 個、pair alpha が 3 個、axis count confidence が 3 個、pair count confidence が 3 個、residual count confidence が 1 個です。

各パラメーターは次の形で書きます。

```json
"king_axis": { "current": 1.0, "tune": true, "step": 0.005, "min": 0.0, "max": 100.0 }
```

| 項目 | 意味 |
| --- | --- |
| `current` | 現在値。候補はこの値の周辺から作られる |
| `tune` | `true` なら ES が動かす。`false` なら固定 |
| `step` | 乗算的な揺らし幅。候補値は `current * exp(random(-step, step))` で作られる |
| `min` | 下限 |
| `max` | 上限 |

`step` は加算幅ではありません。`step = 0.02` なら、おおむね `current` の ±2% の範囲からサンプリングします。`step = 0.10` なら、おおむね 0.90 倍から 1.11 倍です。`tune = true` のパラメーターは `current > 0` である必要があります。

候補が採用されると、その候補のパラメーター値が `current` に書き戻されます。再開時に `king_axis=...` のような長い値を手で転記する必要はありません。

## alpha パラメーター

alpha は factorizer 成分の強さです。

| キー | 意味 |
| --- | --- |
| `shared` | 全 bucket で共有する成分の強さ |
| `king_axis` | king bucket 軸の成分の強さ |
| `hand_axis` | hand bucket 軸の成分の強さ |
| `progress_axis` | progress bucket 軸の成分の強さ |
| `king_hand_pair` | king-hand pair 成分の強さ |
| `king_progress_pair` | king-progress pair 成分の強さ |
| `hand_progress_pair` | hand-progress pair 成分の強さ |

`shared` を動かすと全体の土台が動くので、最初は `shared = 1.0` で固定したほうが結果を読みやすいです。

`bulletou.exe` の `--sfnn-factorizer-alpha pair=...` は、3 つの pair alpha に同じ値を入れる短縮指定です。ES では `king_hand_pair`、`king_progress_pair`、`hand_progress_pair` を個別に調整します。

## count confidence パラメーター

count confidence は `bucket-count` で作った `count.bin` を使います。出現回数が少なく、信用しにくい成分を弱めるための係数です。

| キー | BulletOu のオプション | 意味 |
| --- | --- | --- |
| `residual_count` | `--sfnn-residual-count-gate-confidence` | bucket 固有 residual の count gate confidence |
| `king_axis_count` | `--sfnn-king-axis-count-confidence` | king-axis 用 |
| `hand_axis_count` | `--sfnn-hand-axis-count-confidence` | hand-axis 用 |
| `progress_axis_count` | `--sfnn-progress-axis-count-confidence` | progress-axis 用 |
| `king_hand_pair_count` | `--sfnn-king-hand-pair-count-confidence` | king-hand pair 用 |
| `king_progress_pair_count` | `--sfnn-king-progress-pair-count-confidence` | king-progress pair 用 |
| `hand_progress_pair_count` | `--sfnn-hand-progress-pair-count-confidence` | hand-progress pair 用 |

`0.0` なら、その confidence は無効です。大きい値にすると、十分な出現回数がある成分だけを強く信用します。

通常の `bulletou.exe` には axis 系や pair 系をまとめて指定する共通オプションもあります。ただし ES の `parameters` では、どの成分を動かしているのかを明確にするため、上の表にある個別項目だけを扱います。

## ES を回さず `current` の値だけ使う

ES で調整済みの `parameters.current` だけを使い、ES 自体は回したくない場合は、`es.enabled` を `false` にします。

```json
"es": {
  "enabled": false
}
```

その状態で runner を 1 回起動します。

```powershell
python .\es_local_runner.py --es-settings-file .\es-settings.json
```

この使い方では、13 個の `parameters.current` を手で `bulletou-settings.json` に転記する必要はありません。runner が `parameters.current` を読み、`--sfnn-factorizer-alpha` と count confidence オプションに変換して `bulletou.exe` に渡します。

このモードでは、`superbatches` は `beam` の最後の `after_sbs`、`max_epochs` は `generations` から runner が補います。`validation_rate` と `quantized_validation_rate` は `es-settings.json` の `es` に書いた値を使います。`lr` や `save_rate` などの通常学習項目は `bulletou-settings.json` に書きます。

`enabled: false` では ES の候補生成、worker cache、snapshot保持は使いません。runner は `bulletou.exe` を1回起動し、`parameters.current` をCLI引数に変換して渡すだけです。stdoutログは `output_folder/es-<tag_prefix>/logs/bulletou-settings-run.stdout.log` に書かれます。

## `bulletou.exe --settings-file`

`bulletou.exe` も settings JSON を直接読めます。

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

CLI で明示した値は settings JSON より優先されます。

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --lr 0.000050
```

## 出力先

runner root は `output_folder/es-<tag_prefix>` です。

| path | 役割 |
| --- | --- |
| `current/` | 最新の採用 checkpoint。`--resume` はここから続ける |
| `accepted-checkpoints/sbXXXXXXXX/` | `save_rate` 回採用するごとに保存される公開 checkpoint |
| `summary-learn.log` | すべての候補の結果 |
| `accepted-summary-learn.log` | 採用された候補だけの結果 |
| `parameters-history.jsonl` | 採用されたパラメーターの履歴 |
| `runner-state.json` | resume 用の状態 |
| `logs/` | 候補ごとの stdout log |
| `temp/` | `temp_folder` 未指定時の一時 checkpoint |

ES の `summary-learn.log` と `accepted-summary-learn.log` は、通常の BulletOu の `summary-learn.log` と同じく `test_value_accuracy`, `test_value_loss`, `quantized_value_accuracy`, `quantized_value_loss` の順で指標を書きます。

runner は `current/` と `accepted-checkpoints/sbXXXXXXXX/` に、その時点の `es-settings.json` と `bulletou-settings.json` をコピーします。あとから「この checkpoint はどの条件で作ったのか」を確認できます。

手動停止するなら、`[SAFE TO STOP]` が表示された直後が安全です。`[BEAM END]` は候補の絞り込みが終わったという意味で、公開 checkpoint の保存完了を意味しません。

ES 実行中は、標準では `bulletou worker` を 1 回だけ起動し、その中で候補を試します。これにより、CUDA context、validation cache、qvalid cache、worker warmup を候補ごとに作り直す時間を避けられます。

`use_worker` を `false` にした場合や、runner が worker では安全に扱えない beam 構成を検出した場合は、候補ごとに短い `bulletou.exe` job を起動します。その場合、子プロセスの `[epoch] start epoch 1/1` は「ES 全体の epoch」ではありません。runner は画面出力に `[G0002 S0008 C001]` のような prefix を付けます。これは「generation 2、8 sb stage、candidate 1」の意味です。

通常学習で `bulletou.exe --settings-file` を使った場合も、各 BulletOu checkpoint には `bulletou-settings.json` がコピーされます。
