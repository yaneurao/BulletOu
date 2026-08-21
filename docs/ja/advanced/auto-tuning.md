# ES による自動調整

<a href="../../en/advanced/auto-tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

このページでは、`es_local_runner.py` を使って factorizer の強さや count confidence を自動調整する方法を説明します。

ここでいう ES は evolution strategy です。複数の候補を少しずつ違うパラメーターで学習させ、検証 loss が良い候補だけを残します。勾配推定によって別の微小更新を行う方式ではありません。最後に残った候補の NN 重みとパラメーター値を、そのまま次の世代の開始点にします。

## 使う JSON ファイル

自動調整では、設定を 2 つの JSON に分けます。

| ファイル | 役割 |
| --- | --- |
| `es-settings.json` | ES の世代数、population、beam、調整するパラメーター、現在値を書く |
| `bulletou-settings.json` | `bulletou.exe` に渡す通常の学習設定を書く |

`es-settings.json` の中に `bulletou-settings.json` の path を書きます。runner の実行時は `--es-settings-file` だけ指定すればよいです。

```powershell
python .\es_local_runner.py --es-settings-file .\es-settings.json
```

再開するときは同じ `es-settings.json` を指定して `--resume` を付けます。

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
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "seed": 1,
    "save_rate": 1,
    "candidate_validation_rate": 1,
    "candidate_quantized_validation_rate": 1
  },
  "run": {
    "exe": "C:/shogi/YaneuraOuWorks/BulletOu/target/release/examples/bulletou.exe",
    "bulletou_settings_file": "./bulletou-settings.json",
    "base_checkpoint": "C:/shogi/YaneuraOuWorks/BulletOu/checkpoints/.../0256",
    "output_folder": "D:/BulletOu-snapshots/20260820",
    "temp_folder": "C:/BulletOu-es-temp",
    "tag_prefix": "pair2-qloss"
  },
  "parameters": {
    "shared": { "current": 1.0, "tune": false, "step": 0.0, "min": 0.0, "max": 100.0 },
    "king_axis": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "hand_axis": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "progress_axis": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "pair": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "residual_count": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "king_axis_count": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "king_hand_pair_count": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "king_progress_pair_count": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 },
    "hand_progress_pair_count": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 }
  }
}
```

## `bulletou-settings.json` の例

`bulletou-settings.json` には、通常 `bulletou.exe` に書く学習条件を書きます。JSON のキーは CLI の `--` を外し、ハイフンをアンダースコアにした名前です。たとえば `--lr-min` は `lr_min` です。

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
  "sfnn_saturation_penalty": 1e-7
}
```

ES 実行時は runner が候補ごとに次の値を決めます。そのため、`bulletou-settings.json` には書かないでください。

| runner が決める項目 | 理由 |
| --- | --- |
| `initial_state`, `initial_dataloader_pos` | 候補ごとの開始 checkpoint が違う |
| `output`, `output_folder`, `tag` | 候補ごとの出力先を runner が作る |
| `superbatches`, `max_epochs` | beam の stage ごとに学習 sb 数が違う |
| `save_rate`, `validation_rate`, `quantized_validation_rate` | 候補評価用に runner が指定する |
| `sfnn_factorizer_alpha` | ES の `parameters` から候補ごとに作る |
| `sfnn_*_count_confidence` | ES の `parameters` から候補ごとに作る |

## `es` の項目

| 項目 | 意味 |
| --- | --- |
| `enabled` | `true` なら ES を実行する。`false` なら `parameters.current` だけを使って 1 回だけ通常学習を起動する |
| `generations` | 世代数。1 世代ごとに 1 つの候補が採用される |
| `population` | 世代開始時に作る候補数 |
| `beam` | 何 sb 学習した時点で、何個の候補を残すか |
| `metric` | 候補を比較する指標 |
| `lower_is_better` | 指標が小さいほど良いなら `true` |
| `seed` | 候補生成の乱数 seed |
| `save_rate` | 何回採用するごとに `accepted-checkpoints/` へ保存するか |
| `candidate_validation_rate` | 候補学習中の f32 validation 間隔 |
| `candidate_quantized_validation_rate` | 候補学習中の量子化 validation 間隔 |

`beam` は次のように読みます。

```json
"beam": [
  { "after_sbs": 8, "keep": 8 },
  { "after_sbs": 16, "keep": 4 },
  { "after_sbs": 24, "keep": 2 },
  { "after_sbs": 32, "keep": 1 }
]
```

この例では 16 候補で開始し、8 sb 後に 8 候補、16 sb 後に 4 候補、24 sb 後に 2 候補、32 sb 後に 1 候補へ絞ります。最後の `keep` は必ず `1` にしてください。

`metric` は主に次のどれかを使います。

| 値 | 意味 | 推奨する `lower_is_better` |
| --- | --- | --- |
| `quantized_value_loss` | 量子化後の validation loss | `true` |
| `quantized_value_accuracy` | 量子化後の validation accuracy | `false` |
| `test_value_loss` | f32 weight の validation loss | `true` |
| `test_value_accuracy` | f32 weight の validation accuracy | `false` |

棋力計測に近い候補を探す目的なら、まず `quantized_value_loss` を見るのが無難です。

## `run` の項目

| 項目 | 意味 |
| --- | --- |
| `exe` | 実行する `bulletou.exe` |
| `bulletou_settings_file` | 通常学習設定を書いた JSON |
| `base_checkpoint` | 最初に読む checkpoint。`state.bin` と `dataloader_pos.txt` が必要 |
| `output_folder` | runner root を作る親フォルダ |
| `temp_folder` | 候補の一時 checkpoint を作る場所。高速な SSD を推奨 |
| `tag_prefix` | runner root 名に使う名前 |

runner root は `output_folder/es-<tag_prefix>` になります。

## `parameters` の共通フィールド

各パラメーターは次の形で書きます。

```json
"pair": { "current": 1.0, "tune": true, "step": 0.05, "min": 0.0, "max": 100.0 }
```

| フィールド | 意味 |
| --- | --- |
| `current` | 現在値。候補はこの値の周辺に作られる |
| `tune` | `true` なら ES が動かす。`false` なら固定 |
| `step` | 候補を作る倍率の幅。候補は `current * exp(random(-step, step))` で作る |
| `min` | 下限 |
| `max` | 上限 |

`step` は加算幅ではありません。`step = 0.02` なら候補はおよそ `current` の ±2% の範囲、`step = 0.10` ならおよそ 0.90 倍から 1.11 倍の範囲になります。`tune = true` にするパラメーターは `current > 0` にしてください。

採用された候補の値は `current` に書き戻されます。再開時に手で `king_axis=...` のような長い値を打ち直す必要はありません。

## alpha パラメーター

alpha は factorizer 成分をどれだけ使うかを決める倍率です。

| キー | 意味 |
| --- | --- |
| `shared` | 全 bucket で共有する成分の強さ |
| `king_axis` | king bucket 軸の成分の強さ |
| `hand_axis` | hand bucket 軸の成分の強さ |
| `progress_axis` | progress bucket 軸の成分の強さ |
| `pair` | king-hand、king-progress、hand-progress の pair 成分の強さ |

`shared` を動かすと全体の土台が変わります。比較実験では、まず `shared = 1.0` 固定にして、axis や pair を動かすほうが結果を読みやすいです。

## count confidence パラメーター

count confidence は、`bucket-count` で作った `count.bin` を使って、出現回数が少ない成分を弱めるための値です。

| キー | 対応する BulletOu オプション | 意味 |
| --- | --- | --- |
| `residual_count` | `--sfnn-residual-count-confidence` | bucket 固有 residual の count confidence |
| `king_axis_count` | `--sfnn-king-axis-count-confidence` | king axis 専用 |
| `hand_axis_count` | `--sfnn-hand-axis-count-confidence` | hand axis 専用 |
| `progress_axis_count` | `--sfnn-progress-axis-count-confidence` | progress axis 専用 |
| `king_hand_pair_count` | `--sfnn-king-hand-pair-count-confidence` | king-hand pair 専用 |
| `king_progress_pair_count` | `--sfnn-king-progress-pair-count-confidence` | king-progress pair 専用 |
| `hand_progress_pair_count` | `--sfnn-hand-progress-pair-count-confidence` | hand-progress pair 専用 |

値が `0.0` なら、その count confidence は無効です。値を大きくすると、十分な出現回数がない成分をより強く抑えます。

通常の `bulletou.exe` には axis 系 / pair 系をまとめて指定する共通オプションもあります。ただし ES の `parameters` では、どの成分を動かしているのかを明確にするため、上の表にある個別項目だけを扱います。

## `es.enabled=false` で current 値だけ使う

ES を回さず、`parameters.current` の値だけを使いたい場合は、`es.enabled` を `false` にします。

```json
"es": {
  "enabled": false
}
```

この状態で runner を起動すると、ES は行わず、次のような 1 回だけの `bulletou.exe` 実行になります。

```powershell
python .\es_local_runner.py --es-settings-file .\es-settings.json
```

このとき `bulletou-settings.json` には、通常学習に必要な `superbatches`、`max_epochs`、`save_rate`、`validation_rate` なども書いてください。

## `bulletou.exe --settings-file`

`bulletou.exe` 単体でも JSON 設定を読めます。

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

同じオプションを CLI にも書いた場合は、CLI に書いた値が優先されます。

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --lr 0.000050
```

## 出力フォルダ

ES runner の出力は `output_folder/es-<tag_prefix>` に作られます。

| path | 内容 |
| --- | --- |
| `current/` | 最新の採用 checkpoint。`--resume` はここから再開する |
| `accepted-checkpoints/sbXXXXXXXX/` | `save_rate` 回採用ごとの公開 checkpoint |
| `summary-learn.log` | すべての候補の結果 |
| `accepted-summary-learn.log` | 採用された候補だけの結果 |
| `parameters-history.jsonl` | 採用時のパラメーター履歴 |
| `runner-state.json` | runner の再開情報 |
| `logs/` | 候補ごとの stdout |
| `temp/` | 候補の一時 checkpoint。`temp_folder` 指定時はそちらに作る |

ES runner が保存する `current/` と `accepted-checkpoints/sbXXXXXXXX/` には、その時点の `es-settings.json` と `bulletou-settings.json` がコピーされます。あとから「この checkpoint はどの条件で作ったのか」を確認できます。

`bulletou.exe --settings-file` で通常学習した checkpoint には、`bulletou-settings.json` がコピーされます。
