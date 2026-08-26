# population search による自動調整

<a href="../../en/advanced/auto-tuning.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-2563EB?style=flat-square"></a>

このページでは、`tuning_parameters.py` を使って SFNN の factorizer alpha と count confidence を自動調整する方法を説明します。

ここでいう population search は、複数の候補を実際に短く学習させ、その結果を見て次に使うパラメーターを決める方式です。各世代で `population` 本の候補を少し違うパラメーターで学習させ、設定した指標で比較します。世代の最後に、選ばれたパラメーターで commit run を1回実行し、その checkpoint を次の世代の開始点にします。勾配を推定して別の小さな更新を行う方式ではありません。

## 使う JSON ファイル

runner は 2 つの JSON ファイルを使います。

| ファイル | 役割 |
| --- | --- |
| `tuning-settings.json` | population search の世代数、population、trial 長、調整するパラメーター、現在値を書く |
| `bulletou-settings.json` | 通常の `bulletou.exe` 学習オプションを書く |

`tuning-settings.json` の中に `bulletou-settings.json` の path を書きます。普段は runner に `--settings-file` だけを渡します。

```powershell
python .\tuning_parameters.py --settings-file .\tuning-settings.json
```

同じ runner を再開するときは `--resume` を付けます。

```powershell
python .\tuning_parameters.py --settings-file .\tuning-settings.json --resume
```

## `tuning-settings.json` の例

```json
{
  "version": 1,
  "tuning": {
    "enabled": true,
    "generations": 100,
    "population": 100,
    "trial_sbs": 4,
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "use_worker": true,
    "seed": 1,
    "sampler": "tpe",
    "tpe_startup_trials": 16,
    "tpe_good_fraction": 0.25,
    "tpe_bandwidth": 0.05,
    "max_parameter_change_ratio": 2.0,
    "commit_source": "best",
    "save_rate": 1,
    "validation_rate": 1,
    "quantized_validation_rate": 1
  },
  "run": {
    "exe": "C:/shogi/YaneuraOuWorks/BulletOu/target/release/examples/bulletou.exe",
    "bulletou_settings_file": "./bulletou-settings.json",
    "base_checkpoint": "C:/shogi/YaneuraOuWorks/BulletOu/checkpoints/.../0256",
    "output_folder": "D:/BulletOu-snapshots/20260820",
    "temp_folder": "C:/BulletOu-tuning-temp",
    "tag_prefix": "pair2-es"
  },
  "parameters": {
    "shared": 1.0,
    "king_axis": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_axis": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "progress_axis": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_hand_pair": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_progress_pair": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_progress_pair": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "residual_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_axis_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_hand_pair_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "king_progress_pair_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "hand_progress_pair_count": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 },
    "lr": { "current": 0.000030, "tune": true, "min": 0.000001, "max": 0.001, "log": true },
    "lr_min": { "current": 0.000010, "tune": true, "min": 0.000001, "max": 0.001, "log": true }
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
  "lr_schedule": "step",
  "optimizer": "ranger",
  "optimizer_weight_decay": 0.0,
  "batches_per_update": 1,
  "wrm_in_offset": 0,
  "wrm_target_offset": 0,
  "sfnn_dirty_bucket_update": true,
  "sfnn_freeze_progress": true,
  "teacher_memory_cache_sbs": 4,
  "sfnn_saturation_penalty": 1e-7
}
```

population search 実行時は runner が候補ごとに次の値を決めます。そのため、これらは `bulletou-settings.json` には書かないでください。

| runner が決める項目 | 理由 |
| --- | --- |
| `initial_state`, `initial_dataloader_pos` | 候補ごとに開始 checkpoint が変わる |
| `output`, `output_folder`, `tag` | 候補ごとに出力先を分ける |
| `superbatches`, `max_epochs` | 候補 trial の長さを runner が決める |
| `save_rate`, `validation_rate`, `quantized_validation_rate` | 候補評価用に runner が指定する |
| `sfnn_factorizer_alpha` | population search の `parameters` から候補ごとに作る |
| `sfnn_*_count_confidence` | population search の `parameters` から候補ごとに作る |
| `lr`, `lr_min` | 任意。`tuning-settings.json` の `parameters` に書いた場合、runner が `--lr` / `--lr-min` として渡す。探索中の採用値は `runner-state.json` に保存する |

`progress` 付き arch で進行度パラメーターを固定したい場合は、`bulletou-settings.json` に `sfnn_freeze_progress: true` を入れます。進行度を固定すると validation cache を使いやすくなり、qvalid も軽くなります。

`teacher_memory_cache_sbs` は、worker process の RAM に教師データを保持するための設定です。値は「何 superbatch 分を RAM に保持するか」です。たとえば `4` なら、trial で使う 4 sb 分の `.psv` / `.bin` 教師レコードを RAM に読み込み、同じ worker process 内の候補評価で再利用します。

この cache は worker process の中だけに存在します。worker を終了すると消えます。SSD や `temp_folder` に教師データを退避する機能ではありません。

注意点は次の通りです。

- `.psv` / `.bin` 教師だけ対応します。`.hcpe` / `.hcpe3` / `.pack` では使えません。
- `trial_sbs: 4` なら `teacher_memory_cache_sbs` は 4 以上にするとcacheが効きます。足りない場合は警告を出し、そのtrialではcacheを使わず通常のstreaming読み込みに戻ります。
- `population: 0` の generation は候補比較ではなく1本の長い定着学習なので、runnerは teacher memory cache を自動的に無効化します。
- 1 sb が `610 * 65536` 局面なら、4 sb は約 1.6 億局面で、RAM 使用量は約 6 GiB です。
- worker mode で効く機能です。候補ごとに `bulletou.exe` を起動し直す方式では、process 終了時に cache も消えるため効果がありません。

## `tuning` の項目

| 項目 | 意味 |
| --- | --- |
| `enabled` | `true` なら population search を実行する。`false` なら `parameters.current` だけを使って通常学習を 1 回起動する |
| `generations` | 世代数。1 世代ごとに 1 つの候補を採用する |
| `population` | 1 世代で試す候補数。例では 100 本の候補を試す |
| `trial_sbs` | 候補 1 本あたり何 sb 学習するか。例では 1 trial = 4 sb |
| `metric` | 候補を比較する指標 |
| `lower_is_better` | 小さいほど良い指標なら `true`。`borda_count` では常に順位和が小さいほど良いので、この値は使われない |
| `use_worker` | 長寿命の `bulletou worker` を使う。省略時は `true` |
| `seed` | 候補生成用の乱数 seed |
| `sampler` | `"tpe"` または `"random"`。通常は `"tpe"` を使う |
| `tpe_startup_trials` | TPE の分布推定に使う最低 trial 数。generation 1 ではこの本数までは広くランダム探索する |
| `tpe_good_fraction` | TPE が上位何割を「良かった候補」として扱うか |
| `tpe_bandwidth` | TPE の分布幅の下限。大きいほど候補が広めに散る |
| `max_parameter_change_ratio` | generation 2 以降で、候補値を現在採用中の値から何倍まで動かしてよいか。`2.0` なら `current/2` から `current*2` に制限する |
| `commit_source` | 世代末の commit run に使う値。`"best"` は実測1位、`"recommended"` は上位 trial から推定した値 |
| `save_rate` | 何回採用するごとに `accepted-checkpoints/` へ公開 checkpoint を保存するか |
| `validation_rate` | f32 validation 間隔。population search 有効時は候補ごと、`enabled: false` では通常学習に使う。`0` なら各 trial の末尾だけで測る。`-1` なら無効 |
| `quantized_validation_rate` | 量子化 validation 間隔。population search 有効時は候補ごと、`enabled: false` では通常学習に使う。`0` なら各 trial の末尾だけで測る。`-1` なら無効 |

`metric` が必要とする validation を `-1` にすることはできません。例えば `metric: "borda_count"` は f32/量子化の accuracy/loss をすべて使うので、両方の validation rate を有効にしてください。trial 末尾だけでよい場合は `0` を指定します。

`trial_sbs` は候補 1 本あたりの学習長です。`trial_sbs: 4` なら、各候補を 4 sb だけ学習し、`population` 本すべてが終わってから指標で順位を付けます。

`max_parameter_change_ratio` は、学習が進んだ checkpoint に対してパラメーターを急に大きく動かしすぎるのを防ぐ安全弁です。scratch から始める generation 1 では広く探索し、commit checkpoint ができた後の generation では現在値の近くを探索します。現在値が `0` のパラメーターは `0` のままにします。`0` は成分無効化という特別な意味を持つため、通常は factorizer alpha / count confidence の `min` を `0.1` 以上にしておくほうが扱いやすいです。

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

runner root は `output_folder/tuning-<tag_prefix>` です。

`run.base_checkpoint` は最初の開始点です。1 generation ごとの採用 checkpoint は `runner-state.json` の `current_checkpoint` に記録されます。`--resume` 時は `run.base_checkpoint` ではなく `current_checkpoint` から再開します。`tuning-settings.json` の `run.base_checkpoint` は自動では書き換えません。

## `parameters` の書き方

`shared` を固定する場合、主な調整対象は 13 個です。axis alpha が 3 個、pair alpha が 3 個、axis count confidence が 3 個、pair count confidence が 3 個、residual count confidence が 1 個です。学習率も runner に保持させたい場合は、これに加えて `lr` / `lr_min` を任意で書けます。

各パラメーターは次の形で書きます。

```json
"king_axis": { "current": 1.0, "tune": true, "min": 0.1, "max": 10.0 }
```

| 項目 | 意味 |
| --- | --- |
| `current` | 初回開始値。`enabled: false` ではこの値をそのまま使う。探索再開時の採用値は `runner-state.json` から読む |
| `tune` | `true` なら population search が動かす。`false` なら固定 |
| `min` | 下限 |
| `max` | 上限 |
| `log` | `true` なら対数空間で探索する。学習率のように桁で効く値に使う |

候補値は、基本的には `min` から `max` の範囲で作られます。`log: true` のパラメーターは対数空間で作られます。generation 2 以降で `max_parameter_change_ratio` を指定している場合は、さらに現在採用中の値の近くに制限されます。

runner は `tuning-settings.json` 自体には書き戻しません。採用中の値は `runner-state.json` に保存され、確認用の値は `recommended-parameters.json` に書かれます。再開時に `king_axis=...` のような長い値を手で転記する必要はありません。

学習率も runner に現在値として保持させたい場合は、`lr` と `lr_min` を `parameters` に書けます。

```json
"lr": { "current": 0.000030, "tune": true, "min": 0.000001, "max": 0.001, "log": true },
"lr_min": { "current": 0.000010, "tune": true, "min": 0.000001, "max": 0.001, "log": true }
```

この 2 つを `parameters` に書いた場合、runner は各 `bulletou.exe` 起動時に `--lr` と `--lr-min` を追加します。そのため、`bulletou-settings.json` に `lr` / `lr_min` が残っていても、population search 側の値が上書きします。`parameters` に書かなければ、学習率は `bulletou-settings.json` 側の固定値を使います。

`lr` と `lr_min` の両方を調整対象にする場合、runner は `lr_min <= lr` になるように `lr_min` の候補をその trial の `lr` 以下に制限して作ります。

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

`bulletou.exe` の `--sfnn-factorizer-alpha pair=...` は、3 つの pair alpha に同じ値を入れる短縮指定です。population search では `king_hand_pair`、`king_progress_pair`、`hand_progress_pair` を個別に調整します。

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

通常の `bulletou.exe` には axis 系や pair 系をまとめて指定する共通オプションもあります。ただし population search の `parameters` では、どの成分を動かしているのかを明確にするため、上の表にある個別項目だけを扱います。

## population search を回さず `current` の値だけ使う

population search で調整済みの `parameters.current` だけを使い、population search 自体は回したくない場合は、`tuning.enabled` を `false` にします。

```json
"tuning": {
  "enabled": false
}
```

その状態で runner を 1 回起動します。

```powershell
python .\tuning_parameters.py --settings-file .\tuning-settings.json
```

この使い方では、13 個の `parameters.current` を手で `bulletou-settings.json` に転記する必要はありません。runner が `parameters.current` を読み、`--sfnn-factorizer-alpha` と count confidence オプションに変換して `bulletou.exe` に渡します。

このモードでは、`superbatches` は `trial_sbs`、`max_epochs` は `generations` から runner が補います。`validation_rate` と `quantized_validation_rate` は `tuning-settings.json` の `tuning` に書いた値を使います。`lr` / `lr_min` は `parameters` に書いていれば runner が現在値として渡し、書いていなければ `bulletou-settings.json` の値を使います。`save_rate` は `accepted-checkpoints/` に公開 checkpoint を何 epoch ごとに残すかを表します。

`enabled: false` では population search の候補生成、worker cache、snapshot保持は使いません。runner は `bulletou.exe` を1回起動し、`parameters.current` をCLI引数に変換して渡すだけです。stdoutログは `output_folder/tuning-<tag_prefix>/logs/bulletou-settings-run.stdout.log` に書かれます。

普通学習の出力本体は `output_folder/tuning-<tag_prefix>/bulletou-run/` に作られます。runner はその `summary-learn.log` を読み、`output_folder/tuning-<tag_prefix>/summary-learn.log` と `accepted-summary-learn.log` にも同じ指標を反映します。保存された checkpoint は `accepted-checkpoints/sbXXXXXXXX/` に同期され、最新 checkpoint は `current/` にコピーされます。

そのあと `enabled` を `true` に戻して `--resume` すると、population search は更新済みの `current/` から続行します。`enabled=false` で進めたぶんの `accepted_sbs` と `generation` も `runner-state.json` に保存されるので、公開 checkpoint の番号も巻き戻りません。

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

runner root は `output_folder/tuning-<tag_prefix>` です。

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

population search の `summary-learn.log` と `accepted-summary-learn.log` は、通常の BulletOu の `summary-learn.log` と同じく `test_value_accuracy`, `test_value_loss`, `quantized_value_accuracy`, `quantized_value_loss` の順で指標を書きます。

runner は `current/` と `accepted-checkpoints/sbXXXXXXXX/` に、その時点の `tuning-settings.json` と `bulletou-settings.json` をコピーします。あとから「この checkpoint はどの条件で作ったのか」を確認できます。

手動停止するなら、`[SAFE TO STOP]` が表示された直後が安全です。`[GEN RANK]` はその世代の候補評価と順位付けが終わったという意味で、公開 checkpoint の保存完了を意味しません。

population search 実行中は、標準では `bulletou worker` を 1 回だけ起動し、その中で候補を試します。これにより、CUDA context、validation cache、qvalid cache、worker warmup を候補ごとに作り直す時間を避けられます。

`use_worker` を `false` にした場合は、候補ごとに短い `bulletou.exe` job を起動します。その場合、子プロセスの `[epoch] start epoch 1/1` は「population search 全体の epoch」ではありません。runner は画面出力に `[G0002 S0032 C001]` のような prefix を付けます。これは「generation 2、32 sb trial、candidate 1」の意味です。

通常学習で `bulletou.exe --settings-file` を使った場合も、各 BulletOu checkpoint には `bulletou-settings.json` がコピーされます。
