# 固定長 trial によるパラメーター調整

`tuning_parameters.py` は、同じ開始地点から短い学習 trial を多数走らせ、`lr` / `lr_min` / factorizer / count confidence のよさそうな値を探すための runner です。

外部 package は使っていません。BulletOu 用の軽量な TPE-style sampler です。最初はランダムに試し、その後は良かった trial の分布と悪かった trial の分布を比べ、有望そうな範囲を重点的に試します。

## 何をする runner か

- 各 trial は scratch、または指定した同じ checkpoint から始まります。
- 1 trial の長さは `tuning.trial_sbs` で指定します。数値または配列で書けます。
- trial 中はパラメーターを変えません。
- generation ごとに `population` 本の trial を走らせ、指定した metric が良いものを記録します。
- `recommended-parameters.json` に、上位 trial から推定した推奨パラメーターを書き出します。

短い trial では「たまたま良かった1本」をそのまま採用するとノイズを拾いやすいです。そのため、最終的に使う値は `best_observed` だけでなく、`recommended` も確認してください。

## `log` とは何か

`parameters` の各項目には、必要なら `log: true` を書けます。

`log: true` は、値を倍率ベースで探索する指定です。たとえば `min=0.000001, max=0.001` の学習率では、`0.000001`、`0.00001`、`0.0001`、`0.001` のような桁の違いを自然に探索できます。

一方で、`log: true` は `min=0` と両立しません。`log(0)` が存在しないためです。

factorizer alpha や count confidence で `0` を許したい場合は、`log` を書かないか、`log: false` にしてください。`min` が `0` 以下なら、runner は省略時に線形探索として扱います。

## generation と TPE sampler

`tuning_parameters.py` は generation 単位で候補を作ります。同じ generation 内の trial 結果は、その generation の候補生成には使いません。次の generation から反映します。

```json
"tuning": {
  "generations": 3,
  "population": [100, 50],
  "trial_sbs": [4, 8],
  "sampler": "tpe"
}
```

この例では、

- generation 1: 100 trial、各 trial 4 sb
- generation 2 以降: 50 trial、各 trial 8 sb

になります。`generations` を省略した場合は、`population` / `trial_sbs` の配列長から generation 数を決めます。`population` や `trial_sbs` の配列が `generations` より短い場合は、最後の値を使い続けます。

TPE-style sampler は、前 generation までの結果を使って次の候補を作ります。完了済み trial を metric で並べ、上位を「良かった候補」、残りを「悪かった候補」として、各パラメーターの分布を作ります。そのうえで、良かった候補の分布に近く、悪かった候補の分布から遠い値を優先してサンプルします。

## sampler の項目

これらは学習そのもののパラメーターではなく、「次にどの候補を試すか」を決める sampler 側の設定です。

| 項目 | 意味 | 省略時 |
| --- | --- | --- |
| `sampler` | `"tpe"` または `"random"`。通常は `"tpe"` を使います。 | `"tpe"` |
| `tpe_startup_trials` | TPE を始める前に必要な完了済み trial 数。この本数に到達するまでは、探索範囲全体からランダムにサンプルします。 | `16` |
| `tpe_good_fraction` | TPE が上位何割を「良かった候補」として使うか。`0.25` なら上位25%を使います。 | `0.25` |
| `tpe_bandwidth` | TPE の KDE 幅の下限です。大きいほど候補が広めに散り、小さいほど観測された良い候補の近くに寄ります。 | `0.15` |

## 設定例

```json
{
  "version": 1,
  "tuning": {
    "generations": 3,
    "population": [100, 50],
    "trial_sbs": [4, 8],
    "sampler": "tpe",
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "seed": 20260825,
    "tpe_startup_trials": 16,
    "tpe_good_fraction": 0.25,
    "tpe_bandwidth": 0.15,
    "validation_rate": 0,
    "quantized_validation_rate": 0,
    "keep_all_trials": false
  },
  "run": {
    "exe": "C:/shogi/YaneuraOuWorks/BulletOu/target/release/examples/bulletou.exe",
    "bulletou_settings_file": "./bulletou-settings.json",
    "base_checkpoint": null,
    "output_folder": "D:/BulletOu-snapshots/20260825",
    "temp_folder": "D:/BulletOu-snapshots/20260825",
    "tag_prefix": "tuning-scratch-4sb"
  },
  "parameters": {
    "lr": { "current": 0.0003, "tune": true, "min": 0.000001, "max": 0.001, "log": true },
    "lr_min": { "current": 0.0001, "tune": true, "min": 0.000001, "max": 0.001, "log": true },

    "shared": 1.0,

    "king_axis": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_axis": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "progress_axis": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },

    "king_hand_pair": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_progress_pair": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_progress_pair": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },

    "residual_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_axis_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_axis_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "progress_axis_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_hand_pair_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "king_progress_pair_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 },
    "hand_progress_pair_count": { "current": 1.0, "tune": true, "min": 0, "max": 10.0 }
  }
}
```

`lr` と `lr_min` を両方 tune する場合、runner は `lr_min <= lr` になるように `lr_min` の上限をその trial の `lr` 以下に制限してサンプルします。

## 教師データをRAMに保持する場合

`bulletou.exe worker` では、PSV互換の教師データ (`.psv` / `.bin`) を worker process のRAMに保持できます。
同じ worker process の中で複数の trial を走らせる場合、USB HDDなどから同じ教師範囲を何度も読み直す無駄を避けられます。

`bulletou-settings.json` に次のように書きます。

```json
{
  "teacher_memory_cache_sbs": 4
}
```

`4` は「4 superbatch 分をRAMに保持する」という意味です。1 superbatch が `610 * 65536` 局面なら、4 superbatch は約 1.6 億局面で、RAM使用量は約 6 GiB です。

注意点:

- このcacheは worker process のメモリ上だけにあります。workerを終了すると消えます。
- `.psv` / `.bin` 専用です。`.hcpe3` や `.pack` には使えません。
- trial が4sbなら `teacher_memory_cache_sbs` は4以上にしてください。足りない場合はエラーになります。
- `tuning_parameters.py` は標準で worker を使います。`tuning.use_worker: false` にした場合は、trialごとに `bulletou.exe` を起動するため、このcacheは効きません。
- cache が有効な場合、起動時に `[CACHE] teacher_memory_cache_sbs=...` が表示され、worker側のログに `worker teacher memory cache = loading/ready` が出ます。

## 実行

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json
```

中断後に再開する場合:

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json `
  --resume
```

## 出力

runner root は次の場所です。

```text
<output_folder>/tuning-<tag_prefix>/
```

主な出力:

| path | 意味 |
| --- | --- |
| `summary-learn.log` | 各 trial の結果 |
| `best-checkpoint/` | 観測上の best trial の checkpoint |
| `recommended-parameters.json` | 上位 trial から推定した推奨パラメーター |
| `runner-state.json` | resume 用 |
| `logs/` | trial ごとの stdout |

## checkpoint の保存と削除

`keep_all_trials` は、trial ごとの checkpoint をどれだけ残すかを決めます。

```json
"metric": "quantized_value_loss",
"lower_is_better": true,
"keep_all_trials": false
```

この設定では、`quantized_value_loss` が小さい trial を良い trial とみなします。runner は、その時点で一番良い trial だけを `best-checkpoint/` に残します。best にならなかった trial の出力フォルダと checkpoint は、trial 終了後に削除します。

削除しても、`summary-learn.log` と `logs/trialXXXX.stdout.log` は残るので、各 trial の指標と実行ログはあとから確認できます。

すべての trial checkpoint を残したい場合は、次のどちらかを使います。

- `keep_all_trials: true`
- 実行時に `--keep-temp`

通常は storage 消費を抑えるため、`keep_all_trials: false` のままにしておくのが安全です。

`recommended-parameters.json` の `recommended.parameters` は、上位 trial から重み付き平均で推定した値です。`log: true` のパラメーターは log 空間で平均するため、学習率のように桁で効く値に向いています。
