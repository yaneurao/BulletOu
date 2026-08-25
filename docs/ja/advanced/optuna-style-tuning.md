# Optuna風の固定長 trial 探索

`optuna_style_runner.py` は、同じ開始地点から短い学習 trial を多数走らせ、`lr` / `lr_min` / factorizer / count confidence のよさそうな値を探すための runner です。

外部の Optuna package は使っていません。BulletOu 用の軽量な sampler です。最初はランダムに試し、その後は良かった trial の近くを重点的に試します。

## 何をする runner か

- 各 trial は scratch、または指定した同じ checkpoint から始まります。
- 1 trial の長さは `tuning.trial_sbs` で指定します。
- trial 中はパラメーターを変えません。
- `population` 本の trial を走らせ、指定した metric が良いものを記録します。
- `recommended-parameters.json` に、上位 trial から推定した推奨パラメーターを書き出します。

短い trial では「たまたま良かった1本」をそのまま採用するとノイズを拾いやすいです。そのため、最終的に使う値は `best_observed` だけでなく、`recommended` も確認してください。

## `log` とは何か

`parameters` の各項目には、必要なら `log: true` を書けます。

`log: true` は、値を倍率ベースで探索する指定です。たとえば `min=0.000001, max=0.001` の学習率では、`0.000001`、`0.00001`、`0.0001`、`0.001` のような桁の違いを自然に探索できます。

一方で、`log: true` は `min=0` と両立しません。`log(0)` が存在しないためです。

factorizer alpha や count confidence で `0` を許したい場合は、`log` を書かないか、`log: false` にしてください。`min` が `0` 以下なら、runner は省略時に線形探索として扱います。

## `startup_trials` / `elite_fraction` / `elite_sigma`

この3つは学習そのもののパラメーターではなく、「次にどの候補を試すか」を決める sampler 側の設定です。

| 項目 | 意味 | 省略時 |
| --- | --- | --- |
| `startup_trials` | 最初に完全ランダムで試す trial 数。この本数に到達するまでは、過去の良かった候補の近くを狙わず、探索範囲全体からサンプルします。 | `16` |
| `elite_fraction` | `startup_trials` 後に、完了済み trial の上位何割を「良かった候補」として使うか。`0.25` なら上位25%を使います。 | `0.25` |
| `elite_sigma` | 良かった候補の近くをサンプルするときの散らばり幅。線形探索では `(max - min) * elite_sigma`、`log: true` では log 空間の幅に対する割合です。 | `0.15` |

つまり、次のような流れです。

```text
最初 startup_trials 本はランダムに試す
その後は上位 elite_fraction の候補の近くを elite_sigma の幅で試す
```

## 設定例

```json
{
  "version": 1,
  "tuning": {
    "population": 100,
    "trial_sbs": 4,
    "metric": "quantized_value_loss",
    "lower_is_better": true,
    "seed": 20260825,
    "startup_trials": 16,
    "elite_fraction": 0.25,
    "elite_sigma": 0.15,
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

## 実行

```powershell
python .\optuna_style_runner.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json
```

中断後に再開する場合:

```powershell
python .\optuna_style_runner.py `
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

`recommended-parameters.json` の `recommended.parameters` は、上位 trial から重み付き平均で推定した値です。`log: true` のパラメーターは log 空間で平均するため、学習率のように桁で効く値に向いています。
