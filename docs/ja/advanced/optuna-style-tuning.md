# Optuna風の固定長 trial 探索

`optuna_style_runner.py` は、初期学習から固定パラメーターで短い trial を何本も回し、`lr` / `lr_min` / factorizer / count confidence の候補を探すための runner です。

これは Optuna 本体ではありません。外部 Python package を増やさずに使える、BulletOu 用の軽量な Optuna 風 sampler です。

## population search runner との違い

population search runner は、採用した checkpoint から次の候補へ進みます。そのため、学習済み checkpoint の途中で factorizer や count confidence を動かすと、変更直後の崩れとパラメーター自体の良し悪しが混ざります。

Optuna風 runner は、各 trial を scratch または同じ base checkpoint から開始します。trial 中はパラメーターを変えません。

そのため、次のような用途に向いています。

- 初期学習から、どの固定パラメーターがよいか見る
- `lr` と `lr_min` も探索対象に入れる
- 1 trial = 16sb のような短い比較を大量に回す
- best checkpoint ではなく、上位trialから推定したパラメーターを知る

## 設定例

population search settings と合わせるため、探索範囲は `min` / `max` で指定します。

```json
{
  "version": 1,
  "study": {
    "trials": 64,
    "trial_sbs": 16,
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
    "tag_prefix": "optuna-scratch-16sb"
  },
  "parameters": {
    "lr": { "tune": true, "min": 0.00003, "max": 0.001, "log": true },
    "lr_min_ratio": { "tune": true, "min": 0.03, "max": 1.0, "log": true },

    "shared": 1.0,
    "king_axis": 1.0,
    "hand_axis": 1.0,
    "progress_axis": 1.0,
    "king_hand_pair": 1.0,
    "king_progress_pair": 1.0,
    "hand_progress_pair": 1.0,

    "residual_count": { "tune": true, "min": 0.3, "max": 3.0, "log": true },
    "king_axis_count": { "tune": true, "min": 0.3, "max": 3.0, "log": true },
    "hand_axis_count": { "tune": true, "min": 0.3, "max": 3.0, "log": true },
    "progress_axis_count": { "tune": true, "min": 0.3, "max": 3.0, "log": true },
    "king_hand_pair_count": { "tune": true, "min": 0.3, "max": 3.0, "log": true },
    "king_progress_pair_count": { "tune": true, "min": 0.3, "max": 3.0, "log": true },
    "hand_progress_pair_count": { "tune": true, "min": 0.3, "max": 3.0, "log": true }
  }
}
```

`lr_min_ratio` を使うと、runner は次のように `lr_min` を作ります。

```text
lr_min = lr * lr_min_ratio
```

これにより、`lr_min > lr` のような無効な候補を避けられます。

## 実行

```powershell
python .\optuna_style_runner.py `
  --settings-file D:\BulletOu-snapshots\settings\optuna-style-settings.json
```

途中再開するときは:

```powershell
python .\optuna_style_runner.py `
  --settings-file D:\BulletOu-snapshots\settings\optuna-style-settings.json `
  --resume
```

## 出力

runner root は次の場所です。

```text
<output_folder>/optuna-<tag_prefix>/
```

主な出力は次の通りです。

| path | 意味 |
| --- | --- |
| `summary-learn.log` | 各trialの結果 |
| `best-checkpoint/` | 観測上のbest trialのcheckpoint |
| `recommended-parameters.json` | 上位trialから推定した推奨パラメーター |
| `runner-state.json` | resume用 |
| `logs/` | trialごとのstdout |

## best observed と recommended

このrunnerでは、単に「一番良かったtrial」だけでなく、`recommended-parameters.json` を出力します。

`best_observed` は、実際に一番良い指標を出したtrialです。

`recommended` は、上位trial群から重み付き平均で推定したパラメーターです。log sampling のパラメーターは、log空間で平均します。つまり、`lr` や count confidence では幾何平均に近い値になります。

短いtrialでは、たまたま良かった1本をそのまま採用するとノイズに引っ張られます。固定パラメーターとして次に長く試すなら、通常は `recommended.parameters` を見るほうが安定します。

