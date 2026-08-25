# 固定長 trial によるパラメーター調整

`tuning_parameters.py` は、短い学習 trial を多数走らせ、`lr` / `lr_min` / factorizer / count confidence のよさそうな値を探しながら学習を進めるための runner です。

外部 package は使っていません。BulletOu 用の軽量な TPE-style sampler です。最初はランダムに試し、その後は良かった trial の分布と悪かった trial の分布を比べ、有望そうな範囲を重点的に試します。

## 何をする runner か

- 同じ generation 内の trial は、同じ checkpoint から始まります。
- 1 trial の長さは `tuning.trial_sbs` で指定します。数値または配列で書けます。
- trial 中はパラメーターを変えません。
- generation ごとに `population` 本の trial を走らせ、指定した metric が良いものを記録します。
- generation の最後に、選ばれたパラメーターで commit run を1回だけ実行し、その checkpoint を `current-checkpoint/` に保存します。
- `best-checkpoint/` には、これまでの commit run の中で一番良かった checkpoint を保存します。これは次 generation の開始地点とは別です。
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

TPE-style sampler は、直前 generation の結果を使って次の候補を作ります。完了済み trial を metric で並べ、上位を「良かった候補」、残りを「悪かった候補」として、各パラメーターの分布を作ります。そのうえで、良かった候補の分布に近く、悪かった候補の分布から遠い値を優先してサンプルします。

ただし、直前 generation の trial 数が `tpe_startup_trials` に満たない場合や、trial が1本しかなく悪かった候補の分布を作れない場合は、探索範囲全体から完全ランダムに戻るのではなく、直前 generation の `recommended` parameter の周辺をガウスノイズでサンプルします。つまり、generation 1 は広く探索し、generation 2 以降は前の generation で得た推奨値を中心に探索します。

たとえば generation 2 の trial は、generation 1 の最後に作られた commit checkpoint から追加学習します。また、generation 2 の候補パラメーターは generation 1 の trial 結果をもとに作ります。generation 3 では generation 2 の最後に作られた commit checkpoint から追加学習し、候補パラメーターは generation 2 の結果をもとに作ります。

全世代の trial を混ぜて TPE しないのは、generation が進むほど開始 checkpoint が変わり、metric の土台も変わるためです。違う学習段階の trial を同じ尺度として混ぜると、古い generation が不利になりやすく、TPE の判断が歪みます。

## sampler の項目

これらは学習そのもののパラメーターではなく、「次にどの候補を試すか」を決める sampler 側の設定です。

| 項目 | 意味 | 省略時 |
| --- | --- | --- |
| `sampler` | `"tpe"` または `"random"`。通常は `"tpe"` を使います。 | `"tpe"` |
| `tpe_startup_trials` | TPE の密度推定に必要な完了済み trial 数。generation 1 ではこの本数に届くまで探索範囲全体からランダムにサンプルします。generation 2 以降で直前 generation の trial 数が足りない場合は、直前 generation の `recommended` parameter 周辺をサンプルします。 | `16` |
| `tpe_good_fraction` | TPE が上位何割を「良かった候補」として使うか。`0.25` なら上位25%を使います。 | `0.25` |
| `tpe_bandwidth` | TPE の KDE 幅の下限です。大きいほど候補が広めに散り、小さいほど観測された良い候補の近くに寄ります。 | `0.15` |
| `commit_source` | generation の最後に commit run へ使うパラメーター。`"best"` なら実測1位、`"recommended"` なら上位 trial から推定した値を使います。 | `"best"` |

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
    "commit_source": "best",
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
| `current-checkpoint/` | 最新 generation で採用された checkpoint。次 generation の開始地点 |
| `pending-commit-checkpoint/` | commit run 完了後、`current-checkpoint/` へ反映する前の一時 checkpoint。通常は残りません |
| `best-checkpoint/` | これまでの commit run の中で一番良かった checkpoint |
| `recommended-parameters.json` | 上位 trial から推定した推奨パラメーター |
| `runner-state.json` | resume 用 |
| `logs/` | trial ごとの stdout |

## `recommended-parameters.json` の読み方

`recommended-parameters.json` には、主に次の2種類の値が入ります。

| 項目 | 意味 |
| --- | --- |
| `best_observed` | 実際に走らせた trial のうち、指定した metric が一番良かったもの |
| `recommended.parameters` | 上位 trial から推定した、次に使う候補値 |

`best_observed` は「観測された1本の best」です。短い trial では偶然良かっただけの可能性があります。
一方、`recommended.parameters` は最新 generation の上位 trial をならした値です。長めの本番学習へ使う候補としてはこちらも確認してください。

`recommended.parameters` は、次の手順で計算します。

1. 最新 generation の完了済み trial を metric の良い順に並べます。
2. `tpe_good_fraction` で指定した上位割合だけを使います。
3. その上位 trial を、順位に応じた重み付き平均にします。

たとえば次の設定だとします。

```json
"tpe_startup_trials": 16,
"tpe_good_fraction": 0.25
```

16 trial 終了時点では、上位 `ceil(16 * 0.25) = 4` 本を使います。
重みは best から順に `4, 3, 2, 1` です。

`log: false`、または `log` を省略して線形扱いになっているパラメーターは、次の重み付き算術平均です。

```text
recommended = (4 * p1 + 3 * p2 + 2 * p3 + 1 * p4) / (4 + 3 + 2 + 1)
```

`log: true` のパラメーターは、log 空間で平均します。これは重み付き幾何平均に相当します。

```text
recommended = exp((4 * log(p1) + 3 * log(p2) + 2 * log(p3) + 1 * log(p4)) / (4 + 3 + 2 + 1))
```

`lr` や `lr_min` のように桁で効く値は `log: true` に向いています。
factorizer alpha や count confidence のように `min: 0` を許す値は、`log(0)` が定義できないため、通常は線形平均になります。

この推奨値の計算は、次の trial を作る TPE sampler そのものとは別です。
TPE sampler は良かった trial と悪かった trial の分布を比べて次の候補を作ります。
`recommended.parameters` は、完了済み trial から「いま人間が見るならこのあたり」という値をまとめたものです。

## checkpoint の保存と削除

`keep_all_trials` は、trial ごとの checkpoint をどれだけ残すかを決めます。

```json
"metric": "quantized_value_loss",
"lower_is_better": true,
"keep_all_trials": false
```

この設定では、`quantized_value_loss` が小さい trial を良い trial とみなします。現在の標準動作では、trial ごとの checkpoint は保存しません。各 trial の metric とパラメーターだけを `summary-learn.log` に記録し、generation の最後に `commit_source` で選んだパラメーターを使って commit run を1回だけ実行します。その commit run の checkpoint が `current-checkpoint/` になります。

`commit_source: "best"` では、その generation で実測 metric が一番良かった trial のパラメーターを使います。`commit_source: "recommended"` では、最新 generation の上位 trial から `recommended-parameters.json` と同じ式で推定したパラメーターを使います。`recommended` は未評価の推定値なので、標準ではより安全な `"best"` を使います。

削除しても、`summary-learn.log` と `logs/trialXXXX.stdout.log` は残るので、各 trial の指標と実行ログはあとから確認できます。

すべての trial checkpoint を残したい場合は、次のどちらかを使います。

- `keep_all_trials: true`
- 実行時に `--keep-temp`

通常は storage 消費を抑えるため、`keep_all_trials: false` のままにしておくのが安全です。
