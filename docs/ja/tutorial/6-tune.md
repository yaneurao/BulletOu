# 6. 学習をチューニング — スケジュールと教師ターゲット

<a href="../../en/tutorial/6-tune.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[4. 学習を走らせる](4-train.md) のデフォルト設定で動くことを確認したら、この章で **学習を調整するためのフラグ群** を紹介する。**最初の学習ではすべてデフォルトのままで問題ない**。チューニングが必要になったときに戻ってくる位置付け。

## 6.1 学習スケジュール

ログに出てくる `superbatch` は **checkpoint や学習率を更新するためのまとまり**で、デフォルトで約 1 億局面ぶん。

主要なフラグ:

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--batch-size` | 1 gradient step あたりの局面数 | 16384 |
| `--batches-per-superbatch` | 1 superbatch を構成する mini-batch 数 | `ceil(100M / batch-size)` (≒ 1 superbatch ≒ 1 億局面) |
| `--superbatches` | epoch あたりの superbatch 数の上限 | 上限なし (= EOF まで) |
| `--max-epochs` | 教師データを何周するか | 1 |
| `--save-rate` | N superbatch ごとに checkpoint を保存 | 1 |
| `--lr` / `--lr-gamma` / `--lr-step` | StepLR (`lr-step` superbatch ごとに `lr-gamma` 倍) | 0.001 / 0.1 / 8 |
| `--lambda` | 教師 eval と対局結果 (WDL) のブレンド比 ([§6.2](#62-教師ターゲット-lambda) 参照) | 1.0 (= 純 eval) |

実行例 (1 億局面 × 40 superbatch = 計 40 億局面):

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher teachers/ \
    --superbatches 40
```

教師ファイルが 1 superbatch 未満 (≒ 1 億局面未満) しか無い場合は `--batches-per-superbatch` を小さくする (例: `1024` で 1 superbatch ≒ 1670 万局面) と、何回も save が走るようになる。

### 学習率の動き

`--lr 0.001 --lr-gamma 0.1 --lr-step 8` (デフォルト) の場合の学習率推移:

| superbatch | lr |
|---|---|
| 1 - 8 | 0.001 |
| 9 - 16 | 0.0001 |
| 17 - 24 | 0.00001 |
| 25 - 32 | 0.000001 |
| ... | ... |

学習が走った後で実際の lr 推移を確認するには、`learn.log` の `lr` 列を見れば良い ([§7.2 学習ログの読み方](7-result.md#72-学習ログ-learnlog-の読み方))。

### 複数 epoch 回す

`--max-epochs N` を指定すると教師データを N 周する。各 epoch 開始時に:
- LR scheduler が reset される (superbatch 1 から再開、`lr = --lr` に戻る)
- データローダーが先頭にシークし直す

つまり N 回学習し直すに近い挙動。各 epoch ごとに lr が再下降するので、長時間学習で局所最適から脱出させたいときに使う。

## 6.2 教師ターゲット (`--lambda`)

教師局面ファイル (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) には、各局面ごとに **2 種類のラベル** が記録されている:

1. **教師 eval** — その局面に対する教師エンジンの評価値 (sigmoid 後)
2. **対局結果** — その対局が最終的にどう終わったか (W/D/L = 1.0 / 0.5 / 0.0、side-to-move 視点)

`--lambda <λ>` で、loss target をこの 2 つでどう混ぜるかを指定する (やねうら王内蔵学習器の `lambda` と同じ慣例):

```
target = λ × 教師eval + (1 − λ) × 対局結果
```

| `--lambda` 値 | 意味 |
|---|---|
| `1.0` (デフォルト) | 100% 教師 eval、対局結果は無視 |
| `0.5` | eval 50% + 対局結果 50% (elmo 式の典型値) |
| `0.0` | 100% 対局結果、教師 eval は無視 |
| `0.7` 等 | 中間値も自由に指定可能 |

デフォルトの `1.0` (純 eval) が安全な初期値。教師エンジンの評価値そのものを真似に行く動作になる。

対局結果 (W/D/L = Win / Draw / Loss) も混ぜたいときに `--lambda` を下げる。完全結果ベース (`--lambda 0.0`) は教師エンジンの強さに依存しないが、勾配が疎で収束が遅い傾向。実用は `0.5 〜 0.8` あたりの混合が多い。

```bash
# elmo 式の 50:50 ブレンドで KPPT 学習
./target/release/examples/bulletou \
    --eval-type KPPT \
    --teacher teachers/ \
    --lambda 0.5
```

(`WDL` は Win/Draw/Loss の略。)

---

次へ: [7. 結果を確認](7-result.md) — 学習結果の確認、ログの読み方

前へ: [4. 学習を走らせる](4-train.md)
