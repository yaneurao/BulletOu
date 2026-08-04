# 7. 結果を確認する — 出力ファイルと学習ログ

<a href="../../en/tutorial/7-result.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習が終わったあと (もしくは学習中) にやることをまとめる:
- 出力ディレクトリの中身を確認
- 学習ログ (`learn.log`) を読んで学習が正常に進んでいるかチェック

(学習結果をエンジンに組み込んで動作確認する手順は [8. エンジンに組み込む](8-engine.md) を参照。)

## 7.1 出力を確認する

学習完了後、出力ディレクトリ (例: `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/`) は以下のレイアウト:

```
checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/
├── summary-learn.log                  ← 学習・再開を通して追記される sb 単位のログ
├── 0001/
│   ├── nn.bin                         ← やねうら王 / Stockfish が読み込める NNUE バイナリ
│   ├── state.bin                      ← 再開用の重み + Ranger optimizer state
│   └── learn.log                      ← この保存時点の学習ログ
├── 0002/
├── ...
└── 000N/                              ← 最新 (= 最後に保存されたもの)
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

最新の `000N/` (= 最大番号のディレクトリ) がエンジンに渡す成果物が入ったフォルダ。

KPPT / KPP_KKPT の場合は `nn.bin` の代わりに `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` の 3 ファイル組が `000N/` 配下に入る (3 ファイル全部必要)。

## 7.2 学習ログ (`learn.log`) の読み方

学習中・終了後の loss 推移は `<output>/summary-learn.log` と各 `<output>/0NNN/learn.log` に記録される。普段見るのは `summary-learn.log` です。検証または保存された superbatch ごとに 1 行ずつ増えます。

`--validation-rate` は、検証用ファイルの accuracy/loss を何 superbatch ごとに計算するかを決めます。デフォルトは `--save-rate` と同じです。たとえば `--validation-rate 1 --save-rate 20` なら、保存は 20 sb ごとのまま、検証だけを毎 sb 実行できます。検証だけの行には、対応する `000N/` ディレクトリがありません。学習を中断して再開した場合、最新 checkpoint より後のログ行は、その時点へ戻れないため切り詰められます。

`--test-positions` を省略した場合、検証は `--test-teacher` に含まれる全局面を使います。同じ検証ファイルなら毎回同じ条件で比較できます。`--test-positions N` で一部だけ使う場合、再現性を持たせたいなら `--test-seed 1` のように非ゼロ seed を指定するか、`--test-sample sequential` を使ってください。`--test-seed 0` のランダムサンプルは時刻に依存するため、同じ学習条件でも accuracy/loss が少し揺れます。

### どっちを見るか

- **トップレベル `<output>/summary-learn.log`** — 普段はこれを見る。再開後の行も追記される。
- **各 `0NNN/learn.log`** — その保存時点までのログ。`0005/` の時点だけ見たい、という場合に使う。

### CSV のサンプル

`learn.log` と `summary-learn.log` は CSV です。`summary-learn.log` は右端に `test_teacher` 列があり、どの検証ファイルで accuracy/loss を測ったかが分かります。

```csv
eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,32,-,-,0.6234,0.001000,0.000999,1.000000,2097152,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,64,-,-,0.5891,0.000999,0.000998,1.000000,4194304,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,96,-,-,0.5510,0.000998,0.000997,1.000000,6291456,teachers/
...
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,2,32,-,-,0.4523,0.000934,0.000933,1.000000,102039552,teachers/
...
```

`learn.log` では **32 batch ごとに 1 行** loss を記録します。`summary-learn.log` は、検証または保存された sb ごとに 1 行です。`curr_batch` 列が 1 sb 内の最後の batch に達すると、次の行から `superbatch` が +1 され、`curr_batch` は 1 から再開します。

### 列の意味

| 列 | 意味 | 例 |
|---|---|---|
| `eval` | 学習対象名。KPPT 系では `KPPT/kk` のように成分名も付く | `NNUE_HALFKP-NNUE_halfkp_256x2_32_32` / `KPPT/kk` |
| `epoch` | 今回の起動内の epoch (1 始まり) | `1` |
| `superbatch` | epoch 内 superbatch (1 始まり)。`--positions-per-superbatch` の実効局面数ごとに +1 | `1`, `2`, ... |
| `curr_batch` | superbatch 内 batch (1 始まり)。bullet は 32 batch ごとに 1 行記録 | `32`, `64`, ..., `1525` |
| `test_value_accuracy` | `--test-teacher` の検証 accuracy。sb 境界行だけ実値、それ以外は `-` | `0.583784` |
| `test_value_loss` | `--test-teacher` の検証 loss。sb 境界行だけ実値、それ以外は `-` | `0.129676` |
| `train_value_loss` | bullet が記録する 32-batch 平均 loss | `0.234` |
| `lr_start` | その行の区間開始時の学習率。summary 行では superbatch 開始 LR | `0.001000` |
| `lr_end` | その行の最後の batch で使った学習率。summary 行では superbatch 終端側 LR | `0.000934` |
| `lambda` | `--lambda` 値 (1 回の起動内で定数、6 桁固定) | `1.000000` |
| `positions` | 累計教師局面数 (**再開をまたいでも累積**) | `2097152` |
| `teacher` | `--teacher` の値 | `teachers/` |

NNUE 系は `--arch` を `eval` 列に含める (出力ディレクトリ名と同じ命名)。KPPT 系は `--arch` を使わないので `eval` 列に arch は出ない。

細かいファイル仕様は [`spec/04-checkpoint-layout.md`](../../spec/04-checkpoint-layout.md#learnlog-フォーマット) を参照。

### pandas で読む

```python
import pandas as pd

df = pd.read_csv("checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/summary-learn.log")
print(df.shape)        # 全行数
print(df.tail())       # 最後の数行
print(df["train_value_loss"].describe())   # loss の統計
```

CSV header 付きなので `pd.read_csv` で列名は自動取得される。

### 学習が順調かを見るチェックリスト

学習が正しく動いている場合の典型的な兆候:

1. **`train_value_loss` が概ね単調減少**
   - 学習開始直後は急に下がり、徐々に減衰
   - 1 superbatch を消化するごとに目に見えて下がるのが理想
   - 1 superbatch まわっても下がらない場合は `--lr` が大きすぎる、または教師サイズが学習器のキャパに対して小さすぎる可能性
   - 定期的な loss の跳ね上がりや、局所的な loss の偏りが見える場合は、教師局面のシャッフル不足の可能性が高い。事前シャッフルするか、`--teacher-shuffle-buffer-sbs` を使う。対処は [§3.2 教師局面をシャッフルする](3-data.md#教師局面をシャッフルする) を参照

2. **`lr_start` / `lr_end` がスケジュール通りに動いている**
   - `--lr-schedule step` (デフォルト): 1 superbatch ごとに `lr *= gamma` し、`--lr-min` を下限にする。`gamma` は明示値、または 1 epoch 長から自動計算された値。epoch 境界で `--lr` に戻る
   - `--lr-schedule geometric`: 1 epoch で `--lr` → `--lr-min` へ滑らかに下げ、次 epoch で `--lr` に戻る
   - `--lr-schedule cos`: cosine カーブで `--lr` → `--lr-min` へ下げ、次 epoch で `--lr` に戻る
   - 期待通り変化していないなら学習率系フラグの値を見直す ([§6.3 学習率をどう下げるか](6-tune.md#63-学習率をどう下げるか) 参照)

3. **`positions` が単調増加**
   - 1 superbatch 完了で約 1 億 (= `--positions-per-superbatch` を `--batch-size` の倍数へ切り捨てた値)
   - 教師サイズと照らし合わせると「ちゃんと教師を全部読めているか」が分かる

4. **`superbatch` がきちんと進んでいるか**
   - 教師局面数が 1 億未満だと 1 周回しても `superbatch` は 1 のまま終わる (fallback save で 1 度だけ保存)。これは仕様
   - 大きな教師なら `curr_batch` が実効superbatch内の最終batchに到達するごとに `superbatch` が +1 されているはず
   - `superbatch` がいつまでも 1 のままで `curr_batch` も実効superbatch内の最終batchよりかなり小さい値で止まっているなら、loader が打ち切られている可能性

### 簡単なプロット

```python
import matplotlib.pyplot as plt

# positions を時系列軸に
plt.figure(figsize=(12, 4))
plt.plot(df["positions"], df["train_value_loss"])
plt.xlabel("positions"); plt.ylabel("train_value_loss")
plt.title("training loss curve")
plt.savefig("loss_curve.png")
```

### KPPT の場合 (kk / kkp / kpp 同時記録)

KPPT 系では 1 回の保存につき kk → kkp → kpp の 3 成分のログが連続して書かれます。`eval` 列に `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` のように成分名が付くので、それで絞り込んでからプロットします。

```python
for c in ["kk", "kkp", "kpp"]:
    sub = df[df["eval"] == f"KPPT/{c}"]
    plt.plot(sub["positions"], sub["train_value_loss"], label=c)
plt.legend(); plt.xlabel("positions"); plt.ylabel("loss")
```

KK の loss は KKP / KPP よりも小さいネットワークなので **絶対値の比較ではなく減少傾向** を見ます。

`eval` 列から family / component を分離したいときは:

```python
df[["family", "component"]] = df["eval"].str.split("/", n=1, expand=True)
df["component"] = df["component"].fillna("nnue")   # NNUE 系はスラッシュなし
```

### 再開後のログの見方

再開すると、新しい行が学習ログにそのまま追記されます。再開後は:
- `epoch` が 1 から再開
- `superbatch` が 1 から再開
- **`positions` だけ前 run の最終値からの続き** で増える

`positions` 列を横軸に使えば、再開をまたいでも連続した loss 曲線が描けます。

`epoch` / `superbatch` は 1 回の起動内のカウンタなので、再開をまたぐと同じ番号が複数回出ます。境界を見つけたい場合は、`positions` が増え続けていることを確認しつつ、`(epoch, superbatch)` が 1 に戻った行を探します。

## 7.3 次のステップ

- [8. エンジンに組み込む](8-engine.md) — 学習結果をやねうら王エンジンで動作確認する
- [リファレンス: NNUE HalfKP 学習](../shogi/halfkp.md) — `nn.bin` のバイナリレイアウト、量子化、再開の詳細
- [リファレンス: NNUE K-P 学習](../shogi/kp.md) — HalfKP との比較、入力 feature の構造
- [リファレンス: NNUE HalfKPE9 学習](../shogi/halfkpe9.md) — 利き数情報拡張版
- [リファレンス: KPPT / KPP_KKPT 学習](../shogi/kppt.md) — KPPT / KPP_KKPT の学習
- [仕様: spec/](../../spec/) — 学習対象一覧 / バイナリレイアウト / hash 計算式 / `learn.log` フォーマット

---

前へ: [6. 学習設定を調整する](6-tune.md)
