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
├── summary-learn.log                  ← 全 run / resume を連結した sb 単位の累積ログ
├── 0001/
│   ├── nn.bin                         ← やねうら王 / Stockfish 互換 NNUE バイナリ
│   ├── state.bin                      ← resume 用の重み + Adam moments
│   └── learn.log                      ← この save 時点の学習ログ snapshot
├── 0002/
├── ...
└── 000N/                              ← 最新 (= 最後に保存された) save
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

最新の `000N/` (= 最大番号のディレクトリ) がエンジンに渡す成果物が入ったフォルダ。

KPPT / KPP_KKPT の場合は `nn.bin` の代わりに `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` の 3 ファイル組が `000N/` 配下に入る (3 ファイル全部必要)。

## 7.2 学習ログ (`learn.log`) の読み方

学習中・終了後の loss 推移は `<output>/summary-learn.log` (累積) と各 `<output>/0NNN/learn.log` (各 save 時点の snapshot) に記録される。列数は違い、`summary-learn.log` は superbatch 境界だけ、各 `0NNN/learn.log` は per-batch snapshot。

### どっちを見るか

- **トップレベル `<output>/summary-learn.log`** — 全 run / resume を **連結した累積版**。普段はこれを見る。
- **各 `0NNN/learn.log`** — その save 時点までのスナップショット。学習途中に「save 0005 の時点でどうだったか」を見たいときに使う。

### CSV のサンプル

```csv
eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,32,-,-,0.6234,0.001000,0.000999,1.000000,2097152,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,64,-,-,0.5891,0.000999,0.000998,1.000000,4194304,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,96,-,-,0.5510,0.000998,0.000997,1.000000,6291456,teachers/
...
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,2,32,-,-,0.4523,0.000934,0.000933,1.000000,102039552,teachers/
...
```

bullet は **32 batch ごとに 1 行** loss を記録する。デフォルト (`--positions-per-superbatch 100000000`、`--batch-size` 省略 = 65536) では、実効値は 1525 batch (= 99,942,400 局面) で、1 superbatch あたり約 48 行。`--batch-size 16384` を明示した場合は 6103 batch (= 99,991,552 局面) で約 191 行になる。`curr_batch` 列が実効superbatch内の最終batchに達すると `superbatch` が +1 されて `curr_batch` は 1 から再開する。

### 列の意味

| 列 | 意味 | 例 |
|---|---|---|
| `eval` | 出力ディレクトリ名と同じ `<eval-type>[-<arch>]` 形式 + マルチ component (KPPT 系) ではさらに `/<component>` | `NNUE_HALFKP-NNUE_halfkp_256x2_32_32` / `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` |
| `epoch` | run 内 epoch (1 始まり) | `1` |
| `superbatch` | epoch 内 superbatch (1 始まり)。`--positions-per-superbatch` の実効局面数ごとに +1 | `1`, `2`, ... |
| `curr_batch` | superbatch 内 batch (1 始まり)。bullet は 32 batch ごとに 1 行記録 | `32`, `64`, ..., `1525` |
| `test_value_accuracy` | `--test-teacher` の検証 accuracy。sb 境界行だけ実値、それ以外は `-` | `0.583784` |
| `test_value_loss` | `--test-teacher` の検証 loss。sb 境界行だけ実値、それ以外は `-` | `0.129676` |
| `train_value_loss` | bullet が記録する 32-batch 平均 loss | `0.234` |
| `lr_start` | その行の区間開始時の学習率。summary 行では superbatch 開始 LR | `0.001000` |
| `lr_end` | その行の最後の batch で使った学習率。summary 行では superbatch 終端側 LR | `0.000934` |
| `lambda` | `--lambda` 値 (1 run 内で定数、6 桁固定) | `1.000000` |
| `positions` | 累計教師局面数 (**resume 跨ぎで累積**) | `2097152` |
| `teacher` | `--teacher` の値 | `teachers/` |

NNUE 系は `--arch` を `eval` 列に含める (出力ディレクトリ名と同じ命名)。KPPT 系は `--arch` を使わないので `eval` 列に arch は出ない。

正確な仕様は [`spec/04-checkpoint-layout.md`](../../spec/04-checkpoint-layout.md#learnlog-フォーマット) を参照。

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
   - **periodic な loss スパイク** や局所的な loss の偏りが見える場合は、教師ファイルが事前シャッフルされていない可能性が高い。BulletOu は学習時に追加シャッフルしないので、対処は [§3.2 教師ファイルは事前にシャッフルしておく](3-data.md#教師ファイルは事前にシャッフルしておく) を参照

2. **`lr_start` / `lr_end` がスケジュール通りに動いている**
   - `--lr-schedule step` (デフォルト): 1 superbatch ごとに `lr *= gamma` し、`--lr-min` を下限にする。`gamma` は明示値、または 1 epoch 長から自動計算された値。epoch 境界で `--lr` に戻る
   - `--lr-schedule geometric`: 1 epoch (= `--superbatches × sb_size` 局面) で `--lr` (lr_max) → `--lr-min` を **geometric** (= 対数線形) に減衰、epoch 末で warm restart して lr_max に戻る
   - `--lr-schedule cos`: cosine annealing で `--lr` (lr_max) → `--lr-min` を 1 epoch (= `--superbatches × sb_size` 局面) で 1 周期。各 cycle 末 (= epoch 末) で warm restart して `--lr` に戻る
   - 期待通り変化していないなら lr 系フラグの値を見直す ([§6.1 学習スケジュール](6-tune.md#61-学習スケジュール) 参照)

3. **`positions` が単調増加** (run 内、resume 跨いでも)
   - 1 superbatch 完了で約 1 億 (= `--positions-per-superbatch` を `--batch-size` の倍数へ切り捨てた値)
   - 教師サイズと照らし合わせると「ちゃんと教師を全部読めているか」が分かる

4. **`superbatch` がきちんと進んでいるか**
   - 教師局面数が 1 億未満だと 1 周回しても `superbatch` は 1 のまま終わる (fallback save で 1 度だけ保存)。これは仕様
   - 大きな教師なら `curr_batch` が実効superbatch内の最終batchに到達するごとに `superbatch` が +1 されているはず
   - `superbatch` がいつまでも 1 のままで `curr_batch` も実効superbatch内の最終batchよりかなり小さい値で止まっているなら、loader が打ち切られている可能性 (旧 HCPE loader 極性バグ等)

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

KPPT 系では 1 save につき kk → kkp → kpp の 3 component のログが連続して書かれる。`eval` 列に `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` のように component が付いているので、それでフィルタしてからプロット:

```python
for c in ["kk", "kkp", "kpp"]:
    sub = df[df["eval"] == f"KPPT/{c}"]
    plt.plot(sub["positions"], sub["value_loss"], label=c)
plt.legend(); plt.xlabel("positions"); plt.ylabel("loss")
```

KK component の loss は KKP / KPP よりも小さいネットワークなので **絶対値の比較ではなく減少傾向** を見る。

`eval` 列から family / component を分離したいときは:

```python
df[["family", "component"]] = df["eval"].str.split("/", n=1, expand=True)
df["component"] = df["component"].fillna("nnue")   # NNUE 系はスラッシュなし
```

### resume 後のログの見方

resume すると新 run の行が学習ログにそのまま追記される。新 run では:
- `epoch` が 1 から再開
- `superbatch` が 1 から再開
- **`positions` だけ前 run の最終値からの続き** で増える

`positions` 列を時系列軸に使えば、resume 跨ぎでも連続な loss curve が描ける (上記プロット例と同じ書き方で OK)。

`epoch` / `superbatch` は run 内のカウンタなので、resume を跨ぐと同じ番号が複数回出現する。run の境界を判別したい場合は、`positions` が突然小さくならない (= 単調) ことを確認しつつ、`(epoch, superbatch)` がリセットされた行を見つける。

## 7.3 次のステップ

- [8. エンジンに組み込む](8-engine.md) — 学習結果をやねうら王エンジンで動作確認する
- [リファレンス: NNUE HalfKP 学習](../shogi/halfkp.md) — `nn.bin` のバイナリレイアウト、量子化、resume の詳細
- [リファレンス: NNUE K-P 学習](../shogi/kp.md) — HalfKP との比較、入力 feature の構造
- [リファレンス: NNUE HalfKPE9 学習](../shogi/halfkpe9.md) — 利き数情報拡張版
- [リファレンス: KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習
- [仕様: spec/](../../spec/) — eval-type 一覧 / バイナリレイアウト / hash 計算式 / `learn.log` フォーマット

---

前へ: [6. 学習をチューニング](6-tune.md)
