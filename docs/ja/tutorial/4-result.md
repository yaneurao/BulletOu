# 4. 結果を確認・活用する — 出力ファイル / 学習ログ / エンジン投入

<a href="../../en/tutorial/4-result.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習が終わったあと (もしくは学習中) にやることをまとめる:
- 出力ディレクトリの中身を確認
- 学習ログ (`learn.log`) を読んで学習が正常に進んでいるかチェック
- 出力された評価関数ファイルをエンジンに読み込ませて動作確認

## 4.1 出力を確認する

学習完了後、出力ディレクトリ (例: `checkpoints/NNUE_HALFKP-256x2-32-32/`) は以下のレイアウト:

```
checkpoints/NNUE_HALFKP-256x2-32-32/
├── learn.log                          ← 全 run / resume を連結した累積ログ
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

## 4.2 学習ログ (`learn.log`) の読み方

学習中・終了後の loss 推移は `<output>/learn.log` (累積) と各 `<output>/0NNN/learn.log` (各 save 時点の snapshot) に記録される。どちらも **同じ 9 列 CSV** フォーマット。

### どっちを見るか

- **トップレベル `<output>/learn.log`** — 全 run / resume を **連結した累積版**。普段はこれを見る。
- **各 `0NNN/learn.log`** — その save 時点までのスナップショット。学習途中に「save 0005 の時点でどうだったか」を見たいときに使う。

### CSV のサンプル

```csv
eval,epoch,superbatch,curr_batch,value_loss,lr,lambda,positions,teacher
NNUE_HALFKP-256x2-32-32,1,1,32,0.6234,0.001,1.000,524288,teachers/
NNUE_HALFKP-256x2-32-32,1,1,64,0.5891,0.001,1.000,1048576,teachers/
NNUE_HALFKP-256x2-32-32,1,1,96,0.5510,0.001,1.000,1572864,teachers/
...
NNUE_HALFKP-256x2-32-32,1,2,32,0.4523,0.001,1.000,100532224,teachers/
...
```

bullet は **32 batch ごとに 1 行** loss を記録する。デフォルトの `--batches-per-superbatch ≒ 6104` なら、1 superbatch あたり約 191 行。`curr_batch` 列が `batches_per_superbatch` (= 6104) に達すると `superbatch` が +1 されて `curr_batch` は 1 から再開する。

### 列の意味

| 列 | 意味 | 例 |
|---|---|---|
| `eval` | 出力ディレクトリ名と同じ `<eval-type>[-<arch>]` 形式 + マルチ component (KPPT 系) ではさらに `/<component>` | `NNUE_HALFKP-256x2-32-32` / `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` |
| `epoch` | run 内 epoch (1 始まり) | `1` |
| `superbatch` | epoch 内 superbatch (1 始まり)。`--batches-per-superbatch` (デフォルト 6104) batch ごとに +1 | `1`, `2`, ... |
| `curr_batch` | superbatch 内 batch (1 始まり)。bullet は 32 batch ごとに 1 行記録 | `32`, `64`, ..., `6104` |
| `value_loss` | bullet が記録する 32-batch 平均 loss | `0.234` |
| `lr` | その時点の学習率 (StepLR 由来) | `0.001` |
| `lambda` | `--lambda` 値 (1 run 内で定数、3 桁固定) | `1.000` |
| `positions` | 累計教師局面数 (**resume 跨ぎで累積**) | `524288` |
| `teacher` | `--teacher` の値 | `teachers/` |

NNUE 系は `--arch` を `eval` 列に含める (出力ディレクトリ名と同じ命名)。KPPT 系は `--arch` を使わないので `eval` 列に arch は出ない。

正確な仕様は [`spec/04-checkpoint-layout.md`](../../../spec/04-checkpoint-layout.md#learnlog-フォーマット) を参照。

### pandas で読む

```python
import pandas as pd

df = pd.read_csv("checkpoints/NNUE_HALFKP-256x2-32-32/learn.log")
print(df.shape)        # 全行数
print(df.tail())       # 最後の数行
print(df["value_loss"].describe())   # loss の統計
```

9 列 + CSV header 付きなので `pd.read_csv` で列名は自動取得される。

### 学習が順調かを見るチェックリスト

学習が正しく動いている場合の典型的な兆候:

1. **`value_loss` が概ね単調減少**
   - 学習開始直後は急に下がり、徐々に減衰
   - 1 superbatch を消化するごとに目に見えて下がるのが理想
   - 1 superbatch まわっても下がらない場合は `--lr` が大きすぎる、または教師サイズが学習器のキャパに対して小さすぎる可能性
   - **periodic な loss スパイク** (= 数百 batch ごとに急に跳ねる) が見える場合は、教師ファイルが事前シャッフルされていない可能性が高い。shuffle buffer の境界 (デフォルト 256MB buffer ≒ 約 410 batch ごと) で分布が突然変わって起きる。対処は [§2.2 教師ファイルは事前にシャッフルしておく](2-train.md#教師ファイルは事前にシャッフルしておく) を参照

2. **`lr` が `--lr-step` 周期で `--lr-gamma` 倍されている**
   - 例: `--lr 0.001 --lr-gamma 0.1 --lr-step 8` なら superbatch 1-8 で 0.001、9-16 で 0.0001、...
   - 期待通り step してないなら lr フラグの値を見直す

3. **`positions` が単調増加** (run 内、resume 跨いでも)
   - 1 superbatch 完了で約 1 億 (= `--batches-per-superbatch × --batch-size`)
   - 教師サイズと照らし合わせると「ちゃんと教師を全部読めているか」が分かる

4. **`superbatch` がきちんと進んでいるか**
   - 教師局面数が 1 億未満だと 1 周回しても `superbatch` は 1 のまま終わる (fallback save で 1 度だけ保存)。これは仕様
   - 大きな教師なら `curr_batch` が 6104 (デフォルト `--batches-per-superbatch`) に到達するごとに `superbatch` が +1 されているはず
   - `superbatch` がいつまでも 1 のままで `curr_batch` も小さい値で止まっているなら、loader が打ち切られている可能性 (旧 HCPE loader 極性バグ等)

### 簡単なプロット

```python
import matplotlib.pyplot as plt

# positions を時系列軸に
plt.figure(figsize=(12, 4))
plt.plot(df["positions"], df["value_loss"])
plt.xlabel("positions"); plt.ylabel("value_loss")
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

## 4.3 エンジンに組み込む

学習結果をやねうら王エンジンで動作確認する最小手順。

### NNUE 系 (`nn.bin`)

最新の `000N/nn.bin` をエンジンが探す場所に置く。やねうら王の場合、`EvalDir` オプションでパスを指定する:

```
# エンジン起動後、USI コマンドで:
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/NNUE_HALFKP-256x2-32-32/0005
isready
bench
```

または、`eval/nn.bin` という相対パスでエンジン側に置く場合は、`000N/nn.bin` をそのファイル名で配置する。

`isready` でロードが通れば学習結果が認識できている。`bench` の出力に nn.bin のハッシュが出るので、毎回違う数字になっていれば確かに違う重みを load していることが分かる。

### KPPT 系 (`KK_synthesized.bin` 等の 3 ファイル組)

最新 `000N/` ディレクトリそのものを `EvalDir` に指定する (3 ファイルが揃った状態のディレクトリを指す):

```
setoption name EvalDir value C:/shogi/BulletOu/checkpoints/KPPT/0005
isready
bench
```

3 ファイルすべてが揃っていない場合エンジンは load に失敗する点に注意。

### 学習結果が弱いとき

最初の学習は小さな教師で短い superbatch しか回していないので、評価の質はあまり期待しないこと。本格対局できるレベルにするには:
- 教師サイズを増やす (1 億 → 10 億局面以上)
- `--max-epochs 3` 程度で複数周回す
- `--save-rate` を大きく (例: 10) して、後半の save だけを使う

詳細なハイパーパラメータ調整は各 eval-type のリファレンス ([halfkp.md](../shogi/halfkp.md) / [kp.md](../shogi/kp.md) / [halfkpe9.md](../shogi/halfkpe9.md) / [kppt.md](../shogi/kppt.md)) を参照。

## 4.4 次のステップ

- [リファレンス: NNUE HalfKP 学習](../shogi/halfkp.md) — `nn.bin` のバイナリレイアウト、量子化、resume の詳細
- [リファレンス: NNUE K-P 学習](../shogi/kp.md) — HalfKP との比較、入力 feature の構造
- [リファレンス: NNUE HalfKPE9 学習](../shogi/halfkpe9.md) — 利き数情報拡張版
- [リファレンス: KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習
- [仕様: spec/](../../../spec/) — eval-type 一覧 / バイナリレイアウト / hash 計算式 / `learn.log` フォーマット

---

前へ: [3. 学習をチューニング](3-tune.md)
