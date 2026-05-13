# 9. LayerStack — 局面ごとに別の評価関数を使い分ける

<a href="../../en/tutorial/9-layerstack.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

通常の NNUE は局面に関係なく 1 つの NN で評価値を出す。これに対して **LayerStack 系評価関数** は、局面に応じて **複数のサブネットワークを使い分ける**:

- 序盤 / 中盤 / 終盤、あるいは玉の位置関係などで、評価関数の傾向は実は異なるはず
- そこで「9 個の独立した小さな NN」を持っておき、局面ごとに 1 個だけを選んで評価値を出す
- どのサブネットを使うかの **バケット選択ロジック** をやねうら王エンジンと bulletou で揃える必要がある (= `--layerstack` で指定)

bulletou で LayerStack を使うのは現状 **やねうら王 SFNNwoP1536 ビルド向けの学習** (= `--eval-type SFNN_HALFKA1HM` / `SFNN_HALFKA2HM`) のみ。詳細仕様は [SFNN-1536 学習リファレンス](../shogi/sfnn-1536.md) を参照。

## 9.1 `--layerstack` の選択

| flag | バケット数 | やねうら王 load 可 | 説明 |
|---|---|---|---|
| **`king3-by-king3`** (デフォルト) | 9 | ○ | 自玉段を 3 区分 (1-3 / 4-6 / 7-9 段) × 敵玉段も 3 区分 = 9 通り。やねうら王の `stack_index_for_nnue` と完全一致 |

現状サポートしている `--layerstack` の値はこれ 1 つだけ。将来やねうら王側で別のバケット選択スキームが追加されれば、それに対応した値をここに足す。

### king3-by-king3 のバケット表

両玉の段 (perspective 反転後) を 3 区分にしてから組み合わせる:

|  | 敵玉 1-3 段 | 敵玉 4-6 段 | 敵玉 7-9 段 |
|---|---|---|---|
| **自玉 1-3 段** | bucket 0 | bucket 1 | bucket 2 |
| **自玉 4-6 段** | bucket 3 | bucket 4 | bucket 5 |
| **自玉 7-9 段** | bucket 6 | bucket 7 | bucket 8 |

各 bucket は独立した「fc_0 + fc_1 + fc_2」のセットを持ち、学習中はその bucket に分類された局面だけからその bucket の重みを更新する。

## 9.2 使う場面

LayerStack は **SFNN ファミリ専用** のオプションで、他の eval-type (`NNUE_HALFKP` / `NNUE_KP` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` / `KPPT` 系) では無視される (= 単一 NN 構造のため LayerStack 不要)。

```bash
# SFNN-1536 を king3-by-king3 = 9 バケットで学習
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch 1536x2-15-32 \
    --layerstack king3-by-king3 \
    --teacher teachers/
```

`--output` を省略すると `checkpoints/SFNN_HALFKA2HM-1536x2-15-32-king3-by-king3/` に書かれる (= `--eval-type` + `--arch` + `--layerstack` を連結した命名)。

学習自体のスケジューリング (`--lr` / `--superbatches` 等) は [§6 学習をチューニング](6-tune.md) と共通。結果の確認も [§7 結果を確認する](7-result.md) と同じ。

## 9.3 実機での load 確認

LayerStack の学習結果は通常の NNUE と同じく `nn.bin` として書かれ、やねうら王の **SFNNwoP1536 ビルド** で `setoption EvalDir → isready → bench` で動作確認する。`isready` 時に `info string Warning: NNUE hash mismatch` が出るが、これは想定動作 (load は続行される)。

詳細手順は [§8 エンジンに組み込む](8-engine.md) を参照。やねうら王の SFNN ビルドのビルド方法・USI オプションは [SFNN-1536 リファレンス](../shogi/sfnn-1536.md) も合わせて見ること。

## 9.4 「LayerStack を使わない方が良い」場合

LayerStack は 9 倍のサブネット重みを持つため、学習も推論も単一 NN より重い。

- 教師データが小さい (1 億局面未満など) 場合、9 バケットに分かれる局面数も少なくなり、各バケットの学習効率が落ちる
- やねうら王に投入する目的でなければ、`NNUE_HALFKP` / `NNUE_HALFKPVM` 等の単一 NN を使った方が手軽

実用的には:
- **やねうら王 SFNNwoP1536 互換の評価関数が欲しい** → SFNN_HALFKA2HM + LayerStack を使う
- それ以外 → 通常の NNUE 系で十分

## 9.5 関連

- [SFNN-1536 学習リファレンス](../shogi/sfnn-1536.md) — アーキ・binary layout・量子化スケール
- 既存実装: `examples/shogi_layerstack.rs` — 9 バケット以外の (実験的) bucketing モードあり (rshogi 互換出力、bulletou と並行存続)

---

前へ: [8. エンジンに組み込む](8-engine.md)
