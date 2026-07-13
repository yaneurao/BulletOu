# 9. LayerStack — 局面ごとに別の評価関数を使い分ける

<a href="../../en/tutorial/9-layerstack.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

通常の NNUE は局面に関係なく 1 つの NN で評価値を出す。これに対して **LayerStack 系評価関数** は、局面に応じて **複数のサブネットワークを使い分ける**:

- 序盤 / 中盤 / 終盤、あるいは玉の位置関係などで、評価関数の傾向は実は異なるはず
- そこで「9 個の独立した小さな NN」を持っておき、局面ごとに 1 個だけを選んで評価値を出す
- どのサブネットを使うかの **バケット選択ロジック** をやねうら王エンジンと bulletou で揃える必要がある (= `--arch` の suffix で指定)

bulletou で LayerStack を使うのは現状 **やねうら王 SFNNwoP1536 ビルド向けの学習** (= `--eval-type SFNN_HALFKA1HM` / `SFNN_HALFKA2HM`) のみ。詳細仕様は [SFNN-1536 学習リファレンス](../shogi/sfnn-1536.md) を参照。

## 9.1 LayerStack suffix の選択

| `--arch` suffix | バケット数 | やねうら王 load 可 | 説明 |
|---|---|---|---|
| **`k3k3(king3-by-king3)`** (デフォルト) | 9 | ○ | 自玉段を 3 区分 (1-3 / 4-6 / 7-9 段) × 敵玉段も 3 区分 = 9 通り。やねうら王の `stack_index_for_nnue` と完全一致 |

現状サポートしている suffix はこれ 1 つだけ。将来やねうら王側で別のバケット選択スキームが追加されれば、それに対応した suffix をここに足す。

### k3k3(king3-by-king3) のバケット表

両玉の段 (perspective 反転後) を 3 区分にしてから組み合わせる:

|  | 敵玉 1-3 段 | 敵玉 4-6 段 | 敵玉 7-9 段 |
|---|---|---|---|
| **自玉 1-3 段** | bucket 0 | bucket 1 | bucket 2 |
| **自玉 4-6 段** | bucket 3 | bucket 4 | bucket 5 |
| **自玉 7-9 段** | bucket 6 | bucket 7 | bucket 8 |

各 bucket は独立した「fc_0 + fc_1 + fc_2」のセットを持ち、学習中はその bucket に分類された局面だけからその bucket の重みを更新する。

## 9.2 使う場面

LayerStack は **SFNN ファミリ専用** で、他の eval-type (`NNUE_HALFKP` / `NNUE_KP` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` / `KPPT` 系) は単一 NN 構造のため LayerStack 不要。

```bash
# SFNN-1536 を k3k3(king3-by-king3) = 9 バケットで学習
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

`--output` を省略すると `checkpoints/SFNN_HALFKA2HM-SFNN_halfkahm2_1536_15_32_k3k3/` に書かれる (= `--eval-type` + `--arch` を連結した命名)。

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

## 9.5 双子ニューロン対策としての STE CReLU

LayerStack / SFNN のように入力特徴量が多い評価関数では、feature transformer 直後の CReLU が初期段階で 0 または 1 に張り付くと、複数のニューロンがほぼ同じ出力になり、後段から見て 1 つ分のニューロンのように振る舞うことがある。これを避ける実験用オプションとして `--sfnn-ste-crelu` がある。

```bash
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/ \
    --sfnn-ste-crelu
```

この指定は **学習時だけ** の挙動を変える。forward の評価値は通常の CReLU と同じなので、出力される `nn.bin` の推論形式は変わらない。違うのは feature transformer 直後の CReLU の backward で、クリップされた FT ニューロンにも勾配を通す点。L1後段やL2後段の CReLU は通常通りに扱う。標準 NNUE には適用されず、SFNN / LayerStack 系 (`SFNN_*`) 専用。

飽和状況を見たい場合は、以下も併用する:

```bash
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/ \
    --dump-activation-stats \
    --activation-stats-positions 4096
```

`--sfnn-ste-crelu` は resume 判定に含まれるため、ON/OFF を比較するときは `--tag` を分ける。

## 9.6 関連

- [SFNN-1536 学習リファレンス](../shogi/sfnn-1536.md) — アーキ・binary layout・量子化スケール
- 既存実装: `examples/shogi_layerstack.rs` — 9 バケット以外の (実験的) bucketing モードあり (rshogi 互換出力、bulletou と並行存続)

---

前へ: [8. エンジンに組み込む](8-engine.md)
