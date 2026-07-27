# 9. LayerStack — 局面ごとに別のサブネットを使う

<a href="../../en/tutorial/9-layerstack.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

通常の NNUE は、どの局面でも 1 つの MLP で評価値を出します。SFNN の LayerStack は、複数の小さな MLP stack を持ち、局面ごとに 1 つを選んで使います。

重要なのは、BulletOu とやねうら王で LayerStack index の計算が完全に一致していることです。

## 9.1 LayerStack の軸

やねうら王側では、LayerStack 選択を次の 3 軸として扱います。

| 軸 | 指定できる token | bucket 数 |
|---|---|---:|
| hand | 省略, `hand64`, `hand256`, `hand1024` | 1 / 64 / 256 / 1024 |
| king | 省略, `k3k3`, `k9k9`, `k21k21`, `k29k29` | 1 / 9 / 81 / 441 / 841 |
| progress | 省略, `progress2`, `progress3`, `progress4`, `progress8`, `progress16`, `progress32` | 1 / 2 / 3 / 4 / 8 / 16 / 32 |

最終的な stack 数は次の積です。

```text
LayerStacks = hand_bucket_count * king_bucket_count * progress_bucket_count
```

実行時の index 合成順は、やねうら王と同じです。

```text
idx = hand_bucket
idx = idx * king_bucket_count + king_bucket
idx = idx * progress_bucket_count + progress_bucket
```

architecture 名では token の順番を入れ替えても受け付けますが、BulletOu の表示・保存名では `hand`, `king`, `progress` の順に正規化されます。例えば、

```text
SFNN_halfka2_1024_7_64_k3k3_hand256_progress16
```

は受け付けられ、次の名前として扱われます。

```text
SFNN_halfka2_1024_7_64_hand256_k3k3_progress16
```

## 9.2 例

| `--arch` | LayerStacks | 意味 |
|---|---:|---|
| `SFNN_halfka2_1024_7_64` | 1 | 単一 stack |
| `SFNN_halfka2_1024_7_64_k3k3` | 9 | king 3x3 |
| `SFNN_halfka2_1024_7_64_k29k29` | 841 | king 29x29 |
| `SFNN_halfka2_1024_7_64_hand256` | 256 | hand256 のみ |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 2304 | hand256 x k3k3 |
| `SFNN_halfka2_1024_7_64_progress8` | 8 | progress8 のみ |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 72 | k3k3 x progress8 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 36864 | hand256 x k3k3 x progress16 |

grouped SFNN の表記とも組み合わせられます。

```text
SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k3k3_progress16
```

## 9.3 king bucket

`king3_by_king3`, `king9_by_king9`, `king21_by_king21`, `king29_by_king29` のような長い alias も受け付けます。

### `k3k3`

手番側視点に正規化したあと、自玉段・敵玉段をそれぞれ 3 区分に丸めます。

|  | 敵玉 1-3段 | 敵玉 4-6段 | 敵玉 7-9段 |
|---|---:|---:|---:|
| 自玉 1-3段 | 0 | 1 | 2 |
| 自玉 4-6段 | 3 | 4 | 5 |
| 自玉 7-9段 | 6 | 7 | 8 |

### `k9k9`

自玉段と敵玉段をそのまま使います。

```text
bucket = friend_rank * 9 + enemy_rank
```

### `k21k21`

玉 1 つを 21 bucket に分けます。

```text
if rank < 3: single = 0
else if rank < 6: single = 1
else if rank < 7: single = 2
else: single = 3 + (rank - 7) * 9 + file

bucket = friend_single * 21 + enemy_single
```

### `k29k29`

玉 1 つを 29 bucket に分けます。

```text
if rank < 3: single = 0
else if rank < 6: single = 1
else: single = 2 + (rank - 6) * 9 + file

bucket = friend_single * 29 + enemy_single
```

## 9.4 hand bucket

### `hand64`

片側の手駒を点数化して、次の式で bucket 化します。

```text
bucket_one_side = min((score + 3) / 4, 7)
bucket = side_to_move_bucket * 8 + non_side_bucket
```

点数は次の通りです。

- 歩: 1
- 香/桂: 2
- 銀/金: 3
- 角/飛: 5

### `hand256`

片側の手駒を 4bit の有無で見ます。

- bit0: 歩/香/桂 のいずれかを持つ
- bit1: 銀/金 のいずれかを持つ
- bit2: 角を持つ
- bit3: 飛を持つ

最終 bucket は次の式です。

```text
bucket = side_to_move_bucket * 16 + non_side_bucket
```

### `hand1024`

片側の手駒を 5bit の有無で見ます。

- bit0: 歩を持つ
- bit1: 香/桂 のいずれかを持つ
- bit2: 銀/金 のいずれかを持つ
- bit3: 角を持つ
- bit4: 飛を持つ

最終 bucket は次の式です。

```text
bucket = side_to_move_bucket * 32 + non_side_bucket
```

## 9.5 progress bucket

`progressN` は、やねうら王の SFNN progress parameter から `0..255` の進行度値を計算し、それを指定 bucket 数に丸めます。

```text
progress_bucket = min(progress_0_255 * progress_bucket_count / 256,
                      progress_bucket_count - 1)
```

`progressN` を使うとき、出力される `nn.bin` は次の順にセクションを持ちます。

```text
NNUE header
FeatureTransformer section
Progress section
LayerStack network section 0
LayerStack network section 1
...
```

Progress section はやねうら王互換です。

```text
section_hash = 0x6f50524f  # "oPRO"
int32 bias_q16
int32 weights_q16[81][1548]
```

既存の q16 `Progress::Parameters` payload を使いたい場合だけ、`--sfnn-progress-params <file>` を指定します。ファイル形式は次のどちらかです。

- `bias_q16 + weights_q16[81][1548]`
- `0x6f50524f + bias_q16 + weights_q16[81][1548]`

これは古い実験用の f64 `progress8kpabs` / `progress.bin` 形式ではありません。`progressN` architecture で `--sfnn-progress-params` を省略した場合は、やねうら王の dummy zero と同じく zero progress parameters を書き出します。この場合、全局面は中立付近の progress bucket に入ります。

現状、`--sfnn-factorizer shared` は `progressN` と併用できます。`king=axis` / `hand=axis` は、現在の CUDA factorizer layout が hand/king の 2 軸分解だけを持つため、`progressN` とは併用不可として弾きます。

## 9.6 使い方

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k3k3_progress8 \
    --teacher teachers/
```

hand/king/progress を組み合わせる例です。

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand256_k3k3_progress16 \
    --teacher teachers/
```

`--output` を省略した場合、checkpoint は正規化後の architecture 名から作られるディレクトリに出力されます。

## 9.7 注意点

- LayerStack は SFNN architecture でのみ意味があります。
- bucket 数が増えるほど、1 bucket あたりの教師局面密度が下がります。
- checkpoint サイズと VRAM 使用量は、おおむね LayerStacks 数に比例して増えます。
- 出力した `nn.bin` を読むやねうら王側も、同じ architecture suffix で build してください。

---

前へ: [8. エンジンに組み込む](8-engine.md)
