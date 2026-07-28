# 9. LayerStack — 局面ごとに別の小さなネットワークを使う

<a href="../../en/tutorial/9-layerstack.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

通常の NNUE は、どの局面でも同じ MLP を使って評価値を出します。SFNN の LayerStack は、同じ FeatureTransformer 出力を使いながら、局面の種類ごとに後段の小さな MLP stack を切り替える仕組みです。

大事なのは一点だけです。BulletOu と やねうら王 が、同じ局面に対して完全に同じ LayerStack index を計算しなければなりません。

## 9.1 まず全体像

LayerStack は、次の3つの軸を独立に組み合わせます。

| 軸 | 何を見るか | 指定できる token | bucket 数 |
|---|---|---|---:|
| hand | 先手番側/後手番側の持ち駒 | 省略, `hand64`, `hand256`, `hand1024` | 1 / 64 / 256 / 1024 |
| king | 手番側玉/非手番側玉の位置 | 省略, `k3k3`, `k9k9`, `k9k9z`, `k13k13z`, `k21k21`, `k29k29` | 1 / 9 / 81 / 81 / 169 / 441 / 841 |
| progress | 進行度 | 省略, `progress2`, `progress3`, `progress4`, `progress8`, `progress16`, `progress32` | 1 / 2 / 3 / 4 / 8 / 16 / 32 |

最終的な stack 数は掛け算です。

| 例 | hand | king | progress | LayerStacks |
|---|---:|---:|---:|---:|
| `SFNN_halfka2_1024_7_64` | 1 | 1 | 1 | 1 |
| `SFNN_halfka2_1024_7_64_k3k3` | 1 | 9 | 1 | 9 |
| `SFNN_halfka2_1024_7_64_k9k9z` | 1 | 81 | 1 | 81 |
| `SFNN_halfka2_1024_7_64_k13k13z` | 1 | 169 | 1 | 169 |
| `SFNN_halfka2_1024_7_64_hand256` | 256 | 1 | 1 | 256 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 256 | 9 | 1 | 2304 |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 1 | 9 | 8 | 72 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 256 | 9 | 16 | 36864 |

index の合成順は、やねうら王と同じです。

```text
idx = hand_bucket
idx = idx * king_bucket_count + king_bucket
idx = idx * progress_bucket_count + progress_bucket
```

つまり、`hand256_k3k3_progress16` なら、

```text
idx = (hand256_bucket * 9 + k3k3_bucket) * 16 + progress16_bucket
```

という意味になります。

## 9.2 architecture 名の書き方

LayerStack token は任意の順番で書けますが、BulletOu は保存名では `hand → king → progress` の順に正規化します。

| 入力として許可される名前 | BulletOu が扱う正規名 |
|---|---|
| `SFNN_halfka2_1024_7_64_k3k3_hand256_progress16` | `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` |
| `SFNN_halfka2_1024_7_64_progress8_k9k9z` | `SFNN_halfka2_1024_7_64_k9k9z_progress8` |
| `SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k13k13z` | そのまま |

grouped SFNN / common+shard L1 とも組み合わせられます。

```text
SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k3k3_progress16
```

この例は、`ka2` 入力、FT=3072、L1 は 1024 common + 256 x 8 shard、LayerStack は `hand256 × k3k3 × progress16` です。

## 9.3 king bucket の選び方

king bucket は、手番側玉と非手番側玉をそれぞれ正規化してから bucket 化し、組み合わせます。

| token | 1玉あたりの分類 | 最終 king buckets | ざっくりした性質 |
|---|---:|---:|---|
| 省略 | なし | 1 | 玉位置で分けない |
| `k3k3` | 3 | 9 | 最も軽い。玉の大まかな段だけを見る |
| `k9k9` | 9 | 81 | 玉の段をそのまま見る。file は見ない |
| `k9k9z` | 9 | 81 | `k9k9` と同じ bucket 数だが、自陣側では file/3 も見る |
| `k13k13z` | 13 | 169 | 1〜7段は段ごと、8〜9段は file/3 も見る |
| `k21k21` | 21 | 441 | 自陣深い2段で file を細かく見る |
| `k29k29` | 29 | 841 | 自陣深い3段で file を細かく見る |

long alias も使えます。たとえば `king9_by_king9`, `king9z_by_king9z`, `king9zone_by_king9zone`, `king13z_by_king13z`, `king13zone_by_king13zone` などです。

### `k3k3`

玉の段を3分割します。手番側玉と非手番側玉の組み合わせなので `3 x 3 = 9` buckets です。

|  | enemy rank 1-3 | enemy rank 4-6 | enemy rank 7-9 |
|---|---:|---:|---:|
| friend rank 1-3 | 0 | 1 | 2 |
| friend rank 4-6 | 3 | 4 | 5 |
| friend rank 7-9 | 6 | 7 | 8 |

### `k9k9`

玉の段をそのまま使います。file は見ません。

| 1玉あたり | 値 |
|---|---|
| rank 1 | 0 |
| rank 2 | 1 |
| ... | ... |
| rank 9 | 8 |

最終 bucket:

```text
bucket = friend_rank * 9 + enemy_rank
```

### `k9k9z`

`k9k9` と同じく合計は 81 buckets ですが、1玉あたりの9分類の意味が違います。遠い段は粗くまとめ、自陣側の深い段では file を3分割します。

| rank | file 1-3 | file 4-6 | file 7-9 |
|---|---:|---:|---:|
| 1-3 | 0 | 0 | 0 |
| 4-6 | 1 | 1 | 1 |
| 7 | 2 | 2 | 2 |
| 8 | 3 | 4 | 5 |
| 9 | 6 | 7 | 8 |

最終 bucket:

```text
bucket = friend_single * 9 + enemy_single
```

`k9k9` より「自陣深い場所で玉がどの筋にいるか」を少し見る、という設計です。

### `k13k13z`

1〜7段は段ごとに分け、8〜9段だけ file を3分割します。

| rank | file 1-3 | file 4-6 | file 7-9 |
|---|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 1 | 1 | 1 |
| 3 | 2 | 2 | 2 |
| 4 | 3 | 3 | 3 |
| 5 | 4 | 4 | 4 |
| 6 | 5 | 5 | 5 |
| 7 | 6 | 6 | 6 |
| 8 | 7 | 8 | 9 |
| 9 | 10 | 11 | 12 |

最終 bucket:

```text
bucket = friend_single * 13 + enemy_single
```

`k9k9z` より rank 情報を多く残しつつ、`k21k21` / `k29k29` より stack 数を抑える中間案です。

### `k21k21` / `k29k29`

この2つは、自陣深い段では file を完全に見る方式です。

| token | rank 1-3 | rank 4-6 | rank 7 | rank 8 | rank 9 | 1玉あたり |
|---|---:|---:|---:|---:|---:|---:|
| `k21k21` | 0 | 1 | 2 | 3〜11 | 12〜20 | 21 |
| `k29k29` | 0 | 1 | 2〜10 | 11〜19 | 20〜28 | 29 |

最終 bucket:

```text
k21k21: bucket = friend_single * 21 + enemy_single
k29k29: bucket = friend_single * 29 + enemy_single
```

## 9.4 hand bucket

hand bucket は、手番側の持ち駒 bucket と非手番側の持ち駒 bucket を組み合わせます。

| token | 片側の bucket 数 | 最終 hand buckets | 見ているもの |
|---|---:|---:|---|
| 省略 | 1 | 1 | 持ち駒を見ない |
| `hand64` | 8 | 64 | 持ち駒のざっくりした点数 |
| `hand256` | 16 | 256 | 4種類の存在bit |
| `hand1024` | 32 | 1024 | 5種類の存在bit |

### `hand64`

片側の持ち駒を点数化し、8 bucket に丸めます。

| 駒 | 点数 |
|---|---:|
| 歩 | 1 |
| 香 / 桂 | 2 |
| 銀 / 金 | 3 |
| 角 / 飛 | 5 |

```text
one_side_bucket = min((score + 3) / 4, 7)
bucket = stm_bucket * 8 + non_stm_bucket
```

### `hand256` と `hand1024`

| token | bit | 意味 |
|---|---:|---|
| `hand256` | bit0 | 歩/香/桂を持つ |
| `hand256` | bit1 | 銀/金を持つ |
| `hand256` | bit2 | 角を持つ |
| `hand256` | bit3 | 飛を持つ |
| `hand1024` | bit0 | 歩を持つ |
| `hand1024` | bit1 | 香/桂を持つ |
| `hand1024` | bit2 | 銀/金を持つ |
| `hand1024` | bit3 | 角を持つ |
| `hand1024` | bit4 | 飛を持つ |

```text
hand256:  bucket = stm_4bit_bucket * 16 + non_stm_4bit_bucket
hand1024: bucket = stm_5bit_bucket * 32 + non_stm_5bit_bucket
```

## 9.5 progress bucket

`progressN` は、やねうら王互換の SFNN progress parameter から `0..255` の進行度を計算し、それを N 個の bucket に割り当てます。

| token | progress buckets |
|---|---:|
| `progress2` | 2 |
| `progress3` | 3 |
| `progress4` | 4 |
| `progress8` | 8 |
| `progress16` | 16 |
| `progress32` | 32 |

```text
progress_bucket = min(progress_0_255 * progress_bucket_count / 256,
                      progress_bucket_count - 1)
```

`progressN` を使うと、export される `nn.bin` には Progress section が入ります。

| section | 内容 |
|---|---|
| header | NNUE/SFNN header |
| FeatureTransformer | L0 bias/weight |
| Progress | `0x6f50524f`, `bias_q16`, `weights_q16[81][1548]` |
| LayerStack network | stack 0, stack 1, ... |

既存の q16 progress parameter を使う場合だけ、`--sfnn-progress-params <file>` を指定します。省略した場合は zero parameter を書き出すので、全局面が中立付近の progress bucket に入ります。

注意: 現状の CUDA factorizer layout では、`progressN` と `king=axis` / `hand=axis` factorizer は併用できません。`shared` factorizer は併用できます。

## 9.6 どれを選ぶべきか

目安です。絶対の正解ではないので、同じ教師・同じ検証条件で比較してください。

| 目的 | 候補 | コメント |
|---|---|---|
| まず動作確認したい | suffix なし / `k3k3` | 軽く、失敗時の切り分けが楽 |
| `k3k3` より細かくしたいが、stack 数を抑えたい | `k9k9`, `k9k9z`, `k13k13z` | `k9k9z` は `k9k9` と同じ 81 stacks で意味だけ変わる |
| 玉位置をかなり細かく見たい | `k21k21`, `k29k29` | 教師量が少ないと各 stack の学習密度が落ちる |
| 持ち駒で局面を分けたい | `hand64`, `hand256`, `hand1024` | king bucket と掛け合わせると急激に巨大化する |
| 序盤/中盤/終盤で分けたい | `progress8`, `progress16` | progress parameter の扱いに注意 |

bucket を増やすほど表現力は増えますが、1 stack あたりに届く教師局面は減ります。たとえば `hand256_k13k13z` は `256 * 169 = 43264` stacks です。これはかなり大きいので、教師量・VRAM・checkpointサイズを見ながら試してください。

## 9.7 使用例

king zone bucket だけを試す例:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k13k13z \
    --teacher teachers/
```

hand と king zone を組み合わせる例:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand256_k9k9z \
    --teacher teachers/
```

hand / king / progress を全部組み合わせる例:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand256_k3k3_progress16 \
    --teacher teachers/
```

出力された `nn.bin` をやねうら王で読むときは、やねうら王側も同じ architecture suffix で build してください。

---

前へ: [8. エンジンに組み込む](8-engine.md)
