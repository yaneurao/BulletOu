# LayerStack — 局面ごとに別の小さなネットワークを使う

<a href="../../en/advanced/layerstack.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

通常の NNUE は、どの局面でも同じ後段ネットワークを使って評価値を出します。SFNN の LayerStack は、入力側の大きな変換部分は共有し、その後ろの小さなネットワークを局面の種類ごとに切り替える仕組みです。

たとえば `k3k3` なら玉位置の粗い分類で 9 個の後段ネットワークを持ち、局面ごとにそのうち 1 つを使います。`hand256_k3k3` なら、持ち駒分類 256 通りと玉位置分類 9 通りを掛け合わせて、`256 * 9 = 2304` 個の後段ネットワークを持ちます。

重要なのは、BulletOu と やねうら王が同じ局面に対して同じ番号を選ぶことです。ここが 1 つでもずれると、学習した `nn.bin` をエンジン側で読めても、別の後段ネットワークを参照してしまいます。

この章では、次の言葉を使います。

| 言葉 | この章での意味 |
|---|---|
| LayerStack | 局面ごとに切り替える後段ネットワークの集まり |
| stack | LayerStack の中の 1 個の後段ネットワーク |
| bucket | 玉位置・持ち駒・進行度などで局面を分類した番号 |
| `k3k3` / `hand256` など | architecture 名に付ける「分け方」の指定 |

## 1. LayerStack が何をしているか

LayerStack は「入力特徴量を変える」仕組みではありません。入力特徴量と FeatureTransformer は同じまま、後段だけを切り替えます。

| 部分 | LayerStack で共有されるか | 説明 |
|---|---|---|
| 入力特徴量 | 共有 | `halfka2` / `ka2` などは変わらない |
| FeatureTransformer | 共有 | 入力特徴量を最初に変換する大きな部分。L0 の重みは全 stack 共通 |
| 後段ネットワーク | stack ごとに別 | L1/L2/L3 相当の小さなネットワークを bucket ごとに持つ |
| 出力値 | 局面ごとに 1 stack だけ使う | 選ばれた stack の出力を評価値にする |

直感的には、次のような分業です。

| 分け方 | 狙い |
|---|---|
| 玉位置で分ける | 自玉/相手玉の位置によって評価の癖を変える |
| 持ち駒で分ける | 駒の持ち合い、終盤度、攻め駒の有無で評価の癖を変える |
| 進行度で分ける | 序盤/中盤/終盤で評価の癖を変える |

ただし、bucket を増やすほど 1 stack あたりに届く教師局面は減ります。表現力は増えますが、教師密度は落ちます。ここが LayerStack の一番大きなトレードオフです。

## 2. 3つの軸を掛け合わせる

BulletOu の SFNN LayerStack は、次の3つの軸を独立に組み合わせます。

| 分け方 | 何を見るか | architecture 名に付ける文字列 | bucket 数 |
|---|---|---|---:|
| hand | 手番側/非手番側の持ち駒 | 省略, `hand4`, `hand16`, `hand64`, `hand64z`, `hand256`, `hand1024` | 1 / 4 / 16 / 64 / 64 / 256 / 1024 |
| king | 手番側玉/非手番側玉の位置 | 省略, `k3k3`, `k9k9`, `k9k9z`, `k13k13z`, `k21k21`, `k29k29` | 1 / 9 / 81 / 81 / 169 / 441 / 841 |
| progress | 進行度 | 省略, `progress2`, `progress3`, `progress4`, `progress8`, `progress16`, `progress32` | 1 / 2 / 3 / 4 / 8 / 16 / 32 |

最終的な stack 数は掛け算です。

| architecture | hand | king | progress | LayerStacks |
|---|---:|---:|---:|---:|
| `SFNN_halfka2_1024_7_64` | 1 | 1 | 1 | 1 |
| `SFNN_halfka2_1024_7_64_k3k3` | 1 | 9 | 1 | 9 |
| `SFNN_halfka2_1024_7_64_k9k9z` | 1 | 81 | 1 | 81 |
| `SFNN_halfka2_1024_7_64_k13k13z` | 1 | 169 | 1 | 169 |
| `SFNN_halfka2_1024_7_64_k21k21` | 1 | 441 | 1 | 441 |
| `SFNN_halfka2_1024_7_64_k29k29` | 1 | 841 | 1 | 841 |
| `SFNN_halfka2_1024_7_64_hand4` | 4 | 1 | 1 | 4 |
| `SFNN_halfka2_1024_7_64_hand16` | 16 | 1 | 1 | 16 |
| `SFNN_halfka2_1024_7_64_hand64` | 64 | 1 | 1 | 64 |
| `SFNN_halfka2_1024_7_64_hand64z` | 64 | 1 | 1 | 64 |
| `SFNN_halfka2_1024_7_64_hand256` | 256 | 1 | 1 | 256 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 256 | 9 | 1 | 2304 |
| `SFNN_halfka2_1024_7_64_k3k3_progress8` | 1 | 9 | 8 | 72 |
| `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` | 256 | 9 | 16 | 36864 |

最終番号を作る順番は、やねうら王と同じく `hand → king → progress` です。

```text
idx = hand_bucket
idx = idx * king_bucket_count + king_bucket
idx = idx * progress_bucket_count + progress_bucket
```

たとえば `hand256_k3k3_progress16` は、

```text
idx = (hand256_bucket * 9 + k3k3_bucket) * 16 + progress16_bucket
```

という意味です。

## 3. architecture 名の書き方

`hand256` や `k3k3` などの指定は任意の順番で書けます。ただし BulletOu は、出力ディレクトリ名では `hand → king → progress` の順に整理します。

| 入力として許可される名前 | 出力ディレクトリ名で使われる名前 |
|---|---|
| `SFNN_halfka2_1024_7_64_k3k3_hand256_progress16` | `SFNN_halfka2_1024_7_64_hand256_k3k3_progress16` |
| `SFNN_halfka2_1024_7_64_progress8_k9k9z` | `SFNN_halfka2_1024_7_64_k9k9z_progress8` |
| `SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k13k13z` | そのまま |

SFNN では、中央の `H1` の値で L1 の shortcut の有無も決まります。

| 例 | fc0 の出力数 | shortcut |
|---|---:|---|
| `SFNN_halfka2_1024_7_64_k3k3` | 8 (`7 + 1`) | あり |
| `SFNN_halfka2_1024_8_64_k3k3` | 8 | なし |
| `SFNN_halfka2_1024_15_64_k3k3` | 16 (`15 + 1`) | あり |
| `SFNN_halfka2_1024_16_64_k3k3` | 16 | なし |

つまり、`H1 = 8n - 1` の形では shortcut 用の出力が 1 つ追加され、`H1 = 8n` の形では追加されません。`c0_s1024x4` のような L1 分割は、実際の fc0 出力数を分割します。そのため `4096_7_64_c0_s1024x4` は 8 出力を 4 分割し、`4096_8_64_c0_s1024x4` も 8 出力を 4 分割します。

`k3k3` や `hand256` などを何も付けない名前も有効です。

```text
SFNN_halfka2_1024_7_64
```

この場合は LayerStacks=1、つまり玉位置・持ち駒・進行度で分けません。

L1 層を分割する指定とも組み合わせられます。

```text
SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k3k3_progress16
```

この例は、`ka2` 入力、FT=3072、L1 は共有 1024 channel + 256 channel × 8 個、LayerStack は `hand256 × k3k3 × progress16` です。

## 4. king bucket を読む前の座標ルール

king bucket は、手番側玉と非手番側玉を別々に bucket 化してから組み合わせます。

ここで重要なのは、玉の座標をその玉の陣営から見た向きに正規化していることです。

| 用語 | 意味 |
|---|---|
| 手番側玉 | いま手番の側の玉 |
| 非手番側玉 | 相手側の玉 |
| 正規化 rank 1 | その玉から見て敵陣側 |
| 正規化 rank 9 | その玉から見て自陣最深部 |
| 正規化 file 1〜9 | その玉の陣営から見た筋 |

つまり、先手番でも後手番でも「自玉が自陣深くにいる」状態は同じように rank 8〜9 付近として扱われます。これにより、先手用と後手用で別の規則を作らずに済みます。

最終的な king bucket は、基本的に次の形です。

```text
king_bucket = stm_king_single_bucket * single_bucket_count
            + non_stm_king_single_bucket
```

たとえば `k29k29` なら 1玉あたり 29 bucket なので、

```text
king_bucket = stm_king29 * 29 + non_stm_king29
```

です。`k29k29` の LayerStacks が 841 になるのは、`29 * 29 = 841` だからです。

## 5. king bucket の種類

king bucket は「玉位置をどれくらい細かく見るか」を決めます。

| 指定 | 1玉あたり | 両玉の組み合わせ | 何を重視するか |
|---|---:|---:|---|
| 省略 | 1 | 1 | 玉位置で分けない |
| `k3k3` | 3 | 9 | 玉の大まかな段だけを見る |
| `k9k9` | 9 | 81 | 玉の段をそのまま見る。筋は見ない |
| `k9k9z` | 9 | 81 | `k9k9` と同じ 81 stacks のまま、自陣深部の筋情報を入れる |
| `k13k13z` | 13 | 169 | 1〜7段は段ごと、8〜9段は筋3分割 |
| `k21k21` | 21 | 441 | 8〜9段を全マス区別する |
| `k29k29` | 29 | 841 | 7〜9段を全マス区別する |

`k9k9z` / `k13k13z` の `z` は zone の意味です。単純な rank 分割ではなく、重要そうな領域に解像度を寄せる bucket です。

長い名前でも指定できます。たとえば `king9_by_king9`, `king9z_by_king9z`, `king9zone_by_king9zone`, `king13z_by_king13z`, `king13zone_by_king13zone`, `king21_by_king21`, `king29_by_king29` などです。

## 6. `k3k3` — まず試す粗い玉 bucket

`k3k3` は、玉1つを「敵陣側・中央・自陣側」の3つに分けます。手番側玉と非手番側玉の組み合わせなので `3 * 3 = 9` buckets です。

| 正規化 rank | 1玉の分類 |
|---|---:|
| 1〜3 | 0 |
| 4〜6 | 1 |
| 7〜9 | 2 |

両玉を組み合わせると次のようになります。

| 手番側玉 \ 非手番側玉 | rank 1〜3 | rank 4〜6 | rank 7〜9 |
|---|---:|---:|---:|
| rank 1〜3 | 0 | 1 | 2 |
| rank 4〜6 | 3 | 4 | 5 |
| rank 7〜9 | 6 | 7 | 8 |

`k3k3` は軽く、教師密度も高いので、LayerStack の最初の比較対象として扱いやすいです。一方で、同じ rank 7〜9 にいる玉はすべて同じ扱いになるため、自陣深部で玉が何筋にいるかは区別できません。

## 7. `k9k9` と `k9k9z` — 同じ 81 stacks で何を表現するか

`k9k9` と `k9k9z` は、どちらも 1玉あたり 9 bucket、両玉で `9 * 9 = 81` stacks です。違いは、9 bucket の使い方です。

| 指定 | 9 bucket の使い方 | 向いている狙い |
|---|---|---|
| `k9k9` | rank 1〜9 をそのまま 9 分割 | 玉の段を素直に見たい |
| `k9k9z` | 遠い段を粗くまとめ、自陣 rank 8〜9 の筋を3分割 | 同じ 81 stacks のまま、自陣深部の横位置を見たい |

### `k9k9`

`k9k9` は 1玉の正規化 rank をそのまま bucket にします。

| 正規化 rank | 1玉の bucket |
|---|---:|
| 1 | 0 |
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |
| 5 | 4 |
| 6 | 5 |
| 7 | 6 |
| 8 | 7 |
| 9 | 8 |

筋は見ません。たとえば rank 9 なら、1筋でも5筋でも9筋でも同じ bucket 8 です。

`k9k9` はわかりやすい反面、実戦で重要になりやすい自陣深部の横位置を捨てています。

### `k9k9z`

`k9k9z` は、`k9k9` と同じ 81 stacks のまま、情報の置き場所を変えます。rank 1〜6 をかなり粗くまとめ、そのぶん rank 8〜9 で筋方向の情報を持ちます。

1玉の bucket は次の通りです。

| 正規化 rank | file 1〜3 | file 4〜6 | file 7〜9 | 説明 |
|---|---:|---:|---:|---|
| 1〜3 | 0 | 0 | 0 | 敵陣側はまとめる |
| 4〜6 | 1 | 1 | 1 | 中央付近もまとめる |
| 7 | 2 | 2 | 2 | 自陣寄りだが筋はまだ見ない |
| 8 | 3 | 4 | 5 | 自陣深部なので筋を3分割 |
| 9 | 6 | 7 | 8 | 自陣最深部なので筋を3分割 |

`k9k9` との違いを一言で言うと、こうです。

| 領域 | `k9k9` | `k9k9z` |
|---|---|---|
| rank 1〜6 | rank を細かく見る | 粗くまとめる |
| rank 7 | rank だけ見る | rank だけ見る |
| rank 8〜9 | rank だけ見る | rank と file/3 を見る |
| stack 数 | 81 | 81 |

つまり `k9k9z` は、bucket 数を増やさずに「自陣深部の玉の横位置」を見たいときの設計です。たとえば、同じ rank 9 でも玉が端にいるか中央にいるかで評価の癖を変えたい、という狙いです。

逆に、rank 1〜6 の差をかなり潰すので、そこが効く教師・局面集合では `k9k9` より悪くなる可能性もあります。`k9k9z` は 81 stacks の予算配分を変える指定であり、常に `k9k9` より良いとは限りません。

## 8. `k13k13z` — rank 情報を残しつつ自陣深部だけ zone 化する

`k13k13z` は 1玉を 13 bucket に分け、両玉で `13 * 13 = 169` stacks になります。

`k9k9z` より rank 情報を多く残します。rank 1〜7 は段ごとに分け、rank 8〜9 だけ file/3 で分けます。

| 正規化 rank | file 1〜3 | file 4〜6 | file 7〜9 | 説明 |
|---|---:|---:|---:|---|
| 1 | 0 | 0 | 0 | rank そのもの |
| 2 | 1 | 1 | 1 | rank そのもの |
| 3 | 2 | 2 | 2 | rank そのもの |
| 4 | 3 | 3 | 3 | rank そのもの |
| 5 | 4 | 4 | 4 | rank そのもの |
| 6 | 5 | 5 | 5 | rank そのもの |
| 7 | 6 | 6 | 6 | rank そのもの |
| 8 | 7 | 8 | 9 | 自陣深部なので筋を3分割 |
| 9 | 10 | 11 | 12 | 自陣最深部なので筋を3分割 |

`k13k13z` は、`k9k9z` だと粗すぎるが、`k21k21` / `k29k29` は重すぎる、というときの中間案です。

| 比較 | stack 数 | コメント |
|---|---:|---|
| `k9k9z` | 81 | 軽い。rank 1〜6 は粗い |
| `k13k13z` | 169 | rank 1〜7 を保持し、8〜9段だけ筋を見る |
| `k21k21` | 441 | 8〜9段を全マス区別 |
| `k29k29` | 841 | 7〜9段を全マス区別 |

## 9. `k21k21` と `k29k29` — 自陣付近の玉位置をかなり細かく見る

`k21k21` と `k29k29` は、zone というより「自陣の深い段をマス単位で見る」bucket です。

### `k21k21`

`k21k21` は 1玉を 21 bucket に分けます。

| 正規化 rank | file 1 | file 2 | file 3 | file 4 | file 5 | file 6 | file 7 | file 8 | file 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1〜3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4〜6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 7 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 8 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| 9 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |

rank 8〜9 の 2段 x 9筋 = 18マスをそのまま区別し、それ以外は粗くまとめます。

両玉では `21 * 21 = 441` stacks です。`k29k29` より軽く、自陣最深部の玉位置を見たいときに使います。

### `k29k29`

`k29k29` は 1玉を 29 bucket に分けます。

| 正規化 rank | file 1 | file 2 | file 3 | file 4 | file 5 | file 6 | file 7 | file 8 | file 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1〜3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4〜6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 7 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 8 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| 9 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 |

rank 7〜9 の 3段 x 9筋 = 27マスをそのまま区別し、rank 1〜3 と rank 4〜6 はそれぞれ1つにまとめます。

両玉では `29 * 29 = 841` stacks です。

`k29k29` の狙いは、通常局面でよく現れる「自玉が自陣付近にいる状態」を細かく見ることです。たとえば、同じ自陣でも玉が 7段目に上がっているのか、9段目に深くいるのか、端寄りなのか中央なのかを別 stack にできます。

一方で、stack 数は `k3k3` の約 93 倍です。

| 比較 | LayerStacks | `k3k3` 比 |
|---|---:|---:|
| `k3k3` | 9 | 1.0x |
| `k9k9` / `k9k9z` | 81 | 9.0x |
| `k13k13z` | 169 | 18.8x |
| `k21k21` | 441 | 49.0x |
| `k29k29` | 841 | 93.4x |

したがって、`k29k29` は教師量が足りないと accuracy が伸びるのが遅くなったり、検証 loss が不安定になったりします。表現力は高いですが、十分な教師局面と、必要なら factorizer を併用して比較するのが前提になります。

## 10. hand bucket

hand bucket は、手番側の持ち駒 bucket と非手番側の持ち駒 bucket を組み合わせます。

| 指定 | 片側の bucket 数 | 最終 hand buckets | 見ているもの |
|---|---:|---:|---|
| 省略 | 1 | 1 | 持ち駒を見ない |
| `hand4` | 2 | 4 | 角を持っているか |
| `hand16` | 4 | 16 | 歩・角を持っているか |
| `hand64` | 8 | 64 | 3種類の存在bit |
| `hand64z` | 8 | 64 | 持ち駒のざっくりした点数 zone |
| `hand256` | 16 | 256 | 4種類の存在bit |
| `hand1024` | 32 | 1024 | 5種類の存在bit |

`hand64` と `hand64z` はどちらも最終的には64 bucketですが、見ている情報が違います。`hand64` は持ち駒の種類グループ、`hand64z` は持ち駒の点数 zone を見ます。

### `hand4` と `hand16`

`hand4` / `hand16` は、軽い hand bucket です。片側の bucket を作り、最後に手番側と非手番側を掛け合わせます。

| 指定 | 片側 bucket の作り方 | 最終 bucket |
|---|---|---|
| `hand4` | `角を持つ ? 1 : 0` | `stm_1bit * 2 + non_stm_1bit` |
| `hand16` | bit0=`歩を持つ`, bit1=`角を持つ` | `stm_2bit * 4 + non_stm_2bit` |

`hand4` は角持ちだけをかなり軽く分けたいとき、`hand16` は歩持ちも加えて少しだけ細かくしたいときの実験用です。

### `hand64`

`hand64` は、片側の持ち駒の有無を3つのグループで表します。

| bit | 意味 |
|---:|---|
| bit0 | 歩/香/桂を持つ |
| bit1 | 金/銀/飛を持つ |
| bit2 | 角を持つ |

片側 bucket は 3bit の `0..7` です。最終 bucket は次のように作ります。

```text
hand64_bucket = stm_3bit * 8 + non_stm_3bit
```

`hand64` は、角持ちを独立に見つつ、`hand256` / `hand1024` ほど細かく分けたくない場合に使います。

### `hand64z`

`hand64z` は、片側の持ち駒を点数化し、8 bucket に丸めます。

| 駒 | 点数 |
|---|---:|
| 歩 | 1 |
| 香 / 桂 | 2 |
| 銀 / 金 | 3 |
| 角 / 飛 | 5 |

片側 bucket は `min((score + 3) / 4, 7)` です。点数範囲で書くと次のようになります。

| 片側 bucket | score |
|---:|---|
| 0 | 0 |
| 1 | 1〜4 |
| 2 | 5〜8 |
| 3 | 9〜12 |
| 4 | 13〜16 |
| 5 | 17〜20 |
| 6 | 21〜24 |
| 7 | 25以上 |

最終 bucket は、

```text
hand64z_bucket = stm_bucket * 8 + non_stm_bucket
```

です。

`hand64z` は持ち駒の種類よりも「どれくらい持っているか」を粗く見たいときの bucket です。同じ64 bucketでも、`hand64` とは bucket の分け方が違います。

### `hand256` と `hand1024`

`hand256` / `hand1024` は、持ち駒の有無を bit で表します。

| 指定 | bit | 意味 |
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

| 指定 | 片側 bucket | 最終 bucket |
|---|---|---|
| `hand256` | 4bit, 0〜15 | `stm_4bit * 16 + non_stm_4bit` |
| `hand1024` | 5bit, 0〜31 | `stm_5bit * 32 + non_stm_5bit` |

`hand256` は軽めの種類分類、`hand1024` は歩の有無まで独立に見たい場合の分類です。king bucket と掛け合わせると stack 数が急増するので注意してください。

## 11. progress bucket

`progressN` は、局面から `0..255` の進行度値を計算し、それを N 個の bucket に割り当てる仕組みです。

architecture 名に `progress8` や `progress16` を付けると、LayerStack の第3の分け方として progress bucket が使われます。必要な進行度情報は、BulletOu が `nn.bin` の一部として出力します。

BulletOu は、手駒量と大駒成りをもとに進行度を計算します。これにより、BulletOu 学習時とやねうら王実行時の progress bucket が一致します。

| 指定 | progress buckets |
|---|---:|
| `progress2` | 2 |
| `progress3` | 3 |
| `progress4` | 4 |
| `progress8` | 8 |
| `progress16` | 16 |
| `progress32` | 32 |

bucket 化は次の考え方です。

```text
progress_bucket = min(progress_0_255 * progress_bucket_count / 256,
                      progress_bucket_count - 1)
```

たとえば `progress8` なら、進行度 `0..255` をほぼ 32 刻みで 8 個に分けます。`progress16` なら 16 刻みです。

`progressN` を使うと、出力される `nn.bin` には進行度計算用のデータも入ります。

| `nn.bin` 内の部分 | 内容 |
|---|---|
| header | NNUE/SFNN header |
| FeatureTransformer | L0 bias/weight |
| Progress | `0x6f50524f`, `bias_q16`, `weights_q16[81][1548]` |
| LayerStack network | stack 0, stack 1, ... |

`progressN` は `king=axis` / `hand=axis` factorizer と併用できます。この場合、factorizer は hand / king の分け方にだけ掛かります。progress 方向には掛かりません。たとえば `SFNN_halfka2_1024_7_64_k3k3_hand64_progress2` には `--sfnn-factorizer king=axis,hand=axis` を指定できます。

## 12. 組み合わせるとどれくらい大きくなるか

hand / king / progress は掛け算なので、少し足しただけで急に大きくなります。

| architecture 名に付ける指定 | LayerStacks | コメント |
|---|---:|---|
| なし | 1 | LayerStack なし |
| `k3k3` | 9 | 最小の king bucket |
| `k9k9z` | 81 | `k9k9` と同じ大きさ |
| `k13k13z` | 169 | 中間案 |
| `k29k29` | 841 | king だけならまだ試しやすい |
| `hand4` | 4 | 角持ちだけを見る軽量 hand bucket |
| `hand16` | 16 | 歩/角持ちを見る軽量 hand bucket |
| `hand64_k3k3` | 576 | 軽めの hand + king |
| `hand64z_k3k3` | 576 | 点数 zone による hand + king |
| `hand256_k3k3` | 2304 | よく比較対象にしやすい |
| `hand256_k9k9z` | 20736 | かなり大きい |
| `hand256_k13k13z` | 43264 | 教師量・VRAM・保存サイズに注意 |
| `hand1024_k29k29` | 861184 | 実験としては非常に重い |

大きい指定は、学習速度だけでなく、保存サイズ、検証時間、1 stack あたりの教師密度にも効きます。

## 13. どれを選ぶべきか

目安です。絶対の正解ではないので、同じ教師・同じ検証条件で比較してください。

| 目的 | 候補 | コメント |
|---|---|---|
| まず動作確認したい | 何も付けない / `k3k3` | 軽く、失敗時の切り分けが楽 |
| `k3k3` より細かくしたい | `k9k9` | 素直な rank 分割 |
| 81 stacks のまま自陣深部の筋を見たい | `k9k9z` | `k9k9` と同じ大きさだが予算配分が違う |
| `k9k9z` では粗すぎる | `k13k13z` | rank 1〜7 を保持し、8〜9段だけ zone 化 |
| 自陣最深部をマス単位で見たい | `k21k21` | `k29k29` より軽い |
| 自陣 7〜9段をマス単位で見たい | `k29k29` | 表現力は高いが教師密度が落ちる |
| 持ち駒で局面を分けたい | `hand4`, `hand16`, `hand64`, `hand64z`, `hand256`, `hand1024` | king bucket と掛け合わせると急激に巨大化する |
| 序盤/中盤/終盤で分けたい | `progress8`, `progress16` | hand / king と独立した第3軸として使える |

大きい bucket ほど「最初の accuracy の上がり」は遅く見えることがあります。これは必ずしもバグではなく、1 stack あたりの学習サンプルが減るためです。比較するときは、同じ教師量での短期 accuracy だけでなく、十分回した後の loss、実対局、保存サイズ、学習速度も合わせて見てください。

## 14. 使用例

もっとも軽い king bucket を試す例:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --teacher teachers/
```

`k9k9` と同じ 81 stacks のまま、自陣深部の筋を見たい例:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k9k9z \
    --teacher teachers/
```

`k29k29` で玉位置を細かく見る例:

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_k29k29 \
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

出力された `nn.bin` をやねうら王で読むときは、やねうら王側も同じ architecture 名で build してください。

---

前へ: [応用編トップ](README.md)
