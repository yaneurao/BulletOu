# 9. LayerStack — 局面ごとに別のサブネットを使う

<a href="../../en/tutorial/9-layerstack.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

通常の NNUE は、局面に関係なく 1 つの MLP で評価値を出します。これに対して **LayerStack 系の評価関数** は、複数の小さなサブネットを持ち、局面ごとに 1 つを選んで使います。

- 序盤・中盤・終盤、あるいは玉位置や手駒状態によって、評価関数に欲しい形が変わることがあります。
- そこで bucket ごとに独立した `fc_0 + fc_1 + fc_2` 重みを持ち、推論時に局面から bucket を選びます。
- bucket 選択ロジックは、やねうら王側と BulletOu 側で完全に一致している必要があります。

BulletOu では、LayerStack は SFNN family で使います。`--arch` の末尾 suffix が、やねうら王 build 側の bucket 選択アルゴリズムに対応します。

## 9.1 LayerStack suffix の選択

| `--arch` suffix | buckets | やねうら王でload可 | 説明 |
|---|---:|---|---|
| **`k3k3` / `king3_by_king3`** (default) | 9 | ○ | 自玉段を3区分 × 敵玉段を3区分 |
| **`k9k9` / `king9_by_king9`** | 81 | ○ | 自玉段そのもの × 敵玉段そのもの |
| **`hand64`** | 64 | ○ | 手番側/非手番側の手駒スコア8段階 |
| **`hand64_k3k3` / `hand64_king3_by_king3`** | 576 | ○ | `hand64` × `k3k3` |
| **`hand64_k9k9` / `hand64_king9_by_king9`** | 5184 | ○ | `hand64` × `k9k9` |
| **`hand256`** | 256 | ○ | 手番側/非手番側の4bit手駒有無 bucket |
| **`hand256_k3k3` / `hand256_king3_by_king3`** | 2304 | ○ | `hand256` × `k3k3` |
| **`hand256_k9k9` / `hand256_king9_by_king9`** | 20736 | ○ | `hand256` × `k9k9` |
| **`hand1024`** | 1024 | ○ | 手番側/非手番側の5bit手駒有無 bucket |
| **`hand1024_k3k3` / `hand1024_king3_by_king3`** | 9216 | ○ | `hand1024` × `k3k3` |
| **`hand1024_k9k9` / `hand1024_king9_by_king9`** | 82944 | ○ | `hand1024` × `k9k9`。VRAM と checkpoint サイズが非常に大きい |

### k3k3 bucket

手番側から見た自玉段・敵玉段をそれぞれ 3 区分に丸め、9 通りにします。

|  | 敵玉 1-3段 | 敵玉 4-6段 | 敵玉 7-9段 |
|---|---:|---:|---:|
| **自玉 1-3段** | 0 | 1 | 2 |
| **自玉 4-6段** | 3 | 4 | 5 |
| **自玉 7-9段** | 6 | 7 | 8 |

### k9k9 bucket

手番側から見た自玉段と敵玉段をそのまま使い、`self_rank * 9 + enemy_rank` で 81 通りにします。

### hand64 bucket

片側の手駒を次のスコアに変換し、`bucket = min((score + 3) / 4, 7)` とします。

- 歩: 1
- 香/桂: 2
- 銀/金: 3
- 角/飛: 5

最終 bucket は `手番側 bucket * 8 + 非手番側 bucket` です。

### hand256 bucket

片側の手駒を 4bit の有無に変換します。

- bit0: 歩/香/桂 のいずれかを持つ
- bit1: 銀/金 のいずれかを持つ
- bit2: 角を持つ
- bit3: 飛を持つ

最終 bucket は `手番側 bucket * 16 + 非手番側 bucket` です。

### hand1024 bucket

片側の手駒を 5bit の有無に変換します。

- bit0: 歩を持つ
- bit1: 香/桂 のいずれかを持つ
- bit2: 銀/金 のいずれかを持つ
- bit3: 角を持つ
- bit4: 飛を持つ

最終 bucket は `手番側 bucket * 32 + 非手番側 bucket` です。

手駒 bucket と king bucket を組み合わせる場合、やねうら王と同じく `hand_bucket * king_bucket_count + king_bucket` の順で index を作ります。

## 9.2 使い方

```bash
# k3k3 = 9 stacks
./target/release/examples/bulletou \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/

# hand256 bucket split
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand256 \
    --teacher teachers/

# common+shard L1 と hand256_k3k3 の組み合わせ
./target/release/examples/bulletou \
    --arch SFNN_ka2_3072_7_64_c1024_s256x8_hand256_k3k3 \
    --teacher teachers/
```

## 9.3 注意点

LayerStack は bucket ごとにサブネット重みを持つため、単一 MLP より学習・推論・保存が重くなります。特に `hand1024_k9k9` は 82,944 stacks なので、まずは小さめの FT/H1 サイズ、または `hand256` / `hand1024` 単体から試すのが安全です。

- 教師局面が少ないと、1 bucket あたりの学習密度が落ちます。
- bucket 数が増えるほど checkpoint サイズと VRAM 使用量が増えます。
- やねうら王側も同じ architecture suffix で build してください。

---

前へ: [8. エンジンに組み込む](8-engine.md)
