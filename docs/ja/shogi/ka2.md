# NNUE K-A2 学習

<a href="../../en/shogi/ka2.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[リファレンス目次へ戻る](../README.md)

`--eval-type NNUE_KA2` は、入力特徴量に `FeatureSet<Features::K, Features::A2>` を使うやねうら王 NNUE を学習する (architecture ファイルは `source/eval/nnue/architectures/ka2_*.h`、`nnue_arch_gen.py` で自動生成される)。ネットワーク本体は [HalfKP](halfkp.md) / [K-P](kp.md) と同じ 4 層 ClippedReLU で、入力特徴量だけが違う。

関連: `--eval-type SFNN_KA2` は同じ KA2 入力を SFNN-1536 architecture (LayerStacks=9) で学習する版 ([§4.4](../tutorial/4-train.md) と [SFNN-1536](sfnn-1536.md))。

## アーキテクチャ

L1 / L2 / L3 サイズは `--arch NNUE_ka2_<L1>x2_<L2>_<L3>` で指定する。デフォルトは K-P 系と同じ `NNUE_ka2_256x2_32_32` だが、自由形式の利点を活かして例えば `--arch NNUE_ka2_256x2_64_64` のように **後段の hidden を厚く**して、HalfKA に比べて KA2 が持たない king-anchor クロス積を後段で多少補う、という使い方ができる。以下はデフォルト構成の図:

```
将棋の局面
        │
        │  K + A2 sparse 入力 (1,791 次元 / perspective)
        ▼
   L0 affine + ClippedReLU                ← 両 perspective で重み共有
        ▼
   accumulator (256 次元 × 2 perspective = 連結して 512 次元)
        │
        │  L1 affine (512 → 32) + ClippedReLU
        ▼
        │  L2 affine (32 → 32) + ClippedReLU
        ▼
        │  Out affine (32 → 1)
        ▼
      eval (centipawn ベースのスカラー)
```

## 入力特徴量

`FeatureSet<K, A2>` は 2 つのやねうら王 feature set の合成:

| サブ feature | 次元 | 最大 active | Hash | 意味 |
|---|---|---|---|---|
| **K** (`features/k.h`) | 162 (= 81 × 2) | 2 | `0xD3CEE169` | 自玉位置 (0..80) + 相手玉位置 (0..80) |
| **A2** (`features/a2.h`) | 1629 (= `e_king`) | 40 | `0xA20DCB9B` | 玉を含む全駒の BonaPiece 値、ただし後手玉を自玉 plane に collapse する v2 エンコーディング |

perspective ごとの合計: **1791 次元**、最大 active **42** (K の 2 + A2 の 40)。**両玉とも perspective ごとに 2 回発火する** (K 領域で「玉」として 1 回、A2 領域で「全 40 駒のうちの 1 つ」として 1 回 — v2 collapse 後)。この「2 重発火」は `FeatureSet<K, A2>` の意図通り。

`FeatureSet<Head, Tail>` (`feature_set.h`) は Tail の index を先に並べ、Head の index に `Tail::kDimensions` を加算するので:

- index `0 .. 1547`: A2 玉以外の BonaPiece 値 (P と同じ範囲)
- index `1548 .. 1628`: A2 玉 plane (両玉が collapse、自玉も後手玉もここに発火)
- index `1629 .. 1709`: K 自玉 (perspective から見た 0..80 のマス)
- index `1710 .. 1790`: K 相手玉 (perspective から見た 0..80 のマス)

合成 feature hash (`nn.bin` ヘッダーに入る):
```
FeatureSet<K, A2>::kHashValue
  = K::kHashValue ^ (A2::kHashValue << 1) ^ (A2::kHashValue >> 31)
  = 0xD3CEE169 ^ 0x441B9736 ^ 0x00000001
  = 0x97D5765E
```

## K-P / HalfKA との比較

| | K-P | **K-A2** | HalfKA_hm2 |
|---|---|---|---|
| perspective あたり入力次元 | 1,710 | **1,791** | 73,305 |
| 最大 active | 40 | **42** | 40 |
| 玉を駒特徴量にも入れるか | いいえ (P は玉を除く) | **はい** (A2 が両玉を含む、v2 collapse 後) | はい |
| 入力層に 玉×駒 のクロス積があるか | いいえ | **いいえ** | はい (HalfKA が直接埋め込む) |

K-A2 は表現力的に K-P と HalfKA_hm2 の中間: 駒特徴量側にも両玉が現れる (K-P には無い情報) が、玉と他駒の相互作用を入力層でクロス積として持つわけではない (HalfKA はそれをやる)。実用上は loss 曲線が K-P に近い — 同じ `--arch` で K-P よりわずかに良い、HalfKP / HalfKA より明らかに弱い、という出方になる。

## 実際の使い方

### コマンド

標準の 4 層 NNUE (デフォルト `--arch NNUE_ka2_256x2_32_32`):

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_KA2 \
    --teacher teachers/ \
    --output checkpoints/my-ka2
```

hidden を厚くする (256x2 FT、64 次元 hidden):

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_KA2 \
    --arch NNUE_ka2_256x2_64_64 \
    --teacher teachers/
```

SFNN-1536 architecture を KA2 入力で:

```bash
./target/release/examples/bulletou \
    --eval-type SFNN_KA2 \
    --arch SFNN_ka2_1536_15_32_k3k3 \
    --teacher teachers/
```

スケジュール系フラグ、save layout、`state.bin` からの resume、トップレベル `learn.log` — その他はすべて [HalfKP](halfkp.md) と同一。`--eval-type` (および入力次元) だけが違う。

### 保存レイアウト

```
checkpoints/my-ka2/
├── learn.log
├── 0001/
│   ├── nn.bin                         ← やねうら王 / Stockfish (nnue-pytorch) 互換 NNUE バイナリ
│   ├── state.bin                      ← resume 用の重み + Adam moments
│   └── learn.log
├── 0002/
├── ...
└── 000N/
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

### `nn.bin` フォーマット

バイナリレイアウトは [HalfKP の `nn.bin`](halfkp.md#nnbin-フォーマット) と同じ。違いは:

- ヘッダーの `network_hash` と `feature_transformer_hash` が違う (`FEATURE_HASH_KA2 = 0x97D5765E` が混入される)
- ヘッダー `description` 文字列が `Features=K-A2(Friend)[1791->256x2],...`
- L0 のサイズ: `1791 × L1` (i16) (HalfKP は `125388 × L1`、K-P は `1710 × L1`)

L1 / L2 / Output 層は同じ `--arch` の下で HalfKP / K-P / K-A2 で byte-identical (同じサイズ、同じ i8 row-major SIMD パディング)。

### やねうら王での load

学習結果の `nn.bin` は、`--arch` の triple と一致するアーキテクチャヘッダで build されたやねうら王でしか load できない。対応する edition 名を `make` に渡してビルドする:

```bash
# NNUE_KA2 --arch NNUE_ka2_256x2_32_32 (デフォルト) の場合
make normal YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_ka2_256x2_32_32

# NNUE_KA2 --arch NNUE_ka2_256x2_64_64 の場合
make normal YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_ka2_256x2_64_64

# SFNN_KA2 --arch SFNN_ka2_1536_15_32_k3k3 の場合
make normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_ka2_1536_15_32_k3k3
```

やねうら王の `Makefile` は未知の edition について `nnue_arch_gen.py` を自動実行するので、対応するヘッダはビルド時に動的生成される。edition 名の dim 部分には**ハイフンではなくアンダースコア**を使う (clang の `-Wc99-extensions` 警告を避けるため)。load 手順は [§8 Engine](../tutorial/8-engine.md) を参照。

### 主要 CLI フラグ

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--eval-type` | `NNUE_KA2` (4 層) または `SFNN_KA2` (LayerStacks-1536) | (必須) |
| `--arch` | `NNUE_ka2_<L1>x2_<L2>_<L3>` (`L1` は 32 の倍数) | `NNUE_ka2_256x2_32_32` |
| `--teacher` | 教師ファイル / ディレクトリ / カンマ区切り | (必須) |
| `--output` | チェックポイント親ディレクトリ | `checkpoints/<eval-type>-<arch>` (例: `checkpoints/NNUE_KA2-NNUE_ka2_256x2_32_32`) |

全フラグ一覧は [HalfKP 学習](halfkp.md) を参照 (NNUE family 全体で同じ)。
