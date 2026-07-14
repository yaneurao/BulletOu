# NNUE K-P 学習

<a href="../../en/shogi/kp.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[リファレンス目次へ戻る](../README.md)

`--eval-type NNUE_KP` は、やねうら王の `kp_256x2-32-32` NNUE を学習する。ネットワーク本体 (4 層 ClippedReLU) は [HalfKP](halfkp.md) と完全に同一で、**入力特徴量だけが違う**。

やねうら王側の architecture ファイルは `source/eval/nnue/architectures/kp_256x2-32-32.h` で、`RawFeatures = FeatureSet<Features::K, Features::P>` と宣言されている。

## アーキテクチャ

L1 / L2 / L3 サイズは `--arch NNUE_kp_<L1>x2_<L2>_<L3>` で指定する (NNUE_HALFKP と同じよく使われるサイズが共通、詳細は [§4.3](../tutorial/4-train.md#43---arch-を指定する))。やねうら王が実エンジンとして配布している `NNUE_kp_*` バイナリは現状 `256x2-32-32` のみだが、学習側は他サイズでも生成できる (実験用)。CLI では `NNUE_kp_256x2_32_32` のように完全名で書く。以下はデフォルト構成の図:

```
将棋の局面
        │
        │  K + P sparse 入力 (1,710 次元 / perspective)
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

`FeatureSet<K, P>` は 2 つのやねうら王 feature set の合成:

| サブ feature | 次元 | 最大 active | Hash | 意味 |
|---|---|---|---|---|
| **K** (`features/k.h`) | 162 (= 81 × 2) | 2 | `0xD3CEE169` | 自玉位置 (0..80) + 相手玉位置 (0..80) |
| **P** (`features/p.h`) | 1548 (= `fe_end`) | 38 | `0x764CFB4B` | 玉以外の駒の BonaPiece 値 |

perspective ごとの合計: **1710 次元**、最大 active **40** (玉 2 + 玉以外最大 38)。

`FeatureSet<Head, Tail>` (`feature_set.h`) は Tail の index を先に並べ、Head の index に `Tail::kDimensions` を加算するので:

- index `0 .. 1547`: P (玉以外の BonaPiece 値; index 0 は BonaPiece 慣習により未使用)
- index `1548 .. 1628`: K 自玉 (perspective から見た 0..80 のマス)
- index `1629 .. 1709`: K 相手玉 (perspective から見た 0..80 のマス)

合成 feature hash (`nn.bin` ヘッダーに入る):
```
FeatureSet<K, P>::kHashValue
  = K::kHashValue ^ (P::kHashValue << 1) ^ (P::kHashValue >> 31)
  = 0xD3CEE169 ^ 0xEC99F696
  = 0x3F5717FF
```

## HalfKP との比較

| | HalfKP | K-P |
|---|---|---|
| perspective あたり入力次元 | 125,388 (= 81 × 1548) | 1,710 (= 162 + 1548) |
| クロス積か | はい — (玉 × 駒) 各組合せが固有の特徴量 | いいえ — K と P は単に連結、特徴量レベルでは交差しない |
| L0 重みサイズ | 125,388 × 256 | 1,710 × 256 |
| 表現力 | 高 (玉 × 駒 の相関が直接埋め込まれる) | 低 (相関は L0+L1 を通じて学習する必要がある) |

K-P は HalfKP と並んで NNUE 系評価関数の最初期に追加された (どちらも同じ 4 層 ClippedReLU)。実用は HalfKP が主流 — クロス積入力のほうが強くなることが分かったため。K-P はアブレーション実験や軽量モデル比較用として残る。

## 実際の使い方

### コマンド

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_KP \
    --teacher teachers/ \
    --output checkpoints/my-kp
```

スケジュール系フラグ、save layout、`state.bin` からの resume、トップレベル `learn.log` — その他はすべて [HalfKP](halfkp.md) と同一。`--eval-type` だけが違う。

### 保存レイアウト

```
checkpoints/my-kp/
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

- ヘッダーの `network_hash` と `feature_transformer_hash` が違う (HalfKP の `0x5D69D5B8` ではなく `FEATURE_HASH_KP = 0x3F5717FF` が混入される)
- ヘッダー `description` 文字列が `Features=K-P(Friend)[1710->256x2],...` (HalfKP は `Features=HalfKP(Friend)[125388->256x2],...`)
- L0 のサイズ: `1710 × 256` (i16) (HalfKP は `125388 × 256`)

L1 / L2 / Output 層は同じ `--arch` preset の下で HalfKP と byte-identical (同じサイズ、同じ i8 row-major SIMD パディング)。

### 主要 CLI フラグ

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--eval-type` | `NNUE_KP` | (必須) |
| `--arch` | `NNUE_kp_256x2_32_32`、`NNUE_kp_384x2_8_96`、`NNUE_kp_512x2_8_64`、`NNUE_kp_768x2_16_64`、`NNUE_kp_1024x2_8_32`、`NNUE_kp_1024x2_8_64` | `NNUE_kp_256x2_32_32` |
| `--teacher` | 教師ファイル / ディレクトリ / カンマ区切り | (必須) |
| `--output` | チェックポイント親ディレクトリ | `checkpoints/<eval-type>-<arch>` (例: `checkpoints/NNUE_KP-NNUE_kp_256x2_32_32`) |

全フラグ一覧は [HalfKP 学習](halfkp.md) を参照 (NNUE_HALFKP と NNUE_KP で同じ)。
