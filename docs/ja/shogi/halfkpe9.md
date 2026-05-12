# NNUE HalfKPE9 学習

<a href="../../en/shogi/halfkpe9.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[リファレンス目次へ戻る](../README.md)

`--eval-type NNUE_HALFKPE9` は、やねうら王の `halfkpe9_*` 評価関数を学習する。HalfKP の `(自玉位置, 駒の BonaPiece)` ペアに加えて、**その駒が占めているマスへの利き数情報** を 9 通り (= 自軍利き 0/1/2 × 敵軍利き 0/1/2) ぶん多重化したもの。ネットワーク本体 (4 層 ClippedReLU dual-perspective) は HalfKP / K-P と同じ。

ネットワーク構造は HalfKP / K-P と同じだが、入力次元が **HalfKP の 9 倍** (125,388 → 1,128,492) になるため L0 weight matrix が 9 倍に膨らむ。GPU メモリと学習時間の見積もりは halfkp に比例して増える。

## アーキテクチャ

`--arch` で L1 / L2 / L3 サイズを選択 (NNUE_HALFKP と同じ preset 群):

```
将棋の局面
       │
       ▼ HalfKPE9 sparse 特徴量 (1,128,492 次元 = 81 × 1548 × 9)
       │
       ▼ L0 affine + ClippedReLU       ← 両 perspective で重み共有
       │
       ▼ accumulator (L1 × 2 perspective)
       │
       ▼ L1 affine + ClippedReLU
       ▼ L2 affine + ClippedReLU
       ▼ Out affine
       │
       ▼ eval (centipawn ベースのスカラー)
```

## 入力特徴量

`HalfKP × 9 effect-count buckets` の構造:

| 軸 | 範囲 | 意味 |
|---|---|---|
| **king_sq** | 0..80 | perspective から見た自玉のマス |
| **bonapiece** | 0..1547 | 駒の BonaPiece 値 (perspective 視点) |
| **effect bucket** | 0..8 | `(effect1 × 3 + effect2)` 利き数組合せ |

`effect1` = perspective から見て **自軍** がそのマスに与えている利き数 (0/1/2 にクリップ)、`effect2` = 同じく **敵軍** の利き数。

active index 計算式 (YaneuraOu の `MakeIndex` と完全一致):

```
index = fe_end × king_sq + bonapiece
      + fe_end × SQ_NB × (effect1 × 3 + effect2)
```

- `fe_end` = 1548
- `SQ_NB` = 81

### 利き計算

各教師局面につき、まず 81 マス × 2 色 = 162 セルの利き数 table を計算する (`compute_effect_counts`)。
- 玉を含む全駒種について `for_each_attack()` で利き先を列挙
- slider 駒 (角・飛・馬・竜) は遮蔽考慮で正しく扱われる (`shogi_halfka_hm_threat` モジュール由来の既存ルーチン)

`for_each_attack` は Threat 系 feature でも使われている共通ユーティリティで、HalfKPE9 のために新規実装したものではない。

### 手駒の扱い

手駒は盤上 sq を持たないので `effect1 = effect2 = 0` 固定 → `effect bucket = 0`。これは HalfKP と同じ index 領域に発火するので、(king_sq, hand_bonapiece, 0, 0) の組合せ部分は実質 HalfKP と同等。

### 次元と feature_hash

| 項目 | 値 |
|---|---|
| dim | 81 × 1548 × 9 = **1,128,492** |
| max_active | 38 |
| FEATURE_HASH | **`0x5D69D5B8`** (HalfKP Friend と同値) |

`kHashValue` は YaneuraOu source (`features/half_kpe9.h`) で `0x5D69D5B9 ^ (Friend == 1)` と定義されており、HalfKP の `kHashValue` と同じ値になる。**識別は description 文字列 (`HalfKPE9(Friend)`) と入力次元の差で行う**。

## HalfKP との比較

| | HalfKP | HalfKPE9 |
|---|---|---|
| 入力次元 / perspective | 125,388 | 1,128,492 (= × 9) |
| L0 重み行列 (L1=256) | 125,388 × 256 ≒ 32M | 1,128,492 × 256 ≒ 290M |
| 利き情報 | 無し | 自軍 0/1/2 × 敵軍 0/1/2 の 9 通り |
| 表現力 | 玉位置×駒位置 | 玉位置×駒位置×利き数 |
| 学習時間 | 標準 | HalfKP の数倍 |

## 実際の使い方

### コマンド

```bash
# Build (1 回だけ)
cargo build --release --features device-cuda --example bulletou

# Run
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKPE9 \
    --teacher teachers/
```

`--output` 省略時のデフォルトは `checkpoints/NNUE_HALFKPE9-256x2-32-32/`。

### 保存レイアウト

HalfKP と完全に同じ:

```
checkpoints/NNUE_HALFKPE9-256x2-32-32/
├── learn.log                          ← 10 列 CSV (全 run / resume 累積)
├── 0001/
│   ├── nn.bin                         ← やねうら王 / Stockfish 互換 NNUE バイナリ
│   ├── state.bin                      ← resume 用
│   └── learn.log                      ← snapshot
├── ...
└── 000N/
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

### `nn.bin` フォーマット

[NNUE HalfKP](halfkp.md#nnbin-format) と同じバイナリレイアウト。違いは:
- description 文字列の先頭が `Features=HalfKPE9(Friend)[1128492->...x2]`
- 入力次元 1,128,492 (HalfKP は 125,388)
- L0 重みのサイズが 9 倍

L1 / L2 / Out 層は `--arch` が同じなら HalfKP と byte-identical。

### 主要 CLI フラグ

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--eval-type` | `NNUE_HALFKPE9` | (必須) |
| `--arch` | `256x2-32-32`<br>`384x2-8-96`<br>`512x2-8-64`<br>`768x2-16-64`<br>`1024x2-8-32`<br>`1024x2-8-64` | `256x2-32-32` |
| `--teacher` | 教師ファイル / ディレクトリ / カンマ区切り | (必須) |
| `--output` | チェックポイント親ディレクトリ | `checkpoints/<eval-type>-<arch>` |
| `--lambda` | 教師 eval と対局結果 (WDL) のブレンド比 | 1.0 |

その他フラグは [HalfKP 学習](halfkp.md) を参照 (NNUE 系 4 つで共通)。

## 注意点

- **L0 が大きいので GPU メモリ消費に注意**。1024x2 構成の HalfKPE9 は HalfKP の 9 倍のメモリ。16GB+ の GPU 推奨。
- 利き計算は CPU 側 (dataloader thread) で実行される。Threat 系 feature と同じ仕組みなので、現状の最適化 (LUT による高速化) を共有する。
- **やねうら王の `halfkpe9_*` 評価関数** が動作する engine ビルドが必要 (`EVAL_KPP_NN_HALFKPE9` 等の build flag)。
