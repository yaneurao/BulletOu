# 02. `nn.bin` バイナリ仕様

NNUE 系 eval-type (`NNUE_HALFKP` / `NNUE_KP`) が `<output>/000N/nn.bin` に書き出すバイナリの仕様。**nnue-pytorch / Stockfish / YaneuraOu の NNUE 形式と byte 単位で同一**。

`bullet_lib::value::nnue_save` モジュール参照。

## 概観

```
+---------------------------------------+
| Header                                |
|  - NNUE_VERSION (u32 LE)              |  = 0x7AF32F16
|  - network_hash (u32 LE)              |  ← § hash 計算
|  - desc_len (u32 LE)                  |
|  - description (UTF-8 bytes)          |  ← `Features=...,Network=...`
+---------------------------------------+
| Feature Transformer (FT)              |
|  - ft_hash (u32 LE)                   |  = feature_hash ^ (L1×2)
|  - L0 biases  (i16 × L1, qa)          |
|  - L0 weights (i16 × INPUT × L1, qa)  |  column-major bullet 内部 → そのまま出力
+---------------------------------------+
| Network layer hash (u32 LE)           |  = fc_hash
+---------------------------------------+
| L1                                    |
|  - L1 bias    (i32 × L2, l1_bias_scale)
|  - L1 weights (i8  × L2 × pad32(L1×2), qb)  row-major, SIMD padded
+---------------------------------------+
| L2                                    |
|  - L2 bias    (i32 × L3, 127×qb)      |
|  - L2 weights (i8  × L3 × pad32(L2), qb)    row-major, SIMD padded
+---------------------------------------+
| Output                                |
|  - Out bias    (i32 × 1, 127×qb)      |
|  - Out weights (i8  × 1 × pad32(L3), qb)    row-major, SIMD padded
+---------------------------------------+
```

すべて little-endian。`pad32(n) = ceil(n/32) × 32`。

## 定数

| 名称 | 値 |
|---|---|
| `NNUE_VERSION` | `0x7AF32F16` (u32) |
| L0 量子化 `qa` (ClippedReLU 前提) | `127` (i16) |
| L1-Out 量子化 `qb` | `64` (i16, weights は i8 へ) |
| InputSlice hash base | `0xEC42E90D` |
| FC layer hash base | `0xCC03DAE4` |
| ClippedReLU hash | `0x538D24C7` |

## description 文字列

nnue-pytorch / YaneuraOu パーサが読む人間可読部分。**byte 数は `desc_len` で明示されるので空白や記号は厳密一致でなくとも load 自体は通る** (engine は description を walk して構造を取るので構造一致は必要)。

書式 (HalfKP の例):
```
Features=HalfKP(Friend)[125388->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransformSparseInput[32<-512](InputSlice[512(0:512)])))))
```

`Features=<name>[<input_size>-><L1>x2]` の `<name>` は feature set に依存。`AffineTransform[1<-L3](...AffineTransform[L3<-L2](...AffineTransformSparseInput[L2<-L1×2](InputSlice[L1×2(0:L1×2)])))` の入れ子で L1, L2, L3, Out の構造を表現する。

feature set の name (`<name>`):

| feature set | description name | YaneuraOu 内部の `kName` |
|---|---|---|
| HalfKP | `HalfKP(Friend)` | `"HalfKP"` |
| K-P (= `FeatureSet<K, P>`) | `K-P(Friend)` (BulletOu の暫定表記) | `"K+P"` (`feature_set.h` の `+` 結合) |

K-P の `(Friend)` suffix と `+` vs `-` 表記は engine 側のパーサ容認範囲で調整余地あり。`network_hash` の一致が load 時の本質的な互換性チェック。

## hash 計算

### feature_hash (各 feature set 固有)

[03-feature-sets.md](03-feature-sets.md) 参照。

### fc_hash (= network layer hash)

```python
def fc_hash(L1, L2, L3):
    prev = 0xEC42E90D
    prev ^= (L1 * 2)
    for i, out_features in enumerate([L2, L3, 1]):
        layer = 0xCC03DAE4
        layer = (layer + out_features) & 0xFFFFFFFF
        layer ^= (prev >> 1) ^ ((prev << 31) & 0xFFFFFFFF)
        if i < 2:                          # output 層には ClippedReLU が付かない
            layer = (layer + 0x538D24C7) & 0xFFFFFFFF
        prev = layer
    return prev
```

`256x2-32-32`:
```
prev = 0xEC42E90D
prev ^= 512 = 0xEC42EB0D
# L2 (out=32):
layer = (0xCC03DAE4 + 32) ^ (prev>>1) ^ (prev<<31)
layer += 0x538D24C7   # ClippedReLU
# ... 同様に L3, Out ...
```

### ft_hash

`ft_hash = feature_hash ^ (L1 × 2)`

### network_hash

`network_hash = fc_hash ^ feature_hash ^ (L1 × 2)`

これがヘッダーの `network_hash` フィールドに入る。engine 側はこの値で互換性を判定する。

## 量子化スケール詳細

| 重み | スケール (qa=127, qb=64 の場合) | 備考 |
|---|---|---|
| L0 biases / weights | qa = 127 (i16) | ClippedReLU 出力 range は 0..127 |
| L1 bias | qa × qb = 8128 (i32) | ClippedReLU 出力 (qa スケール) を qb スケールにする |
| L1 weights | qb = 64 (i8) | row-major (= transpose 後)、pad32 |
| L2 bias | 127 × qb = 8128 (i32) | crelu_i32_to_u8 後は常に 127 スケール |
| L2 weights | qb = 64 (i8) | row-major、pad32 |
| Out bias | 127 × qb = 8128 (i32) | 同上 |
| Out weights | qb = 64 (i8) | row-major、pad32 |

l1_bias_scale は活性化関数や pairwise 有無で変わる。`bullet_lib::value::nnue_save::l1_bias_scale()` 参照。

```
CReLU      qa=127  → l1_bias = 127 × qb
CReLU      qa=255  → l1_bias = 255 × qb
SCReLU     qa=255  → l1_bias = 127 × qb     (x² >> 9 で 127 にスケールダウン)
Pairwise   qa=255  → l1_bias = (qa² >> 9) × qb
Pairwise   qa<255  → l1_bias = (qa² >> 7) × qb
```

## SIMD パディング (`pad32`)

L1 / L2 / Out 層の重みは row-major かつ入力次元を 32 の倍数に揃える。32B = AVX2 一行のサイズ。

```rust
fn pad32(n: usize) -> usize { (n + 31) / 32 * 32 }
```

入力次元 < 32 のときも 32 にゼロパディング。パディング部分の重みは 0。

例 (256x2-32-32):
- L1: 入力 = 2 × 256 = 512 (= pad32(512))、padding 不要
- L2: 入力 = 32 → pad32(32) = 32、padding 不要
- Out: 入力 = 32 → pad32(32) = 32、padding 不要

実際には小さいモデルではパディングは多くないが、L2/L3 = 8 のような構成ではパディング差が大きい。
