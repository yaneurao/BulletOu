# 02. `nn.bin` バイナリ仕様

`<output>/000N/nn.bin` に書き出すバイナリの仕様。**nnue-pytorch / Stockfish / YaneuraOu の NNUE 形式と byte 単位で同一**。

eval-type による分岐:
- **標準 NNUE** (`NNUE_HALFKP` / `NNUE_KP` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM`): 単一 MLP、本ページ §概観〜§量子化スケール詳細
- **SFNN-1536 / LayerStacks=9** (`SFNN_HALFKA1HM` / `SFNN_HALFKA2HM`): 9 個のサブネット + LEB128 FT + PSQT shortcut + SqrCReLU pair、§SFNN-1536 layout を参照

実装は `bulletou_lib::value::nnue_save` / `bulletou_lib::value::nnue_save_sfnn1536` モジュール。

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

l1_bias_scale は活性化関数や pairwise 有無で変わる。`bulletou_lib::value::nnue_save::l1_bias_scale()` 参照。

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

## エンジン側 hash check の挙動

やねうら王の `evaluate_nnue.cpp` は読み込み時に複数のハッシュ値をチェックするが、**それぞれ降格レベルが違う**。以下は本ライブラリの save format 設計に直接効いてくる契約事項:

| ヘッダ位置 | フィールド | エンジン側の挙動 (`evaluate_nnue.cpp`) | 結果 |
|---|---|---|---|
| ヘッダ先頭 4 byte | `NNUE_VERSION` | `version != kVersion` で `Tools::ResultCode::FileMismatch` (`evaluate_nnue.cpp:250-254`) | **load 失敗。完全一致必須** |
| ヘッダ 2 番目 u32 | `kHashValue` (= wide network hash) | mismatch を warning ログに格下げ (`evaluate_nnue.cpp:203-209`) | load 続行 |
| ヘッダ 3 番目 u32 + 続く byte | `desc_len` + arch description | 長さだけ読んで stream に流す。**parse なし、表示のみ** | 任意 |
| FT block 先頭 u32 | `FT::GetHashValue()` | `Detail::ReadParameters` が warning に降格 (`evaluate_nnue.cpp:178-179`) | load 続行 |
| 各 Network block 先頭 u32 | `Network::GetHashValue()` | 同上 | load 続行 |

ソース内コメント `evaluate_nnue.cpp:177` 「hash値、古い評価関数ファイルに対して一致するとは限らないので、警告に変更する。」が降格の根拠。**したがって `kHashValue` / FT hash / Network hash は値が任意でも load 自体は通る**が、bulletou は可読性のためやねうら王のソース定義 (`evaluate_nnue.h:25` / `nnue_feature_transformer.h:164` / `sfnnwop-1536.h:54` 等) に揃える。

## SIMD 重み permutation の方向

`AffineTransformExplicit` / `AffineTransformSparseInputExplicit` の `get_weight_index_scrambled` (`USE_SSSE3` または `USE_NEON_DOTPROD` 有効時に発動) は **read 時に file byte index を memory 位置に写像** する関数:

```cpp
// affine_transform_explicit.h:62-67
for (i = 0..out × pad_in):
    weights_[get_weight_index_scrambled(i)] = read_little_endian<i8>(stream);
```

書き手側の解析:

- ファイル position `f` の i8 byte は memory 位置 `weights_[scrambled(f)]` に置かれる
- engine 側の forward では `weights_[out * pad_in + in]` を行優先 (`(out, in)`) で参照する (`affine_transform.h:43-45`)
- `scrambled(f)` を decompose すると `f = out * pad_in + in_chunk * 4 + p` ↔ `scrambled(f) = in_chunk * out * 4 + out * 4 + p`
- ∴ **ファイルは行優先 `(out, in)` 順、engine 側 read 時に chunked memory layout に並べ替えられる**

bulletou は file を行優先で出力するだけで OK。padding (in ≥ in_dim_real) は 0 埋め。**書き手側で chunked layout を組む必要はない**。

(`USE_SSSE3` も `USE_NEON_DOTPROD` も無効な build では `get_weight_index(i) = i` で chunked layout 自体無効。x86_64 では SSSE3 が常に enable されるので、実質的に上記が支配的経路。)

## SFNN-1536 / LayerStacks=9 layout

`SFNN_HALFKA1HM` / `SFNN_HALFKA2HM` 用。やねうら王 `YANEURAOU_ENGINE_NNUE_SFNNwoP1536` ビルドが load する layout。標準 NNUE と次の点が違う:

1. **FT は LEB128 圧縮** (`USE_ELEMENT_WISE_MULTIPLY` 経路、`nnue_feature_transformer.h:180-182`)。SFNN ビルドで自動 enable される
2. **Network は 9 個の独立した stack** (`LayerStacks = 9`、`sfnnwop-1536.h:29`)
3. **PSQT shortcut neuron** (`kHidden1Dims = 8n - 1` の SFNN では `fc_0` の出力次元 = `kHidden1Dims + 1`、最後の 1 neuron は活性化を通さず `fc_2_out[0]` に直接加算。`kHidden1Dims = 8n` の SFNN では shortcut なし)
4. **SqrCReLU + CReLU pair** (`fc_0` 出力を CReLU と SqrCReLU の両方に通して結合 → `fc_1` 入力)
5. **3 種の hash 値はやねうら王側でハードコード固定値**

```
+---------------------------------------+
| Header                                |
|  NNUE_VERSION u32 LE = 0x7AF32F16     |
|  kHashValue   u32 LE = 0x3C203B32     |  ← SFNNwoPSQT 固定、warning のみ
|  desc_len     u32 LE                  |
|  description  UTF-8 bytes             |  ← `"ModelType=SFNNWithoutPsqt;Features=...{LayerStack=9}"`
+---------------------------------------+
| Feature Transformer                   |
|  FT hash u32 LE = 0x5F134AB8          |  ← SFNN 固定 (`nnue_feature_transformer.h:164`)
|  LEB128 block: biases  (i16 × ft_size)|  ← magic + size + signed-LEB128 payload
|  LEB128 block: weights (i16 × ft_size × input_size)
+---------------------------------------+
| × 9 LayerStacks                       |
|  Network hash u32 LE = 0x6333718A    |  ← SFNN 固定 (`sfnnwop-1536.h:54`)
|  fc_0 biases  i32 × (l1_hidden+1)     |  scale = qa × qb = 8128
|  fc_0 weights i8  × (l1_hidden+1) × pad32(ft_size)  scale = qb = 64
|  fc_1 biases  i32 × l2_size           |  scale = 8128
|  fc_1 weights i8  × l2_size × pad32(l1_hidden × 2)  scale = 64
|  fc_2 biases  i32 × 1                 |  scale = 8128
|  fc_2 weights i8  × 1 × pad32(l2_size)             scale = 64
+---------------------------------------+
```

ac_0 (ClippedReLUExplicit) / ac_sqr_0 (SqrClippedReLU) / ac_1 (ClippedReLUExplicit) はパラメータ無しで bytes を消費しない。重みの行優先順序および padding 規約は §SIMD 重み permutation の方向 と共通。

### LEB128 block format

`nnue_common.h:64-209` 由来。FT の biases / weights をそれぞれ別ブロックに圧縮:

```
+--------------------------------------+
| magic    : 17 bytes "COMPRESSED_LEB128" (null 終端なし、Leb128MagicStringSize = sizeof(literal) - 1)
| size     : u32 LE (= payload byte count)
| payload  : signed-LEB128 sequence (各 i16 を 1〜3 byte に可変長エンコード)
+--------------------------------------+
```

signed-LEB128 詳細:
- 各 byte の下位 7 bit が data、MSB が continuation (1 = 続く、0 = 最終 byte)
- 最終 byte の bit 0x40 = sign bit (符号拡張に使う)
- i16 の値域では最大 3 byte 必要 (`bulletou_lib::value::nnue_save_sfnn1536::push_signed_leb128_i16` 参照)

### 量子化スケール (SFNN)

標準 NNUE と同じ:

| 部位 | 型 | スケール |
|---|---|---|
| FT biases / weights | i16 | qa = 127 (Stockfish 標準) |
| fc_0 / fc_1 / fc_2 biases | i32 | qa × qb = 8128 |
| fc_0 / fc_1 / fc_2 weights | i8 | qb = 64 (= 1 << kWeightScaleBits = `nnue_common.h::kWeightScaleBits = 6`) |

### Per-stack byte count

```
4 (Network hash)
+ 64 + 24576       fc_0 (l1_hidden+1=16, ft_size=1536)
+ 128 + 1024       fc_1 (l2_size=32, l2_in=30→pad32=32)
+ 4 + 32           fc_2 (out=1, l2_size=32)
= 25,832 bytes per stack × 9 = 232,488 bytes
```

FT 部は LEB128 圧縮率に依存。1536 × (input_size + 1) × 平均 1.x byte ≒ 100〜200 MB が現実的。
