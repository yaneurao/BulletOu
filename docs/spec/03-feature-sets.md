# 03. NNUE Feature Sets 仕様

NNUE 系の `--eval-type` が使う入力特徴量の仕様。`bulletou_lib::game::inputs` で実装されているもののうち、現状 `bulletou` から到達可能なもの。

## 共通

各 feature set は `SparseInputType` トレイトを実装し、以下を返す:

| メソッド | 意味 |
|---|---|
| `num_inputs()` | 入力次元数 (perspective あたり) |
| `max_active()` | 1 局面で同時 active になる feature の最大数 |
| `map_features(pos, f)` | `f(stm_idx, nstm_idx)` で stm/nstm 視点の発火 index を列挙 |

`stm_idx` / `nstm_idx` はそれぞれの perspective 用 accumulator に加算される feature index (`0..num_inputs()` 範囲)。

## HalfKP

YaneuraOu / Stockfish の古典的 HalfKP。`Features::HalfKP` (`features/half_kp.h`) 相当。

| 項目 | 値 |
|---|---|
| dim | 125,388 (= 81 × 1548) |
| max_active | 38 (玉以外の駒、最大 38) |
| FEATURE_HASH | `0x5D69D5B8` |
| 構造 | `(own_king_sq, friendly_piece_bonapiece)` のクロス積 |

index 計算:
```
halfkp_index = king_sq * FE_END + bonapiece
            (king_sq = 0..80, bonapiece = 0..1547、index 0 は BonaPiece ゼロ予約で未使用)
```

`own_king_sq` は **perspective から見た自玉の座標** (BLACK perspective なら絶対座標、WHITE perspective なら反転後の座標)。

`bonapiece` は BonaPiece 値 (0..fe_end-1)。`from_piece_square(piece, sq, perspective)` または `from_hand_piece(perspective, owner, pt, count)` で計算。

詳細は `crates/bulletou_lib/src/game/inputs/shogi_halfkp.rs`。

## K (玉単体)

`Features::K` (`features/k.h`) 相当。BulletOu では単独使用ではなく後述の `ShogiKp` の構成要素として使う。

| 項目 | 値 |
|---|---|
| dim | 162 (= 81 × 2) |
| max_active | 2 (自玉 + 相手玉) |
| FEATURE_HASH (k.h::kHashValue) | `0xD3CEE169` |
| 内訳 | 自玉 81 slot + 相手玉 81 slot |

YaneuraOu の `K::AppendActiveIndices` は `BonaPiece(king_i) - fe_end` を吐く (kings の BonaPiece は `fe_end..fe_end+162`)。perspective 視点での自玉 / 相手玉の square 座標が返る形。

active index の意味付け:
- `0..80`: 自玉が perspective から見た square 0..80 にいる
- `81..161`: 相手玉が perspective から見た square 0..80 にいる

## P (玉以外の駒)

`Features::P` (`features/p.h`) 相当。同上、`ShogiKp` の構成要素。

| 項目 | 値 |
|---|---|
| dim | 1548 (= `fe_end` = `FE_OLD_END`) |
| max_active | 38 (= `PIECE_NUMBER_KING`、玉以外の駒数上限) |
| FEATURE_HASH (p.h::kHashValue) | `0x764CFB4B` |
| 内訳 | 玉以外の駒の BonaPiece 値 (raw) |

active index = bonapiece 値 (0..1547、index 0 は未使用)。

## FeatureSet 合成規則

YaneuraOu の `feature_set.h` で定義されている `FeatureSet<Head, Tail>` の合成 (sub feature を 1 つに統合する仕組み):

| 合成属性 | 計算式 |
|---|---|
| `kDimensions` | `Head::kDimensions + Tail::kDimensions` |
| `kMaxActiveDimensions` | `Head::kMaxActiveDimensions + Tail::kMaxActiveDimensions` |
| `kHashValue` | `Head::kHashValue ^ (Tail::kHashValue << 1) ^ (Tail::kHashValue >> 31)` |
| 名前 (`GetName()`) | `Head::kName + "+" + Tail::GetName()` (再帰、`+` 結合) |

active index の合成:
- **Tail を先に列挙** (オフセット無し、index は `0..Tail::kDimensions-1`)
- **Head を後に列挙** し、index に `Tail::kDimensions` を加算 (= `Tail::kDimensions..Tail::kDimensions+Head::kDimensions-1`)

## ShogiKp (= K-P)

YaneuraOu の `RawFeatures = FeatureSet<Features::K, Features::P>` (`kp_256x2-32-32.h`) 相当。BulletOu では `bulletou_lib::game::inputs::ShogiKp` として単一の SparseInputType に統合実装。

| 項目 | 値 |
|---|---|
| dim | 1,710 (= 162 + 1548) |
| max_active | 40 (= 2 + 38) |
| FEATURE_HASH_KP | `0x3F5717FF` |

`FEATURE_HASH_KP` の導出:
```
FEATURE_HASH_KP
  = K::kHashValue ^ (P::kHashValue << 1) ^ (P::kHashValue >> 31)
  = 0xD3CEE169 ^ (0x764CFB4B << 1) ^ (0x764CFB4B >> 31)
  = 0xD3CEE169 ^ 0xEC99F696 ^ 0
  = 0x3F5717FF
```

active index 配置 (`FeatureSet<K, P>` の Head=K, Tail=P 合成則より):

| index 範囲 | 意味 |
|---|---|
| `0..1547` | P (玉以外の BonaPiece 値、index 0 未使用) |
| `1548..1628` | K 自玉 (= 1548 + perspective から見た自玉 sq 0..80) |
| `1629..1709` | K 相手玉 (= 1548 + 81 + perspective から見た相手玉 sq 0..80) |

`map_features` 内で:
- 物理的な「stm 側の玉」: STM 視点では自玉 (`1548 + stm_view(stm_king)`)、NSTM 視点では相手玉 (`1548 + 81 + nstm_view(stm_king)`)
- 物理的な「nstm 側の玉」: STM 視点では相手玉 (`1548 + 81 + stm_view(nstm_king)`)、NSTM 視点では自玉 (`1548 + nstm_view(nstm_king)`)
- 物理的な「玉以外の駒 1 個」: STM 視点 / NSTM 視点それぞれの BonaPiece 値 (P 領域に発火)

詳細は `crates/bulletou_lib/src/game/inputs/shogi_kp.rs`。

## A2 (玉含む全駒、v2 collapse)

`Features::A2` (`features/a2.h`) 相当。`ShogiKa2` の構成要素。P を「玉も含めた全駒」に拡張し、後手玉 BonaPiece (`E_KING..fe_end2`) を自玉 plane (`F_KING..E_KING`) に collapse する v2 エンコーディングを使う (HalfKA_hm2 と同じ collapse 規則を非 anchored 版で適用したもの)。

| 項目 | 値 |
|---|---|
| dim | 1,629 (= `E_KING`) |
| max_active | 40 (= `PIECE_NUMBER_NB`、玉含む全駒) |
| FEATURE_HASH (a2.h::kHashValue) | `0xA20DCB9B` |
| 内訳 | `0..1547` 玉以外 BonaPiece (P と同じ範囲) + `1548..1628` 自玉 plane (両玉が collapse) |

active index = `bp >= E_KING ? bp - SQ_NB : bp` (後手玉を自玉 plane へ移動)。

`refresh trigger` は持たない (anchor 無しなので玉が動いても全計算不要)。差分更新は dirtyPiece を全部処理 (P と違って kings をスキップしない)。

## ShogiKa2 (= K-A2)

YaneuraOu の `RawFeatures = FeatureSet<Features::K, Features::A2>` (`SFNNwoPSQT_ka2_*` および `ka2_*x2-*-*`) 相当。BulletOu では `bulletou_lib::game::inputs::ShogiKa2` として単一の SparseInputType に統合実装。

| 項目 | 値 |
|---|---|
| dim | 1,791 (= 162 + 1629) |
| max_active | 42 (= 2 + 40。玉が K と A2 で「2 重カウント」される、`FeatureSet<K, A2>` の意図通り) |
| FEATURE_HASH_KA2 | `0x97D5765E` |

`FEATURE_HASH_KA2` の導出:
```
FEATURE_HASH_KA2
  = K::kHashValue ^ (A2::kHashValue << 1) ^ (A2::kHashValue >> 31)
  = 0xD3CEE169 ^ (0xA20DCB9B << 1) ^ (0xA20DCB9B >> 31)
  = 0xD3CEE169 ^ 0x441B9736 ^ 0x00000001
  = 0x97D5765E
```

active index 配置 (`FeatureSet<K, A2>` の Head=K, Tail=A2 合成則より):

| index 範囲 | 意味 |
|---|---|
| `0..1547` | A2 玉以外 BonaPiece 値 (P 領域と同じ) |
| `1548..1628` | A2 玉 plane (両玉が collapse、自玉も後手玉もここに発火) |
| `1629..1709` | K 自玉 (= 1629 + perspective から見た自玉 sq 0..80) |
| `1710..1790` | K 相手玉 (= 1629 + 81 + perspective から見た相手玉 sq 0..80) |

「玉が K と A2 で 2 重に発火」の意味: 自玉が square s にあるとき、K 領域では `1629 + s` に、A2 領域では `1548 + s` (collapse 後) にそれぞれ index 1 が立つ。同じ物理的な玉から 2 つ active feature が出るので、K-P (max_active=40) より max_active が 2 増えて 42。

詳細は `crates/bulletou_lib/src/game/inputs/shogi_ka2.rs`。

## HalfKPE9

YaneuraOu の `Features::HalfKPE9` (`features/half_kpe9.{h,cpp}`) 相当。HalfKP の input に「**駒のいるマスへの利き数情報**」を 9 通り掛けた変種。

| 項目 | 値 |
|---|---|
| dim | 81 × 1548 × 9 = **1,128,492** (= HalfKP × 9) |
| max_active | 38 (HalfKP と同じ、玉以外の駒) |
| FEATURE_HASH | **`0x5D69D5B8`** (HalfKP と同値) |
| 識別子 (description) | `HalfKPE9(Friend)` |

active index 計算式 (`MakeIndex` 由来):

```
index = fe_end × king_sq + bonapiece
      + fe_end × SQ_NB × (effect1 × 3 + effect2)
```

- `effect1` = perspective から見た自軍がそのマスに与えている利き数 (0/1/2 にクリップ)
- `effect2` = 同じく敵軍の利き数
- `effect bucket = effect1 × 3 + effect2`、0..8 の 9 通り

bucket index 配置:

| `effect1, effect2` | bucket | index 範囲 |
|---|---|---|
| (0, 0) | 0 | `0 .. 125,387` (= HalfKP と同じ index) |
| (0, 1) | 1 | `125,388 .. 250,775` |
| (0, 2) | 2 | `250,776 .. 376,163` |
| (1, 0) | 3 | ... |
| ... | ... | ... |
| (2, 2) | 8 | `1,003,104 .. 1,128,491` |

### 利き計算

各教師局面について、81 マス × 2 色 = 162 セルの利き数 table を作成する (`bulletou_lib::game::inputs::shogi_halfkpe9::compute_effect_counts`)。玉を含む全駒種について `for_each_attack` で利き先マスを列挙 → 該当セルをインクリメント (上限 2)。slider 駒 (角・飛・馬・竜) は遮蔽考慮で正しく扱われる。

### 手駒の扱い

手駒は盤上 sq を持たないので `effect1 = effect2 = 0` 固定。`effect bucket = 0` に発火 (= HalfKP と同じ index 領域)。

### FEATURE_HASH の collision

YaneuraOu の `kHashValue` 定義:
```cpp
static constexpr std::uint32_t kHashValue =
    0x5D69D5B9u ^ (AssociatedKing == Side::kFriend);
```

これは HalfKP の kHashValue と **完全に同じ値**。エンジン側は `kHashValue` だけでは HalfKP / HalfKPE9 を判別できず、description 文字列 (`HalfKPE9(Friend)`) と入力次元の違いで判別する。

## HalfKP_vm

YaneuraOu の `Features::HalfKP_vm` (`features/half_kp_vm.{h,cpp}`) 相当。HalfKP と同じ「(玉位置, 駒)」の sparse 特徴量だが、**玉の左右対称性** (6 筋以降の玉は 4 筋以前にミラー) を畳んで入力次元を約 1/2 にした版。

| 項目 | 値 |
|---|---|
| dim | 45 × 1548 = **69,660** (= HalfKP の約 1/2) |
| max_active | 38 (HalfKP と同じ、玉以外の駒) |
| FEATURE_HASH_HALFKPVM | `0x0B6B1D9A` (= `0x0B6B1D9B ^ 1` for `Side::kFriend`) |
| 識別子 (description) | `HalfKP_vm(Friend)` |

active index 計算式 (`MakeIndex` 由来):

```
sq_k_eff = (king_file >= 5) ? Mir(king_sq) : king_sq      // file ∈ {0..4}, rank ∈ {0..8}
bp_eff   = (king_file >= 5 && bp >= fe_hand_end)          // 盤上駒のみ
              ? FE_HAND_END + piece_idx * SQ_NB + Mir(sq) // sq だけミラー
              : bp                                        // 持駒はミラーしない

index = fe_end × sq_k_eff + bp_eff
      = (file_eff * 9 + rank) * 1548 + bp_eff             // file_eff ∈ {0..4}, ∴ index < 45 * 1548
```

- `Mir(sq)`: 筋反転 (file 0 ↔ 8, file 1 ↔ 7, ..., file 4 fixed)。Rust 実装は `Square::mirror_file()` 同等
- 持駒 BonaPiece (`< fe_hand_end = 90`) は仮想的なエンコーディング (盤面 sq を持たない) のためミラーしない

### FEATURE_HASH

YaneuraOu の `kHashValue` 定義:
```cpp
static constexpr std::uint32_t kHashValue =
    0x0B6B1D9Bu ^ (AssociatedKing == Side::kFriend);
```

`Side::kFriend` (= 1) を XOR して `0x0B6B1D9A` になる。これは HalfKP (`0x5D69D5B8`) / HalfKPE9 (同上) と **別値** なので、engine 側で description 検証なしでも判別できる。

詳細は `crates/bulletou_lib/src/game/inputs/shogi_halfkpvm.rs`。

## HalfKA_hm1 (strict v1)

`Features::HalfKA_hm1` (`features/half_ka_hm1.{h,cpp}`) 相当。HalfKA + 左右対称ミラー、**両玉を別 plane に区別して** 含める。SFNN-1536 / SFNN_HALFKA1HM (ablation 用途) で使用。

| 項目 | 値 |
|---|---|
| dim | 45 × 1710 = **76,950** |
| max_active | 40 (`PIECE_NUMBER_NB`、両玉含む全駒) |
| FEATURE_HASH_HALFKA_HM1 | `0x7f134cb8` (= `0x7f134cb9 ^ 1` for `Side::kFriend`) |
| 識別子 (description) | `HalfKA_hm1(Friend)` |
| 駒入力数 (= `fe_end2` = `e_king + SQ_NB`) | 1710 |

active index 計算式 (`MakeIndex` 由来):

```
sq_k_eff = (king_file >= 5) ? Mir(king_sq) : king_sq
bp_eff   = (king_file >= 5 && bp >= fe_hand_end)          // 盤上駒・王のみ
              ? FE_HAND_END + piece_idx * SQ_NB + Mir(sq) // sq だけミラー
              : bp                                        // 持駒は不変
index = fe_end2 × sq_k_eff + bp_eff                       // 玉は別 plane、collapse なし
```

詳細は `crates/bulletou_lib/src/game/inputs/shogi_halfka_hm1.rs`。

## HalfKA_hm2 (strict v2)

`Features::HalfKA_hm2` (`features/half_ka_hm2.{h,cpp}`) 相当。HalfKA_hm1 の dim 圧縮版で、**後手玉 BonaPiece を自玉 plane に collapse** することで入力次元を ~4.7% 削減。やねうら王 `YANEURAOU_ENGINE_NNUE_SFNNwoP1536` ビルドが実際に使うのはこちら。

| 項目 | 値 |
|---|---|
| dim | 45 × 1629 = **73,305** |
| max_active | 40 (HalfKA_hm1 と同じ) |
| FEATURE_HASH_HALFKA_HM2 | `0x7f234cb8` (= `0x7f234cb9 ^ 1` for `Side::kFriend`) |
| 識別子 (description) | `HalfKA_hm2(Friend)` |
| 駒入力数 (= `e_king`、後手玉を自玉 plane に collapse) | 1629 |

active index 計算式 (`MakeIndex` 由来):

```
sq_k_eff = (king_file >= 5) ? Mir(king_sq) : king_sq      // HalfKA_hm1 と同じ
bp_eff   = (king_file >= 5 && bp >= fe_hand_end)
              ? FE_HAND_END + piece_idx * SQ_NB + Mir(sq)
              : bp
// 2 段階目: 後手王 (>= e_king) を自玉 plane に collapse
bp_eff   = bp_eff >= e_king ? bp_eff - SQ_NB : bp_eff
index = e_king × sq_k_eff + bp_eff                        // dim = 5 × 9 × e_king
```

詳細は `crates/bulletou_lib/src/game/inputs/shogi_halfka_hm2.rs`。

### v1 / v2 の使い分け

| 用途 | 推奨 |
|---|---|
| やねうら王 SFNNwoP1536 ビルドに投入する | **`HalfKA_hm2`** (= 唯一 engine-loadable な variant) |
| 両玉を別 plane で持ったときの強さ比較 (ablation) | `HalfKA_hm1` |

`bulletou_lib::game::inputs::ShogiHalfKA_hm` (= 既存実装、`shogi_halfka.rs`) は **アルゴリズムは v2 (collapse あり)** だが **hash 値は v1 (`0x7f134cb8`)** を返すというプリエクストの不整合がある。これは `examples/shogi_layerstack.rs` の rshogi 互換出力で消費されているため後方互換のため触らない。やねうら王互換 nn.bin を作る経路では上記の strict v1 / v2 (`ShogiHalfKaHm1` / `ShogiHalfKaHm2`) を使うこと。

## HalfKP vs K-P 設計比較

| | HalfKP | K-P |
|---|---|---|
| perspective あたり入力 dim | 125,388 | 1,710 |
| クロス積か | はい (玉位置 × 駒) | いいえ (玉と駒を独立に並べる) |
| L0 重み行列サイズ (256 L1 想定) | 125,388 × 256 | 1,710 × 256 |
| 表現力 | 高 (玉位置 × 駒位置 の組合せ毎に重み) | 低 (相関は L0+L1 経由で学習) |
| やねうら王での歴史的位置付け | 主流 | アブレーション用 (PR #75 で並列追加) |
