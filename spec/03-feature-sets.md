# 03. NNUE Feature Sets 仕様

NNUE 系の `--eval-type` が使う入力特徴量の仕様。`bullet_lib::game::inputs` で実装されているもののうち、現状 `bulletou` から到達可能なもの。

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

詳細は `crates/bullet_lib/src/game/inputs/shogi_halfkp.rs`。

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

YaneuraOu の `RawFeatures = FeatureSet<Features::K, Features::P>` (`kp_256x2-32-32.h`) 相当。BulletOu では `bullet_lib::game::inputs::ShogiKp` として単一の SparseInputType に統合実装。

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

詳細は `crates/bullet_lib/src/game/inputs/shogi_kp.rs`。

## HalfKP vs K-P 設計比較

| | HalfKP | K-P |
|---|---|---|
| perspective あたり入力 dim | 125,388 | 1,710 |
| クロス積か | はい (玉位置 × 駒) | いいえ (玉と駒を独立に並べる) |
| L0 重み行列サイズ (256 L1 想定) | 125,388 × 256 | 1,710 × 256 |
| 表現力 | 高 (玉位置 × 駒位置 の組合せ毎に重み) | 低 (相関は L0+L1 経由で学習) |
| やねうら王での歴史的位置付け | 主流 | アブレーション用 (PR #75 で並列追加) |
