//! ShogiKa2 — YaneuraOu の `SFNNwoPSQT_ka2_1536-15-32-ls9` で使われる
//! `RawFeatures = FeatureSet<Features::K, Features::A2>` に対応する入力。
//!
//! KP との違いは Tail を P (玉以外, 1548 dim) → A2 (玉含む全駒, v2 collapse, 1629 dim)
//! に差し替えたこと。
//!
//! - K (`source/eval/nnue/features/k.h`):
//!   - `kDimensions = SQ_NB * 2 = 162` (自玉 0..80 / 相手玉 81..161)
//!   - `kHashValue = 0xD3CEE169`
//! - A2 (`source/eval/nnue/features/a2.h`):
//!   - `kDimensions = e_king = 1629` (後手玉を自玉 plane に collapse する v2 エンコーディング)
//!   - `kMaxActiveDimensions = PIECE_NUMBER_NB = 40` (玉を含む全駒)
//!   - `kHashValue = 0xA20DCB9B`
//!
//! `FeatureSet<K, A2>::kHashValue` の合成 (`feature_set.h:138`):
//!     Head::kHashValue ^ (Tail::kHashValue << 1) ^ (Tail::kHashValue >> 31)
//!   = 0xD3CEE169 ^ (0xA20DCB9B << 1) ^ (0xA20DCB9B >> 31)
//!   = 0xD3CEE169 ^ 0x441B9736 ^ 0x00000001
//!   = 0x97D5765E

use super::SparseInputType;
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::{E_KING, F_KING},
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, Square},
};

/// `FeatureSet<K, A2>` の合成 feature hash。
pub const FEATURE_HASH_KA2: u32 = 0x97D5765E;

/// e_king = 玉含む全駒 (v2 collapse) の BonaPiece 範囲 = A2 の次元数。
pub const KA2_A2_DIMENSIONS: usize = E_KING as usize; // 1629

/// SQ_NB = 81 マス。K は (自玉, 相手玉) を別 slot にして 2x SQ_NB を専有する。
pub const KA2_K_DIMENSIONS: usize = 81 * 2; // 162

/// KA2 全体の入力次元 = A2 + K = 1629 + 162 = 1791。
pub const KA2_DIMENSIONS: usize = KA2_A2_DIMENSIONS + KA2_K_DIMENSIONS; // 1791

/// 同時 active 数の上限 = A2 (40, 玉含む全駒) + K (2) = 42。
/// 玉は K と A2 の両方で発火するので「2 重カウント」になる (FeatureSet<K, A2> の意図通り)。
pub const KA2_MAX_ACTIVE: usize = 40 + 2;

/// ShogiKa2 — `SFNNwoPSQT_ka2_*` 用の入力特徴量。dual-perspective。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiKa2;

impl SparseInputType for ShogiKa2 {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        KA2_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        KA2_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_ka2_features(&board, f);
    }

    fn shorthand(&self) -> String {
        "shogi-ka2-1791".to_string()
    }

    fn description(&self) -> String {
        "Shogi K+A2: 162 king features + 1629 piece features (kings collapsed onto friend plane) per perspective".to_string()
    }
}

/// 視点 `perspective` から見た square index (0..80)。
/// 後手視点では盤を反転させる (YaneuraOu の piece_list_fw と同じ)。
#[inline]
fn sq_from_perspective(sq: Square, perspective: Color) -> usize {
    if perspective == Color::Black { sq.index() } else { sq.inverse().index() }
}

/// K の active index (perspective から見たもの) を計算。
///
/// `FeatureSet<K, A2>` は K (Head) に Tail (A2) の次元数をシフトとして加えるので:
///   own_king: KA2_A2_DIMENSIONS + sq
///   opp_king: KA2_A2_DIMENSIONS + SQ_NB + sq
#[inline]
fn k_index_own(sq_from_p: usize) -> usize {
    KA2_A2_DIMENSIONS + sq_from_p
}

#[inline]
fn k_index_opp(sq_from_p: usize) -> usize {
    KA2_A2_DIMENSIONS + 81 + sq_from_p
}

/// A2 の v2 collapse: 敵王 BonaPiece (>= E_KING) を自玉 plane (F_KING..E_KING) に投影。
#[inline]
fn a2_index(bp: u16) -> usize {
    let v = bp as usize;
    if v >= E_KING as usize { v - 81 } else { v }
}

/// K + A2 特徴量を `f(stm_idx, nstm_idx)` 形で列挙する。
fn map_ka2_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
    let stm = board.side_to_move;
    let nstm = stm.opponent();

    let stm_king_sq = board.king_square(stm);
    let nstm_king_sq = board.king_square(nstm);
    if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
        return;
    }

    // ---- K: 自玉 / 相手玉 -------------------------------------------------
    let stm_own_king = sq_from_perspective(stm_king_sq, stm);
    let stm_opp_king = sq_from_perspective(nstm_king_sq, stm);
    let nstm_own_king = sq_from_perspective(nstm_king_sq, nstm);
    let nstm_opp_king = sq_from_perspective(stm_king_sq, nstm);

    f(k_index_own(stm_own_king), k_index_opp(nstm_opp_king));
    f(k_index_opp(stm_opp_king), k_index_own(nstm_own_king));

    // ---- A2: 玉以外の盤上駒 (BonaPiece は < E_KING なので collapse は no-op) ----
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);

                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);

                f(a2_index(stm_bp.value()), a2_index(nstm_bp.value()));
            }
        }
    }

    // ---- A2: 両玉 (v2 collapse で敵王を自玉 plane に折りたたむ) -----------
    {
        let stm_friend_bp = (F_KING as usize) + sq_from_perspective(stm_king_sq, stm);
        let stm_enemy_bp = (E_KING as usize) + sq_from_perspective(nstm_king_sq, stm);
        let nstm_friend_bp = (F_KING as usize) + sq_from_perspective(nstm_king_sq, nstm);
        let nstm_enemy_bp = (E_KING as usize) + sq_from_perspective(stm_king_sq, nstm);

        // stm の玉: STM 視点では自玉、NSTM 視点では相手玉
        f(a2_index(stm_friend_bp as u16), a2_index(nstm_enemy_bp as u16));
        // nstm の玉: STM 視点では相手玉、NSTM 視点では自玉
        f(a2_index(stm_enemy_bp as u16), a2_index(nstm_friend_bp as u16));
    }

    // ---- A2: 手駒 ---------------------------------------------------------
    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            let count = board.hand(owner).count(pt);
            if count == 0 {
                continue;
            }
            for i in 1..=count {
                let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                f(a2_index(stm_bp.value()), a2_index(nstm_bp.value()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::shogi::{PieceType, Square};

    use super::*;

    #[test]
    fn dims() {
        let ka2 = ShogiKa2;
        assert_eq!(ka2.num_inputs(), 1791);
        assert_eq!(ka2.max_active(), 42);
    }

    #[test]
    fn k_index_ranges() {
        // 自玉の最初: KA2_A2_DIMENSIONS (= 1629)
        assert_eq!(k_index_own(0), 1629);
        // 自玉の最後: KA2_A2_DIMENSIONS + 80 = 1709
        assert_eq!(k_index_own(80), 1709);
        // 相手玉の最初: KA2_A2_DIMENSIONS + 81 = 1710
        assert_eq!(k_index_opp(0), 1710);
        // 相手玉の最後: KA2_A2_DIMENSIONS + 81 + 80 = 1790 (= KA2_DIMENSIONS - 1)
        assert_eq!(k_index_opp(80), 1790);
    }

    #[test]
    fn a2_collapse_enemy_to_friend_plane() {
        // 敵王 (>= E_KING) は自玉 plane (F_KING..E_KING) に collapse される。
        for sq in 0u16..81 {
            let friend_bp = (F_KING + sq) as u16;
            let enemy_bp = (E_KING + sq) as u16;
            assert_eq!(a2_index(friend_bp), a2_index(enemy_bp));
            assert_eq!(a2_index(friend_bp), (F_KING as usize) + sq as usize);
        }
    }

    #[test]
    fn a2_passthrough_below_e_king() {
        // BonaPiece が E_KING 未満なら collapse なし、そのまま。
        for v in [0u16, 100, 1000, 1547, 1548, 1628] {
            assert_eq!(a2_index(v), v as usize);
        }
    }

    #[test]
    fn feature_hash_matches_yaneuraou() {
        // YaneuraOu の FeatureSet<K, A2>::kHashValue =
        //   K::kHashValue ^ (A2::kHashValue << 1) ^ (A2::kHashValue >> 31)
        // = 0xD3CEE169 ^ (0xA20DCB9B << 1) ^ (0xA20DCB9B >> 31)
        // = 0xD3CEE169 ^ 0x441B9736 ^ 0x00000001
        // = 0x97D5765E
        let k_hash: u32 = 0xD3CEE169;
        let a2_hash: u32 = 0xA20DCB9B;
        let combined = k_hash ^ (a2_hash.wrapping_shl(1)) ^ (a2_hash.wrapping_shr(31));
        assert_eq!(combined, FEATURE_HASH_KA2);
        assert_eq!(FEATURE_HASH_KA2, 0x97D5765E);
    }

    #[test]
    fn map_features_two_kings_only() {
        // 玉だけある盤面: K で 2 件 + A2 で 2 件 (両玉) = 4 件。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        let mut emissions: Vec<(usize, usize)> = Vec::new();
        map_ka2_features(&board, |s, n| emissions.push((s, n)));
        assert_eq!(emissions.len(), 4);

        // 全 index が KA2_DIMENSIONS 未満。
        let max = KA2_DIMENSIONS - 1;
        for (s, n) in &emissions {
            assert!(*s <= max, "stm={s}");
            assert!(*n <= max, "nstm={n}");
        }

        // K の 2 件は >= KA2_A2_DIMENSIONS、A2 の 2 件は < KA2_A2_DIMENSIONS。
        let k_count = emissions.iter().filter(|(s, _)| *s >= KA2_A2_DIMENSIONS).count();
        let a2_count = emissions.iter().filter(|(s, _)| *s < KA2_A2_DIMENSIONS).count();
        assert_eq!(k_count, 2);
        assert_eq!(a2_count, 2);
    }

    #[test]
    fn map_features_two_kings_plus_pawns() {
        // 玉 + 歩 9×2: K で 2 件 + A2 で (歩 18 + 玉 2) = 22 件。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        let mut k_count = 0;
        let mut a2_count = 0;
        map_ka2_features(&board, |s, _n| {
            if s >= KA2_A2_DIMENSIONS {
                k_count += 1;
            } else {
                a2_count += 1;
            }
        });
        assert_eq!(k_count, 2);
        // 歩 18 + 玉 2 = 20
        assert_eq!(a2_count, 20);
    }

    #[test]
    fn map_features_skip_one_king_position() {
        // 片玉データはスキップ。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::NONE,
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);

        let mut count = 0;
        map_ka2_features(&board, |_, _| count += 1);
        assert_eq!(count, 0);
    }
}
