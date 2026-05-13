//! ShogiHalfKaHm2 - 将棋用 HalfKA_hm2 特徴量 (strict v2)
//!
//! やねうら王 `Features::HalfKA_hm2` (`features/half_ka_hm2.{h,cpp}`) と byte-for-byte
//! 一致する HalfKA + half-mirror 特徴量。**後手玉を自玉の plane に collapse** することで
//! v1 (`ShogiHalfKaHm1`) より入力次元を ~4.7% 削減する。SFNNwoPSQT-1536 で実際に使われる
//! のはこの v2 系。
//!
//! - キングバケット: **45** (= 5 筋 × 9 段、玉が 6 筋以降のとき file-mirror)
//! - 駒入力数: **1629** (= `e_king`、両玉を同じ 81 マス plane に投影)
//! - 入力次元: **73,305** (= 45 × 1629)
//! - 最大アクティブ特徴: 40 (全駒)
//! - feature hash: `0x7f234cb8` (= `0x7f234cb9 ^ 1` for `Side::kFriend`)
//!
//! v1 (`ShogiHalfKaHm1`) との違いは index 計算式のみ:
//! - v1: `index = fe_end2 × sq_k + p` (両玉を別 plane に保持、dim=76,950)
//! - **v2**: `index = e_king × sq_k + (p >= e_king ? p - SQ_NB : p)` (後手玉を自玉 plane に collapse、dim=73,305)

use super::SparseInputType;
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::{E_KING, F_KING, FE_HAND_END},
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece},
};

// =============================================================================
// 定数
// =============================================================================

/// nnue-pytorch / YaneuraOu 互換の特徴量 hash (HalfKA_hm2, Friend)。
/// C++ 側: `0x7f234cb9u ^ Side::kFriend(=1)` = `0x7f234cb8`.
pub const FEATURE_HASH_HALFKA_HM2: u32 = 0x7f234cb8;

/// キングバケット数 (5 筋 × 9 段)。玉が file >= 5 なら file-mirror で 0..=4 に畳む。
pub const NUM_KING_BUCKETS: usize = 5 * 9;

/// 駒入力数 (= `e_king` = 1629)。両玉を **同じ 81 マス plane** に投影するため
/// v1 より SQ_NB (= 81) 小さい。
pub const PIECE_INPUTS: usize = E_KING as usize;

/// HalfKA_hm2 の総入力次元 = 45 × 1629 = **73,305**.
pub const HALFKA_HM2_DIMENSIONS: usize = NUM_KING_BUCKETS * PIECE_INPUTS;

/// 最大アクティブ特徴数 (= `PIECE_NUMBER_NB` = 40 駒全部、両玉含む)。
pub const MAX_ACTIVE_FEATURES: usize = 40;

// =============================================================================
// ShogiHalfKaHm2 特徴量型
// =============================================================================

/// ShogiHalfKaHm2 特徴量 (strict v2)。
///
/// やねうら王 `HalfKA_hm2<Side::kFriend>` と完全互換。後手玉 BonaPiece を
/// 自玉 plane に collapse する点が v1 (`ShogiHalfKaHm1`) との違い。
/// やねうら王 `YANEURAOU_ENGINE_NNUE_SFNNwoP1536` ビルドで使われる正規の入力特徴量。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHm2;

impl SparseInputType for ShogiHalfKaHm2 {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM2_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_halfka_hm2_features(&board, f);
    }

    fn shorthand(&self) -> String {
        "shogi-73305x45-hm2".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKA_hm2: 45 king buckets (file-mirror), 1629 piece inputs (enemy king collapsed onto friend plane)".to_string()
    }
}

// =============================================================================
// インデックス計算
// =============================================================================

/// 81 マス (file*9+rank) の file 反転。
#[inline]
fn mirror_file_idx(sq_idx: usize) -> usize {
    let file = sq_idx / 9;
    let rank = sq_idx % 9;
    (8 - file) * 9 + rank
}

/// HalfKA_hm2 のインデックス計算。
///
/// `king_sq_idx`: 視点反転済みの玉位置 (0..=80)。
/// `bp`: 視点反転済みの BonaPiece 値 (0..=1709、両王の plane を含む入力)。
///
/// 2 段階の畳み込み:
/// 1. **File-mirror**: 玉 file >= 5 なら玉位置と盤上駒/王 BonaPiece の sq を mirror
/// 2. **King-plane collapse**: 敵王 BonaPiece (>= e_king) を SQ_NB だけ前にずらして
///    友軍王 plane (= f_king..e_king) に投影
#[inline]
fn halfka_hm2_index(king_sq_idx: usize, bp: usize) -> usize {
    // 1. File-mirror (v1 と同じロジック)
    let (eff_king, mut eff_bp) = if king_sq_idx / 9 >= 5 {
        let mirrored_king = mirror_file_idx(king_sq_idx);
        let mirrored_bp = if bp >= FE_HAND_END {
            let offset = bp - FE_HAND_END;
            let piece_idx = offset / 81;
            let sq = offset % 81;
            FE_HAND_END + piece_idx * 81 + mirror_file_idx(sq)
        } else {
            bp
        };
        (mirrored_king, mirrored_bp)
    } else {
        (king_sq_idx, bp)
    };

    // 2. King-plane collapse: 敵王 (>= e_king) を自玉 plane に投影
    if eff_bp >= E_KING as usize {
        eff_bp -= 81;
    }

    // eff_king は file ∈ {0..=4}, rank ∈ {0..=8} なので file*9+rank ∈ {0..=44}。
    // eff_bp は collapse 後 0..=1628 (= e_king-1) の範囲。
    // HALFKA_HM2_DIMENSIONS = 45 × 1629 に収まる。
    eff_king * PIECE_INPUTS + eff_bp
}

// =============================================================================
// 特徴量列挙
// =============================================================================

/// HalfKA_hm2 特徴量を (stm, nstm) ペアで列挙する。
fn map_halfka_hm2_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
    let stm = board.side_to_move;
    let nstm = stm.opponent();

    let stm_king_sq = board.king_square(stm);
    let nstm_king_sq = board.king_square(nstm);
    if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
        return;
    }

    let stm_ksq = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
    let nstm_ksq = if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };

    // 盤上駒 (王以外)
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);

                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let stm_idx = halfka_hm2_index(stm_ksq, stm_bp.value() as usize);

                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                let nstm_idx = halfka_hm2_index(nstm_ksq, nstm_bp.value() as usize);

                f(stm_idx, nstm_idx);
            }
        }
    }

    // 王の特徴量 (両王とも index 計算で collapse される)。
    {
        let stm_friend_king_sq = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
        let stm_friend_bp = (F_KING as usize) + stm_friend_king_sq;
        let stm_friend_idx = halfka_hm2_index(stm_ksq, stm_friend_bp);

        let stm_enemy_king_sq = if stm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
        let stm_enemy_bp = (E_KING as usize) + stm_enemy_king_sq;
        let stm_enemy_idx = halfka_hm2_index(stm_ksq, stm_enemy_bp);

        let nstm_friend_king_sq = if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
        let nstm_friend_bp = (F_KING as usize) + nstm_friend_king_sq;
        let nstm_friend_idx = halfka_hm2_index(nstm_ksq, nstm_friend_bp);

        let nstm_enemy_king_sq = if nstm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
        let nstm_enemy_bp = (E_KING as usize) + nstm_enemy_king_sq;
        let nstm_enemy_idx = halfka_hm2_index(nstm_ksq, nstm_enemy_bp);

        f(stm_friend_idx, nstm_friend_idx);
        f(stm_enemy_idx, nstm_enemy_idx);
    }

    // 手駒
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
                let stm_idx = halfka_hm2_index(stm_ksq, stm_bp.value() as usize);

                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                let nstm_idx = halfka_hm2_index(nstm_ksq, nstm_bp.value() as usize);

                f(stm_idx, nstm_idx);
            }
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::shogi::{PieceType, Square};

    use super::*;

    #[test]
    fn test_dimensions() {
        assert_eq!(ShogiHalfKaHm2.num_inputs(), 73_305);
        assert_eq!(ShogiHalfKaHm2.max_active(), 40);
        assert_eq!(PIECE_INPUTS, 1629);
        assert_eq!(NUM_KING_BUCKETS, 45);
    }

    #[test]
    fn test_feature_hash_matches_yaneuraou() {
        // YaneuraOu features/half_ka_hm2.h: kHashValue = 0x7f234cb9u ^ Friend(=1)
        assert_eq!(FEATURE_HASH_HALFKA_HM2, 0x7f234cb8_u32);
    }

    #[test]
    fn test_index_no_mirror_below_file_5() {
        for file in 0u8..=4 {
            let king = (file as usize) * 9 + 0;
            let idx = halfka_hm2_index(king, 100);
            assert_eq!(idx, king * PIECE_INPUTS + 100);
            assert!(idx < HALFKA_HM2_DIMENSIONS);
        }
    }

    #[test]
    fn test_index_collapses_enemy_king_to_friend_plane() {
        // king at file 0 (no mirror). 敵王 BonaPiece が 自玉 plane (= F_KING..E_KING) に
        // collapse される ことを確認。
        let king = 0;
        let enemy_king_sq = 40;
        let bp_enemy = (E_KING as usize) + enemy_king_sq;
        let idx = halfka_hm2_index(king, bp_enemy);

        // 期待: 自玉と同じ場所に collapse される
        let expected_collapsed_bp = (F_KING as usize) + enemy_king_sq;
        assert_eq!(idx, king * PIECE_INPUTS + expected_collapsed_bp);
        assert!(idx < HALFKA_HM2_DIMENSIONS);

        // 同じ index になることを確認 (v2 の本質):
        let bp_friend = (F_KING as usize) + enemy_king_sq;
        let idx_friend = halfka_hm2_index(king, bp_friend);
        assert_eq!(idx, idx_friend, "敵王と自玉が同じ sq にあれば同じ index");
    }

    #[test]
    fn test_index_mirror_at_or_above_file_5() {
        let king = 5 * 9 + 2;
        let bp_hand = 10usize;
        let idx = halfka_hm2_index(king, bp_hand);
        assert_eq!(idx, (3 * 9 + 2) * PIECE_INPUTS + bp_hand);

        let king = 8 * 9 + 5;
        let idx = halfka_hm2_index(king, bp_hand);
        assert_eq!(idx, (0 * 9 + 5) * PIECE_INPUTS + bp_hand);
    }

    #[test]
    fn test_index_mirror_and_collapse_combined() {
        // king at file 6 mirrors to file 2. Enemy king BonaPiece sq mirrored AND collapsed.
        let king = 6 * 9 + 4;
        let enemy_king_sq = 7 * 9 + 3; // file 7, rank 3
        let bp = (E_KING as usize) + enemy_king_sq;
        let idx = halfka_hm2_index(king, bp);

        // 1. file-mirror: king → file 2, enemy_king_sq → file 1
        // 2. collapse: E_KING + mirrored_sq → F_KING + mirrored_sq
        let expected_king = mirror_file_idx(king);
        let expected_bp = (F_KING as usize) + mirror_file_idx(enemy_king_sq);
        assert_eq!(idx, expected_king * PIECE_INPUTS + expected_bp);
        assert!(idx < HALFKA_HM2_DIMENSIONS);
    }

    #[test]
    fn test_map_features_initial_position_count() {
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

        let mut count = 0;
        map_halfka_hm2_features(&board, |_, _| count += 1);

        // 歩 18 + 王 2 = 20
        assert_eq!(count, 20);
    }

    #[test]
    fn test_map_features_indices_in_range() {
        for king_file in 0u8..9 {
            let mut board = ShogiBoard {
                side_to_move: Color::Black,
                black_king_sq: Square::new(king_file, 8),
                white_king_sq: Square::new(8 - king_file, 0),
                ..Default::default()
            };
            board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
            board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

            for file in 0..9 {
                board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
                board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
            }

            let max_valid = HALFKA_HM2_DIMENSIONS - 1;
            map_halfka_hm2_features(&board, |stm, nstm| {
                assert!(stm <= max_valid, "STM index {} OOB (king_file={}, max {})", stm, king_file, max_valid);
                assert!(nstm <= max_valid, "NSTM index {} OOB (king_file={}, max {})", nstm, king_file, max_valid);
            });
        }
    }

    #[test]
    fn test_mirror_symmetry() {
        let mut left = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(2, 8),
            white_king_sq: Square::new(2, 0),
            ..Default::default()
        };
        left.board[left.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        left.board[left.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        left.board[Square::new(1, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);

        let mut right = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(6, 8),
            white_king_sq: Square::new(6, 0),
            ..Default::default()
        };
        right.board[right.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        right.board[right.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        right.board[Square::new(7, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);

        let mut left_set: Vec<(usize, usize)> = Vec::new();
        map_halfka_hm2_features(&left, |s, n| left_set.push((s, n)));
        let mut right_set: Vec<(usize, usize)> = Vec::new();
        map_halfka_hm2_features(&right, |s, n| right_set.push((s, n)));

        left_set.sort();
        right_set.sort();
        assert_eq!(left_set, right_set);
    }

    #[test]
    fn test_v2_dimensions_smaller_than_v1() {
        // Sanity: v2 は v1 (= 5*9*1710 = 76,950) より dim が小さい。
        use super::super::shogi_halfka_hm1::HALFKA_HM1_DIMENSIONS;
        assert!(HALFKA_HM2_DIMENSIONS < HALFKA_HM1_DIMENSIONS);
        assert_eq!(HALFKA_HM1_DIMENSIONS - HALFKA_HM2_DIMENSIONS, 45 * 81);
    }
}
