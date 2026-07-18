//! ShogiHalfKaHm1 - 将棋用 HalfKA_hm1 特徴量 (strict v1)
//!
//! やねうら王 `Features::HalfKA_hm1` (`features/half_ka_hm1.{h,cpp}`) と byte-for-byte
//! 一致する HalfKA + half-mirror 特徴量。両玉を **別の plane に区別して** 含める。
//!
//! - キングバケット: **45** (= 5 筋 × 9 段、玉が 6 筋以降のとき file-mirror)
//! - 駒入力数: **1710** (= `fe_end2` = `e_king + SQ_NB`、両玉の plane を別々に保持)
//! - 入力次元: **76,950** (= 45 × 1710)
//! - 最大アクティブ特徴: 40 (全駒)
//! - feature hash: `0x7f134cb8` (= `0x7f134cb9 ^ 1` for `Side::kFriend`)
//!
//! v2 (`ShogiHalfKaHm2`) との違いは index 計算式のみ:
//! - **v1**: `index = fe_end2 × sq_k + p` (両玉を別 plane に保持)
//! - v2: `index = e_king × sq_k + (p >= e_king ? p - SQ_NB : p)` (後手玉を自玉 plane に collapse)

use super::SparseInputType;
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::{E_KING, F_KING, FE_HAND_END},
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece},
};

// =============================================================================
// 定数
// =============================================================================

/// nnue-pytorch / YaneuraOu 互換の特徴量 hash (HalfKA_hm1, Friend)。
/// C++ 側: `0x7f134cb9u ^ Side::kFriend(=1)` = `0x7f134cb8`.
pub const FEATURE_HASH_HALFKA_HM1: u32 = 0x7f134cb8;

/// キングバケット数 (5 筋 × 9 段)。玉が file >= 5 なら file-mirror で 0..=4 に畳む。
pub const NUM_KING_BUCKETS: usize = 5 * 9;

/// 駒入力数 (= `fe_end2` = `e_king + SQ_NB` = 1629 + 81 = **1710**)。
/// 両玉の plane を別々に持つ点が v2 と異なる。
pub const PIECE_INPUTS: usize = (E_KING as usize) + 81;

/// HalfKA_hm1 の総入力次元 = 45 × 1710 = **76,950**.
pub const HALFKA_HM1_DIMENSIONS: usize = NUM_KING_BUCKETS * PIECE_INPUTS;

/// 最大アクティブ特徴数 (= `PIECE_NUMBER_NB` = 40 駒全部、両玉含む)。
pub const MAX_ACTIVE_FEATURES: usize = 40;

// =============================================================================
// ShogiHalfKaHm1 特徴量型
// =============================================================================

/// ShogiHalfKaHm1 特徴量 (strict v1)。
///
/// やねうら王 `HalfKA_hm1<Side::kFriend>` と完全互換。両玉を別 plane に
/// 保持する点が v2 (`ShogiHalfKaHm2`) との違い。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHm1;

impl SparseInputType for ShogiHalfKaHm1 {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM1_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_halfka_hm1_features(&board, f);
    }

    fn shorthand(&self) -> String {
        "shogi-76950x45-hm1".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKA_hm1: 45 king buckets (file-mirror), 1710 piece inputs (both kings on separate planes)".to_string()
    }
}

// =============================================================================
// インデックス計算
// =============================================================================

/// 81 マス (file*9+rank) の file 反転 (file 0 ↔ file 8, ...)。
#[inline]
fn mirror_file_idx(sq_idx: usize) -> usize {
    let file = sq_idx / 9;
    let rank = sq_idx % 9;
    (8 - file) * 9 + rank
}

/// HalfKA_hm1 のインデックス計算。
///
/// `king_sq_idx`: 視点反転済みの玉位置 (0..=80)。
/// `bp`: 視点反転済みの BonaPiece 値 (0..=1709、両王の plane を含む)。
///
/// 玉が file >= 5 なら玉位置と **盤上駒 / 王** の BonaPiece 内の sq を mirror。
/// 持駒 BonaPiece (`< FE_HAND_END`) は仮想エンコードのため不変。
#[inline]
fn halfka_hm1_index(king_sq_idx: usize, bp: usize) -> usize {
    let (eff_king, eff_bp) = if king_sq_idx / 9 >= 5 {
        let mirrored_king = mirror_file_idx(king_sq_idx);
        // 盤上駒・王 (BonaPiece >= FE_HAND_END) の sq 部分を mirror。
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

    // 反転後の eff_king は file ∈ {0..=4}, rank ∈ {0..=8} なので
    // file*9+rank ∈ {0..=44}。HALFKA_HM1_DIMENSIONS = 45 × PIECE_INPUTS に収まる。
    eff_king * PIECE_INPUTS + eff_bp
}

// =============================================================================
// 特徴量列挙
// =============================================================================

/// HalfKA_hm1 特徴量を (stm, nstm) ペアで列挙する。
fn map_halfka_hm1_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
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
                let stm_idx = halfka_hm1_index(stm_ksq, stm_bp.value() as usize);

                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                let nstm_idx = halfka_hm1_index(nstm_ksq, nstm_bp.value() as usize);

                f(stm_idx, nstm_idx);
            }
        }
    }

    // 王の特徴量 (v1 では両王を別 plane に出す)。
    // STM 視点での自玉 / 敵玉
    {
        let stm_friend_king_sq = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
        let stm_friend_bp = (F_KING as usize) + stm_friend_king_sq;
        let stm_friend_idx = halfka_hm1_index(stm_ksq, stm_friend_bp);

        let stm_enemy_king_sq = if stm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
        let stm_enemy_bp = (E_KING as usize) + stm_enemy_king_sq;
        let stm_enemy_idx = halfka_hm1_index(stm_ksq, stm_enemy_bp);

        let nstm_friend_king_sq =
            if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
        let nstm_friend_bp = (F_KING as usize) + nstm_friend_king_sq;
        let nstm_friend_idx = halfka_hm1_index(nstm_ksq, nstm_friend_bp);

        let nstm_enemy_king_sq = if nstm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
        let nstm_enemy_bp = (E_KING as usize) + nstm_enemy_king_sq;
        let nstm_enemy_idx = halfka_hm1_index(nstm_ksq, nstm_enemy_bp);

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
                let stm_idx = halfka_hm1_index(stm_ksq, stm_bp.value() as usize);

                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                let nstm_idx = halfka_hm1_index(nstm_ksq, nstm_bp.value() as usize);

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
        assert_eq!(ShogiHalfKaHm1.num_inputs(), 76_950);
        assert_eq!(ShogiHalfKaHm1.max_active(), 40);
        assert_eq!(PIECE_INPUTS, 1710);
        assert_eq!(NUM_KING_BUCKETS, 45);
    }

    #[test]
    fn test_feature_hash_matches_yaneuraou() {
        // YaneuraOu features/half_ka_hm1.h: kHashValue = 0x7f134cb9u ^ Friend(=1)
        assert_eq!(FEATURE_HASH_HALFKA_HM1, 0x7f134cb8_u32);
    }

    #[test]
    fn test_index_no_mirror_below_file_5() {
        for file in 0u8..=4 {
            let king = (file as usize) * 9 + 0;
            let idx = halfka_hm1_index(king, 100);
            assert_eq!(idx, king * PIECE_INPUTS + 100);
            assert!(idx < HALFKA_HM1_DIMENSIONS);
        }
    }

    #[test]
    fn test_index_mirror_at_or_above_file_5() {
        // king at file 5 (= 6 筋) mirrors to file 3 (= 4 筋).
        let king = 5 * 9 + 2;
        let bp_hand = 10usize;
        let idx = halfka_hm1_index(king, bp_hand);
        assert_eq!(idx, (3 * 9 + 2) * PIECE_INPUTS + bp_hand);

        // king at file 8 (= 9 筋) mirrors to file 0 (= 1 筋).
        let king = 8 * 9 + 5;
        let idx = halfka_hm1_index(king, bp_hand);
        assert_eq!(idx, (0 * 9 + 5) * PIECE_INPUTS + bp_hand);
    }

    #[test]
    fn test_index_mirror_board_piece_bp() {
        // king at file 6 mirrors to file 2. Board piece sq at file 7 → file 1.
        let king = 6 * 9 + 4;
        let piece_idx = 1usize;
        let sq_idx = 7 * 9 + 3;
        let bp = FE_HAND_END + piece_idx * 81 + sq_idx;
        let idx = halfka_hm1_index(king, bp);

        let expected_king = mirror_file_idx(king);
        let expected_bp = FE_HAND_END + piece_idx * 81 + mirror_file_idx(sq_idx);
        assert_eq!(idx, expected_king * PIECE_INPUTS + expected_bp);
        assert!(idx < HALFKA_HM1_DIMENSIONS);
    }

    #[test]
    fn test_index_no_collapse_for_enemy_king() {
        // v1 では敵王 BonaPiece (>= E_KING) は plane が別。
        // king at file 0, BonaPiece pointing into E_KING plane.
        let king = 0 * 9 + 4;
        let bp_e_king = (E_KING as usize) + 40; // enemy king at sq 40
        let idx = halfka_hm1_index(king, bp_e_king);

        // 期待: e_king plane の値そのままで index 計算 (collapse なし)
        assert_eq!(idx, king * PIECE_INPUTS + bp_e_king);
        // dim を超えないこと
        assert!(idx < HALFKA_HM1_DIMENSIONS);
    }

    #[test]
    fn test_map_features_initial_position_count() {
        // 初期局面 (王 2 + 飛 2 + 角 2 + 金 4 + 銀 4 + 桂 4 + 香 4 + 歩 18) = 40 駒。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        // 歩のみ配置 (両軍 9 枚ずつ)
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        let mut count = 0;
        map_halfka_hm1_features(&board, |_, _| count += 1);

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

            let max_valid = HALFKA_HM1_DIMENSIONS - 1;
            map_halfka_hm1_features(&board, |stm, nstm| {
                assert!(stm <= max_valid, "STM index {} OOB (king_file={}, max {})", stm, king_file, max_valid);
                assert!(nstm <= max_valid, "NSTM index {} OOB (king_file={}, max {})", nstm, king_file, max_valid);
            });
        }
    }

    #[test]
    fn test_mirror_symmetry() {
        // 左右対称な盤面は同じ特徴量集合を返す。
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
        map_halfka_hm1_features(&left, |s, n| left_set.push((s, n)));
        let mut right_set: Vec<(usize, usize)> = Vec::new();
        map_halfka_hm1_features(&right, |s, n| right_set.push((s, n)));

        left_set.sort();
        right_set.sort();
        assert_eq!(left_set, right_set);
    }
}
