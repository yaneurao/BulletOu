//! ShogiHalfKPvm - 将棋用 HalfKP_vm 特徴量 (vertical-mirror 半畳み込み版)
//!
//! HalfKP と同じ「King × Piece」sparse 特徴量だが、玉が 6 筋〜9 筋にあるときは
//! 4 筋〜1 筋に左右反転 (file-mirror) して同じパラメータを共有する。
//! 玉位置の左右対称性を畳むことで入力次元が HalfKP の **約 1/2** になる。
//!
//! やねうら王の `Features::HalfKP_vm<Side::kFriend>` (`features/half_kp_vm.{h,cpp}`)
//! を Rust に移植したもの。
//!
//! - キングバケット: **45** (= 5 筋 × 9 段)
//! - 入力次元: **69,660** (45 × 1548)
//! - 最大アクティブ特徴: 38 (王 2 枚を除く)
//! - feature hash: `0x0B6B1D9A` (= `0x0B6B1D9B ^ 1` for `Side::kFriend`)

use super::SparseInputType;
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::{FE_HAND_END, FE_OLD_END},
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece},
};

// =============================================================================
// 定数
// =============================================================================

/// nnue-pytorch / YaneuraOu 互換の特徴量 hash 値 (HalfKP_vm, Friend)。
/// C++ 側: `0x0B6B1D9Bu ^ (AssociatedKing == Side::kFriend)` = `0x0B6B1D9A`.
pub const FEATURE_HASH_HALFKPVM: u32 = 0x0B6B1D9A;

/// キングバケット数 (5 筋 × 9 段)。
/// 玉が 6 筋〜9 筋なら 4 筋〜1 筋に file-mirror するため、玉が取りうる
/// 実質的な筋は 1 筋〜5 筋の 5 種類のみ。
pub const NUM_KING_BUCKETS: usize = 5 * 9;

/// 駒入力数 (fe_end = 1548、王を除く)。
pub const FE_END: usize = FE_OLD_END;

/// HalfKP_vm の総入力次元 = 45 × 1548 = 69_660.
pub const HALFKPVM_DIMENSIONS: usize = NUM_KING_BUCKETS * FE_END;

/// 最大アクティブ特徴数 (王 2 枚を除く 38 駒)。
pub const MAX_ACTIVE_FEATURES: usize = 38;

// =============================================================================
// ShogiHalfKPvm 特徴量型
// =============================================================================

/// ShogiHalfKPvm 特徴量。
///
/// `ShogiHalfKP` と同じインターフェース・同じ駒列挙だが、玉位置 (の視点反転後)
/// の筋が 6 筋以上なら BonaPiece の盤上マスと一緒に左右反転する。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKPvm;

impl SparseInputType for ShogiHalfKPvm {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKPVM_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_halfkpvm_features(&board, f);
    }

    fn shorthand(&self) -> String {
        "shogi-69660x45".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKP_vm: 45 king buckets (file-mirror folded), 1548 piece inputs (no kings)".to_string()
    }
}

// =============================================================================
// インデックス計算
// =============================================================================

/// 81 マス (file*9+rank) のうち file を反転する。Square::mirror_file と同じ
/// 計算だが、ここでは `usize` のままで扱いたいのでローカルに持つ。
#[inline]
fn mirror_file_idx(sq_idx: usize) -> usize {
    let file = sq_idx / 9;
    let rank = sq_idx % 9;
    (8 - file) * 9 + rank
}

/// HalfKP_vm のインデックス計算。
///
/// `king_sq_idx`: 視点反転済みの玉位置 (0..=80、= file*9+rank)。
/// `bp`: 視点反転済みの BonaPiece 値 (0..=1547)。
///
/// 玉が 6 筋〜9 筋 (file >= 5) なら玉位置と **盤上駒** の BonaPiece を file-mirror
/// する。持駒 BonaPiece (`< FE_HAND_END = 90`) は反転しない (盤面マスを持たない
/// 仮想的なエンコーディングのため)。
#[inline]
fn halfkpvm_index(king_sq_idx: usize, bp: usize) -> usize {
    let (eff_king, eff_bp) = if king_sq_idx / 9 >= 5 {
        let mirrored_king = mirror_file_idx(king_sq_idx);
        let mirrored_bp = if bp >= FE_HAND_END {
            // 盤上駒: BonaPiece = FE_HAND_END + piece_idx * 81 + sq
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
    // file*9+rank ∈ {0..=44}。HALFKPVM_DIMENSIONS = 45 * FE_END に収まる。
    eff_king * FE_END + eff_bp
}

// =============================================================================
// 特徴量列挙
// =============================================================================

/// HalfKP_vm 特徴量インデックスを (stm, nstm) ペアで列挙する。
///
/// ロジックは [`shogi_halfkp::map_halfkp_features`] と並行で、
/// `halfkp_index` の代わりに `halfkpvm_index` を使うだけ。
fn map_halfkpvm_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
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
                let stm_idx = halfkpvm_index(stm_ksq, stm_bp.value() as usize);

                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                let nstm_idx = halfkpvm_index(nstm_ksq, nstm_bp.value() as usize);

                f(stm_idx, nstm_idx);
            }
        }
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
                let stm_idx = halfkpvm_index(stm_ksq, stm_bp.value() as usize);

                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                let nstm_idx = halfkpvm_index(nstm_ksq, nstm_bp.value() as usize);

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
        let input = ShogiHalfKPvm;
        assert_eq!(input.num_inputs(), 69_660);
        assert_eq!(input.max_active(), 38);
    }

    #[test]
    fn test_shorthand() {
        let input = ShogiHalfKPvm;
        assert_eq!(input.shorthand(), "shogi-69660x45");
    }

    #[test]
    fn test_feature_hash_matches_yaneuraou() {
        // YaneuraOu features/half_kp_vm.h: kHashValue = 0x0B6B1D9Bu ^ Friend.
        // Friend = 1 → 0x0B6B1D9A.
        assert_eq!(FEATURE_HASH_HALFKPVM, 0x0B6B1D9A_u32);
    }

    #[test]
    fn test_mirror_file_idx_symmetric() {
        // file 0 ↔ 8, file 1 ↔ 7, ..., file 4 fixed
        for f in 0u8..9 {
            for r in 0u8..9 {
                let sq = (f as usize) * 9 + (r as usize);
                let mirrored = mirror_file_idx(sq);
                assert_eq!(mirrored / 9, (8 - f) as usize);
                assert_eq!(mirrored % 9, r as usize);
                assert_eq!(mirror_file_idx(mirrored), sq);
            }
        }
    }

    #[test]
    fn test_halfkpvm_index_no_mirror_below_file_5() {
        // king at file 0..=4: no mirror; result = king_sq * FE_END + bp
        for file in 0u8..=4 {
            let king = (file as usize) * 9 + 0;
            let idx = halfkpvm_index(king, 100);
            assert_eq!(idx, king * FE_END + 100);
            assert!(idx < HALFKPVM_DIMENSIONS);
        }
    }

    #[test]
    fn test_halfkpvm_index_mirror_at_or_above_file_5() {
        // king at file 5 (= 6 筋) should mirror to file 3 (= 4 筋).
        // BonaPiece for hand piece (< FE_HAND_END) is NOT mirrored.
        let king = 5 * 9 + 2; // file 5, rank 2
        let bp_hand = 10usize; // hand piece
        let idx = halfkpvm_index(king, bp_hand);
        let expected_king = 3 * 9 + 2; // mirrored to file 3
        assert_eq!(idx, expected_king * FE_END + bp_hand);

        // king at file 8 (= 9 筋) mirrors to file 0 (= 1 筋).
        let king = 8 * 9 + 5;
        let idx = halfkpvm_index(king, bp_hand);
        let expected_king = 0 * 9 + 5;
        assert_eq!(idx, expected_king * FE_END + bp_hand);
    }

    #[test]
    fn test_halfkpvm_index_mirror_board_piece_bp() {
        // king at file 6 (= 7 筋), board-piece BonaPiece pointing to file 7.
        // After mirror: king goes to file 2, board-piece sq goes to file 1.
        let king = 6 * 9 + 4; // file 6, rank 4
        let piece_idx = 1usize; // some piece type
        let sq_idx = 7 * 9 + 3; // sq at file 7, rank 3
        let bp = FE_HAND_END + piece_idx * 81 + sq_idx;
        let idx = halfkpvm_index(king, bp);

        let expected_king = mirror_file_idx(king); // file 2
        let expected_bp = FE_HAND_END + piece_idx * 81 + mirror_file_idx(sq_idx); // file 1
        assert_eq!(idx, expected_king * FE_END + expected_bp);
        assert!(idx < HALFKPVM_DIMENSIONS);
    }

    #[test]
    fn test_map_features_pawn_count() {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::new(4, 0), // 5一
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        let mut count = 0;
        map_halfkpvm_features(&board, |_, _| count += 1);

        // 歩 18 枚 (王は含まない)
        assert_eq!(count, 18);
    }

    #[test]
    fn test_map_features_indices_in_range() {
        // 玉が左側 (file 0=1 筋) と右側 (file 8=9 筋) どちらでもインデックスが
        // HALFKPVM_DIMENSIONS 内に収まることを確認 (mirror が効くため)。
        for king_file in 0u8..9 {
            let mut board = ShogiBoard {
                side_to_move: Color::Black,
                black_king_sq: Square::new(king_file, 8),
                white_king_sq: Square::new(8 - king_file, 0),
                ..Default::default()
            };
            board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
            board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

            // 適当に駒を配置
            for file in 0..9 {
                board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
                board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
            }

            let max_valid = HALFKPVM_DIMENSIONS - 1;
            map_halfkpvm_features(&board, |stm, nstm| {
                assert!(stm <= max_valid, "STM index {} OOB (max {})", stm, max_valid);
                assert!(nstm <= max_valid, "NSTM index {} OOB (max {})", nstm, max_valid);
            });
        }
    }

    #[test]
    fn test_map_features_no_kings_in_features() {
        // HalfKP と同じく、王自体は特徴量に含めない。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        let mut count = 0;
        map_halfkpvm_features(&board, |_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_mirror_consistency_with_halfkp() {
        // 玉位置を mirror した盤面と元の盤面の HalfKP_vm 特徴量集合は同じ。
        use crate::shogi::types::Color;

        // 左右対称な (file 4 = 5 筋) 上に駒を置いた局面 1。
        let mut left = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(2, 8), // 3 筋に玉 → mirror なし
            white_king_sq: Square::new(2, 0),
            ..Default::default()
        };
        left.board[left.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        left.board[left.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        left.board[Square::new(1, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);

        // 局面 1 の file-mirror 版 (file 2 → 6 → mirror 後 file 2、駒 file 1 → 7 → mirror 後 file 1)。
        let mut right = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(6, 8), // 7 筋に玉 → mirror で 3 筋
            white_king_sq: Square::new(6, 0),
            ..Default::default()
        };
        right.board[right.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        right.board[right.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        right.board[Square::new(7, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);

        let mut left_set: Vec<(usize, usize)> = Vec::new();
        map_halfkpvm_features(&left, |s, n| left_set.push((s, n)));
        let mut right_set: Vec<(usize, usize)> = Vec::new();
        map_halfkpvm_features(&right, |s, n| right_set.push((s, n)));

        left_set.sort();
        right_set.sort();
        assert_eq!(left_set, right_set, "file-mirror した盤面は同じ特徴量集合を生成すべき");
    }
}
