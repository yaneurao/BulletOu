//! ShogiHalfKpe9 — YaneuraOu の `halfkpe9_256x2-32-32.h` で使われる
//! `RawFeatures = HalfKPE9` に対応する入力特徴量。
//!
//! HalfKP の `(own_king_sq, friend_piece_bonapiece)` ペアに、その駒が
//! 占めているマスの **利き数情報** を 9 通り (= 3 × 3) ぶん多重化した
//! バリアント。具体的には:
//!
//!     index = fe_end × sq_k + p
//!           + fe_end × SQ_NB × (effect1 × 3 + effect2)
//!
//! - `effect1` = perspective から見た **自軍** がそのマスへ与えている利き数
//!   (0/1/2 にクリップ)
//! - `effect2` = perspective から見た **相手軍** がそのマスへ与えている利き数
//!   (0/1/2 にクリップ)
//!
//! 出典: `source/eval/nnue/features/half_kpe9.{h,cpp}` および
//! `features/pe9.cpp` の `MakeIndex` / `GetEffectCount` / `AppendActiveIndices`。
//!
//! 手駒は盤上の sq を持たないので effect1 = effect2 = 0 とし、`effect bucket = 0`
//! (= HalfKP と同じ index 領域) に発火する。
//!
//! ## 次元
//!
//! - 入力次元 (perspective あたり): 81 × 1548 × 9 = 1,128,492
//! - max active: 38 (玉以外の駒、HalfKP と同じ)
//! - `FEATURE_HASH`: HalfKP と同値 `0x5D69D5B8` (`0x5D69D5B9 ^ 1` for Friend)
//!   ※ engine は description 文字列 (`HalfKPE9(Friend)`) と入力次元で
//!     HalfKP と HalfKPE9 を判別する。

use super::SparseInputType;
use super::shogi_halfka_hm_threat::{Occupied, for_each_attack};
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::FE_OLD_END,
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, PieceType},
};

/// nnue-pytorch / YaneuraOu 互換の feature hash (HalfKP Friend と同値)。
/// 識別はバイナリ層 hash + description 文字列で行われる。
pub const FEATURE_HASH_HALFKPE9: u32 = 0x5D69D5B8;

/// fe_end (= 1548)。
pub const FE_END: usize = FE_OLD_END;

/// 玉のマス数 (81)。
pub const NUM_KING_SQ: usize = 81;

/// 利き数 bucket 数 = `(0/1/2) × (0/1/2)` = 9。
pub const EFFECT_BUCKETS: usize = 9;

/// HalfKPE9 全体の次元 = 81 × 1548 × 9 = 1,128,492。
pub const HALFKPE9_DIMENSIONS: usize = NUM_KING_SQ * FE_END * EFFECT_BUCKETS;

/// 同時アクティブ最大数 (玉 2 枚を除く 38 駒)。
pub const MAX_ACTIVE_FEATURES: usize = 38;

/// ShogiHalfKpe9 入力特徴量。dual-perspective。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKpe9;

impl SparseInputType for ShogiHalfKpe9 {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKPE9_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_halfkpe9_features(&board, f);
    }

    fn shorthand(&self) -> String {
        "shogi-halfkpe9-1128492x81".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKPE9: 81 king squares × 1548 piece inputs × 9 effect-count buckets".to_string()
    }
}

/// 盤上各マスへの利き数を `[color][sq]` 2 次元配列で返す (0/1/2 にクリップ)。
///
/// 玉を含むすべての駒種を列挙し、`for_each_attack` で各駒の利き先マスを
/// インクリメントする。slider 駒 (角・飛・馬・竜) は遮蔽考慮で
/// `for_each_attack` が正しく扱う。
fn compute_effect_counts(board: &ShogiBoard) -> [[u8; 81]; 2] {
    let occ = Occupied::from_board(board);
    let mut counts = [[0u8; 81]; 2];
    let all_pts = BOARD_PIECE_TYPES.iter().copied().chain(std::iter::once(PieceType::King));
    for color in [Color::Black, Color::White] {
        for pt in all_pts.clone() {
            for from_sq in board.pieces(color, pt) {
                for_each_attack(pt, color, from_sq, &occ, |to_sq| {
                    let idx = to_sq.index();
                    let c = color as usize;
                    if counts[c][idx] < 2 {
                        counts[c][idx] += 1;
                    }
                });
            }
        }
    }
    counts
}

/// HalfKPE9 の発火 index を計算。
#[inline]
fn make_index(king_sq: usize, bonapiece: usize, effect1: u8, effect2: u8) -> usize {
    let eff_bucket = (effect1 as usize) * 3 + (effect2 as usize);
    eff_bucket * NUM_KING_SQ * FE_END + king_sq * FE_END + bonapiece
}

/// HalfKPE9 の発火 index を `f(stm_idx, nstm_idx)` 形式で列挙する。
/// 片玉/詰将棋データ (玉位置 = SQ_NB) はスキップ。
fn map_halfkpe9_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
    let stm = board.side_to_move;
    let nstm = stm.opponent();

    let stm_king_sq = board.king_square(stm);
    let nstm_king_sq = board.king_square(nstm);
    if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
        return;
    }

    let stm_ksq =
        if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
    let nstm_ksq =
        if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };

    let effect_counts = compute_effect_counts(board);
    let cnt_stm = &effect_counts[stm as usize];
    let cnt_nstm = &effect_counts[nstm as usize];

    // ---- 盤上の玉以外の駒 ------------------------------------------------
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);
                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);

                let sq_abs = sq.index();
                // perspective ごとに「自軍」「相手軍」が入れ替わる
                let stm_e1 = cnt_stm[sq_abs]; // STM 視点での自軍
                let stm_e2 = cnt_nstm[sq_abs]; // STM 視点での相手軍
                let nstm_e1 = cnt_nstm[sq_abs]; // NSTM 視点での自軍
                let nstm_e2 = cnt_stm[sq_abs]; // NSTM 視点での相手軍

                let stm_idx = make_index(stm_ksq, stm_bp.value() as usize, stm_e1, stm_e2);
                let nstm_idx = make_index(nstm_ksq, nstm_bp.value() as usize, nstm_e1, nstm_e2);
                f(stm_idx, nstm_idx);
            }
        }
    }

    // ---- 手駒 (盤上 sq を持たないので effect1 = effect2 = 0) -------------
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
                let stm_idx = make_index(stm_ksq, stm_bp.value() as usize, 0, 0);
                let nstm_idx = make_index(nstm_ksq, nstm_bp.value() as usize, 0, 0);
                f(stm_idx, nstm_idx);
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
        let f = ShogiHalfKpe9;
        assert_eq!(f.num_inputs(), 1_128_492);
        assert_eq!(f.max_active(), 38);
        assert_eq!(HALFKPE9_DIMENSIONS, 81 * 1548 * 9);
    }

    #[test]
    fn feature_hash_matches_yaneuraou() {
        // YaneuraOu features/half_kpe9.h の Friend variant:
        //   kHashValue = 0x5D69D5B9u ^ (AssociatedKing == Side::kFriend)
        //              = 0x5D69D5B9 ^ 1 = 0x5D69D5B8
        // (= HalfKP Friend の hash と同値。description で識別される。)
        assert_eq!(FEATURE_HASH_HALFKPE9, 0x5D69D5B8u32);
    }

    #[test]
    fn make_index_layout() {
        // (effect1, effect2) = (0, 0) は HalfKP と同じ index に発火する
        // (= bucket 0)
        let kp_like = make_index(0, 0, 0, 0);
        assert_eq!(kp_like, 0);
        let kp_like_max = make_index(80, 1547, 0, 0);
        assert_eq!(kp_like_max, 80 * 1548 + 1547);

        // (effect1, effect2) = (2, 2) は最大 bucket (8) に発火
        let max_bucket_start = make_index(0, 0, 2, 2);
        assert_eq!(max_bucket_start, 8 * 81 * 1548);

        // 最大値
        let absolute_max = make_index(80, 1547, 2, 2);
        assert_eq!(absolute_max, HALFKPE9_DIMENSIONS - 1);
    }

    #[test]
    fn make_index_effect_bucket_order() {
        // effect bucket index = effect1 * 3 + effect2
        //   (0,0)=0  (0,1)=1  (0,2)=2
        //   (1,0)=3  (1,1)=4  (1,2)=5
        //   (2,0)=6  (2,1)=7  (2,2)=8
        let unit = NUM_KING_SQ * FE_END;
        assert_eq!(make_index(0, 0, 0, 1), 1 * unit);
        assert_eq!(make_index(0, 0, 1, 0), 3 * unit);
        assert_eq!(make_index(0, 0, 1, 1), 4 * unit);
        assert_eq!(make_index(0, 0, 2, 0), 6 * unit);
    }

    fn make_test_board_with_pawns() -> ShogiBoard {
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
        board
    }

    #[test]
    fn map_features_emits_38_pieces_at_most() {
        let board = make_test_board_with_pawns();
        let mut count = 0;
        map_halfkpe9_features(&board, |_, _| count += 1);
        // 玉以外 = 18 (歩のみ) なので 18 件 emit (HalfKP と同じ count)
        assert_eq!(count, 18);
    }

    #[test]
    fn map_features_all_in_range() {
        let board = make_test_board_with_pawns();
        let max = HALFKPE9_DIMENSIONS - 1;
        map_halfkpe9_features(&board, |stm_idx, nstm_idx| {
            assert!(stm_idx <= max, "stm {stm_idx} > max {max}");
            assert!(nstm_idx <= max, "nstm {nstm_idx} > max {max}");
        });
    }

    #[test]
    fn map_features_skip_one_king() {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::NONE,
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        let mut count = 0;
        map_halfkpe9_features(&board, |_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn effect_counts_initial_position_pawn_attacked() {
        // 玉 + 歩 9 枚ずつ。先手の各歩 (5七) には先手側の利き数が 0、
        // 後手側からの利きは隣の歩からは 0 (歩は前方のみ)。
        // 黒の歩 (file=0, rank=6) は file=0, rank=5 を攻撃する。
        // file=1, rank=5 にいる駒は無いが、利き table 上は値 1 になっているはず
        let board = make_test_board_with_pawns();
        let counts = compute_effect_counts(&board);
        // 黒の歩 9 枚はそれぞれ 1 マス前 (rank=5) を攻撃する
        for file in 0..9 {
            let target = Square::new(file, 5).index();
            assert_eq!(counts[Color::Black as usize][target], 1, "file {file} attacked by black pawn");
        }
        // 同様に白の歩 (rank=2) はそれぞれ 1 マス前 (rank=3、相手目線では下) を攻撃
        for file in 0..9 {
            let target = Square::new(file, 3).index();
            assert_eq!(counts[Color::White as usize][target], 1, "file {file} attacked by white pawn");
        }
    }
}
