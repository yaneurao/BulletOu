//! ShogiKp — YaneuraOu の `k-p_256x2-32-32` (`kp_256x2-32-32.h`) で使われる
//! `RawFeatures = FeatureSet<Features::K, Features::P>` に対応する入力。
//!
//! HalfKP との違い:
//! - HalfKP は **自玉×駒** の (king, piece) 組合せを各 perspective ぶん発火させる
//!   (玉 81 通り × 駒 1548 = 125,388 dim per perspective)。
//! - KP は K (玉) と P (玉以外の駒) を独立した sparse 特徴として並べる。
//!   (`FeatureSet<K, P>` の合成規約により、Tail=P が先、Head=K が
//!   `+Tail::kDimensions` シフトで後ろに。)
//!
//! `Features::K` (`source/eval/nnue/features/k.h`):
//! - `kDimensions = SQ_NB * 2 = 162` (自玉 0..80 / 相手玉 81..161)
//! - `kMaxActiveDimensions = 2` (自玉 + 相手玉)
//! - `kHashValue = 0xD3CEE169`
//!
//! `Features::P` (`source/eval/nnue/features/p.h`):
//! - `kDimensions = fe_end = 1548`
//! - `kMaxActiveDimensions = PIECE_NUMBER_KING = 38` (玉以外の駒)
//! - `kHashValue = 0x764CFB4B`
//!
//! `FeatureSet<K, P>::kHashValue` の合成 (`feature_set.h`):
//!     Head::kHashValue ^ (Tail::kHashValue << 1) ^ (Tail::kHashValue >> 31)
//!   = 0xD3CEE169 ^ (0x764CFB4B << 1) ^ 0
//!   = 0xD3CEE169 ^ 0xEC99F696
//!   = 0x3F5717FF

use super::SparseInputType;
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::FE_OLD_END,
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, Square},
};

/// `FeatureSet<K, P>` の合成 feature hash (= K::kHashValue ^ (P::kHashValue << 1))。
pub const FEATURE_HASH_KP: u32 = 0x3F5717FF;

/// fe_end = 玉以外の BonaPiece 範囲 = P の次元数。
pub const KP_P_DIMENSIONS: usize = FE_OLD_END; // 1548

/// SQ_NB = 81 マス。K は (自玉, 相手玉) を別 slot にして 2x SQ_NB を専有する。
pub const KP_K_DIMENSIONS: usize = 81 * 2; // 162

/// KP 全体の入力次元 = P + K = 1548 + 162 = 1710。
pub const KP_DIMENSIONS: usize = KP_P_DIMENSIONS + KP_K_DIMENSIONS; // 1710

/// 同時 active 数の上限 = P の上限 (38) + K の上限 (2) = 40。
pub const KP_MAX_ACTIVE: usize = 38 + 2;

/// ShogiKp — `k-p_256x2-32-32` 用の入力特徴量。dual-perspective。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiKp;

impl SparseInputType for ShogiKp {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        KP_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        KP_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_kp_features(&board, f);
    }

    fn shorthand(&self) -> String {
        "shogi-kp-1710".to_string()
    }

    fn description(&self) -> String {
        "Shogi K+P: 162 king features + 1548 piece (non-king) features per perspective".to_string()
    }
}

/// 視点 `perspective` から見た square index (0..80)。
/// 後手視点では盤を反転させる (YaneuraOu の piece_list_fw と同じ)。
/// Fill fixed-size STM/NSTM KP sparse feature buffers for one packed shogi
/// position, returning the number of features written on each perspective.
///
/// Unused slots are intentionally left untouched; callers that pass reused
/// buffers must clear the tail themselves. Fresh teacher batches allocate the
/// whole sparse buffer with `-1`, so they can skip per-position tail writes.
pub fn fill_kp_feature_indices(pos: &PackedSfenValue, stm: &mut [i32], nstm: &mut [i32]) -> (usize, usize) {
    debug_assert!(stm.len() >= KP_MAX_ACTIVE);
    debug_assert!(nstm.len() >= KP_MAX_ACTIVE);
    let mut stm_count = 0usize;
    let mut nstm_count = 0usize;
    ShogiKp.map_features(pos, |stm_idx, nstm_idx| {
        debug_assert!(stm_count < stm.len());
        debug_assert!(nstm_count < nstm.len());
        stm[stm_count] = stm_idx as i32;
        nstm[nstm_count] = nstm_idx as i32;
        stm_count += 1;
        nstm_count += 1;
    });
    debug_assert!(stm_count <= KP_MAX_ACTIVE);
    debug_assert!(nstm_count <= KP_MAX_ACTIVE);
    (stm_count, nstm_count)
}

#[inline]
fn sq_from_perspective(sq: Square, perspective: Color) -> usize {
    if perspective == Color::Black { sq.index() } else { sq.inverse().index() }
}

/// K の active index (perspective から見たもの) を計算。
///
/// YaneuraOu の `K::AppendActiveIndices` は `BonaPiece(king_i) - fe_end` を吐く。
/// 自玉の BonaPiece は `fe_end + sq`、相手玉の BonaPiece は `fe_end + SQ_NB + sq`
/// (perspective から見た square)。`FeatureSet<K, P>` は K に Tail (P) の次元数を
/// シフトとして加えるので、最終 active index は:
///   own_king: KP_P_DIMENSIONS + sq
///   opp_king: KP_P_DIMENSIONS + SQ_NB + sq
#[inline]
fn k_index_own(sq_from_p: usize) -> usize {
    KP_P_DIMENSIONS + sq_from_p
}

#[inline]
fn k_index_opp(sq_from_p: usize) -> usize {
    KP_P_DIMENSIONS + 81 + sq_from_p
}

/// K + P 特徴量を `f(stm_idx, nstm_idx)` 形で列挙する。
///
/// 1 つの物理的な駒は STM 視点と NSTM 視点でそれぞれ index を持つので、`f` は
/// 駒 1 つにつき 1 回呼ぶ。
/// 片玉/詰将棋データ (玉位置 = SQ_NB) はスキップ。
fn map_kp_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
    let stm = board.side_to_move;
    let nstm = stm.opponent();

    let stm_king_sq = board.king_square(stm);
    let nstm_king_sq = board.king_square(nstm);
    if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
        return;
    }

    // ---- K: 自玉 / 相手玉 -------------------------------------------------
    //
    // STM 視点では「stm の玉」が自玉、「nstm の玉」が相手玉。
    // NSTM 視点ではその逆。
    let stm_own_king = sq_from_perspective(stm_king_sq, stm);
    let stm_opp_king = sq_from_perspective(nstm_king_sq, stm);
    let nstm_own_king = sq_from_perspective(nstm_king_sq, nstm);
    let nstm_opp_king = sq_from_perspective(stm_king_sq, nstm);

    // stm の玉: STM 視点では自玉、NSTM 視点では相手玉
    f(k_index_own(stm_own_king), k_index_opp(nstm_opp_king));
    // nstm の玉: STM 視点では相手玉、NSTM 視点では自玉
    f(k_index_opp(stm_opp_king), k_index_own(nstm_own_king));

    // ---- P: 玉以外の盤上駒 ------------------------------------------------
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);

                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);

                // P 特徴量は FeatureSet の Tail 側なので shift 無し: index = bp.value()。
                f(stm_bp.value() as usize, nstm_bp.value() as usize);
            }
        }
    }

    // ---- P: 手駒 ----------------------------------------------------------
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
                f(stm_bp.value() as usize, nstm_bp.value() as usize);
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
        let kp = ShogiKp;
        assert_eq!(kp.num_inputs(), 1710);
        assert_eq!(kp.max_active(), 40);
    }

    #[test]
    fn k_index_ranges() {
        // 自玉の最初: KP_P_DIMENSIONS (= 1548)
        assert_eq!(k_index_own(0), 1548);
        // 自玉の最後: KP_P_DIMENSIONS + 80 = 1628
        assert_eq!(k_index_own(80), 1628);
        // 相手玉の最初: KP_P_DIMENSIONS + 81 = 1629
        assert_eq!(k_index_opp(0), 1629);
        // 相手玉の最後: KP_P_DIMENSIONS + 81 + 80 = 1709 (= KP_DIMENSIONS - 1)
        assert_eq!(k_index_opp(80), 1709);
    }

    #[test]
    fn feature_hash_matches_yaneuraou() {
        // YaneuraOu の FeatureSet<K, P>::kHashValue =
        //   K::kHashValue ^ (P::kHashValue << 1) ^ (P::kHashValue >> 31)
        // = 0xD3CEE169 ^ (0x764CFB4B << 1) ^ 0
        // = 0xD3CEE169 ^ 0xEC99F696
        // = 0x3F5717FF
        let k_hash: u32 = 0xD3CEE169;
        let p_hash: u32 = 0x764CFB4B;
        let combined = k_hash ^ (p_hash.wrapping_shl(1)) ^ (p_hash.wrapping_shr(31));
        assert_eq!(combined, FEATURE_HASH_KP);
        assert_eq!(FEATURE_HASH_KP, 0x3F5717FF);
    }

    #[test]
    fn map_features_two_kings_only() {
        // 玉だけある盤面 → K の 2 個だけ emit (= 2 件)。P は 0 件。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        let mut emissions: Vec<(usize, usize)> = Vec::new();
        map_kp_features(&board, |s, n| emissions.push((s, n)));
        assert_eq!(emissions.len(), 2);

        // 全 index が KP_P_DIMENSIONS 以上 = K 領域に入っているはず。
        for (s, n) in &emissions {
            assert!(*s >= KP_P_DIMENSIONS && *s < KP_DIMENSIONS, "stm={s}");
            assert!(*n >= KP_P_DIMENSIONS && *n < KP_DIMENSIONS, "nstm={n}");
        }
    }

    #[test]
    fn map_features_two_kings_plus_pawns() {
        // 玉 + 歩 9 枚 × 2 = 玉 2 (K) + 歩 18 (P) = 20 emit。
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
        let mut p_count = 0;
        map_kp_features(&board, |s, _n| {
            if s >= KP_P_DIMENSIONS {
                k_count += 1;
            } else {
                p_count += 1;
            }
        });
        assert_eq!(k_count, 2);
        assert_eq!(p_count, 18);
    }

    #[test]
    fn map_features_all_in_range() {
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

        let max = KP_DIMENSIONS - 1;
        map_kp_features(&board, |s, n| {
            assert!(s <= max, "stm idx {s} exceeds {max}");
            assert!(n <= max, "nstm idx {n} exceeds {max}");
        });
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
        map_kp_features(&board, |_, _| count += 1);
        assert_eq!(count, 0);
    }
}
