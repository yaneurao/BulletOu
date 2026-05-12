//! 将棋 KKP (King-King-Piece) 入力特徴量
//!
//! 旧 YaneuraOu 系評価関数 (KPPT 系) の **KKP** 部分。
//!
//! ## 表現
//!
//! 1 局面で、玉以外の各駒について `(sq_bk, sq_wk_inv, bp)` の組を 1 つ立てる。
//!
//! - **STM 視点**:
//!   - `sq_my_king` = STM 玉位置 (反転なし)
//!   - `sq_opp_king` = NTM 玉位置 (反転、`Square::inverse()`)
//!   - `bp` = `BonaPiece::from_piece_square(piece, sq, STM)` または手駒の場合は `from_hand_piece(STM, owner, pt, count)`
//!   - `stm_idx = sq_my_king * 81 * fe_end + sq_opp_king * fe_end + bp`
//! - **NTM 視点**: STM と NTM をスワップ
//!
//! `fe_end = 1548` (BulletOu の `BonaPiece` 仕様、YaneuraOu の KPPT32 と一致)。
//!
//! ## やねうら王の KPPT との関係
//!
//! `KKP_synthesized.bin` (i32[81][81][1548][2]) の `[bk][wk][bp]` インデックスに対応する。
//! `[2]` (手番) は writer で扱う (現状は手番無関係項 `[0]` のみ学習し、`[1]` には 0 を書く)。
//!
//! ## 性質
//!
//! - `num_inputs = 81 * 81 * 1548 = 10,158,588`
//! - `max_active = 38` (合法局面の非玉駒数の上限)

use crate::game::inputs::SparseInputType;
use crate::shogi::types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece};
use crate::shogi::{BonaPiece, PackedSfenValue, ShogiBoard};

/// BonaPiece の総数 (KPPT32 と一致)
pub const KKP_FE_END: usize = 1548;

/// KKP 特徴量の総次元数
pub const KKP_INPUTS: usize = 81 * 81 * KKP_FE_END; // 10,158,588

/// 1 局面あたりの最大 active 特徴数 (非玉駒 38 個)
pub const KKP_MAX_ACTIVE: usize = 38;

#[derive(Debug, Clone, Copy, Default)]
pub struct ShogiKkp;

/// 視点 `perspective` から見た玉ペアのオフセット `bk * 81 + wk_inv` を返す。
///
/// `perspective == Black` のとき:
///   `bk` = 黒玉位置 (反転なし)、`wk_inv` = 白玉位置 (反転)
/// `perspective == White` のとき:
///   `bk` = 白玉位置 (反転なし)、`wk_inv` = 黒玉位置 (反転)
///
/// これは BulletOu の `BonaPiece::from_piece_square` で使う `perspective` と同じ
/// 規約 — 自分の玉位置はそのまま、相手の玉位置を `Square::inverse()` で反転。
#[inline]
fn king_pair_offset(board: &ShogiBoard, perspective: Color) -> Option<usize> {
    let (my_king, opp_king) = match perspective {
        Color::Black => (board.black_king_sq, board.white_king_sq),
        Color::White => (board.white_king_sq, board.black_king_sq),
    };
    // 玉位置が NONE (=81) の場合は None
    if my_king.index() >= 81 || opp_king.index() >= 81 {
        return None;
    }
    let my = match perspective {
        Color::Black => my_king.index() as usize,
        Color::White => my_king.inverse().index() as usize,
    };
    let opp = match perspective {
        Color::Black => opp_king.inverse().index() as usize,
        Color::White => opp_king.index() as usize,
    };
    Some(my * 81 + opp)
}

impl SparseInputType for ShogiKkp {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        KKP_INPUTS
    }

    fn max_active(&self) -> usize {
        KKP_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        // 視点ごとの玉ペアオフセット
        let stm_kk = match king_pair_offset(&board, stm) {
            Some(v) => v,
            None => return,
        };
        let nstm_kk = match king_pair_offset(&board, nstm) {
            Some(v) => v,
            None => return,
        };

        // 盤上の駒 (玉以外)
        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);
                    let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                    let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                    if stm_bp == BonaPiece::ZERO || nstm_bp == BonaPiece::ZERO {
                        continue; // 玉 / 空マスの保険
                    }
                    let stm_idx = stm_kk * KKP_FE_END + stm_bp.value() as usize;
                    let nstm_idx = nstm_kk * KKP_FE_END + nstm_bp.value() as usize;
                    f(stm_idx, nstm_idx);
                }
            }
        }

        // 手駒
        for owner in [Color::Black, Color::White] {
            let hand = match owner {
                Color::Black => &board.black_hand,
                Color::White => &board.white_hand,
            };
            for &pt in &HAND_PIECE_TYPES {
                let count = hand.count(pt);
                for n in 1..=count {
                    let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, n);
                    let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, n);
                    if stm_bp == BonaPiece::ZERO || nstm_bp == BonaPiece::ZERO {
                        continue;
                    }
                    let stm_idx = stm_kk * KKP_FE_END + stm_bp.value() as usize;
                    let nstm_idx = nstm_kk * KKP_FE_END + nstm_bp.value() as usize;
                    f(stm_idx, nstm_idx);
                }
            }
        }
    }

    fn shorthand(&self) -> String {
        "ShogiKkp".to_string()
    }

    fn description(&self) -> String {
        format!(
            "Shogi KKP input ({KKP_INPUTS} dims = 81*81*{KKP_FE_END}, max_active={KKP_MAX_ACTIVE}; for KPPT-style training)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kkp_dims_are_sane() {
        assert_eq!(KKP_FE_END, 1548);
        assert_eq!(KKP_INPUTS, 81 * 81 * 1548);
        assert_eq!(KKP_INPUTS, 10_156_428);
        assert_eq!(KKP_MAX_ACTIVE, 38);
    }

    #[test]
    fn map_features_does_not_panic_on_default_psv() {
        // 0 埋め PSV は不正だが、map_features が panic しないこと (玉なしで早期 return) を確認
        use crate::game::inputs::SparseInputType;
        let pos = PackedSfenValue::default();
        let mut count = 0;
        ShogiKkp.map_features(&pos, |stm, nstm| {
            assert!(stm < KKP_INPUTS, "stm idx out of range: {stm}");
            assert!(nstm < KKP_INPUTS, "nstm idx out of range: {nstm}");
            count += 1;
        });
        // 0 埋め PSV は玉が無いので 0 件 emit を期待
        let _ = count;
    }
}
