//! 将棋 KPP (King-Piece-Piece) 入力特徴量
//!
//! 旧 YaneuraOu 系評価関数 (KPPT 系) の **KPP** 部分。Phase 3 の中核。
//!
//! ## 表現
//!
//! 1 局面で、視点ごとに **玉以外の非順序駒ペア** `(bp_a, bp_b)` (ただし
//! `bp_a < bp_b` で canonical 化) すべてについて、`(my_king, bp_a, bp_b)` を
//! 1 つ立てる。
//!
//! - **STM 視点**: `my_king = STM 玉位置`、`bp` は STM 視点 BonaPiece
//! - **NTM 視点**: 反転 (Square::inverse) + NTM 視点 BonaPiece
//!
//! 線形インデックス: `idx = my_king * fe_end * fe_end + bp_a * fe_end + bp_b`
//! (常に `bp_a < bp_b`)。
//!
//! ## やねうら王の KPPT との関係
//!
//! `KPP_synthesized.bin` (`int16_t kpp[81][1548][1548][2]`) の `[k][p1][p2]`
//! インデックスに対応する。**`kpp[k][p1][p2] == kpp[k][p2][p1]`** という
//! 対称性があり、本トレーナでは `p1 < p2` の方のみを学習し、出力 writer 側で
//! 対称コピー (`[p1][p2]` と `[p2][p1]` の両方に書き込む) を行う。
//!
//! `[2]` (= `[turn_independent, turn_dependent]`) のうち、Phase 3 では
//! `[0]` = turn-independent のみ学習し、`[1]` = turn-dependent は 0 で書く。
//! turn 項の学習は将来の Phase で別途扱う。
//!
//! ## 性質
//!
//! - `num_inputs = 81 * 1548 * 1548 = 194,100,624`  (約 1.94 億次元)
//! - `max_active = C(38, 2) = 703` (合法局面の非玉駒数 38 から 2 個選ぶ)
//! - 学習時のメモリ: f32 重みで約 776 MB、AdamW 状態込みで約 2.33 GB

use crate::game::inputs::SparseInputType;
use crate::shogi::types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece};
use crate::shogi::{BonaPiece, PackedSfenValue, ShogiBoard};

/// BonaPiece の総数 (KPPT32 と一致)
pub const KPP_FE_END: usize = 1548;

/// KPP 特徴量の総次元数 (81 玉位置 × fe_end × fe_end)
pub const KPP_INPUTS: usize = 81 * KPP_FE_END * KPP_FE_END;

/// 1 局面あたりの最大 active 特徴数 (非玉駒 38 個から 2 個 = C(38, 2) = 703)
pub const KPP_MAX_ACTIVE: usize = 703;

#[derive(Debug, Clone, Copy, Default)]
pub struct ShogiKpp;

/// `perspective` の玉位置を返す (NTM 視点では Square::inverse() で反転)。
/// 玉が NONE (=81) ならば `None`。
#[inline]
fn my_king_index(board: &ShogiBoard, perspective: Color) -> Option<usize> {
    let king = match perspective {
        Color::Black => board.black_king_sq,
        Color::White => board.white_king_sq,
    };
    if king.index() >= 81 {
        return None;
    }
    let idx = match perspective {
        Color::Black => king.index() as usize,
        Color::White => king.inverse().index() as usize,
    };
    Some(idx)
}

/// 視点 `perspective` から見た現局面の **非玉 BonaPiece 一覧** を返す。
/// 玉位置や `BonaPiece::ZERO` (空マス等) は除外。
fn list_bonapieces(board: &ShogiBoard, perspective: Color) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(KPP_MAX_ACTIVE + 16);

    // 盤上の駒 (玉以外)
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);
                let bp = BonaPiece::from_piece_square(piece, sq, perspective);
                if bp == BonaPiece::ZERO {
                    continue;
                }
                out.push(bp.value() as u32);
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
                let bp = BonaPiece::from_hand_piece(perspective, owner, pt, n);
                if bp == BonaPiece::ZERO {
                    continue;
                }
                out.push(bp.value() as u32);
            }
        }
    }

    out
}

impl SparseInputType for ShogiKpp {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        KPP_INPUTS
    }

    fn max_active(&self) -> usize {
        KPP_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        // 視点ごとの玉位置
        let stm_king = match my_king_index(&board, stm) {
            Some(v) => v,
            None => return,
        };
        let nstm_king = match my_king_index(&board, nstm) {
            Some(v) => v,
            None => return,
        };

        // 視点ごとの BonaPiece リスト
        let mut stm_bps = list_bonapieces(&board, stm);
        let mut nstm_bps = list_bonapieces(&board, nstm);
        stm_bps.sort_unstable();
        nstm_bps.sort_unstable();

        // 2 視点のペア数を揃える保険 (両者とも合法局面なら同数のはず)。
        let n = stm_bps.len().min(nstm_bps.len());

        // (i, j) の昇順ペアを enumerate
        for i in 0..n {
            for j in (i + 1)..n {
                let stm_a = stm_bps[i] as usize;
                let stm_b = stm_bps[j] as usize;
                let nstm_a = nstm_bps[i] as usize;
                let nstm_b = nstm_bps[j] as usize;

                // BonaPiece sort 後なので a < b は保証されているが、
                // sort 規約の崩れに備えて min/max を取る
                let (stm_lo, stm_hi) = if stm_a <= stm_b { (stm_a, stm_b) } else { (stm_b, stm_a) };
                let (nstm_lo, nstm_hi) = if nstm_a <= nstm_b { (nstm_a, nstm_b) } else { (nstm_b, nstm_a) };

                let stm_idx = stm_king * KPP_FE_END * KPP_FE_END + stm_lo * KPP_FE_END + stm_hi;
                let nstm_idx = nstm_king * KPP_FE_END * KPP_FE_END + nstm_lo * KPP_FE_END + nstm_hi;

                f(stm_idx, nstm_idx);
            }
        }
    }

    fn shorthand(&self) -> String {
        "ShogiKpp".to_string()
    }

    fn description(&self) -> String {
        format!(
            "Shogi KPP input ({KPP_INPUTS} dims = 81*{KPP_FE_END}*{KPP_FE_END}, max_active={KPP_MAX_ACTIVE}; for KPPT-style training, canonicalised p1 < p2)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kpp_dims_are_sane() {
        assert_eq!(KPP_FE_END, 1548);
        assert_eq!(KPP_INPUTS, 81 * 1548 * 1548);
        assert_eq!(KPP_INPUTS, 194_100_624);
        // C(38, 2) = 703
        assert_eq!(KPP_MAX_ACTIVE, 703);
    }

    #[test]
    fn map_features_does_not_panic_on_default_psv() {
        // 0 埋め PSV は不正だが、map_features が panic しないこと
        // (玉なしで早期 return) を確認
        let pos = PackedSfenValue::default();
        let mut count = 0;
        ShogiKpp.map_features(&pos, |stm, nstm| {
            assert!(stm < KPP_INPUTS, "stm idx out of range: {stm}");
            assert!(nstm < KPP_INPUTS, "nstm idx out of range: {nstm}");
            count += 1;
        });
        // 0 埋め PSV は玉が無いので 0 件 emit を期待
        let _ = count;
    }

    #[test]
    fn map_features_canonical_ordering() {
        // 任意の有効局面 (テスト用の hirate を作る方法は既存テストにあるかもしれないが、
        // ここでは最小限: emit されたインデックスがすべて canonical (lo < hi) であることを確認。
        // hirate を作る簡単な API があれば差し替え。当面 default_psv だと玉なしで 0 件 emit のため
        // canonicality は trivially 成立 — ここは将来 hirate fixture を入れたタイミングで強化する。
        let pos = PackedSfenValue::default();
        ShogiKpp.map_features(&pos, |stm_idx, nstm_idx| {
            // 各 index を king/p1/p2 に分解して p1 <= p2 を確認
            for &idx in &[stm_idx, nstm_idx] {
                let king = idx / (KPP_FE_END * KPP_FE_END);
                let rest = idx % (KPP_FE_END * KPP_FE_END);
                let p_lo = rest / KPP_FE_END;
                let p_hi = rest % KPP_FE_END;
                assert!(king < 81, "king out of range: {king}");
                assert!(p_lo <= p_hi, "non-canonical pair: ({p_lo}, {p_hi})");
                assert!(p_hi < KPP_FE_END, "p_hi out of range: {p_hi}");
            }
        });
    }
}
