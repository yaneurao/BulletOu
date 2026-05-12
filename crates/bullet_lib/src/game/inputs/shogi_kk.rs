//! 将棋 KK (King-King) 入力特徴量
//!
//! 旧 YaneuraOu 系評価関数 (KPPT 系) の **KK** 部分に相当する sparse 特徴量。
//! KKP / KPP と組み合わせて使う。
//!
//! ## 表現
//!
//! 1 局面につき、玉ペア (先手玉位置, 後手玉位置) が決まる。これを sparse な 1-hot
//! ベクトル (81 * 81 = 6561 次元) として与え、dual perspective で学習する。
//!
//! - **STM 視点** (`stm_index`): STM の玉位置 × NTM の玉位置 (NTM 側は反転)
//! - **NTM 視点** (`ntm_index`): NTM の玉位置 × STM の玉位置 (STM 側は反転)
//!
//! 反転は `Square::inverse()` (= 180° 回転、白を黒視点に揃える既存の慣例)。
//!
//! ## やねうら王の KPPT との関係
//!
//! やねうら王の `KK_synthesized.bin` は `weights[81][81][2]` (i32, 手番別 2 channel)
//! の形状で、玉ペアごとに 2 つの評価値を持つ。`ShogiKk` はその「玉ペア
//! インデックス化」部分に対応し、学習後の重みは `crate::value::yaneuraou_kppt`
//! の writer がやねうら王形式に変換して書き出す。

use crate::game::inputs::SparseInputType;
use crate::shogi::types::Color;
use crate::shogi::{PackedSfenValue, ShogiBoard};

/// KK 特徴量の総次元数
pub const KK_INPUTS: usize = 81 * 81; // 6561

/// 1 局面あたりの最大 active 特徴数 (KK は玉ペア 1 つしか立たない)
pub const KK_MAX_ACTIVE: usize = 1;

#[derive(Debug, Clone, Copy, Default)]
pub struct ShogiKk;

impl SparseInputType for ShogiKk {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        KK_INPUTS
    }

    fn max_active(&self) -> usize {
        KK_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        let bk = board.black_king_sq.index() as usize;
        let wk_inv = board.white_king_sq.inverse().index() as usize;

        // STM = Black の場合:
        //   STM 視点: 自分=Black、相手=White (反転)
        //     idx = bk * 81 + inv(wk)
        //   NTM 視点: 自分=White (反転)、相手=Black
        //     idx = inv(wk) * 81 + bk
        // STM = White の場合: 上の Black/White を入れ替え
        let (stm_idx, ntm_idx) = match board.side_to_move {
            Color::Black => (bk * 81 + wk_inv, wk_inv * 81 + bk),
            Color::White => (wk_inv * 81 + bk, bk * 81 + wk_inv),
        };

        f(stm_idx, ntm_idx);
    }

    fn shorthand(&self) -> String {
        "ShogiKk".to_string()
    }

    fn description(&self) -> String {
        format!("Shogi KK input ({KK_INPUTS} dims, 1-hot per position; for KPPT-style training)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kk_dims_are_sane() {
        assert_eq!(KK_INPUTS, 6561);
        assert_eq!(KK_MAX_ACTIVE, 1);
    }

    #[test]
    fn map_features_returns_one_pair() {
        // 平手の開始局面: 先手玉 5九 (file=4, rank=8 = index 4*9+8 = 44),
        //                 後手玉 5一 (file=4, rank=0 = index 4*9+0 = 36)
        // STM = Black なので stm_idx = 44 * 81 + inv(36) = 44*81 + (80-36) = 3564 + 44 = 3608
        // ntm_idx = (80-36) * 81 + 44 = 44 * 81 + 44 = 3608 (この対称ケースだけ偶然一致)
        // ↑ 反転が同じ値になる玉対称ケースを使ってもチェックにならないので、
        // ここでは callback の呼び出し回数だけ検証する。
        use crate::game::inputs::SparseInputType;
        let pos = PackedSfenValue::default(); // 平手で動かない初期値。本物の hcp は別途用意。
        let mut count = 0;
        ShogiKk.map_features(&pos, |stm, ntm| {
            // 範囲チェック
            assert!(stm < KK_INPUTS, "stm out of range: {stm}");
            assert!(ntm < KK_INPUTS, "ntm out of range: {ntm}");
            count += 1;
        });
        assert_eq!(count, 1, "KK should emit exactly one feature per position");
    }
}
