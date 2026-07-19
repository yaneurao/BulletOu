//! ShogiHalfKP - 将棋用 HalfKP 特徴量
//!
//! King + Piece 特徴量（王は特徴量に含めない）。
//! nnue-pytorch の HalfKP 実装に準拠。
//!
//! - キングバケット: 81バケット (全マス)
//! - 入力次元: 125,388 (81 × 1548)
//! - 最大アクティブ特徴: 38 (王2枚を除く)

use super::{Factorises, SparseInputType};
#[cfg(test)]
use crate::shogi::ShogiBoard;
use crate::shogi::{
    BonaPiece, PackedSfenValue,
    bona_piece::FE_OLD_END,
    packed_sfen::{BitStream, decode_board_piece, decode_hand_piece},
    types::{Color, HAND_PIECE_TYPES, PieceType, Square},
};

// =============================================================================
// 定数
// =============================================================================

/// nnue-pytorch互換の特徴量hash値 (HalfKP)
pub const FEATURE_HASH: u32 = 0x5D69D5B8;

/// キングバケット数 (全81マス)
pub const NUM_KING_SQ: usize = 81;

/// 駒入力数 (fe_end = 1548、王を除く)
pub const FE_END: usize = FE_OLD_END; // 1548

/// HalfKP piece-input virtual rows used by the FT factorizer.
pub const HALFKP_PIECE_INPUTS: usize = FE_END;

/// HalfKP の総入力次元
pub const HALFKP_DIMENSIONS: usize = NUM_KING_SQ * FE_END; // 125,388

/// 最大アクティブ特徴数 (王2枚を除く38駒)
pub const MAX_ACTIVE_FEATURES: usize = 38;

// =============================================================================
// ShogiHalfKP 特徴量型
// =============================================================================

/// ShogiHalfKP 特徴量
///
/// nnue-pytorch / YaneuraOu 互換の HalfKP 特徴量。
/// 王は特徴量に含めない（HalfKA_hm との違い）。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKP;

impl SparseInputType for ShogiHalfKP {
    /// 学習データの型: PackedSfenValue (40バイト)
    type RequiredDataType = PackedSfenValue;

    /// 特徴量の総次元数: 81 × 1548 = 125,388
    fn num_inputs(&self) -> usize {
        HALFKP_DIMENSIONS
    }

    /// 同時にアクティブになる最大特徴数: 38
    ///
    /// 将棋の合法局面では王を除く駒は最大38個:
    /// - 盤上駒（王除く）+ 手駒 = 38
    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    /// 特徴量インデックスを列挙
    ///
    /// PackedSfenValue をデコードして ShogiBoard を作成し、
    /// 各駒について (stm_index, nstm_index) を生成。
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        map_halfkp_features_from_packed(pos, f);
    }

    /// 短縮名
    fn shorthand(&self) -> String {
        "shogi-125388x81".to_string()
    }

    /// 説明
    fn description(&self) -> String {
        "Shogi HalfKP: 81 king squares, 1548 piece inputs (no kings in features)".to_string()
    }
}

/// Piece-input factorizer for [`ShogiHalfKP`].
///
/// A normal HalfKP feature is laid out as `king_square * FE_END + bona_piece`.
/// The factorized virtual row keeps only the `bona_piece` part, matching tatara's
/// default Simple FT factorizer in `Base` mode.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKPPieceFactorizer;

impl SparseInputType for ShogiHalfKPPieceFactorizer {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKP_PIECE_INPUTS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        ShogiHalfKP.map_features(pos, |stm, nstm| {
            f(stm % HALFKP_PIECE_INPUTS, nstm % HALFKP_PIECE_INPUTS);
        });
    }

    fn shorthand(&self) -> String {
        "shogi-halfkp-piece-factorizer".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKP piece-input factorizer".to_string()
    }
}

impl Factorises<ShogiHalfKP> for ShogiHalfKPPieceFactorizer {
    fn derive_feature(&self, _input: &ShogiHalfKP, feat: usize) -> Option<usize> {
        (feat < HALFKP_DIMENSIONS).then_some(feat % HALFKP_PIECE_INPUTS)
    }
}

// =============================================================================
// HalfKP 特徴量計算
// =============================================================================

/// HalfKP 特徴量インデックスを列挙
///
/// stm (side-to-move) 視点と nstm (not-side-to-move) 視点の両方を返す。
/// 片玉・詰将棋データ（玉位置が SQ_NB=81）の場合は何もしない。
/// Fill fixed-size STM/NSTM HalfKP sparse feature buffers for one packed shogi
/// position, returning the number of features written on each perspective.
///
/// Unused slots are intentionally left untouched; callers that pass reused
/// buffers must clear the tail themselves. Fresh teacher batches allocate the
/// whole sparse buffer with `-1`, so they can skip per-position tail writes.
pub fn fill_halfkp_feature_indices(pos: &PackedSfenValue, stm: &mut [i32], nstm: &mut [i32]) -> (usize, usize) {
    debug_assert!(stm.len() >= MAX_ACTIVE_FEATURES);
    debug_assert!(nstm.len() >= MAX_ACTIVE_FEATURES);
    let mut stm_count = 0usize;
    let mut nstm_count = 0usize;
    map_halfkp_features_from_packed(pos, |stm_idx, nstm_idx| {
        debug_assert!(stm_count < stm.len());
        debug_assert!(nstm_count < nstm.len());
        stm[stm_count] = stm_idx as i32;
        nstm[nstm_count] = nstm_idx as i32;
        stm_count += 1;
        nstm_count += 1;
    });
    debug_assert!(stm_count <= MAX_ACTIVE_FEATURES);
    debug_assert!(nstm_count <= MAX_ACTIVE_FEATURES);
    (stm_count, nstm_count)
}

fn map_halfkp_features_from_packed<F: FnMut(usize, usize)>(pos: &PackedSfenValue, mut f: F) {
    let mut stream = BitStream::new(&pos.sfen().data);

    let stm = if stream.read_bit() { Color::White } else { Color::Black };
    let nstm = stm.opponent();

    let black_king_sq = Square(stream.read_bits(7) as u8);
    let white_king_sq = Square(stream.read_bits(7) as u8);

    let stm_king_sq = match stm {
        Color::Black => black_king_sq,
        Color::White => white_king_sq,
    };
    let nstm_king_sq = match nstm {
        Color::Black => black_king_sq,
        Color::White => white_king_sq,
    };
    if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
        return;
    }

    let stm_ksq = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
    let nstm_ksq = if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };

    for sq_idx in 0..81u8 {
        if sq_idx == black_king_sq.0 || sq_idx == white_king_sq.0 {
            continue;
        }

        let piece = decode_board_piece(&mut stream);
        if piece.is_none() || piece.piece_type == PieceType::King {
            continue;
        }

        let sq = Square(sq_idx);
        let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
        if stm_bp == BonaPiece::ZERO {
            continue;
        }
        let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
        f(halfkp_index(stm_ksq, stm_bp.value() as usize), halfkp_index(nstm_ksq, nstm_bp.value() as usize));
    }

    let mut hand_counts = [[0u8; HAND_PIECE_TYPES.len()]; 2];
    while stream.cursor() < 256 {
        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        if is_piecebox || piece.is_none() {
            continue;
        }
        if let Some(piece_index) = hand_piece_index(piece.piece_type) {
            let owner_index = color_index(piece.color);
            hand_counts[owner_index][piece_index] = hand_counts[owner_index][piece_index].saturating_add(1);
        }
    }

    for owner in [Color::Black, Color::White] {
        for (piece_index, &pt) in HAND_PIECE_TYPES.iter().enumerate() {
            let count = hand_counts[color_index(owner)][piece_index];
            for i in 1..=count {
                let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                f(halfkp_index(stm_ksq, stm_bp.value() as usize), halfkp_index(nstm_ksq, nstm_bp.value() as usize));
            }
        }
    }
}

#[cfg(test)]
fn map_halfkp_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
    // STM と NSTM の視点
    let stm = board.side_to_move;
    let nstm = stm.opponent();

    // 玉位置の妥当性チェック（SQ_NB=81 は「玉なし」を意味する）
    let stm_king_sq = board.king_square(stm);
    let nstm_king_sq = board.king_square(nstm);
    if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
        // 片玉/詰将棋データはスキップ
        return;
    }

    // 視点に応じた玉位置（後手視点では反転）
    let stm_ksq = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };

    let nstm_ksq = if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };

    // 盤上の駒（王以外）。従来は駒種×色ごとに81マスを何度も走査していたが、
    // HalfKP では各非玉駒を1回だけ列挙すれば十分。
    for (sq_idx, &piece) in board.board.iter().enumerate() {
        if piece.is_none() || piece.piece_type == PieceType::King {
            continue;
        }
        let sq = Square::from_index(sq_idx);

        // STM 視点での BonaPiece
        let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
        if stm_bp == BonaPiece::ZERO {
            continue;
        }
        let stm_idx = halfkp_index(stm_ksq, stm_bp.value() as usize);

        // NSTM 視点での BonaPiece
        let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
        let nstm_idx = halfkp_index(nstm_ksq, nstm_bp.value() as usize);

        f(stm_idx, nstm_idx);
    }

    // 注意: HalfKP では王は特徴量に含めない（HalfKA_hm との違い）

    // 手駒の特徴量
    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            let count = board.hand(owner).count(pt);
            if count == 0 {
                continue;
            }

            // 各枚数分の特徴量を追加
            for i in 1..=count {
                // STM 視点
                let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let stm_idx = halfkp_index(stm_ksq, stm_bp.value() as usize);

                // NSTM 視点
                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                let nstm_idx = halfkp_index(nstm_ksq, nstm_bp.value() as usize);

                f(stm_idx, nstm_idx);
            }
        }
    }
}

/// HalfKP の特徴インデックスを計算
///
/// feature_index = king_sq * FE_END + bonapiece
#[inline]
fn color_index(color: Color) -> usize {
    match color {
        Color::Black => 0,
        Color::White => 1,
    }
}

#[inline]
fn hand_piece_index(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Lance => Some(1),
        PieceType::Knight => Some(2),
        PieceType::Silver => Some(3),
        PieceType::Gold => Some(4),
        PieceType::Bishop => Some(5),
        PieceType::Rook => Some(6),
        _ => None,
    }
}

#[inline]
fn halfkp_index(king_sq: usize, bonapiece: usize) -> usize {
    king_sq * FE_END + bonapiece
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::game::inputs::Factorised;
    use crate::shogi::{Piece, PieceType, Square};

    use super::*;

    #[test]
    fn test_dimensions() {
        let input = ShogiHalfKP;
        assert_eq!(input.num_inputs(), 125_388);
        assert_eq!(input.max_active(), 38);
    }

    #[test]
    fn test_shorthand() {
        let input = ShogiHalfKP;
        assert_eq!(input.shorthand(), "shogi-125388x81");
    }

    #[test]
    fn test_halfkp_index() {
        // king_sq=0, bp=0 → 0
        assert_eq!(halfkp_index(0, 0), 0);

        // king_sq=1, bp=0 → 1548
        assert_eq!(halfkp_index(1, 0), FE_END);

        // king_sq=80, bp=1547 → 80*1548 + 1547 = 125,387
        assert_eq!(halfkp_index(80, 1547), 125_387);
    }

    #[test]
    fn test_piece_factorizer_dimensions_and_derivation() {
        let factorizer = ShogiHalfKPPieceFactorizer;

        assert_eq!(factorizer.num_inputs(), HALFKP_PIECE_INPUTS);
        assert_eq!(factorizer.max_active(), MAX_ACTIVE_FEATURES);
        assert_eq!(factorizer.derive_feature(&ShogiHalfKP, halfkp_index(17, 123)), Some(123));
        assert_eq!(factorizer.derive_feature(&ShogiHalfKP, HALFKP_DIMENSIONS), None);
    }

    #[test]
    fn test_piece_factorizer_merge_folds_virtual_rows() {
        let input = Factorised::from_parts(ShogiHalfKP, ShogiHalfKPPieceFactorizer);
        let layer_size = 2;
        let mut unmerged = vec![0.0; input.num_inputs() * layer_size];

        let piece = 10;
        let normal_feat = halfkp_index(3, piece);
        let normal_start = (HALFKP_PIECE_INPUTS + normal_feat) * layer_size;
        unmerged[normal_start] = 1.25;
        unmerged[normal_start + 1] = -2.5;

        let factor_start = piece * layer_size;
        unmerged[factor_start] = 0.75;
        unmerged[factor_start + 1] = 0.5;

        let merged = input.merge_factoriser(unmerged);
        let merged_start = normal_feat * layer_size;
        assert_eq!(merged[merged_start], 2.0);
        assert_eq!(merged[merged_start + 1], -2.0);

        let same_piece_other_king = halfkp_index(5, piece);
        let other_start = same_piece_other_king * layer_size;
        assert_eq!(merged[other_start], 0.75);
        assert_eq!(merged[other_start + 1], 0.5);
    }

    #[derive(Default)]
    struct TestPackedSfenWriter {
        bytes: [u8; 32],
        cursor: usize,
    }

    impl TestPackedSfenWriter {
        fn write_bit(&mut self, bit: bool) {
            assert!(self.cursor < 256);
            if bit {
                self.bytes[self.cursor / 8] |= 1 << (self.cursor % 8);
            }
            self.cursor += 1;
        }

        fn write_bits(&mut self, value: u32, len: u8) {
            for i in 0..len {
                self.write_bit(((value >> i) & 1) != 0);
            }
        }

        fn write_board_piece(&mut self, piece: Piece) {
            if piece.is_none() {
                self.write_bit(false);
                return;
            }

            let (pattern, len, promoted) = board_piece_code(piece.piece_type);
            self.write_bits(pattern, len);
            if piece.piece_type.unpromote() != PieceType::Gold {
                self.write_bit(promoted);
            }
            self.write_bit(piece.color == Color::White);
        }

        fn write_hand_piece(&mut self, piece: Piece) {
            let (pattern, len, _) = board_piece_code(piece.piece_type);
            self.write_bits(pattern >> 1, len - 1);
            if piece.piece_type != PieceType::Gold {
                self.write_bit(false);
            }
            self.write_bit(piece.color == Color::White);
        }

        fn write_piecebox_pawn(&mut self) {
            self.write_bit(false);
            self.write_bit(true);
            self.write_bit(false);
        }

        fn write_piecebox_lance_prefix(&mut self, bits: usize) {
            // Lance hand code is 001 (LSB-first value 1, len 3), followed by the
            // piecebox bit. The final colour bit may be omitted at the 256-bit
            // boundary because the decoder treats OOB reads as false.
            let seq = [true, false, false, true, false];
            for &bit in seq.iter().take(bits) {
                self.write_bit(bit);
            }
        }

        fn finish_with_piecebox_padding(&mut self) {
            while self.cursor < 256 {
                let remaining = 256 - self.cursor;
                match remaining {
                    1 => self.write_bit(true),
                    2 => {
                        self.write_bit(false);
                        self.write_bit(true);
                    }
                    4 => self.write_piecebox_lance_prefix(4),
                    5 => self.write_piecebox_lance_prefix(5),
                    _ => self.write_piecebox_pawn(),
                }
            }
        }
    }

    fn board_piece_code(pt: PieceType) -> (u32, u8, bool) {
        match pt {
            PieceType::Pawn => (0x01, 2, false),
            PieceType::Lance => (0x03, 4, false),
            PieceType::Knight => (0x0b, 4, false),
            PieceType::Silver => (0x07, 4, false),
            PieceType::Bishop => (0x1f, 6, false),
            PieceType::Rook => (0x3f, 6, false),
            PieceType::Gold => (0x0f, 5, false),
            PieceType::ProPawn => (0x01, 2, true),
            PieceType::ProLance => (0x03, 4, true),
            PieceType::ProKnight => (0x0b, 4, true),
            PieceType::ProSilver => (0x07, 4, true),
            PieceType::Horse => (0x1f, 6, true),
            PieceType::Dragon => (0x3f, 6, true),
            _ => panic!("unsupported test piece type: {pt:?}"),
        }
    }

    fn packed_halfkp_test_position(side_to_move: Color) -> PackedSfenValue {
        let black_king = Square::new(4, 8);
        let white_king = Square::new(4, 0);
        let mut board = [Piece::NONE; 81];
        board[Square::new(0, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
        board[Square::new(8, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        board[Square::new(2, 4).index()] = Piece::new(Color::Black, PieceType::ProSilver);
        board[Square::new(6, 4).index()] = Piece::new(Color::White, PieceType::Dragon);

        let mut writer = TestPackedSfenWriter::default();
        writer.write_bit(side_to_move == Color::White);
        writer.write_bits(black_king.0 as u32, 7);
        writer.write_bits(white_king.0 as u32, 7);
        for sq_idx in 0..81u8 {
            if sq_idx == black_king.0 || sq_idx == white_king.0 {
                continue;
            }
            writer.write_board_piece(board[sq_idx as usize]);
        }
        writer.write_hand_piece(Piece::new(Color::Black, PieceType::Pawn));
        writer.write_hand_piece(Piece::new(Color::Black, PieceType::Pawn));
        writer.write_hand_piece(Piece::new(Color::White, PieceType::Rook));
        writer.write_hand_piece(Piece::new(Color::White, PieceType::Gold));
        writer.finish_with_piecebox_padding();

        let mut pos = PackedSfenValue::default();
        pos.as_bytes_mut()[..32].copy_from_slice(&writer.bytes);
        pos
    }

    #[test]
    fn test_direct_packed_mapper_matches_board_mapper() {
        for side_to_move in [Color::Black, Color::White] {
            let pos = packed_halfkp_test_position(side_to_move);
            let board = ShogiBoard::from_packed_sfen(&pos);

            let mut board_features = Vec::new();
            map_halfkp_features(&board, |stm, nstm| board_features.push((stm, nstm)));

            let mut direct_features = Vec::new();
            map_halfkp_features_from_packed(&pos, |stm, nstm| direct_features.push((stm, nstm)));
            assert_eq!(direct_features, board_features);

            let mut stm = [-1; MAX_ACTIVE_FEATURES];
            let mut nstm = [-1; MAX_ACTIVE_FEATURES];
            let (stm_count, nstm_count) = fill_halfkp_feature_indices(&pos, &mut stm, &mut nstm);
            assert_eq!(stm_count, board_features.len());
            assert_eq!(nstm_count, board_features.len());
            let filled_features: Vec<_> = (0..stm_count).map(|i| (stm[i] as usize, nstm[i] as usize)).collect();
            assert_eq!(filled_features, board_features);
        }
    }

    #[test]
    fn test_map_features_count() {
        // ダミーの局面を作成（手動で設定）
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::new(4, 0), // 5一
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        // 歩を9枚ずつ配置
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        let mut count = 0;
        map_halfkp_features(&board, |_, _| count += 1);

        // 歩18枚（王は含まない）= 18
        assert_eq!(count, 18);
    }

    #[test]
    fn test_map_features_no_kings() {
        // HalfKP では王は特徴量に含めないことを確認
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::new(4, 0), // 5一
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        let mut count = 0;
        map_halfkp_features(&board, |_, _| count += 1);

        // 王は特徴量に含めないので 0
        assert_eq!(count, 0);
    }

    #[test]
    fn test_map_features_sq_nb_guard() {
        // 片玉データ（玉位置が SQ_NB=81）のテスト
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::NONE,      // SQ_NB (81)
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);

        let mut count = 0;
        map_halfkp_features(&board, |_, _| count += 1);

        // 片玉データはスキップされるため、カウントは 0
        assert_eq!(count, 0);
    }

    #[test]
    fn test_feature_indices_in_range() {
        // 特徴インデックスが範囲内であることを確認
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::new(4, 0), // 5一
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        // 歩を配置
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        let max_valid_index = HALFKP_DIMENSIONS - 1;
        map_halfkp_features(&board, |stm_idx, nstm_idx| {
            assert!(stm_idx <= max_valid_index, "STM index {} exceeds max {}", stm_idx, max_valid_index);
            assert!(nstm_idx <= max_valid_index, "NSTM index {} exceeds max {}", nstm_idx, max_valid_index);
        });
    }
}
