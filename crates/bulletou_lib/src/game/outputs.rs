use std::{path::Path, sync::OnceLock};

use bulletformat::{ChessBoard, chess::MarlinFormat};

use crate::shogi::{
    BonaPiece, Color, Hand, PackedSfenValue, Piece, PieceType, ShogiBoard,
    bona_piece::{
        E_DRAGON, E_HAND_BISHOP, E_HAND_GOLD, E_HAND_KNIGHT, E_HAND_LANCE, E_HAND_PAWN, E_HAND_ROOK, E_HAND_SILVER,
        E_HORSE, F_DRAGON, F_HAND_BISHOP, F_HAND_GOLD, F_HAND_KNIGHT, F_HAND_LANCE, F_HAND_PAWN, F_HAND_ROOK,
        F_HAND_SILVER, F_HORSE, FE_OLD_END,
    },
    types::{BOARD_PIECE_TYPES, HAND_PIECE_TYPES, Square},
};

pub trait OutputBuckets<T>: Send + Sync + Copy + Default + 'static {
    const BUCKETS: usize;

    fn bucket(&self, pos: &T) -> usize;
}

#[deprecated(note = "You do not need to specify this anymore, it is the default!")]
#[derive(Clone, Copy, Default)]
pub struct Single;

#[allow(deprecated)]
impl<T: 'static> OutputBuckets<T> for Single {
    const BUCKETS: usize = 1;

    fn bucket(&self, _: &T) -> usize {
        0
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCount<const N: usize>;
impl<const N: usize> OutputBuckets<ChessBoard> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ChessBoard) -> usize {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as usize - 2) / divisor
    }
}

impl<const N: usize> OutputBuckets<MarlinFormat> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &MarlinFormat) -> usize {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as usize - 2) / divisor
    }
}

/// 将棋 LayerStacks 用出力バケット
///
/// 両玉の相対段に基づいて N バケットに分類する。
/// rshogi の `compute_bucket_index` / `compute_king_ranks` と同一ロジック。
///
/// 標準では N=9 (3×3 マトリクス):
/// ```text
///        e_rank 0-2  e_rank 3-5  e_rank 6-8
/// f_rank 0-2:    0        1            2
/// f_rank 3-5:    3        4            5
/// f_rank 6-8:    6        7            8
/// ```
#[derive(Clone, Copy, Default)]
pub struct ShogiKingRankBucket<const N: usize>;

impl<const N: usize> OutputBuckets<PackedSfenValue> for ShogiKingRankBucket<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        let board = pos.decode();

        let side_to_move = board.side_to_move;
        let f_king = board.king_square(side_to_move);
        let e_king = board.king_square(side_to_move.opponent());

        // 味方玉の段（味方から見た相対段）
        let f_rank = match side_to_move {
            crate::shogi::Color::Black => f_king.rank() as usize,
            crate::shogi::Color::White => 8 - f_king.rank() as usize,
        };

        // 相手玉の段（相手から見た相対段）
        let e_rank = match side_to_move {
            crate::shogi::Color::Black => 8 - e_king.rank() as usize,
            crate::shogi::Color::White => e_king.rank() as usize,
        };

        const F_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
        const E_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

        match N {
            81 => f_rank.min(8) * 9 + e_rank.min(8),
            9 => F_TO_INDEX[f_rank.min(8)] + E_TO_INDEX[e_rank.min(8)],
            _ => (F_TO_INDEX[f_rank.min(8)] + E_TO_INDEX[e_rank.min(8)]).min(N.saturating_sub(1)),
        }
    }
}

#[inline]
fn shogi_stm_ntm_hands_from_board(board: &ShogiBoard) -> (Hand, Hand) {
    let stm = board.side_to_move;
    let stm_hand = if stm == Color::Black { board.black_hand } else { board.white_hand };
    let ntm_hand = if stm == Color::Black { board.white_hand } else { board.black_hand };
    (stm_hand, ntm_hand)
}

#[inline]
fn shogi_perspective_king_squares_from_board(board: &ShogiBoard) -> (Square, Square) {
    let stm = board.side_to_move;
    let f_king = board.king_square(stm);
    let e_king = board.king_square(stm.opponent());
    let f_sq = if stm == Color::Black { f_king } else { f_king.inverse() };
    let e_sq = if stm == Color::Black { e_king.inverse() } else { e_king };
    (f_sq, e_sq)
}

#[inline]
fn shogi_king_rank_bucket_from_board<const N: usize>(board: &ShogiBoard) -> usize {
    let (f_sq, e_sq) = shogi_perspective_king_squares_from_board(board);
    let f_rank = f_sq.rank() as usize;
    let e_rank = e_sq.rank() as usize;

    const F_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
    const E_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    match N {
        81 => f_rank.min(8) * 9 + e_rank.min(8),
        9 => F_TO_INDEX[f_rank.min(8)] + E_TO_INDEX[e_rank.min(8)],
        _ => (F_TO_INDEX[f_rank.min(8)] + E_TO_INDEX[e_rank.min(8)]).min(N.saturating_sub(1)),
    }
}

#[inline]
fn shogi_file3_bucket(file: u8) -> usize {
    usize::from(file.min(8)) / 3
}

/// SFNN `k9k9z` bucket for one king square after perspective normalization.
///
/// This formula must match the engine-side `king9_zone_single_bucket()`:
/// ranks 1-3 -> bucket 0, ranks 4-6 -> bucket 1, rank 7 -> bucket 2,
/// rank 8 keeps file/3 detail in buckets 3..=5, and rank 9 keeps file/3
/// detail in buckets 6..=8.
#[inline]
pub fn shogi_king9_zone_single_bucket(sq: Square) -> usize {
    let rank = sq.rank().min(8) as usize;
    if rank < 3 {
        0
    } else if rank < 6 {
        1
    } else if rank == 6 {
        2
    } else {
        3 + (rank - 7) * 3 + shogi_file3_bucket(sq.file())
    }
}

/// YaneuraOu SFNN `k9k9z` LayerStack bucket.
///
/// YaneuraOu normalizes each king square to the side's perspective, then
/// computes `stm_king9z * 9 + non_stm_king9z`.
#[derive(Clone, Copy, Default)]
pub struct ShogiKing9ZoneByKing9ZoneBucket;

impl ShogiKing9ZoneByKing9ZoneBucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let f_king = board.king_square(stm);
        let e_king = board.king_square(stm.opponent());
        let f_sq = if stm == Color::Black { f_king } else { f_king.inverse() };
        let e_sq = if stm == Color::Black { e_king.inverse() } else { e_king };
        shogi_king9_zone_single_bucket(f_sq) * 9 + shogi_king9_zone_single_bucket(e_sq)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiKing9ZoneByKing9ZoneBucket {
    const BUCKETS: usize = 9 * 9;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// SFNN `k13k13z` bucket for one king square after perspective normalization.
///
/// This formula must match the engine-side `king13_zone_single_bucket()`:
/// ranks 1-7 are kept as seven rank buckets, rank 8 keeps file/3 detail in
/// buckets 7..=9, and rank 9 keeps file/3 detail in buckets 10..=12.
#[inline]
pub fn shogi_king13_zone_single_bucket(sq: Square) -> usize {
    let rank = sq.rank().min(8) as usize;
    if rank < 7 { rank } else { 7 + (rank - 7) * 3 + shogi_file3_bucket(sq.file()) }
}

/// YaneuraOu SFNN `k13k13z` LayerStack bucket.
///
/// YaneuraOu normalizes each king square to the side's perspective, then
/// computes `stm_king13z * 13 + non_stm_king13z`.
#[derive(Clone, Copy, Default)]
pub struct ShogiKing13ZoneByKing13ZoneBucket;

impl ShogiKing13ZoneByKing13ZoneBucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let f_king = board.king_square(stm);
        let e_king = board.king_square(stm.opponent());
        let f_sq = if stm == Color::Black { f_king } else { f_king.inverse() };
        let e_sq = if stm == Color::Black { e_king.inverse() } else { e_king };
        shogi_king13_zone_single_bucket(f_sq) * 13 + shogi_king13_zone_single_bucket(e_sq)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiKing13ZoneByKing13ZoneBucket {
    const BUCKETS: usize = 13 * 13;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// SFNN `k21k21` bucket for one king square after perspective normalization.
///
/// This follows the same family as `king29_single_bucket()`, but keeps only
/// the two deepest home-rank rows at full file resolution:
/// ranks 1-3 -> bucket 0, ranks 4-6 -> bucket 1, rank 7 -> bucket 2,
/// ranks 8-9 keep full 2x9 square detail and map to buckets 3..=20.
#[inline]
pub fn shogi_king21_single_bucket(sq: Square) -> usize {
    let rank = sq.rank().min(8) as usize;
    let file = sq.file().min(8) as usize;
    if rank < 3 {
        0
    } else if rank < 6 {
        1
    } else if rank < 7 {
        2
    } else {
        3 + (rank - 7) * 9 + file
    }
}

/// YaneuraOu SFNN `k21k21` LayerStack bucket.
///
/// YaneuraOu normalizes each king square to the side's perspective, then
/// computes `stm_king21 * 21 + non_stm_king21`.
#[derive(Clone, Copy, Default)]
pub struct ShogiKing21ByKing21Bucket;

impl ShogiKing21ByKing21Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let f_king = board.king_square(stm);
        let e_king = board.king_square(stm.opponent());
        let f_sq = if stm == Color::Black { f_king } else { f_king.inverse() };
        let e_sq = if stm == Color::Black { e_king.inverse() } else { e_king };
        shogi_king21_single_bucket(f_sq) * 21 + shogi_king21_single_bucket(e_sq)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiKing21ByKing21Bucket {
    const BUCKETS: usize = 21 * 21;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// SFNN `k29k29` bucket for one king square after perspective normalization.
///
/// This formula must match the engine-side `king29_single_bucket()`:
/// ranks 1-3 -> bucket 0, ranks 4-6 -> bucket 1, ranks 7-9 keep full 3x9
/// square detail and map to buckets 2..=28.
#[inline]
pub fn shogi_king29_single_bucket(sq: Square) -> usize {
    let rank = sq.rank().min(8) as usize;
    let file = sq.file().min(8) as usize;
    if rank < 3 {
        0
    } else if rank < 6 {
        1
    } else {
        2 + (rank - 6) * 9 + file
    }
}

/// YaneuraOu SFNN `k29k29` LayerStack bucket.
///
/// YaneuraOu normalizes each king square to the side's perspective, then
/// computes `stm_king29 * 29 + non_stm_king29`.
#[derive(Clone, Copy, Default)]
pub struct ShogiKing29ByKing29Bucket;

impl ShogiKing29ByKing29Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let f_king = board.king_square(stm);
        let e_king = board.king_square(stm.opponent());
        let f_sq = if stm == Color::Black { f_king } else { f_king.inverse() };
        let e_sq = if stm == Color::Black { e_king.inverse() } else { e_king };
        shogi_king29_single_bucket(f_sq) * 29 + shogi_king29_single_bucket(e_sq)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiKing29ByKing29Bucket {
    const BUCKETS: usize = 29 * 29;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// SFNN `hand4` bucket for one side's hand.
///
/// This formula must match the engine-side `hand4_single_bucket()`:
/// bit0 = bishop exists.
#[inline]
pub fn shogi_hand4_single_bucket(hand: Hand) -> usize {
    usize::from(hand.bishop() > 0)
}

/// SFNN `hand16` bucket for one side's hand.
///
/// This formula must match the engine-side `hand16_single_bucket()`:
/// bit0 = pawn exists, bit1 = bishop exists.
#[inline]
pub fn shogi_hand16_single_bucket(hand: Hand) -> usize {
    let mut bucket = 0usize;
    if hand.pawn() > 0 {
        bucket |= 1;
    }
    if hand.bishop() > 0 {
        bucket |= 2;
    }
    bucket
}

/// SFNN `hand64` bucket for one side's hand.
///
/// This formula must match the engine-side `hand64_single_bucket()`:
/// bit0 = pawn/lance/knight exists, bit1 = gold/silver/rook exists,
/// bit2 = bishop exists.
#[inline]
pub fn shogi_hand64_single_bucket(hand: Hand) -> usize {
    let mut bucket = 0usize;
    if hand.pawn() + hand.lance() + hand.knight() > 0 {
        bucket |= 1;
    }
    if hand.gold() + hand.silver() + hand.rook() > 0 {
        bucket |= 2;
    }
    if hand.bishop() > 0 {
        bucket |= 4;
    }
    bucket
}

/// SFNN `hand64z` bucket for one side's hand.
///
/// This score-zone bucket is separate from the `hand64` hand-presence bucket.
#[inline]
pub fn shogi_hand64z_single_bucket(hand: Hand) -> usize {
    let score = usize::from(hand.pawn())
        + usize::from(hand.lance() + hand.knight()) * 2
        + usize::from(hand.silver() + hand.gold()) * 3
        + usize::from(hand.bishop() + hand.rook()) * 5;
    ((score + 3) / 4).min(7)
}

/// SFNN `hand256` bucket for one side's hand.
///
/// This formula must match the engine-side `hand256_single_bucket()`:
/// bit0 = pawn/lance/knight exists, bit1 = silver/gold exists,
/// bit2 = bishop exists, bit3 = rook exists.
#[inline]
pub fn shogi_hand256_single_bucket(hand: Hand) -> usize {
    let mut bucket = 0usize;
    if hand.pawn() + hand.lance() + hand.knight() > 0 {
        bucket |= 1;
    }
    if hand.silver() + hand.gold() > 0 {
        bucket |= 2;
    }
    if hand.bishop() > 0 {
        bucket |= 4;
    }
    if hand.rook() > 0 {
        bucket |= 8;
    }
    bucket
}

/// SFNN `hand1024` bucket for one side's hand.
///
/// This formula must match the engine-side `hand1024_single_bucket()`:
/// bit0 = pawn exists, bit1 = lance/knight exists, bit2 = silver/gold exists,
/// bit3 = bishop exists, bit4 = rook exists.
#[inline]
pub fn shogi_hand1024_single_bucket(hand: Hand) -> usize {
    let mut bucket = 0usize;
    if hand.pawn() > 0 {
        bucket |= 1;
    }
    if hand.lance() + hand.knight() > 0 {
        bucket |= 2;
    }
    if hand.silver() + hand.gold() > 0 {
        bucket |= 4;
    }
    if hand.bishop() > 0 {
        bucket |= 8;
    }
    if hand.rook() > 0 {
        bucket |= 16;
    }
    bucket
}

/// YaneuraOu SFNN `hand4` LayerStack bucket.
///
/// Bucket index is `stm_hand_bucket * 2 + non_stm_hand_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand4Bucket;

impl ShogiHand4Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_hand = if stm == Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == Color::Black { board.white_hand } else { board.black_hand };
        shogi_hand4_single_bucket(stm_hand) * 2 + shogi_hand4_single_bucket(ntm_hand)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand4Bucket {
    const BUCKETS: usize = 4;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand16` LayerStack bucket.
///
/// Bucket index is `stm_hand_bucket * 4 + non_stm_hand_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand16Bucket;

impl ShogiHand16Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_hand = if stm == Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == Color::Black { board.white_hand } else { board.black_hand };
        shogi_hand16_single_bucket(stm_hand) * 4 + shogi_hand16_single_bucket(ntm_hand)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand16Bucket {
    const BUCKETS: usize = 16;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand64` LayerStack bucket.
///
/// Bucket index is `stm_hand_bucket * 8 + non_stm_hand_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand64Bucket;

impl ShogiHand64Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_hand = if stm == Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == Color::Black { board.white_hand } else { board.black_hand };
        shogi_hand64_single_bucket(stm_hand) * 8 + shogi_hand64_single_bucket(ntm_hand)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand64Bucket {
    const BUCKETS: usize = 64;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand64z` LayerStack bucket.
///
/// Bucket index is `stm_hand_bucket * 8 + non_stm_hand_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand64zBucket;

impl ShogiHand64zBucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_hand = if stm == Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == Color::Black { board.white_hand } else { board.black_hand };
        shogi_hand64z_single_bucket(stm_hand) * 8 + shogi_hand64z_single_bucket(ntm_hand)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand64zBucket {
    const BUCKETS: usize = 64;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand256` LayerStack bucket.
///
/// Bucket index is `stm_hand_bucket * 16 + non_stm_hand_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand256Bucket;

impl ShogiHand256Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_hand = if stm == Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == Color::Black { board.white_hand } else { board.black_hand };
        shogi_hand256_single_bucket(stm_hand) * 16 + shogi_hand256_single_bucket(ntm_hand)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand256Bucket {
    const BUCKETS: usize = 256;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand1024` LayerStack bucket.
///
/// Bucket index is `stm_hand_bucket * 32 + non_stm_hand_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand1024Bucket;

impl ShogiHand1024Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_hand = if stm == Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == Color::Black { board.white_hand } else { board.black_hand };
        shogi_hand1024_single_bucket(stm_hand) * 32 + shogi_hand1024_single_bucket(ntm_hand)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand1024Bucket {
    const BUCKETS: usize = 1024;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand64_k3k3` LayerStack bucket.
///
/// YaneuraOu computes `hand64_bucket * 9 + king3_by_king3_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand64KingRankBucket;

impl ShogiHand64KingRankBucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand64Bucket::bucket_index(pos) * 9 + ShogiKingRankBucket::<9>.bucket(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand64KingRankBucket {
    const BUCKETS: usize = 64 * 9;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand256_k3k3` LayerStack bucket.
///
/// YaneuraOu computes `hand256_bucket * 9 + king3_by_king3_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand256KingRankBucket;

impl ShogiHand256KingRankBucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand256Bucket::bucket_index(pos) * 9 + ShogiKingRankBucket::<9>.bucket(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand256KingRankBucket {
    const BUCKETS: usize = 256 * 9;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand1024_k3k3` LayerStack bucket.
///
/// YaneuraOu computes `hand1024_bucket * 9 + king3_by_king3_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand1024KingRankBucket;

impl ShogiHand1024KingRankBucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand1024Bucket::bucket_index(pos) * 9 + ShogiKingRankBucket::<9>.bucket(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand1024KingRankBucket {
    const BUCKETS: usize = 1024 * 9;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand64_k9k9` LayerStack bucket.
///
/// YaneuraOu computes `hand64_bucket * 81 + king9_by_king9_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand64KingRank81Bucket;

impl ShogiHand64KingRank81Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand64Bucket::bucket_index(pos) * 81 + ShogiKingRankBucket::<81>.bucket(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand64KingRank81Bucket {
    const BUCKETS: usize = 64 * 81;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand256_k9k9` LayerStack bucket.
///
/// YaneuraOu computes `hand256_bucket * 81 + king9_by_king9_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand256KingRank81Bucket;

impl ShogiHand256KingRank81Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand256Bucket::bucket_index(pos) * 81 + ShogiKingRankBucket::<81>.bucket(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand256KingRank81Bucket {
    const BUCKETS: usize = 256 * 81;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand1024_k9k9` LayerStack bucket.
///
/// YaneuraOu computes `hand1024_bucket * 81 + king9_by_king9_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand1024KingRank81Bucket;

impl ShogiHand1024KingRank81Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand1024Bucket::bucket_index(pos) * 81 + ShogiKingRankBucket::<81>.bucket(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand1024KingRank81Bucket {
    const BUCKETS: usize = 1024 * 81;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand64_k21k21` LayerStack bucket.
///
/// YaneuraOu computes `hand64_bucket * 441 + king21_by_king21_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand64King21ByKing21Bucket;

impl ShogiHand64King21ByKing21Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand64Bucket::bucket_index(pos) * 441 + ShogiKing21ByKing21Bucket::bucket_index(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand64King21ByKing21Bucket {
    const BUCKETS: usize = 64 * 441;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand256_k21k21` LayerStack bucket.
///
/// YaneuraOu computes `hand256_bucket * 441 + king21_by_king21_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand256King21ByKing21Bucket;

impl ShogiHand256King21ByKing21Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand256Bucket::bucket_index(pos) * 441 + ShogiKing21ByKing21Bucket::bucket_index(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand256King21ByKing21Bucket {
    const BUCKETS: usize = 256 * 441;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand1024_k21k21` LayerStack bucket.
///
/// YaneuraOu computes `hand1024_bucket * 441 + king21_by_king21_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand1024King21ByKing21Bucket;

impl ShogiHand1024King21ByKing21Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand1024Bucket::bucket_index(pos) * 441 + ShogiKing21ByKing21Bucket::bucket_index(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand1024King21ByKing21Bucket {
    const BUCKETS: usize = 1024 * 441;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand64_k29k29` LayerStack bucket.
///
/// YaneuraOu computes `hand64_bucket * 841 + king29_by_king29_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand64King29ByKing29Bucket;

impl ShogiHand64King29ByKing29Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand64Bucket::bucket_index(pos) * 841 + ShogiKing29ByKing29Bucket::bucket_index(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand64King29ByKing29Bucket {
    const BUCKETS: usize = 64 * 841;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand256_k29k29` LayerStack bucket.
///
/// YaneuraOu computes `hand256_bucket * 841 + king29_by_king29_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand256King29ByKing29Bucket;

impl ShogiHand256King29ByKing29Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand256Bucket::bucket_index(pos) * 841 + ShogiKing29ByKing29Bucket::bucket_index(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand256King29ByKing29Bucket {
    const BUCKETS: usize = 256 * 841;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// YaneuraOu SFNN `hand1024_k29k29` LayerStack bucket.
///
/// YaneuraOu computes `hand1024_bucket * 841 + king29_by_king29_bucket`.
#[derive(Clone, Copy, Default)]
pub struct ShogiHand1024King29ByKing29Bucket;

impl ShogiHand1024King29ByKing29Bucket {
    #[inline]
    pub fn bucket_index(pos: &PackedSfenValue) -> usize {
        ShogiHand1024Bucket::bucket_index(pos) * 841 + ShogiKing29ByKing29Bucket::bucket_index(pos)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiHand1024King29ByKing29Bucket {
    const BUCKETS: usize = 1024 * 841;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        Self::bucket_index(pos)
    }
}

/// Hand axis for YaneuraOu-compatible SFNN LayerStack buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShogiSfnnHandBucketKind {
    #[default]
    None,
    Hand4,
    Hand16,
    Hand64,
    Hand64z,
    Hand256,
    Hand1024,
}

impl ShogiSfnnHandBucketKind {
    pub const fn bucket_count(self) -> usize {
        match self {
            Self::None => 1,
            Self::Hand4 => 4,
            Self::Hand16 => 16,
            Self::Hand64 => 64,
            Self::Hand64z => 64,
            Self::Hand256 => 256,
            Self::Hand1024 => 1024,
        }
    }

    pub fn bucket(self, pos: &PackedSfenValue) -> usize {
        match self {
            Self::None => 0,
            Self::Hand4 => ShogiHand4Bucket::bucket_index(pos),
            Self::Hand16 => ShogiHand16Bucket::bucket_index(pos),
            Self::Hand64 => ShogiHand64Bucket::bucket_index(pos),
            Self::Hand64z => ShogiHand64zBucket::bucket_index(pos),
            Self::Hand256 => ShogiHand256Bucket::bucket_index(pos),
            Self::Hand1024 => ShogiHand1024Bucket::bucket_index(pos),
        }
    }

    pub fn bucket_from_board(self, board: &ShogiBoard) -> usize {
        let (stm_hand, ntm_hand) = shogi_stm_ntm_hands_from_board(board);
        match self {
            Self::None => 0,
            Self::Hand4 => shogi_hand4_single_bucket(stm_hand) * 2 + shogi_hand4_single_bucket(ntm_hand),
            Self::Hand16 => shogi_hand16_single_bucket(stm_hand) * 4 + shogi_hand16_single_bucket(ntm_hand),
            Self::Hand64 => shogi_hand64_single_bucket(stm_hand) * 8 + shogi_hand64_single_bucket(ntm_hand),
            Self::Hand64z => shogi_hand64z_single_bucket(stm_hand) * 8 + shogi_hand64z_single_bucket(ntm_hand),
            Self::Hand256 => shogi_hand256_single_bucket(stm_hand) * 16 + shogi_hand256_single_bucket(ntm_hand),
            Self::Hand1024 => shogi_hand1024_single_bucket(stm_hand) * 32 + shogi_hand1024_single_bucket(ntm_hand),
        }
    }
}

/// King axis for YaneuraOu-compatible SFNN LayerStack buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShogiSfnnKingBucketKind {
    None,
    #[default]
    KingRank9,
    KingRank81,
    King9ZoneByKing9Zone,
    King13ZoneByKing13Zone,
    King21ByKing21,
    King29ByKing29,
}

impl ShogiSfnnKingBucketKind {
    pub const fn bucket_count(self) -> usize {
        match self {
            Self::None => 1,
            Self::KingRank9 => 9,
            Self::KingRank81 => 81,
            Self::King9ZoneByKing9Zone => 81,
            Self::King13ZoneByKing13Zone => 169,
            Self::King21ByKing21 => 441,
            Self::King29ByKing29 => 841,
        }
    }

    pub const fn axis_dim(self) -> usize {
        match self {
            Self::None => 0,
            Self::KingRank9 => 3,
            Self::KingRank81 => 9,
            Self::King9ZoneByKing9Zone => 9,
            Self::King13ZoneByKing13Zone => 13,
            Self::King21ByKing21 => 21,
            Self::King29ByKing29 => 29,
        }
    }

    pub fn bucket(self, pos: &PackedSfenValue) -> usize {
        match self {
            Self::None => 0,
            Self::KingRank9 => ShogiKingRankBucket::<9>.bucket(pos),
            Self::KingRank81 => ShogiKingRankBucket::<81>.bucket(pos),
            Self::King9ZoneByKing9Zone => ShogiKing9ZoneByKing9ZoneBucket::bucket_index(pos),
            Self::King13ZoneByKing13Zone => ShogiKing13ZoneByKing13ZoneBucket::bucket_index(pos),
            Self::King21ByKing21 => ShogiKing21ByKing21Bucket::bucket_index(pos),
            Self::King29ByKing29 => ShogiKing29ByKing29Bucket::bucket_index(pos),
        }
    }

    pub fn bucket_from_board(self, board: &ShogiBoard) -> usize {
        let (f_sq, e_sq) = shogi_perspective_king_squares_from_board(board);
        match self {
            Self::None => 0,
            Self::KingRank9 => shogi_king_rank_bucket_from_board::<9>(board),
            Self::KingRank81 => shogi_king_rank_bucket_from_board::<81>(board),
            Self::King9ZoneByKing9Zone => {
                shogi_king9_zone_single_bucket(f_sq) * 9 + shogi_king9_zone_single_bucket(e_sq)
            }
            Self::King13ZoneByKing13Zone => {
                shogi_king13_zone_single_bucket(f_sq) * 13 + shogi_king13_zone_single_bucket(e_sq)
            }
            Self::King21ByKing21 => shogi_king21_single_bucket(f_sq) * 21 + shogi_king21_single_bucket(e_sq),
            Self::King29ByKing29 => shogi_king29_single_bucket(f_sq) * 29 + shogi_king29_single_bucket(e_sq),
        }
    }
}

/// Progress axis for YaneuraOu-compatible SFNN LayerStack buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShogiSfnnProgressBucketKind {
    #[default]
    None,
    Progress2,
    Progress3,
    Progress4,
    Progress8,
    Progress16,
    Progress32,
}

impl ShogiSfnnProgressBucketKind {
    pub const fn bucket_count(self) -> usize {
        match self {
            Self::None => 1,
            Self::Progress2 => 2,
            Self::Progress3 => 3,
            Self::Progress4 => 4,
            Self::Progress8 => 8,
            Self::Progress16 => 16,
            Self::Progress32 => 32,
        }
    }

    pub fn bucket(self, pos: &PackedSfenValue) -> usize {
        let count = self.bucket_count();
        if count == 1 { 0 } else { shogi_sfnn_progress_bucket(pos, count) }
    }

    pub fn bucket_from_board(self, board: &ShogiBoard) -> usize {
        let count = self.bucket_count();
        if count == 1 {
            0
        } else {
            shogi_sfnn_progress_bucket_from_value(shogi_sfnn_progress_0_to_255_from_board(board), count)
        }
    }
}

/// Runtime-selectable YaneuraOu-compatible SFNN LayerStack bucket.
///
/// The LayerStack index is composed exactly like YaneuraOu:
///
/// `idx = ((hand_bucket * king_bucket_count) + king_bucket) * progress_bucket_count + progress_bucket`
///
/// The associated `OutputBuckets::BUCKETS` for the wrapper below is the maximum
/// supported count; CUDA direct training uses the actual architecture stack
/// count separately and validates each per-sample bucket against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShogiSfnnLayerStackBucketKind {
    pub hand: ShogiSfnnHandBucketKind,
    pub king: ShogiSfnnKingBucketKind,
    pub progress: ShogiSfnnProgressBucketKind,
}

impl Default for ShogiSfnnLayerStackBucketKind {
    fn default() -> Self {
        Self::KingRank9
    }
}

#[allow(non_upper_case_globals)]
impl ShogiSfnnLayerStackBucketKind {
    pub const fn new(
        hand: ShogiSfnnHandBucketKind,
        king: ShogiSfnnKingBucketKind,
        progress: ShogiSfnnProgressBucketKind,
    ) -> Self {
        Self { hand, king, progress }
    }

    pub const Single: Self =
        Self::new(ShogiSfnnHandBucketKind::None, ShogiSfnnKingBucketKind::None, ShogiSfnnProgressBucketKind::None);
    pub const KingRank9: Self =
        Self::new(ShogiSfnnHandBucketKind::None, ShogiSfnnKingBucketKind::KingRank9, ShogiSfnnProgressBucketKind::None);
    pub const KingRank81: Self = Self::new(
        ShogiSfnnHandBucketKind::None,
        ShogiSfnnKingBucketKind::KingRank81,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const King9ZoneByKing9Zone: Self = Self::new(
        ShogiSfnnHandBucketKind::None,
        ShogiSfnnKingBucketKind::King9ZoneByKing9Zone,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const King13ZoneByKing13Zone: Self = Self::new(
        ShogiSfnnHandBucketKind::None,
        ShogiSfnnKingBucketKind::King13ZoneByKing13Zone,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const King21ByKing21: Self = Self::new(
        ShogiSfnnHandBucketKind::None,
        ShogiSfnnKingBucketKind::King21ByKing21,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const King29ByKing29: Self = Self::new(
        ShogiSfnnHandBucketKind::None,
        ShogiSfnnKingBucketKind::King29ByKing29,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand4: Self =
        Self::new(ShogiSfnnHandBucketKind::Hand4, ShogiSfnnKingBucketKind::None, ShogiSfnnProgressBucketKind::None);
    pub const Hand16: Self =
        Self::new(ShogiSfnnHandBucketKind::Hand16, ShogiSfnnKingBucketKind::None, ShogiSfnnProgressBucketKind::None);
    pub const Hand64: Self =
        Self::new(ShogiSfnnHandBucketKind::Hand64, ShogiSfnnKingBucketKind::None, ShogiSfnnProgressBucketKind::None);
    pub const Hand64z: Self =
        Self::new(ShogiSfnnHandBucketKind::Hand64z, ShogiSfnnKingBucketKind::None, ShogiSfnnProgressBucketKind::None);
    pub const Hand64KingRank9: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand64,
        ShogiSfnnKingBucketKind::KingRank9,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand64KingRank81: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand64,
        ShogiSfnnKingBucketKind::KingRank81,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand64King21ByKing21: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand64,
        ShogiSfnnKingBucketKind::King21ByKing21,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand64King29ByKing29: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand64,
        ShogiSfnnKingBucketKind::King29ByKing29,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand256: Self =
        Self::new(ShogiSfnnHandBucketKind::Hand256, ShogiSfnnKingBucketKind::None, ShogiSfnnProgressBucketKind::None);
    pub const Hand256KingRank9: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand256,
        ShogiSfnnKingBucketKind::KingRank9,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand256KingRank81: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand256,
        ShogiSfnnKingBucketKind::KingRank81,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand256King21ByKing21: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand256,
        ShogiSfnnKingBucketKind::King21ByKing21,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand256King29ByKing29: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand256,
        ShogiSfnnKingBucketKind::King29ByKing29,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand1024: Self =
        Self::new(ShogiSfnnHandBucketKind::Hand1024, ShogiSfnnKingBucketKind::None, ShogiSfnnProgressBucketKind::None);
    pub const Hand1024KingRank9: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand1024,
        ShogiSfnnKingBucketKind::KingRank9,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand1024KingRank81: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand1024,
        ShogiSfnnKingBucketKind::KingRank81,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand1024King21ByKing21: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand1024,
        ShogiSfnnKingBucketKind::King21ByKing21,
        ShogiSfnnProgressBucketKind::None,
    );
    pub const Hand1024King29ByKing29: Self = Self::new(
        ShogiSfnnHandBucketKind::Hand1024,
        ShogiSfnnKingBucketKind::King29ByKing29,
        ShogiSfnnProgressBucketKind::None,
    );

    pub const fn hand_bucket_count(self) -> usize {
        self.hand.bucket_count()
    }

    pub const fn king_bucket_count(self) -> usize {
        self.king.bucket_count()
    }

    pub const fn progress_bucket_count(self) -> usize {
        self.progress.bucket_count()
    }

    pub const fn num_stacks(self) -> usize {
        self.hand_bucket_count() * self.king_bucket_count() * self.progress_bucket_count()
    }

    pub fn bucket(self, pos: &PackedSfenValue) -> usize {
        let board = pos.decode();
        self.bucket_from_board(&board)
    }

    pub fn bucket_from_board(self, board: &ShogiBoard) -> usize {
        let hand_bucket = self.hand.bucket_from_board(board);
        let king_bucket = self.king.bucket_from_board(board);
        let progress_bucket = self.progress.bucket_from_board(board);
        (hand_bucket * self.king_bucket_count() + king_bucket) * self.progress_bucket_count() + progress_bucket
    }
}

#[derive(Clone, Copy, Default)]
pub struct ShogiSfnnLayerStackBucket {
    kind: ShogiSfnnLayerStackBucketKind,
}

impl ShogiSfnnLayerStackBucket {
    pub const fn new(kind: ShogiSfnnLayerStackBucketKind) -> Self {
        Self { kind }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiSfnnLayerStackBucket {
    const BUCKETS: usize = 1024 * 841 * 32;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        self.kind.bucket(pos)
    }
}

/// Default boundaries for shogi ply-based 9-bucket split.
///
/// bucket0: <=30, bucket1: <=44, ..., bucket7: <=138, bucket8: >=139
pub const SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS: [u16; 8] = [30, 44, 58, 72, 86, 100, 116, 138];

/// Number of features used by progress-based bucket model.
pub const SHOGI_PROGRESS8_NUM_FEATURES: usize = 6;

/// Feature order for progress-based bucket model (coeff_v1).
pub const SHOGI_PROGRESS8_FEATURE_ORDER: [&str; SHOGI_PROGRESS8_NUM_FEATURES] = [
    "x_board_non_king",
    "x_hand_total",
    "x_major_board",
    "x_promoted_board",
    "x_stm_king_rank_rel",
    "x_ntm_king_rank_rel",
];

/// Number of buckets for progress8.
pub const SHOGI_PROGRESS8_NUM_BUCKETS: usize = 8;

/// Number of KP-absolute weights: `81 * FE_OLD_END`.
pub const SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS: usize = 81 * FE_OLD_END;

/// Number of weights in YaneuraOu SFNN progress parameters:
/// `SQ_NB * Eval::fe_end`.
pub const SHOGI_SFNN_PROGRESS_WEIGHT_COUNT: usize = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;

/// Number of scalar progress values used by YaneuraOu SFNN progress buckets.
pub const SHOGI_SFNN_PROGRESS_VALUE_COUNT: usize = 256;

/// YaneuraOu hash block for SFNN progress parameters (`"oPRO"`).
pub const SHOGI_SFNN_PROGRESS_HASH: u32 = 0x6F50_524F;

/// YaneuraOu-compatible SFNN progress parameters.
///
/// The values are q16 logits:
///
/// `sum_q16 = bias_q16 + sum(active_kp_abs_weights_q16)`
///
/// `sum_q16` is converted to a scalar `0..=255`; each architecture-specific
/// progress bucket then uses `progress * bucket_count / 256`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShogiSfnnProgressQ16Params {
    pub bias_q16: i32,
    pub weights_q16: Box<[i32]>,
}

impl ShogiSfnnProgressQ16Params {
    pub fn zero() -> Self {
        Self { bias_q16: 0, weights_q16: vec![0; SHOGI_SFNN_PROGRESS_WEIGHT_COUNT].into_boxed_slice() }
    }

    /// Deterministic built-in progress parameters used by BulletOu when an
    /// SFNN `progressN` architecture is selected.
    ///
    /// This is intentionally not loaded from a user-provided side file. The
    /// parameters are exported into the `nn.bin` Progress section, so
    /// YaneuraOu and BulletOu see the same bucket assignment.
    ///
    /// The heuristic is deliberately simple: hand material increases progress,
    /// and promoted major pieces on board add a small amount. Normal board
    /// pieces are left at zero so the start position maps near the opening
    /// side of the scale.
    pub fn material_heuristic() -> Self {
        fn q16(x: f32) -> i32 {
            (x * 65_536.0).round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
        }

        fn fill_bp(bp_weights: &mut [i32], start: u16, len: usize, value: i32) {
            let start = start as usize;
            let end = start.saturating_add(len).min(bp_weights.len());
            for weight in &mut bp_weights[start..end] {
                *weight = value;
            }
        }

        let mut bp_weights = vec![0i32; FE_OLD_END];
        let hand_point = 0.07_f32;
        for &(f_start, e_start, len, piece_points) in &[
            (F_HAND_PAWN, E_HAND_PAWN, 18usize, 1.0_f32),
            (F_HAND_LANCE, E_HAND_LANCE, 4usize, 2.0_f32),
            (F_HAND_KNIGHT, E_HAND_KNIGHT, 4usize, 2.0_f32),
            (F_HAND_SILVER, E_HAND_SILVER, 4usize, 3.0_f32),
            (F_HAND_GOLD, E_HAND_GOLD, 4usize, 3.0_f32),
            (F_HAND_BISHOP, E_HAND_BISHOP, 2usize, 5.0_f32),
            (F_HAND_ROOK, E_HAND_ROOK, 2usize, 5.0_f32),
        ] {
            let value = q16(hand_point * piece_points);
            fill_bp(&mut bp_weights, f_start, len, value);
            fill_bp(&mut bp_weights, e_start, len, value);
        }

        // Horses and dragons usually imply the game has progressed even if the
        // captured material is not currently in hand.
        fill_bp(&mut bp_weights, F_HORSE, 81, q16(0.18));
        fill_bp(&mut bp_weights, E_HORSE, 81, q16(0.18));
        fill_bp(&mut bp_weights, F_DRAGON, 81, q16(0.22));
        fill_bp(&mut bp_weights, E_DRAGON, 81, q16(0.22));

        let mut weights_q16 = Vec::with_capacity(SHOGI_SFNN_PROGRESS_WEIGHT_COUNT);
        for _sq in 0..81 {
            weights_q16.extend_from_slice(&bp_weights);
        }

        Self { bias_q16: q16(-3.0), weights_q16: weights_q16.into_boxed_slice() }
    }

    pub fn new(bias_q16: i32, weights_q16: Vec<i32>) -> Result<Self, String> {
        if weights_q16.len() != SHOGI_SFNN_PROGRESS_WEIGHT_COUNT {
            return Err(format!(
                "SFNN progress q16 weight count mismatch: got {}, expected {}",
                weights_q16.len(),
                SHOGI_SFNN_PROGRESS_WEIGHT_COUNT
            ));
        }
        Ok(Self { bias_q16, weights_q16: weights_q16.into_boxed_slice() })
    }
}

static SHOGI_PROGRESS_KP_ABS_WEIGHTS: OnceLock<Box<[f32]>> = OnceLock::new();
static SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS: [f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS] =
    [0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
static SHOGI_SFNN_PROGRESS_Q16_PARAMS: OnceLock<ShogiSfnnProgressQ16Params> = OnceLock::new();
static SHOGI_SFNN_PROGRESS_IS_MATERIAL_HEURISTIC: OnceLock<bool> = OnceLock::new();
static SHOGI_SFNN_PROGRESS_Q16_THRESHOLDS: OnceLock<[i64; SHOGI_SFNN_PROGRESS_VALUE_COUNT - 1]> = OnceLock::new();

pub fn set_shogi_sfnn_progress_q16_params(params: ShogiSfnnProgressQ16Params) -> Result<(), String> {
    if let Some(existing) = SHOGI_SFNN_PROGRESS_Q16_PARAMS.get() {
        if existing == &params {
            return Ok(());
        }
        return Err("different SFNN progress q16 parameters are already loaded in this process".to_string());
    }

    let is_material_heuristic = params == ShogiSfnnProgressQ16Params::material_heuristic();
    SHOGI_SFNN_PROGRESS_IS_MATERIAL_HEURISTIC
        .set(is_material_heuristic)
        .map_err(|_| "SFNN progress q16 parameter kind is already loaded in this process".to_string())?;
    SHOGI_SFNN_PROGRESS_Q16_PARAMS
        .set(params)
        .map_err(|_| "SFNN progress q16 parameters are already loaded in this process".to_string())
}

pub fn shogi_sfnn_progress_q16_params() -> Option<&'static ShogiSfnnProgressQ16Params> {
    SHOGI_SFNN_PROGRESS_Q16_PARAMS.get()
}

fn shogi_sfnn_progress_q16_thresholds() -> &'static [i64; SHOGI_SFNN_PROGRESS_VALUE_COUNT - 1] {
    SHOGI_SFNN_PROGRESS_Q16_THRESHOLDS.get_or_init(|| {
        std::array::from_fn(|i| {
            let p = (i + 1) as f64 / SHOGI_SFNN_PROGRESS_VALUE_COUNT as f64;
            (p.ln() - (1.0 - p).ln()).mul_add(65_536.0, 0.0).round() as i64
        })
    })
}

/// Convert a q16 logit sum to YaneuraOu's scalar progress value `0..=255`.
pub fn shogi_sfnn_progress_0_to_255_from_sum_q16(sum_q16: i64) -> u8 {
    shogi_sfnn_progress_q16_thresholds().partition_point(|&threshold| threshold <= sum_q16) as u8
}

/// Compute YaneuraOu-compatible SFNN progress value `0..=255`.
pub fn shogi_sfnn_progress_0_to_255(pos: &PackedSfenValue) -> u8 {
    let Some(params) = shogi_sfnn_progress_q16_params() else {
        return shogi_sfnn_progress_0_to_255_from_sum_q16(0);
    };
    let board = pos.decode();
    shogi_sfnn_progress_0_to_255_from_sum_q16(shogi_sfnn_progress_sum_q16_from_board(&board, params))
}

fn shogi_sfnn_progress_0_to_255_from_board(board: &ShogiBoard) -> u8 {
    let Some(params) = shogi_sfnn_progress_q16_params() else {
        return shogi_sfnn_progress_0_to_255_from_sum_q16(0);
    };
    shogi_sfnn_progress_0_to_255_from_sum_q16(shogi_sfnn_progress_sum_q16_from_board(board, params))
}

fn shogi_sfnn_progress_sum_q16_from_board(board: &ShogiBoard, params: &ShogiSfnnProgressQ16Params) -> i64 {
    if SHOGI_SFNN_PROGRESS_IS_MATERIAL_HEURISTIC.get().copied().unwrap_or(false) {
        return shogi_sfnn_material_heuristic_sum_q16_from_board(board);
    }

    if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
        return i64::from(params.bias_q16);
    }

    let weights = &params.weights_q16;
    let bk_base = board.black_king_sq.index() * FE_OLD_END;
    let wk_base = board.white_king_sq.inverse().index() * FE_OLD_END;
    let mut sum_q16 = i64::from(params.bias_q16);

    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);

                let bp_b = BonaPiece::from_piece_square(piece, sq, Color::Black);
                if bp_b != BonaPiece::ZERO {
                    sum_q16 += i64::from(weights[bk_base + bp_b.value() as usize]);
                }

                let bp_w = BonaPiece::from_piece_square(piece, sq, Color::White);
                if bp_w != BonaPiece::ZERO {
                    sum_q16 += i64::from(weights[wk_base + bp_w.value() as usize]);
                }
            }
        }
    }

    for owner in [Color::Black, Color::White] {
        let hand = if owner == Color::Black { board.black_hand } else { board.white_hand };
        for &pt in &HAND_PIECE_TYPES {
            let count = hand.count(pt);
            for c in 1..=count {
                let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c);
                if bp_b != BonaPiece::ZERO {
                    sum_q16 += i64::from(weights[bk_base + bp_b.value() as usize]);
                }

                let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c);
                if bp_w != BonaPiece::ZERO {
                    sum_q16 += i64::from(weights[wk_base + bp_w.value() as usize]);
                }
            }
        }
    }

    sum_q16
}

fn shogi_sfnn_material_heuristic_sum_q16_from_board(board: &ShogiBoard) -> i64 {
    #[inline]
    fn q16(x: f32) -> i64 {
        (x * 65_536.0).round().clamp(i32::MIN as f32, i32::MAX as f32) as i64
    }

    let mut sum_q16 = q16(-3.0);
    let hand_point = 0.07_f32;
    let pawn = q16(hand_point * 1.0);
    let lance_knight = q16(hand_point * 2.0);
    let silver_gold = q16(hand_point * 3.0);
    let bishop_rook = q16(hand_point * 5.0);
    let horse = q16(0.18);
    let dragon = q16(0.22);

    for piece in &board.board {
        match piece.piece_type {
            PieceType::Horse => sum_q16 += 2 * horse,
            PieceType::Dragon => sum_q16 += 2 * dragon,
            _ => {}
        }
    }

    for hand in [board.black_hand, board.white_hand] {
        sum_q16 += 2 * i64::from(hand.pawn()) * pawn;
        sum_q16 += 2 * i64::from(hand.lance()) * lance_knight;
        sum_q16 += 2 * i64::from(hand.knight()) * lance_knight;
        sum_q16 += 2 * i64::from(hand.silver()) * silver_gold;
        sum_q16 += 2 * i64::from(hand.gold()) * silver_gold;
        sum_q16 += 2 * i64::from(hand.bishop()) * bishop_rook;
        sum_q16 += 2 * i64::from(hand.rook()) * bishop_rook;
    }

    sum_q16
}

/// Map scalar progress `0..=255` into an arbitrary YaneuraOu progress bucket count.
pub fn shogi_sfnn_progress_bucket_from_value(progress: u8, bucket_count: usize) -> usize {
    if bucket_count <= 1 {
        return 0;
    }
    let raw = usize::from(progress) * bucket_count / SHOGI_SFNN_PROGRESS_VALUE_COUNT;
    raw.min(bucket_count - 1)
}

/// Compute a YaneuraOu-compatible SFNN progress bucket for `bucket_count`.
pub fn shogi_sfnn_progress_bucket(pos: &PackedSfenValue, bucket_count: usize) -> usize {
    shogi_sfnn_progress_bucket_from_value(shogi_sfnn_progress_0_to_255(pos), bucket_count)
}

/// Progress-based 8 bucket assignment (logistic regression).
///
/// `p = sigmoid(bias + Σ(w_i * ((x_i - mean_i) / std_i)))`
/// `bucket = min(7, floor(p * 8.0))`
#[derive(Clone, Copy)]
pub struct ShogiProgressBucket8 {
    pub mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub std: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub bias: f32,
    pub z_clip: [f32; 2],
}

impl ShogiProgressBucket8 {
    pub const fn new(
        mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        std: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        bias: f32,
        z_clip: [f32; 2],
    ) -> Self {
        Self { mean, std, weights, bias, z_clip }
    }

    /// Extract raw progress-model features in coeff_v1 order.
    pub fn extract_features(pos: &PackedSfenValue) -> [f32; SHOGI_PROGRESS8_NUM_FEATURES] {
        let board = pos.decode();

        let board_non_king = board
            .board
            .iter()
            .filter(|p| p.piece_type != crate::shogi::PieceType::None && p.piece_type != crate::shogi::PieceType::King)
            .count() as f32;

        let hand_total =
            board.black_hand.counts.iter().chain(board.white_hand.counts.iter()).map(|&v| v as f32).sum::<f32>();

        let major_board = board
            .board
            .iter()
            .filter(|p| {
                matches!(
                    p.piece_type,
                    crate::shogi::PieceType::Bishop
                        | crate::shogi::PieceType::Rook
                        | crate::shogi::PieceType::Horse
                        | crate::shogi::PieceType::Dragon
                )
            })
            .count() as f32;

        let promoted_board = board.board.iter().filter(|p| p.piece_type.is_promoted()).count() as f32;

        let stm = board.side_to_move;
        let f_king = board.king_square(stm);
        let e_king = board.king_square(stm.opponent());

        let stm_king_rank_rel = match stm {
            crate::shogi::Color::Black => f_king.rank() as f32,
            crate::shogi::Color::White => (8 - f_king.rank()) as f32,
        };

        let ntm_king_rank_rel = match stm {
            crate::shogi::Color::Black => (8 - e_king.rank()) as f32,
            crate::shogi::Color::White => e_king.rank() as f32,
        };

        [board_non_king, hand_total, major_board, promoted_board, stm_king_rank_rel, ntm_king_rank_rel]
    }

    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let x = Self::extract_features(pos);

        let mut z = self.bias;
        for (i, &x_i) in x.iter().enumerate() {
            let std = if self.std[i] > 0.0 { self.std[i] } else { 1.0 };
            let x_norm = (x_i - self.mean[i]) / std;
            z += self.weights[i] * x_norm;
        }

        let z_min = self.z_clip[0].min(self.z_clip[1]);
        let z_max = self.z_clip[0].max(self.z_clip[1]);
        let z_clamped = z.clamp(z_min, z_max);
        let p = 1.0 / (1.0 + (-z_clamped).exp());
        p.clamp(0.0, 1.0)
    }
}

impl Default for ShogiProgressBucket8 {
    fn default() -> Self {
        // docs/progress-bucket-coeff-script-spec-v1.md の JSON 例を既定値として採用。
        Self {
            mean: [30.12, 8.45, 2.18, 1.63, 6.71, 6.24],
            std: [3.77, 4.02, 0.66, 1.40, 1.31, 1.27],
            weights: [-0.81, 0.56, -0.32, 0.48, 0.11, -0.09],
            bias: -0.15,
            z_clip: [-8.0, 8.0],
        }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgressBucket8 {
    const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as usize
    }
}

/// Progress-based 8 bucket assignment using YaneuraOu/tanuki- style KP-absolute features.
///
/// Weights are process-global so this type stays `Copy` and can be embedded in `OutputBuckets`.
#[derive(Clone, Copy, Default)]
pub struct ShogiProgressKPAbs;

impl ShogiProgressKPAbs {
    fn weights() -> &'static [f32] {
        SHOGI_PROGRESS_KP_ABS_WEIGHTS.get().map_or(&SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS, |weights| weights.as_ref())
    }

    /// Enumerates all active KP-absolute feature indices for the position.
    ///
    /// This is the exact feature expansion used by `progress8kpabs`.
    pub fn for_each_active_index(pos: &PackedSfenValue, mut f: impl FnMut(usize)) {
        let board = pos.decode();
        Self::for_each_active_index_from_board(&board, &mut f);
    }

    fn for_each_active_index_from_board(board: &ShogiBoard, mut f: impl FnMut(usize)) {
        if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
            return;
        }

        let sq_bk = board.black_king_sq.index();
        let sq_wk = board.white_king_sq.inverse().index();

        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);

                    let bp_b = BonaPiece::from_piece_square(piece, sq, Color::Black);
                    if bp_b != BonaPiece::ZERO {
                        f(sq_bk * FE_OLD_END + bp_b.value() as usize);
                    }

                    let bp_w = BonaPiece::from_piece_square(piece, sq, Color::White);
                    if bp_w != BonaPiece::ZERO {
                        f(sq_wk * FE_OLD_END + bp_w.value() as usize);
                    }
                }
            }
        }

        for owner in [Color::Black, Color::White] {
            let hand = if owner == Color::Black { board.black_hand } else { board.white_hand };
            for &pt in &HAND_PIECE_TYPES {
                let count = hand.count(pt);
                for c in 1..=count {
                    let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c);
                    if bp_b != BonaPiece::ZERO {
                        f(sq_bk * FE_OLD_END + bp_b.value() as usize);
                    }

                    let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c);
                    if bp_w != BonaPiece::ZERO {
                        f(sq_wk * FE_OLD_END + bp_w.value() as usize);
                    }
                }
            }
        }
    }

    /// Collects all active KP-absolute feature indices into `out`.
    pub fn collect_active_indices(pos: &PackedSfenValue, out: &mut Vec<usize>) {
        out.clear();
        Self::for_each_active_index(pos, |idx| out.push(idx));
    }

    /// Loads KP-absolute weights from a binary coefficient file.
    ///
    /// Only one KP-absolute model can be loaded per process.
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
        let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();
        if bytes.len() != expected {
            return Err(format!("progress.bin size mismatch: got {} bytes, expected {}", bytes.len(), expected));
        }

        let weights: Vec<f32> = bytes
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size is checked")) as f32)
            .collect();

        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .set(weights.into_boxed_slice())
            .map_err(|_| "KP-absolute progress weights are already loaded in this process".to_string())?;

        Ok(Self)
    }

    /// Estimates progress in `0.0..=1.0`.
    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let weights = Self::weights();
        let mut sum = 0.0f32;
        Self::for_each_active_index(pos, |idx| sum += weights[idx]);

        let p = 1.0 / (1.0 + (-sum).exp());
        p.clamp(0.0, 1.0)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgressKPAbs {
    const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as usize
    }
}

/// Number of features used by Gikou-lite progress model.
pub const SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES: usize = 34;

/// Feature order for Gikou-lite progress model (coeff_v2).
pub const SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER: [&str; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] = [
    "x_board_non_king",
    "x_hand_total",
    "x_major_board",
    "x_promoted_board",
    "x_stm_king_rank_rel",
    "x_ntm_king_rank_rel",
    "x_stm_all_to_own_king_d1",
    "x_stm_all_to_own_king_d2",
    "x_stm_all_to_own_king_d3p",
    "x_stm_all_to_opp_king_d1",
    "x_stm_all_to_opp_king_d2",
    "x_stm_all_to_opp_king_d3p",
    "x_ntm_all_to_own_king_d1",
    "x_ntm_all_to_own_king_d2",
    "x_ntm_all_to_own_king_d3p",
    "x_ntm_all_to_opp_king_d1",
    "x_ntm_all_to_opp_king_d2",
    "x_ntm_all_to_opp_king_d3p",
    "x_stm_major_to_own_king_d1",
    "x_stm_major_to_own_king_d2",
    "x_stm_major_to_own_king_d3p",
    "x_stm_major_to_opp_king_d1",
    "x_stm_major_to_opp_king_d2",
    "x_stm_major_to_opp_king_d3p",
    "x_ntm_major_to_own_king_d1",
    "x_ntm_major_to_own_king_d2",
    "x_ntm_major_to_own_king_d3p",
    "x_ntm_major_to_opp_king_d1",
    "x_ntm_major_to_opp_king_d2",
    "x_ntm_major_to_opp_king_d3p",
    "x_stm_hand_total",
    "x_ntm_hand_total",
    "x_stm_hand_major",
    "x_ntm_hand_major",
];

#[inline]
fn chebyshev_distance(a: crate::shogi::Square, b: crate::shogi::Square) -> u8 {
    let df = a.file().abs_diff(b.file());
    let dr = a.rank().abs_diff(b.rank());
    df.max(dr)
}

#[inline]
fn distance_bin(d: u8) -> usize {
    if d <= 1 {
        0
    } else if d == 2 {
        1
    } else {
        2
    }
}

#[inline]
fn is_major_piece(pt: crate::shogi::PieceType) -> bool {
    matches!(
        pt,
        crate::shogi::PieceType::Bishop
            | crate::shogi::PieceType::Rook
            | crate::shogi::PieceType::Horse
            | crate::shogi::PieceType::Dragon
    )
}

/// Progress-based 8 bucket assignment (Gikou-lite logistic regression).
///
/// `p = sigmoid(bias + Σ(w_i * ((x_i - mean_i) / std_i)))`
/// `bucket = min(7, floor(p * 8.0))`
#[derive(Clone, Copy)]
pub struct ShogiProgressBucket8GikouLite {
    pub mean: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
    pub std: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
    pub weights: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
    pub bias: f32,
    pub z_clip: [f32; 2],
}

impl ShogiProgressBucket8GikouLite {
    pub const fn new(
        mean: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
        std: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
        weights: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
        bias: f32,
        z_clip: [f32; 2],
    ) -> Self {
        Self { mean, std, weights, bias, z_clip }
    }

    /// Extract raw progress-model features in coeff_v2 order.
    pub fn extract_features(pos: &PackedSfenValue) -> [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_king = board.king_square(stm);
        let ntm_king = board.king_square(stm.opponent());

        let mut out = [0.0f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES];
        let baseline = ShogiProgressBucket8::extract_features(pos);
        out[..SHOGI_PROGRESS8_NUM_FEATURES].copy_from_slice(&baseline);

        for (idx, piece) in board.board.iter().enumerate() {
            let pt = piece.piece_type;
            if matches!(pt, crate::shogi::PieceType::None | crate::shogi::PieceType::King) {
                continue;
            }

            let is_stm_piece = piece.color == stm;
            let side_offset = if is_stm_piece { 6usize } else { 12usize };
            let major_offset = if is_stm_piece { 18usize } else { 24usize };

            let sq = crate::shogi::Square::from_index(idx);
            let own_king = if is_stm_piece { stm_king } else { ntm_king };
            let opp_king = if is_stm_piece { ntm_king } else { stm_king };

            let own_bin = distance_bin(chebyshev_distance(sq, own_king));
            let opp_bin = distance_bin(chebyshev_distance(sq, opp_king));

            out[side_offset + own_bin] += 1.0;
            out[side_offset + 3 + opp_bin] += 1.0;

            if is_major_piece(pt) {
                out[major_offset + own_bin] += 1.0;
                out[major_offset + 3 + opp_bin] += 1.0;
            }
        }

        let stm_hand = if stm == crate::shogi::Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == crate::shogi::Color::Black { board.white_hand } else { board.black_hand };

        out[30] = stm_hand.counts.iter().map(|&v| v as f32).sum::<f32>();
        out[31] = ntm_hand.counts.iter().map(|&v| v as f32).sum::<f32>();
        // bishop + rook in hand
        out[32] = (stm_hand.counts[5] + stm_hand.counts[6]) as f32;
        out[33] = (ntm_hand.counts[5] + ntm_hand.counts[6]) as f32;

        out
    }

    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let x = Self::extract_features(pos);

        let mut z = self.bias;
        for (i, &x_i) in x.iter().enumerate() {
            let std = if self.std[i] > 0.0 { self.std[i] } else { 1.0 };
            let x_norm = (x_i - self.mean[i]) / std;
            z += self.weights[i] * x_norm;
        }

        let z_min = self.z_clip[0].min(self.z_clip[1]);
        let z_max = self.z_clip[0].max(self.z_clip[1]);
        let z_clamped = z.clamp(z_min, z_max);
        let p = 1.0 / (1.0 + (-z_clamped).exp());
        p.clamp(0.0, 1.0)
    }
}

impl Default for ShogiProgressBucket8GikouLite {
    fn default() -> Self {
        // coeff_v2 を指定しない場合の最小既定値（ほぼ中立）。
        Self {
            mean: [0.0; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
            std: [1.0; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
            weights: [0.0; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
            bias: 0.0,
            z_clip: [-8.0, 8.0],
        }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgressBucket8GikouLite {
    const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as usize
    }
}

/// 将棋 LayerStacks 用の手数ベース 9 バケット。
///
/// `game_ply` を固定境界で 9 分割する:
/// - b0: <= bounds[0]
/// - ...
/// - b7: <= bounds[7]
/// - b8: > bounds[7]
#[derive(Clone, Copy)]
pub struct ShogiPlyBucket9 {
    pub bounds: [u16; 8],
}

impl Default for ShogiPlyBucket9 {
    fn default() -> Self {
        Self { bounds: SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiPlyBucket9 {
    const BUCKETS: usize = 9;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        let ply = pos.game_ply();
        for (i, &bound) in self.bounds.iter().enumerate() {
            if ply <= bound {
                return i;
            }
        }
        8
    }
}

/// Runtime-selectable 9-bucket mode for shogi LayerStacks.
///
/// This enum stays `Copy` because `OutputBuckets` requires it, so boxing the
/// larger progress variants is not an option here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Default)]
pub enum ShogiLayerStackBucket9 {
    #[default]
    KingRank9,
    Ply9([u16; 8]),
    Progress8(ShogiProgressBucket8),
    Progress8GikouLite(ShogiProgressBucket8GikouLite),
    Progress8KPAbs(ShogiProgressKPAbs),
}

impl OutputBuckets<PackedSfenValue> for ShogiLayerStackBucket9 {
    const BUCKETS: usize = 9;

    fn bucket(&self, pos: &PackedSfenValue) -> usize {
        match self {
            Self::KingRank9 => ShogiKingRankBucket::<9>.bucket(pos),
            Self::Ply9(bounds) => {
                let ply = pos.game_ply();
                for (i, &bound) in bounds.iter().enumerate() {
                    if ply <= bound {
                        return i;
                    }
                }
                8
            }
            // 9bucket互換モード:
            // progress8 は bucket 0..7 を使用し、bucket 8 は未使用となる。
            Self::Progress8(progress) => progress.bucket(pos),
            // 9bucket互換モード:
            // progress8-gikou-lite も bucket 0..7 を使用し、bucket 8 は未使用となる。
            Self::Progress8GikouLite(progress) => progress.bucket(pos),
            // 9bucket互換モード:
            // KP-absolute も bucket 0..7 を使用し、bucket 8 は未使用となる。
            Self::Progress8KPAbs(progress) => progress.bucket(pos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::Square;

    fn psv_with_ply(ply: u16) -> PackedSfenValue {
        let mut psv = PackedSfenValue::default();
        let bytes = psv.as_bytes_mut();
        let le = ply.to_le_bytes();
        bytes[36] = le[0];
        bytes[37] = le[1];
        psv
    }

    fn write_bits_lsb_first(bytes: &mut [u8], cursor: &mut usize, value: u32, bits: u8) {
        for i in 0..bits {
            if ((value >> i) & 1) != 0 {
                let bit = *cursor + usize::from(i);
                bytes[bit / 8] |= 1 << (bit % 8);
            }
        }
        *cursor += usize::from(bits);
    }

    fn psv_with_kings(stm: Color, black_king: Square, white_king: Square) -> PackedSfenValue {
        let mut psv = PackedSfenValue::default();
        let bytes = psv.as_bytes_mut();
        let mut cursor = 0usize;
        write_bits_lsb_first(bytes, &mut cursor, u32::from(stm == Color::White), 1);
        write_bits_lsb_first(bytes, &mut cursor, black_king.index() as u32, 7);
        write_bits_lsb_first(bytes, &mut cursor, white_king.index() as u32, 7);
        psv
    }

    #[test]
    fn test_shogi_hand4_and_hand16_single_bucket_formulas() {
        let empty = Hand::EMPTY;
        assert_eq!(shogi_hand4_single_bucket(empty), 0);
        assert_eq!(shogi_hand16_single_bucket(empty), 0);

        let mut hand = Hand::EMPTY;
        hand.set_pawn(1);
        assert_eq!(shogi_hand4_single_bucket(hand), 0);
        assert_eq!(shogi_hand16_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_bishop(1);
        assert_eq!(shogi_hand4_single_bucket(hand), 1);
        assert_eq!(shogi_hand16_single_bucket(hand), 2);

        hand.set_pawn(1);
        assert_eq!(shogi_hand16_single_bucket(hand), 3);
    }

    #[test]
    fn test_shogi_hand64_single_bucket_formula() {
        let empty = Hand::EMPTY;
        assert_eq!(shogi_hand64_single_bucket(empty), 0);

        let mut hand = Hand::EMPTY;
        hand.set_pawn(1);
        assert_eq!(shogi_hand64_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_pawn(5);
        assert_eq!(shogi_hand64_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_lance(1);
        hand.set_knight(1);
        assert_eq!(shogi_hand64_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_silver(1);
        hand.set_gold(1);
        assert_eq!(shogi_hand64_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_bishop(1);
        assert_eq!(shogi_hand64_single_bucket(hand), 4);

        hand = Hand::EMPTY;
        hand.set_bishop(1);
        hand.set_rook(1);
        assert_eq!(shogi_hand64_single_bucket(hand), 6);

        hand = Hand::EMPTY;
        hand.set_pawn(18);
        hand.set_lance(4);
        hand.set_knight(4);
        hand.set_silver(4);
        hand.set_gold(4);
        hand.set_bishop(2);
        hand.set_rook(2);
        assert_eq!(shogi_hand64_single_bucket(hand), 7);
    }

    #[test]
    fn test_shogi_hand64z_single_bucket_formula() {
        let empty = Hand::EMPTY;
        assert_eq!(shogi_hand64z_single_bucket(empty), 0);

        let mut hand = Hand::EMPTY;
        hand.set_pawn(1);
        assert_eq!(shogi_hand64z_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_pawn(5);
        assert_eq!(shogi_hand64z_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_lance(1);
        hand.set_knight(1);
        assert_eq!(shogi_hand64z_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_silver(1);
        hand.set_gold(1);
        assert_eq!(shogi_hand64z_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_bishop(1);
        assert_eq!(shogi_hand64z_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_bishop(1);
        hand.set_rook(1);
        assert_eq!(shogi_hand64z_single_bucket(hand), 3);

        hand = Hand::EMPTY;
        hand.set_pawn(18);
        hand.set_lance(4);
        hand.set_knight(4);
        hand.set_silver(4);
        hand.set_gold(4);
        hand.set_bishop(2);
        hand.set_rook(2);
        assert_eq!(shogi_hand64z_single_bucket(hand), 7);
    }

    #[test]
    fn test_shogi_hand256_single_bucket_formula() {
        let empty = Hand::EMPTY;
        assert_eq!(shogi_hand256_single_bucket(empty), 0);

        let mut hand = Hand::EMPTY;
        hand.set_pawn(1);
        assert_eq!(shogi_hand256_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_lance(1);
        assert_eq!(shogi_hand256_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_knight(1);
        assert_eq!(shogi_hand256_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_silver(1);
        assert_eq!(shogi_hand256_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_gold(1);
        assert_eq!(shogi_hand256_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_bishop(1);
        assert_eq!(shogi_hand256_single_bucket(hand), 4);

        hand = Hand::EMPTY;
        hand.set_rook(1);
        assert_eq!(shogi_hand256_single_bucket(hand), 8);

        hand = Hand::EMPTY;
        hand.set_pawn(18);
        hand.set_lance(4);
        hand.set_knight(4);
        hand.set_silver(4);
        hand.set_gold(4);
        hand.set_bishop(2);
        hand.set_rook(2);
        assert_eq!(shogi_hand256_single_bucket(hand), 15);
    }

    #[test]
    fn test_shogi_hand1024_single_bucket_formula() {
        let empty = Hand::EMPTY;
        assert_eq!(shogi_hand1024_single_bucket(empty), 0);

        let mut hand = Hand::EMPTY;
        hand.set_pawn(1);
        assert_eq!(shogi_hand1024_single_bucket(hand), 1);

        hand = Hand::EMPTY;
        hand.set_lance(1);
        assert_eq!(shogi_hand1024_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_knight(1);
        assert_eq!(shogi_hand1024_single_bucket(hand), 2);

        hand = Hand::EMPTY;
        hand.set_silver(1);
        assert_eq!(shogi_hand1024_single_bucket(hand), 4);

        hand = Hand::EMPTY;
        hand.set_gold(1);
        assert_eq!(shogi_hand1024_single_bucket(hand), 4);

        hand = Hand::EMPTY;
        hand.set_bishop(1);
        assert_eq!(shogi_hand1024_single_bucket(hand), 8);

        hand = Hand::EMPTY;
        hand.set_rook(1);
        assert_eq!(shogi_hand1024_single_bucket(hand), 16);

        hand = Hand::EMPTY;
        hand.set_pawn(18);
        hand.set_lance(4);
        hand.set_knight(4);
        hand.set_silver(4);
        hand.set_gold(4);
        hand.set_bishop(2);
        hand.set_rook(2);
        assert_eq!(shogi_hand1024_single_bucket(hand), 31);
    }

    #[test]
    fn test_shogi_sfnn_layerstack_bucket_counts() {
        assert_eq!(ShogiSfnnLayerStackBucketKind::Single.num_stacks(), 1);
        assert_eq!(ShogiSfnnLayerStackBucketKind::KingRank9.num_stacks(), 9);
        assert_eq!(ShogiSfnnLayerStackBucketKind::KingRank81.num_stacks(), 81);
        assert_eq!(ShogiSfnnLayerStackBucketKind::King9ZoneByKing9Zone.num_stacks(), 81);
        assert_eq!(ShogiSfnnLayerStackBucketKind::King13ZoneByKing13Zone.num_stacks(), 169);
        assert_eq!(ShogiSfnnLayerStackBucketKind::King21ByKing21.num_stacks(), 441);
        assert_eq!(ShogiSfnnLayerStackBucketKind::King29ByKing29.num_stacks(), 841);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand4.num_stacks(), 4);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand16.num_stacks(), 16);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand64.num_stacks(), 64);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand64z.num_stacks(), 64);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand64KingRank9.num_stacks(), 576);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand64KingRank81.num_stacks(), 5184);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand64King21ByKing21.num_stacks(), 28224);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand64King29ByKing29.num_stacks(), 53824);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand256.num_stacks(), 256);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand256KingRank9.num_stacks(), 2304);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand256KingRank81.num_stacks(), 20736);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand256King21ByKing21.num_stacks(), 112896);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand256King29ByKing29.num_stacks(), 215296);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand1024.num_stacks(), 1024);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand1024KingRank9.num_stacks(), 9216);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand1024KingRank81.num_stacks(), 82944);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand1024King21ByKing21.num_stacks(), 451584);
        assert_eq!(ShogiSfnnLayerStackBucketKind::Hand1024King29ByKing29.num_stacks(), 861184);
        assert_eq!(
            ShogiSfnnLayerStackBucketKind::new(
                ShogiSfnnHandBucketKind::None,
                ShogiSfnnKingBucketKind::None,
                ShogiSfnnProgressBucketKind::Progress8,
            )
            .num_stacks(),
            8
        );
        assert_eq!(
            ShogiSfnnLayerStackBucketKind::new(
                ShogiSfnnHandBucketKind::Hand256,
                ShogiSfnnKingBucketKind::KingRank9,
                ShogiSfnnProgressBucketKind::Progress16,
            )
            .num_stacks(),
            256 * 9 * 16
        );
        let startpos_like = psv_with_kings(Color::Black, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiSfnnLayerStackBucket::default().bucket(&startpos_like), 8);
        assert_eq!(ShogiSfnnLayerStackBucket::new(ShogiSfnnLayerStackBucketKind::Single).bucket(&startpos_like), 0);
        assert_eq!(
            ShogiSfnnLayerStackBucket::new(ShogiSfnnLayerStackBucketKind::KingRank81).bucket(&startpos_like),
            80
        );
        assert_eq!(
            ShogiSfnnLayerStackBucket::new(ShogiSfnnLayerStackBucketKind::King9ZoneByKing9Zone).bucket(&startpos_like),
            70
        );
        assert_eq!(
            ShogiSfnnLayerStackBucket::new(ShogiSfnnLayerStackBucketKind::King13ZoneByKing13Zone)
                .bucket(&startpos_like),
            154
        );
        assert_eq!(
            ShogiSfnnLayerStackBucket::new(ShogiSfnnLayerStackBucketKind::King29ByKing29).bucket(&startpos_like),
            720
        );
        assert_eq!(
            ShogiSfnnLayerStackBucket::new(ShogiSfnnLayerStackBucketKind::King21ByKing21).bucket(&startpos_like),
            352
        );
        assert_eq!(shogi_sfnn_progress_0_to_255_from_sum_q16(0), 128);
        assert_eq!(shogi_sfnn_progress_bucket_from_value(0, 8), 0);
        assert_eq!(shogi_sfnn_progress_bucket_from_value(128, 8), 4);
        assert_eq!(shogi_sfnn_progress_bucket_from_value(255, 8), 7);
        assert_eq!(shogi_sfnn_progress_bucket_from_value(255, 32), 31);
    }

    #[test]
    fn test_shogi_sfnn_progress_material_heuristic_shape() {
        let params = ShogiSfnnProgressQ16Params::material_heuristic();
        assert_eq!(params.weights_q16.len(), SHOGI_SFNN_PROGRESS_WEIGHT_COUNT);
        assert!(params.bias_q16 < 0);
        assert!(params.weights_q16.iter().any(|&w| w > 0));
    }

    #[test]
    fn test_shogi_sfnn_progress_material_heuristic_fast_sum_matches_weight_sum() {
        let params = ShogiSfnnProgressQ16Params::material_heuristic();
        let mut board =
            ShogiBoard { black_king_sq: Square::new(4, 8), white_king_sq: Square::new(4, 0), ..Default::default() };
        board.board[Square::new(2, 3).index()] = Piece::new(Color::Black, PieceType::Horse);
        board.board[Square::new(6, 5).index()] = Piece::new(Color::White, PieceType::Dragon);
        board.black_hand.set_pawn(3);
        board.black_hand.set_bishop(1);
        board.white_hand.set_lance(2);
        board.white_hand.set_rook(1);

        let generic = shogi_sfnn_progress_sum_q16_from_board(&board, &params);
        let fast = shogi_sfnn_material_heuristic_sum_q16_from_board(&board);
        assert_eq!(fast, generic);
    }

    #[test]
    fn test_shogi_king9_by_king9_bucket_formula() {
        let pos = psv_with_kings(Color::Black, Square::new(4, 2), Square::new(4, 7));
        assert_eq!(ShogiKingRankBucket::<81>.bucket(&pos), 2 * 9 + 1);
        assert_eq!(ShogiKingRankBucket::<9>.bucket(&pos), 0);

        let pos = psv_with_kings(Color::White, Square::new(4, 2), Square::new(4, 7));
        assert_eq!(ShogiKingRankBucket::<81>.bucket(&pos), 1 * 9 + 2);
        assert_eq!(ShogiKingRankBucket::<9>.bucket(&pos), 0);

        let startpos_like = psv_with_kings(Color::Black, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKingRankBucket::<81>.bucket(&startpos_like), 80);
        assert_eq!(ShogiKingRankBucket::<9>.bucket(&startpos_like), 8);
    }

    #[test]
    fn test_shogi_king9_zone_by_king9_zone_bucket_formula() {
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(0, 0)), 0);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(8, 2)), 0);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(0, 3)), 1);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(8, 5)), 1);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(0, 6)), 2);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(0, 7)), 3);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(8, 7)), 5);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(0, 8)), 6);
        assert_eq!(shogi_king9_zone_single_bucket(Square::new(8, 8)), 8);

        let startpos_like = psv_with_kings(Color::Black, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing9ZoneByKing9ZoneBucket::bucket_index(&startpos_like), 7 * 9 + 7);

        let startpos_like_white = psv_with_kings(Color::White, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing9ZoneByKing9ZoneBucket::bucket_index(&startpos_like_white), 7 * 9 + 7);
    }

    #[test]
    fn test_shogi_king13_zone_by_king13_zone_bucket_formula() {
        assert_eq!(shogi_king13_zone_single_bucket(Square::new(0, 0)), 0);
        assert_eq!(shogi_king13_zone_single_bucket(Square::new(8, 6)), 6);
        assert_eq!(shogi_king13_zone_single_bucket(Square::new(0, 7)), 7);
        assert_eq!(shogi_king13_zone_single_bucket(Square::new(8, 7)), 9);
        assert_eq!(shogi_king13_zone_single_bucket(Square::new(0, 8)), 10);
        assert_eq!(shogi_king13_zone_single_bucket(Square::new(8, 8)), 12);

        let startpos_like = psv_with_kings(Color::Black, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing13ZoneByKing13ZoneBucket::bucket_index(&startpos_like), 11 * 13 + 11);

        let startpos_like_white = psv_with_kings(Color::White, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing13ZoneByKing13ZoneBucket::bucket_index(&startpos_like_white), 11 * 13 + 11);
    }

    #[test]
    fn test_shogi_king29_by_king29_bucket_formula() {
        assert_eq!(shogi_king29_single_bucket(Square::new(0, 0)), 0);
        assert_eq!(shogi_king29_single_bucket(Square::new(8, 2)), 0);
        assert_eq!(shogi_king29_single_bucket(Square::new(0, 3)), 1);
        assert_eq!(shogi_king29_single_bucket(Square::new(8, 5)), 1);
        assert_eq!(shogi_king29_single_bucket(Square::new(0, 6)), 2);
        assert_eq!(shogi_king29_single_bucket(Square::new(8, 6)), 10);
        assert_eq!(shogi_king29_single_bucket(Square::new(0, 7)), 11);
        assert_eq!(shogi_king29_single_bucket(Square::new(8, 8)), 28);

        let startpos_like = psv_with_kings(Color::Black, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing29ByKing29Bucket::bucket_index(&startpos_like), 24 * 29 + 24);

        let startpos_like_white = psv_with_kings(Color::White, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing29ByKing29Bucket::bucket_index(&startpos_like_white), 24 * 29 + 24);

        let pos = psv_with_kings(Color::Black, Square::new(0, 6), Square::new(8, 2));
        assert_eq!(
            ShogiKing29ByKing29Bucket::bucket_index(&pos),
            shogi_king29_single_bucket(Square::new(0, 6)) * 29 + shogi_king29_single_bucket(Square::new(0, 6))
        );
    }

    #[test]
    fn test_shogi_king21_by_king21_bucket_formula() {
        assert_eq!(shogi_king21_single_bucket(Square::new(0, 0)), 0);
        assert_eq!(shogi_king21_single_bucket(Square::new(8, 2)), 0);
        assert_eq!(shogi_king21_single_bucket(Square::new(0, 3)), 1);
        assert_eq!(shogi_king21_single_bucket(Square::new(8, 5)), 1);
        assert_eq!(shogi_king21_single_bucket(Square::new(0, 6)), 2);
        assert_eq!(shogi_king21_single_bucket(Square::new(0, 7)), 3);
        assert_eq!(shogi_king21_single_bucket(Square::new(8, 7)), 11);
        assert_eq!(shogi_king21_single_bucket(Square::new(0, 8)), 12);
        assert_eq!(shogi_king21_single_bucket(Square::new(8, 8)), 20);

        let startpos_like = psv_with_kings(Color::Black, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing21ByKing21Bucket::bucket_index(&startpos_like), 16 * 21 + 16);

        let startpos_like_white = psv_with_kings(Color::White, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(ShogiKing21ByKing21Bucket::bucket_index(&startpos_like_white), 16 * 21 + 16);

        let pos = psv_with_kings(Color::Black, Square::new(0, 7), Square::new(8, 1));
        assert_eq!(
            ShogiKing21ByKing21Bucket::bucket_index(&pos),
            shogi_king21_single_bucket(Square::new(0, 7)) * 21 + shogi_king21_single_bucket(Square::new(0, 7))
        );
    }

    #[test]
    fn test_shogi_hand64_king9_by_king9_bucket_formula() {
        let pos = psv_with_kings(Color::Black, Square::new(4, 2), Square::new(4, 7));
        assert_eq!(
            ShogiHand64KingRank81Bucket::bucket_index(&pos),
            ShogiHand64Bucket::bucket_index(&pos) * 81 + ShogiKingRankBucket::<81>.bucket(&pos)
        );

        let startpos_like = psv_with_kings(Color::Black, Square::new(4, 8), Square::new(4, 0));
        assert_eq!(
            ShogiHand64KingRank81Bucket::bucket_index(&startpos_like),
            ShogiHand64Bucket::bucket_index(&startpos_like) * 81 + 80
        );
    }

    #[test]
    fn test_shogi_hand256_and_hand1024_king_bucket_formulas() {
        let pos = psv_with_kings(Color::White, Square::new(4, 1), Square::new(4, 8));
        assert_eq!(
            ShogiHand256KingRankBucket::bucket_index(&pos),
            ShogiHand256Bucket::bucket_index(&pos) * 9 + ShogiKingRankBucket::<9>.bucket(&pos)
        );
        assert_eq!(
            ShogiHand256KingRank81Bucket::bucket_index(&pos),
            ShogiHand256Bucket::bucket_index(&pos) * 81 + ShogiKingRankBucket::<81>.bucket(&pos)
        );
        assert_eq!(
            ShogiHand1024KingRankBucket::bucket_index(&pos),
            ShogiHand1024Bucket::bucket_index(&pos) * 9 + ShogiKingRankBucket::<9>.bucket(&pos)
        );
        assert_eq!(
            ShogiHand1024KingRank81Bucket::bucket_index(&pos),
            ShogiHand1024Bucket::bucket_index(&pos) * 81 + ShogiKingRankBucket::<81>.bucket(&pos)
        );
    }

    #[test]
    fn test_shogi_hand_king29_bucket_formulas() {
        let pos = psv_with_kings(Color::White, Square::new(2, 6), Square::new(6, 0));
        assert_eq!(
            ShogiHand64King21ByKing21Bucket::bucket_index(&pos),
            ShogiHand64Bucket::bucket_index(&pos) * 441 + ShogiKing21ByKing21Bucket::bucket_index(&pos)
        );
        assert_eq!(
            ShogiHand256King21ByKing21Bucket::bucket_index(&pos),
            ShogiHand256Bucket::bucket_index(&pos) * 441 + ShogiKing21ByKing21Bucket::bucket_index(&pos)
        );
        assert_eq!(
            ShogiHand1024King21ByKing21Bucket::bucket_index(&pos),
            ShogiHand1024Bucket::bucket_index(&pos) * 441 + ShogiKing21ByKing21Bucket::bucket_index(&pos)
        );
        assert_eq!(
            ShogiHand64King29ByKing29Bucket::bucket_index(&pos),
            ShogiHand64Bucket::bucket_index(&pos) * 841 + ShogiKing29ByKing29Bucket::bucket_index(&pos)
        );
        assert_eq!(
            ShogiHand256King29ByKing29Bucket::bucket_index(&pos),
            ShogiHand256Bucket::bucket_index(&pos) * 841 + ShogiKing29ByKing29Bucket::bucket_index(&pos)
        );
        assert_eq!(
            ShogiHand1024King29ByKing29Bucket::bucket_index(&pos),
            ShogiHand1024Bucket::bucket_index(&pos) * 841 + ShogiKing29ByKing29Bucket::bucket_index(&pos)
        );
    }

    #[test]
    fn test_shogi_ply_bucket9_default_bounds() {
        let bucket = ShogiPlyBucket9::default();
        assert_eq!(bucket.bucket(&psv_with_ply(0)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(30)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(31)), 1);
        assert_eq!(bucket.bucket(&psv_with_ply(138)), 7);
        assert_eq!(bucket.bucket(&psv_with_ply(139)), 8);
        assert_eq!(bucket.bucket(&psv_with_ply(400)), 8);
    }

    #[test]
    fn test_shogi_layerstack_bucket9_ply_mode() {
        let bucket = ShogiLayerStackBucket9::Ply9([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(bucket.bucket(&psv_with_ply(10)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(21)), 2);
        assert_eq!(bucket.bucket(&psv_with_ply(80)), 7);
        assert_eq!(bucket.bucket(&psv_with_ply(81)), 8);
    }

    #[test]
    fn test_shogi_progress_bucket8_range() {
        let bucket = ShogiProgressBucket8::default();
        for ply in [1u16, 30, 60, 100, 150, 220, 300] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8 bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_layerstack_bucket9_progress_mode_range() {
        let bucket = ShogiLayerStackBucket9::Progress8(ShogiProgressBucket8::default());
        for ply in [1u16, 40, 80, 120, 200, 400] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-in-9 bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_progress_bucket8_gikou_lite_range() {
        let bucket = ShogiProgressBucket8GikouLite::default();
        for ply in [1u16, 30, 60, 100, 150, 220, 300] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-gikou-lite bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_progress_bucket8_gikou_lite_prefix_matches_v1_features() {
        let psv = psv_with_ply(60);
        let v1 = ShogiProgressBucket8::extract_features(&psv);
        let v2 = ShogiProgressBucket8GikouLite::extract_features(&psv);
        assert_eq!(&v2[..SHOGI_PROGRESS8_NUM_FEATURES], &v1);
    }

    #[test]
    fn test_shogi_layerstack_bucket9_progress_gikou_lite_mode_range() {
        let bucket = ShogiLayerStackBucket9::Progress8GikouLite(ShogiProgressBucket8GikouLite::default());
        for ply in [1u16, 40, 80, 120, 200, 400] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-gikou-lite-in-9 bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_progress_kp_abs_default_is_neutral_on_invalid_position() {
        let bucket = ShogiProgressKPAbs;
        let psv = psv_with_ply(60);
        assert_eq!(bucket.progress(&psv), 0.5);
        assert_eq!(bucket.bucket(&psv), 4);
    }

    #[test]
    fn test_shogi_layerstack_bucket9_progress_kp_abs_mode_range() {
        let bucket = ShogiLayerStackBucket9::Progress8KPAbs(ShogiProgressKPAbs);
        for ply in [1u16, 40, 80, 120, 200, 400] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-kpabs-in-9 bucket must be in 0..=7, got {}", b);
        }
    }
}
