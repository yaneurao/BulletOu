//! ShogiShardKp - shardKP 実験用の K+P 入力。
//!
//! 設計上の目標は `NNUE_shardkp_c256_s128x64_f6_16_16`:
//! - common accumulator: 256
//! - shard accumulator: 128 x 64
//! - 1 つの K+P 特徴は common と 6 個の shard に接続する
//!
//! 現時点の BulletOu の sparse matmul は「発火した特徴が L1 全行に加算される」
//! 演算なので、ここではまず学習実験を始めるため、K+P 特徴を
//! common/shard 接続 ID に展開する。最終的に row-sparse な推論器へ移す場合は、
//! `shard_for_feature()` と `connection_index()` の規約を使って、接続先 shard
//! の行だけを更新する専用 FeatureTransformer を実装する。

use super::{
    shogi_kp::{ShogiKp, KP_DIMENSIONS, KP_MAX_ACTIVE},
    SparseInputType,
};
use crate::shogi::PackedSfenValue;

pub const SHARDKP_COMMON_DIMENSIONS: usize = 256;
pub const SHARDKP_SHARD_DIMENSIONS: usize = 128;
pub const SHARDKP_SHARD_COUNT: usize = 64;
pub const SHARDKP_FANOUT: usize = 6;
pub const SHARDKP_CONNECTIONS_PER_FEATURE: usize = 1 + SHARDKP_FANOUT;
pub const SHARDKP_TOTAL_L1: usize = SHARDKP_COMMON_DIMENSIONS + SHARDKP_SHARD_DIMENSIONS * SHARDKP_SHARD_COUNT;

/// K+P 特徴を common/shard 接続 ID に展開した入力次元。
pub const SHARDKP_DIMENSIONS: usize = KP_DIMENSIONS * SHARDKP_CONNECTIONS_PER_FEATURE;

/// K+P の最大 active 数 40 を、common + 6 shard 接続へ展開する。
pub const SHARDKP_MAX_ACTIVE: usize = KP_MAX_ACTIVE * SHARDKP_CONNECTIONS_PER_FEATURE;

/// YaneuraOu互換 NNUE ではないため、この hash は BulletOu 内部実験用。
pub const FEATURE_HASH_SHARDKP: u32 = 0x534B5031; // "SKP1"

/// ShogiShardKp - K+P を common + 6 shard 接続に展開する dual-perspective 入力。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiShardKp;

impl SparseInputType for ShogiShardKp {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        SHARDKP_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        SHARDKP_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.map_features_split(pos, |stm, nstm| {
            if let (Some(stm), Some(nstm)) = (stm, nstm) {
                f(stm, nstm);
            }
        });
    }

    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        ShogiKp.map_features_split(pos, |stm, nstm| {
            for connection in 0..SHARDKP_CONNECTIONS_PER_FEATURE {
                f(stm.map(|idx| connection_index(idx, connection)), nstm.map(|idx| connection_index(idx, connection)));
            }
        });
    }

    fn shorthand(&self) -> String {
        "shogi-shardkp-c256-s128x64-f6".to_string()
    }

    fn description(&self) -> String {
        format!(
            "Shogi shardKP prototype: K+P expanded to common {} + shard {}x{} fanout {}",
            SHARDKP_COMMON_DIMENSIONS, SHARDKP_SHARD_DIMENSIONS, SHARDKP_SHARD_COUNT, SHARDKP_FANOUT
        )
    }
}

#[inline]
pub const fn connection_index(kp_feature: usize, connection: usize) -> usize {
    kp_feature * SHARDKP_CONNECTIONS_PER_FEATURE + connection
}

/// `connection == 0` は common。`connection >= 1` のとき接続先 shard を返す。
#[inline]
pub fn shard_for_feature(kp_feature: usize, connection: usize) -> Option<usize> {
    if connection == 0 || connection >= SHARDKP_CONNECTIONS_PER_FEATURE {
        return None;
    }

    let mut x = (kp_feature as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= (connection as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    Some((x as usize) & (SHARDKP_SHARD_COUNT - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims() {
        let input = ShogiShardKp;
        assert_eq!(SHARDKP_TOTAL_L1, 8448);
        assert_eq!(input.num_inputs(), 1710 * 7);
        assert_eq!(input.max_active(), 40 * 7);
    }

    #[test]
    fn connection_indices_are_contiguous() {
        assert_eq!(connection_index(0, 0), 0);
        assert_eq!(connection_index(0, 6), 6);
        assert_eq!(connection_index(1, 0), 7);
        assert_eq!(connection_index(KP_DIMENSIONS - 1, 6), SHARDKP_DIMENSIONS - 1);
    }

    #[test]
    fn shard_hash_is_in_range() {
        for feature in [0, 1, 80, 1547, 1548, 1709] {
            assert_eq!(shard_for_feature(feature, 0), None);
            for connection in 1..=SHARDKP_FANOUT {
                let shard = shard_for_feature(feature, connection).unwrap();
                assert!(shard < SHARDKP_SHARD_COUNT);
            }
        }
    }
}
