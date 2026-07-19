//! ShogiShardKp - shardKP experimental K+P input.
//!
//! Target architecture: `NNUE_shardkp_c256_s128x64_f6_16_16`.
//! - common accumulator: 256
//! - shard accumulator: 128 x 64
//! - each K+P feature is expanded to one common connection plus six shard
//!   connection IDs.
//!
//! This is the experimental dense-L0 version. BulletOu's sparse matmul adds
//! each active feature to every accumulator row, so the connection IDs below do
//! not yet enforce row-sparse shard routing. They preserve the branch's feature
//! indexing convention so a future row-sparse transformer can reuse the same
//! `shard_for_feature()` / `connection_index()` rules.

use super::{
    SparseInputType,
    shogi_kp::{KP_DIMENSIONS, KP_MAX_ACTIVE, ShogiKp},
};
use crate::shogi::PackedSfenValue;

pub const SHARDKP_COMMON_DIMENSIONS: usize = 256;
pub const SHARDKP_SHARD_DIMENSIONS: usize = 128;
pub const SHARDKP_SHARD_COUNT: usize = 64;
pub const SHARDKP_FANOUT: usize = 6;
pub const SHARDKP_CONNECTIONS_PER_FEATURE: usize = 1 + SHARDKP_FANOUT;
pub const SHARDKP_TOTAL_L1: usize = SHARDKP_COMMON_DIMENSIONS + SHARDKP_SHARD_DIMENSIONS * SHARDKP_SHARD_COUNT;

/// Input dimension after expanding each K+P feature into common/shard
/// connection IDs.
pub const SHARDKP_DIMENSIONS: usize = KP_DIMENSIONS * SHARDKP_CONNECTIONS_PER_FEATURE;

/// K+P max-active 40 expanded to common + six shard connection IDs.
pub const SHARDKP_MAX_ACTIVE: usize = KP_MAX_ACTIVE * SHARDKP_CONNECTIONS_PER_FEATURE;

/// BulletOu-internal experimental hash. This is not a standard YaneuraOu
/// feature hash unless the engine side also implements shardKP.
pub const FEATURE_HASH_SHARDKP: u32 = 0x534B5031; // "SKP1"

/// K+P expanded to common + shard connection IDs, dual perspective.
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

/// `connection == 0` is the common accumulator. For `connection >= 1`, return
/// the target shard index used by the intended row-sparse transformer.
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
