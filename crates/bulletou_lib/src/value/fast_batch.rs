//! Fixed-layout batch representation for the future shogi NNUE/SFNN fast backend.
//!
//! The existing generic trainer feeds batches through `PreparedBatchHost`, which
//! is a name-keyed tensor map. That is flexible, but the cuda-oxide path should
//! pass a compact fixed layout directly to fused kernels. This module defines
//! that host-side layout without changing the current Bullet backend.

use crate::{game::inputs::SparseInputType, value::loader::PreparedData};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastBatchLayout {
    pub batch_size: usize,
    pub max_active: usize,
    pub output_size: usize,
    pub hand_count_dim: usize,
}

impl FastBatchLayout {
    pub fn sparse_len(self) -> usize {
        self.batch_size.saturating_mul(self.max_active)
    }

    pub fn target_len(self) -> usize {
        self.batch_size.saturating_mul(self.output_size)
    }

    pub fn hand_count_len(self) -> usize {
        self.batch_size.saturating_mul(self.hand_count_dim)
    }
}

#[derive(Debug, Clone)]
pub struct FastBatchHost {
    pub layout: FastBatchLayout,
    pub stm: Vec<i32>,
    pub nstm: Vec<i32>,
    pub buckets: Vec<i32>,
    pub targets: Vec<f32>,
    pub weights: Vec<f32>,
    pub hand_count: Option<Vec<f32>>,
}

impl FastBatchHost {
    pub fn validate(&self) -> Result<(), String> {
        let layout = self.layout;
        if self.stm.len() != layout.sparse_len() {
            return Err(format!(
                "stm length mismatch: got {}, expected {}",
                self.stm.len(),
                layout.sparse_len(),
            ));
        }
        if self.nstm.len() != layout.sparse_len() {
            return Err(format!(
                "nstm length mismatch: got {}, expected {}",
                self.nstm.len(),
                layout.sparse_len(),
            ));
        }
        if self.buckets.len() != layout.batch_size {
            return Err(format!(
                "buckets length mismatch: got {}, expected {}",
                self.buckets.len(),
                layout.batch_size,
            ));
        }
        if self.targets.len() != layout.target_len() {
            return Err(format!(
                "targets length mismatch: got {}, expected {}",
                self.targets.len(),
                layout.target_len(),
            ));
        }
        if self.weights.len() != layout.batch_size {
            return Err(format!(
                "weights length mismatch: got {}, expected {}",
                self.weights.len(),
                layout.batch_size,
            ));
        }
        match (&self.hand_count, layout.hand_count_dim) {
            (Some(hand_count), dim) if dim > 0 && hand_count.len() != layout.hand_count_len() => Err(format!(
                "hand_count length mismatch: got {}, expected {}",
                hand_count.len(),
                layout.hand_count_len(),
            )),
            (Some(_), 0) => Err("hand_count buffer exists but hand_count_dim is 0".to_string()),
            (None, dim) if dim > 0 => Err(format!("hand_count_dim is {dim} but hand_count buffer is missing")),
            _ => Ok(()),
        }
    }
}

impl<I, O> From<PreparedData<I, O>> for FastBatchHost
where
    I: SparseInputType,
{
    fn from(prepared: PreparedData<I, O>) -> Self {
        let batch_size = prepared.batch_size;
        let max_active = prepared.input_getter.max_active();
        let output_size = if batch_size == 0 {
            0
        } else {
            prepared.targets.len() / batch_size
        };
        let hand_count_dim = if batch_size == 0 {
            0
        } else {
            prepared.hand_count.as_ref().map(|v| v.len() / batch_size).unwrap_or(0)
        };

        Self {
            layout: FastBatchLayout {
                batch_size,
                max_active,
                output_size,
                hand_count_dim,
            },
            stm: prepared.stm,
            nstm: prepared.nstm,
            buckets: prepared.buckets,
            targets: prepared.targets,
            weights: prepared.weights,
            hand_count: prepared.hand_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_lengths_are_derived_from_shape() {
        let layout = FastBatchLayout {
            batch_size: 8,
            max_active: 32,
            output_size: 3,
            hand_count_dim: 14,
        };

        assert_eq!(layout.sparse_len(), 256);
        assert_eq!(layout.target_len(), 24);
        assert_eq!(layout.hand_count_len(), 112);
    }

    #[test]
    fn validate_accepts_matching_buffers() {
        let layout = FastBatchLayout {
            batch_size: 2,
            max_active: 3,
            output_size: 1,
            hand_count_dim: 0,
        };
        let batch = FastBatchHost {
            layout,
            stm: vec![0; layout.sparse_len()],
            nstm: vec![0; layout.sparse_len()],
            buckets: vec![0; layout.batch_size],
            targets: vec![0.0; layout.target_len()],
            weights: vec![1.0; layout.batch_size],
            hand_count: None,
        };

        assert!(batch.validate().is_ok());
    }

    #[test]
    fn validate_rejects_shape_mismatch() {
        let layout = FastBatchLayout {
            batch_size: 2,
            max_active: 3,
            output_size: 1,
            hand_count_dim: 0,
        };
        let batch = FastBatchHost {
            layout,
            stm: vec![0; layout.sparse_len() - 1],
            nstm: vec![0; layout.sparse_len()],
            buckets: vec![0; layout.batch_size],
            targets: vec![0.0; layout.target_len()],
            weights: vec![1.0; layout.batch_size],
            hand_count: None,
        };

        let err = batch.validate().unwrap_err();
        assert!(err.contains("stm length mismatch"));
    }
}
