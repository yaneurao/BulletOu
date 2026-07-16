//! Scalar NNUE forward reference for the fixed-layout fast backend.
//!
//! This is intentionally CPU-only. It gives cuda-oxide kernels a small golden
//! implementation to compare against before fused GPU training kernels are
//! introduced.

use std::{collections::BTreeMap, fmt};

use bullet_compiler::tensor::TValue;
use bullet_gpu::runtime::Gpu;
use bullet_trainer::model::Model;

use crate::value::FastBatchHost;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardShape {
    pub input_size: usize,
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

pub const NNUE_HALFKP_256X2_32_32: NnueForwardShape = NnueForwardShape { input_size: 125_388, l1: 256, l2: 32, l3: 32 };

#[derive(Debug, Clone, Copy)]
pub struct NnueForwardWeights<'a> {
    pub shape: NnueForwardShape,
    pub l0w: &'a [f32],
    pub l0b: &'a [f32],
    pub l1w: &'a [f32],
    pub l1b: &'a [f32],
    pub l2w: &'a [f32],
    pub l2b: &'a [f32],
    pub outw: &'a [f32],
    pub outb: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct NnueForwardOwnedWeights {
    pub shape: NnueForwardShape,
    pub l0w: Vec<f32>,
    pub l0b: Vec<f32>,
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub outw: Vec<f32>,
    pub outb: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FastNnueError {
    BatchLayout(String),
    MissingWeight { name: &'static str },
    WeightType { name: &'static str, expected: &'static str, actual: &'static str },
    WeightLength { name: &'static str, expected: usize, actual: usize },
}

impl fmt::Display for FastNnueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchLayout(message) => write!(f, "invalid fast batch layout: {message}"),
            Self::MissingWeight { name } => write!(f, "missing NNUE weight `{name}`"),
            Self::WeightType { name, expected, actual } => {
                write!(f, "weight type mismatch for {name}: expected {expected}, got {actual}")
            }
            Self::WeightLength { name, expected, actual } => {
                write!(f, "weight length mismatch for {name}: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for FastNnueError {}

impl NnueForwardOwnedWeights {
    pub fn from_weight_map(
        shape: NnueForwardShape,
        weights: &BTreeMap<String, Vec<f32>>,
    ) -> Result<Self, FastNnueError> {
        Self::from_weight_getter(shape, |name| weights.get(name).cloned().ok_or(FastNnueError::MissingWeight { name }))
    }

    pub fn from_model<G: Gpu>(shape: NnueForwardShape, model: &Model<G>) -> Result<Self, FastNnueError> {
        Self::from_weight_getter(shape, |name| match model.get_weights(name) {
            Some(TValue::F32(values)) => Ok(values),
            Some(TValue::I32(_)) => Err(FastNnueError::WeightType { name, expected: "f32", actual: "i32" }),
            None => Err(FastNnueError::MissingWeight { name }),
        })
    }

    pub fn from_weight_getter<F>(shape: NnueForwardShape, mut get: F) -> Result<Self, FastNnueError>
    where
        F: FnMut(&'static str) -> Result<Vec<f32>, FastNnueError>,
    {
        let weights = Self {
            shape,
            l0w: get("l0w")?,
            l0b: get("l0b")?,
            l1w: get("l1w")?,
            l1b: get("l1b")?,
            l2w: get("l2w")?,
            l2b: get("l2b")?,
            outw: get("outw")?,
            outb: get("outb")?,
        };

        weights.validate()?;
        Ok(weights)
    }

    pub fn as_borrowed(&self) -> NnueForwardWeights<'_> {
        NnueForwardWeights {
            shape: self.shape,
            l0w: &self.l0w,
            l0b: &self.l0b,
            l1w: &self.l1w,
            l1b: &self.l1b,
            l2w: &self.l2w,
            l2b: &self.l2b,
            outw: &self.outw,
            outb: &self.outb,
        }
    }

    pub fn validate(&self) -> Result<(), FastNnueError> {
        self.as_borrowed().validate()
    }

    pub fn forward_batch(&self, batch: &FastBatchHost) -> Result<Vec<f32>, FastNnueError> {
        self.as_borrowed().forward_batch(batch)
    }
}

impl<'a> NnueForwardWeights<'a> {
    pub fn validate(&self) -> Result<(), FastNnueError> {
        let shape = self.shape;
        expect_len("l0w", shape.input_size * shape.l1, self.l0w.len())?;
        expect_len("l0b", shape.l1, self.l0b.len())?;
        expect_len("l1w", shape.l1 * 2 * shape.l2, self.l1w.len())?;
        expect_len("l1b", shape.l2, self.l1b.len())?;
        expect_len("l2w", shape.l2 * shape.l3, self.l2w.len())?;
        expect_len("l2b", shape.l3, self.l2b.len())?;
        expect_len("outw", shape.l3, self.outw.len())?;
        expect_len("outb", 1, self.outb.len())?;
        Ok(())
    }

    pub fn forward_batch(&self, batch: &FastBatchHost) -> Result<Vec<f32>, FastNnueError> {
        self.validate()?;
        batch.validate().map_err(FastNnueError::BatchLayout)?;

        let shape = self.shape;
        let batch_size = batch.layout.batch_size;
        let max_active = batch.layout.max_active;
        let mut outputs = Vec::with_capacity(batch_size);

        let mut stm_l0 = vec![0.0; shape.l1];
        let mut nstm_l0 = vec![0.0; shape.l1];
        let mut combined = vec![0.0; shape.l1 * 2];
        let mut hidden1 = vec![0.0; shape.l2];
        let mut hidden2 = vec![0.0; shape.l3];

        for sample in 0..batch_size {
            affine_sparse_padded(
                self.l0w,
                self.l0b,
                shape.l1,
                shape.input_size,
                &batch.stm[sample * max_active..(sample + 1) * max_active],
                &mut stm_l0,
            );
            affine_sparse_padded(
                self.l0w,
                self.l0b,
                shape.l1,
                shape.input_size,
                &batch.nstm[sample * max_active..(sample + 1) * max_active],
                &mut nstm_l0,
            );

            crelu_in_place(&mut stm_l0);
            crelu_in_place(&mut nstm_l0);
            combined[..shape.l1].copy_from_slice(&stm_l0);
            combined[shape.l1..].copy_from_slice(&nstm_l0);

            affine_dense(self.l1w, self.l1b, &combined, shape.l2, &mut hidden1);
            crelu_in_place(&mut hidden1);
            affine_dense(self.l2w, self.l2b, &hidden1, shape.l3, &mut hidden2);
            crelu_in_place(&mut hidden2);

            let output = dot(self.outw, &hidden2) + self.outb[0];
            outputs.push(output);
        }

        Ok(outputs)
    }
}

fn expect_len(name: &'static str, expected: usize, actual: usize) -> Result<(), FastNnueError> {
    if expected == actual {
        Ok(())
    } else {
        Err(FastNnueError::WeightLength { name, expected, actual })
    }
}

fn affine_sparse_padded(weights: &[f32], bias: &[f32], rows: usize, cols: usize, active: &[i32], out: &mut [f32]) {
    out.copy_from_slice(&bias[..rows]);
    for &feature in active {
        if feature < 0 || feature as usize >= cols {
            continue;
        }
        let base = feature as usize * rows;
        for row in 0..rows {
            out[row] += weights[base + row];
        }
    }
}

fn affine_dense(weights: &[f32], bias: &[f32], input: &[f32], rows: usize, out: &mut [f32]) {
    out.copy_from_slice(&bias[..rows]);
    for (input_idx, &x) in input.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        let base = input_idx * rows;
        for row in 0..rows {
            out[row] += weights[base + row] * x;
        }
    }
}

fn crelu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.clamp(0.0, 1.0);
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&l, &r)| l * r).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{FastBatchHost, FastBatchLayout};
    use std::collections::BTreeMap;

    #[test]
    fn validates_weight_lengths() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let weights = tiny_weights(shape);

        weights.validate().unwrap();
    }

    #[test]
    fn scalar_forward_ignores_sparse_sentinel() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let weights = tiny_weights(shape);
        let batch = FastBatchHost {
            layout: FastBatchLayout { batch_size: 1, max_active: 3, output_size: 1, hand_count_dim: 0 },
            stm: vec![0, 1, -1],
            nstm: vec![2, -1, -1],
            buckets: vec![0],
            targets: vec![0.0],
            weights: vec![1.0],
            hand_count: None,
        };

        let outputs = weights.forward_batch(&batch).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!((outputs[0] - 1.208).abs() < 1.0e-6, "got {}", outputs[0]);
    }

    #[test]
    fn owned_weights_delegate_to_borrowed_forward() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let borrowed = tiny_weights(shape);
        let owned = NnueForwardOwnedWeights {
            shape,
            l0w: borrowed.l0w.to_vec(),
            l0b: borrowed.l0b.to_vec(),
            l1w: borrowed.l1w.to_vec(),
            l1b: borrowed.l1b.to_vec(),
            l2w: borrowed.l2w.to_vec(),
            l2b: borrowed.l2b.to_vec(),
            outw: borrowed.outw.to_vec(),
            outb: borrowed.outb.to_vec(),
        };
        let batch = FastBatchHost {
            layout: FastBatchLayout { batch_size: 1, max_active: 3, output_size: 1, hand_count_dim: 0 },
            stm: vec![0, 1, -1],
            nstm: vec![2, -1, -1],
            buckets: vec![0],
            targets: vec![0.0],
            weights: vec![1.0],
            hand_count: None,
        };

        let borrowed_outputs = borrowed.forward_batch(&batch).unwrap();
        let owned_outputs = owned.forward_batch(&batch).unwrap();

        assert_eq!(borrowed_outputs, owned_outputs);
    }

    #[test]
    fn owned_weights_can_be_loaded_from_weight_map() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let borrowed = tiny_weights(shape);
        let mut map = BTreeMap::new();
        map.insert("l0w".to_string(), borrowed.l0w.to_vec());
        map.insert("l0b".to_string(), borrowed.l0b.to_vec());
        map.insert("l1w".to_string(), borrowed.l1w.to_vec());
        map.insert("l1b".to_string(), borrowed.l1b.to_vec());
        map.insert("l2w".to_string(), borrowed.l2w.to_vec());
        map.insert("l2b".to_string(), borrowed.l2b.to_vec());
        map.insert("outw".to_string(), borrowed.outw.to_vec());
        map.insert("outb".to_string(), borrowed.outb.to_vec());

        let owned = NnueForwardOwnedWeights::from_weight_map(shape, &map).unwrap();

        assert_eq!(owned.as_borrowed().l0w, borrowed.l0w);
    }

    #[test]
    fn owned_weights_report_missing_weight() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let map = BTreeMap::new();

        let err = NnueForwardOwnedWeights::from_weight_map(shape, &map).unwrap_err();

        assert_eq!(err, FastNnueError::MissingWeight { name: "l0w" });
    }

    fn tiny_weights(shape: NnueForwardShape) -> NnueForwardWeights<'static> {
        assert_eq!(shape.input_size, 4);
        assert_eq!(shape.l1, 2);
        assert_eq!(shape.l2, 2);
        assert_eq!(shape.l3, 1);
        NnueForwardWeights {
            shape,
            l0w: &[
                0.2, 0.3, // feature 0
                0.4, -0.1, // feature 1
                -0.3, 0.5, // feature 2
                0.7, 0.9, // feature 3
            ],
            l0b: &[0.1, 0.2],
            l1w: &[
                0.5, -0.2, // combined 0
                0.1, 0.3, // combined 1
                -0.4, 0.2, // combined 2
                0.6, 0.1, // combined 3
            ],
            l1b: &[0.05, 0.1],
            l2w: &[
                0.7,  // hidden1 0
                -0.2, // hidden1 1
            ],
            l2b: &[0.2],
            outw: &[1.5],
            outb: &[0.05],
        }
    }
}
