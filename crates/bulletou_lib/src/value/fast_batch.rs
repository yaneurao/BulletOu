//! Fixed-layout batch representation for the future shogi NNUE/SFNN fast backend.
//!
//! The existing generic trainer feeds batches through `PreparedBatchHost`, which
//! is a name-keyed tensor map. That is flexible, but the fixed-layout GPU path
//! should pass compact batches directly to fused kernels. This module defines
//! that host-side layout without changing the current Bullet backend.

use std::{borrow::Cow, fmt};

use bullet_compiler::tensor::TValue;
use bullet_trainer::run::dataloader::PreparedBatchHost;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardComparison {
    pub len: usize,
    pub max_abs_diff: f32,
    pub max_abs_index: usize,
    pub mean_abs_diff: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FastReferenceError {
    InvalidTolerance(f32),
    Layout(String),
    MissingInput { name: &'static str },
    UnexpectedInput { name: String },
    TensorTypeMismatch { name: &'static str, expected: &'static str, actual: &'static str },
    TensorLengthMismatch { name: &'static str, expected: usize, actual: usize },
    I32ValueMismatch { name: &'static str, index: usize, expected: i32, actual: i32 },
    F32ValueMismatch { name: &'static str, index: usize, expected: f32, actual: f32, abs_diff: f32, tolerance: f32 },
    OutputLengthMismatch { reference: usize, candidate: usize },
    OutputValueMismatch { index: usize, reference: f32, candidate: f32, abs_diff: f32, tolerance: f32 },
}

impl fmt::Display for FastReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance(tolerance) => write!(f, "invalid tolerance: {tolerance}"),
            Self::Layout(message) => write!(f, "fast batch layout error: {message}"),
            Self::MissingInput { name } => write!(f, "missing input tensor: {name}"),
            Self::UnexpectedInput { name } => write!(f, "unexpected input tensor: {name}"),
            Self::TensorTypeMismatch { name, expected, actual } => {
                write!(f, "tensor type mismatch for {name}: expected {expected}, got {actual}")
            }
            Self::TensorLengthMismatch { name, expected, actual } => {
                write!(f, "tensor length mismatch for {name}: expected {expected}, got {actual}")
            }
            Self::I32ValueMismatch { name, index, expected, actual } => {
                write!(f, "i32 mismatch for {name}[{index}]: expected {expected}, got {actual}")
            }
            Self::F32ValueMismatch { name, index, expected, actual, abs_diff, tolerance } => write!(
                f,
                "f32 mismatch for {name}[{index}]: expected {expected}, got {actual}, abs_diff={abs_diff}, tolerance={tolerance}"
            ),
            Self::OutputLengthMismatch { reference, candidate } => {
                write!(f, "output length mismatch: reference={reference}, candidate={candidate}")
            }
            Self::OutputValueMismatch { index, reference, candidate, abs_diff, tolerance } => write!(
                f,
                "output mismatch at {index}: reference={reference}, candidate={candidate}, abs_diff={abs_diff}, tolerance={tolerance}"
            ),
        }
    }
}

impl std::error::Error for FastReferenceError {}

impl FastBatchHost {
    pub fn validate(&self) -> Result<(), String> {
        let layout = self.layout;
        if self.stm.len() != layout.sparse_len() {
            return Err(format!("stm length mismatch: got {}, expected {}", self.stm.len(), layout.sparse_len(),));
        }
        if self.nstm.len() != layout.sparse_len() {
            return Err(format!("nstm length mismatch: got {}, expected {}", self.nstm.len(), layout.sparse_len(),));
        }
        if self.buckets.len() != layout.batch_size {
            return Err(
                format!("buckets length mismatch: got {}, expected {}", self.buckets.len(), layout.batch_size,),
            );
        }
        if self.targets.len() != layout.target_len() {
            return Err(format!(
                "targets length mismatch: got {}, expected {}",
                self.targets.len(),
                layout.target_len(),
            ));
        }
        if self.weights.len() != layout.batch_size {
            return Err(
                format!("weights length mismatch: got {}, expected {}", self.weights.len(), layout.batch_size,),
            );
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

    pub fn into_prepared_batch_host(self) -> PreparedBatchHost {
        let mut inputs = Vec::with_capacity(5 + usize::from(self.hand_count.is_some()));
        inputs.push((Cow::Borrowed("stm"), TValue::I32(self.stm)));
        inputs.push((Cow::Borrowed("nstm"), TValue::I32(self.nstm)));
        inputs.push((Cow::Borrowed("buckets"), TValue::I32(self.buckets)));
        inputs.push((Cow::Borrowed("targets"), TValue::F32(self.targets)));
        inputs.push((Cow::Borrowed("entry_weights"), TValue::F32(self.weights)));

        if let Some(hand_count) = self.hand_count {
            inputs.push((Cow::Borrowed("hand_count"), TValue::F32(hand_count)));
        }

        PreparedBatchHost { batch_size: self.layout.batch_size, inputs }
    }

    pub fn stm_sample(&self, sample: usize) -> Option<&[i32]> {
        self.sparse_sample(&self.stm, sample)
    }

    pub fn nstm_sample(&self, sample: usize) -> Option<&[i32]> {
        self.sparse_sample(&self.nstm, sample)
    }

    fn sparse_sample<'a>(&self, sparse: &'a [i32], sample: usize) -> Option<&'a [i32]> {
        if sample >= self.layout.batch_size {
            return None;
        }

        let start = sample * self.layout.max_active;
        let end = start + self.layout.max_active;
        sparse.get(start..end)
    }

    pub fn compare_prepared_batch(
        &self,
        prepared: &PreparedBatchHost,
        tolerance: f32,
    ) -> Result<(), FastReferenceError> {
        validate_tolerance(tolerance)?;
        self.validate().map_err(FastReferenceError::Layout)?;

        if prepared.batch_size != self.layout.batch_size {
            return Err(FastReferenceError::TensorLengthMismatch {
                name: "batch_size",
                expected: self.layout.batch_size,
                actual: prepared.batch_size,
            });
        }

        self.reject_unexpected_inputs(prepared)?;
        compare_i32_tensor(prepared, "stm", &self.stm)?;
        compare_i32_tensor(prepared, "nstm", &self.nstm)?;
        compare_i32_tensor(prepared, "buckets", &self.buckets)?;
        compare_f32_tensor(prepared, "targets", &self.targets, tolerance)?;
        compare_f32_tensor(prepared, "entry_weights", &self.weights, tolerance)?;

        if let Some(hand_count) = &self.hand_count {
            compare_f32_tensor(prepared, "hand_count", hand_count, tolerance)?;
        }

        Ok(())
    }

    fn reject_unexpected_inputs(&self, prepared: &PreparedBatchHost) -> Result<(), FastReferenceError> {
        let expects_hand_count = self.hand_count.is_some();
        for (name, _) in &prepared.inputs {
            let name = name.as_ref();
            let expected = matches!(name, "stm" | "nstm" | "buckets" | "targets" | "entry_weights")
                || (expects_hand_count && name == "hand_count");
            if !expected {
                return Err(FastReferenceError::UnexpectedInput { name: name.to_string() });
            }
        }
        Ok(())
    }
}

pub fn active_feature_indices(active: &[i32], feature_count: usize) -> impl Iterator<Item = usize> + '_ {
    active.iter().filter_map(move |&feature| {
        if feature < 0 || feature as usize >= feature_count { None } else { Some(feature as usize) }
    })
}

pub fn compare_forward_outputs(
    reference: &[f32],
    candidate: &[f32],
    tolerance: f32,
) -> Result<ForwardComparison, FastReferenceError> {
    validate_tolerance(tolerance)?;

    if reference.len() != candidate.len() {
        return Err(FastReferenceError::OutputLengthMismatch {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }

    let mut max_abs_diff = 0.0_f32;
    let mut max_abs_index = 0usize;
    let mut sum_abs_diff = 0.0_f32;

    for (index, (&reference, &candidate)) in reference.iter().zip(candidate).enumerate() {
        let abs_diff = (reference - candidate).abs();
        sum_abs_diff += abs_diff;
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            max_abs_index = index;
        }
        if abs_diff > tolerance {
            return Err(FastReferenceError::OutputValueMismatch { index, reference, candidate, abs_diff, tolerance });
        }
    }

    let mean_abs_diff = if reference.is_empty() { 0.0 } else { sum_abs_diff / reference.len() as f32 };

    Ok(ForwardComparison { len: reference.len(), max_abs_diff, max_abs_index, mean_abs_diff })
}

fn validate_tolerance(tolerance: f32) -> Result<(), FastReferenceError> {
    if tolerance.is_finite() && tolerance >= 0.0 {
        Ok(())
    } else {
        Err(FastReferenceError::InvalidTolerance(tolerance))
    }
}

fn input<'a>(prepared: &'a PreparedBatchHost, name: &'static str) -> Result<&'a TValue, FastReferenceError> {
    prepared
        .inputs
        .iter()
        .find_map(|(actual_name, value)| (actual_name.as_ref() == name).then_some(value))
        .ok_or(FastReferenceError::MissingInput { name })
}

fn compare_i32_tensor(
    prepared: &PreparedBatchHost,
    name: &'static str,
    expected: &[i32],
) -> Result<(), FastReferenceError> {
    let actual = match input(prepared, name)? {
        TValue::I32(actual) => actual,
        TValue::F32(_) => {
            return Err(FastReferenceError::TensorTypeMismatch { name, expected: "i32", actual: "f32" });
        }
    };

    if actual.len() != expected.len() {
        return Err(FastReferenceError::TensorLengthMismatch { name, expected: expected.len(), actual: actual.len() });
    }

    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if expected != actual {
            return Err(FastReferenceError::I32ValueMismatch { name, index, expected, actual });
        }
    }

    Ok(())
}

fn compare_f32_tensor(
    prepared: &PreparedBatchHost,
    name: &'static str,
    expected: &[f32],
    tolerance: f32,
) -> Result<(), FastReferenceError> {
    let actual = match input(prepared, name)? {
        TValue::F32(actual) => actual,
        TValue::I32(_) => {
            return Err(FastReferenceError::TensorTypeMismatch { name, expected: "f32", actual: "i32" });
        }
    };

    if actual.len() != expected.len() {
        return Err(FastReferenceError::TensorLengthMismatch { name, expected: expected.len(), actual: actual.len() });
    }

    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let abs_diff = (expected - actual).abs();
        if abs_diff > tolerance {
            return Err(FastReferenceError::F32ValueMismatch { name, index, expected, actual, abs_diff, tolerance });
        }
    }

    Ok(())
}

impl<I, O> From<PreparedData<I, O>> for FastBatchHost
where
    I: SparseInputType,
{
    fn from(prepared: PreparedData<I, O>) -> Self {
        let batch_size = prepared.batch_size;
        let max_active = prepared.input_getter.max_active();
        let output_size = if batch_size == 0 { 0 } else { prepared.targets.len() / batch_size };
        let hand_count_dim = if batch_size == 0 {
            0
        } else {
            prepared.hand_count.as_ref().map(|v| v.len() / batch_size).unwrap_or(0)
        };

        Self {
            layout: FastBatchLayout { batch_size, max_active, output_size, hand_count_dim },
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
        let layout = FastBatchLayout { batch_size: 8, max_active: 32, output_size: 3, hand_count_dim: 14 };

        assert_eq!(layout.sparse_len(), 256);
        assert_eq!(layout.target_len(), 24);
        assert_eq!(layout.hand_count_len(), 112);
    }

    #[test]
    fn validate_accepts_matching_buffers() {
        let layout = FastBatchLayout { batch_size: 2, max_active: 3, output_size: 1, hand_count_dim: 0 };
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
        let layout = FastBatchLayout { batch_size: 2, max_active: 3, output_size: 1, hand_count_dim: 0 };
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

    #[test]
    fn prepared_batch_conversion_matches_fast_layout() {
        let layout = FastBatchLayout { batch_size: 2, max_active: 3, output_size: 1, hand_count_dim: 2 };
        let batch = FastBatchHost {
            layout,
            stm: vec![1, 2, 0, 3, 4, 0],
            nstm: vec![5, 6, 0, 7, 8, 0],
            buckets: vec![0, 1],
            targets: vec![0.25, 0.75],
            weights: vec![1.0, 0.5],
            hand_count: Some(vec![0.0, 1.0, 2.0, 3.0]),
        };

        let prepared = batch.clone().into_prepared_batch_host();

        batch.compare_prepared_batch(&prepared, 0.0).unwrap();
    }

    #[test]
    fn sparse_samples_are_sliced_per_position() {
        let layout = FastBatchLayout { batch_size: 2, max_active: 3, output_size: 1, hand_count_dim: 0 };
        let batch = FastBatchHost {
            layout,
            stm: vec![1, 2, -1, 3, 4, -1],
            nstm: vec![5, -1, -1, 6, 7, -1],
            buckets: vec![0, 0],
            targets: vec![0.0; layout.target_len()],
            weights: vec![1.0; layout.batch_size],
            hand_count: None,
        };

        assert_eq!(batch.stm_sample(0), Some([1, 2, -1].as_slice()));
        assert_eq!(batch.stm_sample(1), Some([3, 4, -1].as_slice()));
        assert_eq!(batch.nstm_sample(1), Some([6, 7, -1].as_slice()));
        assert_eq!(batch.stm_sample(2), None);
    }

    #[test]
    fn active_feature_iterator_ignores_sentinel_and_out_of_range() {
        let active: Vec<_> = active_feature_indices(&[0, 2, -1, 99, 3], 4).collect();

        assert_eq!(active, vec![0, 2, 3]);
    }

    #[test]
    fn prepared_batch_comparison_reports_mismatch() {
        let layout = FastBatchLayout { batch_size: 1, max_active: 2, output_size: 1, hand_count_dim: 0 };
        let batch = FastBatchHost {
            layout,
            stm: vec![1, 0],
            nstm: vec![2, 0],
            buckets: vec![0],
            targets: vec![0.25],
            weights: vec![1.0],
            hand_count: None,
        };
        let mut prepared = batch.clone().into_prepared_batch_host();
        let (_, TValue::F32(targets)) =
            prepared.inputs.iter_mut().find(|(name, _)| name.as_ref() == "targets").unwrap()
        else {
            panic!("targets should be f32");
        };
        targets[0] = 0.3;

        let err = batch.compare_prepared_batch(&prepared, 0.001).unwrap_err();

        assert!(matches!(err, FastReferenceError::F32ValueMismatch { name: "targets", index: 0, .. }));
    }

    #[test]
    fn forward_output_comparison_summarises_diffs() {
        let reference = [1.0, 2.0, 3.0];
        let candidate = [1.0, 2.002, 2.999];

        let comparison = compare_forward_outputs(&reference, &candidate, 0.01).unwrap();

        assert_eq!(comparison.len, 3);
        assert_eq!(comparison.max_abs_index, 1);
        assert!(comparison.max_abs_diff > 0.0019);
        assert!(comparison.mean_abs_diff > 0.0009);
    }

    #[test]
    fn forward_output_comparison_rejects_large_diff() {
        let err = compare_forward_outputs(&[0.0], &[0.5], 0.1).unwrap_err();

        assert!(matches!(err, FastReferenceError::OutputValueMismatch { index: 0, .. }));
    }
}
