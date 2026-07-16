//! Scalar value-loss reference for the fixed-layout fast backend.
//!
//! This is intentionally CPU-only. It gives cuda-oxide loss kernels a compact
//! golden implementation before the training path grows backward/optimizer
//! kernels.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarValueLossKind {
    SigmoidMse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarValueLossTrace {
    pub kind: ScalarValueLossKind,
    pub per_sample: Vec<f32>,
    pub weighted_sum: f32,
    pub mean: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FastLossError {
    LengthMismatch { name: &'static str, expected: usize, actual: usize },
    EmptyBatch,
}

impl fmt::Display for FastLossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { name, expected, actual } => {
                write!(f, "{name} length mismatch: expected {expected}, got {actual}")
            }
            Self::EmptyBatch => write!(f, "loss batch must contain at least one sample"),
        }
    }
}

impl std::error::Error for FastLossError {}

pub fn scalar_value_loss_trace(
    kind: ScalarValueLossKind,
    outputs: &[f32],
    targets: &[f32],
    entry_weights: &[f32],
) -> Result<ScalarValueLossTrace, FastLossError> {
    if outputs.is_empty() {
        return Err(FastLossError::EmptyBatch);
    }
    expect_len("targets", outputs.len(), targets.len())?;
    expect_len("entry_weights", outputs.len(), entry_weights.len())?;

    let mut per_sample = Vec::with_capacity(outputs.len());
    let mut weighted_sum = 0.0_f32;
    for ((&output, &target), &entry_weight) in outputs.iter().zip(targets).zip(entry_weights) {
        let loss = match kind {
            ScalarValueLossKind::SigmoidMse => {
                let prediction = sigmoid(output);
                let error = prediction - target;
                error * error
            }
        };
        let weighted = entry_weight * loss;
        per_sample.push(weighted);
        weighted_sum += weighted;
    }

    let mean = weighted_sum / outputs.len() as f32;
    Ok(ScalarValueLossTrace { kind, per_sample, weighted_sum, mean })
}

fn expect_len(name: &'static str, expected: usize, actual: usize) -> Result<(), FastLossError> {
    if expected == actual {
        Ok(())
    } else {
        Err(FastLossError::LengthMismatch { name, expected, actual })
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_mse_loss_matches_known_values() {
        let outputs = [-2.0, 0.0, 2.0];
        let targets = [0.0, 0.5, 1.0];
        let weights = [1.0, 0.5, 2.0];

        let trace = scalar_value_loss_trace(ScalarValueLossKind::SigmoidMse, &outputs, &targets, &weights).unwrap();

        assert_eq!(trace.kind, ScalarValueLossKind::SigmoidMse);
        assert_close_slice("per_sample", &trace.per_sample, &[0.014209336, 0.0, 0.028418668]);
        assert_close("weighted_sum", trace.weighted_sum, 0.042628005);
        assert_close("mean", trace.mean, 0.014209335);
    }

    #[test]
    fn zero_entry_weight_masks_sample_loss() {
        let outputs = [10.0, -10.0];
        let targets = [0.0, 1.0];
        let weights = [0.0, 1.0];

        let trace = scalar_value_loss_trace(ScalarValueLossKind::SigmoidMse, &outputs, &targets, &weights).unwrap();

        assert_eq!(trace.per_sample[0], 0.0);
        assert!(trace.per_sample[1] > 0.999);
    }

    #[test]
    fn reports_length_mismatch() {
        let err = scalar_value_loss_trace(ScalarValueLossKind::SigmoidMse, &[0.0], &[], &[1.0]).unwrap_err();

        assert_eq!(err, FastLossError::LengthMismatch { name: "targets", expected: 1, actual: 0 });
    }

    #[test]
    fn rejects_empty_batch() {
        let err = scalar_value_loss_trace(ScalarValueLossKind::SigmoidMse, &[], &[], &[]).unwrap_err();

        assert_eq!(err, FastLossError::EmptyBatch);
    }

    fn assert_close_slice(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{name}[{idx}]"), actual, expected);
        }
    }

    fn assert_close(name: &str, actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-6, "{name}: expected {expected}, got {actual}");
    }
}
