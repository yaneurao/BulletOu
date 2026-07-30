//! Scalar value-loss reference for the fixed-layout fast backend.
//!
//! This is intentionally CPU-only. It gives fixed-layout GPU loss kernels a
//! compact golden implementation before the training path grows
//! backward/optimizer kernels.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValueLossKind {
    SigmoidMse,
    WinRateModel { pow_exp: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarValueLossTrace {
    pub kind: ScalarValueLossKind,
    pub per_sample: Vec<f32>,
    pub mean_output_gradients: Vec<f32>,
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
    let mut mean_output_gradients = Vec::with_capacity(outputs.len());
    let mut weighted_sum = 0.0_f32;
    let inv_batch = 1.0_f32 / outputs.len() as f32;
    for ((&output, &target), &entry_weight) in outputs.iter().zip(targets).zip(entry_weights) {
        let (loss, output_gradient) = match kind {
            ScalarValueLossKind::SigmoidMse => {
                let prediction = sigmoid(output);
                let error = prediction - target;
                let loss = error * error;
                let gradient = 2.0 * error * prediction * (1.0 - prediction);
                (loss, gradient)
            }
            ScalarValueLossKind::WinRateModel { pow_exp } => win_rate_model_loss_and_gradient(output, target, pow_exp),
        };
        let weighted = entry_weight * loss;
        per_sample.push(weighted);
        mean_output_gradients.push(entry_weight * output_gradient * inv_batch);
        weighted_sum += weighted;
    }

    let mean = weighted_sum / outputs.len() as f32;
    Ok(ScalarValueLossTrace { kind, per_sample, mean_output_gradients, weighted_sum, mean })
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

fn win_rate_model_loss_and_gradient(output: f32, target: f32, pow_exp: f32) -> (f32, f32) {
    const NNUE2SCORE: f32 = 600.0;
    const IN_OFFSET: f32 = 270.0;
    const IN_SCALING: f32 = 340.0;

    let scorenet = output * NNUE2SCORE;
    let q = sigmoid((scorenet - IN_OFFSET) / IN_SCALING);
    let qm = sigmoid((-scorenet - IN_OFFSET) / IN_SCALING);
    let prediction = (1.0 + q - qm) * 0.5;
    let error = prediction - target;
    let abs_error = error.abs();
    let loss = abs_error.powf(pow_exp);
    let q_prime = q * (1.0 - q);
    let qm_prime = qm * (1.0 - qm);
    let prediction_gradient = 0.5 * (NNUE2SCORE / IN_SCALING) * (q_prime + qm_prime);
    let loss_gradient = pow_exp * error.signum() * abs_error.powf(pow_exp - 1.0);
    (loss, loss_gradient * prediction_gradient)
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
        assert_close_slice("mean_output_gradients", &trace.mean_output_gradients, &[0.008343695, 0.0, -0.01668739]);
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
    fn win_rate_model_pow_loss_matches_known_values() {
        let outputs = [-4.0, -1.25, 0.0, 1.5, 4.0];
        let targets = [0.0, 0.25, 0.5, 0.75, 1.0];
        let weights = [1.0, 0.0, 0.5, 2.0, 0.25];

        let trace =
            scalar_value_loss_trace(ScalarValueLossKind::WinRateModel { pow_exp: 2.5 }, &outputs, &targets, &weights)
                .unwrap();

        assert_eq!(trace.kind, ScalarValueLossKind::WinRateModel { pow_exp: 2.5 });
        assert_close_slice("per_sample", &trace.per_sample, &[4.4222793e-8, 0.0, 0.0, 0.022698434, 1.1055698e-8]);
        assert_close_slice(
            "mean_output_gradients",
            &trace.mean_output_gradients,
            &[3.8956034e-8, -0.0, 0.0, 0.008843387, -9.7390085e-9],
        );
        assert_close("weighted_sum", trace.weighted_sum, 0.022698488);
        assert_close("mean", trace.mean, 0.0045396974);
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

    #[cfg(feature = "cuda-cpp-backend")]
    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn cuda_cpp_scalar_loss_matches_cpu_reference() {
        let outputs = [-2.0, 0.0, 2.0];
        let targets = [0.0, 0.5, 1.0];
        let weights = [1.0, 0.5, 2.0];
        let cpu = scalar_value_loss_trace(ScalarValueLossKind::SigmoidMse, &outputs, &targets, &weights).unwrap();

        let gpu = bulletou_cuda_cpp::scalar_loss_host(
            0,
            bulletou_cuda_cpp::ScalarLossKind::SigmoidMse,
            1.0,
            bulletou_cuda_cpp::ScalarLossHostBatch { outputs: &outputs, targets: &targets, entry_weights: &weights },
        )
        .unwrap();

        assert_close_slice("per_sample", &gpu.per_sample, &cpu.per_sample);
        assert_close_slice("mean_output_gradients", &gpu.mean_output_gradients, &cpu.mean_output_gradients);
        assert_close("weighted_sum", gpu.weighted_sum, cpu.weighted_sum);
        assert_close("mean", gpu.mean, cpu.mean);
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
