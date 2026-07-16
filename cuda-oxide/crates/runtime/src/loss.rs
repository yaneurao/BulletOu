//! Fixed-layout scalar value-loss workspace for cuda-oxide kernels.

#[cfg(feature = "cuda")]
use crate::{CudaStream, DeviceBuffer, Result};

pub const LOSS_SIGMOID_MSE_REDUCE_KERNEL: &str = "loss_sigmoid_mse_reduce";
pub const LOSS_NNUE_PYTORCH_WRM_REDUCE_KERNEL: &str = "loss_nnue_pytorch_wrm_reduce";
pub const LOSS_KERNEL_NAMES: [&str; 2] = [LOSS_SIGMOID_MSE_REDUCE_KERNEL, LOSS_NNUE_PYTORCH_WRM_REDUCE_KERNEL];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarLossKind {
    SigmoidMse,
    NnuePytorchWrm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarLossLayout {
    pub batch_size: usize,
}

impl ScalarLossLayout {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    pub fn validate(self) -> std::result::Result<(), LossLayoutError> {
        if self.batch_size == 0 {
            Err(LossLayoutError::EmptyBatch)
        } else {
            Ok(())
        }
    }

    pub fn per_sample_len(self) -> usize {
        self.batch_size
    }

    pub fn reduced_len(self) -> usize {
        1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarLossLaunchPlan {
    pub reduce_threads: usize,
}

impl ScalarLossLaunchPlan {
    pub fn new(layout: ScalarLossLayout) -> Self {
        // Correctness baseline: one thread per sample writes debug
        // per-sample loss; thread 0 also computes the reduced sum/mean.
        Self { reduce_threads: layout.batch_size }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarLossHostBatch<'a> {
    pub outputs: &'a [f32],
    pub targets: &'a [f32],
    pub entry_weights: &'a [f32],
    pub batch_size: usize,
}

impl<'a> ScalarLossHostBatch<'a> {
    pub fn validate(&self) -> std::result::Result<(), LossLayoutError> {
        ScalarLossLayout::new(self.batch_size).validate()?;
        expect_len("outputs", self.batch_size, self.outputs.len())?;
        expect_len("targets", self.batch_size, self.targets.len())?;
        expect_len("entry_weights", self.batch_size, self.entry_weights.len())?;
        Ok(())
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum LossLayoutError {
    #[error("loss batch must contain at least one sample")]
    EmptyBatch,
    #[error("batch length mismatch for {name}: expected {expected}, got {actual}")]
    BatchLength { name: &'static str, expected: usize, actual: usize },
    #[error("layout value mismatch for {name}: expected {expected}, got {actual}")]
    LayoutValue { name: &'static str, expected: usize, actual: usize },
}

fn expect_len(name: &'static str, expected: usize, actual: usize) -> std::result::Result<(), LossLayoutError> {
    if expected == actual {
        Ok(())
    } else {
        Err(LossLayoutError::BatchLength { name, expected, actual })
    }
}

#[cfg(feature = "cuda")]
pub struct ScalarLossDeviceBatch {
    pub batch_size: usize,
    pub outputs: DeviceBuffer<f32>,
    pub targets: DeviceBuffer<f32>,
    pub entry_weights: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl ScalarLossDeviceBatch {
    pub fn from_host(stream: &CudaStream, batch: &ScalarLossHostBatch<'_>) -> Result<Self> {
        batch.validate()?;
        Ok(Self {
            batch_size: batch.batch_size,
            outputs: DeviceBuffer::from_host(stream, batch.outputs)?,
            targets: DeviceBuffer::from_host(stream, batch.targets)?,
            entry_weights: DeviceBuffer::from_host(stream, batch.entry_weights)?,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct ScalarLossWorkspace {
    pub layout: ScalarLossLayout,
    pub per_sample: DeviceBuffer<f32>,
    pub weighted_sum: DeviceBuffer<f32>,
    pub mean: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl ScalarLossWorkspace {
    pub fn new(stream: &CudaStream, layout: ScalarLossLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            per_sample: DeviceBuffer::<f32>::zeroed(stream, layout.per_sample_len())?,
            weighted_sum: DeviceBuffer::<f32>::zeroed(stream, layout.reduced_len())?,
            mean: DeviceBuffer::<f32>::zeroed(stream, layout.reduced_len())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_names_are_stable() {
        assert_eq!(LOSS_KERNEL_NAMES, ["loss_sigmoid_mse_reduce", "loss_nnue_pytorch_wrm_reduce"]);
    }

    #[test]
    fn layout_counts_buffers() {
        let layout = ScalarLossLayout::new(8);

        assert_eq!(layout.per_sample_len(), 8);
        assert_eq!(layout.reduced_len(), 1);
        assert_eq!(ScalarLossLaunchPlan::new(layout).reduce_threads, 8);
    }

    #[test]
    fn host_batch_validates() {
        let batch = ScalarLossHostBatch {
            outputs: &[0.0, 1.0],
            targets: &[0.5, 0.75],
            entry_weights: &[1.0, 0.5],
            batch_size: 2,
        };

        batch.validate().unwrap();
    }

    #[test]
    fn host_batch_reports_length_mismatch() {
        let batch =
            ScalarLossHostBatch { outputs: &[0.0], targets: &[0.5, 0.75], entry_weights: &[1.0, 0.5], batch_size: 2 };

        let err = batch.validate().unwrap_err();

        assert_eq!(err, LossLayoutError::BatchLength { name: "outputs", expected: 2, actual: 1 });
    }

    #[test]
    fn layout_rejects_empty_batch() {
        let err = ScalarLossLayout::new(0).validate().unwrap_err();

        assert_eq!(err, LossLayoutError::EmptyBatch);
    }
}
