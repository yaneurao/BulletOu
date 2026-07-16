//! Fixed-layout NNUE forward workspace for cuda-oxide kernels.
//!
//! This module intentionally mirrors the root `bulletou_lib` fast NNUE shape
//! without depending on the root workspace. The bridge code can assert both
//! layouts match before uploading buffers.

#[cfg(feature = "cuda")]
use crate::{CudaStream, DeviceBuffer, Result};

pub const NNUE_SPARSE_L0_CRELU_KERNEL: &str = "nnue_sparse_l0_crelu";
pub const NNUE_CONCAT_L0_KERNEL: &str = "nnue_concat_l0";
pub const NNUE_DENSE_L1_CRELU_KERNEL: &str = "nnue_dense_l1_crelu";
pub const NNUE_DENSE_L2_CRELU_KERNEL: &str = "nnue_dense_l2_crelu";
pub const NNUE_DENSE_OUTPUT_KERNEL: &str = "nnue_dense_output";
pub const NNUE_FORWARD_KERNEL_NAMES: [&str; 5] = [
    NNUE_SPARSE_L0_CRELU_KERNEL,
    NNUE_CONCAT_L0_KERNEL,
    NNUE_DENSE_L1_CRELU_KERNEL,
    NNUE_DENSE_L2_CRELU_KERNEL,
    NNUE_DENSE_OUTPUT_KERNEL,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardShape {
    pub input_size: usize,
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

pub const NNUE_HALFKP_256X2_32_32: NnueForwardShape = NnueForwardShape {
    input_size: 125_388,
    l1: 256,
    l2: 32,
    l3: 32,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardWeightLayout {
    pub shape: NnueForwardShape,
}

impl NnueForwardWeightLayout {
    pub fn new(shape: NnueForwardShape) -> Self {
        Self { shape }
    }

    pub fn l0w_len(self) -> usize {
        self.shape.input_size.saturating_mul(self.shape.l1)
    }

    pub fn l0b_len(self) -> usize {
        self.shape.l1
    }

    pub fn l1w_len(self) -> usize {
        self.shape.l1.saturating_mul(2).saturating_mul(self.shape.l2)
    }

    pub fn l1b_len(self) -> usize {
        self.shape.l2
    }

    pub fn l2w_len(self) -> usize {
        self.shape.l2.saturating_mul(self.shape.l3)
    }

    pub fn l2b_len(self) -> usize {
        self.shape.l3
    }

    pub fn outw_len(self) -> usize {
        self.shape.l3
    }

    pub fn outb_len(self) -> usize {
        1
    }

    pub fn validate_host_weights(self, weights: &NnueForwardHostWeights<'_>) -> std::result::Result<(), NnueLayoutError> {
        expect_len("l0w", self.l0w_len(), weights.l0w.len())?;
        expect_len("l0b", self.l0b_len(), weights.l0b.len())?;
        expect_len("l1w", self.l1w_len(), weights.l1w.len())?;
        expect_len("l1b", self.l1b_len(), weights.l1b.len())?;
        expect_len("l2w", self.l2w_len(), weights.l2w.len())?;
        expect_len("l2b", self.l2b_len(), weights.l2b.len())?;
        expect_len("outw", self.outw_len(), weights.outw.len())?;
        expect_len("outb", self.outb_len(), weights.outb.len())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NnueForwardHostWeights<'a> {
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

impl<'a> NnueForwardHostWeights<'a> {
    pub fn validate(&self) -> std::result::Result<(), NnueLayoutError> {
        NnueForwardWeightLayout::new(self.shape).validate_host_weights(self)
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum NnueLayoutError {
    #[error("weight length mismatch for {name}: expected {expected}, got {actual}")]
    WeightLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("batch length mismatch for {name}: expected {expected}, got {actual}")]
    BatchLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("layout value mismatch for {name}: expected {expected}, got {actual}")]
    LayoutValue {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
}

fn expect_len(name: &'static str, expected: usize, actual: usize) -> std::result::Result<(), NnueLayoutError> {
    if expected == actual {
        Ok(())
    } else {
        Err(NnueLayoutError::WeightLength { name, expected, actual })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NnueForwardHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl<'a> NnueForwardHostBatch<'a> {
    pub fn validate(&self) -> std::result::Result<(), NnueLayoutError> {
        let expected = self.batch_size.saturating_mul(self.max_active);
        expect_batch_len("stm_indices", expected, self.stm_indices.len())?;
        expect_batch_len("nstm_indices", expected, self.nstm_indices.len())?;
        Ok(())
    }
}

fn expect_batch_len(name: &'static str, expected: usize, actual: usize) -> std::result::Result<(), NnueLayoutError> {
    if expected == actual {
        Ok(())
    } else {
        Err(NnueLayoutError::BatchLength { name, expected, actual })
    }
}

#[cfg(feature = "cuda")]
pub struct NnueForwardDeviceBatch {
    pub batch_size: usize,
    pub max_active: usize,
    pub stm_indices: DeviceBuffer<i32>,
    pub nstm_indices: DeviceBuffer<i32>,
}

#[cfg(feature = "cuda")]
impl NnueForwardDeviceBatch {
    pub fn from_host(stream: &CudaStream, batch: &NnueForwardHostBatch<'_>) -> Result<Self> {
        batch.validate()?;
        Ok(Self {
            batch_size: batch.batch_size,
            max_active: batch.max_active,
            stm_indices: DeviceBuffer::from_host(stream, batch.stm_indices)?,
            nstm_indices: DeviceBuffer::from_host(stream, batch.nstm_indices)?,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct NnueForwardDeviceWeights {
    pub shape: NnueForwardShape,
    pub l0w: DeviceBuffer<f32>,
    pub l0b: DeviceBuffer<f32>,
    pub l1w: DeviceBuffer<f32>,
    pub l1b: DeviceBuffer<f32>,
    pub l2w: DeviceBuffer<f32>,
    pub l2b: DeviceBuffer<f32>,
    pub outw: DeviceBuffer<f32>,
    pub outb: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl NnueForwardDeviceWeights {
    pub fn from_host(stream: &CudaStream, weights: &NnueForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            shape: weights.shape,
            l0w: DeviceBuffer::from_host(stream, weights.l0w)?,
            l0b: DeviceBuffer::from_host(stream, weights.l0b)?,
            l1w: DeviceBuffer::from_host(stream, weights.l1w)?,
            l1b: DeviceBuffer::from_host(stream, weights.l1b)?,
            l2w: DeviceBuffer::from_host(stream, weights.l2w)?,
            l2b: DeviceBuffer::from_host(stream, weights.l2b)?,
            outw: DeviceBuffer::from_host(stream, weights.outw)?,
            outb: DeviceBuffer::from_host(stream, weights.outb)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardWorkspaceLayout {
    pub shape: NnueForwardShape,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardLaunchPlan {
    pub sparse_l0_threads_per_perspective: usize,
    pub concat_l0_threads: usize,
    pub dense_l1_threads: usize,
    pub dense_l2_threads: usize,
    pub dense_output_threads: usize,
}

impl NnueForwardLaunchPlan {
    pub fn new(layout: NnueForwardWorkspaceLayout) -> Self {
        Self {
            sparse_l0_threads_per_perspective: layout.l0_len(),
            concat_l0_threads: layout.combined_len(),
            dense_l1_threads: layout.hidden1_len(),
            dense_l2_threads: layout.hidden2_len(),
            dense_output_threads: layout.output_len(),
        }
    }
}

impl NnueForwardWorkspaceLayout {
    pub fn new(shape: NnueForwardShape, batch_size: usize) -> Self {
        Self { shape, batch_size }
    }

    pub fn l0_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1)
    }

    pub fn combined_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1).saturating_mul(2)
    }

    pub fn hidden1_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2)
    }

    pub fn hidden2_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l3)
    }

    pub fn output_len(self) -> usize {
        self.batch_size
    }

    pub fn total_activation_f32_len(self) -> usize {
        self.l0_len()
            .saturating_mul(2)
            .saturating_add(self.combined_len())
            .saturating_add(self.hidden1_len())
            .saturating_add(self.hidden2_len())
            .saturating_add(self.output_len())
    }
}

#[cfg(feature = "cuda")]
pub struct NnueForwardWorkspace {
    pub layout: NnueForwardWorkspaceLayout,
    pub stm_l0: DeviceBuffer<f32>,
    pub nstm_l0: DeviceBuffer<f32>,
    pub combined: DeviceBuffer<f32>,
    pub hidden1: DeviceBuffer<f32>,
    pub hidden2: DeviceBuffer<f32>,
    pub output: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl NnueForwardWorkspace {
    pub fn new(stream: &CudaStream, layout: NnueForwardWorkspaceLayout) -> Result<Self> {
        Ok(Self {
            layout,
            stm_l0: DeviceBuffer::<f32>::zeroed(stream, layout.l0_len())?,
            nstm_l0: DeviceBuffer::<f32>::zeroed(stream, layout.l0_len())?,
            combined: DeviceBuffer::<f32>::zeroed(stream, layout.combined_len())?,
            hidden1: DeviceBuffer::<f32>::zeroed(stream, layout.hidden1_len())?,
            hidden2: DeviceBuffer::<f32>::zeroed(stream, layout.hidden2_len())?,
            output: DeviceBuffer::<f32>::zeroed(stream, layout.output_len())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_shape() -> NnueForwardShape {
        NnueForwardShape {
            input_size: 4,
            l1: 2,
            l2: 3,
            l3: 1,
        }
    }

    #[test]
    fn weight_layout_counts_fixed_nnue_weights() {
        let layout = NnueForwardWeightLayout::new(NNUE_HALFKP_256X2_32_32);

        assert_eq!(layout.l0w_len(), 125_388 * 256);
        assert_eq!(layout.l0b_len(), 256);
        assert_eq!(layout.l1w_len(), 256 * 2 * 32);
        assert_eq!(layout.l1b_len(), 32);
        assert_eq!(layout.l2w_len(), 32 * 32);
        assert_eq!(layout.l2b_len(), 32);
        assert_eq!(layout.outw_len(), 32);
        assert_eq!(layout.outb_len(), 1);
    }

    #[test]
    fn forward_kernel_names_are_stable() {
        assert_eq!(
            NNUE_FORWARD_KERNEL_NAMES,
            [
                "nnue_sparse_l0_crelu",
                "nnue_concat_l0",
                "nnue_dense_l1_crelu",
                "nnue_dense_l2_crelu",
                "nnue_dense_output",
            ]
        );
    }

    #[test]
    fn workspace_layout_counts_forward_activations() {
        let shape = tiny_shape();
        let layout = NnueForwardWorkspaceLayout::new(shape, 5);

        assert_eq!(layout.l0_len(), 10);
        assert_eq!(layout.combined_len(), 20);
        assert_eq!(layout.hidden1_len(), 15);
        assert_eq!(layout.hidden2_len(), 5);
        assert_eq!(layout.output_len(), 5);
        assert_eq!(layout.total_activation_f32_len(), 65);
    }

    #[test]
    fn launch_plan_counts_logical_threads() {
        let shape = tiny_shape();
        let layout = NnueForwardWorkspaceLayout::new(shape, 5);
        let plan = NnueForwardLaunchPlan::new(layout);

        assert_eq!(plan.sparse_l0_threads_per_perspective, 10);
        assert_eq!(plan.concat_l0_threads, 20);
        assert_eq!(plan.dense_l1_threads, 15);
        assert_eq!(plan.dense_l2_threads, 5);
        assert_eq!(plan.dense_output_threads, 5);
    }

    #[test]
    fn host_weights_validate_against_shape() {
        let shape = tiny_shape();
        let weights = NnueForwardHostWeights {
            shape,
            l0w: &[0.0; 8],
            l0b: &[0.0; 2],
            l1w: &[0.0; 12],
            l1b: &[0.0; 3],
            l2w: &[0.0; 3],
            l2b: &[0.0; 1],
            outw: &[0.0; 1],
            outb: &[0.0; 1],
        };

        weights.validate().unwrap();
    }

    #[test]
    fn host_weights_report_length_mismatch() {
        let shape = tiny_shape();
        let weights = NnueForwardHostWeights {
            shape,
            l0w: &[0.0; 7],
            l0b: &[0.0; 2],
            l1w: &[0.0; 12],
            l1b: &[0.0; 3],
            l2w: &[0.0; 3],
            l2b: &[0.0; 1],
            outw: &[0.0; 1],
            outb: &[0.0; 1],
        };

        let err = weights.validate().unwrap_err();

        assert_eq!(
            err,
            NnueLayoutError::WeightLength {
                name: "l0w",
                expected: 8,
                actual: 7,
            }
        );
    }

    #[test]
    fn host_batch_validates_fixed_width_sparse_indices() {
        let batch = NnueForwardHostBatch {
            stm_indices: &[0, 1, -1, 2, -1, -1],
            nstm_indices: &[3, -1, -1, 0, 2, -1],
            batch_size: 2,
            max_active: 3,
        };

        batch.validate().unwrap();
    }

    #[test]
    fn host_batch_reports_length_mismatch() {
        let batch = NnueForwardHostBatch {
            stm_indices: &[0, 1, -1],
            nstm_indices: &[3, -1, -1, 0, 2, -1],
            batch_size: 2,
            max_active: 3,
        };

        let err = batch.validate().unwrap_err();

        assert_eq!(
            err,
            NnueLayoutError::BatchLength {
                name: "stm_indices",
                expected: 6,
                actual: 3,
            }
        );
    }
}
