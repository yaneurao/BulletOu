//! Fixed-layout SFNN forward workspace for cuda-oxide kernels.
//!
//! This mirrors the root `bulletou_lib::value::fast_sfnn` layout without
//! depending on the root workspace.

#[cfg(feature = "cuda")]
use crate::{CudaStream, DeviceBuffer, Result};

pub const SFNN_SPARSE_L0_CRELU_KERNEL: &str = "sfnn_sparse_l0_crelu";
pub const SFNN_PAIRWISE_CONCAT_KERNEL: &str = "sfnn_pairwise_concat";
pub const SFNN_STACKED_L1_KERNEL: &str = "sfnn_stacked_l1";
pub const SFNN_SHARED_L1_ADD_KERNEL: &str = "sfnn_shared_l1_add";
pub const SFNN_L2_INPUT_KERNEL: &str = "sfnn_l2_input";
pub const SFNN_STACKED_L2_CRELU_KERNEL: &str = "sfnn_stacked_l2_crelu";
pub const SFNN_STACKED_L3_OUTPUT_KERNEL: &str = "sfnn_stacked_l3_output";
pub const SFNN_FORWARD_KERNEL_NAMES: [&str; 7] = [
    SFNN_SPARSE_L0_CRELU_KERNEL,
    SFNN_PAIRWISE_CONCAT_KERNEL,
    SFNN_STACKED_L1_KERNEL,
    SFNN_SHARED_L1_ADD_KERNEL,
    SFNN_L2_INPUT_KERNEL,
    SFNN_STACKED_L2_CRELU_KERNEL,
    SFNN_STACKED_L3_OUTPUT_KERNEL,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardShape {
    pub input_size: usize,
    pub ft_size: usize,
    pub l1_hidden: usize,
    pub l2_size: usize,
    pub num_stacks: usize,
}

pub const SFNN_HALFKA2_1024_7_64_K3K3: SfnnForwardShape =
    SfnnForwardShape { input_size: 131_949, ft_size: 1024, l1_hidden: 7, l2_size: 64, num_stacks: 9 };

impl SfnnForwardShape {
    pub fn l1_out(self) -> usize {
        self.l1_hidden + 1
    }

    pub fn l2_in(self) -> usize {
        self.l1_hidden * 2
    }

    pub fn pairwise_size(self) -> usize {
        self.ft_size / 2
    }

    pub fn validate(self) -> std::result::Result<(), SfnnLayoutError> {
        if self.input_size == 0 {
            return Err(SfnnLayoutError::Shape { message: "input_size must be > 0".to_string() });
        }
        if self.ft_size == 0 {
            return Err(SfnnLayoutError::Shape { message: "ft_size must be > 0".to_string() });
        }
        if self.ft_size % 2 != 0 {
            return Err(SfnnLayoutError::Shape { message: format!("ft_size must be even, got {}", self.ft_size) });
        }
        if self.l1_hidden == 0 {
            return Err(SfnnLayoutError::Shape { message: "l1_hidden must be > 0".to_string() });
        }
        if self.l2_size == 0 {
            return Err(SfnnLayoutError::Shape { message: "l2_size must be > 0".to_string() });
        }
        if self.num_stacks == 0 {
            return Err(SfnnLayoutError::Shape { message: "num_stacks must be > 0".to_string() });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardWeightLayout {
    pub shape: SfnnForwardShape,
}

impl SfnnForwardWeightLayout {
    pub fn new(shape: SfnnForwardShape) -> Self {
        Self { shape }
    }

    pub fn l0w_len(self) -> usize {
        self.shape.input_size.saturating_mul(self.shape.ft_size)
    }

    pub fn l0b_len(self) -> usize {
        self.shape.ft_size
    }

    pub fn l1w_len(self) -> usize {
        self.shape.ft_size.saturating_mul(self.shape.num_stacks).saturating_mul(self.shape.l1_out())
    }

    pub fn l1b_len(self) -> usize {
        self.shape.num_stacks.saturating_mul(self.shape.l1_out())
    }

    pub fn l1fw_len(self) -> usize {
        self.shape.ft_size.saturating_mul(self.shape.l1_out())
    }

    pub fn l1fb_len(self) -> usize {
        self.shape.l1_out()
    }

    pub fn l2w_len(self) -> usize {
        self.shape.l2_in().saturating_mul(self.shape.num_stacks).saturating_mul(self.shape.l2_size)
    }

    pub fn l2b_len(self) -> usize {
        self.shape.num_stacks.saturating_mul(self.shape.l2_size)
    }

    pub fn l3w_len(self) -> usize {
        self.shape.l2_size.saturating_mul(self.shape.num_stacks)
    }

    pub fn l3b_len(self) -> usize {
        self.shape.num_stacks
    }

    pub fn validate_host_weights(
        self,
        weights: &SfnnForwardHostWeights<'_>,
    ) -> std::result::Result<(), SfnnLayoutError> {
        self.shape.validate()?;
        expect_len("l0w", self.l0w_len(), weights.l0w.len())?;
        expect_len("l0b", self.l0b_len(), weights.l0b.len())?;
        expect_len("l1w", self.l1w_len(), weights.l1w.len())?;
        expect_len("l1b", self.l1b_len(), weights.l1b.len())?;
        match (weights.l1fw, weights.l1fb) {
            (Some(l1fw), Some(l1fb)) => {
                expect_len("l1fw", self.l1fw_len(), l1fw.len())?;
                expect_len("l1fb", self.l1fb_len(), l1fb.len())?;
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(SfnnLayoutError::Shape { message: "l1fw requires l1fb".to_string() });
            }
            (None, Some(_)) => {
                return Err(SfnnLayoutError::Shape { message: "l1fb requires l1fw".to_string() });
            }
        }
        expect_len("l2w", self.l2w_len(), weights.l2w.len())?;
        expect_len("l2b", self.l2b_len(), weights.l2b.len())?;
        expect_len("l3w", self.l3w_len(), weights.l3w.len())?;
        expect_len("l3b", self.l3b_len(), weights.l3b.len())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnForwardHostWeights<'a> {
    pub shape: SfnnForwardShape,
    pub l0w: &'a [f32],
    pub l0b: &'a [f32],
    pub l1w: &'a [f32],
    pub l1b: &'a [f32],
    pub l1fw: Option<&'a [f32]>,
    pub l1fb: Option<&'a [f32]>,
    pub l2w: &'a [f32],
    pub l2b: &'a [f32],
    pub l3w: &'a [f32],
    pub l3b: &'a [f32],
}

impl<'a> SfnnForwardHostWeights<'a> {
    pub fn validate(&self) -> std::result::Result<(), SfnnLayoutError> {
        SfnnForwardWeightLayout::new(self.shape).validate_host_weights(self)
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum SfnnLayoutError {
    #[error("invalid SFNN shape: {message}")]
    Shape { message: String },
    #[error("weight length mismatch for {name}: expected {expected}, got {actual}")]
    WeightLength { name: &'static str, expected: usize, actual: usize },
    #[error("batch length mismatch for {name}: expected {expected}, got {actual}")]
    BatchLength { name: &'static str, expected: usize, actual: usize },
    #[error("layout value mismatch for {name}: expected {expected}, got {actual}")]
    LayoutValue { name: &'static str, expected: usize, actual: usize },
}

fn expect_len(name: &'static str, expected: usize, actual: usize) -> std::result::Result<(), SfnnLayoutError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SfnnLayoutError::WeightLength { name, expected, actual })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnForwardHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub buckets: &'a [i32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl<'a> SfnnForwardHostBatch<'a> {
    pub fn validate(&self) -> std::result::Result<(), SfnnLayoutError> {
        let expected_sparse = self.batch_size.saturating_mul(self.max_active);
        expect_batch_len("stm_indices", expected_sparse, self.stm_indices.len())?;
        expect_batch_len("nstm_indices", expected_sparse, self.nstm_indices.len())?;
        expect_batch_len("buckets", self.batch_size, self.buckets.len())?;
        Ok(())
    }
}

fn expect_batch_len(name: &'static str, expected: usize, actual: usize) -> std::result::Result<(), SfnnLayoutError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SfnnLayoutError::BatchLength { name, expected, actual })
    }
}

#[cfg(feature = "cuda")]
pub struct SfnnForwardDeviceBatch {
    pub batch_size: usize,
    pub max_active: usize,
    pub stm_indices: DeviceBuffer<i32>,
    pub nstm_indices: DeviceBuffer<i32>,
    pub buckets: DeviceBuffer<i32>,
}

#[cfg(feature = "cuda")]
impl SfnnForwardDeviceBatch {
    pub fn from_host(stream: &CudaStream, batch: &SfnnForwardHostBatch<'_>) -> Result<Self> {
        batch.validate()?;
        Ok(Self {
            batch_size: batch.batch_size,
            max_active: batch.max_active,
            stm_indices: DeviceBuffer::from_host(stream, batch.stm_indices)?,
            nstm_indices: DeviceBuffer::from_host(stream, batch.nstm_indices)?,
            buckets: DeviceBuffer::from_host(stream, batch.buckets)?,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct SfnnForwardDeviceWeights {
    pub shape: SfnnForwardShape,
    pub l0w: DeviceBuffer<f32>,
    pub l0b: DeviceBuffer<f32>,
    pub l1w: DeviceBuffer<f32>,
    pub l1b: DeviceBuffer<f32>,
    pub l1fw: Option<DeviceBuffer<f32>>,
    pub l1fb: Option<DeviceBuffer<f32>>,
    pub l2w: DeviceBuffer<f32>,
    pub l2b: DeviceBuffer<f32>,
    pub l3w: DeviceBuffer<f32>,
    pub l3b: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl SfnnForwardDeviceWeights {
    pub fn from_host(stream: &CudaStream, weights: &SfnnForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            shape: weights.shape,
            l0w: DeviceBuffer::from_host(stream, weights.l0w)?,
            l0b: DeviceBuffer::from_host(stream, weights.l0b)?,
            l1w: DeviceBuffer::from_host(stream, weights.l1w)?,
            l1b: DeviceBuffer::from_host(stream, weights.l1b)?,
            l1fw: match weights.l1fw {
                Some(values) => Some(DeviceBuffer::from_host(stream, values)?),
                None => None,
            },
            l1fb: match weights.l1fb {
                Some(values) => Some(DeviceBuffer::from_host(stream, values)?),
                None => None,
            },
            l2w: DeviceBuffer::from_host(stream, weights.l2w)?,
            l2b: DeviceBuffer::from_host(stream, weights.l2b)?,
            l3w: DeviceBuffer::from_host(stream, weights.l3w)?,
            l3b: DeviceBuffer::from_host(stream, weights.l3b)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardWorkspaceLayout {
    pub shape: SfnnForwardShape,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardLaunchPlan {
    pub sparse_l0_threads_per_perspective: usize,
    pub pairwise_concat_threads: usize,
    pub stacked_l1_threads: usize,
    pub shared_l1_threads: usize,
    pub l2_input_threads: usize,
    pub stacked_l2_threads: usize,
    pub stacked_l3_threads: usize,
}

impl SfnnForwardLaunchPlan {
    pub fn new(layout: SfnnForwardWorkspaceLayout) -> Self {
        Self {
            sparse_l0_threads_per_perspective: layout.l0_len(),
            pairwise_concat_threads: layout.combined_len(),
            stacked_l1_threads: layout.l1_len(),
            shared_l1_threads: layout.l1_len(),
            l2_input_threads: layout.l2_input_len(),
            stacked_l2_threads: layout.l2_len(),
            stacked_l3_threads: layout.output_len(),
        }
    }
}

impl SfnnForwardWorkspaceLayout {
    pub fn new(shape: SfnnForwardShape, batch_size: usize) -> Self {
        Self { shape, batch_size }
    }

    pub fn l0_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn combined_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn l1_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1_out())
    }

    pub fn l2_input_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_in())
    }

    pub fn l2_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_size)
    }

    pub fn output_len(self) -> usize {
        self.batch_size
    }

    pub fn total_activation_f32_len(self) -> usize {
        self.l0_len()
            .saturating_mul(2)
            .saturating_add(self.combined_len())
            .saturating_add(self.l1_len())
            .saturating_add(self.l2_input_len())
            .saturating_add(self.l2_len())
            .saturating_add(self.output_len())
    }
}

#[cfg(feature = "cuda")]
pub struct SfnnForwardWorkspace {
    pub layout: SfnnForwardWorkspaceLayout,
    pub stm_l0: DeviceBuffer<f32>,
    pub nstm_l0: DeviceBuffer<f32>,
    pub combined: DeviceBuffer<f32>,
    pub l1: DeviceBuffer<f32>,
    pub l2_input: DeviceBuffer<f32>,
    pub l2: DeviceBuffer<f32>,
    pub output: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl SfnnForwardWorkspace {
    pub fn new(stream: &CudaStream, layout: SfnnForwardWorkspaceLayout) -> Result<Self> {
        layout.shape.validate()?;
        Ok(Self {
            layout,
            stm_l0: DeviceBuffer::<f32>::zeroed(stream, layout.l0_len())?,
            nstm_l0: DeviceBuffer::<f32>::zeroed(stream, layout.l0_len())?,
            combined: DeviceBuffer::<f32>::zeroed(stream, layout.combined_len())?,
            l1: DeviceBuffer::<f32>::zeroed(stream, layout.l1_len())?,
            l2_input: DeviceBuffer::<f32>::zeroed(stream, layout.l2_input_len())?,
            l2: DeviceBuffer::<f32>::zeroed(stream, layout.l2_len())?,
            output: DeviceBuffer::<f32>::zeroed(stream, layout.output_len())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_shape() -> SfnnForwardShape {
        SfnnForwardShape { input_size: 4, ft_size: 4, l1_hidden: 2, l2_size: 3, num_stacks: 2 }
    }

    #[test]
    fn weight_layout_counts_fixed_sfnn_weights() {
        let layout = SfnnForwardWeightLayout::new(SFNN_HALFKA2_1024_7_64_K3K3);

        assert_eq!(layout.l0w_len(), 131_949 * 1024);
        assert_eq!(layout.l0b_len(), 1024);
        assert_eq!(layout.l1w_len(), 1024 * 9 * 8);
        assert_eq!(layout.l1b_len(), 9 * 8);
        assert_eq!(layout.l1fw_len(), 1024 * 8);
        assert_eq!(layout.l1fb_len(), 8);
        assert_eq!(layout.l2w_len(), 14 * 9 * 64);
        assert_eq!(layout.l2b_len(), 9 * 64);
        assert_eq!(layout.l3w_len(), 64 * 9);
        assert_eq!(layout.l3b_len(), 9);
    }

    #[test]
    fn forward_kernel_names_are_stable() {
        assert_eq!(
            SFNN_FORWARD_KERNEL_NAMES,
            [
                "sfnn_sparse_l0_crelu",
                "sfnn_pairwise_concat",
                "sfnn_stacked_l1",
                "sfnn_shared_l1_add",
                "sfnn_l2_input",
                "sfnn_stacked_l2_crelu",
                "sfnn_stacked_l3_output",
            ]
        );
    }

    #[test]
    fn workspace_layout_counts_forward_activations() {
        let shape = tiny_shape();
        let layout = SfnnForwardWorkspaceLayout::new(shape, 5);

        assert_eq!(layout.l0_len(), 20);
        assert_eq!(layout.combined_len(), 20);
        assert_eq!(layout.l1_len(), 15);
        assert_eq!(layout.l2_input_len(), 20);
        assert_eq!(layout.l2_len(), 15);
        assert_eq!(layout.output_len(), 5);
        assert_eq!(layout.total_activation_f32_len(), 115);
    }

    #[test]
    fn launch_plan_counts_logical_threads() {
        let shape = tiny_shape();
        let layout = SfnnForwardWorkspaceLayout::new(shape, 5);
        let plan = SfnnForwardLaunchPlan::new(layout);

        assert_eq!(plan.sparse_l0_threads_per_perspective, 20);
        assert_eq!(plan.pairwise_concat_threads, 20);
        assert_eq!(plan.stacked_l1_threads, 15);
        assert_eq!(plan.shared_l1_threads, 15);
        assert_eq!(plan.l2_input_threads, 20);
        assert_eq!(plan.stacked_l2_threads, 15);
        assert_eq!(plan.stacked_l3_threads, 5);
    }

    #[test]
    fn host_weights_validate_against_shape() {
        let shape = tiny_shape();
        let weights = SfnnForwardHostWeights {
            shape,
            l0w: &[0.0; 16],
            l0b: &[0.0; 4],
            l1w: &[0.0; 24],
            l1b: &[0.0; 6],
            l1fw: None,
            l1fb: None,
            l2w: &[0.0; 24],
            l2b: &[0.0; 6],
            l3w: &[0.0; 6],
            l3b: &[0.0; 2],
        };

        weights.validate().unwrap();
    }

    #[test]
    fn host_weights_report_length_mismatch() {
        let shape = tiny_shape();
        let weights = SfnnForwardHostWeights {
            shape,
            l0w: &[0.0; 15],
            l0b: &[0.0; 4],
            l1w: &[0.0; 24],
            l1b: &[0.0; 6],
            l1fw: None,
            l1fb: None,
            l2w: &[0.0; 24],
            l2b: &[0.0; 6],
            l3w: &[0.0; 6],
            l3b: &[0.0; 2],
        };

        let err = weights.validate().unwrap_err();

        assert_eq!(err, SfnnLayoutError::WeightLength { name: "l0w", expected: 16, actual: 15 });
    }

    #[test]
    fn host_batch_validates_sparse_indices_and_buckets() {
        let batch = SfnnForwardHostBatch {
            stm_indices: &[0, 1, -1, 2, -1, -1],
            nstm_indices: &[3, -1, -1, 0, 2, -1],
            buckets: &[0, 1],
            batch_size: 2,
            max_active: 3,
        };

        batch.validate().unwrap();
    }

    #[test]
    fn host_batch_reports_bucket_length_mismatch() {
        let batch = SfnnForwardHostBatch {
            stm_indices: &[0, 1, -1, 2, -1, -1],
            nstm_indices: &[3, -1, -1, 0, 2, -1],
            buckets: &[0],
            batch_size: 2,
            max_active: 3,
        };

        let err = batch.validate().unwrap_err();

        assert_eq!(err, SfnnLayoutError::BatchLength { name: "buckets", expected: 2, actual: 1 });
    }

    #[test]
    fn shape_validation_requires_even_ft_size() {
        let shape = SfnnForwardShape { input_size: 4, ft_size: 3, l1_hidden: 2, l2_size: 3, num_stacks: 2 };

        let err = shape.validate().unwrap_err();

        assert!(matches!(err, SfnnLayoutError::Shape { message } if message.contains("ft_size")));
    }
}
