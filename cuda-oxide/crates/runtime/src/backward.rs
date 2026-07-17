//! Fixed-layout backward workspaces for cuda-oxide kernels.

use crate::sfnn::SfnnForwardShape;

#[cfg(feature = "cuda")]
use crate::{CudaStream, DeviceBuffer, Result};

pub const DENSE_OUTPUT_BACKWARD_KERNEL: &str = "dense_output_backward";
pub const DENSE_CRELU_BACKWARD_KERNEL: &str = "dense_crelu_backward";
pub const NNUE_L0_CRELU_BACKWARD_KERNEL: &str = "nnue_l0_crelu_backward";
pub const NNUE_L0_SPARSE_BACKWARD_KERNEL: &str = "nnue_l0_sparse_backward";
pub const SFNN_STACKED_L3_BACKWARD_KERNEL: &str = "sfnn_stacked_l3_backward";
pub const SFNN_STACKED_CRELU_BACKWARD_KERNEL: &str = "sfnn_stacked_crelu_backward";
pub const SFNN_L2_INPUT_BACKWARD_KERNEL: &str = "sfnn_l2_input_backward";
pub const SFNN_STACKED_AFFINE_BACKWARD_KERNEL: &str = "sfnn_stacked_affine_backward";
pub const SFNN_PAIRWISE_BACKWARD_KERNEL: &str = "sfnn_pairwise_backward";
pub const SFNN_L0_SPARSE_BACKWARD_KERNEL: &str = "sfnn_l0_sparse_backward";
pub const BACKWARD_KERNEL_NAMES: [&str; 10] = [
    DENSE_OUTPUT_BACKWARD_KERNEL,
    DENSE_CRELU_BACKWARD_KERNEL,
    NNUE_L0_CRELU_BACKWARD_KERNEL,
    NNUE_L0_SPARSE_BACKWARD_KERNEL,
    SFNN_STACKED_L3_BACKWARD_KERNEL,
    SFNN_STACKED_CRELU_BACKWARD_KERNEL,
    SFNN_L2_INPUT_BACKWARD_KERNEL,
    SFNN_STACKED_AFFINE_BACKWARD_KERNEL,
    SFNN_PAIRWISE_BACKWARD_KERNEL,
    SFNN_L0_SPARSE_BACKWARD_KERNEL,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseOutputBackwardLayout {
    pub batch_size: usize,
    pub input_len: usize,
}

impl DenseOutputBackwardLayout {
    pub fn new(batch_size: usize, input_len: usize) -> Self {
        Self { batch_size, input_len }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.input_len == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else {
            Ok(())
        }
    }

    pub fn input_values_len(self) -> usize {
        self.batch_size * self.input_len
    }

    pub fn output_gradients_len(self) -> usize {
        self.batch_size
    }

    pub fn weight_len(self) -> usize {
        self.input_len
    }

    pub fn input_gradients_len(self) -> usize {
        self.batch_size * self.input_len
    }

    pub fn bias_len(self) -> usize {
        1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseCReluBackwardLayout {
    pub batch_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl DenseCReluBackwardLayout {
    pub fn new(batch_size: usize, input_dim: usize, output_dim: usize) -> Self {
        Self { batch_size, input_dim, output_dim }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.input_dim == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else if self.output_dim == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else {
            Ok(())
        }
    }

    pub fn input_values_len(self) -> usize {
        self.batch_size * self.input_dim
    }

    pub fn activations_len(self) -> usize {
        self.batch_size * self.output_dim
    }

    pub fn output_gradients_len(self) -> usize {
        self.batch_size * self.output_dim
    }

    pub fn weight_len(self) -> usize {
        self.input_dim * self.output_dim
    }

    pub fn bias_len(self) -> usize {
        self.output_dim
    }

    pub fn input_gradients_len(self) -> usize {
        self.batch_size * self.input_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueL0CReluBackwardLayout {
    pub batch_size: usize,
    pub l1: usize,
}

impl NnueL0CReluBackwardLayout {
    pub fn new(batch_size: usize, l1: usize) -> Self {
        Self { batch_size, l1 }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.l1 == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else {
            Ok(())
        }
    }

    pub fn per_perspective_len(self) -> usize {
        self.batch_size * self.l1
    }

    pub fn combined_len(self) -> usize {
        self.per_perspective_len() * 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueL0SparseBackwardLayout {
    pub batch_size: usize,
    pub max_active: usize,
    pub input_size: usize,
    pub l1: usize,
}

impl NnueL0SparseBackwardLayout {
    pub fn new(batch_size: usize, max_active: usize, input_size: usize, l1: usize) -> Self {
        Self { batch_size, max_active, input_size, l1 }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.max_active == 0 {
            Err(BackwardLayoutError::EmptySparse)
        } else if self.input_size == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else if self.l1 == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else {
            Ok(())
        }
    }

    pub fn sparse_values_len(self) -> usize {
        self.batch_size * self.max_active
    }

    pub fn per_perspective_gradient_len(self) -> usize {
        self.batch_size * self.l1
    }

    pub fn weight_len(self) -> usize {
        self.input_size * self.l1
    }

    pub fn bias_len(self) -> usize {
        self.l1
    }

    pub fn gradient_threads(self) -> usize {
        self.weight_len().max(self.bias_len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnStackedL3BackwardLayout {
    pub batch_size: usize,
    pub l2_size: usize,
    pub l1_out: usize,
    pub num_stacks: usize,
}

impl SfnnStackedL3BackwardLayout {
    pub fn new(batch_size: usize, l2_size: usize, l1_out: usize, num_stacks: usize) -> Self {
        Self { batch_size, l2_size, l1_out, num_stacks }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.l2_size == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else if self.l1_out == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else if self.num_stacks == 0 {
            Err(BackwardLayoutError::EmptyStack)
        } else {
            Ok(())
        }
    }

    pub fn output_gradients_len(self) -> usize {
        self.batch_size
    }

    pub fn input_gradients_len(self) -> usize {
        self.batch_size * self.l2_size
    }

    pub fn l1_gradients_len(self) -> usize {
        self.batch_size * self.l1_out
    }

    pub fn weight_len(self) -> usize {
        self.l2_size * self.num_stacks
    }

    pub fn bias_len(self) -> usize {
        self.num_stacks
    }

    pub fn gradient_threads(self) -> usize {
        self.input_gradients_len().max(self.l1_gradients_len()).max(self.weight_len()).max(self.bias_len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnStackedCReluBackwardLayout {
    pub batch_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_stacks: usize,
}

impl SfnnStackedCReluBackwardLayout {
    pub fn new(batch_size: usize, input_dim: usize, output_dim: usize, num_stacks: usize) -> Self {
        Self { batch_size, input_dim, output_dim, num_stacks }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.input_dim == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else if self.output_dim == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else if self.num_stacks == 0 {
            Err(BackwardLayoutError::EmptyStack)
        } else {
            Ok(())
        }
    }

    pub fn input_values_len(self) -> usize {
        self.batch_size * self.input_dim
    }

    pub fn activations_len(self) -> usize {
        self.batch_size * self.output_dim
    }

    pub fn output_gradients_len(self) -> usize {
        self.batch_size * self.output_dim
    }

    pub fn input_gradients_len(self) -> usize {
        self.batch_size * self.input_dim
    }

    pub fn weight_len(self) -> usize {
        self.input_dim * self.num_stacks * self.output_dim
    }

    pub fn bias_len(self) -> usize {
        self.num_stacks * self.output_dim
    }

    pub fn gradient_threads(self) -> usize {
        self.input_gradients_len().max(self.weight_len()).max(self.bias_len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnL2InputBackwardLayout {
    pub batch_size: usize,
    pub l1_hidden: usize,
}

impl SfnnL2InputBackwardLayout {
    pub fn new(batch_size: usize, l1_hidden: usize) -> Self {
        Self { batch_size, l1_hidden }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.l1_hidden == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else {
            Ok(())
        }
    }

    pub fn l1_len(self) -> usize {
        self.batch_size * (self.l1_hidden + 1)
    }

    pub fn l2_input_len(self) -> usize {
        self.batch_size * self.l1_hidden * 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnStackedAffineBackwardLayout {
    pub batch_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_stacks: usize,
}

impl SfnnStackedAffineBackwardLayout {
    pub fn new(batch_size: usize, input_dim: usize, output_dim: usize, num_stacks: usize) -> Self {
        Self { batch_size, input_dim, output_dim, num_stacks }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.input_dim == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else if self.output_dim == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else if self.num_stacks == 0 {
            Err(BackwardLayoutError::EmptyStack)
        } else {
            Ok(())
        }
    }

    pub fn input_values_len(self) -> usize {
        self.batch_size * self.input_dim
    }

    pub fn output_gradients_len(self) -> usize {
        self.batch_size * self.output_dim
    }

    pub fn input_gradients_len(self) -> usize {
        self.batch_size * self.input_dim
    }

    pub fn weight_len(self) -> usize {
        self.input_dim * self.num_stacks * self.output_dim
    }

    pub fn bias_len(self) -> usize {
        self.num_stacks * self.output_dim
    }

    pub fn gradient_threads(self) -> usize {
        self.input_gradients_len().max(self.weight_len()).max(self.bias_len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnPairwiseBackwardLayout {
    pub batch_size: usize,
    pub ft_size: usize,
}

impl SfnnPairwiseBackwardLayout {
    pub fn new(batch_size: usize, ft_size: usize) -> Self {
        Self { batch_size, ft_size }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.ft_size == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else {
            Ok(())
        }
    }

    pub fn l0_len(self) -> usize {
        self.batch_size * self.ft_size
    }

    pub fn combined_gradients_len(self) -> usize {
        self.batch_size * self.ft_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnL0SparseBackwardLayout {
    pub batch_size: usize,
    pub max_active: usize,
    pub input_size: usize,
    pub ft_size: usize,
}

impl SfnnL0SparseBackwardLayout {
    pub fn new(batch_size: usize, max_active: usize, input_size: usize, ft_size: usize) -> Self {
        Self { batch_size, max_active, input_size, ft_size }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.max_active == 0 {
            Err(BackwardLayoutError::EmptySparse)
        } else if self.input_size == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else if self.ft_size == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else {
            Ok(())
        }
    }

    pub fn sparse_values_len(self) -> usize {
        self.batch_size * self.max_active
    }

    pub fn l0_len(self) -> usize {
        self.batch_size * self.ft_size
    }

    pub fn weight_len(self) -> usize {
        self.input_size * self.ft_size
    }

    pub fn bias_len(self) -> usize {
        self.ft_size
    }

    pub fn gradient_threads(self) -> usize {
        self.l0_len().max(self.weight_len()).max(self.bias_len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnBackwardWorkspaceLayout {
    pub shape: SfnnForwardShape,
    pub batch_size: usize,
    pub max_active: usize,
}

impl SfnnBackwardWorkspaceLayout {
    pub fn new(shape: SfnnForwardShape, batch_size: usize, max_active: usize) -> Self {
        Self { shape, batch_size, max_active }
    }

    pub fn validate(self) -> std::result::Result<(), BackwardLayoutError> {
        if self.batch_size == 0 {
            Err(BackwardLayoutError::EmptyBatch)
        } else if self.max_active == 0 {
            Err(BackwardLayoutError::EmptySparse)
        } else if self.shape.input_size == 0 || self.shape.ft_size == 0 {
            Err(BackwardLayoutError::EmptyInput)
        } else if self.shape.l1_hidden == 0 || self.shape.l2_size == 0 {
            Err(BackwardLayoutError::EmptyOutput)
        } else if self.shape.num_stacks == 0 {
            Err(BackwardLayoutError::EmptyStack)
        } else {
            Ok(())
        }
    }

    pub fn l2_gradients_len(self) -> usize {
        self.batch_size * self.shape.l2_size
    }

    pub fn l1_gradients_len(self) -> usize {
        self.batch_size * self.shape.l1_out()
    }

    pub fn l2_input_gradients_len(self) -> usize {
        self.batch_size * self.shape.l2_in()
    }

    pub fn combined_gradients_len(self) -> usize {
        self.batch_size * self.shape.ft_size
    }

    pub fn l0_gradients_len(self) -> usize {
        self.batch_size * self.shape.ft_size
    }

    pub fn l0w_gradients_len(self) -> usize {
        self.shape.input_size * self.shape.ft_size
    }

    pub fn l0b_gradients_len(self) -> usize {
        self.shape.ft_size
    }

    pub fn l1w_gradients_len(self) -> usize {
        self.shape.ft_size * self.shape.num_stacks * self.shape.l1_out()
    }

    pub fn l1b_gradients_len(self) -> usize {
        self.shape.num_stacks * self.shape.l1_out()
    }

    pub fn l2w_gradients_len(self) -> usize {
        self.shape.l2_in() * self.shape.num_stacks * self.shape.l2_size
    }

    pub fn l2b_gradients_len(self) -> usize {
        self.shape.num_stacks * self.shape.l2_size
    }

    pub fn l3w_gradients_len(self) -> usize {
        self.shape.l2_size * self.shape.num_stacks
    }

    pub fn l3b_gradients_len(self) -> usize {
        self.shape.num_stacks
    }

    pub fn total_gradient_f32_len(self) -> usize {
        self.l2_gradients_len()
            .saturating_add(self.l1_gradients_len())
            .saturating_add(self.l2_input_gradients_len())
            .saturating_add(self.combined_gradients_len())
            .saturating_add(self.l0_gradients_len().saturating_mul(4))
            .saturating_add(self.l0w_gradients_len())
            .saturating_add(self.l0b_gradients_len())
            .saturating_add(self.l1w_gradients_len())
            .saturating_add(self.l1b_gradients_len())
            .saturating_add(self.l2w_gradients_len())
            .saturating_add(self.l2b_gradients_len())
            .saturating_add(self.l3w_gradients_len())
            .saturating_add(self.l3b_gradients_len())
    }
}

#[cfg(feature = "cuda")]
pub struct SfnnBackwardWorkspace {
    pub layout: SfnnBackwardWorkspaceLayout,
    pub l2_gradients: DeviceBuffer<f32>,
    pub l1_gradients: DeviceBuffer<f32>,
    pub l2_input_gradients: DeviceBuffer<f32>,
    pub combined_gradients: DeviceBuffer<f32>,
    pub stm_l0_gradients: DeviceBuffer<f32>,
    pub nstm_l0_gradients: DeviceBuffer<f32>,
    pub stm_l0_pre_gradients: DeviceBuffer<f32>,
    pub nstm_l0_pre_gradients: DeviceBuffer<f32>,
    pub l0w_gradients: DeviceBuffer<f32>,
    pub l0b_gradients: DeviceBuffer<f32>,
    pub l1w_gradients: DeviceBuffer<f32>,
    pub l1b_gradients: DeviceBuffer<f32>,
    pub l2w_gradients: DeviceBuffer<f32>,
    pub l2b_gradients: DeviceBuffer<f32>,
    pub l3w_gradients: DeviceBuffer<f32>,
    pub l3b_gradients: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl SfnnBackwardWorkspace {
    pub fn new(stream: &CudaStream, layout: SfnnBackwardWorkspaceLayout) -> Result<Self> {
        layout.shape.validate()?;
        layout.validate()?;
        Ok(Self {
            layout,
            l2_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l2_gradients_len())?,
            l1_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l1_gradients_len())?,
            l2_input_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l2_input_gradients_len())?,
            combined_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.combined_gradients_len())?,
            stm_l0_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l0_gradients_len())?,
            nstm_l0_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l0_gradients_len())?,
            stm_l0_pre_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l0_gradients_len())?,
            nstm_l0_pre_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l0_gradients_len())?,
            l0w_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l0w_gradients_len())?,
            l0b_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l0b_gradients_len())?,
            l1w_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l1w_gradients_len())?,
            l1b_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l1b_gradients_len())?,
            l2w_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l2w_gradients_len())?,
            l2b_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l2b_gradients_len())?,
            l3w_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l3w_gradients_len())?,
            l3b_gradients: DeviceBuffer::<f32>::zeroed(stream, layout.l3b_gradients_len())?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseOutputBackwardLaunchPlan {
    pub threads: usize,
}

impl DenseOutputBackwardLaunchPlan {
    pub fn new(layout: DenseOutputBackwardLayout) -> Self {
        let input_gradient_threads = layout.input_gradients_len();
        let weight_gradient_threads = layout.weight_len();
        Self { threads: input_gradient_threads.max(weight_gradient_threads).max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseCReluBackwardLaunchPlan {
    pub threads: usize,
}

impl DenseCReluBackwardLaunchPlan {
    pub fn new(layout: DenseCReluBackwardLayout) -> Self {
        Self { threads: layout.input_gradients_len().max(layout.weight_len()).max(layout.bias_len()).max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueL0CReluBackwardLaunchPlan {
    pub threads: usize,
}

impl NnueL0CReluBackwardLaunchPlan {
    pub fn new(layout: NnueL0CReluBackwardLayout) -> Self {
        Self { threads: layout.combined_len().max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueL0SparseBackwardLaunchPlan {
    pub threads: usize,
}

impl NnueL0SparseBackwardLaunchPlan {
    pub fn new(layout: NnueL0SparseBackwardLayout) -> Self {
        Self { threads: layout.gradient_threads().max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnStackedL3BackwardLaunchPlan {
    pub threads: usize,
}

impl SfnnStackedL3BackwardLaunchPlan {
    pub fn new(layout: SfnnStackedL3BackwardLayout) -> Self {
        Self { threads: layout.gradient_threads().max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnStackedCReluBackwardLaunchPlan {
    pub threads: usize,
}

impl SfnnStackedCReluBackwardLaunchPlan {
    pub fn new(layout: SfnnStackedCReluBackwardLayout) -> Self {
        Self { threads: layout.gradient_threads().max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnL2InputBackwardLaunchPlan {
    pub threads: usize,
}

impl SfnnL2InputBackwardLaunchPlan {
    pub fn new(layout: SfnnL2InputBackwardLayout) -> Self {
        Self { threads: layout.l1_len().max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnStackedAffineBackwardLaunchPlan {
    pub threads: usize,
}

impl SfnnStackedAffineBackwardLaunchPlan {
    pub fn new(layout: SfnnStackedAffineBackwardLayout) -> Self {
        Self { threads: layout.gradient_threads().max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnPairwiseBackwardLaunchPlan {
    pub threads: usize,
}

impl SfnnPairwiseBackwardLaunchPlan {
    pub fn new(layout: SfnnPairwiseBackwardLayout) -> Self {
        Self { threads: layout.l0_len().max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnL0SparseBackwardLaunchPlan {
    pub threads: usize,
}

impl SfnnL0SparseBackwardLaunchPlan {
    pub fn new(layout: SfnnL0SparseBackwardLayout) -> Self {
        Self { threads: layout.gradient_threads().max(1) }
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum BackwardLayoutError {
    #[error("backward batch must contain at least one sample")]
    EmptyBatch,
    #[error("backward input length must be at least one")]
    EmptyInput,
    #[error("backward output length must be at least one")]
    EmptyOutput,
    #[error("backward sparse list must contain at least one slot")]
    EmptySparse,
    #[error("backward stack count must be at least one")]
    EmptyStack,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_names_are_stable() {
        assert_eq!(
            BACKWARD_KERNEL_NAMES,
            [
                "dense_output_backward",
                "dense_crelu_backward",
                "nnue_l0_crelu_backward",
                "nnue_l0_sparse_backward",
                "sfnn_stacked_l3_backward",
                "sfnn_stacked_crelu_backward",
                "sfnn_l2_input_backward",
                "sfnn_stacked_affine_backward",
                "sfnn_pairwise_backward",
                "sfnn_l0_sparse_backward"
            ]
        );
    }

    #[test]
    fn layout_counts_buffers() {
        let layout = DenseOutputBackwardLayout::new(3, 4);

        assert_eq!(layout.input_values_len(), 12);
        assert_eq!(layout.output_gradients_len(), 3);
        assert_eq!(layout.weight_len(), 4);
        assert_eq!(layout.input_gradients_len(), 12);
        assert_eq!(layout.bias_len(), 1);
        assert_eq!(DenseOutputBackwardLaunchPlan::new(layout).threads, 12);
    }

    #[test]
    fn layout_rejects_empty_values() {
        assert_eq!(DenseOutputBackwardLayout::new(0, 4).validate().unwrap_err(), BackwardLayoutError::EmptyBatch);
        assert_eq!(DenseOutputBackwardLayout::new(3, 0).validate().unwrap_err(), BackwardLayoutError::EmptyInput);
    }

    #[test]
    fn crelu_layout_counts_buffers() {
        let layout = DenseCReluBackwardLayout::new(3, 4, 5);

        assert_eq!(layout.input_values_len(), 12);
        assert_eq!(layout.activations_len(), 15);
        assert_eq!(layout.output_gradients_len(), 15);
        assert_eq!(layout.weight_len(), 20);
        assert_eq!(layout.bias_len(), 5);
        assert_eq!(layout.input_gradients_len(), 12);
        assert_eq!(DenseCReluBackwardLaunchPlan::new(layout).threads, 20);
    }

    #[test]
    fn crelu_layout_rejects_empty_values() {
        assert_eq!(DenseCReluBackwardLayout::new(0, 4, 5).validate().unwrap_err(), BackwardLayoutError::EmptyBatch);
        assert_eq!(DenseCReluBackwardLayout::new(3, 0, 5).validate().unwrap_err(), BackwardLayoutError::EmptyInput);
        assert_eq!(DenseCReluBackwardLayout::new(3, 4, 0).validate().unwrap_err(), BackwardLayoutError::EmptyOutput);
    }

    #[test]
    fn nnue_l0_crelu_layout_counts_buffers() {
        let layout = NnueL0CReluBackwardLayout::new(3, 4);

        assert_eq!(layout.per_perspective_len(), 12);
        assert_eq!(layout.combined_len(), 24);
        assert_eq!(NnueL0CReluBackwardLaunchPlan::new(layout).threads, 24);
    }

    #[test]
    fn nnue_l0_crelu_layout_rejects_empty_values() {
        assert_eq!(NnueL0CReluBackwardLayout::new(0, 4).validate().unwrap_err(), BackwardLayoutError::EmptyBatch);
        assert_eq!(NnueL0CReluBackwardLayout::new(3, 0).validate().unwrap_err(), BackwardLayoutError::EmptyOutput);
    }

    #[test]
    fn nnue_l0_sparse_layout_counts_buffers() {
        let layout = NnueL0SparseBackwardLayout::new(3, 5, 7, 4);

        assert_eq!(layout.sparse_values_len(), 15);
        assert_eq!(layout.per_perspective_gradient_len(), 12);
        assert_eq!(layout.weight_len(), 28);
        assert_eq!(layout.bias_len(), 4);
        assert_eq!(layout.gradient_threads(), 28);
        assert_eq!(NnueL0SparseBackwardLaunchPlan::new(layout).threads, 28);
    }

    #[test]
    fn nnue_l0_sparse_layout_rejects_empty_values() {
        assert_eq!(
            NnueL0SparseBackwardLayout::new(0, 5, 7, 4).validate().unwrap_err(),
            BackwardLayoutError::EmptyBatch
        );
        assert_eq!(
            NnueL0SparseBackwardLayout::new(3, 0, 7, 4).validate().unwrap_err(),
            BackwardLayoutError::EmptySparse
        );
        assert_eq!(
            NnueL0SparseBackwardLayout::new(3, 5, 0, 4).validate().unwrap_err(),
            BackwardLayoutError::EmptyInput
        );
        assert_eq!(
            NnueL0SparseBackwardLayout::new(3, 5, 7, 0).validate().unwrap_err(),
            BackwardLayoutError::EmptyOutput
        );
    }

    #[test]
    fn sfnn_stacked_l3_layout_counts_buffers() {
        let layout = SfnnStackedL3BackwardLayout::new(3, 5, 4, 2);

        assert_eq!(layout.output_gradients_len(), 3);
        assert_eq!(layout.input_gradients_len(), 15);
        assert_eq!(layout.l1_gradients_len(), 12);
        assert_eq!(layout.weight_len(), 10);
        assert_eq!(layout.bias_len(), 2);
        assert_eq!(layout.gradient_threads(), 15);
        assert_eq!(SfnnStackedL3BackwardLaunchPlan::new(layout).threads, 15);
    }

    #[test]
    fn sfnn_stacked_l3_layout_rejects_empty_values() {
        assert_eq!(
            SfnnStackedL3BackwardLayout::new(0, 5, 4, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyBatch
        );
        assert_eq!(
            SfnnStackedL3BackwardLayout::new(3, 0, 4, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyInput
        );
        assert_eq!(
            SfnnStackedL3BackwardLayout::new(3, 5, 0, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyOutput
        );
        assert_eq!(
            SfnnStackedL3BackwardLayout::new(3, 5, 4, 0).validate().unwrap_err(),
            BackwardLayoutError::EmptyStack
        );
    }

    #[test]
    fn sfnn_stacked_crelu_layout_counts_buffers() {
        let layout = SfnnStackedCReluBackwardLayout::new(3, 4, 5, 2);

        assert_eq!(layout.input_values_len(), 12);
        assert_eq!(layout.activations_len(), 15);
        assert_eq!(layout.output_gradients_len(), 15);
        assert_eq!(layout.input_gradients_len(), 12);
        assert_eq!(layout.weight_len(), 40);
        assert_eq!(layout.bias_len(), 10);
        assert_eq!(layout.gradient_threads(), 40);
        assert_eq!(SfnnStackedCReluBackwardLaunchPlan::new(layout).threads, 40);
    }

    #[test]
    fn sfnn_stacked_crelu_layout_rejects_empty_values() {
        assert_eq!(
            SfnnStackedCReluBackwardLayout::new(0, 4, 5, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyBatch
        );
        assert_eq!(
            SfnnStackedCReluBackwardLayout::new(3, 0, 5, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyInput
        );
        assert_eq!(
            SfnnStackedCReluBackwardLayout::new(3, 4, 0, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyOutput
        );
        assert_eq!(
            SfnnStackedCReluBackwardLayout::new(3, 4, 5, 0).validate().unwrap_err(),
            BackwardLayoutError::EmptyStack
        );
    }

    #[test]
    fn sfnn_l2_input_layout_counts_buffers() {
        let layout = SfnnL2InputBackwardLayout::new(3, 4);

        assert_eq!(layout.l1_len(), 15);
        assert_eq!(layout.l2_input_len(), 24);
        assert_eq!(SfnnL2InputBackwardLaunchPlan::new(layout).threads, 15);
    }

    #[test]
    fn sfnn_l2_input_layout_rejects_empty_values() {
        assert_eq!(SfnnL2InputBackwardLayout::new(0, 4).validate().unwrap_err(), BackwardLayoutError::EmptyBatch);
        assert_eq!(SfnnL2InputBackwardLayout::new(3, 0).validate().unwrap_err(), BackwardLayoutError::EmptyOutput);
    }

    #[test]
    fn sfnn_stacked_affine_layout_counts_buffers() {
        let layout = SfnnStackedAffineBackwardLayout::new(3, 4, 5, 2);

        assert_eq!(layout.input_values_len(), 12);
        assert_eq!(layout.output_gradients_len(), 15);
        assert_eq!(layout.input_gradients_len(), 12);
        assert_eq!(layout.weight_len(), 40);
        assert_eq!(layout.bias_len(), 10);
        assert_eq!(layout.gradient_threads(), 40);
        assert_eq!(SfnnStackedAffineBackwardLaunchPlan::new(layout).threads, 40);
    }

    #[test]
    fn sfnn_stacked_affine_layout_rejects_empty_values() {
        assert_eq!(
            SfnnStackedAffineBackwardLayout::new(0, 4, 5, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyBatch
        );
        assert_eq!(
            SfnnStackedAffineBackwardLayout::new(3, 0, 5, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyInput
        );
        assert_eq!(
            SfnnStackedAffineBackwardLayout::new(3, 4, 0, 2).validate().unwrap_err(),
            BackwardLayoutError::EmptyOutput
        );
        assert_eq!(
            SfnnStackedAffineBackwardLayout::new(3, 4, 5, 0).validate().unwrap_err(),
            BackwardLayoutError::EmptyStack
        );
    }

    #[test]
    fn sfnn_pairwise_layout_counts_buffers() {
        let layout = SfnnPairwiseBackwardLayout::new(3, 8);

        assert_eq!(layout.l0_len(), 24);
        assert_eq!(layout.combined_gradients_len(), 24);
        assert_eq!(SfnnPairwiseBackwardLaunchPlan::new(layout).threads, 24);
    }

    #[test]
    fn sfnn_pairwise_layout_rejects_empty_values() {
        assert_eq!(SfnnPairwiseBackwardLayout::new(0, 8).validate().unwrap_err(), BackwardLayoutError::EmptyBatch);
        assert_eq!(SfnnPairwiseBackwardLayout::new(3, 0).validate().unwrap_err(), BackwardLayoutError::EmptyInput);
    }

    #[test]
    fn sfnn_l0_sparse_layout_counts_buffers() {
        let layout = SfnnL0SparseBackwardLayout::new(3, 5, 7, 4);

        assert_eq!(layout.sparse_values_len(), 15);
        assert_eq!(layout.l0_len(), 12);
        assert_eq!(layout.weight_len(), 28);
        assert_eq!(layout.bias_len(), 4);
        assert_eq!(layout.gradient_threads(), 28);
        assert_eq!(SfnnL0SparseBackwardLaunchPlan::new(layout).threads, 28);
    }

    #[test]
    fn sfnn_l0_sparse_layout_rejects_empty_values() {
        assert_eq!(
            SfnnL0SparseBackwardLayout::new(0, 5, 7, 4).validate().unwrap_err(),
            BackwardLayoutError::EmptyBatch
        );
        assert_eq!(
            SfnnL0SparseBackwardLayout::new(3, 0, 7, 4).validate().unwrap_err(),
            BackwardLayoutError::EmptySparse
        );
        assert_eq!(
            SfnnL0SparseBackwardLayout::new(3, 5, 0, 4).validate().unwrap_err(),
            BackwardLayoutError::EmptyInput
        );
        assert_eq!(
            SfnnL0SparseBackwardLayout::new(3, 5, 7, 0).validate().unwrap_err(),
            BackwardLayoutError::EmptyOutput
        );
    }

    #[test]
    fn sfnn_backward_workspace_layout_counts_buffers() {
        let shape = SfnnForwardShape { input_size: 7, ft_size: 4, l1_hidden: 2, l2_size: 3, num_stacks: 2 };
        let layout = SfnnBackwardWorkspaceLayout::new(shape, 3, 5);

        assert_eq!(layout.l2_gradients_len(), 9);
        assert_eq!(layout.l1_gradients_len(), 9);
        assert_eq!(layout.l2_input_gradients_len(), 12);
        assert_eq!(layout.combined_gradients_len(), 12);
        assert_eq!(layout.l0_gradients_len(), 12);
        assert_eq!(layout.l0w_gradients_len(), 28);
        assert_eq!(layout.l0b_gradients_len(), 4);
        assert_eq!(layout.l1w_gradients_len(), 24);
        assert_eq!(layout.l1b_gradients_len(), 6);
        assert_eq!(layout.l2w_gradients_len(), 24);
        assert_eq!(layout.l2b_gradients_len(), 6);
        assert_eq!(layout.l3w_gradients_len(), 6);
        assert_eq!(layout.l3b_gradients_len(), 2);
        assert_eq!(layout.total_gradient_f32_len(), 190);
    }

    #[test]
    fn sfnn_backward_workspace_layout_rejects_empty_values() {
        let shape = SfnnForwardShape { input_size: 7, ft_size: 4, l1_hidden: 2, l2_size: 3, num_stacks: 2 };

        assert_eq!(
            SfnnBackwardWorkspaceLayout::new(shape, 0, 5).validate().unwrap_err(),
            BackwardLayoutError::EmptyBatch
        );
        assert_eq!(
            SfnnBackwardWorkspaceLayout::new(shape, 3, 0).validate().unwrap_err(),
            BackwardLayoutError::EmptySparse
        );
        assert_eq!(
            SfnnBackwardWorkspaceLayout::new(SfnnForwardShape { input_size: 0, ..shape }, 3, 5).validate().unwrap_err(),
            BackwardLayoutError::EmptyInput
        );
        assert_eq!(
            SfnnBackwardWorkspaceLayout::new(SfnnForwardShape { l2_size: 0, ..shape }, 3, 5).validate().unwrap_err(),
            BackwardLayoutError::EmptyOutput
        );
        assert_eq!(
            SfnnBackwardWorkspaceLayout::new(SfnnForwardShape { num_stacks: 0, ..shape }, 3, 5).validate().unwrap_err(),
            BackwardLayoutError::EmptyStack
        );
    }
}
