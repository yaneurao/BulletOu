//! Fixed-layout backward workspaces for cuda-oxide kernels.

pub const DENSE_OUTPUT_BACKWARD_KERNEL: &str = "dense_output_backward";
pub const DENSE_CRELU_BACKWARD_KERNEL: &str = "dense_crelu_backward";
pub const BACKWARD_KERNEL_NAMES: [&str; 2] = [DENSE_OUTPUT_BACKWARD_KERNEL, DENSE_CRELU_BACKWARD_KERNEL];

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

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum BackwardLayoutError {
    #[error("backward batch must contain at least one sample")]
    EmptyBatch,
    #[error("backward input length must be at least one")]
    EmptyInput,
    #[error("backward output length must be at least one")]
    EmptyOutput,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_names_are_stable() {
        assert_eq!(BACKWARD_KERNEL_NAMES, ["dense_output_backward", "dense_crelu_backward"]);
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
}
