//! Fixed-layout NNUE forward workspace for cuda-oxide kernels.
//!
//! This module intentionally mirrors the root `bulletou_lib` fast NNUE shape
//! without depending on the root workspace. The bridge code can assert both
//! layouts match before uploading buffers.

#[cfg(feature = "cuda")]
use crate::{CudaStream, DeviceBuffer, Result};

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardWorkspaceLayout {
    pub shape: NnueForwardShape,
    pub batch_size: usize,
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
    fn workspace_layout_counts_forward_activations() {
        let shape = NnueForwardShape {
            input_size: 4,
            l1: 2,
            l2: 3,
            l3: 1,
        };
        let layout = NnueForwardWorkspaceLayout::new(shape, 5);

        assert_eq!(layout.l0_len(), 10);
        assert_eq!(layout.combined_len(), 20);
        assert_eq!(layout.hidden1_len(), 15);
        assert_eq!(layout.hidden2_len(), 5);
        assert_eq!(layout.output_len(), 5);
        assert_eq!(layout.total_activation_f32_len(), 65);
    }
}
