//! Optimizer workspaces and launch layouts for cuda-oxide kernels.

use crate::{nnue::NnueForwardWeightLayout, sfnn::SfnnForwardWeightLayout};

#[cfg(feature = "cuda")]
use crate::{CudaStream, DeviceBuffer, Result, nnue::NnueForwardHostWeights, sfnn::SfnnForwardHostWeights};

pub const ADAMW_UPDATE_KERNEL: &str = "adamw_update";
pub const RADAM_UPDATE_KERNEL: &str = "radam_update";
pub const RANGER_LOOKAHEAD_KERNEL: &str = "ranger_lookahead";
pub const OPTIMIZER_KERNEL_NAMES: [&str; 3] = [ADAMW_UPDATE_KERNEL, RADAM_UPDATE_KERNEL, RANGER_LOOKAHEAD_KERNEL];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWUpdateParams {
    pub gradient_factor: f32,
    pub learning_rate: f32,
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for AdamWUpdateParams {
    fn default() -> Self {
        Self {
            gradient_factor: 1.0,
            learning_rate: 0.001,
            decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 0.00000001,
            min_weight: -1.98,
            max_weight: 1.98,
        }
    }
}

impl AdamWUpdateParams {
    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if !self.gradient_factor.is_finite() {
            Err(OptimizerLayoutError::NonFiniteParam { name: "gradient_factor" })
        } else if !(self.learning_rate.is_finite() && self.learning_rate >= 0.0) {
            Err(OptimizerLayoutError::InvalidLearningRate)
        } else if !self.decay.is_finite() {
            Err(OptimizerLayoutError::NonFiniteParam { name: "decay" })
        } else if !(self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1)) {
            Err(OptimizerLayoutError::InvalidBeta { name: "beta1" })
        } else if !(self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2)) {
            Err(OptimizerLayoutError::InvalidBeta { name: "beta2" })
        } else if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            Err(OptimizerLayoutError::InvalidEpsilon)
        } else if !(self.min_weight.is_finite() && self.max_weight.is_finite() && self.min_weight <= self.max_weight) {
            Err(OptimizerLayoutError::InvalidClamp)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizerStateLayout {
    pub len: usize,
}

impl OptimizerStateLayout {
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if self.len == 0 { Err(OptimizerLayoutError::EmptyParameters) } else { Ok(()) }
    }

    pub fn state_len(self) -> usize {
        self.len
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MomentumVelocityHostState<'a> {
    pub momentum: &'a [f32],
    pub velocity: &'a [f32],
}

impl<'a> MomentumVelocityHostState<'a> {
    pub fn validate(self, layout: OptimizerStateLayout) -> std::result::Result<(), OptimizerLayoutError> {
        layout.validate()?;
        expect_state_len("momentum", layout.state_len(), self.momentum.len())?;
        expect_state_len("velocity", layout.state_len(), self.velocity.len())?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct MomentumVelocityDeviceState {
    pub layout: OptimizerStateLayout,
    pub momentum: DeviceBuffer<f32>,
    pub velocity: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl MomentumVelocityDeviceState {
    pub fn new_zeroed(stream: &CudaStream, layout: OptimizerStateLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            momentum: DeviceBuffer::zeroed(stream, layout.state_len())?,
            velocity: DeviceBuffer::zeroed(stream, layout.state_len())?,
        })
    }

    pub fn from_host(
        stream: &CudaStream,
        layout: OptimizerStateLayout,
        state: MomentumVelocityHostState<'_>,
    ) -> Result<Self> {
        state.validate(layout)?;
        Ok(Self {
            layout,
            momentum: DeviceBuffer::from_host(stream, state.momentum)?,
            velocity: DeviceBuffer::from_host(stream, state.velocity)?,
        })
    }
}

#[cfg(feature = "cuda")]
pub type AdamWOptimizerState = MomentumVelocityDeviceState;

#[cfg(feature = "cuda")]
pub type RAdamOptimizerState = MomentumVelocityDeviceState;

#[derive(Debug, Clone, Copy)]
pub struct RangerOptimizerHostState<'a> {
    pub momentum: &'a [f32],
    pub velocity: &'a [f32],
    pub slow_params: &'a [f32],
}

impl<'a> RangerOptimizerHostState<'a> {
    pub fn validate(self, layout: OptimizerStateLayout) -> std::result::Result<(), OptimizerLayoutError> {
        layout.validate()?;
        expect_state_len("momentum", layout.state_len(), self.momentum.len())?;
        expect_state_len("velocity", layout.state_len(), self.velocity.len())?;
        expect_state_len("slow_params", layout.state_len(), self.slow_params.len())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NnueRangerOptimizerHostStates<'a> {
    pub l0w: RangerOptimizerHostState<'a>,
    pub l0b: RangerOptimizerHostState<'a>,
    pub l1w: RangerOptimizerHostState<'a>,
    pub l1b: RangerOptimizerHostState<'a>,
    pub l2w: RangerOptimizerHostState<'a>,
    pub l2b: RangerOptimizerHostState<'a>,
    pub outw: RangerOptimizerHostState<'a>,
    pub outb: RangerOptimizerHostState<'a>,
}

impl<'a> NnueRangerOptimizerHostStates<'a> {
    pub fn validate(self, layout: NnueOptimizerStateLayout) -> std::result::Result<(), OptimizerLayoutError> {
        self.l0w.validate(layout.l0w_state_layout())?;
        self.l0b.validate(layout.l0b_state_layout())?;
        self.l1w.validate(layout.l1w_state_layout())?;
        self.l1b.validate(layout.l1b_state_layout())?;
        self.l2w.validate(layout.l2w_state_layout())?;
        self.l2b.validate(layout.l2b_state_layout())?;
        self.outw.validate(layout.outw_state_layout())?;
        self.outb.validate(layout.outb_state_layout())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnRangerOptimizerHostStates<'a> {
    pub l0w: RangerOptimizerHostState<'a>,
    pub l0b: RangerOptimizerHostState<'a>,
    pub l1w: RangerOptimizerHostState<'a>,
    pub l1b: RangerOptimizerHostState<'a>,
    pub l1fw: Option<RangerOptimizerHostState<'a>>,
    pub l1fb: Option<RangerOptimizerHostState<'a>>,
    pub l2w: RangerOptimizerHostState<'a>,
    pub l2b: RangerOptimizerHostState<'a>,
    pub l3w: RangerOptimizerHostState<'a>,
    pub l3b: RangerOptimizerHostState<'a>,
}

impl<'a> SfnnRangerOptimizerHostStates<'a> {
    pub fn validate(self, layout: SfnnOptimizerStateLayout) -> std::result::Result<(), OptimizerLayoutError> {
        self.l0w.validate(layout.l0w_state_layout())?;
        self.l0b.validate(layout.l0b_state_layout())?;
        self.l1w.validate(layout.l1w_state_layout())?;
        self.l1b.validate(layout.l1b_state_layout())?;
        if let Some(state) = self.l1fw {
            state.validate(layout.l1fw_state_layout())?;
        }
        if let Some(state) = self.l1fb {
            state.validate(layout.l1fb_state_layout())?;
        }
        self.l2w.validate(layout.l2w_state_layout())?;
        self.l2b.validate(layout.l2b_state_layout())?;
        self.l3w.validate(layout.l3w_state_layout())?;
        self.l3b.validate(layout.l3b_state_layout())?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct RangerOptimizerState {
    pub layout: OptimizerStateLayout,
    pub momentum: DeviceBuffer<f32>,
    pub velocity: DeviceBuffer<f32>,
    pub slow_params: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl RangerOptimizerState {
    pub fn new_zeroed(stream: &CudaStream, layout: OptimizerStateLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            momentum: DeviceBuffer::zeroed(stream, layout.state_len())?,
            velocity: DeviceBuffer::zeroed(stream, layout.state_len())?,
            slow_params: DeviceBuffer::zeroed(stream, layout.state_len())?,
        })
    }

    pub fn from_host(
        stream: &CudaStream,
        layout: OptimizerStateLayout,
        state: RangerOptimizerHostState<'_>,
    ) -> Result<Self> {
        state.validate(layout)?;
        Ok(Self {
            layout,
            momentum: DeviceBuffer::from_host(stream, state.momentum)?,
            velocity: DeviceBuffer::from_host(stream, state.velocity)?,
            slow_params: DeviceBuffer::from_host(stream, state.slow_params)?,
        })
    }

    pub fn zeroed_with_host_slow_params(
        stream: &CudaStream,
        layout: OptimizerStateLayout,
        slow_params: &[f32],
    ) -> Result<Self> {
        layout.validate()?;
        expect_state_len("slow_params", layout.state_len(), slow_params.len())?;
        Ok(Self {
            layout,
            momentum: DeviceBuffer::zeroed(stream, layout.state_len())?,
            velocity: DeviceBuffer::zeroed(stream, layout.state_len())?,
            slow_params: DeviceBuffer::from_host(stream, slow_params)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueOptimizerStateLayout {
    pub weights: NnueForwardWeightLayout,
}

impl NnueOptimizerStateLayout {
    pub fn new(weights: NnueForwardWeightLayout) -> Self {
        Self { weights }
    }

    pub fn l0w_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l0w_len())
    }

    pub fn l0b_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l0b_len())
    }

    pub fn l1w_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l1w_len())
    }

    pub fn l1b_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l1b_len())
    }

    pub fn l2w_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l2w_len())
    }

    pub fn l2b_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l2b_len())
    }

    pub fn outw_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.outw_len())
    }

    pub fn outb_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.outb_len())
    }

    pub fn parameter_f32_len(self) -> usize {
        self.weights
            .l0w_len()
            .saturating_add(self.weights.l0b_len())
            .saturating_add(self.weights.l1w_len())
            .saturating_add(self.weights.l1b_len())
            .saturating_add(self.weights.l2w_len())
            .saturating_add(self.weights.l2b_len())
            .saturating_add(self.weights.outw_len())
            .saturating_add(self.weights.outb_len())
    }

    pub fn momentum_velocity_state_f32_len(self) -> usize {
        self.parameter_f32_len().saturating_mul(2)
    }

    pub fn ranger_state_f32_len(self) -> usize {
        self.parameter_f32_len().saturating_mul(3)
    }
}

#[cfg(feature = "cuda")]
pub struct NnueRangerOptimizerStates {
    pub layout: NnueOptimizerStateLayout,
    pub l0w: RangerOptimizerState,
    pub l0b: RangerOptimizerState,
    pub l1w: RangerOptimizerState,
    pub l1b: RangerOptimizerState,
    pub l2w: RangerOptimizerState,
    pub l2b: RangerOptimizerState,
    pub outw: RangerOptimizerState,
    pub outb: RangerOptimizerState,
}

#[cfg(feature = "cuda")]
impl NnueRangerOptimizerStates {
    pub fn new_zeroed(stream: &CudaStream, weights: NnueForwardWeightLayout) -> Result<Self> {
        let layout = NnueOptimizerStateLayout::new(weights);
        Ok(Self {
            layout,
            l0w: RangerOptimizerState::new_zeroed(stream, layout.l0w_state_layout())?,
            l0b: RangerOptimizerState::new_zeroed(stream, layout.l0b_state_layout())?,
            l1w: RangerOptimizerState::new_zeroed(stream, layout.l1w_state_layout())?,
            l1b: RangerOptimizerState::new_zeroed(stream, layout.l1b_state_layout())?,
            l2w: RangerOptimizerState::new_zeroed(stream, layout.l2w_state_layout())?,
            l2b: RangerOptimizerState::new_zeroed(stream, layout.l2b_state_layout())?,
            outw: RangerOptimizerState::new_zeroed(stream, layout.outw_state_layout())?,
            outb: RangerOptimizerState::new_zeroed(stream, layout.outb_state_layout())?,
        })
    }

    pub fn from_host_weights(stream: &CudaStream, weights: &NnueForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        let layout = NnueOptimizerStateLayout::new(NnueForwardWeightLayout::new(weights.shape));
        Ok(Self {
            layout,
            l0w: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l0w_state_layout(), weights.l0w)?,
            l0b: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l0b_state_layout(), weights.l0b)?,
            l1w: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l1w_state_layout(), weights.l1w)?,
            l1b: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l1b_state_layout(), weights.l1b)?,
            l2w: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l2w_state_layout(), weights.l2w)?,
            l2b: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l2b_state_layout(), weights.l2b)?,
            outw: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.outw_state_layout(), weights.outw)?,
            outb: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.outb_state_layout(), weights.outb)?,
        })
    }

    pub fn from_host_states(
        stream: &CudaStream,
        weights: NnueForwardWeightLayout,
        states: NnueRangerOptimizerHostStates<'_>,
    ) -> Result<Self> {
        let layout = NnueOptimizerStateLayout::new(weights);
        states.validate(layout)?;
        Ok(Self {
            layout,
            l0w: RangerOptimizerState::from_host(stream, layout.l0w_state_layout(), states.l0w)?,
            l0b: RangerOptimizerState::from_host(stream, layout.l0b_state_layout(), states.l0b)?,
            l1w: RangerOptimizerState::from_host(stream, layout.l1w_state_layout(), states.l1w)?,
            l1b: RangerOptimizerState::from_host(stream, layout.l1b_state_layout(), states.l1b)?,
            l2w: RangerOptimizerState::from_host(stream, layout.l2w_state_layout(), states.l2w)?,
            l2b: RangerOptimizerState::from_host(stream, layout.l2b_state_layout(), states.l2b)?,
            outw: RangerOptimizerState::from_host(stream, layout.outw_state_layout(), states.outw)?,
            outb: RangerOptimizerState::from_host(stream, layout.outb_state_layout(), states.outb)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnOptimizerStateLayout {
    pub weights: SfnnForwardWeightLayout,
}

impl SfnnOptimizerStateLayout {
    pub fn new(weights: SfnnForwardWeightLayout) -> Self {
        Self { weights }
    }

    pub fn l0w_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l0w_len())
    }

    pub fn l0b_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l0b_len())
    }

    pub fn l1w_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l1w_len())
    }

    pub fn l1b_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l1b_len())
    }

    pub fn l1fw_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l1fw_len())
    }

    pub fn l1fb_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l1fb_len())
    }

    pub fn l2w_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l2w_len())
    }

    pub fn l2b_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l2b_len())
    }

    pub fn l3w_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l3w_len())
    }

    pub fn l3b_state_layout(self) -> OptimizerStateLayout {
        OptimizerStateLayout::new(self.weights.l3b_len())
    }

    pub fn parameter_f32_len(self) -> usize {
        self.weights
            .l0w_len()
            .saturating_add(self.weights.l0b_len())
            .saturating_add(self.weights.l1w_len())
            .saturating_add(self.weights.l1b_len())
            .saturating_add(self.weights.l2w_len())
            .saturating_add(self.weights.l2b_len())
            .saturating_add(self.weights.l3w_len())
            .saturating_add(self.weights.l3b_len())
    }

    pub fn momentum_velocity_state_f32_len(self) -> usize {
        self.parameter_f32_len().saturating_mul(2)
    }

    pub fn ranger_state_f32_len(self) -> usize {
        self.parameter_f32_len().saturating_mul(3)
    }
}

#[cfg(feature = "cuda")]
pub struct SfnnRangerOptimizerStates {
    pub layout: SfnnOptimizerStateLayout,
    pub l0w: RangerOptimizerState,
    pub l0b: RangerOptimizerState,
    pub l1w: RangerOptimizerState,
    pub l1b: RangerOptimizerState,
    pub l1fw: Option<RangerOptimizerState>,
    pub l1fb: Option<RangerOptimizerState>,
    pub l2w: RangerOptimizerState,
    pub l2b: RangerOptimizerState,
    pub l3w: RangerOptimizerState,
    pub l3b: RangerOptimizerState,
}

#[cfg(feature = "cuda")]
impl SfnnRangerOptimizerStates {
    pub fn new_zeroed(stream: &CudaStream, weights: SfnnForwardWeightLayout) -> Result<Self> {
        let layout = SfnnOptimizerStateLayout::new(weights);
        Ok(Self {
            layout,
            l0w: RangerOptimizerState::new_zeroed(stream, layout.l0w_state_layout())?,
            l0b: RangerOptimizerState::new_zeroed(stream, layout.l0b_state_layout())?,
            l1w: RangerOptimizerState::new_zeroed(stream, layout.l1w_state_layout())?,
            l1b: RangerOptimizerState::new_zeroed(stream, layout.l1b_state_layout())?,
            l1fw: None,
            l1fb: None,
            l2w: RangerOptimizerState::new_zeroed(stream, layout.l2w_state_layout())?,
            l2b: RangerOptimizerState::new_zeroed(stream, layout.l2b_state_layout())?,
            l3w: RangerOptimizerState::new_zeroed(stream, layout.l3w_state_layout())?,
            l3b: RangerOptimizerState::new_zeroed(stream, layout.l3b_state_layout())?,
        })
    }

    pub fn zeroed_for_host_weights(stream: &CudaStream, weights: &SfnnForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        let layout = SfnnOptimizerStateLayout::new(SfnnForwardWeightLayout::new(weights.shape));
        Ok(Self {
            layout,
            l0w: RangerOptimizerState::new_zeroed(stream, layout.l0w_state_layout())?,
            l0b: RangerOptimizerState::new_zeroed(stream, layout.l0b_state_layout())?,
            l1w: RangerOptimizerState::new_zeroed(stream, layout.l1w_state_layout())?,
            l1b: RangerOptimizerState::new_zeroed(stream, layout.l1b_state_layout())?,
            l1fw: match weights.l1fw {
                Some(_) => Some(RangerOptimizerState::new_zeroed(stream, layout.l1fw_state_layout())?),
                None => None,
            },
            l1fb: match weights.l1fb {
                Some(_) => Some(RangerOptimizerState::new_zeroed(stream, layout.l1fb_state_layout())?),
                None => None,
            },
            l2w: RangerOptimizerState::new_zeroed(stream, layout.l2w_state_layout())?,
            l2b: RangerOptimizerState::new_zeroed(stream, layout.l2b_state_layout())?,
            l3w: RangerOptimizerState::new_zeroed(stream, layout.l3w_state_layout())?,
            l3b: RangerOptimizerState::new_zeroed(stream, layout.l3b_state_layout())?,
        })
    }

    pub fn from_host_weights(stream: &CudaStream, weights: &SfnnForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        let layout = SfnnOptimizerStateLayout::new(SfnnForwardWeightLayout::new(weights.shape));
        Ok(Self {
            layout,
            l0w: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l0w_state_layout(), weights.l0w)?,
            l0b: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l0b_state_layout(), weights.l0b)?,
            l1w: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l1w_state_layout(), weights.l1w)?,
            l1b: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l1b_state_layout(), weights.l1b)?,
            l1fw: match weights.l1fw {
                Some(values) => Some(RangerOptimizerState::zeroed_with_host_slow_params(
                    stream,
                    layout.l1fw_state_layout(),
                    values,
                )?),
                None => None,
            },
            l1fb: match weights.l1fb {
                Some(values) => Some(RangerOptimizerState::zeroed_with_host_slow_params(
                    stream,
                    layout.l1fb_state_layout(),
                    values,
                )?),
                None => None,
            },
            l2w: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l2w_state_layout(), weights.l2w)?,
            l2b: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l2b_state_layout(), weights.l2b)?,
            l3w: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l3w_state_layout(), weights.l3w)?,
            l3b: RangerOptimizerState::zeroed_with_host_slow_params(stream, layout.l3b_state_layout(), weights.l3b)?,
        })
    }

    pub fn from_host_states(
        stream: &CudaStream,
        weights: SfnnForwardWeightLayout,
        states: SfnnRangerOptimizerHostStates<'_>,
    ) -> Result<Self> {
        let layout = SfnnOptimizerStateLayout::new(weights);
        states.validate(layout)?;
        Ok(Self {
            layout,
            l0w: RangerOptimizerState::from_host(stream, layout.l0w_state_layout(), states.l0w)?,
            l0b: RangerOptimizerState::from_host(stream, layout.l0b_state_layout(), states.l0b)?,
            l1w: RangerOptimizerState::from_host(stream, layout.l1w_state_layout(), states.l1w)?,
            l1b: RangerOptimizerState::from_host(stream, layout.l1b_state_layout(), states.l1b)?,
            l1fw: match states.l1fw {
                Some(state) => Some(RangerOptimizerState::from_host(stream, layout.l1fw_state_layout(), state)?),
                None => None,
            },
            l1fb: match states.l1fb {
                Some(state) => Some(RangerOptimizerState::from_host(stream, layout.l1fb_state_layout(), state)?),
                None => None,
            },
            l2w: RangerOptimizerState::from_host(stream, layout.l2w_state_layout(), states.l2w)?,
            l2b: RangerOptimizerState::from_host(stream, layout.l2b_state_layout(), states.l2b)?,
            l3w: RangerOptimizerState::from_host(stream, layout.l3w_state_layout(), states.l3w)?,
            l3b: RangerOptimizerState::from_host(stream, layout.l3b_state_layout(), states.l3b)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RAdamUpdateParams {
    pub gradient_factor: f32,
    pub learning_rate: f32,
    pub step: usize,
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub n_sma_threshold: f32,
    pub epsilon: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for RAdamUpdateParams {
    fn default() -> Self {
        Self {
            gradient_factor: 1.0,
            learning_rate: 0.001,
            step: 1,
            decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            n_sma_threshold: 5.0,
            epsilon: 0.00000001,
            min_weight: f32::MIN,
            max_weight: f32::MAX,
        }
    }
}

impl RAdamUpdateParams {
    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if !self.gradient_factor.is_finite() {
            Err(OptimizerLayoutError::NonFiniteParam { name: "gradient_factor" })
        } else if !(self.learning_rate.is_finite() && self.learning_rate >= 0.0) {
            Err(OptimizerLayoutError::InvalidLearningRate)
        } else if self.step == 0 {
            Err(OptimizerLayoutError::InvalidStep)
        } else if !self.decay.is_finite() {
            Err(OptimizerLayoutError::NonFiniteParam { name: "decay" })
        } else if !(self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1)) {
            Err(OptimizerLayoutError::InvalidBeta { name: "beta1" })
        } else if !(self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2)) {
            Err(OptimizerLayoutError::InvalidBeta { name: "beta2" })
        } else if !(self.n_sma_threshold.is_finite() && self.n_sma_threshold >= 0.0) {
            Err(OptimizerLayoutError::InvalidNSmaThreshold)
        } else if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            Err(OptimizerLayoutError::InvalidEpsilon)
        } else if !(self.min_weight.is_finite() && self.max_weight.is_finite() && self.min_weight <= self.max_weight) {
            Err(OptimizerLayoutError::InvalidClamp)
        } else {
            let step = self.step_scale()?;
            if step.step_size.is_finite() {
                Ok(())
            } else {
                Err(OptimizerLayoutError::NonFiniteParam { name: "step_size" })
            }
        }
    }

    pub fn step_scale(self) -> std::result::Result<RAdamStepScale, OptimizerLayoutError> {
        if self.step == 0 {
            return Err(OptimizerLayoutError::InvalidStep);
        }

        let step = self.step as f32;
        let beta2_t = self.beta2.powf(step);
        let n_sma_max = 2.0 / (1.0 - self.beta2) - 1.0;
        let n_sma = n_sma_max - 2.0 * step * beta2_t / (1.0 - beta2_t);

        let denom = 1.0 - self.beta1.powf(step);
        let use_denom = n_sma > self.n_sma_threshold;
        let step_size = if use_denom {
            let p1 = (n_sma - 4.0) / (n_sma_max - 4.0);
            let p2 = (n_sma - 2.0) / n_sma;
            let p3 = n_sma_max / (n_sma_max - 2.0);
            ((1.0 - beta2_t) * p1 * p2 * p3).sqrt() / denom
        } else {
            1.0 / denom
        };

        Ok(RAdamStepScale { step_size, use_denom })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RAdamStepScale {
    pub step_size: f32,
    pub use_denom: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangerLookaheadParams {
    pub alpha: f32,
}

impl Default for RangerLookaheadParams {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl RangerLookaheadParams {
    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if self.alpha.is_finite() && (0.0..=1.0).contains(&self.alpha) {
            Ok(())
        } else {
            Err(OptimizerLayoutError::InvalidAlpha)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangerUpdateParams {
    pub radam: RAdamUpdateParams,
    pub lookahead: RangerLookaheadParams,
    pub k: usize,
}

impl Default for RangerUpdateParams {
    fn default() -> Self {
        Self {
            radam: RAdamUpdateParams {
                decay: 0.01,
                beta1: 0.99,
                beta2: 0.999,
                min_weight: -1.98,
                max_weight: 1.98,
                ..Default::default()
            },
            lookahead: RangerLookaheadParams::default(),
            k: 6,
        }
    }
}

impl RangerUpdateParams {
    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        self.radam.validate()?;
        self.lookahead.validate()?;
        if self.k == 0 { Err(OptimizerLayoutError::InvalidLookaheadPeriod) } else { Ok(()) }
    }

    pub fn should_lookahead(self) -> std::result::Result<bool, OptimizerLayoutError> {
        self.validate()?;
        Ok(self.radam.step.is_multiple_of(self.k))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdamWUpdateLayout {
    pub len: usize,
}

impl AdamWUpdateLayout {
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if self.len == 0 { Err(OptimizerLayoutError::EmptyParameters) } else { Ok(()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdamWUpdateLaunchPlan {
    pub threads: usize,
}

impl AdamWUpdateLaunchPlan {
    pub fn new(layout: AdamWUpdateLayout) -> Self {
        Self { threads: layout.len.max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RAdamUpdateLayout {
    pub len: usize,
}

impl RAdamUpdateLayout {
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if self.len == 0 { Err(OptimizerLayoutError::EmptyParameters) } else { Ok(()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RAdamUpdateLaunchPlan {
    pub threads: usize,
}

impl RAdamUpdateLaunchPlan {
    pub fn new(layout: RAdamUpdateLayout) -> Self {
        Self { threads: layout.len.max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangerLookaheadLayout {
    pub len: usize,
}

impl RangerLookaheadLayout {
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if self.len == 0 { Err(OptimizerLayoutError::EmptyParameters) } else { Ok(()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangerLookaheadLaunchPlan {
    pub threads: usize,
}

impl RangerLookaheadLaunchPlan {
    pub fn new(layout: RangerLookaheadLayout) -> Self {
        Self { threads: layout.len.max(1) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangerUpdateLayout {
    pub len: usize,
}

impl RangerUpdateLayout {
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    pub fn validate(self) -> std::result::Result<(), OptimizerLayoutError> {
        if self.len == 0 { Err(OptimizerLayoutError::EmptyParameters) } else { Ok(()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangerUpdateLaunchPlan {
    pub threads: usize,
}

impl RangerUpdateLaunchPlan {
    pub fn new(layout: RangerUpdateLayout) -> Self {
        Self { threads: layout.len.max(1) }
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum OptimizerLayoutError {
    #[error("optimizer parameter buffer must contain at least one element")]
    EmptyParameters,
    #[error("optimizer learning rate must be finite and non-negative")]
    InvalidLearningRate,
    #[error("optimizer step must be greater than zero")]
    InvalidStep,
    #[error("optimizer {name} must be finite and in [0, 1)")]
    InvalidBeta { name: &'static str },
    #[error("optimizer lookahead alpha must be finite and in [0, 1]")]
    InvalidAlpha,
    #[error("optimizer lookahead period k must be greater than zero")]
    InvalidLookaheadPeriod,
    #[error("optimizer n_sma_threshold must be finite and non-negative")]
    InvalidNSmaThreshold,
    #[error("optimizer epsilon must be finite and positive")]
    InvalidEpsilon,
    #[error("optimizer clamp range must be finite and min <= max")]
    InvalidClamp,
    #[error("optimizer state length mismatch for {name}: expected {expected}, got {actual}")]
    StateLength { name: &'static str, expected: usize, actual: usize },
    #[error("optimizer parameter {name} must be finite")]
    NonFiniteParam { name: &'static str },
}

fn expect_state_len(
    name: &'static str,
    expected: usize,
    actual: usize,
) -> std::result::Result<(), OptimizerLayoutError> {
    if expected == actual {
        Ok(())
    } else {
        Err(OptimizerLayoutError::StateLength { name, expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nnue::{NnueForwardShape, NnueForwardWeightLayout},
        sfnn::{SfnnForwardShape, SfnnForwardWeightLayout},
    };

    #[test]
    fn optimizer_kernel_names_are_stable() {
        assert_eq!(OPTIMIZER_KERNEL_NAMES, ["adamw_update", "radam_update", "ranger_lookahead"]);
    }

    #[test]
    fn adamw_layout_counts_threads() {
        let layout = AdamWUpdateLayout::new(17);

        assert_eq!(AdamWUpdateLaunchPlan::new(layout).threads, 17);
    }

    #[test]
    fn adamw_layout_rejects_empty_parameters() {
        assert_eq!(AdamWUpdateLayout::new(0).validate().unwrap_err(), OptimizerLayoutError::EmptyParameters);
    }

    #[test]
    fn adamw_params_validate_defaults() {
        AdamWUpdateParams::default().validate().unwrap();
    }

    #[test]
    fn adamw_params_reject_invalid_values() {
        assert_eq!(
            AdamWUpdateParams { learning_rate: -1.0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidLearningRate
        );
        assert_eq!(
            AdamWUpdateParams { beta1: 1.0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidBeta { name: "beta1" }
        );
        assert_eq!(
            AdamWUpdateParams { epsilon: 0.0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidEpsilon
        );
        assert_eq!(
            AdamWUpdateParams { min_weight: 2.0, max_weight: 1.0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidClamp
        );
    }

    #[test]
    fn optimizer_state_layout_rejects_empty_parameters() {
        assert_eq!(OptimizerStateLayout::new(0).validate().unwrap_err(), OptimizerLayoutError::EmptyParameters);
    }

    #[test]
    fn momentum_velocity_host_state_validates_lengths() {
        let layout = OptimizerStateLayout::new(3);
        MomentumVelocityHostState { momentum: &[0.0; 3], velocity: &[0.0; 3] }.validate(layout).unwrap();

        assert_eq!(
            MomentumVelocityHostState { momentum: &[0.0; 2], velocity: &[0.0; 3] }.validate(layout).unwrap_err(),
            OptimizerLayoutError::StateLength { name: "momentum", expected: 3, actual: 2 }
        );
    }

    #[test]
    fn ranger_optimizer_host_state_validates_lengths() {
        let layout = OptimizerStateLayout::new(3);
        RangerOptimizerHostState { momentum: &[0.0; 3], velocity: &[0.0; 3], slow_params: &[0.0; 3] }
            .validate(layout)
            .unwrap();

        assert_eq!(
            RangerOptimizerHostState { momentum: &[0.0; 3], velocity: &[0.0; 3], slow_params: &[0.0; 2] }
                .validate(layout)
                .unwrap_err(),
            OptimizerLayoutError::StateLength { name: "slow_params", expected: 3, actual: 2 }
        );
    }

    #[test]
    fn nnue_optimizer_state_layout_counts_parameter_groups() {
        let weights = NnueForwardWeightLayout::new(NnueForwardShape { input_size: 4, l1: 2, l2: 3, l3: 1 });
        let layout = NnueOptimizerStateLayout::new(weights);

        assert_eq!(layout.l0w_state_layout().state_len(), 8);
        assert_eq!(layout.l0b_state_layout().state_len(), 2);
        assert_eq!(layout.l1w_state_layout().state_len(), 12);
        assert_eq!(layout.l1b_state_layout().state_len(), 3);
        assert_eq!(layout.l2w_state_layout().state_len(), 3);
        assert_eq!(layout.l2b_state_layout().state_len(), 1);
        assert_eq!(layout.outw_state_layout().state_len(), 1);
        assert_eq!(layout.outb_state_layout().state_len(), 1);
        assert_eq!(layout.parameter_f32_len(), 31);
        assert_eq!(layout.momentum_velocity_state_f32_len(), 62);
        assert_eq!(layout.ranger_state_f32_len(), 93);
    }

    #[test]
    fn nnue_ranger_host_states_validate_group_lengths() {
        let weights = NnueForwardWeightLayout::new(NnueForwardShape { input_size: 4, l1: 2, l2: 3, l3: 1 });
        let layout = NnueOptimizerStateLayout::new(weights);
        let l0w_state = RangerOptimizerHostState { momentum: &[0.0; 8], velocity: &[0.0; 8], slow_params: &[0.0; 8] };
        let l0b_state = RangerOptimizerHostState { momentum: &[0.0; 2], velocity: &[0.0; 2], slow_params: &[0.0; 2] };
        let l1w_state =
            RangerOptimizerHostState { momentum: &[0.0; 12], velocity: &[0.0; 12], slow_params: &[0.0; 12] };
        let l1b_state = RangerOptimizerHostState { momentum: &[0.0; 3], velocity: &[0.0; 3], slow_params: &[0.0; 3] };
        let len1_state = RangerOptimizerHostState { momentum: &[0.0; 1], velocity: &[0.0; 1], slow_params: &[0.0; 1] };

        NnueRangerOptimizerHostStates {
            l0w: l0w_state,
            l0b: l0b_state,
            l1w: l1w_state,
            l1b: l1b_state,
            l2w: l1b_state,
            l2b: len1_state,
            outw: len1_state,
            outb: len1_state,
        }
        .validate(layout)
        .unwrap();

        assert_eq!(
            NnueRangerOptimizerHostStates {
                l0w: RangerOptimizerHostState { momentum: &[0.0; 7], velocity: &[0.0; 8], slow_params: &[0.0; 8] },
                l0b: l0b_state,
                l1w: l1w_state,
                l1b: l1b_state,
                l2w: l1b_state,
                l2b: len1_state,
                outw: len1_state,
                outb: len1_state,
            }
            .validate(layout)
            .unwrap_err(),
            OptimizerLayoutError::StateLength { name: "momentum", expected: 8, actual: 7 }
        );
    }

    #[test]
    fn sfnn_optimizer_state_layout_counts_parameter_groups() {
        let weights = SfnnForwardWeightLayout::new(SfnnForwardShape {
            input_size: 4,
            ft_size: 4,
            l1_hidden: 2,
            l2_size: 3,
            num_stacks: 2,
        });
        let layout = SfnnOptimizerStateLayout::new(weights);

        assert_eq!(layout.l0w_state_layout().state_len(), 16);
        assert_eq!(layout.l0b_state_layout().state_len(), 4);
        assert_eq!(layout.l1w_state_layout().state_len(), 24);
        assert_eq!(layout.l1b_state_layout().state_len(), 6);
        assert_eq!(layout.l2w_state_layout().state_len(), 24);
        assert_eq!(layout.l2b_state_layout().state_len(), 6);
        assert_eq!(layout.l3w_state_layout().state_len(), 6);
        assert_eq!(layout.l3b_state_layout().state_len(), 2);
        assert_eq!(layout.parameter_f32_len(), 88);
        assert_eq!(layout.momentum_velocity_state_f32_len(), 176);
        assert_eq!(layout.ranger_state_f32_len(), 264);
    }

    #[test]
    fn radam_layout_counts_threads() {
        let layout = RAdamUpdateLayout::new(19);

        assert_eq!(RAdamUpdateLaunchPlan::new(layout).threads, 19);
    }

    #[test]
    fn radam_layout_rejects_empty_parameters() {
        assert_eq!(RAdamUpdateLayout::new(0).validate().unwrap_err(), OptimizerLayoutError::EmptyParameters);
    }

    #[test]
    fn radam_params_validate_defaults() {
        RAdamUpdateParams::default().validate().unwrap();
    }

    #[test]
    fn radam_step_scale_matches_warmup_branch() {
        let scale = RAdamUpdateParams { step: 1, ..Default::default() }.step_scale().unwrap();

        assert!(!scale.use_denom);
        assert!((scale.step_size - 10.0).abs() < 0.00001);
    }

    #[test]
    fn radam_step_scale_uses_denominator_after_threshold() {
        let scale = RAdamUpdateParams { step: 6, ..Default::default() }.step_scale().unwrap();

        assert!(scale.use_denom);
        assert!(scale.step_size.is_finite());
        assert!(scale.step_size > 0.0);
    }

    #[test]
    fn radam_params_reject_invalid_values() {
        assert_eq!(
            RAdamUpdateParams { step: 0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidStep
        );
        assert_eq!(
            RAdamUpdateParams { n_sma_threshold: -1.0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidNSmaThreshold
        );
        assert_eq!(
            RAdamUpdateParams { beta2: 1.0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidBeta { name: "beta2" }
        );
    }

    #[test]
    fn ranger_lookahead_layout_counts_threads() {
        let layout = RangerLookaheadLayout::new(23);

        assert_eq!(RangerLookaheadLaunchPlan::new(layout).threads, 23);
    }

    #[test]
    fn ranger_lookahead_layout_rejects_empty_parameters() {
        assert_eq!(RangerLookaheadLayout::new(0).validate().unwrap_err(), OptimizerLayoutError::EmptyParameters);
    }

    #[test]
    fn ranger_lookahead_params_validate_defaults() {
        RangerLookaheadParams::default().validate().unwrap();
    }

    #[test]
    fn ranger_lookahead_params_reject_invalid_alpha() {
        assert_eq!(RangerLookaheadParams { alpha: -0.1 }.validate().unwrap_err(), OptimizerLayoutError::InvalidAlpha);
        assert_eq!(RangerLookaheadParams { alpha: 1.1 }.validate().unwrap_err(), OptimizerLayoutError::InvalidAlpha);
    }

    #[test]
    fn ranger_update_layout_counts_threads() {
        let layout = RangerUpdateLayout::new(29);

        assert_eq!(RangerUpdateLaunchPlan::new(layout).threads, 29);
    }

    #[test]
    fn ranger_update_layout_rejects_empty_parameters() {
        assert_eq!(RangerUpdateLayout::new(0).validate().unwrap_err(), OptimizerLayoutError::EmptyParameters);
    }

    #[test]
    fn ranger_update_params_validate_defaults() {
        RangerUpdateParams::default().validate().unwrap();
    }

    #[test]
    fn ranger_update_params_reject_invalid_k() {
        assert_eq!(
            RangerUpdateParams { k: 0, ..Default::default() }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidLookaheadPeriod
        );
    }

    #[test]
    fn ranger_update_params_detect_lookahead_step() {
        assert!(
            !RangerUpdateParams {
                radam: RAdamUpdateParams { step: 5, ..Default::default() },
                k: 3,
                ..Default::default()
            }
            .should_lookahead()
            .unwrap()
        );
        assert!(
            RangerUpdateParams {
                radam: RAdamUpdateParams { step: 6, ..Default::default() },
                k: 3,
                ..Default::default()
            }
            .should_lookahead()
            .unwrap()
        );
    }
}
