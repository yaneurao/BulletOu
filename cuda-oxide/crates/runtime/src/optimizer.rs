//! Optimizer workspaces and launch layouts for cuda-oxide kernels.

pub const ADAMW_UPDATE_KERNEL: &str = "adamw_update";
pub const OPTIMIZER_KERNEL_NAMES: [&str; 1] = [ADAMW_UPDATE_KERNEL];

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
        } else if !(self.min_weight.is_finite()
            && self.max_weight.is_finite()
            && self.min_weight <= self.max_weight)
        {
            Err(OptimizerLayoutError::InvalidClamp)
        } else {
            Ok(())
        }
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
        if self.len == 0 {
            Err(OptimizerLayoutError::EmptyParameters)
        } else {
            Ok(())
        }
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

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum OptimizerLayoutError {
    #[error("optimizer parameter buffer must contain at least one element")]
    EmptyParameters,
    #[error("optimizer learning rate must be finite and non-negative")]
    InvalidLearningRate,
    #[error("optimizer {name} must be finite and in [0, 1)")]
    InvalidBeta { name: &'static str },
    #[error("optimizer epsilon must be finite and positive")]
    InvalidEpsilon,
    #[error("optimizer clamp range must be finite and min <= max")]
    InvalidClamp,
    #[error("optimizer parameter {name} must be finite")]
    NonFiniteParam { name: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_kernel_names_are_stable() {
        assert_eq!(OPTIMIZER_KERNEL_NAMES, ["adamw_update"]);
    }

    #[test]
    fn adamw_layout_counts_threads() {
        let layout = AdamWUpdateLayout::new(17);

        assert_eq!(AdamWUpdateLaunchPlan::new(layout).threads, 17);
    }

    #[test]
    fn adamw_layout_rejects_empty_parameters() {
        assert_eq!(
            AdamWUpdateLayout::new(0).validate().unwrap_err(),
            OptimizerLayoutError::EmptyParameters
        );
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
            AdamWUpdateParams { min_weight: 2.0, max_weight: 1.0, ..Default::default() }
                .validate()
                .unwrap_err(),
            OptimizerLayoutError::InvalidClamp
        );
    }
}
