//! Optimizer workspaces and launch layouts for cuda-oxide kernels.

pub const ADAMW_UPDATE_KERNEL: &str = "adamw_update";
pub const RADAM_UPDATE_KERNEL: &str = "radam_update";
pub const RANGER_LOOKAHEAD_KERNEL: &str = "ranger_lookahead";
pub const OPTIMIZER_KERNEL_NAMES: [&str; 3] =
    [ADAMW_UPDATE_KERNEL, RADAM_UPDATE_KERNEL, RANGER_LOOKAHEAD_KERNEL];

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
        } else if !(self.min_weight.is_finite()
            && self.max_weight.is_finite()
            && self.min_weight <= self.max_weight)
        {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RAdamUpdateLayout {
    pub len: usize,
}

impl RAdamUpdateLayout {
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
        if self.len == 0 {
            Err(OptimizerLayoutError::EmptyParameters)
        } else {
            Ok(())
        }
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
    #[error("optimizer n_sma_threshold must be finite and non-negative")]
    InvalidNSmaThreshold,
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
        assert_eq!(OPTIMIZER_KERNEL_NAMES, ["adamw_update", "radam_update", "ranger_lookahead"]);
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

    #[test]
    fn radam_layout_counts_threads() {
        let layout = RAdamUpdateLayout::new(19);

        assert_eq!(RAdamUpdateLaunchPlan::new(layout).threads, 19);
    }

    #[test]
    fn radam_layout_rejects_empty_parameters() {
        assert_eq!(
            RAdamUpdateLayout::new(0).validate().unwrap_err(),
            OptimizerLayoutError::EmptyParameters
        );
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
        assert_eq!(
            RangerLookaheadLayout::new(0).validate().unwrap_err(),
            OptimizerLayoutError::EmptyParameters
        );
    }

    #[test]
    fn ranger_lookahead_params_validate_defaults() {
        RangerLookaheadParams::default().validate().unwrap();
    }

    #[test]
    fn ranger_lookahead_params_reject_invalid_alpha() {
        assert_eq!(
            RangerLookaheadParams { alpha: -0.1 }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidAlpha
        );
        assert_eq!(
            RangerLookaheadParams { alpha: 1.1 }.validate().unwrap_err(),
            OptimizerLayoutError::InvalidAlpha
        );
    }
}
