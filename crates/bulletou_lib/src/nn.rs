use bullet_trainer::model::Model as TrainerModel;

pub use bullet_trainer::model::{
    Shape,
    builder::{Affine, InitSettings, ModelBuilder, ModelNode},
};

/// Legacy Bullet model-builder examples still type-check through the mock
/// runtime. Production GPU training is handled by BulletOu's `cuda-cpp`
/// backend, not by the old `bullet-gpu` hardware runtime.
pub type ExecutionContext = bullet_gpu::runtime::mock::MockGpu;

pub type Model = TrainerModel<ExecutionContext>;

pub mod optimiser {
    use super::ExecutionContext;

    use bullet_trainer::optimiser::{self, OptimiserState, radam};

    pub type AdamWOptimiser = optimiser::adam::AdamW<ExecutionContext>;
    pub type RAdamOptimiser = radam::RAdam<ExecutionContext>;
    pub type RangerOptimiser = optimiser::ranger::Ranger<ExecutionContext>;
    pub use optimiser::{Optimiser, adam::AdamWParams, ranger::RangerParams};

    pub trait OptimiserType: Default {
        type Optimiser: OptimiserState<ExecutionContext>;
    }

    #[derive(Default)]
    pub struct AdamW;
    impl OptimiserType for AdamW {
        type Optimiser = AdamWOptimiser;
    }

    #[derive(Default)]
    pub struct RAdam;
    impl OptimiserType for RAdam {
        type Optimiser = RAdamOptimiser;
    }

    #[derive(Default)]
    pub struct Ranger;
    impl OptimiserType for Ranger {
        type Optimiser = RangerOptimiser;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct RAdamParams {
        pub decay: f32,
        pub beta1: f32,
        pub beta2: f32,
        pub epsilon: f32,
        pub min_weight: f32,
        pub max_weight: f32,
    }

    impl Default for RAdamParams {
        fn default() -> Self {
            Self { decay: 0.0, beta1: 0.9, beta2: 0.999, epsilon: 0.00000001, min_weight: -1.98, max_weight: 1.98 }
        }
    }

    impl From<RAdamParams> for radam::RAdamParams {
        fn from(value: RAdamParams) -> Self {
            radam::RAdamParams {
                beta1: value.beta1,
                beta2: value.beta2,
                n_sma_threshold: 5.0,
                decay: value.decay,
                epsilon: value.epsilon,
                clip: Some((value.min_weight, value.max_weight)),
            }
        }
    }
}
