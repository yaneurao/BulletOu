//! Host launch sequence for the minimal AdamW update kernel.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    optimizer::{AdamWUpdateLaunchPlan, AdamWUpdateLayout, AdamWUpdateParams},
    CudaModule, CudaStream, DeviceBuffer, LaunchConfig, Result,
};
use cuda_host::cuda_launch;

#[allow(dead_code)]
pub(crate) fn launch_adamw_update(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: AdamWUpdateLayout,
    params: AdamWUpdateParams,
    gradients: &DeviceBuffer<f32>,
    mut weights: &mut DeviceBuffer<f32>,
    mut momentum: &mut DeviceBuffer<f32>,
    mut velocity: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    params.validate()?;
    let plan = AdamWUpdateLaunchPlan::new(layout);
    let len = layout.len as u32;

    unsafe {
        // SAFETY: kernel ABI matches `adamw_update`; all buffers are device
        // allocations owned by the same CUDA context and live until the caller
        // synchronizes.
        cuda_launch! {
            kernel: crate::kernels::optimizer::adamw_update,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(gradients),
                slice_mut(weights),
                slice_mut(momentum),
                slice_mut(velocity),
                len,
                params.gradient_factor,
                params.learning_rate,
                params.decay,
                params.beta1,
                params.beta2,
                params.epsilon,
                params.min_weight,
                params.max_weight
            ]
        }
    }?;

    Ok(())
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
