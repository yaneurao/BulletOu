//! Host launch sequence for minimal optimizer update kernels.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    optimizer::{
        AdamWUpdateLaunchPlan, AdamWUpdateLayout, AdamWUpdateParams, RAdamUpdateLaunchPlan, RAdamUpdateLayout,
        RAdamUpdateParams, RangerLookaheadLaunchPlan, RangerLookaheadLayout, RangerLookaheadParams,
        RangerUpdateLayout, RangerUpdateParams,
    },
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

#[allow(dead_code)]
pub(crate) fn launch_radam_update(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: RAdamUpdateLayout,
    params: RAdamUpdateParams,
    gradients: &DeviceBuffer<f32>,
    mut weights: &mut DeviceBuffer<f32>,
    mut momentum: &mut DeviceBuffer<f32>,
    mut velocity: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    params.validate()?;
    let step_scale = params.step_scale()?;
    let plan = RAdamUpdateLaunchPlan::new(layout);
    let len = layout.len as u32;
    let use_denom = u32::from(step_scale.use_denom);

    unsafe {
        // SAFETY: kernel ABI matches `radam_update`; all buffers are device
        // allocations owned by the same CUDA context and live until the caller
        // synchronizes.
        cuda_launch! {
            kernel: crate::kernels::optimizer::radam_update,
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
                step_scale.step_size,
                use_denom,
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

#[allow(dead_code)]
pub(crate) fn launch_ranger_lookahead(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: RangerLookaheadLayout,
    params: RangerLookaheadParams,
    mut weights: &mut DeviceBuffer<f32>,
    mut slow_params: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    params.validate()?;
    let plan = RangerLookaheadLaunchPlan::new(layout);
    let len = layout.len as u32;

    unsafe {
        // SAFETY: kernel ABI matches `ranger_lookahead`; all buffers are
        // device allocations owned by the same CUDA context and live until the
        // caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::optimizer::ranger_lookahead,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice_mut(weights),
                slice_mut(slow_params),
                len,
                params.alpha
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_ranger_update(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: RangerUpdateLayout,
    params: RangerUpdateParams,
    gradients: &DeviceBuffer<f32>,
    weights: &mut DeviceBuffer<f32>,
    momentum: &mut DeviceBuffer<f32>,
    velocity: &mut DeviceBuffer<f32>,
    slow_params: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    params.validate()?;
    launch_radam_update(
        stream,
        module,
        RAdamUpdateLayout::new(layout.len),
        params.radam,
        gradients,
        weights,
        momentum,
        velocity,
    )?;

    if params.should_lookahead()? {
        launch_ranger_lookahead(
            stream,
            module,
            RangerLookaheadLayout::new(layout.len),
            params.lookahead,
            weights,
            slow_params,
        )?;
    }

    Ok(())
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
