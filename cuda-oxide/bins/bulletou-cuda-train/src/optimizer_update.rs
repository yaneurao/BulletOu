//! Host launch sequence for minimal optimizer update kernels.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    CudaModule, CudaStream, DeviceBuffer, Error, LaunchConfig, Result,
    backward::{NnueBackwardWorkspace, SfnnBackwardWorkspace},
    nnue::NnueForwardDeviceWeights,
    optimizer::{
        AdamWUpdateLaunchPlan, AdamWUpdateLayout, AdamWUpdateParams, NnueRangerOptimizerStates, RAdamUpdateLaunchPlan,
        RAdamUpdateLayout, RAdamUpdateParams, RangerLookaheadLaunchPlan, RangerLookaheadLayout, RangerLookaheadParams,
        RangerOptimizerState, RangerUpdateLayout, RangerUpdateParams, SfnnRangerOptimizerStates,
    },
    sfnn::SfnnForwardDeviceWeights,
};
use cuda_host::cuda_launch;

const NNUE_QUANT_CLAMP_MIN: f32 = -127.0 / 64.0;
const NNUE_QUANT_CLAMP_MAX: f32 = 127.0 / 64.0;
const NNUE_NO_CLAMP_MIN: f32 = f32::MIN;
const NNUE_NO_CLAMP_MAX: f32 = f32::MAX;

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
    state: &mut RangerOptimizerState,
) -> Result<()> {
    layout.validate()?;
    params.validate()?;
    ensure_ranger_state_len("ranger_update", layout, state)?;
    launch_radam_update(
        stream,
        module,
        RAdamUpdateLayout::new(layout.len),
        params.radam,
        gradients,
        weights,
        &mut state.momentum,
        &mut state.velocity,
    )?;

    if params.should_lookahead()? {
        launch_ranger_lookahead(
            stream,
            module,
            RangerLookaheadLayout::new(layout.len),
            params.lookahead,
            weights,
            &mut state.slow_params,
        )?;
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_nnue_ranger_update(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    params: RangerUpdateParams,
    weights: &mut NnueForwardDeviceWeights,
    gradients: &NnueBackwardWorkspace,
    states: &mut NnueRangerOptimizerStates,
) -> Result<()> {
    ensure_nnue_update_shapes(weights, gradients, states)?;
    let layout = states.layout;
    let no_clamp_params = nnue_ranger_params_with_clamp(params, NNUE_NO_CLAMP_MIN, NNUE_NO_CLAMP_MAX);
    let quant_clamp_params = nnue_ranger_params_with_clamp(params, NNUE_QUANT_CLAMP_MIN, NNUE_QUANT_CLAMP_MAX);

    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.l0w_state_layout().state_len()),
        no_clamp_params,
        &gradients.l0w_gradients,
        &mut weights.l0w,
        &mut states.l0w,
    )?;
    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.l0b_state_layout().state_len()),
        no_clamp_params,
        &gradients.l0b_gradients,
        &mut weights.l0b,
        &mut states.l0b,
    )?;
    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.l1w_state_layout().state_len()),
        quant_clamp_params,
        &gradients.l1w_gradients,
        &mut weights.l1w,
        &mut states.l1w,
    )?;
    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.l1b_state_layout().state_len()),
        quant_clamp_params,
        &gradients.l1b_gradients,
        &mut weights.l1b,
        &mut states.l1b,
    )?;
    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.l2w_state_layout().state_len()),
        quant_clamp_params,
        &gradients.l2w_gradients,
        &mut weights.l2w,
        &mut states.l2w,
    )?;
    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.l2b_state_layout().state_len()),
        quant_clamp_params,
        &gradients.l2b_gradients,
        &mut weights.l2b,
        &mut states.l2b,
    )?;
    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.outw_state_layout().state_len()),
        quant_clamp_params,
        &gradients.outw_gradients,
        &mut weights.outw,
        &mut states.outw,
    )?;
    launch_ranger_update(
        stream,
        module,
        RangerUpdateLayout::new(layout.outb_state_layout().state_len()),
        no_clamp_params,
        &gradients.outb_gradients,
        &mut weights.outb,
        &mut states.outb,
    )?;

    Ok(())
}

fn nnue_ranger_params_with_clamp(
    mut params: RangerUpdateParams,
    min_weight: f32,
    max_weight: f32,
) -> RangerUpdateParams {
    params.radam.min_weight = min_weight;
    params.radam.max_weight = max_weight;
    params
}

#[allow(dead_code)]
pub(crate) fn launch_radam_update_reset_gradients(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: RAdamUpdateLayout,
    params: RAdamUpdateParams,
    mut gradients: &mut DeviceBuffer<f32>,
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
        // SAFETY: kernel ABI matches `radam_update_reset_gradients`; all
        // buffers are device allocations owned by the same CUDA context and
        // live until the caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::optimizer::radam_update_reset_gradients,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice_mut(gradients),
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
pub(crate) fn launch_ranger_update_reset_gradients(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: RangerUpdateLayout,
    params: RangerUpdateParams,
    gradients: &mut DeviceBuffer<f32>,
    weights: &mut DeviceBuffer<f32>,
    state: &mut RangerOptimizerState,
) -> Result<()> {
    layout.validate()?;
    params.validate()?;
    ensure_ranger_state_len("ranger_update_reset_gradients", layout, state)?;
    launch_radam_update_reset_gradients(
        stream,
        module,
        RAdamUpdateLayout::new(layout.len),
        params.radam,
        gradients,
        weights,
        &mut state.momentum,
        &mut state.velocity,
    )?;

    if params.should_lookahead()? {
        launch_ranger_lookahead(
            stream,
            module,
            RangerLookaheadLayout::new(layout.len),
            params.lookahead,
            weights,
            &mut state.slow_params,
        )?;
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_ranger_update(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    params: RangerUpdateParams,
    weights: &mut SfnnForwardDeviceWeights,
    gradients: &mut SfnnBackwardWorkspace,
    states: &mut SfnnRangerOptimizerStates,
) -> Result<()> {
    ensure_sfnn_update_shapes(weights, gradients, states)?;
    let layout = states.layout;
    let no_clamp_params = nnue_ranger_params_with_clamp(params, NNUE_NO_CLAMP_MIN, NNUE_NO_CLAMP_MAX);
    let quant_clamp_params = nnue_ranger_params_with_clamp(params, NNUE_QUANT_CLAMP_MIN, NNUE_QUANT_CLAMP_MAX);

    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l0w_state_layout().state_len()),
        no_clamp_params,
        &mut gradients.l0w_gradients,
        &mut weights.l0w,
        &mut states.l0w,
    )?;
    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l0b_state_layout().state_len()),
        no_clamp_params,
        &mut gradients.l0b_gradients,
        &mut weights.l0b,
        &mut states.l0b,
    )?;
    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l1w_state_layout().state_len()),
        quant_clamp_params,
        &mut gradients.l1w_gradients,
        &mut weights.l1w,
        &mut states.l1w,
    )?;
    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l1b_state_layout().state_len()),
        quant_clamp_params,
        &mut gradients.l1b_gradients,
        &mut weights.l1b,
        &mut states.l1b,
    )?;
    match (&mut weights.l1fw, &mut states.l1fw) {
        (Some(l1fw), Some(l1fw_state)) => {
            launch_ranger_update_reset_gradients(
                stream,
                module,
                RangerUpdateLayout::new(layout.l1fw_state_layout().state_len()),
                quant_clamp_params,
                &mut gradients.l1fw_gradients,
                l1fw,
                l1fw_state,
            )?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::Smoke("SFNN shared L1 weight/state mismatch for l1fw update".to_string()));
        }
    }
    match (&mut weights.l1fb, &mut states.l1fb) {
        (Some(l1fb), Some(l1fb_state)) => {
            launch_ranger_update_reset_gradients(
                stream,
                module,
                RangerUpdateLayout::new(layout.l1fb_state_layout().state_len()),
                quant_clamp_params,
                &mut gradients.l1fb_gradients,
                l1fb,
                l1fb_state,
            )?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::Smoke("SFNN shared L1 weight/state mismatch for l1fb update".to_string()));
        }
    }
    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l2w_state_layout().state_len()),
        quant_clamp_params,
        &mut gradients.l2w_gradients,
        &mut weights.l2w,
        &mut states.l2w,
    )?;
    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l2b_state_layout().state_len()),
        quant_clamp_params,
        &mut gradients.l2b_gradients,
        &mut weights.l2b,
        &mut states.l2b,
    )?;
    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l3w_state_layout().state_len()),
        quant_clamp_params,
        &mut gradients.l3w_gradients,
        &mut weights.l3w,
        &mut states.l3w,
    )?;
    launch_ranger_update_reset_gradients(
        stream,
        module,
        RangerUpdateLayout::new(layout.l3b_state_layout().state_len()),
        no_clamp_params,
        &mut gradients.l3b_gradients,
        &mut weights.l3b,
        &mut states.l3b,
    )?;

    Ok(())
}

fn ensure_ranger_state_len(name: &'static str, layout: RangerUpdateLayout, state: &RangerOptimizerState) -> Result<()> {
    let expected = layout.len;
    let actual = state.layout.state_len();
    if expected == actual {
        Ok(())
    } else {
        Err(Error::Smoke(format!("{name} optimizer state length mismatch: expected {expected}, got {actual}")))
    }
}

fn ensure_nnue_update_shapes(
    weights: &NnueForwardDeviceWeights,
    gradients: &NnueBackwardWorkspace,
    states: &NnueRangerOptimizerStates,
) -> Result<()> {
    let shape = weights.shape;
    if gradients.layout.shape != shape {
        return Err(Error::Smoke(format!(
            "NNUE gradient shape mismatch: weights={shape:?}, gradients={:?}",
            gradients.layout.shape
        )));
    }
    if states.layout.weights.shape != shape {
        return Err(Error::Smoke(format!(
            "NNUE optimizer state shape mismatch: weights={shape:?}, states={:?}",
            states.layout.weights.shape
        )));
    }
    Ok(())
}

fn ensure_sfnn_update_shapes(
    weights: &SfnnForwardDeviceWeights,
    gradients: &SfnnBackwardWorkspace,
    states: &SfnnRangerOptimizerStates,
) -> Result<()> {
    let shape = weights.shape;
    if gradients.layout.shape != shape {
        return Err(Error::Smoke(format!(
            "SFNN gradient shape mismatch: weights={shape:?}, gradients={:?}",
            gradients.layout.shape
        )));
    }
    if states.layout.weights.shape != shape {
        return Err(Error::Smoke(format!(
            "SFNN optimizer state shape mismatch: weights={shape:?}, states={:?}",
            states.layout.weights.shape
        )));
    }
    Ok(())
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
