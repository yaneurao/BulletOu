//! Host launch sequence for minimal dense backward kernels.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    backward::{
        DenseCReluBackwardLaunchPlan, DenseCReluBackwardLayout, DenseOutputBackwardLaunchPlan,
        DenseOutputBackwardLayout, NnueL0CReluBackwardLaunchPlan, NnueL0CReluBackwardLayout,
    },
    CudaModule, CudaStream, DeviceBuffer, LaunchConfig, Result,
};
use cuda_host::cuda_launch;

#[allow(dead_code)]
pub(crate) fn launch_dense_output_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: DenseOutputBackwardLayout,
    inputs: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
    mut bias_gradient: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = DenseOutputBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let input_len = layout.input_len as u32;

    unsafe {
        // SAFETY: kernel ABI matches `dense_output_backward`; all buffers are
        // device allocations owned by the same CUDA context and live until the
        // caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::backward::dense_output_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(inputs),
                slice(output_gradients),
                slice(weights),
                slice_mut(input_gradients),
                slice_mut(weight_gradients),
                slice_mut(bias_gradient),
                batch,
                input_len
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_dense_crelu_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: DenseCReluBackwardLayout,
    inputs: &DeviceBuffer<f32>,
    activations: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
    mut bias_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = DenseCReluBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let input_dim = layout.input_dim as u32;
    let output_dim = layout.output_dim as u32;

    unsafe {
        // SAFETY: kernel ABI matches `dense_crelu_backward`; all buffers are
        // device allocations owned by the same CUDA context and live until the
        // caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::backward::dense_crelu_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(inputs),
                slice(activations),
                slice(output_gradients),
                slice(weights),
                slice_mut(input_gradients),
                slice_mut(weight_gradients),
                slice_mut(bias_gradients),
                batch,
                input_dim,
                output_dim
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_nnue_l0_crelu_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: NnueL0CReluBackwardLayout,
    combined_gradients: &DeviceBuffer<f32>,
    stm_activations: &DeviceBuffer<f32>,
    nstm_activations: &DeviceBuffer<f32>,
    mut stm_gradients: &mut DeviceBuffer<f32>,
    mut nstm_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = NnueL0CReluBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let l1 = layout.l1 as u32;

    unsafe {
        // SAFETY: kernel ABI matches `nnue_l0_crelu_backward`; all buffers are
        // device allocations owned by the same CUDA context and live until the
        // caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::backward::nnue_l0_crelu_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(combined_gradients),
                slice(stm_activations),
                slice(nstm_activations),
                slice_mut(stm_gradients),
                slice_mut(nstm_gradients),
                batch,
                l1
            ]
        }
    }?;

    Ok(())
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
