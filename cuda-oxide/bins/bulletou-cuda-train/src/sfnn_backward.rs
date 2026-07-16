//! Host launch sequence for minimal SFNN backward kernels.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    backward::{
        SfnnL2InputBackwardLaunchPlan, SfnnL2InputBackwardLayout, SfnnStackedCReluBackwardLaunchPlan,
        SfnnStackedCReluBackwardLayout, SfnnStackedL3BackwardLaunchPlan, SfnnStackedL3BackwardLayout,
    },
    CudaModule, CudaStream, DeviceBuffer, LaunchConfig, Result,
};
use cuda_host::cuda_launch;

#[allow(dead_code)]
pub(crate) fn launch_sfnn_stacked_l3_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnStackedL3BackwardLayout,
    inputs: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    buckets: &DeviceBuffer<i32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
    mut l1_gradients: &mut DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
    mut bias_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = SfnnStackedL3BackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let input_dim = layout.l2_size as u32;
    let l1_out = layout.l1_out as u32;
    let num_stacks = layout.num_stacks as u32;

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_stacked_l3_backward`; all buffers
        // are device allocations owned by the same CUDA context and live until
        // the caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_stacked_l3_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(inputs),
                slice(output_gradients),
                slice(weights),
                slice(buckets),
                slice_mut(input_gradients),
                slice_mut(l1_gradients),
                slice_mut(weight_gradients),
                slice_mut(bias_gradients),
                batch,
                input_dim,
                l1_out,
                num_stacks
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_stacked_crelu_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnStackedCReluBackwardLayout,
    inputs: &DeviceBuffer<f32>,
    activations: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    buckets: &DeviceBuffer<i32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
    mut bias_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = SfnnStackedCReluBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let input_dim = layout.input_dim as u32;
    let output_dim = layout.output_dim as u32;
    let num_stacks = layout.num_stacks as u32;

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_stacked_crelu_backward`; all
        // buffers are device allocations owned by the same CUDA context and
        // live until the caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_stacked_crelu_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(inputs),
                slice(activations),
                slice(output_gradients),
                slice(weights),
                slice(buckets),
                slice_mut(input_gradients),
                slice_mut(weight_gradients),
                slice_mut(bias_gradients),
                batch,
                input_dim,
                output_dim,
                num_stacks
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_l2_input_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnL2InputBackwardLayout,
    l1: &DeviceBuffer<f32>,
    l2_input: &DeviceBuffer<f32>,
    l2_input_gradients: &DeviceBuffer<f32>,
    mut l1_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = SfnnL2InputBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let l1_hidden = layout.l1_hidden as u32;

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_l2_input_backward`; the kernel has
        // one writer for each L1 gradient element and only adds to hidden
        // columns, preserving the L3 skip-column gradient.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_l2_input_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(l1),
                slice(l2_input),
                slice(l2_input_gradients),
                slice_mut(l1_gradients),
                batch,
                l1_hidden
            ]
        }
    }?;

    Ok(())
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
