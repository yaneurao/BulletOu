//! Host launch sequence for minimal dense backward kernels.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    CudaModule, CudaStream, DeviceBuffer, LaunchConfig, Result,
    backward::{
        DenseCReluBackwardLaunchPlan, DenseCReluBackwardLayout, DenseOutputBackwardLaunchPlan,
        DenseOutputBackwardLayout, NnueL0CReluBackwardLaunchPlan, NnueL0CReluBackwardLayout,
        NnueL0SparseBackwardLaunchPlan, NnueL0SparseBackwardLayout,
    },
};
use cuda_host::cuda_launch;

use crate::cublas::CublasHandle;

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
pub(crate) fn launch_dense_crelu_backward_cublas(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    cublas: &CublasHandle,
    layout: DenseCReluBackwardLayout,
    inputs: &DeviceBuffer<f32>,
    activations: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
    mut bias_gradients: &mut DeviceBuffer<f32>,
    mut pre_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    launch_dense_crelu_pre_gradients(stream, module, layout, activations, output_gradients, &mut pre_gradients)?;

    cublas.sgemm_x_yt_rowmajor(
        layout.batch_size,
        layout.input_dim,
        layout.output_dim,
        pre_gradients,
        weights,
        &mut input_gradients,
    )?;
    cublas.sgemm_xt_y_rowmajor(
        layout.input_dim,
        layout.output_dim,
        layout.batch_size,
        inputs,
        pre_gradients,
        &mut weight_gradients,
    )?;

    launch_dense_bias_gradients(stream, module, layout, pre_gradients, &mut bias_gradients)
}

#[allow(dead_code)]
pub(crate) fn launch_dense_crelu_backward_split(
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
    mut pre_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    launch_dense_crelu_pre_gradients(stream, module, layout, activations, output_gradients, &mut pre_gradients)?;
    launch_dense_pre_input_gradients(stream, module, layout, pre_gradients, weights, &mut input_gradients)?;
    launch_dense_pre_weight_gradients(stream, module, layout, inputs, pre_gradients, &mut weight_gradients)?;
    launch_dense_bias_gradients(stream, module, layout, pre_gradients, &mut bias_gradients)
}

fn launch_dense_crelu_pre_gradients(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: DenseCReluBackwardLayout,
    activations: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    mut pre_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let threads = layout.output_gradients_len();
    let batch = layout.batch_size as u32;
    let output_dim = layout.output_dim as u32;

    unsafe {
        // SAFETY: kernel ABI matches `dense_crelu_pre_gradients`; all buffers
        // are same-context device allocations and the output has B*out_dim elements.
        cuda_launch! {
            kernel: crate::kernels::backward::dense_crelu_pre_gradients,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(threads),
            args: [
                slice(activations),
                slice(output_gradients),
                slice_mut(pre_gradients),
                batch,
                output_dim
            ]
        }
    }?;

    Ok(())
}

fn launch_dense_pre_input_gradients(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: DenseCReluBackwardLayout,
    pre_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let batch = layout.batch_size as u32;
    let input_dim = layout.input_dim as u32;
    let output_dim = layout.output_dim as u32;

    unsafe {
        // SAFETY: kernel ABI matches `dense_pre_input_gradients`.
        cuda_launch! {
            kernel: crate::kernels::backward::dense_pre_input_gradients,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(layout.input_gradients_len()),
            args: [
                slice(pre_gradients),
                slice(weights),
                slice_mut(input_gradients),
                batch,
                input_dim,
                output_dim
            ]
        }
    }?;

    Ok(())
}

fn launch_dense_pre_weight_gradients(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: DenseCReluBackwardLayout,
    inputs: &DeviceBuffer<f32>,
    pre_gradients: &DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let batch = layout.batch_size as u32;
    let input_dim = layout.input_dim as u32;
    let output_dim = layout.output_dim as u32;

    unsafe {
        // SAFETY: kernel ABI matches `dense_pre_weight_gradients`.
        cuda_launch! {
            kernel: crate::kernels::backward::dense_pre_weight_gradients,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(layout.weight_len()),
            args: [
                slice(inputs),
                slice(pre_gradients),
                slice_mut(weight_gradients),
                batch,
                input_dim,
                output_dim
            ]
        }
    }?;

    Ok(())
}

fn launch_dense_bias_gradients(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: DenseCReluBackwardLayout,
    pre_gradients: &DeviceBuffer<f32>,
    mut bias_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let batch = layout.batch_size as u32;
    let output_dim = layout.output_dim as u32;

    unsafe {
        // SAFETY: kernel ABI matches `dense_bias_gradients`; one thread owns
        // each bias output element and scans the batch dimension.
        cuda_launch! {
            kernel: crate::kernels::backward::dense_bias_gradients,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(layout.bias_len()),
            args: [
                slice(pre_gradients),
                slice_mut(bias_gradients),
                batch,
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

#[allow(dead_code)]
pub(crate) fn launch_nnue_l0_sparse_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: NnueL0SparseBackwardLayout,
    stm_indices: &DeviceBuffer<i32>,
    nstm_indices: &DeviceBuffer<i32>,
    stm_gradients: &DeviceBuffer<f32>,
    nstm_gradients: &DeviceBuffer<f32>,
    mut l0w_gradients: &mut DeviceBuffer<f32>,
    mut l0b_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = NnueL0SparseBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let max_active = layout.max_active as u32;
    let input_size = layout.input_size as u32;
    let l1 = layout.l1 as u32;

    unsafe {
        // SAFETY: kernel ABI matches `nnue_l0_sparse_zero_gradients`; each
        // launched thread owns one output gradient element in the zero phase.
        cuda_launch! {
            kernel: crate::kernels::backward::nnue_l0_sparse_zero_gradients,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice_mut(l0w_gradients),
                slice_mut(l0b_gradients),
                input_size,
                l1
            ]
        }
    }?;

    unsafe {
        // SAFETY: kernel ABI matches `nnue_l0_sparse_backward`; gradient
        // writes are shared scatter-adds and are performed through device
        // atomics inside the kernel.
        cuda_launch! {
            kernel: crate::kernels::backward::nnue_l0_sparse_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.scatter_threads),
            args: [
                slice(stm_indices),
                slice(nstm_indices),
                slice(stm_gradients),
                slice(nstm_gradients),
                slice_mut(l0w_gradients),
                slice_mut(l0b_gradients),
                batch,
                max_active,
                input_size,
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
