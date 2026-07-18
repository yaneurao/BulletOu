//! Host launch sequence for minimal SFNN backward kernels.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    CudaModule, CudaStream, DeviceBuffer, LaunchConfig, Result,
    backward::{
        SfnnL0SparseBackwardLayout, SfnnL2InputBackwardLaunchPlan, SfnnL2InputBackwardLayout,
        SfnnPairwiseBackwardLaunchPlan, SfnnPairwiseBackwardLayout, SfnnSharedL1BackwardLaunchPlan,
        SfnnSharedL1BackwardLayout, SfnnStackedAffineBackwardLaunchPlan, SfnnStackedAffineBackwardLayout,
        SfnnStackedCReluBackwardLaunchPlan, SfnnStackedCReluBackwardLayout, SfnnStackedL3BackwardLaunchPlan,
        SfnnStackedL3BackwardLayout,
    },
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
    let scatter_threads = layout.batch_size.saturating_mul(layout.input_dim).saturating_mul(layout.output_dim);
    let threads = plan.threads.max(scatter_threads);

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_stacked_crelu_backward`; all
        // buffers are device allocations owned by the same CUDA context and
        // live until the caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_stacked_crelu_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(threads),
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

#[allow(dead_code)]
pub(crate) fn launch_sfnn_stacked_affine_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnStackedAffineBackwardLayout,
    inputs: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    buckets: &DeviceBuffer<i32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
    mut bias_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = SfnnStackedAffineBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let input_dim = layout.input_dim as u32;
    let output_dim = layout.output_dim as u32;
    let num_stacks = layout.num_stacks as u32;
    let scatter_threads = layout.batch_size.saturating_mul(layout.input_dim).saturating_mul(layout.output_dim);
    let threads = plan.threads.max(scatter_threads);

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_stacked_affine_backward`; all
        // buffers are device allocations owned by the same CUDA context and
        // live until the caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_stacked_affine_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(threads),
            args: [
                slice(inputs),
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
pub(crate) fn launch_sfnn_shared_l1_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnSharedL1BackwardLayout,
    inputs: &DeviceBuffer<f32>,
    output_gradients: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    mut input_gradients: &mut DeviceBuffer<f32>,
    mut weight_gradients: &mut DeviceBuffer<f32>,
    mut bias_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = SfnnSharedL1BackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let input_dim = layout.input_dim as u32;
    let output_dim = layout.output_dim as u32;
    let scatter_threads = layout.batch_size.saturating_mul(layout.input_dim).saturating_mul(layout.output_dim);
    let threads = plan.threads.max(scatter_threads);

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_shared_l1_backward`; the kernel
        // adds into an already initialized input-gradient buffer and writes
        // one shared L1 weight/bias gradient per launched index.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_shared_l1_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(threads),
            args: [
                slice(inputs),
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
pub(crate) fn launch_sfnn_pairwise_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnPairwiseBackwardLayout,
    stm_l0: &DeviceBuffer<f32>,
    nstm_l0: &DeviceBuffer<f32>,
    combined_gradients: &DeviceBuffer<f32>,
    mut stm_gradients: &mut DeviceBuffer<f32>,
    mut nstm_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let plan = SfnnPairwiseBackwardLaunchPlan::new(layout);
    let batch = layout.batch_size as u32;
    let ft_size = layout.ft_size as u32;

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_pairwise_backward`; each thread
        // writes one stm/nstm L0-gradient element at the same disjoint index.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_pairwise_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.threads),
            args: [
                slice(stm_l0),
                slice(nstm_l0),
                slice(combined_gradients),
                slice_mut(stm_gradients),
                slice_mut(nstm_gradients),
                batch,
                ft_size
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_l0_sparse_backward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnL0SparseBackwardLayout,
    stm_indices: &DeviceBuffer<i32>,
    nstm_indices: &DeviceBuffer<i32>,
    stm_activations: &DeviceBuffer<f32>,
    nstm_activations: &DeviceBuffer<f32>,
    stm_output_gradients: &DeviceBuffer<f32>,
    nstm_output_gradients: &DeviceBuffer<f32>,
    mut stm_pre_gradients: &mut DeviceBuffer<f32>,
    mut nstm_pre_gradients: &mut DeviceBuffer<f32>,
    mut l0w_gradients: &mut DeviceBuffer<f32>,
    mut l0b_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let batch = layout.batch_size as u32;
    let max_active = layout.max_active as u32;
    let input_size = layout.input_size as u32;
    let ft_size = layout.ft_size as u32;
    let scatter_threads = layout.l0_len();

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_l0_sparse_backward`; pre-gradient
        // writes are disjoint, while weight/bias gradient accumulation is
        // performed through device atomics inside the kernel.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_l0_sparse_backward,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(scatter_threads),
            args: [
                slice(stm_indices),
                slice(nstm_indices),
                slice(stm_activations),
                slice(nstm_activations),
                slice(stm_output_gradients),
                slice(nstm_output_gradients),
                slice_mut(stm_pre_gradients),
                slice_mut(nstm_pre_gradients),
                slice_mut(l0w_gradients),
                slice_mut(l0b_gradients),
                batch,
                max_active,
                input_size,
                ft_size
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_pairwise_l0_sparse_backward_train(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnL0SparseBackwardLayout,
    stm_indices: &DeviceBuffer<i32>,
    nstm_indices: &DeviceBuffer<i32>,
    stm_activations: &DeviceBuffer<f32>,
    nstm_activations: &DeviceBuffer<f32>,
    combined_gradients: &DeviceBuffer<f32>,
    mut l0w_gradients: &mut DeviceBuffer<f32>,
    mut l0b_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let batch = layout.batch_size as u32;
    let max_active = layout.max_active as u32;
    let input_size = layout.input_size as u32;
    let ft_size = layout.ft_size as u32;
    let threads = layout.l0_len();

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_pairwise_l0_sparse_backward_train`.
        // Each thread owns one sample/FT row. Weight and bias gradients are
        // accumulated through device atomics.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_pairwise_l0_sparse_backward_train,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(threads),
            args: [
                slice(stm_indices),
                slice(nstm_indices),
                slice(stm_activations),
                slice(nstm_activations),
                slice(combined_gradients),
                slice_mut(l0w_gradients),
                slice_mut(l0b_gradients),
                batch,
                max_active,
                input_size,
                ft_size
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_halfka2_ft_factorized_l0_reduce_virtual_grad(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    layout: SfnnL0SparseBackwardLayout,
    mut l0w_gradients: &mut DeviceBuffer<f32>,
) -> Result<()> {
    layout.validate()?;
    let input_size = layout.input_size as u32;
    let ft_size = layout.ft_size as u32;
    let virtual_rows = 1_629_usize;
    let threads = virtual_rows * layout.ft_size;

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_halfka2_ft_factorized_l0_reduce_virtual_grad`.
        // The kernel only reads base HalfKA2 L0 gradient rows and overwrites
        // the disjoint virtual FT-factorizer rows.
        cuda_launch! {
            kernel: crate::kernels::backward::sfnn_halfka2_ft_factorized_l0_reduce_virtual_grad,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(threads),
            args: [
                slice_mut(l0w_gradients),
                input_size,
                ft_size
            ]
        }
    }?;

    Ok(())
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
