//! Host launch sequence for the minimal SFNN forward kernels.
//!
//! The kernel entry points live in this binary crate because cuda-oxide emits
//! `#[kernel]` symbols only for binary-local functions. Runtime-owned layout
//! and buffer types stay in `bulletou-cuda-oxide-runtime`.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    CudaModule, CudaStream, DeviceBuffer, LaunchConfig, Result,
    sfnn::{
        SfnnForwardDeviceBatch, SfnnForwardDeviceWeights, SfnnForwardLaunchPlan, SfnnForwardWorkspace, SfnnLayoutError,
    },
};
use cuda_host::cuda_launch;

#[allow(dead_code)]
pub(crate) fn launch_sfnn_forward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    workspace: &mut SfnnForwardWorkspace,
) -> Result<()> {
    launch_sfnn_forward_with_l0(stream, module, batch, weights, workspace, &weights.l0w, weights.shape.input_size)
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_forward_with_l0(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    workspace: &mut SfnnForwardWorkspace,
    l0w: &DeviceBuffer<f32>,
    l0_input_size: usize,
) -> Result<()> {
    validate_forward_layout(batch, weights, workspace)?;

    let layout = workspace.layout;
    let shape = layout.shape;
    let plan = SfnnForwardLaunchPlan::new(layout);
    let batch_size = layout.batch_size as u32;
    let max_active = batch.max_active as u32;
    let input_size = l0_input_size as u32;
    let ft_size = shape.ft_size as u32;
    let l1_out = shape.l1_out() as u32;
    let l1_hidden = shape.l1_hidden as u32;
    let l2_in = shape.l2_in() as u32;
    let l2_size = shape.l2_size as u32;
    let num_stacks = shape.num_stacks as u32;

    unsafe {
        // SAFETY: kernel ABI matches `sfnn_sparse_l0_crelu`; all buffers are
        // device allocations owned by the same CUDA context and live until the
        // caller synchronizes or launches subsequent same-stream work.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_sparse_l0_crelu,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.sparse_l0_threads_per_perspective),
            args: [
                slice(batch.stm_indices),
                slice(l0w),
                slice(weights.l0b),
                slice_mut(workspace.stm_l0),
                batch_size, max_active, input_size, ft_size
            ]
        }
    }?;
    unsafe {
        // SAFETY: same ABI and lifetime guarantees as the stm L0 launch, using
        // the opponent-perspective sparse input and output buffer.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_sparse_l0_crelu,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.sparse_l0_threads_per_perspective),
            args: [
                slice(batch.nstm_indices),
                slice(l0w),
                slice(weights.l0b),
                slice_mut(workspace.nstm_l0),
                batch_size, max_active, input_size, ft_size
            ]
        }
    }?;
    unsafe {
        // SAFETY: pairwise concat reads the two L0 buffers written earlier in
        // the same stream and writes the combined buffer once.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_pairwise_concat,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.pairwise_concat_threads),
            args: [
                slice(workspace.stm_l0),
                slice(workspace.nstm_l0),
                slice_mut(workspace.combined),
                batch_size, ft_size
            ]
        }
    }?;
    unsafe {
        // SAFETY: stacked dense dimensions are derived from validated layout
        // and weight shape. Buckets are copied from validated host batch.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_stacked_l1,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.stacked_l1_threads),
            args: [
                slice(workspace.combined),
                slice(weights.l1w),
                slice(weights.l1b),
                slice(batch.buckets),
                slice_mut(workspace.l1),
                batch_size, ft_size, l1_out, num_stacks
            ]
        }
    }?;
    if let (Some(l1fw), Some(l1fb)) = (&weights.l1fw, &weights.l1fb) {
        unsafe {
            // SAFETY: shared L1 dimensions are derived from the same validated
            // shape as stacked L1. The kernel adds into the L1 buffer produced
            // by the previous launch in the same stream.
            cuda_launch! {
                kernel: crate::kernels::sfnn::sfnn_shared_l1_add,
                stream: stream.clone(),
                module: module.clone(),
                config: cfg_1d(plan.shared_l1_threads),
                args: [
                    slice(workspace.combined),
                    slice(l1fw),
                    slice(l1fb),
                    slice_mut(workspace.l1),
                    batch_size, ft_size, l1_out
                ]
            }
        }?;
    }
    unsafe {
        // SAFETY: l2_input reads L1 output and writes exactly
        // batch * l1_hidden * 2 values.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_l2_input,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.l2_input_threads),
            args: [
                slice(workspace.l1),
                slice_mut(workspace.l2_input),
                batch_size, l1_hidden
            ]
        }
    }?;
    unsafe {
        // SAFETY: stacked dense dimensions are derived from validated layout
        // and weight shape.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_stacked_l2_crelu,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.stacked_l2_threads),
            args: [
                slice(workspace.l2_input),
                slice(weights.l2w),
                slice(weights.l2b),
                slice(batch.buckets),
                slice_mut(workspace.l2),
                batch_size, l2_in, l2_size, num_stacks
            ]
        }
    }?;
    unsafe {
        // SAFETY: output layer reads L2 and L1 skip, then writes one scalar per
        // sample into the output buffer.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_stacked_l3_output,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.stacked_l3_threads),
            args: [
                slice(workspace.l2),
                slice(workspace.l1),
                slice(weights.l3w),
                slice(weights.l3b),
                slice(batch.buckets),
                slice_mut(workspace.output),
                batch_size, l2_size, l1_hidden, num_stacks
            ]
        }
    }?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn launch_sfnn_halfka2_fold_factorized_l0w(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    train_l0w: &DeviceBuffer<f32>,
    mut forward_l0w: &mut DeviceBuffer<f32>,
    ft_size: usize,
) -> Result<()> {
    let threads = 131_949_usize.saturating_mul(ft_size);
    let ft_size = ft_size as u32;
    unsafe {
        // SAFETY: kernel ABI matches `sfnn_halfka2_fold_factorized_l0w`.
        // `forward_l0w` is the base HalfKA2 shape and each thread writes one
        // disjoint folded weight.
        cuda_launch! {
            kernel: crate::kernels::sfnn::sfnn_halfka2_fold_factorized_l0w,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(threads),
            args: [
                slice(train_l0w),
                slice_mut(forward_l0w),
                ft_size
            ]
        }
    }?;
    Ok(())
}

fn validate_forward_layout(
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    workspace: &SfnnForwardWorkspace,
) -> Result<()> {
    expect_layout_value("batch_size", workspace.layout.batch_size, batch.batch_size)?;
    expect_layout_value("shape.input_size", workspace.layout.shape.input_size, weights.shape.input_size)?;
    expect_layout_value("shape.ft_size", workspace.layout.shape.ft_size, weights.shape.ft_size)?;
    expect_layout_value("shape.l1_hidden", workspace.layout.shape.l1_hidden, weights.shape.l1_hidden)?;
    expect_layout_value("shape.l2_size", workspace.layout.shape.l2_size, weights.shape.l2_size)?;
    expect_layout_value("shape.num_stacks", workspace.layout.shape.num_stacks, weights.shape.num_stacks)?;
    Ok(())
}

fn expect_layout_value(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(SfnnLayoutError::LayoutValue { name, expected, actual }.into())
    }
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
