//! Host launch sequence for the minimal NNUE forward kernels.
//!
//! The kernel entry points live in this binary crate because cuda-oxide emits
//! `#[kernel]` symbols only for binary-local functions. Runtime-owned layout
//! and buffer types stay in `bulletou-cuda-oxide-runtime`.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    CudaModule, CudaStream, LaunchConfig, Result,
    nnue::{
        NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardLaunchPlan,
        NnueForwardWorkspace, NnueLayoutError,
    },
};
use cuda_host::cuda_launch;

#[allow(dead_code)]
pub(crate) fn launch_nnue_forward(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: &NnueForwardDeviceBatch,
    weights: &NnueForwardDeviceWeights,
    workspace: &mut NnueForwardWorkspace,
) -> Result<()> {
    validate_forward_layout(batch, weights, workspace)?;

    let layout = workspace.layout;
    let shape = layout.shape;
    let plan = NnueForwardLaunchPlan::new(layout);
    let batch_size = layout.batch_size as u32;
    let max_active = batch.max_active as u32;
    let input_size = shape.input_size as u32;
    let l1 = shape.l1 as u32;
    let l2 = shape.l2 as u32;
    let l3 = shape.l3 as u32;

    unsafe {
        // SAFETY: kernel ABI matches `nnue_sparse_l0_crelu`; all buffers are
        // device allocations owned by the same CUDA context and live until the
        // caller synchronizes or launches subsequent same-stream work.
        cuda_launch! {
            kernel: crate::kernels::nnue::nnue_sparse_l0_crelu,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.sparse_l0_threads_per_perspective),
            args: [
                slice(batch.stm_indices),
                slice(weights.l0w),
                slice(weights.l0b),
                slice_mut(workspace.stm_l0),
                batch_size, max_active, input_size, l1
            ]
        }
    }?;
    unsafe {
        // SAFETY: same ABI and lifetime guarantees as the stm L0 launch, using
        // the opponent-perspective sparse input and output buffer.
        cuda_launch! {
            kernel: crate::kernels::nnue::nnue_sparse_l0_crelu,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.sparse_l0_threads_per_perspective),
            args: [
                slice(batch.nstm_indices),
                slice(weights.l0w),
                slice(weights.l0b),
                slice_mut(workspace.nstm_l0),
                batch_size, max_active, input_size, l1
            ]
        }
    }?;
    unsafe {
        // SAFETY: concat reads the two L0 buffers written earlier in the same
        // stream and writes the combined buffer once.
        cuda_launch! {
            kernel: crate::kernels::nnue::nnue_concat_l0,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.concat_l0_threads),
            args: [
                slice(workspace.stm_l0),
                slice(workspace.nstm_l0),
                slice_mut(workspace.combined),
                batch_size, l1
            ]
        }
    }?;
    unsafe {
        // SAFETY: dense layer dimensions are derived from the validated
        // workspace layout and weight shape.
        cuda_launch! {
            kernel: crate::kernels::nnue::nnue_dense_l1_crelu,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.dense_l1_threads),
            args: [
                slice(workspace.combined),
                slice(weights.l1w),
                slice(weights.l1b),
                slice_mut(workspace.hidden1),
                batch_size, l1 * 2, l2
            ]
        }
    }?;
    unsafe {
        // SAFETY: dense layer dimensions are derived from the validated
        // workspace layout and weight shape.
        cuda_launch! {
            kernel: crate::kernels::nnue::nnue_dense_l2_crelu,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.dense_l2_threads),
            args: [
                slice(workspace.hidden1),
                slice(weights.l2w),
                slice(weights.l2b),
                slice_mut(workspace.hidden2),
                batch_size, l2, l3
            ]
        }
    }?;
    unsafe {
        // SAFETY: output layer reads hidden2 and writes exactly one scalar per
        // sample into the output buffer.
        cuda_launch! {
            kernel: crate::kernels::nnue::nnue_dense_output,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.dense_output_threads),
            args: [
                slice(workspace.hidden2),
                slice(weights.outw),
                slice(weights.outb),
                slice_mut(workspace.output),
                batch_size, l3
            ]
        }
    }?;

    Ok(())
}

fn validate_forward_layout(
    batch: &NnueForwardDeviceBatch,
    weights: &NnueForwardDeviceWeights,
    workspace: &NnueForwardWorkspace,
) -> Result<()> {
    expect_layout_value("batch_size", workspace.layout.batch_size, batch.batch_size)?;
    expect_layout_value(
        "shape.input_size",
        workspace.layout.shape.input_size,
        weights.shape.input_size,
    )?;
    expect_layout_value("shape.l1", workspace.layout.shape.l1, weights.shape.l1)?;
    expect_layout_value("shape.l2", workspace.layout.shape.l2, weights.shape.l2)?;
    expect_layout_value("shape.l3", workspace.layout.shape.l3, weights.shape.l3)?;
    Ok(())
}

fn expect_layout_value(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(NnueLayoutError::LayoutValue { name, expected, actual }.into())
    }
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
