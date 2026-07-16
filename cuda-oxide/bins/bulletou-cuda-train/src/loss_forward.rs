//! Host launch sequence for the scalar value-loss kernel.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    loss::{LossLayoutError, ScalarLossDeviceBatch, ScalarLossLaunchPlan, ScalarLossWorkspace},
    CudaModule, CudaStream, LaunchConfig, Result,
};
use cuda_host::cuda_launch;

#[allow(dead_code)]
pub(crate) fn launch_sigmoid_mse_loss(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: &ScalarLossDeviceBatch,
    workspace: &mut ScalarLossWorkspace,
) -> Result<()> {
    validate_loss_layout(batch, workspace)?;

    let layout = workspace.layout;
    let plan = ScalarLossLaunchPlan::new(layout);
    let batch_size = layout.batch_size as u32;

    unsafe {
        // SAFETY: kernel ABI matches `loss_sigmoid_mse_reduce`; all buffers are
        // device allocations owned by the same CUDA context and live until the
        // caller synchronizes.
        cuda_launch! {
            kernel: crate::kernels::loss::loss_sigmoid_mse_reduce,
            stream: stream.clone(),
            module: module.clone(),
            config: cfg_1d(plan.reduce_threads),
            args: [
                slice(batch.outputs),
                slice(batch.targets),
                slice(batch.entry_weights),
                slice_mut(workspace.per_sample),
                slice_mut(workspace.weighted_sum),
                slice_mut(workspace.mean),
                batch_size
            ]
        }
    }?;

    Ok(())
}

fn validate_loss_layout(batch: &ScalarLossDeviceBatch, workspace: &ScalarLossWorkspace) -> Result<()> {
    expect_layout_value("batch_size", workspace.layout.batch_size, batch.batch_size)
}

fn expect_layout_value(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(LossLayoutError::LayoutValue { name, expected, actual }.into())
    }
}

fn cfg_1d(threads: usize) -> LaunchConfig {
    let threads = threads.clamp(1, u32::MAX as usize) as u32;
    LaunchConfig::for_num_elems(threads)
}
