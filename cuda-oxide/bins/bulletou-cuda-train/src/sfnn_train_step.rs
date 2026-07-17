//! Host-side SFNN train-step runner used by the cuda-oxide teacher harness.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    backward::{
        SfnnBackwardWorkspace, SfnnBackwardWorkspaceLayout, SfnnL0SparseBackwardLayout, SfnnL2InputBackwardLayout,
        SfnnPairwiseBackwardLayout, SfnnStackedAffineBackwardLayout, SfnnStackedCReluBackwardLayout,
        SfnnStackedL3BackwardLayout,
    },
    loss::{ScalarLossLayout, ScalarLossWorkspace},
    optimizer::{RangerUpdateParams, SfnnRangerOptimizerStates},
    sfnn::{
        SfnnForwardDeviceBatch, SfnnForwardDeviceWeights, SfnnForwardHostWeights, SfnnForwardShape,
        SfnnForwardWorkspace, SfnnForwardWorkspaceLayout,
    },
    CudaModule, CudaStream, DeviceBuffer, Error, Result,
};

use crate::{loss_forward, optimizer_update, sfnn_backward, sfnn_forward};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SfnnTrainLossKind {
    SigmoidMse,
    NnuePytorchWrm,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SfnnTrainStepHostBatch<'a> {
    pub(crate) stm_indices: &'a [i32],
    pub(crate) nstm_indices: &'a [i32],
    pub(crate) buckets: &'a [i32],
    pub(crate) targets: &'a [f32],
    pub(crate) entry_weights: &'a [f32],
    pub(crate) batch_size: usize,
    pub(crate) max_active: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SfnnTrainStepLossReadback {
    pub(crate) weighted_sum: Vec<f32>,
    pub(crate) mean: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct SfnnTrainParamGroupReadback {
    pub(crate) weights: Vec<f32>,
    pub(crate) momentum: Vec<f32>,
    pub(crate) velocity: Vec<f32>,
    pub(crate) slow_params: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct SfnnTrainStateReadback {
    pub(crate) l0w: SfnnTrainParamGroupReadback,
    pub(crate) l0b: SfnnTrainParamGroupReadback,
    pub(crate) l1w: SfnnTrainParamGroupReadback,
    pub(crate) l1b: SfnnTrainParamGroupReadback,
    pub(crate) l2w: SfnnTrainParamGroupReadback,
    pub(crate) l2b: SfnnTrainParamGroupReadback,
    pub(crate) l3w: SfnnTrainParamGroupReadback,
    pub(crate) l3b: SfnnTrainParamGroupReadback,
}

pub(crate) struct SfnnLossRangerStepRunner {
    shape: SfnnForwardShape,
    batch_size: usize,
    max_active: usize,
    device_weights: SfnnForwardDeviceWeights,
    optimizer_states: SfnnRangerOptimizerStates,
    device_batch: SfnnForwardDeviceBatch,
    targets: DeviceBuffer<f32>,
    entry_weights: DeviceBuffer<f32>,
    forward_workspace: SfnnForwardWorkspace,
    loss_workspace: ScalarLossWorkspace,
    backward_workspace: SfnnBackwardWorkspace,
}

impl SfnnLossRangerStepRunner {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        weights: &SfnnForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        weights.validate()?;
        let shape = weights.shape;
        let sparse_len = batch_size.saturating_mul(max_active);
        let device_weights = SfnnForwardDeviceWeights::from_host(stream, weights)?;
        let optimizer_states = SfnnRangerOptimizerStates::from_host_weights(stream, weights)?;
        let device_batch = SfnnForwardDeviceBatch {
            batch_size,
            max_active,
            stm_indices: DeviceBuffer::<i32>::zeroed(stream, sparse_len)?,
            nstm_indices: DeviceBuffer::<i32>::zeroed(stream, sparse_len)?,
            buckets: DeviceBuffer::<i32>::zeroed(stream, batch_size)?,
        };
        let targets = DeviceBuffer::<f32>::zeroed(stream, batch_size)?;
        let entry_weights = DeviceBuffer::<f32>::zeroed(stream, batch_size)?;
        let forward_workspace = SfnnForwardWorkspace::new(stream, SfnnForwardWorkspaceLayout::new(shape, batch_size))?;
        let loss_workspace = ScalarLossWorkspace::new(stream, ScalarLossLayout::new(batch_size))?;
        let backward_workspace =
            SfnnBackwardWorkspace::new(stream, SfnnBackwardWorkspaceLayout::new(shape, batch_size, max_active))?;

        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_weights,
            optimizer_states,
            device_batch,
            targets,
            entry_weights,
            forward_workspace,
            loss_workspace,
            backward_workspace,
        })
    }

    pub(crate) fn step(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        params: RangerUpdateParams,
        loss_kind: SfnnTrainLossKind,
        batch: SfnnTrainStepHostBatch<'_>,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        self.device_batch.stm_indices.copy_from_host(stream, batch.stm_indices)?;
        self.device_batch.nstm_indices.copy_from_host(stream, batch.nstm_indices)?;
        self.device_batch.buckets.copy_from_host(stream, batch.buckets)?;
        self.targets.copy_from_host(stream, batch.targets)?;
        self.entry_weights.copy_from_host(stream, batch.entry_weights)?;

        sfnn_forward::launch_sfnn_forward(
            stream,
            module,
            &self.device_batch,
            &self.device_weights,
            &mut self.forward_workspace,
        )?;

        match loss_kind {
            SfnnTrainLossKind::SigmoidMse => loss_forward::launch_sigmoid_mse_loss_from_buffers(
                stream,
                module,
                &self.forward_workspace.output,
                &self.targets,
                &self.entry_weights,
                &mut self.loss_workspace,
            )?,
            SfnnTrainLossKind::NnuePytorchWrm => loss_forward::launch_nnue_pytorch_wrm_loss_from_buffers(
                stream,
                module,
                &self.forward_workspace.output,
                &self.targets,
                &self.entry_weights,
                &mut self.loss_workspace,
            )?,
        }

        let l3_layout =
            SfnnStackedL3BackwardLayout::new(self.batch_size, self.shape.l2_size, self.shape.l1_out(), self.shape.num_stacks);
        sfnn_backward::launch_sfnn_stacked_l3_backward(
            stream,
            module,
            l3_layout,
            &self.forward_workspace.l2,
            &self.loss_workspace.mean_output_gradients,
            &self.device_weights.l3w,
            &self.device_batch.buckets,
            &mut self.backward_workspace.l2_gradients,
            &mut self.backward_workspace.l1_gradients,
            &mut self.backward_workspace.l3w_gradients,
            &mut self.backward_workspace.l3b_gradients,
        )?;

        let l2_layout =
            SfnnStackedCReluBackwardLayout::new(self.batch_size, self.shape.l2_in(), self.shape.l2_size, self.shape.num_stacks);
        sfnn_backward::launch_sfnn_stacked_crelu_backward(
            stream,
            module,
            l2_layout,
            &self.forward_workspace.l2_input,
            &self.forward_workspace.l2,
            &self.backward_workspace.l2_gradients,
            &self.device_weights.l2w,
            &self.device_batch.buckets,
            &mut self.backward_workspace.l2_input_gradients,
            &mut self.backward_workspace.l2w_gradients,
            &mut self.backward_workspace.l2b_gradients,
        )?;

        let l2_input_layout = SfnnL2InputBackwardLayout::new(self.batch_size, self.shape.l1_hidden);
        sfnn_backward::launch_sfnn_l2_input_backward(
            stream,
            module,
            l2_input_layout,
            &self.forward_workspace.l1,
            &self.forward_workspace.l2_input,
            &self.backward_workspace.l2_input_gradients,
            &mut self.backward_workspace.l1_gradients,
        )?;

        let l1_layout =
            SfnnStackedAffineBackwardLayout::new(self.batch_size, self.shape.ft_size, self.shape.l1_out(), self.shape.num_stacks);
        sfnn_backward::launch_sfnn_stacked_affine_backward(
            stream,
            module,
            l1_layout,
            &self.forward_workspace.combined,
            &self.backward_workspace.l1_gradients,
            &self.device_weights.l1w,
            &self.device_batch.buckets,
            &mut self.backward_workspace.combined_gradients,
            &mut self.backward_workspace.l1w_gradients,
            &mut self.backward_workspace.l1b_gradients,
        )?;

        let pairwise_layout = SfnnPairwiseBackwardLayout::new(self.batch_size, self.shape.ft_size);
        sfnn_backward::launch_sfnn_pairwise_backward(
            stream,
            module,
            pairwise_layout,
            &self.forward_workspace.stm_l0,
            &self.forward_workspace.nstm_l0,
            &self.backward_workspace.combined_gradients,
            &mut self.backward_workspace.stm_l0_gradients,
            &mut self.backward_workspace.nstm_l0_gradients,
        )?;

        let l0_layout =
            SfnnL0SparseBackwardLayout::new(self.batch_size, self.max_active, self.shape.input_size, self.shape.ft_size);
        sfnn_backward::launch_sfnn_l0_sparse_backward(
            stream,
            module,
            l0_layout,
            &self.device_batch.stm_indices,
            &self.device_batch.nstm_indices,
            &self.forward_workspace.stm_l0,
            &self.forward_workspace.nstm_l0,
            &self.backward_workspace.stm_l0_gradients,
            &self.backward_workspace.nstm_l0_gradients,
            &mut self.backward_workspace.stm_l0_pre_gradients,
            &mut self.backward_workspace.nstm_l0_pre_gradients,
            &mut self.backward_workspace.l0w_gradients,
            &mut self.backward_workspace.l0b_gradients,
        )?;

        optimizer_update::launch_sfnn_ranger_update(
            stream,
            module,
            params,
            &mut self.device_weights,
            &self.backward_workspace,
            &mut self.optimizer_states,
        )
    }

    pub(crate) fn read_loss(&self, stream: &Arc<CudaStream>) -> Result<SfnnTrainStepLossReadback> {
        Ok(SfnnTrainStepLossReadback {
            weighted_sum: self.loss_workspace.weighted_sum.to_host_vec(stream)?,
            mean: self.loss_workspace.mean.to_host_vec(stream)?,
        })
    }

    pub(crate) fn read_state(&self, stream: &Arc<CudaStream>) -> Result<SfnnTrainStateReadback> {
        macro_rules! read_group {
            ($field:ident) => {
                SfnnTrainParamGroupReadback {
                    weights: self.device_weights.$field.to_host_vec(stream)?,
                    momentum: self.optimizer_states.$field.momentum.to_host_vec(stream)?,
                    velocity: self.optimizer_states.$field.velocity.to_host_vec(stream)?,
                    slow_params: self.optimizer_states.$field.slow_params.to_host_vec(stream)?,
                }
            };
        }

        Ok(SfnnTrainStateReadback {
            l0w: read_group!(l0w),
            l0b: read_group!(l0b),
            l1w: read_group!(l1w),
            l1b: read_group!(l1b),
            l2w: read_group!(l2w),
            l2b: read_group!(l2b),
            l3w: read_group!(l3w),
            l3b: read_group!(l3b),
        })
    }

    fn validate_batch(&self, batch: SfnnTrainStepHostBatch<'_>) -> Result<()> {
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(Error::Smoke(format!(
                "SFNN train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }

        let sparse_len = self.batch_size.saturating_mul(self.max_active);
        if batch.stm_indices.len() != sparse_len || batch.nstm_indices.len() != sparse_len {
            return Err(Error::Smoke(format!(
                "SFNN train-step sparse length mismatch: stm={} nstm={} expected={}",
                batch.stm_indices.len(),
                batch.nstm_indices.len(),
                sparse_len
            )));
        }

        if batch.buckets.len() != self.batch_size
            || batch.targets.len() != self.batch_size
            || batch.entry_weights.len() != self.batch_size
        {
            return Err(Error::Smoke(format!(
                "SFNN train-step dense length mismatch: buckets={} targets={} entry_weights={} expected={}",
                batch.buckets.len(),
                batch.targets.len(),
                batch.entry_weights.len(),
                self.batch_size
            )));
        }

        Ok(())
    }
}
