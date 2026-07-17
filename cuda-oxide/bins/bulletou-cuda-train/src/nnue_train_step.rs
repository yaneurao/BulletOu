//! Host-side NNUE train-step runner used by the cuda-oxide smoke harness.
//!
//! This is deliberately still small and explicit: it owns the persistent
//! device weights, Ranger state, and workspaces, while each call to `step`
//! uploads one fixed-layout host batch and enqueues forward -> loss ->
//! backward -> Ranger update.  The fixture smoke can drive this today; the
//! real trainer loop can later feed the same runner from a dataloader stream.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    backward::{
        DenseCReluBackwardLayout, DenseOutputBackwardLayout, NnueBackwardWorkspace, NnueBackwardWorkspaceLayout,
        NnueL0CReluBackwardLayout, NnueL0SparseBackwardLayout,
    },
    loss::{ScalarLossLayout, ScalarLossWorkspace},
    nnue::{
        NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardHostBatch, NnueForwardHostWeights,
        NnueForwardShape, NnueForwardWorkspace, NnueForwardWorkspaceLayout,
    },
    optimizer::{NnueRangerOptimizerStates, RangerUpdateParams},
    CudaModule, CudaStream, DeviceBuffer, Error, Result,
};

use crate::{dense_backward, loss_forward, nnue_forward, optimizer_update};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NnueTrainLossKind {
    SigmoidMse,
    NnuePytorchWrm,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NnueTrainStepHostBatch<'a> {
    pub(crate) stm_indices: &'a [i32],
    pub(crate) nstm_indices: &'a [i32],
    pub(crate) targets: &'a [f32],
    pub(crate) entry_weights: &'a [f32],
    pub(crate) batch_size: usize,
    pub(crate) max_active: usize,
}

pub(crate) struct NnueLossRangerStepRunner {
    shape: NnueForwardShape,
    batch_size: usize,
    max_active: usize,
    pub(crate) device_weights: NnueForwardDeviceWeights,
    pub(crate) optimizer_states: NnueRangerOptimizerStates,
    forward_workspace: NnueForwardWorkspace,
    pub(crate) loss_workspace: ScalarLossWorkspace,
    backward_workspace: NnueBackwardWorkspace,
}

impl NnueLossRangerStepRunner {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        initial_weights: &NnueForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::Smoke("NNUE train-step runner requires batch_size > 0".to_string()));
        }
        if max_active == 0 {
            return Err(Error::Smoke("NNUE train-step runner requires max_active > 0".to_string()));
        }

        let shape = initial_weights.shape;
        let device_weights = NnueForwardDeviceWeights::from_host(stream, initial_weights)?;
        let optimizer_states = NnueRangerOptimizerStates::from_host_weights(stream, initial_weights)?;
        let forward_workspace =
            NnueForwardWorkspace::new(stream, NnueForwardWorkspaceLayout::new(shape, batch_size))?;
        let loss_workspace = ScalarLossWorkspace::new(stream, ScalarLossLayout::new(batch_size))?;
        let backward_workspace =
            NnueBackwardWorkspace::new(stream, NnueBackwardWorkspaceLayout::new(shape, batch_size, max_active))?;

        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_weights,
            optimizer_states,
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
        loss_kind: NnueTrainLossKind,
        batch: NnueTrainStepHostBatch<'_>,
    ) -> Result<()> {
        self.validate_batch(batch)?;

        let host_batch = NnueForwardHostBatch {
            stm_indices: batch.stm_indices,
            nstm_indices: batch.nstm_indices,
            batch_size: batch.batch_size,
            max_active: batch.max_active,
        };
        let device_batch = NnueForwardDeviceBatch::from_host(stream, &host_batch)?;
        let targets = DeviceBuffer::from_host(stream, batch.targets)?;
        let entry_weights = DeviceBuffer::from_host(stream, batch.entry_weights)?;

        nnue_forward::launch_nnue_forward(
            stream,
            module,
            &device_batch,
            &self.device_weights,
            &mut self.forward_workspace,
        )?;

        match loss_kind {
            NnueTrainLossKind::SigmoidMse => loss_forward::launch_sigmoid_mse_loss_from_buffers(
                stream,
                module,
                &self.forward_workspace.output,
                &targets,
                &entry_weights,
                &mut self.loss_workspace,
            )?,
            NnueTrainLossKind::NnuePytorchWrm => loss_forward::launch_nnue_pytorch_wrm_loss_from_buffers(
                stream,
                module,
                &self.forward_workspace.output,
                &targets,
                &entry_weights,
                &mut self.loss_workspace,
            )?,
        }

        dense_backward::launch_dense_output_backward(
            stream,
            module,
            DenseOutputBackwardLayout::new(self.batch_size, self.shape.l3),
            &self.forward_workspace.hidden2,
            &self.loss_workspace.mean_output_gradients,
            &self.device_weights.outw,
            &mut self.backward_workspace.hidden2_gradients,
            &mut self.backward_workspace.outw_gradients,
            &mut self.backward_workspace.outb_gradients,
        )?;

        dense_backward::launch_dense_crelu_backward(
            stream,
            module,
            DenseCReluBackwardLayout::new(self.batch_size, self.shape.l2, self.shape.l3),
            &self.forward_workspace.hidden1,
            &self.forward_workspace.hidden2,
            &self.backward_workspace.hidden2_gradients,
            &self.device_weights.l2w,
            &mut self.backward_workspace.hidden1_gradients,
            &mut self.backward_workspace.l2w_gradients,
            &mut self.backward_workspace.l2b_gradients,
        )?;

        dense_backward::launch_dense_crelu_backward(
            stream,
            module,
            DenseCReluBackwardLayout::new(self.batch_size, self.shape.l1 * 2, self.shape.l2),
            &self.forward_workspace.combined,
            &self.forward_workspace.hidden1,
            &self.backward_workspace.hidden1_gradients,
            &self.device_weights.l1w,
            &mut self.backward_workspace.combined_gradients,
            &mut self.backward_workspace.l1w_gradients,
            &mut self.backward_workspace.l1b_gradients,
        )?;

        dense_backward::launch_nnue_l0_crelu_backward(
            stream,
            module,
            NnueL0CReluBackwardLayout::new(self.batch_size, self.shape.l1),
            &self.backward_workspace.combined_gradients,
            &self.forward_workspace.stm_l0,
            &self.forward_workspace.nstm_l0,
            &mut self.backward_workspace.stm_l0_gradients,
            &mut self.backward_workspace.nstm_l0_gradients,
        )?;

        dense_backward::launch_nnue_l0_sparse_backward(
            stream,
            module,
            NnueL0SparseBackwardLayout::new(
                self.batch_size,
                self.max_active,
                self.shape.input_size,
                self.shape.l1,
            ),
            &device_batch.stm_indices,
            &device_batch.nstm_indices,
            &self.backward_workspace.stm_l0_gradients,
            &self.backward_workspace.nstm_l0_gradients,
            &mut self.backward_workspace.l0w_gradients,
            &mut self.backward_workspace.l0b_gradients,
        )?;

        optimizer_update::launch_nnue_ranger_update(
            stream,
            module,
            params,
            &mut self.device_weights,
            &self.backward_workspace,
            &mut self.optimizer_states,
        )?;

        Ok(())
    }

    fn validate_batch(&self, batch: NnueTrainStepHostBatch<'_>) -> Result<()> {
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(Error::Smoke(format!(
                "NNUE train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }

        let sparse_len = self.batch_size.saturating_mul(self.max_active);
        if batch.stm_indices.len() != sparse_len || batch.nstm_indices.len() != sparse_len {
            return Err(Error::Smoke(format!(
                "NNUE train-step sparse length mismatch: stm={} nstm={} expected={}",
                batch.stm_indices.len(),
                batch.nstm_indices.len(),
                sparse_len
            )));
        }

        if batch.targets.len() != self.batch_size || batch.entry_weights.len() != self.batch_size {
            return Err(Error::Smoke(format!(
                "NNUE train-step target length mismatch: targets={} entry_weights={} expected={}",
                batch.targets.len(),
                batch.entry_weights.len(),
                self.batch_size
            )));
        }

        Ok(())
    }
}
