//! Host-side NNUE train-step runner used by the cuda-oxide smoke harness.
//!
//! This is deliberately still small and explicit: it owns the persistent
//! device weights, Ranger state, fixed-layout batch buffers, and workspaces,
//! while each call to `step` refills the batch buffers and enqueues forward ->
//! loss -> backward -> Ranger update.  The fixture smoke can drive this today;
//! the real trainer loop can later feed the same runner from a dataloader
//! stream.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    CudaModule, CudaStream, DeviceBuffer, Error, Result,
    backward::{
        DenseCReluBackwardLayout, DenseOutputBackwardLayout, NnueBackwardWorkspace, NnueBackwardWorkspaceLayout,
        NnueL0CReluBackwardLayout, NnueL0SparseBackwardLayout,
    },
    loss::{ScalarLossLayout, ScalarLossWorkspace},
    nnue::{
        NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardHostWeights, NnueForwardShape,
        NnueForwardWeightLayout, NnueForwardWorkspace, NnueForwardWorkspaceLayout,
    },
    optimizer::{NnueRangerOptimizerHostStates, NnueRangerOptimizerStates, RangerUpdateParams},
};
use cuda_core::{CudaEvent, PinnedHostBuffer};

use crate::{cublas::CublasHandle, dense_backward, loss_forward, nnue_forward, optimizer_update};

const NNUE_TRAIN_PIPELINE_SLOTS: usize = 2;

fn cublas_dense_backward_enabled() -> bool {
    match std::env::var("BULLETOU_CUBLAS_DENSE_BACKWARD") {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"),
        Err(_) => true,
    }
}

fn cublas_tf32_enabled() -> bool {
    match std::env::var("BULLETOU_CUBLAS_TF32") {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON"),
        Err(_) => false,
    }
}

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

pub(crate) struct NnueTrainStepLossReadback {
    pub(crate) weighted_sum: Vec<f32>,
    pub(crate) mean: Vec<f32>,
    pub(crate) per_sample: Option<Vec<f32>>,
    pub(crate) mean_output_gradients: Option<Vec<f32>>,
}

pub(crate) struct NnueTrainParamGroupReadback {
    pub(crate) weights: Vec<f32>,
    pub(crate) momentum: Vec<f32>,
    pub(crate) velocity: Vec<f32>,
    pub(crate) slow_params: Vec<f32>,
}

pub(crate) struct NnueTrainWeightsReadback {
    pub(crate) l0w: Vec<f32>,
    pub(crate) l0b: Vec<f32>,
    pub(crate) l1w: Vec<f32>,
    pub(crate) l1b: Vec<f32>,
    pub(crate) l2w: Vec<f32>,
    pub(crate) l2b: Vec<f32>,
    pub(crate) outw: Vec<f32>,
    pub(crate) outb: Vec<f32>,
}

pub(crate) struct NnueTrainStateReadback {
    pub(crate) l0w: NnueTrainParamGroupReadback,
    pub(crate) l0b: NnueTrainParamGroupReadback,
    pub(crate) l1w: NnueTrainParamGroupReadback,
    pub(crate) l1b: NnueTrainParamGroupReadback,
    pub(crate) l2w: NnueTrainParamGroupReadback,
    pub(crate) l2b: NnueTrainParamGroupReadback,
    pub(crate) outw: NnueTrainParamGroupReadback,
    pub(crate) outb: NnueTrainParamGroupReadback,
}

struct PendingLossReadback {
    slot: usize,
    include_debug: bool,
}

struct NnueTrainStepSlot {
    device_batch: NnueForwardDeviceBatch,
    targets: DeviceBuffer<f32>,
    entry_weights: DeviceBuffer<f32>,
    forward_workspace: NnueForwardWorkspace,
    loss_workspace: ScalarLossWorkspace,
    backward_workspace: NnueBackwardWorkspace,
    upload_stm_indices: PinnedHostBuffer<i32>,
    upload_nstm_indices: PinnedHostBuffer<i32>,
    upload_targets: PinnedHostBuffer<f32>,
    upload_entry_weights: PinnedHostBuffer<f32>,
    readback_weighted_sum: PinnedHostBuffer<f32>,
    readback_mean: PinnedHostBuffer<f32>,
    readback_per_sample: PinnedHostBuffer<f32>,
    readback_mean_output_gradients: PinnedHostBuffer<f32>,
    upload_done: Option<CudaEvent>,
    compute_done: Option<CudaEvent>,
    readback_done: Option<CudaEvent>,
}

impl NnueTrainStepSlot {
    fn new(stream: &Arc<CudaStream>, shape: NnueForwardShape, batch_size: usize, max_active: usize) -> Result<Self> {
        let sparse_len = batch_size.saturating_mul(max_active);
        let ctx = stream.context();
        Ok(Self {
            device_batch: NnueForwardDeviceBatch {
                batch_size,
                max_active,
                stm_indices: DeviceBuffer::<i32>::zeroed(stream, sparse_len)?,
                nstm_indices: DeviceBuffer::<i32>::zeroed(stream, sparse_len)?,
            },
            targets: DeviceBuffer::<f32>::zeroed(stream, batch_size)?,
            entry_weights: DeviceBuffer::<f32>::zeroed(stream, batch_size)?,
            forward_workspace: NnueForwardWorkspace::new(stream, NnueForwardWorkspaceLayout::new(shape, batch_size))?,
            loss_workspace: ScalarLossWorkspace::new(stream, ScalarLossLayout::new(batch_size))?,
            backward_workspace: NnueBackwardWorkspace::new(
                stream,
                NnueBackwardWorkspaceLayout::new(shape, batch_size, max_active),
            )?,
            upload_stm_indices: PinnedHostBuffer::<i32>::zeroed(ctx, sparse_len)?,
            upload_nstm_indices: PinnedHostBuffer::<i32>::zeroed(ctx, sparse_len)?,
            upload_targets: PinnedHostBuffer::<f32>::zeroed(ctx, batch_size)?,
            upload_entry_weights: PinnedHostBuffer::<f32>::zeroed(ctx, batch_size)?,
            readback_weighted_sum: PinnedHostBuffer::<f32>::zeroed(ctx, 1)?,
            readback_mean: PinnedHostBuffer::<f32>::zeroed(ctx, 1)?,
            readback_per_sample: PinnedHostBuffer::<f32>::zeroed(ctx, batch_size)?,
            readback_mean_output_gradients: PinnedHostBuffer::<f32>::zeroed(ctx, batch_size)?,
            upload_done: None,
            compute_done: None,
            readback_done: None,
        })
    }
}

pub(crate) struct NnueLossRangerStepRunner {
    shape: NnueForwardShape,
    batch_size: usize,
    max_active: usize,
    device_weights: NnueForwardDeviceWeights,
    optimizer_states: NnueRangerOptimizerStates,
    slots: Vec<NnueTrainStepSlot>,
    upload_stream: Arc<CudaStream>,
    readback_stream: Arc<CudaStream>,
    cublas: CublasHandle,
    use_cublas_dense_backward: bool,
    next_slot: usize,
    last_step_slot: Option<usize>,
    pending_loss: Option<PendingLossReadback>,
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

        Self::from_device_parts(stream, shape, batch_size, max_active, device_weights, optimizer_states)
    }

    pub(crate) fn with_optimizer_state(
        stream: &Arc<CudaStream>,
        initial_weights: &NnueForwardHostWeights<'_>,
        optimizer_state: NnueRangerOptimizerHostStates<'_>,
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
        let optimizer_states =
            NnueRangerOptimizerStates::from_host_states(stream, NnueForwardWeightLayout::new(shape), optimizer_state)?;

        Self::from_device_parts(stream, shape, batch_size, max_active, device_weights, optimizer_states)
    }

    fn from_device_parts(
        stream: &Arc<CudaStream>,
        shape: NnueForwardShape,
        batch_size: usize,
        max_active: usize,
        device_weights: NnueForwardDeviceWeights,
        optimizer_states: NnueRangerOptimizerStates,
    ) -> Result<Self> {
        let upload_stream = stream.fork()?;
        let readback_stream = stream.fork()?;
        let cublas = CublasHandle::new(stream, cublas_tf32_enabled())?;
        let use_cublas_dense_backward = cublas_dense_backward_enabled();
        let mut slots = Vec::with_capacity(NNUE_TRAIN_PIPELINE_SLOTS);
        for _ in 0..NNUE_TRAIN_PIPELINE_SLOTS {
            slots.push(NnueTrainStepSlot::new(stream, shape, batch_size, max_active)?);
        }

        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_weights,
            optimizer_states,
            slots,
            upload_stream,
            readback_stream,
            cublas,
            use_cublas_dense_backward,
            next_slot: 0,
            last_step_slot: None,
            pending_loss: None,
        })
    }

    pub(crate) fn step(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        params: RangerUpdateParams,
        loss_kind: NnueTrainLossKind,
        batch: NnueTrainStepHostBatch<'_>,
        profile: bool,
    ) -> Result<()> {
        let slot = self.next_slot_index();
        self.synchronize_slot_for_blocking_reuse(slot)?;
        self.enqueue_step_with_blocking_upload(stream, module, params, loss_kind, batch, slot, profile)?;
        self.last_step_slot = Some(slot);
        Ok(())
    }

    pub(crate) fn step_pipelined(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        params: RangerUpdateParams,
        loss_kind: NnueTrainLossKind,
        batch: NnueTrainStepHostBatch<'_>,
        include_debug_readback: bool,
        readback_loss: bool,
        drain_before_enqueue: bool,
        profile: bool,
    ) -> Result<Option<NnueTrainStepLossReadback>> {
        let slot = self.next_slot;
        let mut previous = self.pending_loss.take();
        let mut completed = None;

        if drain_before_enqueue || previous.as_ref().is_some_and(|pending| pending.slot == slot) {
            if let Some(pending) = previous.take() {
                completed = Some(self.collect_pending_loss(pending)?);
            }
        }

        self.prepare_slot_for_async_reuse(slot)?;
        self.enqueue_step_with_async_upload_and_readback(
            stream,
            module,
            params,
            loss_kind,
            batch,
            slot,
            include_debug_readback,
            readback_loss,
            profile,
        )?;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        self.last_step_slot = Some(slot);

        if completed.is_none() {
            if let Some(pending) = previous {
                completed = Some(self.collect_pending_loss(pending)?);
            }
        }
        if readback_loss {
            self.pending_loss = Some(PendingLossReadback { slot, include_debug: include_debug_readback });
        }
        Ok(completed)
    }

    pub(crate) fn finish_pipelined_loss(&mut self) -> Result<Option<NnueTrainStepLossReadback>> {
        match self.pending_loss.take() {
            Some(pending) => Ok(Some(self.collect_pending_loss(pending)?)),
            None => Ok(None),
        }
    }

    fn next_slot_index(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        slot
    }

    fn synchronize_slot_for_blocking_reuse(&mut self, slot: usize) -> Result<()> {
        if let Some(event) = self.slots[slot].upload_done.take() {
            event.synchronize()?;
        }
        if let Some(event) = self.slots[slot].compute_done.take() {
            event.synchronize()?;
        }
        if let Some(event) = self.slots[slot].readback_done.take() {
            event.synchronize()?;
        }
        Ok(())
    }

    fn prepare_slot_for_async_reuse(&mut self, slot: usize) -> Result<()> {
        if let Some(event) = self.slots[slot].upload_done.take() {
            event.synchronize()?;
        }
        if let Some(event) = self.slots[slot].readback_done.take() {
            event.synchronize()?;
        }
        if let Some(event) = self.slots[slot].compute_done.take() {
            self.upload_stream.wait(&event)?;
        }
        Ok(())
    }

    fn enqueue_step_with_blocking_upload(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        params: RangerUpdateParams,
        loss_kind: NnueTrainLossKind,
        batch: NnueTrainStepHostBatch<'_>,
        slot: usize,
        profile: bool,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        {
            let slot_ref = &mut self.slots[slot];
            slot_ref.device_batch.stm_indices.copy_from_host(stream, batch.stm_indices)?;
            slot_ref.device_batch.nstm_indices.copy_from_host(stream, batch.nstm_indices)?;
            slot_ref.targets.copy_from_host(stream, batch.targets)?;
            slot_ref.entry_weights.copy_from_host(stream, batch.entry_weights)?;
        }
        self.launch_compute_on_slot(stream, module, params, loss_kind, slot, profile)
    }

    fn enqueue_step_with_async_upload_and_readback(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        params: RangerUpdateParams,
        loss_kind: NnueTrainLossKind,
        batch: NnueTrainStepHostBatch<'_>,
        slot: usize,
        include_debug_readback: bool,
        readback_loss: bool,
        profile: bool,
    ) -> Result<()> {
        self.validate_batch(batch)?;
        {
            let slot_ref = &mut self.slots[slot];
            slot_ref.upload_stm_indices.as_mut_slice().copy_from_slice(batch.stm_indices);
            slot_ref.upload_nstm_indices.as_mut_slice().copy_from_slice(batch.nstm_indices);
            slot_ref.upload_targets.as_mut_slice().copy_from_slice(batch.targets);
            slot_ref.upload_entry_weights.as_mut_slice().copy_from_slice(batch.entry_weights);
            // SAFETY: the slot's pinned upload buffers are not reused until
            // `upload_done` is synchronized in `prepare_slot_for_async_reuse`.
            unsafe {
                slot_ref
                    .device_batch
                    .stm_indices
                    .copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_stm_indices)?;
                slot_ref
                    .device_batch
                    .nstm_indices
                    .copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_nstm_indices)?;
                slot_ref.targets.copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_targets)?;
                slot_ref
                    .entry_weights
                    .copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_entry_weights)?;
            }
        }
        let upload_done = self.upload_stream.record_event(None)?;
        stream.wait(&upload_done)?;
        self.slots[slot].upload_done = Some(upload_done);

        self.launch_compute_on_slot(stream, module, params, loss_kind, slot, profile)?;
        if readback_loss {
            self.enqueue_loss_readback(stream, slot, include_debug_readback)?;
        }
        Ok(())
    }

    fn launch_compute_on_slot(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        params: RangerUpdateParams,
        loss_kind: NnueTrainLossKind,
        slot: usize,
        profile: bool,
    ) -> Result<()> {
        fn profile_stage(stream: &Arc<CudaStream>, mark: &mut std::time::Instant, name: &str) -> Result<()> {
            stream.synchronize()?;
            let now = std::time::Instant::now();
            println!("  profile_step : {name:<16} {:>9.3} ms", now.duration_since(*mark).as_secs_f64() * 1000.0);
            *mark = now;
            Ok(())
        }

        let mut profile_mark = std::time::Instant::now();
        let slot_ref = &mut self.slots[slot];
        nnue_forward::launch_nnue_forward(
            stream,
            module,
            &slot_ref.device_batch,
            &self.device_weights,
            &mut slot_ref.forward_workspace,
        )?;
        if profile {
            profile_stage(stream, &mut profile_mark, "forward")?;
        }

        match loss_kind {
            NnueTrainLossKind::SigmoidMse => loss_forward::launch_sigmoid_mse_loss_from_buffers(
                stream,
                module,
                &slot_ref.forward_workspace.output,
                &slot_ref.targets,
                &slot_ref.entry_weights,
                &mut slot_ref.loss_workspace,
            )?,
            NnueTrainLossKind::NnuePytorchWrm => loss_forward::launch_nnue_pytorch_wrm_loss_from_buffers(
                stream,
                module,
                &slot_ref.forward_workspace.output,
                &slot_ref.targets,
                &slot_ref.entry_weights,
                &mut slot_ref.loss_workspace,
            )?,
        }
        if profile {
            profile_stage(stream, &mut profile_mark, "loss")?;
        }

        dense_backward::launch_dense_output_backward(
            stream,
            module,
            DenseOutputBackwardLayout::new(self.batch_size, self.shape.l3),
            &slot_ref.forward_workspace.hidden2,
            &slot_ref.loss_workspace.mean_output_gradients,
            &self.device_weights.outw,
            &mut slot_ref.backward_workspace.hidden2_gradients,
            &mut slot_ref.backward_workspace.outw_gradients,
            &mut slot_ref.backward_workspace.outb_gradients,
        )?;
        if profile {
            profile_stage(stream, &mut profile_mark, "out_bwd")?;
        }

        let l2_layout = DenseCReluBackwardLayout::new(self.batch_size, self.shape.l2, self.shape.l3);
        if self.use_cublas_dense_backward {
            dense_backward::launch_dense_crelu_backward_cublas(
                stream,
                module,
                &self.cublas,
                l2_layout,
                &slot_ref.forward_workspace.hidden1,
                &slot_ref.forward_workspace.hidden2,
                &slot_ref.backward_workspace.hidden2_gradients,
                &self.device_weights.l2w,
                &mut slot_ref.backward_workspace.hidden1_gradients,
                &mut slot_ref.backward_workspace.l2w_gradients,
                &mut slot_ref.backward_workspace.l2b_gradients,
                &mut slot_ref.backward_workspace.hidden2_pre_gradients,
            )?;
        } else {
            dense_backward::launch_dense_crelu_backward(
                stream,
                module,
                l2_layout,
                &slot_ref.forward_workspace.hidden1,
                &slot_ref.forward_workspace.hidden2,
                &slot_ref.backward_workspace.hidden2_gradients,
                &self.device_weights.l2w,
                &mut slot_ref.backward_workspace.hidden1_gradients,
                &mut slot_ref.backward_workspace.l2w_gradients,
                &mut slot_ref.backward_workspace.l2b_gradients,
            )?;
        }
        if profile {
            profile_stage(stream, &mut profile_mark, "l2_bwd")?;
        }

        let l1_layout = DenseCReluBackwardLayout::new(self.batch_size, self.shape.l1 * 2, self.shape.l2);
        if self.use_cublas_dense_backward {
            dense_backward::launch_dense_crelu_backward_cublas(
                stream,
                module,
                &self.cublas,
                l1_layout,
                &slot_ref.forward_workspace.combined,
                &slot_ref.forward_workspace.hidden1,
                &slot_ref.backward_workspace.hidden1_gradients,
                &self.device_weights.l1w,
                &mut slot_ref.backward_workspace.combined_gradients,
                &mut slot_ref.backward_workspace.l1w_gradients,
                &mut slot_ref.backward_workspace.l1b_gradients,
                &mut slot_ref.backward_workspace.hidden1_pre_gradients,
            )?;
        } else {
            dense_backward::launch_dense_crelu_backward(
                stream,
                module,
                l1_layout,
                &slot_ref.forward_workspace.combined,
                &slot_ref.forward_workspace.hidden1,
                &slot_ref.backward_workspace.hidden1_gradients,
                &self.device_weights.l1w,
                &mut slot_ref.backward_workspace.combined_gradients,
                &mut slot_ref.backward_workspace.l1w_gradients,
                &mut slot_ref.backward_workspace.l1b_gradients,
            )?;
        }
        if profile {
            profile_stage(stream, &mut profile_mark, "l1_bwd")?;
        }

        dense_backward::launch_nnue_l0_crelu_backward(
            stream,
            module,
            NnueL0CReluBackwardLayout::new(self.batch_size, self.shape.l1),
            &slot_ref.backward_workspace.combined_gradients,
            &slot_ref.forward_workspace.stm_l0,
            &slot_ref.forward_workspace.nstm_l0,
            &mut slot_ref.backward_workspace.stm_l0_gradients,
            &mut slot_ref.backward_workspace.nstm_l0_gradients,
        )?;
        if profile {
            profile_stage(stream, &mut profile_mark, "l0_crelu")?;
        }

        dense_backward::launch_nnue_l0_sparse_backward(
            stream,
            module,
            NnueL0SparseBackwardLayout::new(self.batch_size, self.max_active, self.shape.input_size, self.shape.l1),
            &slot_ref.device_batch.stm_indices,
            &slot_ref.device_batch.nstm_indices,
            &slot_ref.backward_workspace.stm_l0_gradients,
            &slot_ref.backward_workspace.nstm_l0_gradients,
            &mut slot_ref.backward_workspace.l0w_gradients,
            &mut slot_ref.backward_workspace.l0b_gradients,
        )?;
        if profile {
            profile_stage(stream, &mut profile_mark, "l0_sparse")?;
        }

        optimizer_update::launch_nnue_ranger_update(
            stream,
            module,
            params,
            &mut self.device_weights,
            &slot_ref.backward_workspace,
            &mut self.optimizer_states,
        )?;
        if profile {
            profile_stage(stream, &mut profile_mark, "optimizer")?;
        }

        self.slots[slot].compute_done = Some(stream.record_event(None)?);
        Ok(())
    }

    fn enqueue_loss_readback(&mut self, stream: &Arc<CudaStream>, slot: usize, include_debug: bool) -> Result<()> {
        let loss_ready = stream.record_event(None)?;
        self.readback_stream.wait(&loss_ready)?;
        {
            let slot_ref = &mut self.slots[slot];
            // SAFETY: readback pinned buffers are not read or reused until the
            // recorded `readback_done` event is synchronized in
            // `collect_pending_loss` or slot reuse.
            unsafe {
                slot_ref
                    .loss_workspace
                    .weighted_sum
                    .copy_to_pinned_host_async(&self.readback_stream, &mut slot_ref.readback_weighted_sum)?;
                slot_ref
                    .loss_workspace
                    .mean
                    .copy_to_pinned_host_async(&self.readback_stream, &mut slot_ref.readback_mean)?;
                if include_debug {
                    slot_ref
                        .loss_workspace
                        .per_sample
                        .copy_to_pinned_host_async(&self.readback_stream, &mut slot_ref.readback_per_sample)?;
                    slot_ref.loss_workspace.mean_output_gradients.copy_to_pinned_host_async(
                        &self.readback_stream,
                        &mut slot_ref.readback_mean_output_gradients,
                    )?;
                }
            }
        }
        self.slots[slot].readback_done = Some(self.readback_stream.record_event(None)?);
        Ok(())
    }

    fn collect_pending_loss(&mut self, pending: PendingLossReadback) -> Result<NnueTrainStepLossReadback> {
        let slot_ref = &mut self.slots[pending.slot];
        let event = slot_ref
            .readback_done
            .take()
            .ok_or_else(|| Error::Smoke("internal NNUE train pipeline loss readback was not enqueued".to_string()))?;
        event.synchronize()?;
        Ok(NnueTrainStepLossReadback {
            weighted_sum: slot_ref.readback_weighted_sum.as_slice().to_vec(),
            mean: slot_ref.readback_mean.as_slice().to_vec(),
            per_sample: pending.include_debug.then(|| slot_ref.readback_per_sample.as_slice().to_vec()),
            mean_output_gradients: pending
                .include_debug
                .then(|| slot_ref.readback_mean_output_gradients.as_slice().to_vec()),
        })
    }

    pub(crate) fn read_loss(&self, stream: &Arc<CudaStream>, include_debug: bool) -> Result<NnueTrainStepLossReadback> {
        let slot_ref = self
            .last_step_slot
            .and_then(|slot| self.slots.get(slot))
            .ok_or_else(|| Error::Smoke("NNUE train-step loss readback requested before any step".to_string()))?;
        Ok(NnueTrainStepLossReadback {
            weighted_sum: slot_ref.loss_workspace.weighted_sum.to_host_vec(stream)?,
            mean: slot_ref.loss_workspace.mean.to_host_vec(stream)?,
            per_sample: include_debug.then(|| slot_ref.loss_workspace.per_sample.to_host_vec(stream)).transpose()?,
            mean_output_gradients: include_debug
                .then(|| slot_ref.loss_workspace.mean_output_gradients.to_host_vec(stream))
                .transpose()?,
        })
    }

    pub(crate) fn read_weights(&self, stream: &Arc<CudaStream>) -> Result<NnueTrainWeightsReadback> {
        Ok(NnueTrainWeightsReadback {
            l0w: self.device_weights.l0w.to_host_vec(stream)?,
            l0b: self.device_weights.l0b.to_host_vec(stream)?,
            l1w: self.device_weights.l1w.to_host_vec(stream)?,
            l1b: self.device_weights.l1b.to_host_vec(stream)?,
            l2w: self.device_weights.l2w.to_host_vec(stream)?,
            l2b: self.device_weights.l2b.to_host_vec(stream)?,
            outw: self.device_weights.outw.to_host_vec(stream)?,
            outb: self.device_weights.outb.to_host_vec(stream)?,
        })
    }

    pub(crate) fn read_state(&self, stream: &Arc<CudaStream>) -> Result<NnueTrainStateReadback> {
        macro_rules! read_group {
            ($field:ident) => {
                NnueTrainParamGroupReadback {
                    weights: self.device_weights.$field.to_host_vec(stream)?,
                    momentum: self.optimizer_states.$field.momentum.to_host_vec(stream)?,
                    velocity: self.optimizer_states.$field.velocity.to_host_vec(stream)?,
                    slow_params: self.optimizer_states.$field.slow_params.to_host_vec(stream)?,
                }
            };
        }

        Ok(NnueTrainStateReadback {
            l0w: read_group!(l0w),
            l0b: read_group!(l0b),
            l1w: read_group!(l1w),
            l1b: read_group!(l1b),
            l2w: read_group!(l2w),
            l2b: read_group!(l2b),
            outw: read_group!(outw),
            outb: read_group!(outb),
        })
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
