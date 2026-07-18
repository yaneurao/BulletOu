//! Host-side SFNN train-step runner used by the cuda-oxide teacher harness.

use std::sync::Arc;

use bulletou_cuda_oxide_runtime::{
    CudaModule, CudaStream, DeviceBuffer, Error, Result,
    backward::{
        SfnnBackwardWorkspace, SfnnBackwardWorkspaceLayout, SfnnL0SparseBackwardLayout, SfnnL2InputBackwardLayout,
        SfnnStackedAffineBackwardLayout, SfnnStackedCReluBackwardLayout, SfnnStackedL3BackwardLayout,
    },
    loss::{ScalarLossLayout, ScalarLossWorkspace},
    optimizer::{RangerUpdateParams, SfnnRangerOptimizerHostStates, SfnnRangerOptimizerStates},
    sfnn::{
        SfnnForwardDeviceBatch, SfnnForwardDeviceWeights, SfnnForwardHostWeights, SfnnForwardShape,
        SfnnForwardWeightLayout, SfnnForwardWorkspace, SfnnForwardWorkspaceLayout,
    },
};
use cuda_core::{CudaEvent, PinnedHostBuffer};

use crate::{loss_forward, optimizer_update, sfnn_backward, sfnn_forward};

const SFNN_TRAIN_PIPELINE_SLOTS: usize = 2;

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
    pub(crate) l1fw: Option<SfnnTrainParamGroupReadback>,
    pub(crate) l1fb: Option<SfnnTrainParamGroupReadback>,
    pub(crate) l2w: SfnnTrainParamGroupReadback,
    pub(crate) l2b: SfnnTrainParamGroupReadback,
    pub(crate) l3w: SfnnTrainParamGroupReadback,
    pub(crate) l3b: SfnnTrainParamGroupReadback,
}

struct SfnnTrainStepSlot {
    device_batch: SfnnForwardDeviceBatch,
    targets: DeviceBuffer<f32>,
    entry_weights: DeviceBuffer<f32>,
    upload_stm_indices: PinnedHostBuffer<i32>,
    upload_nstm_indices: PinnedHostBuffer<i32>,
    upload_buckets: PinnedHostBuffer<i32>,
    upload_targets: PinnedHostBuffer<f32>,
    upload_entry_weights: PinnedHostBuffer<f32>,
    upload_done: Option<CudaEvent>,
    compute_done: Option<CudaEvent>,
}

impl SfnnTrainStepSlot {
    fn new(stream: &Arc<CudaStream>, batch_size: usize, max_active: usize) -> Result<Self> {
        let sparse_len = batch_size.saturating_mul(max_active);
        let ctx = stream.context();
        Ok(Self {
            device_batch: SfnnForwardDeviceBatch {
                batch_size,
                max_active,
                stm_indices: DeviceBuffer::<i32>::zeroed(stream, sparse_len)?,
                nstm_indices: DeviceBuffer::<i32>::zeroed(stream, sparse_len)?,
                buckets: DeviceBuffer::<i32>::zeroed(stream, batch_size)?,
            },
            targets: DeviceBuffer::<f32>::zeroed(stream, batch_size)?,
            entry_weights: DeviceBuffer::<f32>::zeroed(stream, batch_size)?,
            upload_stm_indices: PinnedHostBuffer::<i32>::zeroed(ctx, sparse_len)?,
            upload_nstm_indices: PinnedHostBuffer::<i32>::zeroed(ctx, sparse_len)?,
            upload_buckets: PinnedHostBuffer::<i32>::zeroed(ctx, batch_size)?,
            upload_targets: PinnedHostBuffer::<f32>::zeroed(ctx, batch_size)?,
            upload_entry_weights: PinnedHostBuffer::<f32>::zeroed(ctx, batch_size)?,
            upload_done: None,
            compute_done: None,
        })
    }
}

pub(crate) struct SfnnLossRangerStepRunner {
    shape: SfnnForwardShape,
    batch_size: usize,
    max_active: usize,
    device_weights: SfnnForwardDeviceWeights,
    folded_l0w: Option<DeviceBuffer<f32>>,
    optimizer_states: SfnnRangerOptimizerStates,
    slots: Vec<SfnnTrainStepSlot>,
    upload_stream: Arc<CudaStream>,
    next_slot: usize,
    forward_workspace: SfnnForwardWorkspace,
    loss_workspace: ScalarLossWorkspace,
    backward_workspace: SfnnBackwardWorkspace,
}

impl SfnnLossRangerStepRunner {
    pub(crate) fn new_from_scratch(
        stream: &Arc<CudaStream>,
        weights: &SfnnForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        Self::new_with_optimizer_state_factory(
            stream,
            weights,
            batch_size,
            max_active,
            SfnnRangerOptimizerStates::zeroed_for_host_weights,
        )
    }

    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        weights: &SfnnForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        Self::new_with_optimizer_state_factory(
            stream,
            weights,
            batch_size,
            max_active,
            SfnnRangerOptimizerStates::from_host_weights,
        )
    }

    fn new_with_optimizer_state_factory(
        stream: &Arc<CudaStream>,
        weights: &SfnnForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
        optimizer_states: fn(&CudaStream, &SfnnForwardHostWeights<'_>) -> Result<SfnnRangerOptimizerStates>,
    ) -> Result<Self> {
        weights.validate()?;
        let shape = weights.shape;
        let device_weights = SfnnForwardDeviceWeights::from_host(stream, weights)?;
        let optimizer_states = optimizer_states(stream, weights)?;
        let mut slots = Vec::with_capacity(SFNN_TRAIN_PIPELINE_SLOTS);
        for _ in 0..SFNN_TRAIN_PIPELINE_SLOTS {
            slots.push(SfnnTrainStepSlot::new(stream, batch_size, max_active)?);
        }
        let upload_stream = stream.fork()?;
        let forward_workspace = SfnnForwardWorkspace::new(stream, SfnnForwardWorkspaceLayout::new(shape, batch_size))?;
        let loss_workspace = ScalarLossWorkspace::new(stream, ScalarLossLayout::new(batch_size))?;
        let backward_workspace =
            SfnnBackwardWorkspace::new(stream, SfnnBackwardWorkspaceLayout::new(shape, batch_size, max_active))?;
        let folded_l0w = sfnn_halfka2_folded_l0w(stream, shape, batch_size)?;

        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_weights,
            folded_l0w,
            optimizer_states,
            slots,
            upload_stream,
            next_slot: 0,
            forward_workspace,
            loss_workspace,
            backward_workspace,
        })
    }

    pub(crate) fn with_optimizer_state(
        stream: &Arc<CudaStream>,
        weights: &SfnnForwardHostWeights<'_>,
        optimizer_state: SfnnRangerOptimizerHostStates<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        weights.validate()?;
        let shape = weights.shape;
        let device_weights = SfnnForwardDeviceWeights::from_host(stream, weights)?;
        let optimizer_states =
            SfnnRangerOptimizerStates::from_host_states(stream, SfnnForwardWeightLayout::new(shape), optimizer_state)?;
        let mut slots = Vec::with_capacity(SFNN_TRAIN_PIPELINE_SLOTS);
        for _ in 0..SFNN_TRAIN_PIPELINE_SLOTS {
            slots.push(SfnnTrainStepSlot::new(stream, batch_size, max_active)?);
        }
        let upload_stream = stream.fork()?;
        let forward_workspace = SfnnForwardWorkspace::new(stream, SfnnForwardWorkspaceLayout::new(shape, batch_size))?;
        let loss_workspace = ScalarLossWorkspace::new(stream, ScalarLossLayout::new(batch_size))?;
        let backward_workspace =
            SfnnBackwardWorkspace::new(stream, SfnnBackwardWorkspaceLayout::new(shape, batch_size, max_active))?;
        let folded_l0w = sfnn_halfka2_folded_l0w(stream, shape, batch_size)?;

        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_weights,
            folded_l0w,
            optimizer_states,
            slots,
            upload_stream,
            next_slot: 0,
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
        sigmoid_output_scale: f32,
        batch: SfnnTrainStepHostBatch<'_>,
        profile: bool,
    ) -> Result<()> {
        let slot = self.next_slot_index();
        self.prepare_slot_for_async_reuse(slot)?;
        self.validate_batch(batch)?;
        let mut profile_last = if profile {
            stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        {
            let slot_ref = &mut self.slots[slot];
            slot_ref.upload_stm_indices.as_mut_slice().copy_from_slice(batch.stm_indices);
            slot_ref.upload_nstm_indices.as_mut_slice().copy_from_slice(batch.nstm_indices);
            slot_ref.upload_buckets.as_mut_slice().copy_from_slice(batch.buckets);
            slot_ref.upload_targets.as_mut_slice().copy_from_slice(batch.targets);
            slot_ref.upload_entry_weights.as_mut_slice().copy_from_slice(batch.entry_weights);
            // SAFETY: the slot's pinned upload buffers are not reused until
            // `upload_done` is synchronized by `prepare_slot_for_async_reuse`.
            unsafe {
                slot_ref
                    .device_batch
                    .stm_indices
                    .copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_stm_indices)?;
                slot_ref
                    .device_batch
                    .nstm_indices
                    .copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_nstm_indices)?;
                slot_ref
                    .device_batch
                    .buckets
                    .copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_buckets)?;
                slot_ref.targets.copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_targets)?;
                slot_ref
                    .entry_weights
                    .copy_from_pinned_host_async(&self.upload_stream, &slot_ref.upload_entry_weights)?;
            }
        }
        let upload_done = self.upload_stream.record_event(None)?;
        stream.wait(&upload_done)?;
        self.slots[slot].upload_done = Some(upload_done);
        profile_stage(stream, &mut profile_last, "upload")?;

        self.launch_compute_on_slot(stream, module, params, loss_kind, sigmoid_output_scale, slot, &mut profile_last)?;
        self.slots[slot].compute_done = Some(stream.record_event(None)?);
        Ok(())
    }

    fn next_slot_index(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        slot
    }

    fn prepare_slot_for_async_reuse(&mut self, slot: usize) -> Result<()> {
        if let Some(event) = self.slots[slot].upload_done.take() {
            event.synchronize()?;
        }
        if let Some(event) = self.slots[slot].compute_done.take() {
            self.upload_stream.wait(&event)?;
        }
        Ok(())
    }

    fn launch_compute_on_slot(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        params: RangerUpdateParams,
        loss_kind: SfnnTrainLossKind,
        sigmoid_output_scale: f32,
        slot: usize,
        profile_last: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        let slot_ref = &mut self.slots[slot];
        if let Some(folded_l0w) = &mut self.folded_l0w {
            sfnn_forward::launch_sfnn_halfka2_fold_factorized_l0w(
                stream,
                module,
                &self.device_weights.l0w,
                folded_l0w,
                self.shape.ft_size,
            )?;
            profile_stage(stream, profile_last, "fold_l0w")?;
            sfnn_forward::launch_sfnn_forward_with_l0(
                stream,
                module,
                &slot_ref.device_batch,
                &self.device_weights,
                &mut self.forward_workspace,
                folded_l0w,
                SFNN_HALFKA2_BASE_INPUT_SIZE,
            )?;
        } else {
            sfnn_forward::launch_sfnn_forward(
                stream,
                module,
                &slot_ref.device_batch,
                &self.device_weights,
                &mut self.forward_workspace,
            )?;
        }
        profile_stage(stream, profile_last, "forward")?;

        match loss_kind {
            SfnnTrainLossKind::SigmoidMse => loss_forward::launch_sigmoid_mse_loss_from_buffers(
                stream,
                module,
                &self.forward_workspace.output,
                &slot_ref.targets,
                &slot_ref.entry_weights,
                sigmoid_output_scale,
                &mut self.loss_workspace,
            )?,
            SfnnTrainLossKind::NnuePytorchWrm => loss_forward::launch_nnue_pytorch_wrm_loss_from_buffers(
                stream,
                module,
                &self.forward_workspace.output,
                &slot_ref.targets,
                &slot_ref.entry_weights,
                &mut self.loss_workspace,
            )?,
        }
        profile_stage(stream, profile_last, "loss")?;

        let l3_layout = SfnnStackedL3BackwardLayout::new(
            self.batch_size,
            self.shape.l2_size,
            self.shape.l1_out(),
            self.shape.num_stacks,
        );
        sfnn_backward::launch_sfnn_stacked_l3_backward(
            stream,
            module,
            l3_layout,
            &self.forward_workspace.l2,
            &self.loss_workspace.mean_output_gradients,
            &self.device_weights.l3w,
            &slot_ref.device_batch.buckets,
            &mut self.backward_workspace.l2_gradients,
            &mut self.backward_workspace.l1_gradients,
            &mut self.backward_workspace.l3w_gradients,
            &mut self.backward_workspace.l3b_gradients,
        )?;
        profile_stage(stream, profile_last, "backward_l3")?;

        let l2_layout = SfnnStackedCReluBackwardLayout::new(
            self.batch_size,
            self.shape.l2_in(),
            self.shape.l2_size,
            self.shape.num_stacks,
        );
        sfnn_backward::launch_sfnn_stacked_crelu_backward(
            stream,
            module,
            l2_layout,
            &self.forward_workspace.l2_input,
            &self.forward_workspace.l2,
            &self.backward_workspace.l2_gradients,
            &self.device_weights.l2w,
            &slot_ref.device_batch.buckets,
            &mut self.backward_workspace.l2_input_gradients,
            &mut self.backward_workspace.l2w_gradients,
            &mut self.backward_workspace.l2b_gradients,
        )?;
        profile_stage(stream, profile_last, "backward_l2_crelu")?;

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
        profile_stage(stream, profile_last, "backward_l2_input")?;

        let l1_layout = SfnnStackedAffineBackwardLayout::new(
            self.batch_size,
            self.shape.ft_size,
            self.shape.l1_out(),
            self.shape.num_stacks,
        );
        if let (Some(l1fw), Some(_l1fb)) = (&self.device_weights.l1fw, &self.device_weights.l1fb) {
            sfnn_backward::launch_sfnn_factorized_l1_backward(
                stream,
                module,
                l1_layout,
                &self.forward_workspace.combined,
                &self.backward_workspace.l1_gradients,
                &self.device_weights.l1w,
                l1fw,
                &slot_ref.device_batch.buckets,
                &mut self.backward_workspace.combined_gradients,
                &mut self.backward_workspace.l1w_gradients,
                &mut self.backward_workspace.l1b_gradients,
                &mut self.backward_workspace.l1fw_gradients,
                &mut self.backward_workspace.l1fb_gradients,
            )?;
            profile_stage(stream, profile_last, "backward_l1_factorized")?;
        } else {
            sfnn_backward::launch_sfnn_stacked_affine_backward(
                stream,
                module,
                l1_layout,
                &self.forward_workspace.combined,
                &self.backward_workspace.l1_gradients,
                &self.device_weights.l1w,
                &slot_ref.device_batch.buckets,
                &mut self.backward_workspace.combined_gradients,
                &mut self.backward_workspace.l1w_gradients,
                &mut self.backward_workspace.l1b_gradients,
            )?;
            profile_stage(stream, profile_last, "backward_l1")?;
        }

        let l0_layout = SfnnL0SparseBackwardLayout::new(
            self.batch_size,
            self.max_active,
            self.shape.input_size,
            self.shape.ft_size,
        );
        sfnn_backward::launch_sfnn_pairwise_l0_sparse_backward_train(
            stream,
            module,
            l0_layout,
            &slot_ref.device_batch.stm_indices,
            &slot_ref.device_batch.nstm_indices,
            &self.forward_workspace.stm_l0,
            &self.forward_workspace.nstm_l0,
            &self.backward_workspace.combined_gradients,
            &mut self.backward_workspace.l0w_gradients,
            &mut self.backward_workspace.l0b_gradients,
        )?;
        if sfnn_halfka2_ft_factorized_input_size(self.shape.input_size) {
            sfnn_backward::launch_sfnn_halfka2_ft_factorized_l0_reduce_virtual_grad(
                stream,
                module,
                l0_layout,
                &mut self.backward_workspace.l0w_gradients,
            )?;
        }
        profile_stage(stream, profile_last, "backward_pairwise_l0")?;

        optimizer_update::launch_sfnn_ranger_update(
            stream,
            module,
            params,
            &mut self.device_weights,
            &mut self.backward_workspace,
            &mut self.optimizer_states,
        )?;
        profile_stage(stream, profile_last, "optimizer")?;
        Ok(())
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
        macro_rules! read_optional_group {
            ($field:ident) => {
                match (&self.device_weights.$field, &self.optimizer_states.$field) {
                    (Some(weights), Some(state)) => Some(SfnnTrainParamGroupReadback {
                        weights: weights.to_host_vec(stream)?,
                        momentum: state.momentum.to_host_vec(stream)?,
                        velocity: state.velocity.to_host_vec(stream)?,
                        slow_params: state.slow_params.to_host_vec(stream)?,
                    }),
                    (None, None) => None,
                    _ => {
                        return Err(Error::Smoke(format!(
                            "SFNN optional optimizer state mismatch for {}",
                            stringify!($field)
                        )));
                    }
                }
            };
        }

        Ok(SfnnTrainStateReadback {
            l0w: read_group!(l0w),
            l0b: read_group!(l0b),
            l1w: read_group!(l1w),
            l1b: read_group!(l1b),
            l1fw: read_optional_group!(l1fw),
            l1fb: read_optional_group!(l1fb),
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

const SFNN_HALFKA2_BASE_INPUT_SIZE: usize = 131_949;
const SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS: usize = 1_629;
const SFNN_HALFKA2_FT_FACTORIZE_INPUT_SIZE: usize =
    SFNN_HALFKA2_BASE_INPUT_SIZE + SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS;
const SFNN_HALFKA2_FOLD_L0W_MIN_BATCH: usize = 65_536;

fn sfnn_halfka2_ft_factorized_input_size(input_size: usize) -> bool {
    input_size == SFNN_HALFKA2_FT_FACTORIZE_INPUT_SIZE
}

fn sfnn_halfka2_folded_l0w(
    stream: &Arc<CudaStream>,
    shape: SfnnForwardShape,
    batch_size: usize,
) -> Result<Option<DeviceBuffer<f32>>> {
    if sfnn_halfka2_ft_factorized_input_size(shape.input_size) && batch_size >= SFNN_HALFKA2_FOLD_L0W_MIN_BATCH {
        let len = SFNN_HALFKA2_BASE_INPUT_SIZE.saturating_mul(shape.ft_size);
        Ok(Some(DeviceBuffer::<f32>::zeroed(stream, len)?))
    } else {
        Ok(None)
    }
}

fn profile_stage(stream: &Arc<CudaStream>, last: &mut Option<std::time::Instant>, label: &'static str) -> Result<()> {
    if let Some(previous) = last {
        stream.synchronize()?;
        let now = std::time::Instant::now();
        println!("  profile_sfnn_step : {label:<20} {:>9.3} ms", now.duration_since(*previous).as_secs_f64() * 1000.0);
        *previous = now;
    }
    Ok(())
}
