//! Minimal scalar value-loss kernels.
//!
//! This is the correctness baseline for CO-008. It launches one thread per
//! sample for debug readback and lets thread 0 compute the scalar reduction.
//! Later passes can replace that reduction with a parallel implementation once
//! the formula is validated.

use cuda_device::{device, kernel, thread, DisjointSlice};

#[kernel]
pub fn loss_sigmoid_mse_reduce(
    outputs: &[f32],
    targets: &[f32],
    entry_weights: &[f32],
    mut per_sample: DisjointSlice<f32>,
    mut weighted_sum: DisjointSlice<f32>,
    mut mean: DisjointSlice<f32>,
    batch: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    if tid_value < batch as usize {
        let weighted = sigmoid_mse_weighted(outputs[tid_value], targets[tid_value], entry_weights[tid_value]);
        if let Some(out) = per_sample.get_mut(tid) {
            *out = weighted;
        }
    }

    if tid_value == 0 {
        let mut sum = 0.0_f32;
        for idx in 0..(batch as usize) {
            sum += sigmoid_mse_weighted(outputs[idx], targets[idx], entry_weights[idx]);
        }

        if let Some(out) = weighted_sum.get_mut(thread::index_1d()) {
            *out = sum;
        }
        if let Some(out) = mean.get_mut(thread::index_1d()) {
            *out = sum / (batch as f32);
        }
    }
}

#[kernel]
pub fn loss_nnue_pytorch_wrm_reduce(
    outputs: &[f32],
    targets: &[f32],
    entry_weights: &[f32],
    mut per_sample: DisjointSlice<f32>,
    mut weighted_sum: DisjointSlice<f32>,
    mut mean: DisjointSlice<f32>,
    batch: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    if tid_value < batch as usize {
        let weighted = nnue_pytorch_wrm_weighted(outputs[tid_value], targets[tid_value], entry_weights[tid_value]);
        if let Some(out) = per_sample.get_mut(tid) {
            *out = weighted;
        }
    }

    if tid_value == 0 {
        let mut sum = 0.0_f32;
        for idx in 0..(batch as usize) {
            sum += nnue_pytorch_wrm_weighted(outputs[idx], targets[idx], entry_weights[idx]);
        }

        if let Some(out) = weighted_sum.get_mut(thread::index_1d()) {
            *out = sum;
        }
        if let Some(out) = mean.get_mut(thread::index_1d()) {
            *out = sum / (batch as f32);
        }
    }
}

#[device]
fn sigmoid_mse_weighted(output: f32, target: f32, entry_weight: f32) -> f32 {
    let prediction = loss_sigmoid(output);
    let error = prediction - target;
    entry_weight * error * error
}

#[device]
fn nnue_pytorch_wrm_weighted(output: f32, target: f32, entry_weight: f32) -> f32 {
    const NNUE2SCORE: f32 = 600.0;
    const IN_OFFSET: f32 = 270.0;
    const IN_SCALING: f32 = 340.0;
    const POW_EXP: f32 = 2.5;

    let scorenet = output * NNUE2SCORE;
    let q = loss_sigmoid((scorenet - IN_OFFSET) / IN_SCALING);
    let qm = loss_sigmoid((-scorenet - IN_OFFSET) / IN_SCALING);
    let prediction = (1.0_f32 + q - qm) * 0.5_f32;
    let error = abs_f32(prediction - target);
    entry_weight * core::intrinsics::powf32(error, POW_EXP)
}

#[device]
fn loss_sigmoid(value: f32) -> f32 {
    let exp_neg = core::intrinsics::expf32(-value);
    1.0_f32 / (1.0_f32 + exp_neg)
}

#[device]
fn abs_f32(value: f32) -> f32 {
    if value < 0.0_f32 {
        -value
    } else {
        value
    }
}
