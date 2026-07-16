//! Minimal NNUE forward kernels.
//!
//! These kernels are intentionally simple. They are the correctness baseline
//! for CO-006; later passes can fuse or tile them after CPU/GPU traces match.

use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn nnue_sparse_l0_crelu(
    indices: &[i32],
    weights: &[f32],
    bias: &[f32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    max_active: u32,
    input_size: u32,
    rows: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (rows as usize);
    if tid.get() >= total {
        return;
    }

    let row = tid.get() % (rows as usize);
    let sample = tid.get() / (rows as usize);
    let mut sum = bias[row];

    let sparse_base = sample * (max_active as usize);
    for slot in 0..(max_active as usize) {
        let feature = indices[sparse_base + slot];
        if feature >= 0 && (feature as u32) < input_size {
            let weight_base = (feature as usize) * (rows as usize);
            sum += weights[weight_base + row];
        }
    }

    let clipped = crelu(sum);
    if let Some(out) = output.get_mut(tid) {
        *out = clipped;
    }
}

#[kernel]
pub fn nnue_concat_l0(
    stm_l0: &[f32],
    nstm_l0: &[f32],
    mut combined: DisjointSlice<f32>,
    batch: u32,
    rows: u32,
) {
    let tid = thread::index_1d();
    let combined_rows = (rows as usize) * 2;
    let total = (batch as usize) * combined_rows;
    if tid.get() >= total {
        return;
    }

    let col = tid.get() % combined_rows;
    let sample = tid.get() / combined_rows;
    let src = sample * (rows as usize) + (col % (rows as usize));
    let value = if col < rows as usize {
        stm_l0[src]
    } else {
        nstm_l0[src]
    };

    if let Some(out) = combined.get_mut(tid) {
        *out = value;
    }
}

#[kernel]
pub fn nnue_dense_l1_crelu(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
) {
    dense_crelu(input, weights, bias, output, batch, input_dim, output_dim);
}

#[kernel]
pub fn nnue_dense_l2_crelu(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
) {
    dense_crelu(input, weights, bias, output, batch, input_dim, output_dim);
}

#[kernel]
pub fn nnue_dense_output(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
) {
    let tid = thread::index_1d();
    if tid.get() >= batch as usize {
        return;
    }

    let sample = tid.get();
    let input_base = sample * (input_dim as usize);
    let mut sum = bias[0];
    for idx in 0..(input_dim as usize) {
        sum += input[input_base + idx] * weights[idx];
    }

    if let Some(out) = output.get_mut(tid) {
        *out = sum;
    }
}

fn dense_crelu(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (output_dim as usize);
    if tid.get() >= total {
        return;
    }

    let out_col = tid.get() % (output_dim as usize);
    let sample = tid.get() / (output_dim as usize);
    let input_base = sample * (input_dim as usize);
    let mut sum = bias[out_col];

    for in_col in 0..(input_dim as usize) {
        let weight_base = in_col * (output_dim as usize);
        sum += input[input_base + in_col] * weights[weight_base + out_col];
    }

    if let Some(out) = output.get_mut(tid) {
        *out = crelu(sum);
    }
}

fn crelu(value: f32) -> f32 {
    if value < 0.0_f32 {
        0.0_f32
    } else if value > 1.0_f32 {
        1.0_f32
    } else {
        value
    }
}
