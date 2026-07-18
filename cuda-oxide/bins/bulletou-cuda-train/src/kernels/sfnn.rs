//! Minimal SFNN forward kernels.
//!
//! These are intentionally scalar and correctness-first. Once CPU/GPU traces
//! match, later passes can fuse/tile the expensive layers.

use cuda_device::{DisjointSlice, device, kernel, thread};

const SFNN_HALFKA2_BASE_INPUT_SIZE: u32 = 131_949;
const SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS: u32 = 1_629;
const SFNN_HALFKA2_FT_FACTORIZE_INPUT_SIZE: u32 = SFNN_HALFKA2_BASE_INPUT_SIZE + SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS;

#[kernel]
pub fn sfnn_halfka2_fold_factorized_l0w(
    train_weights: &[f32],
    mut forward_weights: DisjointSlice<f32>,
    ft_size: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let rows = ft_size as usize;
    let base_features = SFNN_HALFKA2_BASE_INPUT_SIZE as usize;
    let piece_inputs = SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS as usize;
    let total = base_features * rows;
    if tid_value >= total {
        return;
    }

    let feature = tid_value / rows;
    let row = tid_value - feature * rows;
    let virtual_feature = base_features + feature % piece_inputs;
    let virtual_idx = virtual_feature * rows + row;
    let value = train_weights[tid_value] + train_weights[virtual_idx];
    if let Some(out) = forward_weights.get_mut(tid) {
        *out = value;
    }
}

#[kernel]
pub fn sfnn_sparse_l0_crelu(
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
            let feature = feature as u32;
            let weight_base = (feature as usize) * (rows as usize);
            sum += weights[weight_base + row];
            if let Some(virtual_feature) = sfnn_forward_halfka2_ft_factorized_virtual_feature(feature, input_size) {
                let virtual_weight_base = (virtual_feature as usize) * (rows as usize);
                sum += weights[virtual_weight_base + row];
            }
        }
    }

    if let Some(out) = output.get_mut(tid) {
        *out = sfnn_crelu(sum);
    }
}

#[kernel]
pub fn sfnn_sparse_l0_pairwise_concat(
    stm_indices: &[i32],
    nstm_indices: &[i32],
    weights: &[f32],
    bias: &[f32],
    mut stm_output: DisjointSlice<f32>,
    mut nstm_output: DisjointSlice<f32>,
    mut combined: DisjointSlice<f32>,
    batch: u32,
    max_active: u32,
    input_size: u32,
    rows: u32,
) {
    let tid = thread::index_1d();
    let row_pairs = (rows as usize) / 2;
    let total = (batch as usize) * row_pairs;
    if tid.get() >= total {
        return;
    }

    let pair = tid.get() % row_pairs;
    let sample = tid.get() / row_pairs;
    let rows = rows as usize;
    let row0 = pair;
    let row1 = row_pairs + pair;
    let l0_base = sample * rows;
    let sparse_base = sample * (max_active as usize);
    let mut stm_sum0 = bias[row0];
    let mut stm_sum1 = bias[row1];
    let mut nstm_sum0 = bias[row0];
    let mut nstm_sum1 = bias[row1];

    for slot in 0..(max_active as usize) {
        let stm_feature = stm_indices[sparse_base + slot];
        if stm_feature >= 0 && (stm_feature as u32) < input_size {
            let feature = stm_feature as u32;
            let weight_base = (feature as usize) * rows;
            stm_sum0 += weights[weight_base + row0];
            stm_sum1 += weights[weight_base + row1];
            if let Some(virtual_feature) = sfnn_forward_halfka2_ft_factorized_virtual_feature(feature, input_size) {
                let virtual_weight_base = (virtual_feature as usize) * rows;
                stm_sum0 += weights[virtual_weight_base + row0];
                stm_sum1 += weights[virtual_weight_base + row1];
            }
        }

        let nstm_feature = nstm_indices[sparse_base + slot];
        if nstm_feature >= 0 && (nstm_feature as u32) < input_size {
            let feature = nstm_feature as u32;
            let weight_base = (feature as usize) * rows;
            nstm_sum0 += weights[weight_base + row0];
            nstm_sum1 += weights[weight_base + row1];
            if let Some(virtual_feature) = sfnn_forward_halfka2_ft_factorized_virtual_feature(feature, input_size) {
                let virtual_weight_base = (virtual_feature as usize) * rows;
                nstm_sum0 += weights[virtual_weight_base + row0];
                nstm_sum1 += weights[virtual_weight_base + row1];
            }
        }
    }

    let stm0 = sfnn_crelu(stm_sum0);
    let stm1 = sfnn_crelu(stm_sum1);
    let nstm0 = sfnn_crelu(nstm_sum0);
    let nstm1 = sfnn_crelu(nstm_sum1);
    let idx0 = l0_base + row0;
    let idx1 = l0_base + row1;
    let combined_base = sample * rows;
    unsafe {
        *stm_output.get_unchecked_mut(idx0) = stm0;
        *stm_output.get_unchecked_mut(idx1) = stm1;
        *nstm_output.get_unchecked_mut(idx0) = nstm0;
        *nstm_output.get_unchecked_mut(idx1) = nstm1;
        *combined.get_unchecked_mut(combined_base + pair) = stm0 * stm1 * (127.0_f32 / 128.0_f32);
        *combined.get_unchecked_mut(combined_base + row_pairs + pair) = nstm0 * nstm1 * (127.0_f32 / 128.0_f32);
    }
}

#[device]
fn sfnn_forward_halfka2_ft_factorized_virtual_feature(feature: u32, input_size: u32) -> Option<u32> {
    if input_size == SFNN_HALFKA2_FT_FACTORIZE_INPUT_SIZE && feature < SFNN_HALFKA2_BASE_INPUT_SIZE {
        Some(SFNN_HALFKA2_BASE_INPUT_SIZE + feature % SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS)
    } else {
        None
    }
}

#[kernel]
pub fn sfnn_pairwise_concat(
    stm_l0: &[f32],
    nstm_l0: &[f32],
    mut combined: DisjointSlice<f32>,
    batch: u32,
    ft_size: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_size as usize);
    if tid.get() >= total {
        return;
    }

    let ft_size = ft_size as usize;
    let pairwise = ft_size / 2;
    let col = tid.get() % ft_size;
    let sample = tid.get() / ft_size;
    let pair = col % pairwise;
    let l0_base = sample * ft_size;
    let source = if col < pairwise { stm_l0 } else { nstm_l0 };
    let value = source[l0_base + pair] * source[l0_base + pairwise + pair] * (127.0_f32 / 128.0_f32);

    if let Some(out) = combined.get_mut(tid) {
        *out = value;
    }
}

#[kernel]
pub fn sfnn_stacked_l1(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    buckets: &[i32],
    output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
    num_stacks: u32,
) {
    stacked_affine(input, weights, bias, buckets, output, batch, input_dim, output_dim, num_stacks, false);
}

#[kernel]
pub fn sfnn_stacked_l1_factorized(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    shared_weights: &[f32],
    shared_bias: &[f32],
    buckets: &[i32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
    num_stacks: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (output_dim as usize);
    if tid.get() >= total {
        return;
    }

    let out_col = tid.get() % (output_dim as usize);
    let sample = tid.get() / (output_dim as usize);
    let stack_i32 = buckets[sample];
    if stack_i32 < 0 || (stack_i32 as u32) >= num_stacks {
        return;
    }

    let stack = stack_i32 as usize;
    let rows = output_dim as usize;
    let input_cols = input_dim as usize;
    let stack_base = stack * rows * input_cols;
    let mut sum = bias[stack * rows + out_col] + shared_bias[out_col];
    let input_base = sample * input_cols;
    for in_col in 0..input_cols {
        let input_value = input[input_base + in_col];
        let stacked_weight = weights[stack_base + out_col * input_cols + in_col];
        let shared_weight = shared_weights[in_col * rows + out_col];
        sum += input_value * (stacked_weight + shared_weight);
    }

    if let Some(out) = output.get_mut(tid) {
        *out = sum;
    }
}

#[kernel]
pub fn sfnn_shared_l1_add(
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
        sum += input[input_base + in_col] * weights[in_col * (output_dim as usize) + out_col];
    }

    if let Some(out) = output.get_mut(tid) {
        *out += sum;
    }
}

#[kernel]
pub fn sfnn_l2_input(l1: &[f32], mut output: DisjointSlice<f32>, batch: u32, l1_hidden: u32) {
    let tid = thread::index_1d();
    let l2_input_dim = (l1_hidden as usize) * 2;
    let total = (batch as usize) * l2_input_dim;
    if tid.get() >= total {
        return;
    }

    let col = tid.get() % l2_input_dim;
    let sample = tid.get() / l2_input_dim;
    let hidden = l1_hidden as usize;
    let source_col = col % hidden;
    let l1_out = hidden + 1;
    let value = l1[sample * l1_out + source_col];
    let transformed = if col < hidden {
        let abs_value = if value < 0.0_f32 { -value } else { value };
        sfnn_crelu(abs_value * abs_value * (127.0_f32 / 128.0_f32))
    } else {
        sfnn_crelu(value)
    };

    if let Some(out) = output.get_mut(tid) {
        *out = transformed;
    }
}

#[kernel]
pub fn sfnn_stacked_l2_crelu(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    buckets: &[i32],
    output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
    num_stacks: u32,
) {
    stacked_affine(input, weights, bias, buckets, output, batch, input_dim, output_dim, num_stacks, true);
}

#[kernel]
pub fn sfnn_stacked_l3_output(
    input: &[f32],
    l1: &[f32],
    weights: &[f32],
    bias: &[f32],
    buckets: &[i32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    l1_hidden: u32,
    num_stacks: u32,
) {
    let tid = thread::index_1d();
    if tid.get() >= batch as usize {
        return;
    }

    let sample = tid.get();
    let stack_i32 = buckets[sample];
    if stack_i32 < 0 || (stack_i32 as u32) >= num_stacks {
        return;
    }

    let stack = stack_i32 as usize;
    let input_base = sample * (input_dim as usize);
    let mut sum = bias[stack];
    let weight_base = stack * (input_dim as usize);
    for idx in 0..(input_dim as usize) {
        sum += input[input_base + idx] * weights[weight_base + idx];
    }

    let skip = l1[sample * ((l1_hidden as usize) + 1) + (l1_hidden as usize)];
    if let Some(out) = output.get_mut(tid) {
        *out = sum + skip;
    }
}

#[device]
fn stacked_affine(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    buckets: &[i32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
    num_stacks: u32,
    apply_crelu: bool,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (output_dim as usize);
    if tid.get() >= total {
        return;
    }

    let out_col = tid.get() % (output_dim as usize);
    let sample = tid.get() / (output_dim as usize);
    let stack_i32 = buckets[sample];
    if stack_i32 < 0 || (stack_i32 as u32) >= num_stacks {
        return;
    }

    let stack = stack_i32 as usize;
    let rows = output_dim as usize;
    let input_cols = input_dim as usize;
    let stack_base = stack * rows * input_cols;
    let mut sum = bias[stack * rows + out_col];
    let input_base = sample * input_cols;
    for in_col in 0..input_cols {
        sum += input[input_base + in_col] * weights[stack_base + out_col * input_cols + in_col];
    }

    if apply_crelu {
        sum = sfnn_crelu(sum);
    }

    if let Some(out) = output.get_mut(tid) {
        *out = sum;
    }
}

#[device]
fn sfnn_crelu(value: f32) -> f32 {
    if value < 0.0_f32 {
        0.0_f32
    } else if value > 1.0_f32 {
        1.0_f32
    } else {
        value
    }
}
