//! Minimal SFNN forward kernels.
//!
//! These are intentionally scalar and correctness-first. Once CPU/GPU traces
//! match, later passes can fuse/tile the expensive layers.

use cuda_device::{DisjointSlice, device, kernel, thread};

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
            let weight_base = (feature as usize) * (rows as usize);
            sum += weights[weight_base + row];
        }
    }

    if let Some(out) = output.get_mut(tid) {
        *out = sfnn_crelu(sum);
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
    let l0_base = sample * ft_size + pair * 2;
    let source = if col < pairwise { stm_l0 } else { nstm_l0 };
    let value = source[l0_base] * source[l0_base + 1] * (127.0_f32 / 128.0_f32);

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
    for idx in 0..(input_dim as usize) {
        sum += input[input_base + idx] * weights[idx * (num_stacks as usize) + stack];
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
    let stack_stride = (num_stacks as usize) * rows;
    let mut sum = bias[stack * rows + out_col];
    let input_base = sample * (input_dim as usize);
    for in_col in 0..(input_dim as usize) {
        sum += input[input_base + in_col] * weights[in_col * stack_stride + stack * rows + out_col];
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
