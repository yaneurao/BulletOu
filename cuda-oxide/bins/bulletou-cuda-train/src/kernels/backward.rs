//! Minimal backward kernels for CO-009.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, kernel, thread};

const SFNN_HALFKA2_BASE_INPUT_SIZE: u32 = 131_949;
const SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS: u32 = 1_629;
const SFNN_HALFKA2_FT_FACTORIZE_INPUT_SIZE: u32 = SFNN_HALFKA2_BASE_INPUT_SIZE + SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS;

#[kernel]
pub fn dense_output_backward(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    mut input_gradients: DisjointSlice<f32>,
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradient: DisjointSlice<f32>,
    batch: u32,
    input_len: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let rows = input_len as usize;
    let input_gradient_len = batch_size * rows;

    if tid_value < input_gradient_len {
        let sample = tid_value / rows;
        let row = tid_value - sample * rows;
        if let Some(out) = input_gradients.get_mut(tid) {
            *out = output_gradients[sample] * weights[row];
        }
    }

    if tid_value < rows {
        let mut sum = 0.0_f32;
        for sample in 0..batch_size {
            sum += output_gradients[sample] * inputs[sample * rows + tid_value];
        }
        if let Some(out) = weight_gradients.get_mut(thread::index_1d()) {
            *out = sum;
        }
    }

    if tid_value == 0 {
        let mut sum = 0.0_f32;
        for sample in 0..batch_size {
            sum += output_gradients[sample];
        }
        if let Some(out) = bias_gradient.get_mut(thread::index_1d()) {
            *out = sum;
        }
    }
}

#[kernel]
pub fn dense_crelu_backward(
    inputs: &[f32],
    activations: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    mut input_gradients: DisjointSlice<f32>,
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let input_rows = input_dim as usize;
    let output_rows = output_dim as usize;
    let input_gradient_len = batch_size * input_rows;
    let weight_len = input_rows * output_rows;

    if tid_value < input_gradient_len {
        let sample = tid_value / input_rows;
        let in_col = tid_value - sample * input_rows;
        let mut sum = 0.0_f32;
        for out_col in 0..output_rows {
            let grad = dense_crelu_pre_gradient(activations, output_gradients, sample, out_col, output_rows);
            sum += grad * weights[in_col * output_rows + out_col];
        }
        if let Some(out) = input_gradients.get_mut(tid) {
            *out = sum;
        }
    }

    if tid_value < weight_len {
        let in_col = tid_value / output_rows;
        let out_col = tid_value - in_col * output_rows;
        let mut sum = 0.0_f32;
        for sample in 0..batch_size {
            let grad = dense_crelu_pre_gradient(activations, output_gradients, sample, out_col, output_rows);
            sum += grad * inputs[sample * input_rows + in_col];
        }
        if let Some(out) = weight_gradients.get_mut(thread::index_1d()) {
            *out = sum;
        }
    }

    if tid_value < output_rows {
        let mut sum = 0.0_f32;
        for sample in 0..batch_size {
            sum += dense_crelu_pre_gradient(activations, output_gradients, sample, tid_value, output_rows);
        }
        if let Some(out) = bias_gradients.get_mut(thread::index_1d()) {
            *out = sum;
        }
    }
}

#[kernel]
pub fn dense_crelu_pre_gradients(
    activations: &[f32],
    output_gradients: &[f32],
    mut pre_gradients: DisjointSlice<f32>,
    batch: u32,
    output_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (output_dim as usize);
    if tid.get() >= total {
        return;
    }

    let value = crelu_pre_gradient_from_value(activations[tid.get()], output_gradients[tid.get()]);
    if let Some(out) = pre_gradients.get_mut(tid) {
        *out = value;
    }
}

#[kernel]
pub fn dense_pre_input_gradients(
    pre_gradients: &[f32],
    weights: &[f32],
    mut input_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let input_rows = input_dim as usize;
    let output_rows = output_dim as usize;
    let input_gradient_len = batch_size * input_rows;

    if tid_value >= input_gradient_len {
        return;
    }

    let sample = tid_value / input_rows;
    let in_col = tid_value - sample * input_rows;
    let mut sum = 0.0_f32;
    for out_col in 0..output_rows {
        sum += pre_gradients[sample * output_rows + out_col] * weights[in_col * output_rows + out_col];
    }
    if let Some(out) = input_gradients.get_mut(tid) {
        *out = sum;
    }
}

#[kernel]
pub fn dense_pre_weight_gradients(
    inputs: &[f32],
    pre_gradients: &[f32],
    mut weight_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let input_rows = input_dim as usize;
    let output_rows = output_dim as usize;
    let weight_len = input_rows * output_rows;

    if tid_value >= weight_len {
        return;
    }

    let in_col = tid_value / output_rows;
    let out_col = tid_value - in_col * output_rows;
    let mut sum = 0.0_f32;
    for sample in 0..batch_size {
        sum += pre_gradients[sample * output_rows + out_col] * inputs[sample * input_rows + in_col];
    }
    if let Some(out) = weight_gradients.get_mut(tid) {
        *out = sum;
    }
}

#[kernel]
pub fn dense_bias_gradients(
    pre_gradients: &[f32],
    mut bias_gradients: DisjointSlice<f32>,
    batch: u32,
    output_dim: u32,
) {
    let tid = thread::index_1d();
    let out_col = tid.get();
    let rows = output_dim as usize;
    if out_col >= rows {
        return;
    }

    let mut sum = 0.0_f32;
    for sample in 0..(batch as usize) {
        sum += pre_gradients[sample * rows + out_col];
    }
    if let Some(out) = bias_gradients.get_mut(tid) {
        *out = sum;
    }
}

#[kernel]
pub fn nnue_l0_crelu_backward(
    combined_gradients: &[f32],
    stm_activations: &[f32],
    nstm_activations: &[f32],
    mut stm_gradients: DisjointSlice<f32>,
    mut nstm_gradients: DisjointSlice<f32>,
    batch: u32,
    l1: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let rows = l1 as usize;
    let combined_stride = rows * 2;
    let combined_len = batch_size * combined_stride;

    if tid_value < combined_len {
        let sample = tid_value / combined_stride;
        let col = tid_value - sample * combined_stride;
        if col < rows {
            let perspective_idx = sample * rows + col;
            let grad = crelu_pre_gradient_from_value(stm_activations[perspective_idx], combined_gradients[tid_value]);
            // SAFETY: for col < rows, each 1D thread maps to exactly one
            // unique stm perspective index within the validated output slice.
            unsafe {
                *stm_gradients.get_unchecked_mut(perspective_idx) = grad;
            }
        } else {
            let row = col - rows;
            let perspective_idx = sample * rows + row;
            let grad = crelu_pre_gradient_from_value(nstm_activations[perspective_idx], combined_gradients[tid_value]);
            // SAFETY: for col >= rows, each 1D thread maps to exactly one
            // unique nstm perspective index within the validated output slice.
            unsafe {
                *nstm_gradients.get_unchecked_mut(perspective_idx) = grad;
            }
        }
    }
}

#[kernel]
pub fn nnue_l0_sparse_zero_gradients(
    mut l0w_gradients: DisjointSlice<f32>,
    mut l0b_gradients: DisjointSlice<f32>,
    input_size: u32,
    l1: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let rows = l1 as usize;
    let features = input_size as usize;
    let weight_len = features * rows;

    if tid_value < weight_len {
        if let Some(out) = l0w_gradients.get_mut(tid) {
            *out = 0.0;
        }
    }

    if tid_value < rows {
        if let Some(out) = l0b_gradients.get_mut(thread::index_1d()) {
            *out = 0.0;
        }
    }
}

#[kernel]
pub fn nnue_l0_sparse_backward(
    stm_indices: &[i32],
    nstm_indices: &[i32],
    stm_gradients: &[f32],
    nstm_gradients: &[f32],
    mut l0w_gradients: DisjointSlice<f32>,
    mut l0b_gradients: DisjointSlice<f32>,
    batch: u32,
    max_active: u32,
    input_size: u32,
    l1: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let slots = max_active as usize;
    let rows = l1 as usize;
    let features = input_size as usize;
    let scatter_len = batch_size * slots * rows;

    if tid_value < scatter_len {
        let row = tid_value % rows;
        let sparse_entry = tid_value / rows;
        let sample = sparse_entry / slots;
        let slot = sparse_entry - sample * slots;
        let sparse_base = sample * slots + slot;

        let stm_feature = stm_indices[sparse_base];
        if stm_feature >= 0 && (stm_feature as usize) < features {
            let weight_idx = (stm_feature as usize) * rows + row;
            let cell = unsafe { &*(l0w_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
            cell.fetch_add(stm_gradients[sample * rows + row], AtomicOrdering::Relaxed);
        }

        let nstm_feature = nstm_indices[sparse_base];
        if nstm_feature >= 0 && (nstm_feature as usize) < features {
            let weight_idx = (nstm_feature as usize) * rows + row;
            let cell = unsafe { &*(l0w_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
            cell.fetch_add(nstm_gradients[sample * rows + row], AtomicOrdering::Relaxed);
        }
    }

    if tid_value < batch_size * rows {
        let row = tid_value % rows;
        let sample = tid_value / rows;
        let cell = unsafe { &*(l0b_gradients.as_mut_ptr().add(row) as *const DeviceAtomicF32) };
        cell.fetch_add(
            stm_gradients[sample * rows + row] + nstm_gradients[sample * rows + row],
            AtomicOrdering::Relaxed,
        );
    }
}

#[kernel]
pub fn sfnn_dense_zero_gradients(
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradients: DisjointSlice<f32>,
    weight_len: u32,
    bias_len: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    if tid_value < weight_len as usize {
        if let Some(out) = weight_gradients.get_mut(tid) {
            *out = 0.0;
        }
    }
    if tid_value < bias_len as usize {
        if let Some(out) = bias_gradients.get_mut(thread::index_1d()) {
            *out = 0.0;
        }
    }
}

#[kernel]
pub fn sfnn_stacked_l3_backward(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    buckets: &[i32],
    mut input_gradients: DisjointSlice<f32>,
    mut l1_gradients: DisjointSlice<f32>,
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    l1_out: u32,
    num_stacks: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let rows = input_dim as usize;
    let l1_rows = l1_out as usize;
    let stacks = num_stacks as usize;
    let input_gradient_len = batch_size * rows;
    let l1_gradient_len = batch_size * l1_rows;

    if tid_value < input_gradient_len {
        let sample = tid_value / rows;
        let row = tid_value - sample * rows;
        let stack_i32 = buckets[sample];
        let value = if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            output_gradients[sample] * weights[(stack_i32 as usize) * rows + row]
        } else {
            0.0_f32
        };
        if let Some(out) = input_gradients.get_mut(tid) {
            *out = value;
        }

        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let weight_idx = stack * rows + row;
            let weight_cell = unsafe { &*(weight_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
            weight_cell.fetch_add(output_gradients[sample] * inputs[sample * rows + row], AtomicOrdering::Relaxed);
        }
    }

    if tid_value < l1_gradient_len {
        let sample = tid_value / l1_rows;
        let col = tid_value - sample * l1_rows;
        let value = if col + 1 == l1_rows { output_gradients[sample] } else { 0.0_f32 };
        if let Some(out) = l1_gradients.get_mut(thread::index_1d()) {
            *out = value;
        }
    }

    if tid_value < batch_size {
        let stack_i32 = buckets[tid_value];
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let cell = unsafe { &*(bias_gradients.as_mut_ptr().add(stack_i32 as usize) as *const DeviceAtomicF32) };
            cell.fetch_add(output_gradients[tid_value], AtomicOrdering::Relaxed);
        }
    }
}

#[kernel]
pub fn sfnn_stacked_crelu_backward(
    inputs: &[f32],
    activations: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    buckets: &[i32],
    mut input_gradients: DisjointSlice<f32>,
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
    num_stacks: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let input_rows = input_dim as usize;
    let output_rows = output_dim as usize;
    let stacks = num_stacks as usize;
    let input_gradient_len = batch_size * input_rows;
    let weight_scatter_len = batch_size * input_rows * output_rows;
    let bias_scatter_len = batch_size * output_rows;

    if tid_value < input_gradient_len {
        let sample = tid_value / input_rows;
        let in_col = tid_value - sample * input_rows;
        let stack_i32 = buckets[sample];
        let mut sum = 0.0_f32;
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let stack_base = stack * output_rows * input_rows;
            for out_col in 0..output_rows {
                let grad = dense_crelu_pre_gradient(activations, output_gradients, sample, out_col, output_rows);
                sum += grad * weights[stack_base + out_col * input_rows + in_col];
            }
        }
        if let Some(out) = input_gradients.get_mut(tid) {
            *out = sum;
        }
    }

    if tid_value < weight_scatter_len {
        let out_col = tid_value % output_rows;
        let input_entry = tid_value / output_rows;
        let in_col = input_entry % input_rows;
        let sample = input_entry / input_rows;
        let stack_i32 = buckets[sample];
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let grad = dense_crelu_pre_gradient(activations, output_gradients, sample, out_col, output_rows);
            let weight_idx = stack * output_rows * input_rows + out_col * input_rows + in_col;
            let cell = unsafe { &*(weight_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
            cell.fetch_add(grad * inputs[sample * input_rows + in_col], AtomicOrdering::Relaxed);
        }
    }

    if tid_value < bias_scatter_len {
        let out_col = tid_value % output_rows;
        let sample = tid_value / output_rows;
        let stack_i32 = buckets[sample];
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let bias_idx = stack * output_rows + out_col;
            let grad = dense_crelu_pre_gradient(activations, output_gradients, sample, out_col, output_rows);
            let cell = unsafe { &*(bias_gradients.as_mut_ptr().add(bias_idx) as *const DeviceAtomicF32) };
            cell.fetch_add(grad, AtomicOrdering::Relaxed);
        }
    }
}

#[kernel]
pub fn sfnn_l2_input_backward(
    l1: &[f32],
    l2_input: &[f32],
    l2_input_gradients: &[f32],
    mut l1_gradients: DisjointSlice<f32>,
    batch: u32,
    l1_hidden: u32,
) {
    let tid = thread::index_1d();
    let hidden = l1_hidden as usize;
    let l1_out = hidden + 1;
    let total = (batch as usize) * l1_out;
    if tid.get() >= total {
        return;
    }

    let sample = tid.get() / l1_out;
    let col = tid.get() - sample * l1_out;
    if col >= hidden {
        return;
    }

    let l2_input_dim = hidden * 2;
    let l2_base = sample * l2_input_dim;
    let value = l1[tid.get()];
    let square_idx = l2_base + col;
    let linear_idx = l2_base + hidden + col;
    let square_grad = crelu_pre_gradient_from_value(l2_input[square_idx], l2_input_gradients[square_idx])
        * (2.0_f32 * value * (127.0_f32 / 128.0_f32));
    let linear_grad = crelu_pre_gradient_from_value(l2_input[linear_idx], l2_input_gradients[linear_idx]);

    if let Some(out) = l1_gradients.get_mut(tid) {
        *out += square_grad + linear_grad;
    }
}

#[kernel]
pub fn sfnn_stacked_affine_backward(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    buckets: &[i32],
    mut input_gradients: DisjointSlice<f32>,
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
    num_stacks: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let input_rows = input_dim as usize;
    let output_rows = output_dim as usize;
    let stacks = num_stacks as usize;
    let input_gradient_len = batch_size * input_rows;
    let weight_scatter_len = batch_size * input_rows * output_rows;
    let bias_scatter_len = batch_size * output_rows;

    if tid_value < input_gradient_len {
        let sample = tid_value / input_rows;
        let in_col = tid_value - sample * input_rows;
        let stack_i32 = buckets[sample];
        let mut sum = 0.0_f32;
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let stack_base = stack * output_rows * input_rows;
            for out_col in 0..output_rows {
                sum += output_gradients[sample * output_rows + out_col]
                    * weights[stack_base + out_col * input_rows + in_col];
            }
        }
        if let Some(out) = input_gradients.get_mut(tid) {
            *out = sum;
        }
    }

    if tid_value < weight_scatter_len {
        let out_col = tid_value % output_rows;
        let input_entry = tid_value / output_rows;
        let in_col = input_entry % input_rows;
        let sample = input_entry / input_rows;
        let stack_i32 = buckets[sample];
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let weight_idx = stack * output_rows * input_rows + out_col * input_rows + in_col;
            let grad = output_gradients[sample * output_rows + out_col] * inputs[sample * input_rows + in_col];
            let cell = unsafe { &*(weight_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
            cell.fetch_add(grad, AtomicOrdering::Relaxed);
        }
    }

    if tid_value < bias_scatter_len {
        let out_col = tid_value % output_rows;
        let sample = tid_value / output_rows;
        let stack_i32 = buckets[sample];
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let bias_idx = stack * output_rows + out_col;
            let cell = unsafe { &*(bias_gradients.as_mut_ptr().add(bias_idx) as *const DeviceAtomicF32) };
            cell.fetch_add(output_gradients[sample * output_rows + out_col], AtomicOrdering::Relaxed);
        }
    }
}

#[kernel]
pub fn sfnn_shared_l1_backward(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    mut input_gradients: DisjointSlice<f32>,
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let input_rows = input_dim as usize;
    let output_rows = output_dim as usize;
    let input_gradient_len = batch_size * input_rows;
    let weight_scatter_len = batch_size * input_rows * output_rows;
    let bias_scatter_len = batch_size * output_rows;

    if tid_value < input_gradient_len {
        let sample = tid_value / input_rows;
        let in_col = tid_value - sample * input_rows;
        let mut sum = 0.0_f32;
        for out_col in 0..output_rows {
            sum += output_gradients[sample * output_rows + out_col] * weights[in_col * output_rows + out_col];
        }
        if let Some(out) = input_gradients.get_mut(tid) {
            *out += sum;
        }
    }

    if tid_value < weight_scatter_len {
        let out_col = tid_value % output_rows;
        let input_entry = tid_value / output_rows;
        let in_col = input_entry % input_rows;
        let sample = input_entry / input_rows;
        let weight_idx = in_col * output_rows + out_col;
        let grad = output_gradients[sample * output_rows + out_col] * inputs[sample * input_rows + in_col];
        let cell = unsafe { &*(weight_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
        cell.fetch_add(grad, AtomicOrdering::Relaxed);
    }

    if tid_value < bias_scatter_len {
        let out_col = tid_value % output_rows;
        let sample = tid_value / output_rows;
        let cell = unsafe { &*(bias_gradients.as_mut_ptr().add(out_col) as *const DeviceAtomicF32) };
        cell.fetch_add(output_gradients[sample * output_rows + out_col], AtomicOrdering::Relaxed);
    }
}

#[kernel]
pub fn sfnn_factorized_l1_backward(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    shared_weights: &[f32],
    buckets: &[i32],
    mut input_gradients: DisjointSlice<f32>,
    mut weight_gradients: DisjointSlice<f32>,
    mut bias_gradients: DisjointSlice<f32>,
    mut shared_weight_gradients: DisjointSlice<f32>,
    mut shared_bias_gradients: DisjointSlice<f32>,
    batch: u32,
    input_dim: u32,
    output_dim: u32,
    num_stacks: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let input_rows = input_dim as usize;
    let output_rows = output_dim as usize;
    let stacks = num_stacks as usize;
    let input_gradient_len = batch_size * input_rows;
    let weight_scatter_len = batch_size * input_rows * output_rows;
    let bias_scatter_len = batch_size * output_rows;

    if tid_value < input_gradient_len {
        let sample = tid_value / input_rows;
        let in_col = tid_value - sample * input_rows;
        let stack_i32 = buckets[sample];
        let mut sum = 0.0_f32;
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let stack_base = stack * output_rows * input_rows;
            for out_col in 0..output_rows {
                let grad = output_gradients[sample * output_rows + out_col];
                let stacked_weight = weights[stack_base + out_col * input_rows + in_col];
                let shared_weight = shared_weights[in_col * output_rows + out_col];
                sum += grad * (stacked_weight + shared_weight);
            }
        }
        if let Some(out) = input_gradients.get_mut(tid) {
            *out = sum;
        }
    }

    if tid_value < weight_scatter_len {
        let out_col = tid_value % output_rows;
        let input_entry = tid_value / output_rows;
        let in_col = input_entry % input_rows;
        let sample = input_entry / input_rows;
        let stack_i32 = buckets[sample];
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let stack = stack_i32 as usize;
            let grad = output_gradients[sample * output_rows + out_col] * inputs[sample * input_rows + in_col];

            let weight_idx = stack * output_rows * input_rows + out_col * input_rows + in_col;
            let weight_cell = unsafe { &*(weight_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
            weight_cell.fetch_add(grad, AtomicOrdering::Relaxed);

            let shared_weight_idx = in_col * output_rows + out_col;
            let shared_weight_cell =
                unsafe { &*(shared_weight_gradients.as_mut_ptr().add(shared_weight_idx) as *const DeviceAtomicF32) };
            shared_weight_cell.fetch_add(grad, AtomicOrdering::Relaxed);
        }
    }

    if tid_value < bias_scatter_len {
        let out_col = tid_value % output_rows;
        let sample = tid_value / output_rows;
        let stack_i32 = buckets[sample];
        if stack_i32 >= 0 && (stack_i32 as usize) < stacks {
            let grad = output_gradients[sample * output_rows + out_col];

            let stack = stack_i32 as usize;
            let bias_idx = stack * output_rows + out_col;
            let bias_cell = unsafe { &*(bias_gradients.as_mut_ptr().add(bias_idx) as *const DeviceAtomicF32) };
            bias_cell.fetch_add(grad, AtomicOrdering::Relaxed);

            let shared_bias_cell =
                unsafe { &*(shared_bias_gradients.as_mut_ptr().add(out_col) as *const DeviceAtomicF32) };
            shared_bias_cell.fetch_add(grad, AtomicOrdering::Relaxed);
        }
    }
}

#[kernel]
pub fn sfnn_pairwise_backward(
    stm_l0: &[f32],
    nstm_l0: &[f32],
    combined_gradients: &[f32],
    mut stm_gradients: DisjointSlice<f32>,
    mut nstm_gradients: DisjointSlice<f32>,
    batch: u32,
    ft_size: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let total = (batch as usize) * (ft_size as usize);
    if tid_value >= total {
        return;
    }

    let ft = ft_size as usize;
    let pairwise = ft / 2;
    let sample = tid_value / ft;
    let col = tid_value - sample * ft;
    let pair = col % pairwise;
    let mate_col = if col < pairwise { pairwise + pair } else { pair };
    let l0_base = sample * ft;
    let combined_base = sample * ft;
    let scale = 127.0_f32 / 128.0_f32;
    let stm_grad = combined_gradients[combined_base + pair] * stm_l0[l0_base + mate_col] * scale;
    let nstm_grad = combined_gradients[combined_base + pairwise + pair] * nstm_l0[l0_base + mate_col] * scale;

    if let Some(out) = stm_gradients.get_mut(tid) {
        *out = stm_grad;
    }
    if let Some(out) = nstm_gradients.get_mut(thread::index_1d()) {
        *out = nstm_grad;
    }
}

#[kernel]
pub fn sfnn_pairwise_l0_sparse_backward_train(
    stm_indices: &[i32],
    nstm_indices: &[i32],
    stm_activations: &[f32],
    nstm_activations: &[f32],
    combined_gradients: &[f32],
    mut l0w_gradients: DisjointSlice<f32>,
    mut l0b_gradients: DisjointSlice<f32>,
    batch: u32,
    max_active: u32,
    input_size: u32,
    ft_size: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let slots = max_active as usize;
    let rows = ft_size as usize;
    let features = input_size as usize;
    let l0_len = batch_size * rows;

    if tid_value >= l0_len {
        return;
    }

    let row = tid_value % rows;
    let sample = tid_value / rows;
    let sparse_base = sample * slots;
    let pairwise = rows / 2;
    let pair = row % pairwise;
    let mate_col = if row < pairwise { pairwise + pair } else { pair };
    let l0_base = sample * rows;
    let combined_base = sample * rows;
    let scale = 127.0_f32 / 128.0_f32;
    let stm_output_grad = combined_gradients[combined_base + pair] * stm_activations[l0_base + mate_col] * scale;
    let nstm_output_grad =
        combined_gradients[combined_base + pairwise + pair] * nstm_activations[l0_base + mate_col] * scale;
    let stm_grad = crelu_pre_gradient_from_value(stm_activations[tid_value], stm_output_grad);
    let nstm_grad = crelu_pre_gradient_from_value(nstm_activations[tid_value], nstm_output_grad);
    let has_stm_grad = stm_grad != 0.0_f32;
    let has_nstm_grad = nstm_grad != 0.0_f32;

    if has_stm_grad || has_nstm_grad {
        let bias_cell = unsafe { &*(l0b_gradients.as_mut_ptr().add(row) as *const DeviceAtomicF32) };
        bias_cell.fetch_add(stm_grad + nstm_grad, AtomicOrdering::Relaxed);
    } else {
        return;
    }

    for slot in 0..slots {
        if has_stm_grad {
            let stm_feature = stm_indices[sparse_base + slot];
            if stm_feature >= 0 && (stm_feature as usize) < features {
                let weight_idx = (stm_feature as usize) * rows + row;
                let cell = unsafe { &*(l0w_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
                cell.fetch_add(stm_grad, AtomicOrdering::Relaxed);
            }
        }

        if has_nstm_grad {
            let nstm_feature = nstm_indices[sparse_base + slot];
            if nstm_feature >= 0 && (nstm_feature as usize) < features {
                let weight_idx = (nstm_feature as usize) * rows + row;
                let cell = unsafe { &*(l0w_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
                cell.fetch_add(nstm_grad, AtomicOrdering::Relaxed);
            }
        }
    }
}

#[kernel]
pub fn sfnn_l0_sparse_zero_gradients(
    mut l0w_gradients: DisjointSlice<f32>,
    mut l0b_gradients: DisjointSlice<f32>,
    input_size: u32,
    ft_size: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let rows = ft_size as usize;
    let features = input_size as usize;
    let weight_len = features * rows;

    if tid_value < weight_len {
        if let Some(out) = l0w_gradients.get_mut(tid) {
            *out = 0.0;
        }
    }

    if tid_value < rows {
        if let Some(out) = l0b_gradients.get_mut(thread::index_1d()) {
            *out = 0.0;
        }
    }
}

#[kernel]
pub fn sfnn_l0_sparse_backward(
    stm_indices: &[i32],
    nstm_indices: &[i32],
    stm_activations: &[f32],
    nstm_activations: &[f32],
    stm_output_gradients: &[f32],
    nstm_output_gradients: &[f32],
    mut stm_pre_gradients: DisjointSlice<f32>,
    mut nstm_pre_gradients: DisjointSlice<f32>,
    mut l0w_gradients: DisjointSlice<f32>,
    mut l0b_gradients: DisjointSlice<f32>,
    batch: u32,
    max_active: u32,
    input_size: u32,
    ft_size: u32,
) {
    let tid = thread::index_1d();
    let tid_value = tid.get();
    let batch_size = batch as usize;
    let slots = max_active as usize;
    let rows = ft_size as usize;
    let features = input_size as usize;
    let l0_len = batch_size * rows;

    if tid_value < l0_len {
        let row = tid_value % rows;
        let sample = tid_value / rows;
        let sparse_base = sample * slots;
        let stm_grad = crelu_pre_gradient_from_value(stm_activations[tid_value], stm_output_gradients[tid_value]);
        let nstm_grad = crelu_pre_gradient_from_value(nstm_activations[tid_value], nstm_output_gradients[tid_value]);
        if let Some(out) = stm_pre_gradients.get_mut(tid) {
            *out = stm_grad;
        }
        if let Some(out) = nstm_pre_gradients.get_mut(thread::index_1d()) {
            *out = nstm_grad;
        }

        let cell = unsafe { &*(l0b_gradients.as_mut_ptr().add(row) as *const DeviceAtomicF32) };
        cell.fetch_add(stm_grad + nstm_grad, AtomicOrdering::Relaxed);

        for slot in 0..slots {
            let stm_feature = stm_indices[sparse_base + slot];
            if stm_feature >= 0 && (stm_feature as usize) < features {
                let stm_feature = stm_feature as u32;
                let weight_idx = (stm_feature as usize) * rows + row;
                let cell = unsafe { &*(l0w_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
                cell.fetch_add(stm_grad, AtomicOrdering::Relaxed);
            }

            let nstm_feature = nstm_indices[sparse_base + slot];
            if nstm_feature >= 0 && (nstm_feature as usize) < features {
                let nstm_feature = nstm_feature as u32;
                let weight_idx = (nstm_feature as usize) * rows + row;
                let cell = unsafe { &*(l0w_gradients.as_mut_ptr().add(weight_idx) as *const DeviceAtomicF32) };
                cell.fetch_add(nstm_grad, AtomicOrdering::Relaxed);
            }
        }
    }
}

#[kernel]
pub fn sfnn_halfka2_ft_factorized_l0_reduce_virtual_grad(
    mut l0w_gradients: DisjointSlice<f32>,
    input_size: u32,
    ft_size: u32,
) {
    if input_size != SFNN_HALFKA2_FT_FACTORIZE_INPUT_SIZE {
        return;
    }

    let tid = thread::index_1d();
    let tid_value = tid.get();
    let rows = ft_size as usize;
    let piece_inputs = SFNN_HALFKA2_FT_FACTORIZE_PIECE_INPUTS as usize;
    let base_input_size = SFNN_HALFKA2_BASE_INPUT_SIZE as usize;
    let total = piece_inputs * rows;
    if tid_value >= total {
        return;
    }

    let piece = tid_value / rows;
    let row = tid_value - piece * rows;
    let row_stride = piece_inputs * rows;
    let base = piece * rows + row;
    let repeats = base_input_size / piece_inputs;
    let ptr = l0w_gradients.as_mut_ptr();
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut kb = 0_usize;
    let unroll_end = repeats.saturating_sub(3);
    while kb < unroll_end {
        unsafe {
            sum0 += ptr.add(base + kb * row_stride).read();
            sum1 += ptr.add(base + (kb + 1) * row_stride).read();
            sum2 += ptr.add(base + (kb + 2) * row_stride).read();
            sum3 += ptr.add(base + (kb + 3) * row_stride).read();
        }
        kb += 4;
    }
    while kb < repeats {
        unsafe {
            sum0 += ptr.add(base + kb * row_stride).read();
        }
        kb += 1;
    }
    let virtual_idx = base_input_size * rows + piece * rows + row;
    unsafe {
        ptr.add(virtual_idx).write((sum0 + sum1) + (sum2 + sum3));
    }
}

#[cuda_device::device]
fn dense_crelu_pre_gradient(
    activations: &[f32],
    output_gradients: &[f32],
    sample: usize,
    out_col: usize,
    output_rows: usize,
) -> f32 {
    let idx = sample * output_rows + out_col;
    crelu_pre_gradient_from_value(activations[idx], output_gradients[idx])
}

#[cuda_device::device]
fn crelu_pre_gradient_from_value(activation: f32, output_gradient: f32) -> f32 {
    if activation > 0.0_f32 && activation < 1.0_f32 { output_gradient } else { 0.0_f32 }
}
