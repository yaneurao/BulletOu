//! Minimal backward kernels for CO-009.

use cuda_device::{kernel, thread, DisjointSlice};

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

#[cuda_device::device]
fn dense_crelu_pre_gradient(
    activations: &[f32],
    output_gradients: &[f32],
    sample: usize,
    out_col: usize,
    output_rows: usize,
) -> f32 {
    let idx = sample * output_rows + out_col;
    let activation = activations[idx];
    if activation > 0.0_f32 && activation < 1.0_f32 {
        output_gradients[idx]
    } else {
        0.0_f32
    }
}
