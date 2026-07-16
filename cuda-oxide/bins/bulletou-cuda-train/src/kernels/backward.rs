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
