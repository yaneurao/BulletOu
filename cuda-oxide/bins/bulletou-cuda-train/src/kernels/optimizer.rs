//! Minimal optimizer kernels for CO-010.

use cuda_device::{device, kernel, thread, DisjointSlice};

#[kernel]
pub fn adamw_update(
    gradients: &[f32],
    mut weights: DisjointSlice<f32>,
    mut momentum: DisjointSlice<f32>,
    mut velocity: DisjointSlice<f32>,
    len: u32,
    gradient_factor: f32,
    learning_rate: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    min_weight: f32,
    max_weight: f32,
) {
    let tid = thread::index_1d();
    let idx = tid.get();
    if idx >= len as usize {
        return;
    }

    let grad = gradient_factor * gradients[idx];
    unsafe {
        let weight = weights.get_unchecked_mut(idx);
        let momentum_value = momentum.get_unchecked_mut(idx);
        let velocity_value = velocity.get_unchecked_mut(idx);

        *weight *= 1.0_f32 - decay * learning_rate;
        *momentum_value = beta1 * *momentum_value + (1.0_f32 - beta1) * grad;
        *velocity_value = beta2 * *velocity_value + (1.0_f32 - beta2) * grad * grad;
        *weight -= learning_rate * (*momentum_value / (core::intrinsics::sqrtf32(*velocity_value) + epsilon));
        *weight = clamp_f32(*weight, min_weight, max_weight);
    }
}

#[kernel]
pub fn radam_update(
    gradients: &[f32],
    mut weights: DisjointSlice<f32>,
    mut momentum: DisjointSlice<f32>,
    mut velocity: DisjointSlice<f32>,
    len: u32,
    gradient_factor: f32,
    learning_rate: f32,
    step_size: f32,
    use_denom: u32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    min_weight: f32,
    max_weight: f32,
) {
    let tid = thread::index_1d();
    let idx = tid.get();
    if idx >= len as usize {
        return;
    }

    let rate = learning_rate * step_size;
    let grad = gradient_factor * gradients[idx];
    unsafe {
        let weight = weights.get_unchecked_mut(idx);
        let momentum_value = momentum.get_unchecked_mut(idx);
        let velocity_value = velocity.get_unchecked_mut(idx);

        *weight *= 1.0_f32 - decay * rate;
        *momentum_value = beta1 * *momentum_value + (1.0_f32 - beta1) * grad;
        *velocity_value = beta2 * *velocity_value + (1.0_f32 - beta2) * grad * grad;

        let mut value = *momentum_value;
        if use_denom != 0 {
            value /= core::intrinsics::sqrtf32(*velocity_value) + epsilon;
        }
        *weight -= rate * value;
        *weight = clamp_f32(*weight, min_weight, max_weight);
    }
}

#[device]
fn clamp_f32(value: f32, min_value: f32, max_value: f32) -> f32 {
    if value < min_value {
        min_value
    } else if value > max_value {
        max_value
    } else {
        value
    }
}
