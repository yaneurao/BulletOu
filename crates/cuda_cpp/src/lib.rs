use std::{error, ffi::CStr, fmt, os::raw::c_char};

#[derive(Debug, Clone, PartialEq)]
pub struct CudaCppError {
    message: String,
}

impl CudaCppError {
    fn from_last_error(code: i32) -> Self {
        let mut bytes = vec![0i8; 1024];
        let fallback = format!("C++/CUDA backend failed with code {code}");
        // SAFETY: `bytes` is a valid writable C buffer.
        let status = unsafe { ffi::bulletou_cuda_cpp_last_error(bytes.as_mut_ptr(), bytes.len()) };
        if status == 0 {
            // SAFETY: the backend always nul-terminates on success.
            let message = unsafe { CStr::from_ptr(bytes.as_ptr()) }.to_string_lossy().into_owned();
            if message.is_empty() { Self { message: fallback } } else { Self { message } }
        } else {
            Self { message: fallback }
        }
    }

    fn message(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for CudaCppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl error::Error for CudaCppError {}

pub type Result<T> = std::result::Result<T, CudaCppError>;

pub fn device_name(device: i32) -> Result<String> {
    let mut bytes = vec![0i8; 256];
    // SAFETY: `bytes` is a valid writable C buffer.
    check(unsafe { ffi::bulletou_cuda_cpp_device_name(device, bytes.as_mut_ptr(), bytes.len()) })?;
    // SAFETY: backend returns a nul-terminated string on success.
    Ok(unsafe { CStr::from_ptr(bytes.as_ptr()) }.to_string_lossy().into_owned())
}

pub fn axpy_host(device: i32, a: f32, x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
    if x.len() != y.len() {
        return Err(CudaCppError::message(format!("axpy length mismatch: x={} y={}", x.len(), y.len())));
    }
    let mut out = vec![0.0; x.len()];
    // SAFETY: all slices have identical length and valid pointers for `len` elements.
    check(unsafe { ffi::bulletou_cuda_cpp_axpy_host(device, x.len(), a, x.as_ptr(), y.as_ptr(), out.as_mut_ptr()) })?;
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RAdamUpdateParams {
    pub step: u64,
    pub gradient_factor: f32,
    pub learning_rate: f32,
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub n_sma_threshold: f32,
    pub epsilon: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for RAdamUpdateParams {
    fn default() -> Self {
        Self {
            step: 1,
            gradient_factor: 1.0,
            learning_rate: 0.001,
            decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            n_sma_threshold: 5.0,
            epsilon: 0.00000001,
            min_weight: -1.98,
            max_weight: 1.98,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RAdamStepScale {
    pub step_size: f32,
    pub use_denom: bool,
}

impl RAdamUpdateParams {
    pub fn validate(self) -> Result<()> {
        if self.step == 0 {
            Err(CudaCppError::message("RAdam step must be greater than zero"))
        } else if !self.gradient_factor.is_finite() {
            Err(CudaCppError::message("RAdam gradient_factor must be finite"))
        } else if !(self.learning_rate.is_finite() && self.learning_rate >= 0.0) {
            Err(CudaCppError::message("RAdam learning_rate must be finite and non-negative"))
        } else if !self.decay.is_finite() {
            Err(CudaCppError::message("RAdam decay must be finite"))
        } else if !(self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1)) {
            Err(CudaCppError::message("RAdam beta1 must be finite and in [0, 1)"))
        } else if !(self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2)) {
            Err(CudaCppError::message("RAdam beta2 must be finite and in [0, 1)"))
        } else if !(self.n_sma_threshold.is_finite() && self.n_sma_threshold >= 0.0) {
            Err(CudaCppError::message("RAdam n_sma_threshold must be finite and non-negative"))
        } else if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            Err(CudaCppError::message("RAdam epsilon must be finite and positive"))
        } else if !(self.min_weight.is_finite() && self.max_weight.is_finite() && self.min_weight <= self.max_weight) {
            Err(CudaCppError::message("RAdam clamp range must be finite and ordered"))
        } else {
            Ok(())
        }
    }

    pub fn step_scale(self) -> Result<RAdamStepScale> {
        self.validate()?;
        let step = self.step as f32;
        let beta2_t = self.beta2.powf(step);
        let n_sma_max = 2.0 / (1.0 - self.beta2) - 1.0;
        let n_sma = n_sma_max - 2.0 * step * beta2_t / (1.0 - beta2_t);
        let bias_correction1 = 1.0 - self.beta1.powf(step);

        let use_denom = n_sma > self.n_sma_threshold;
        let step_size = if use_denom {
            let p1 = (n_sma - 4.0) / (n_sma_max - 4.0);
            let p2 = (n_sma - 2.0) / n_sma;
            let p3 = n_sma_max / (n_sma_max - 2.0);
            (p1 * p2 * p3).sqrt() / bias_correction1
        } else {
            1.0 / bias_correction1
        };
        Ok(RAdamStepScale { step_size, use_denom })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangerUpdateParams {
    pub radam: RAdamUpdateParams,
    pub lookahead_alpha: f32,
    pub lookahead_period: u64,
}

impl Default for RangerUpdateParams {
    fn default() -> Self {
        Self {
            radam: RAdamUpdateParams { decay: 0.0, ..RAdamUpdateParams::default() },
            lookahead_alpha: 0.5,
            lookahead_period: 6,
        }
    }
}

impl RangerUpdateParams {
    pub fn validate(self) -> Result<()> {
        self.radam.validate()?;
        if !(self.lookahead_alpha.is_finite() && (0.0..=1.0).contains(&self.lookahead_alpha)) {
            Err(CudaCppError::message("Ranger lookahead_alpha must be finite and in [0, 1]"))
        } else if self.lookahead_period == 0 {
            Err(CudaCppError::message("Ranger lookahead_period must be greater than zero"))
        } else {
            Ok(())
        }
    }

    pub fn should_lookahead(self) -> Result<bool> {
        self.validate()?;
        Ok(self.radam.step.is_multiple_of(self.lookahead_period))
    }
}

pub struct RangerStateMut<'a> {
    pub gradients: &'a mut [f32],
    pub weights: &'a mut [f32],
    pub momentum: &'a mut [f32],
    pub velocity: &'a mut [f32],
    pub slow_params: &'a mut [f32],
}

impl RangerStateMut<'_> {
    fn validate(&self) -> Result<usize> {
        let len = self.gradients.len();
        for (name, actual) in [
            ("weights", self.weights.len()),
            ("momentum", self.momentum.len()),
            ("velocity", self.velocity.len()),
            ("slow_params", self.slow_params.len()),
        ] {
            if actual != len {
                return Err(CudaCppError::message(format!("Ranger length mismatch: gradients={len}, {name}={actual}")));
            }
        }
        Ok(len)
    }
}

pub fn ranger_update_host(device: i32, params: RangerUpdateParams, state: RangerStateMut<'_>) -> Result<()> {
    params.validate()?;
    let len = state.validate()?;
    let scale = params.radam.step_scale()?;
    let do_lookahead = params.should_lookahead()?;

    // SAFETY: all mutable slices are validated to have `len` elements and valid pointers.
    check(unsafe {
        ffi::bulletou_cuda_cpp_ranger_update_host(
            device,
            len,
            params.radam.gradient_factor,
            params.radam.learning_rate,
            scale.step_size,
            i32::from(scale.use_denom),
            params.radam.decay,
            params.radam.beta1,
            params.radam.beta2,
            params.radam.epsilon,
            params.radam.min_weight,
            params.radam.max_weight,
            i32::from(do_lookahead),
            params.lookahead_alpha,
            state.gradients.as_mut_ptr(),
            state.weights.as_mut_ptr(),
            state.momentum.as_mut_ptr(),
            state.velocity.as_mut_ptr(),
            state.slow_params.as_mut_ptr(),
        )
    })
}

fn check(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(CudaCppError::from_last_error(code)) }
}

mod ffi {
    use super::c_char;

    unsafe extern "C" {
        pub fn bulletou_cuda_cpp_last_error(out: *mut c_char, out_len: usize) -> i32;
        pub fn bulletou_cuda_cpp_device_name(device: i32, out: *mut c_char, out_len: usize) -> i32;
        pub fn bulletou_cuda_cpp_axpy_host(
            device: i32,
            len: usize,
            a: f32,
            x: *const f32,
            y: *const f32,
            out: *mut f32,
        ) -> i32;
        pub fn bulletou_cuda_cpp_ranger_update_host(
            device: i32,
            len: usize,
            gradient_factor: f32,
            learning_rate: f32,
            step_size: f32,
            use_denom: i32,
            decay: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            min_weight: f32,
            max_weight: f32,
            do_lookahead: i32,
            lookahead_alpha: f32,
            gradients: *mut f32,
            weights: *mut f32,
            momentum: *mut f32,
            velocity: *mut f32,
            slow_params: *mut f32,
        ) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radam_step_scale_matches_cuda_oxide_reference_points() {
        let step1 = RAdamUpdateParams { step: 1, ..Default::default() }.step_scale().unwrap();
        assert!(!step1.use_denom);
        assert!((step1.step_size - 10.0).abs() < 0.00001);

        let step6 = RAdamUpdateParams { step: 6, ..Default::default() }.step_scale().unwrap();
        assert!(step6.step_size.is_finite());
        assert!(step6.step_size > 0.0);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn axpy_gpu_smoke() {
        let out = axpy_host(0, 2.0, &[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(out, vec![12.0, 24.0, 36.0]);
    }
}
