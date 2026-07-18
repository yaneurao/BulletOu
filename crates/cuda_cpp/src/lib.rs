use std::{error, ffi::CStr, fmt, os::raw::c_char, ptr::NonNull};

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

#[derive(Debug)]
pub struct Context {
    raw: NonNull<ffi::BulletOuCudaCppContext>,
}

impl Context {
    pub fn new(device: i32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer.
        check(unsafe { ffi::bulletou_cuda_cpp_context_create(device, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| CudaCppError::message("C++/CUDA context_create returned null"))?;
        Ok(Self { raw })
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `self.raw` is owned by this wrapper and valid until Drop.
        check(unsafe { ffi::bulletou_cuda_cpp_context_synchronize(self.raw.as_ptr()) })
    }

    fn as_ptr(&self) -> *mut ffi::BulletOuCudaCppContext {
        self.raw.as_ptr()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and should be destroyed once.
        let _ = unsafe { ffi::bulletou_cuda_cpp_context_destroy(self.raw.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct F32Buffer {
    raw: NonNull<ffi::BulletOuCudaCppF32Buffer>,
    len: usize,
}

impl F32Buffer {
    pub fn new(ctx: &Context, len: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer and `ctx` is valid.
        check(unsafe { ffi::bulletou_cuda_cpp_f32_buffer_create(ctx.as_ptr(), len, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| CudaCppError::message("C++/CUDA f32_buffer_create returned null"))?;
        Ok(Self { raw, len })
    }

    pub fn from_host(ctx: &Context, values: &[f32]) -> Result<Self> {
        let buffer = Self::new(ctx, values.len())?;
        buffer.upload(ctx, values)?;
        Ok(buffer)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn upload(&self, ctx: &Context, values: &[f32]) -> Result<()> {
        if values.len() > self.len {
            return Err(CudaCppError::message(format!(
                "upload length {} exceeds device buffer length {}",
                values.len(),
                self.len
            )));
        }
        // SAFETY: host slice is valid for `values.len()`; backend validates device buffer length.
        check(unsafe {
            ffi::bulletou_cuda_cpp_f32_upload(ctx.as_ptr(), self.raw.as_ptr(), values.as_ptr(), values.len())
        })
    }

    pub fn download(&self, ctx: &Context) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.len];
        self.download_prefix(ctx, &mut out)?;
        Ok(out)
    }

    pub fn download_prefix(&self, ctx: &Context, out: &mut [f32]) -> Result<()> {
        if out.len() > self.len {
            return Err(CudaCppError::message(format!(
                "download length {} exceeds device buffer length {}",
                out.len(),
                self.len
            )));
        }
        // SAFETY: host slice is valid for `out.len()`; backend validates device buffer length.
        check(unsafe {
            ffi::bulletou_cuda_cpp_f32_download(ctx.as_ptr(), self.raw.as_ptr(), out.as_mut_ptr(), out.len())
        })
    }

    pub fn fill(&self, ctx: &Context, value: f32) -> Result<()> {
        // SAFETY: backend validates device buffer length.
        check(unsafe { ffi::bulletou_cuda_cpp_f32_fill(ctx.as_ptr(), self.raw.as_ptr(), value, self.len) })
    }

    fn as_ptr(&self) -> *mut ffi::BulletOuCudaCppF32Buffer {
        self.raw.as_ptr()
    }
}

impl Drop for F32Buffer {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and should be destroyed once.
        let _ = unsafe { ffi::bulletou_cuda_cpp_f32_buffer_destroy(self.raw.as_ptr()) };
    }
}

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

pub fn axpy_device(ctx: &Context, len: usize, a: f32, x: &F32Buffer, y: &F32Buffer, out: &F32Buffer) -> Result<()> {
    // SAFETY: backend validates buffer lengths and device ownership.
    check(unsafe { ffi::bulletou_cuda_cpp_axpy_device(ctx.as_ptr(), len, a, x.as_ptr(), y.as_ptr(), out.as_ptr()) })
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

pub struct RangerDeviceStateMut<'a> {
    pub gradients: &'a F32Buffer,
    pub weights: &'a F32Buffer,
    pub momentum: &'a F32Buffer,
    pub velocity: &'a F32Buffer,
    pub slow_params: &'a F32Buffer,
}

impl RangerDeviceStateMut<'_> {
    fn validate(&self) -> Result<usize> {
        let len = self.gradients.len();
        for (name, actual) in [
            ("weights", self.weights.len()),
            ("momentum", self.momentum.len()),
            ("velocity", self.velocity.len()),
            ("slow_params", self.slow_params.len()),
        ] {
            if actual != len {
                return Err(CudaCppError::message(format!(
                    "Ranger device length mismatch: gradients={len}, {name}={actual}"
                )));
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

pub fn ranger_update_device(ctx: &Context, params: RangerUpdateParams, state: RangerDeviceStateMut<'_>) -> Result<()> {
    params.validate()?;
    let len = state.validate()?;
    let scale = params.radam.step_scale()?;
    let do_lookahead = params.should_lookahead()?;

    // SAFETY: backend validates device ownership and lengths.
    check(unsafe {
        ffi::bulletou_cuda_cpp_ranger_update_device(
            ctx.as_ptr(),
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
            state.gradients.as_ptr(),
            state.weights.as_ptr(),
            state.momentum.as_ptr(),
            state.velocity.as_ptr(),
            state.slow_params.as_ptr(),
        )
    })
}

fn check(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(CudaCppError::from_last_error(code)) }
}

mod ffi {
    use super::c_char;

    #[repr(C)]
    pub struct BulletOuCudaCppContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct BulletOuCudaCppF32Buffer {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn bulletou_cuda_cpp_last_error(out: *mut c_char, out_len: usize) -> i32;
        pub fn bulletou_cuda_cpp_device_name(device: i32, out: *mut c_char, out_len: usize) -> i32;
        pub fn bulletou_cuda_cpp_context_create(device: i32, out: *mut *mut BulletOuCudaCppContext) -> i32;
        pub fn bulletou_cuda_cpp_context_destroy(ctx: *mut BulletOuCudaCppContext) -> i32;
        pub fn bulletou_cuda_cpp_context_synchronize(ctx: *mut BulletOuCudaCppContext) -> i32;
        pub fn bulletou_cuda_cpp_f32_buffer_create(
            ctx: *mut BulletOuCudaCppContext,
            len: usize,
            out: *mut *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_f32_buffer_destroy(buffer: *mut BulletOuCudaCppF32Buffer) -> i32;
        pub fn bulletou_cuda_cpp_f32_upload(
            ctx: *mut BulletOuCudaCppContext,
            dst: *mut BulletOuCudaCppF32Buffer,
            src: *const f32,
            len: usize,
        ) -> i32;
        pub fn bulletou_cuda_cpp_f32_download(
            ctx: *mut BulletOuCudaCppContext,
            src: *mut BulletOuCudaCppF32Buffer,
            dst: *mut f32,
            len: usize,
        ) -> i32;
        pub fn bulletou_cuda_cpp_f32_fill(
            ctx: *mut BulletOuCudaCppContext,
            dst: *mut BulletOuCudaCppF32Buffer,
            value: f32,
            len: usize,
        ) -> i32;
        pub fn bulletou_cuda_cpp_axpy_device(
            ctx: *mut BulletOuCudaCppContext,
            len: usize,
            a: f32,
            x: *mut BulletOuCudaCppF32Buffer,
            y: *mut BulletOuCudaCppF32Buffer,
            out: *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
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
        pub fn bulletou_cuda_cpp_ranger_update_device(
            ctx: *mut BulletOuCudaCppContext,
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
            gradients: *mut BulletOuCudaCppF32Buffer,
            weights: *mut BulletOuCudaCppF32Buffer,
            momentum: *mut BulletOuCudaCppF32Buffer,
            velocity: *mut BulletOuCudaCppF32Buffer,
            slow_params: *mut BulletOuCudaCppF32Buffer,
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

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn persistent_device_api_smoke() {
        let ctx = Context::new(0).unwrap();
        let x = F32Buffer::from_host(&ctx, &[1.0, 2.0, 3.0]).unwrap();
        let y = F32Buffer::from_host(&ctx, &[10.0, 20.0, 30.0]).unwrap();
        let out = F32Buffer::new(&ctx, 3).unwrap();
        axpy_device(&ctx, 3, 2.0, &x, &y, &out).unwrap();
        assert_eq!(out.download(&ctx).unwrap(), vec![12.0, 24.0, 36.0]);
    }
}
