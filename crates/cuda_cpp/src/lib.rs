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

    pub fn begin_capture(&self) -> Result<()> {
        // SAFETY: `self.raw` is owned by this wrapper and valid until Drop.
        check(unsafe { ffi::bulletou_cuda_cpp_graph_begin_capture(self.raw.as_ptr()) })
    }

    pub fn end_capture(&self) -> Result<GraphExec> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer and `self.raw` is valid.
        check(unsafe { ffi::bulletou_cuda_cpp_graph_end_capture(self.raw.as_ptr(), &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| CudaCppError::message("C++/CUDA graph_end_capture returned null"))?;
        Ok(GraphExec { raw })
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
pub struct Event {
    raw: NonNull<ffi::BulletOuCudaCppEvent>,
}

impl Event {
    pub fn new(ctx: &Context) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer and `ctx` is valid.
        check(unsafe { ffi::bulletou_cuda_cpp_event_create(ctx.as_ptr(), &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| CudaCppError::message("C++/CUDA event_create returned null"))?;
        Ok(Self { raw })
    }

    pub fn record(&self, ctx: &Context) -> Result<()> {
        // SAFETY: `self.raw` and `ctx` are valid.
        check(unsafe { ffi::bulletou_cuda_cpp_event_record(ctx.as_ptr(), self.raw.as_ptr()) })
    }

    pub fn wait(&self, ctx: &Context) -> Result<()> {
        // SAFETY: `self.raw` and `ctx` are valid.
        check(unsafe { ffi::bulletou_cuda_cpp_event_wait(ctx.as_ptr(), self.raw.as_ptr()) })
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `self.raw` is valid until Drop.
        check(unsafe { ffi::bulletou_cuda_cpp_event_synchronize(self.raw.as_ptr()) })
    }

    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32> {
        let mut out = 0.0;
        // SAFETY: both events are valid and `out` is a valid out pointer.
        check(unsafe { ffi::bulletou_cuda_cpp_event_elapsed_ms(start.raw.as_ptr(), self.raw.as_ptr(), &mut out) })?;
        Ok(out)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and should be destroyed once.
        let _ = unsafe { ffi::bulletou_cuda_cpp_event_destroy(self.raw.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct GraphExec {
    raw: NonNull<ffi::BulletOuCudaCppGraphExec>,
}

impl GraphExec {
    pub fn launch(&self, ctx: &Context) -> Result<()> {
        // SAFETY: `self.raw` and `ctx` are valid.
        check(unsafe { ffi::bulletou_cuda_cpp_graph_launch(ctx.as_ptr(), self.raw.as_ptr()) })
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and should be destroyed once.
        let _ = unsafe { ffi::bulletou_cuda_cpp_graph_destroy(self.raw.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct F32Buffer {
    raw: NonNull<ffi::BulletOuCudaCppF32Buffer>,
    len: usize,
}

#[derive(Debug)]
pub struct I32Buffer {
    raw: NonNull<ffi::BulletOuCudaCppI32Buffer>,
    len: usize,
}

#[derive(Debug)]
pub struct F32PinnedBuffer {
    raw: NonNull<ffi::BulletOuCudaCppPinnedF32Buffer>,
    len: usize,
}

#[derive(Debug)]
pub struct I32PinnedBuffer {
    raw: NonNull<ffi::BulletOuCudaCppPinnedI32Buffer>,
    len: usize,
}

#[derive(Debug)]
pub struct F32UploadSlot {
    buffer: F32Buffer,
    ready: Event,
}

#[derive(Debug)]
pub struct I32UploadSlot {
    buffer: I32Buffer,
    ready: Event,
}

impl F32UploadSlot {
    pub fn new(upload_ctx: &Context, len: usize) -> Result<Self> {
        Ok(Self { buffer: F32Buffer::new(upload_ctx, len)?, ready: Event::new(upload_ctx)? })
    }

    pub fn upload(&self, upload_ctx: &Context, values: &[f32]) -> Result<()> {
        self.buffer.upload(upload_ctx, values)?;
        self.ready.record(upload_ctx)
    }

    pub fn wait_on<'a>(&'a self, compute_ctx: &Context) -> Result<&'a F32Buffer> {
        self.ready.wait(compute_ctx)?;
        Ok(&self.buffer)
    }

    pub fn buffer(&self) -> &F32Buffer {
        &self.buffer
    }
}

impl I32UploadSlot {
    pub fn new(upload_ctx: &Context, len: usize) -> Result<Self> {
        Ok(Self { buffer: I32Buffer::new(upload_ctx, len)?, ready: Event::new(upload_ctx)? })
    }

    pub fn upload(&self, upload_ctx: &Context, values: &[i32]) -> Result<()> {
        self.buffer.upload(upload_ctx, values)?;
        self.ready.record(upload_ctx)
    }

    pub fn wait_on<'a>(&'a self, compute_ctx: &Context) -> Result<&'a I32Buffer> {
        self.ready.wait(compute_ctx)?;
        Ok(&self.buffer)
    }

    pub fn buffer(&self) -> &I32Buffer {
        &self.buffer
    }
}

impl F32PinnedBuffer {
    pub fn new(ctx: &Context, len: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer and `ctx` is valid.
        check(unsafe { ffi::bulletou_cuda_cpp_pinned_f32_buffer_create(ctx.as_ptr(), len, &mut raw) })?;
        let raw = NonNull::new(raw)
            .ok_or_else(|| CudaCppError::message("C++/CUDA pinned_f32_buffer_create returned null"))?;
        Ok(Self { raw, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn upload_to_device(&self, ctx: &Context, dst: &F32Buffer, values: &[f32]) -> Result<()> {
        if values.len() > self.len {
            return Err(CudaCppError::message(format!(
                "staged f32 upload length {} exceeds pinned buffer length {}",
                values.len(),
                self.len
            )));
        }
        if values.len() > dst.len() {
            return Err(CudaCppError::message(format!(
                "staged f32 upload length {} exceeds device buffer length {}",
                values.len(),
                dst.len()
            )));
        }
        // SAFETY: host slice is copied into an owned pinned host buffer before enqueueing the device copy.
        check(unsafe {
            ffi::bulletou_cuda_cpp_f32_upload_staged(
                ctx.as_ptr(),
                dst.as_ptr(),
                self.raw.as_ptr(),
                values.as_ptr(),
                values.len(),
            )
        })
    }
}

impl Drop for F32PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and should be destroyed once.
        let _ = unsafe { ffi::bulletou_cuda_cpp_pinned_f32_buffer_destroy(self.raw.as_ptr()) };
    }
}

impl I32PinnedBuffer {
    pub fn new(ctx: &Context, len: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer and `ctx` is valid.
        check(unsafe { ffi::bulletou_cuda_cpp_pinned_i32_buffer_create(ctx.as_ptr(), len, &mut raw) })?;
        let raw = NonNull::new(raw)
            .ok_or_else(|| CudaCppError::message("C++/CUDA pinned_i32_buffer_create returned null"))?;
        Ok(Self { raw, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn upload_to_device(&self, ctx: &Context, dst: &I32Buffer, values: &[i32]) -> Result<()> {
        if values.len() > self.len {
            return Err(CudaCppError::message(format!(
                "staged i32 upload length {} exceeds pinned buffer length {}",
                values.len(),
                self.len
            )));
        }
        if values.len() > dst.len() {
            return Err(CudaCppError::message(format!(
                "staged i32 upload length {} exceeds device buffer length {}",
                values.len(),
                dst.len()
            )));
        }
        // SAFETY: host slice is copied into an owned pinned host buffer before enqueueing the device copy.
        check(unsafe {
            ffi::bulletou_cuda_cpp_i32_upload_staged(
                ctx.as_ptr(),
                dst.as_ptr(),
                self.raw.as_ptr(),
                values.as_ptr(),
                values.len(),
            )
        })
    }
}

impl Drop for I32PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and should be destroyed once.
        let _ = unsafe { ffi::bulletou_cuda_cpp_pinned_i32_buffer_destroy(self.raw.as_ptr()) };
    }
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

impl I32Buffer {
    pub fn new(ctx: &Context, len: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer and `ctx` is valid.
        check(unsafe { ffi::bulletou_cuda_cpp_i32_buffer_create(ctx.as_ptr(), len, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| CudaCppError::message("C++/CUDA i32_buffer_create returned null"))?;
        Ok(Self { raw, len })
    }

    pub fn from_host(ctx: &Context, values: &[i32]) -> Result<Self> {
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

    pub fn upload(&self, ctx: &Context, values: &[i32]) -> Result<()> {
        if values.len() > self.len {
            return Err(CudaCppError::message(format!(
                "i32 upload length {} exceeds device buffer length {}",
                values.len(),
                self.len
            )));
        }
        // SAFETY: host slice is valid for `values.len()`; backend validates device buffer length.
        check(unsafe {
            ffi::bulletou_cuda_cpp_i32_upload(ctx.as_ptr(), self.raw.as_ptr(), values.as_ptr(), values.len())
        })
    }

    pub fn download(&self, ctx: &Context) -> Result<Vec<i32>> {
        let mut out = vec![0; self.len];
        self.download_prefix(ctx, &mut out)?;
        Ok(out)
    }

    pub fn download_prefix(&self, ctx: &Context, out: &mut [i32]) -> Result<()> {
        if out.len() > self.len {
            return Err(CudaCppError::message(format!(
                "i32 download length {} exceeds device buffer length {}",
                out.len(),
                self.len
            )));
        }
        // SAFETY: host slice is valid for `out.len()`; backend validates device buffer length.
        check(unsafe {
            ffi::bulletou_cuda_cpp_i32_download(ctx.as_ptr(), self.raw.as_ptr(), out.as_mut_ptr(), out.len())
        })
    }

    fn as_ptr(&self) -> *mut ffi::BulletOuCudaCppI32Buffer {
        self.raw.as_ptr()
    }
}

impl Drop for I32Buffer {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and should be destroyed once.
        let _ = unsafe { ffi::bulletou_cuda_cpp_i32_buffer_destroy(self.raw.as_ptr()) };
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardShape {
    pub input_size: usize,
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

pub const NNUE_HALFKP_256X2_32_32: NnueForwardShape = NnueForwardShape { input_size: 125_388, l1: 256, l2: 32, l3: 32 };

pub fn nnue_l0w_len(shape: NnueForwardShape) -> Result<usize> {
    checked_product("l0w", &[shape.input_size, shape.l1])
}

fn nnue_l0w_len_saturating(shape: NnueForwardShape) -> usize {
    shape.input_size.saturating_mul(shape.l1)
}

#[derive(Debug, Clone, Copy)]
pub struct NnueForwardHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl NnueForwardHostBatch<'_> {
    pub fn validate(self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(CudaCppError::message("NNUE batch_size must be greater than zero"));
        }
        if self.max_active == 0 {
            return Err(CudaCppError::message("NNUE max_active must be greater than zero"));
        }
        let sparse_len = self
            .batch_size
            .checked_mul(self.max_active)
            .ok_or_else(|| CudaCppError::message("NNUE sparse batch length overflow"))?;
        expect_len("stm_indices", sparse_len, self.stm_indices.len())?;
        expect_len("nstm_indices", sparse_len, self.nstm_indices.len())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NnueForwardHostWeights<'a> {
    pub shape: NnueForwardShape,
    pub l0w: &'a [f32],
    pub l0b: &'a [f32],
    pub l1w: &'a [f32],
    pub l1b: &'a [f32],
    pub l2w: &'a [f32],
    pub l2b: &'a [f32],
    pub outw: &'a [f32],
    pub outb: &'a [f32],
}

impl NnueForwardHostWeights<'_> {
    pub fn validate(self) -> Result<()> {
        let shape = self.shape;
        validate_nnue_shape(shape)?;
        expect_len("l0w", nnue_l0w_len(shape)?, self.l0w.len())?;
        expect_len("l0b", shape.l1, self.l0b.len())?;
        expect_len("l1w", checked_product("l1w", &[shape.l1, 2, shape.l2])?, self.l1w.len())?;
        expect_len("l1b", shape.l2, self.l1b.len())?;
        expect_len("l2w", checked_product("l2w", &[shape.l2, shape.l3])?, self.l2w.len())?;
        expect_len("l2b", shape.l3, self.l2b.len())?;
        expect_len("outw", shape.l3, self.outw.len())?;
        expect_len("outb", 1, self.outb.len())
    }
}

#[derive(Debug)]
pub struct NnueForwardDeviceBatch {
    pub batch_size: usize,
    pub max_active: usize,
    pub stm_indices: I32Buffer,
    pub nstm_indices: I32Buffer,
}

impl NnueForwardDeviceBatch {
    pub fn from_host(ctx: &Context, batch: NnueForwardHostBatch<'_>) -> Result<Self> {
        batch.validate()?;
        Ok(Self {
            batch_size: batch.batch_size,
            max_active: batch.max_active,
            stm_indices: I32Buffer::from_host(ctx, batch.stm_indices)?,
            nstm_indices: I32Buffer::from_host(ctx, batch.nstm_indices)?,
        })
    }

    fn validate(&self) -> Result<()> {
        let sparse_len = self
            .batch_size
            .checked_mul(self.max_active)
            .ok_or_else(|| CudaCppError::message("NNUE sparse batch length overflow"))?;
        expect_len("device stm_indices", sparse_len, self.stm_indices.len())?;
        expect_len("device nstm_indices", sparse_len, self.nstm_indices.len())
    }
}

#[derive(Debug)]
pub struct NnueForwardDeviceWeights {
    pub shape: NnueForwardShape,
    pub l0w: F32Buffer,
    pub l0b: F32Buffer,
    pub l1w: F32Buffer,
    pub l1b: F32Buffer,
    pub l2w: F32Buffer,
    pub l2b: F32Buffer,
    pub outw: F32Buffer,
    pub outb: F32Buffer,
}

impl NnueForwardDeviceWeights {
    pub fn from_host(ctx: &Context, weights: NnueForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            shape: weights.shape,
            l0w: F32Buffer::from_host(ctx, weights.l0w)?,
            l0b: F32Buffer::from_host(ctx, weights.l0b)?,
            l1w: F32Buffer::from_host(ctx, weights.l1w)?,
            l1b: F32Buffer::from_host(ctx, weights.l1b)?,
            l2w: F32Buffer::from_host(ctx, weights.l2w)?,
            l2b: F32Buffer::from_host(ctx, weights.l2b)?,
            outw: F32Buffer::from_host(ctx, weights.outw)?,
            outb: F32Buffer::from_host(ctx, weights.outb)?,
        })
    }

    fn validate(&self) -> Result<()> {
        let shape = self.shape;
        validate_nnue_shape(shape)?;
        expect_len("device l0w", nnue_l0w_len(shape)?, self.l0w.len())?;
        expect_len("device l0b", shape.l1, self.l0b.len())?;
        expect_len("device l1w", checked_product("l1w", &[shape.l1, 2, shape.l2])?, self.l1w.len())?;
        expect_len("device l1b", shape.l2, self.l1b.len())?;
        expect_len("device l2w", checked_product("l2w", &[shape.l2, shape.l3])?, self.l2w.len())?;
        expect_len("device l2b", shape.l3, self.l2b.len())?;
        expect_len("device outw", shape.l3, self.outw.len())?;
        expect_len("device outb", 1, self.outb.len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueForwardWorkspaceLayout {
    pub shape: NnueForwardShape,
    pub batch_size: usize,
}

impl NnueForwardWorkspaceLayout {
    pub fn new(shape: NnueForwardShape, batch_size: usize) -> Self {
        Self { shape, batch_size }
    }

    pub fn l0_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1)
    }

    pub fn combined_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1).saturating_mul(2)
    }

    pub fn hidden1_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2)
    }

    pub fn hidden2_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l3)
    }

    pub fn output_len(self) -> usize {
        self.batch_size
    }

    fn validate(self) -> Result<()> {
        validate_nnue_shape(self.shape)?;
        if self.batch_size == 0 {
            Err(CudaCppError::message("NNUE workspace batch_size must be greater than zero"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct NnueForwardWorkspace {
    pub layout: NnueForwardWorkspaceLayout,
    pub stm_l0: F32Buffer,
    pub nstm_l0: F32Buffer,
    pub combined: F32Buffer,
    pub hidden1: F32Buffer,
    pub hidden2: F32Buffer,
    pub output: F32Buffer,
}

impl NnueForwardWorkspace {
    pub fn new(ctx: &Context, layout: NnueForwardWorkspaceLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            stm_l0: F32Buffer::new(ctx, layout.l0_len())?,
            nstm_l0: F32Buffer::new(ctx, layout.l0_len())?,
            combined: F32Buffer::new(ctx, layout.combined_len())?,
            hidden1: F32Buffer::new(ctx, layout.hidden1_len())?,
            hidden2: F32Buffer::new(ctx, layout.hidden2_len())?,
            output: F32Buffer::new(ctx, layout.output_len())?,
        })
    }

    fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        expect_len("workspace stm_l0", self.layout.l0_len(), self.stm_l0.len())?;
        expect_len("workspace nstm_l0", self.layout.l0_len(), self.nstm_l0.len())?;
        expect_len("workspace combined", self.layout.combined_len(), self.combined.len())?;
        expect_len("workspace hidden1", self.layout.hidden1_len(), self.hidden1.len())?;
        expect_len("workspace hidden2", self.layout.hidden2_len(), self.hidden2.len())?;
        expect_len("workspace output", self.layout.output_len(), self.output.len())
    }

    pub fn download_output(&self, ctx: &Context) -> Result<Vec<f32>> {
        self.output.download(ctx)
    }
}

pub fn nnue_forward_host(
    device: i32,
    batch: NnueForwardHostBatch<'_>,
    weights: NnueForwardHostWeights<'_>,
) -> Result<Vec<f32>> {
    batch.validate()?;
    weights.validate()?;
    let mut out = vec![0.0; batch.batch_size];
    let shape = weights.shape;
    // SAFETY: all slices have been length-validated against the shape and batch layout.
    check(unsafe {
        ffi::bulletou_cuda_cpp_nnue_forward_host(
            device,
            shape.input_size,
            shape.l1,
            shape.l2,
            shape.l3,
            batch.batch_size,
            batch.max_active,
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            weights.l0w.as_ptr(),
            weights.l0b.as_ptr(),
            weights.l1w.as_ptr(),
            weights.l1b.as_ptr(),
            weights.l2w.as_ptr(),
            weights.l2b.as_ptr(),
            weights.outw.as_ptr(),
            weights.outb.as_ptr(),
            out.as_mut_ptr(),
        )
    })?;
    Ok(out)
}

pub fn nnue_forward_device(
    ctx: &Context,
    batch: &NnueForwardDeviceBatch,
    weights: &NnueForwardDeviceWeights,
    workspace: &NnueForwardWorkspace,
) -> Result<()> {
    batch.validate()?;
    weights.validate()?;
    workspace.validate()?;
    if workspace.layout.batch_size != batch.batch_size {
        return Err(CudaCppError::message(format!(
            "NNUE workspace batch mismatch: workspace={} batch={}",
            workspace.layout.batch_size, batch.batch_size
        )));
    }
    if workspace.layout.shape != weights.shape {
        return Err(CudaCppError::message(format!(
            "NNUE workspace shape mismatch: workspace={:?} weights={:?}",
            workspace.layout.shape, weights.shape
        )));
    }
    let shape = weights.shape;
    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    check(unsafe {
        ffi::bulletou_cuda_cpp_nnue_forward_device(
            ctx.as_ptr(),
            shape.input_size,
            shape.l1,
            shape.l2,
            shape.l3,
            batch.batch_size,
            batch.max_active,
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            weights.l0w.as_ptr(),
            weights.l0b.as_ptr(),
            weights.l1w.as_ptr(),
            weights.l1b.as_ptr(),
            weights.l2w.as_ptr(),
            weights.l2b.as_ptr(),
            weights.outw.as_ptr(),
            weights.outb.as_ptr(),
            workspace.stm_l0.as_ptr(),
            workspace.nstm_l0.as_ptr(),
            workspace.combined.as_ptr(),
            workspace.hidden1.as_ptr(),
            workspace.hidden2.as_ptr(),
            workspace.output.as_ptr(),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardShape {
    pub input_size: usize,
    pub ft_size: usize,
    pub l1_hidden: usize,
    pub l2_size: usize,
    pub num_stacks: usize,
    /// `1` means normal dense stacked L1 unless the common+shard fields
    /// below are non-zero. Values greater than one mean grouped L1:
    /// `ft_size / l1_group_count` inputs connect to
    /// `l1_out / l1_group_count` outputs in each group.
    ///
    /// For common+shard L1 this is the shard group count: each output
    /// group sees the common prefix plus its own shard.
    pub l1_group_count: usize,
    /// Common input prefix size for common+shard SFNN L1. This may be `0`
    /// for pure grouped L1 expressed as `c0_sMxG`.
    pub l1_common_size: usize,
    /// Per-shard input size for common+shard SFNN L1. `0` disables
    /// common+shard mode unless `l1_common_size` is also non-zero, which is
    /// rejected as an invalid partial common+shard shape.
    pub l1_shard_size: usize,
}

impl SfnnForwardShape {
    pub fn l1_out(self) -> usize {
        self.l1_hidden + 1
    }

    pub fn l2_in(self) -> usize {
        self.l1_hidden * 2
    }

    pub fn pairwise_size(self) -> usize {
        self.ft_size / 2
    }

    pub fn l1_group_count(self) -> usize {
        self.l1_group_count
    }

    pub fn has_grouped_l1(self) -> bool {
        self.l1_group_count > 1 && !self.has_common_shard_l1()
    }

    pub fn has_common_shard_l1(self) -> bool {
        self.l1_common_size != 0 || self.l1_shard_size != 0
    }

    pub fn has_compact_l1(self) -> bool {
        self.has_grouped_l1() || self.has_common_shard_l1()
    }

    pub fn l1_group_input(self) -> usize {
        self.ft_size / self.l1_group_count
    }

    pub fn l1_group_output(self) -> usize {
        self.l1_out() / self.l1_group_count
    }

    pub fn l1_common_shard_input(self) -> usize {
        self.l1_common_size + self.l1_shard_size
    }

    pub fn l1w_len(self) -> Result<usize> {
        if self.has_common_shard_l1() {
            if self.l1_group_count == 0
                || self.l1_shard_size == 0
                || self.l1_common_size + self.l1_shard_size * self.l1_group_count != self.ft_size
                || self.l1_out() % self.l1_group_count != 0
            {
                return Err(CudaCppError::message(format!(
                    "SFNN common+shard L1 shape dimensions are invalid: {self:?}"
                )));
            }
            checked_product("sfnn common+shard l1w", &[self.num_stacks, self.l1_out(), self.l1_common_shard_input()])
        } else if self.has_grouped_l1() {
            if self.l1_group_count == 0
                || self.ft_size % self.l1_group_count != 0
                || self.l1_out() % self.l1_group_count != 0
            {
                return Err(CudaCppError::message(format!("SFNN grouped-L1 shape dimensions are invalid: {self:?}")));
            }
            checked_product(
                "sfnn grouped l1w",
                &[self.num_stacks, self.l1_group_count, self.l1_group_output(), self.l1_group_input()],
            )
        } else {
            checked_product("sfnn l1w", &[self.num_stacks, self.l1_out(), self.ft_size])
        }
    }

    pub fn l1w_len_saturating(self) -> usize {
        if self.has_common_shard_l1() {
            self.num_stacks.saturating_mul(self.l1_out()).saturating_mul(self.l1_common_shard_input())
        } else if self.has_grouped_l1() {
            self.num_stacks
                .saturating_mul(self.l1_group_count)
                .saturating_mul(self.l1_group_output())
                .saturating_mul(self.l1_group_input())
        } else {
            self.num_stacks.saturating_mul(self.l1_out()).saturating_mul(self.ft_size)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnForwardHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub buckets: &'a [i32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl SfnnForwardHostBatch<'_> {
    pub fn validate(self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(CudaCppError::message("SFNN batch_size must be greater than zero"));
        }
        if self.max_active == 0 {
            return Err(CudaCppError::message("SFNN max_active must be greater than zero"));
        }
        let sparse_len = self
            .batch_size
            .checked_mul(self.max_active)
            .ok_or_else(|| CudaCppError::message("SFNN sparse batch length overflow"))?;
        expect_len("sfnn stm_indices", sparse_len, self.stm_indices.len())?;
        expect_len("sfnn nstm_indices", sparse_len, self.nstm_indices.len())?;
        expect_len("sfnn buckets", self.batch_size, self.buckets.len())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnForwardHostWeights<'a> {
    pub shape: SfnnForwardShape,
    pub l0w: &'a [f32],
    pub l0b: &'a [f32],
    pub l1w: &'a [f32],
    pub l1b: &'a [f32],
    pub l1fw: Option<&'a [f32]>,
    pub l1fb: Option<&'a [f32]>,
    pub l2w: &'a [f32],
    pub l2b: &'a [f32],
    pub l3w: &'a [f32],
    pub l3b: &'a [f32],
}

impl SfnnForwardHostWeights<'_> {
    pub fn validate(self) -> Result<()> {
        let shape = self.shape;
        validate_sfnn_shape(shape)?;
        expect_len("sfnn l0w", checked_product("sfnn l0w", &[shape.input_size, shape.ft_size])?, self.l0w.len())?;
        expect_len("sfnn l0b", shape.ft_size, self.l0b.len())?;
        expect_len("sfnn l1w", shape.l1w_len()?, self.l1w.len())?;
        expect_len("sfnn l1b", checked_product("sfnn l1b", &[shape.num_stacks, shape.l1_out()])?, self.l1b.len())?;
        match (self.l1fw, self.l1fb) {
            (Some(l1fw), Some(l1fb)) => {
                if shape.has_compact_l1() {
                    return Err(CudaCppError::message("SFNN compact L1 does not support factorized L1 weights"));
                }
                expect_len("sfnn l1fw", checked_product("sfnn l1fw", &[shape.ft_size, shape.l1_out()])?, l1fw.len())?;
                expect_len("sfnn l1fb", shape.l1_out(), l1fb.len())?;
            }
            (None, None) => {}
            (Some(_), None) => return Err(CudaCppError::message("SFNN l1fw requires l1fb")),
            (None, Some(_)) => return Err(CudaCppError::message("SFNN l1fb requires l1fw")),
        }
        expect_len(
            "sfnn l2w",
            checked_product("sfnn l2w", &[shape.num_stacks, shape.l2_size, shape.l2_in()])?,
            self.l2w.len(),
        )?;
        expect_len("sfnn l2b", checked_product("sfnn l2b", &[shape.num_stacks, shape.l2_size])?, self.l2b.len())?;
        expect_len("sfnn l3w", checked_product("sfnn l3w", &[shape.num_stacks, shape.l2_size])?, self.l3w.len())?;
        expect_len("sfnn l3b", shape.num_stacks, self.l3b.len())
    }
}

#[derive(Debug)]
pub struct SfnnForwardDeviceBatch {
    pub batch_size: usize,
    pub max_active: usize,
    pub stm_indices: I32Buffer,
    pub nstm_indices: I32Buffer,
    pub buckets: I32Buffer,
}

impl SfnnForwardDeviceBatch {
    pub fn from_host(ctx: &Context, batch: SfnnForwardHostBatch<'_>) -> Result<Self> {
        batch.validate()?;
        Ok(Self {
            batch_size: batch.batch_size,
            max_active: batch.max_active,
            stm_indices: I32Buffer::from_host(ctx, batch.stm_indices)?,
            nstm_indices: I32Buffer::from_host(ctx, batch.nstm_indices)?,
            buckets: I32Buffer::from_host(ctx, batch.buckets)?,
        })
    }

    fn validate(&self) -> Result<()> {
        let sparse_len = self
            .batch_size
            .checked_mul(self.max_active)
            .ok_or_else(|| CudaCppError::message("SFNN sparse batch length overflow"))?;
        expect_len("device sfnn stm_indices", sparse_len, self.stm_indices.len())?;
        expect_len("device sfnn nstm_indices", sparse_len, self.nstm_indices.len())?;
        expect_len("device sfnn buckets", self.batch_size, self.buckets.len())
    }
}

#[derive(Debug)]
pub struct SfnnForwardDeviceWeights {
    pub shape: SfnnForwardShape,
    pub l0w: F32Buffer,
    pub l0b: F32Buffer,
    pub l1w: F32Buffer,
    pub l1b: F32Buffer,
    pub l1fw: Option<F32Buffer>,
    pub l1fb: Option<F32Buffer>,
    pub l2w: F32Buffer,
    pub l2b: F32Buffer,
    pub l3w: F32Buffer,
    pub l3b: F32Buffer,
}

impl SfnnForwardDeviceWeights {
    pub fn from_host(ctx: &Context, weights: SfnnForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            shape: weights.shape,
            l0w: F32Buffer::from_host(ctx, weights.l0w)?,
            l0b: F32Buffer::from_host(ctx, weights.l0b)?,
            l1w: F32Buffer::from_host(ctx, weights.l1w)?,
            l1b: F32Buffer::from_host(ctx, weights.l1b)?,
            l1fw: weights.l1fw.map(|values| F32Buffer::from_host(ctx, values)).transpose()?,
            l1fb: weights.l1fb.map(|values| F32Buffer::from_host(ctx, values)).transpose()?,
            l2w: F32Buffer::from_host(ctx, weights.l2w)?,
            l2b: F32Buffer::from_host(ctx, weights.l2b)?,
            l3w: F32Buffer::from_host(ctx, weights.l3w)?,
            l3b: F32Buffer::from_host(ctx, weights.l3b)?,
        })
    }

    fn validate(&self) -> Result<()> {
        let shape = self.shape;
        validate_sfnn_shape(shape)?;
        expect_len(
            "device sfnn l0w",
            checked_product("sfnn l0w", &[shape.input_size, shape.ft_size])?,
            self.l0w.len(),
        )?;
        expect_len("device sfnn l0b", shape.ft_size, self.l0b.len())?;
        expect_len("device sfnn l1w", shape.l1w_len()?, self.l1w.len())?;
        expect_len(
            "device sfnn l1b",
            checked_product("sfnn l1b", &[shape.num_stacks, shape.l1_out()])?,
            self.l1b.len(),
        )?;
        match (&self.l1fw, &self.l1fb) {
            (Some(l1fw), Some(l1fb)) => {
                if shape.has_compact_l1() {
                    return Err(CudaCppError::message("device SFNN compact L1 does not support factorized L1 weights"));
                }
                expect_len(
                    "device sfnn l1fw",
                    checked_product("sfnn l1fw", &[shape.ft_size, shape.l1_out()])?,
                    l1fw.len(),
                )?;
                expect_len("device sfnn l1fb", shape.l1_out(), l1fb.len())?;
            }
            (None, None) => {}
            (Some(_), None) => return Err(CudaCppError::message("device SFNN l1fw requires l1fb")),
            (None, Some(_)) => return Err(CudaCppError::message("device SFNN l1fb requires l1fw")),
        }
        expect_len(
            "device sfnn l2w",
            checked_product("sfnn l2w", &[shape.num_stacks, shape.l2_size, shape.l2_in()])?,
            self.l2w.len(),
        )?;
        expect_len(
            "device sfnn l2b",
            checked_product("sfnn l2b", &[shape.num_stacks, shape.l2_size])?,
            self.l2b.len(),
        )?;
        expect_len(
            "device sfnn l3w",
            checked_product("sfnn l3w", &[shape.num_stacks, shape.l2_size])?,
            self.l3w.len(),
        )?;
        expect_len("device sfnn l3b", shape.num_stacks, self.l3b.len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardWorkspaceLayout {
    pub shape: SfnnForwardShape,
    pub batch_size: usize,
}

impl SfnnForwardWorkspaceLayout {
    pub fn new(shape: SfnnForwardShape, batch_size: usize) -> Self {
        Self { shape, batch_size }
    }

    pub fn l0_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn combined_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn l1_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1_out())
    }

    pub fn l2_input_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_in())
    }

    pub fn l2_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_size)
    }

    pub fn output_len(self) -> usize {
        self.batch_size
    }

    fn validate(self) -> Result<()> {
        validate_sfnn_shape(self.shape)?;
        if self.batch_size == 0 {
            Err(CudaCppError::message("SFNN workspace batch_size must be greater than zero"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct SfnnForwardWorkspace {
    pub layout: SfnnForwardWorkspaceLayout,
    pub stm_l0: F32Buffer,
    pub nstm_l0: F32Buffer,
    pub combined: F32Buffer,
    pub l1: F32Buffer,
    pub l2_input: F32Buffer,
    pub l2: F32Buffer,
    pub output: F32Buffer,
}

impl SfnnForwardWorkspace {
    pub fn new(ctx: &Context, layout: SfnnForwardWorkspaceLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            stm_l0: F32Buffer::new(ctx, layout.l0_len())?,
            nstm_l0: F32Buffer::new(ctx, layout.l0_len())?,
            combined: F32Buffer::new(ctx, layout.combined_len())?,
            l1: F32Buffer::new(ctx, layout.l1_len())?,
            l2_input: F32Buffer::new(ctx, layout.l2_input_len())?,
            l2: F32Buffer::new(ctx, layout.l2_len())?,
            output: F32Buffer::new(ctx, layout.output_len())?,
        })
    }

    fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        expect_len("sfnn workspace stm_l0", self.layout.l0_len(), self.stm_l0.len())?;
        expect_len("sfnn workspace nstm_l0", self.layout.l0_len(), self.nstm_l0.len())?;
        expect_len("sfnn workspace combined", self.layout.combined_len(), self.combined.len())?;
        expect_len("sfnn workspace l1", self.layout.l1_len(), self.l1.len())?;
        expect_len("sfnn workspace l2_input", self.layout.l2_input_len(), self.l2_input.len())?;
        expect_len("sfnn workspace l2", self.layout.l2_len(), self.l2.len())?;
        expect_len("sfnn workspace output", self.layout.output_len(), self.output.len())
    }

    pub fn download_output(&self, ctx: &Context) -> Result<Vec<f32>> {
        self.output.download(ctx)
    }
}

pub fn sfnn_forward_device(
    ctx: &Context,
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    workspace: &SfnnForwardWorkspace,
) -> Result<()> {
    batch.validate()?;
    weights.validate()?;
    workspace.validate()?;
    if workspace.layout.batch_size != batch.batch_size {
        return Err(CudaCppError::message(format!(
            "SFNN workspace batch mismatch: workspace={} batch={}",
            workspace.layout.batch_size, batch.batch_size
        )));
    }
    if workspace.layout.shape != weights.shape {
        return Err(CudaCppError::message(format!(
            "SFNN workspace shape mismatch: workspace={:?} weights={:?}",
            workspace.layout.shape, weights.shape
        )));
    }
    let shape = weights.shape;
    let (l1fw, l1fb, has_l1f) = match (&weights.l1fw, &weights.l1fb) {
        (Some(l1fw), Some(l1fb)) => (l1fw.as_ptr(), l1fb.as_ptr(), 1),
        (None, None) => (std::ptr::null_mut(), std::ptr::null_mut(), 0),
        _ => return Err(CudaCppError::message("SFNN factorized L1 state is partial")),
    };
    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    check(unsafe {
        ffi::bulletou_cuda_cpp_sfnn_forward_device(
            ctx.as_ptr(),
            shape.input_size,
            shape.ft_size,
            shape.l1_hidden,
            shape.l2_size,
            shape.num_stacks,
            shape.l1_group_count,
            shape.l1_common_size,
            shape.l1_shard_size,
            batch.batch_size,
            batch.max_active,
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            batch.buckets.as_ptr(),
            weights.l0w.as_ptr(),
            weights.l0b.as_ptr(),
            weights.l1w.as_ptr(),
            weights.l1b.as_ptr(),
            l1fw,
            l1fb,
            has_l1f,
            weights.l2w.as_ptr(),
            weights.l2b.as_ptr(),
            weights.l3w.as_ptr(),
            weights.l3b.as_ptr(),
            workspace.stm_l0.as_ptr(),
            workspace.nstm_l0.as_ptr(),
            workspace.combined.as_ptr(),
            workspace.l1.as_ptr(),
            workspace.l2_input.as_ptr(),
            workspace.l2.as_ptr(),
            workspace.output.as_ptr(),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarLossKind {
    SigmoidMse,
    NnuePytorchWrm,
}

impl ScalarLossKind {
    fn as_ffi(self) -> i32 {
        match self {
            Self::SigmoidMse => 0,
            Self::NnuePytorchWrm => 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarLossHostBatch<'a> {
    pub outputs: &'a [f32],
    pub targets: &'a [f32],
    pub entry_weights: &'a [f32],
}

impl ScalarLossHostBatch<'_> {
    pub fn batch_size(self) -> usize {
        self.outputs.len()
    }

    pub fn validate(self) -> Result<()> {
        let batch_size = self.batch_size();
        if batch_size == 0 {
            return Err(CudaCppError::message("scalar loss batch must not be empty"));
        }
        expect_len("loss targets", batch_size, self.targets.len())?;
        expect_len("loss entry_weights", batch_size, self.entry_weights.len())
    }
}

#[derive(Debug)]
pub struct ScalarLossDeviceBatch {
    pub batch_size: usize,
    pub outputs: F32Buffer,
    pub targets: F32Buffer,
    pub entry_weights: F32Buffer,
}

impl ScalarLossDeviceBatch {
    pub fn from_host(ctx: &Context, batch: ScalarLossHostBatch<'_>) -> Result<Self> {
        batch.validate()?;
        Ok(Self {
            batch_size: batch.batch_size(),
            outputs: F32Buffer::from_host(ctx, batch.outputs)?,
            targets: F32Buffer::from_host(ctx, batch.targets)?,
            entry_weights: F32Buffer::from_host(ctx, batch.entry_weights)?,
        })
    }

    pub fn from_device(outputs: F32Buffer, targets: F32Buffer, entry_weights: F32Buffer) -> Result<Self> {
        let batch_size = outputs.len();
        if batch_size == 0 {
            return Err(CudaCppError::message("scalar loss batch must not be empty"));
        }
        expect_len("loss device targets", batch_size, targets.len())?;
        expect_len("loss device entry_weights", batch_size, entry_weights.len())?;
        Ok(Self { batch_size, outputs, targets, entry_weights })
    }

    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(CudaCppError::message("scalar loss batch must not be empty"));
        }
        expect_len("loss device outputs", self.batch_size, self.outputs.len())?;
        expect_len("loss device targets", self.batch_size, self.targets.len())?;
        expect_len("loss device entry_weights", self.batch_size, self.entry_weights.len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarLossWorkspaceLayout {
    pub batch_size: usize,
}

impl ScalarLossWorkspaceLayout {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    pub fn per_sample_len(self) -> usize {
        self.batch_size
    }

    pub fn mean_output_gradients_len(self) -> usize {
        self.batch_size
    }

    pub fn reduced_len(self) -> usize {
        1
    }

    fn validate(self) -> Result<()> {
        if self.batch_size == 0 {
            Err(CudaCppError::message("scalar loss workspace batch must not be empty"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct ScalarLossWorkspace {
    pub layout: ScalarLossWorkspaceLayout,
    pub per_sample: F32Buffer,
    pub mean_output_gradients: F32Buffer,
    pub weighted_sum: F32Buffer,
    pub mean: F32Buffer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarLossReadback {
    pub per_sample: Vec<f32>,
    pub mean_output_gradients: Vec<f32>,
    pub weighted_sum: f32,
    pub mean: f32,
}

impl ScalarLossWorkspace {
    pub fn new(ctx: &Context, layout: ScalarLossWorkspaceLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            per_sample: F32Buffer::new(ctx, layout.per_sample_len())?,
            mean_output_gradients: F32Buffer::new(ctx, layout.mean_output_gradients_len())?,
            weighted_sum: F32Buffer::new(ctx, layout.reduced_len())?,
            mean: F32Buffer::new(ctx, layout.reduced_len())?,
        })
    }

    fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        expect_len("loss workspace per_sample", self.layout.per_sample_len(), self.per_sample.len())?;
        expect_len(
            "loss workspace mean_output_gradients",
            self.layout.mean_output_gradients_len(),
            self.mean_output_gradients.len(),
        )?;
        expect_len("loss workspace weighted_sum", self.layout.reduced_len(), self.weighted_sum.len())?;
        expect_len("loss workspace mean", self.layout.reduced_len(), self.mean.len())
    }

    pub fn download(&self, ctx: &Context) -> Result<ScalarLossReadback> {
        let per_sample = self.per_sample.download(ctx)?;
        let mean_output_gradients = self.mean_output_gradients.download(ctx)?;
        let weighted_sum = self.weighted_sum.download(ctx)?;
        let mean = self.mean.download(ctx)?;
        Ok(ScalarLossReadback { per_sample, mean_output_gradients, weighted_sum: weighted_sum[0], mean: mean[0] })
    }
}

pub fn scalar_loss_host(
    device: i32,
    kind: ScalarLossKind,
    output_inv_scale: f32,
    batch: ScalarLossHostBatch<'_>,
) -> Result<ScalarLossReadback> {
    batch.validate()?;
    let batch_size = batch.batch_size();
    let mut per_sample = vec![0.0; batch_size];
    let mut mean_output_gradients = vec![0.0; batch_size];
    let mut weighted_sum = [0.0];
    let mut mean = [0.0];
    // SAFETY: all host slices have been length-validated against `batch_size`.
    check(unsafe {
        ffi::bulletou_cuda_cpp_scalar_loss_host(
            device,
            kind.as_ffi(),
            output_inv_scale,
            batch_size,
            batch.outputs.as_ptr(),
            batch.targets.as_ptr(),
            batch.entry_weights.as_ptr(),
            per_sample.as_mut_ptr(),
            mean_output_gradients.as_mut_ptr(),
            weighted_sum.as_mut_ptr(),
            mean.as_mut_ptr(),
        )
    })?;
    Ok(ScalarLossReadback { per_sample, mean_output_gradients, weighted_sum: weighted_sum[0], mean: mean[0] })
}

pub fn scalar_loss_device(
    ctx: &Context,
    kind: ScalarLossKind,
    output_inv_scale: f32,
    batch: &ScalarLossDeviceBatch,
    workspace: &ScalarLossWorkspace,
) -> Result<()> {
    batch.validate()?;
    scalar_loss_device_from_buffers(
        ctx,
        kind,
        output_inv_scale,
        batch.batch_size,
        &batch.outputs,
        &batch.targets,
        &batch.entry_weights,
        workspace,
    )
}

pub fn scalar_loss_device_from_buffers(
    ctx: &Context,
    kind: ScalarLossKind,
    output_inv_scale: f32,
    batch_size: usize,
    outputs: &F32Buffer,
    targets: &F32Buffer,
    entry_weights: &F32Buffer,
    workspace: &ScalarLossWorkspace,
) -> Result<()> {
    scalar_loss_device_from_buffers_with_finalize(
        ctx,
        kind,
        output_inv_scale,
        batch_size,
        outputs,
        targets,
        entry_weights,
        workspace,
        true,
    )
}

fn scalar_loss_device_from_buffers_with_finalize(
    ctx: &Context,
    kind: ScalarLossKind,
    output_inv_scale: f32,
    batch_size: usize,
    outputs: &F32Buffer,
    targets: &F32Buffer,
    entry_weights: &F32Buffer,
    workspace: &ScalarLossWorkspace,
    finalize_loss: bool,
) -> Result<()> {
    if batch_size == 0 {
        return Err(CudaCppError::message("scalar loss batch must not be empty"));
    }
    expect_len("loss device outputs", batch_size, outputs.len())?;
    expect_len("loss device targets", batch_size, targets.len())?;
    expect_len("loss device entry_weights", batch_size, entry_weights.len())?;
    workspace.validate()?;
    if workspace.layout.batch_size != batch_size {
        return Err(CudaCppError::message(format!(
            "scalar loss workspace batch mismatch: workspace={} batch={}",
            workspace.layout.batch_size, batch_size
        )));
    }
    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    check(unsafe {
        ffi::bulletou_cuda_cpp_scalar_loss_device_with_finalize(
            ctx.as_ptr(),
            kind.as_ffi(),
            output_inv_scale,
            batch_size,
            outputs.as_ptr(),
            targets.as_ptr(),
            entry_weights.as_ptr(),
            workspace.per_sample.as_ptr(),
            workspace.mean_output_gradients.as_ptr(),
            workspace.weighted_sum.as_ptr(),
            workspace.mean.as_ptr(),
            i32::from(finalize_loss),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueBackwardWorkspaceLayout {
    pub shape: NnueForwardShape,
    pub batch_size: usize,
    pub max_active: usize,
}

impl NnueBackwardWorkspaceLayout {
    pub fn new(shape: NnueForwardShape, batch_size: usize, max_active: usize) -> Self {
        Self { shape, batch_size, max_active }
    }

    pub fn hidden2_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l3)
    }

    pub fn hidden1_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2)
    }

    pub fn combined_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1).saturating_mul(2)
    }

    pub fn l0_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1)
    }

    pub fn l0w_gradients_len(self) -> usize {
        nnue_l0w_len_saturating(self.shape)
    }

    pub fn l0b_gradients_len(self) -> usize {
        self.shape.l1
    }

    pub fn l1w_gradients_len(self) -> usize {
        self.shape.l1.saturating_mul(2).saturating_mul(self.shape.l2)
    }

    pub fn l1b_gradients_len(self) -> usize {
        self.shape.l2
    }

    pub fn l2w_gradients_len(self) -> usize {
        self.shape.l2.saturating_mul(self.shape.l3)
    }

    pub fn l2b_gradients_len(self) -> usize {
        self.shape.l3
    }

    pub fn outw_gradients_len(self) -> usize {
        self.shape.l3
    }

    pub fn outb_gradients_len(self) -> usize {
        1
    }

    fn validate(self) -> Result<()> {
        validate_nnue_shape(self.shape)?;
        if self.batch_size == 0 {
            Err(CudaCppError::message("NNUE backward batch_size must be greater than zero"))
        } else if self.max_active == 0 {
            Err(CudaCppError::message("NNUE backward max_active must be greater than zero"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct NnueBackwardWorkspace {
    pub layout: NnueBackwardWorkspaceLayout,
    pub hidden2_gradients: F32Buffer,
    pub hidden1_gradients: F32Buffer,
    pub combined_gradients: F32Buffer,
    pub stm_l0_gradients: F32Buffer,
    pub nstm_l0_gradients: F32Buffer,
    pub l0w_gradients: F32Buffer,
    pub l0b_gradients: F32Buffer,
    pub l1w_gradients: F32Buffer,
    pub l1b_gradients: F32Buffer,
    pub l2w_gradients: F32Buffer,
    pub l2b_gradients: F32Buffer,
    pub outw_gradients: F32Buffer,
    pub outb_gradients: F32Buffer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NnueBackwardReadback {
    pub hidden2_gradients: Vec<f32>,
    pub hidden1_gradients: Vec<f32>,
    pub combined_gradients: Vec<f32>,
    pub stm_l0_gradients: Vec<f32>,
    pub nstm_l0_gradients: Vec<f32>,
    pub l0w_gradients: Vec<f32>,
    pub l0b_gradients: Vec<f32>,
    pub l1w_gradients: Vec<f32>,
    pub l1b_gradients: Vec<f32>,
    pub l2w_gradients: Vec<f32>,
    pub l2b_gradients: Vec<f32>,
    pub outw_gradients: Vec<f32>,
    pub outb_gradients: Vec<f32>,
}

impl NnueBackwardWorkspace {
    pub fn new(ctx: &Context, layout: NnueBackwardWorkspaceLayout) -> Result<Self> {
        layout.validate()?;
        let workspace = Self {
            layout,
            hidden2_gradients: F32Buffer::new(ctx, layout.hidden2_gradients_len())?,
            hidden1_gradients: F32Buffer::new(ctx, layout.hidden1_gradients_len())?,
            combined_gradients: F32Buffer::new(ctx, layout.combined_gradients_len())?,
            stm_l0_gradients: F32Buffer::new(ctx, layout.l0_gradients_len())?,
            nstm_l0_gradients: F32Buffer::new(ctx, layout.l0_gradients_len())?,
            l0w_gradients: F32Buffer::new(ctx, layout.l0w_gradients_len())?,
            l0b_gradients: F32Buffer::new(ctx, layout.l0b_gradients_len())?,
            l1w_gradients: F32Buffer::new(ctx, layout.l1w_gradients_len())?,
            l1b_gradients: F32Buffer::new(ctx, layout.l1b_gradients_len())?,
            l2w_gradients: F32Buffer::new(ctx, layout.l2w_gradients_len())?,
            l2b_gradients: F32Buffer::new(ctx, layout.l2b_gradients_len())?,
            outw_gradients: F32Buffer::new(ctx, layout.outw_gradients_len())?,
            outb_gradients: F32Buffer::new(ctx, layout.outb_gradients_len())?,
        };
        workspace.l0w_gradients.fill(ctx, 0.0)?;
        workspace.l0b_gradients.fill(ctx, 0.0)?;
        Ok(workspace)
    }

    fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        expect_len("backward hidden2_gradients", self.layout.hidden2_gradients_len(), self.hidden2_gradients.len())?;
        expect_len("backward hidden1_gradients", self.layout.hidden1_gradients_len(), self.hidden1_gradients.len())?;
        expect_len("backward combined_gradients", self.layout.combined_gradients_len(), self.combined_gradients.len())?;
        expect_len("backward stm_l0_gradients", self.layout.l0_gradients_len(), self.stm_l0_gradients.len())?;
        expect_len("backward nstm_l0_gradients", self.layout.l0_gradients_len(), self.nstm_l0_gradients.len())?;
        expect_len("backward l0w_gradients", self.layout.l0w_gradients_len(), self.l0w_gradients.len())?;
        expect_len("backward l0b_gradients", self.layout.l0b_gradients_len(), self.l0b_gradients.len())?;
        expect_len("backward l1w_gradients", self.layout.l1w_gradients_len(), self.l1w_gradients.len())?;
        expect_len("backward l1b_gradients", self.layout.l1b_gradients_len(), self.l1b_gradients.len())?;
        expect_len("backward l2w_gradients", self.layout.l2w_gradients_len(), self.l2w_gradients.len())?;
        expect_len("backward l2b_gradients", self.layout.l2b_gradients_len(), self.l2b_gradients.len())?;
        expect_len("backward outw_gradients", self.layout.outw_gradients_len(), self.outw_gradients.len())?;
        expect_len("backward outb_gradients", self.layout.outb_gradients_len(), self.outb_gradients.len())
    }

    pub fn download(&self, ctx: &Context) -> Result<NnueBackwardReadback> {
        Ok(NnueBackwardReadback {
            hidden2_gradients: self.hidden2_gradients.download(ctx)?,
            hidden1_gradients: self.hidden1_gradients.download(ctx)?,
            combined_gradients: self.combined_gradients.download(ctx)?,
            stm_l0_gradients: self.stm_l0_gradients.download(ctx)?,
            nstm_l0_gradients: self.nstm_l0_gradients.download(ctx)?,
            l0w_gradients: self.l0w_gradients.download(ctx)?,
            l0b_gradients: self.l0b_gradients.download(ctx)?,
            l1w_gradients: self.l1w_gradients.download(ctx)?,
            l1b_gradients: self.l1b_gradients.download(ctx)?,
            l2w_gradients: self.l2w_gradients.download(ctx)?,
            l2b_gradients: self.l2b_gradients.download(ctx)?,
            outw_gradients: self.outw_gradients.download(ctx)?,
            outb_gradients: self.outb_gradients.download(ctx)?,
        })
    }
}

pub fn nnue_backward_device(
    ctx: &Context,
    batch: &NnueForwardDeviceBatch,
    weights: &NnueForwardDeviceWeights,
    forward: &NnueForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &NnueBackwardWorkspace,
) -> Result<()> {
    nnue_backward_device_with_l0_zero(ctx, batch, weights, forward, loss, backward, true)
}

fn nnue_backward_device_reusing_zeroed_l0_gradients(
    ctx: &Context,
    batch: &NnueForwardDeviceBatch,
    weights: &NnueForwardDeviceWeights,
    forward: &NnueForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &NnueBackwardWorkspace,
) -> Result<()> {
    nnue_backward_device_with_l0_zero(ctx, batch, weights, forward, loss, backward, false)
}

fn nnue_backward_device_with_l0_zero(
    ctx: &Context,
    batch: &NnueForwardDeviceBatch,
    weights: &NnueForwardDeviceWeights,
    forward: &NnueForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &NnueBackwardWorkspace,
    zero_l0_gradients: bool,
) -> Result<()> {
    batch.validate()?;
    weights.validate()?;
    forward.validate()?;
    loss.validate()?;
    backward.validate()?;
    let shape = weights.shape;
    if forward.layout.shape != shape || backward.layout.shape != shape {
        return Err(CudaCppError::message(format!(
            "NNUE backward shape mismatch: weights={shape:?} forward={:?} backward={:?}",
            forward.layout.shape, backward.layout.shape
        )));
    }
    if forward.layout.batch_size != batch.batch_size
        || loss.layout.batch_size != batch.batch_size
        || backward.layout.batch_size != batch.batch_size
    {
        return Err(CudaCppError::message(format!(
            "NNUE backward batch mismatch: batch={} forward={} loss={} backward={}",
            batch.batch_size, forward.layout.batch_size, loss.layout.batch_size, backward.layout.batch_size
        )));
    }
    if backward.layout.max_active != batch.max_active {
        return Err(CudaCppError::message(format!(
            "NNUE backward max_active mismatch: batch={} backward={}",
            batch.max_active, backward.layout.max_active
        )));
    }

    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    check(unsafe {
        ffi::bulletou_cuda_cpp_nnue_backward_device(
            ctx.as_ptr(),
            shape.input_size,
            shape.l1,
            shape.l2,
            shape.l3,
            batch.batch_size,
            batch.max_active,
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            forward.combined.as_ptr(),
            forward.hidden1.as_ptr(),
            forward.hidden2.as_ptr(),
            forward.stm_l0.as_ptr(),
            forward.nstm_l0.as_ptr(),
            weights.l1w.as_ptr(),
            weights.l2w.as_ptr(),
            weights.outw.as_ptr(),
            loss.mean_output_gradients.as_ptr(),
            backward.hidden2_gradients.as_ptr(),
            backward.hidden1_gradients.as_ptr(),
            backward.combined_gradients.as_ptr(),
            backward.stm_l0_gradients.as_ptr(),
            backward.nstm_l0_gradients.as_ptr(),
            backward.l0w_gradients.as_ptr(),
            backward.l0b_gradients.as_ptr(),
            backward.l1w_gradients.as_ptr(),
            backward.l1b_gradients.as_ptr(),
            backward.l2w_gradients.as_ptr(),
            backward.l2b_gradients.as_ptr(),
            backward.outw_gradients.as_ptr(),
            backward.outb_gradients.as_ptr(),
            i32::from(zero_l0_gradients),
        )
    })
}

pub fn nnue_train_warmup_device(
    ctx: &Context,
    weights: &NnueForwardDeviceWeights,
    forward: &NnueForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &NnueBackwardWorkspace,
) -> Result<()> {
    weights.validate()?;
    forward.validate()?;
    loss.validate()?;
    backward.validate()?;
    let shape = weights.shape;
    if forward.layout.shape != shape || backward.layout.shape != shape {
        return Err(CudaCppError::message(format!(
            "NNUE warmup shape mismatch: weights={shape:?} forward={:?} backward={:?}",
            forward.layout.shape, backward.layout.shape
        )));
    }
    if loss.layout.batch_size != forward.layout.batch_size || backward.layout.batch_size != forward.layout.batch_size {
        return Err(CudaCppError::message(format!(
            "NNUE warmup batch mismatch: forward={} loss={} backward={}",
            forward.layout.batch_size, loss.layout.batch_size, backward.layout.batch_size
        )));
    }

    // SAFETY: all device buffers have been length-validated; the backend only warms dense-backward scratch buffers
    // and does not update trainable weights or optimizer state.
    check(unsafe {
        ffi::bulletou_cuda_cpp_nnue_train_warmup_device(
            ctx.as_ptr(),
            shape.input_size,
            shape.l1,
            shape.l2,
            shape.l3,
            forward.layout.batch_size,
            backward.layout.max_active,
            forward.combined.as_ptr(),
            forward.hidden1.as_ptr(),
            forward.hidden2.as_ptr(),
            weights.l1w.as_ptr(),
            weights.l2w.as_ptr(),
            weights.outw.as_ptr(),
            loss.mean_output_gradients.as_ptr(),
            backward.hidden2_gradients.as_ptr(),
            backward.hidden1_gradients.as_ptr(),
            backward.combined_gradients.as_ptr(),
            backward.l1w_gradients.as_ptr(),
            backward.l1b_gradients.as_ptr(),
            backward.l2w_gradients.as_ptr(),
            backward.l2b_gradients.as_ptr(),
            backward.outw_gradients.as_ptr(),
            backward.outb_gradients.as_ptr(),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnBackwardWorkspaceLayout {
    pub shape: SfnnForwardShape,
    pub batch_size: usize,
    pub max_active: usize,
}

impl SfnnBackwardWorkspaceLayout {
    pub fn new(shape: SfnnForwardShape, batch_size: usize, max_active: usize) -> Self {
        Self { shape, batch_size, max_active }
    }

    pub fn l2_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_size)
    }

    pub fn l1_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1_out())
    }

    pub fn l2_input_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_in())
    }

    pub fn combined_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn l0_gradients_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn l0w_gradients_len(self) -> usize {
        self.shape.input_size.saturating_mul(self.shape.ft_size)
    }

    pub fn l0b_gradients_len(self) -> usize {
        self.shape.ft_size
    }

    pub fn l1w_gradients_len(self) -> usize {
        self.shape.l1w_len_saturating()
    }

    pub fn l1b_gradients_len(self) -> usize {
        self.shape.num_stacks.saturating_mul(self.shape.l1_out())
    }

    pub fn l1fw_gradients_len(self) -> usize {
        self.shape.ft_size.saturating_mul(self.shape.l1_out())
    }

    pub fn l1fb_gradients_len(self) -> usize {
        self.shape.l1_out()
    }

    pub fn l2w_gradients_len(self) -> usize {
        self.shape.num_stacks.saturating_mul(self.shape.l2_size).saturating_mul(self.shape.l2_in())
    }

    pub fn l2b_gradients_len(self) -> usize {
        self.shape.num_stacks.saturating_mul(self.shape.l2_size)
    }

    pub fn l3w_gradients_len(self) -> usize {
        self.shape.num_stacks.saturating_mul(self.shape.l2_size)
    }

    pub fn l3b_gradients_len(self) -> usize {
        self.shape.num_stacks
    }

    fn validate(self) -> Result<()> {
        validate_sfnn_shape(self.shape)?;
        if self.batch_size == 0 {
            Err(CudaCppError::message("SFNN backward batch_size must be greater than zero"))
        } else if self.max_active == 0 {
            Err(CudaCppError::message("SFNN backward max_active must be greater than zero"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct SfnnBackwardWorkspace {
    pub layout: SfnnBackwardWorkspaceLayout,
    pub l2_gradients: F32Buffer,
    pub l1_gradients: F32Buffer,
    pub l2_input_gradients: F32Buffer,
    pub combined_gradients: F32Buffer,
    pub stm_l0_gradients: F32Buffer,
    pub nstm_l0_gradients: F32Buffer,
    pub stm_l0_pre_gradients: F32Buffer,
    pub nstm_l0_pre_gradients: F32Buffer,
    pub l0w_gradients: F32Buffer,
    pub l0b_gradients: F32Buffer,
    pub l1w_gradients: F32Buffer,
    pub l1b_gradients: F32Buffer,
    pub l1fw_gradients: F32Buffer,
    pub l1fb_gradients: F32Buffer,
    pub l2w_gradients: F32Buffer,
    pub l2b_gradients: F32Buffer,
    pub l3w_gradients: F32Buffer,
    pub l3b_gradients: F32Buffer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SfnnBackwardReadback {
    pub l2_gradients: Vec<f32>,
    pub l1_gradients: Vec<f32>,
    pub l2_input_gradients: Vec<f32>,
    pub combined_gradients: Vec<f32>,
    pub stm_l0_gradients: Vec<f32>,
    pub nstm_l0_gradients: Vec<f32>,
    pub stm_l0_pre_gradients: Vec<f32>,
    pub nstm_l0_pre_gradients: Vec<f32>,
    pub l0w_gradients: Vec<f32>,
    pub l0b_gradients: Vec<f32>,
    pub l1w_gradients: Vec<f32>,
    pub l1b_gradients: Vec<f32>,
    pub l1fw_gradients: Vec<f32>,
    pub l1fb_gradients: Vec<f32>,
    pub l2w_gradients: Vec<f32>,
    pub l2b_gradients: Vec<f32>,
    pub l3w_gradients: Vec<f32>,
    pub l3b_gradients: Vec<f32>,
}

impl SfnnBackwardWorkspace {
    pub fn new(ctx: &Context, layout: SfnnBackwardWorkspaceLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            l2_gradients: F32Buffer::new(ctx, layout.l2_gradients_len())?,
            l1_gradients: F32Buffer::new(ctx, layout.l1_gradients_len())?,
            l2_input_gradients: F32Buffer::new(ctx, layout.l2_input_gradients_len())?,
            combined_gradients: F32Buffer::new(ctx, layout.combined_gradients_len())?,
            stm_l0_gradients: F32Buffer::new(ctx, layout.l0_gradients_len())?,
            nstm_l0_gradients: F32Buffer::new(ctx, layout.l0_gradients_len())?,
            stm_l0_pre_gradients: F32Buffer::new(ctx, layout.l0_gradients_len())?,
            nstm_l0_pre_gradients: F32Buffer::new(ctx, layout.l0_gradients_len())?,
            l0w_gradients: F32Buffer::new(ctx, layout.l0w_gradients_len())?,
            l0b_gradients: F32Buffer::new(ctx, layout.l0b_gradients_len())?,
            l1w_gradients: F32Buffer::new(ctx, layout.l1w_gradients_len())?,
            l1b_gradients: F32Buffer::new(ctx, layout.l1b_gradients_len())?,
            l1fw_gradients: F32Buffer::new(ctx, layout.l1fw_gradients_len())?,
            l1fb_gradients: F32Buffer::new(ctx, layout.l1fb_gradients_len())?,
            l2w_gradients: F32Buffer::new(ctx, layout.l2w_gradients_len())?,
            l2b_gradients: F32Buffer::new(ctx, layout.l2b_gradients_len())?,
            l3w_gradients: F32Buffer::new(ctx, layout.l3w_gradients_len())?,
            l3b_gradients: F32Buffer::new(ctx, layout.l3b_gradients_len())?,
        })
    }

    fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        expect_len("sfnn backward l2_gradients", self.layout.l2_gradients_len(), self.l2_gradients.len())?;
        expect_len("sfnn backward l1_gradients", self.layout.l1_gradients_len(), self.l1_gradients.len())?;
        expect_len(
            "sfnn backward l2_input_gradients",
            self.layout.l2_input_gradients_len(),
            self.l2_input_gradients.len(),
        )?;
        expect_len(
            "sfnn backward combined_gradients",
            self.layout.combined_gradients_len(),
            self.combined_gradients.len(),
        )?;
        expect_len("sfnn backward stm_l0_gradients", self.layout.l0_gradients_len(), self.stm_l0_gradients.len())?;
        expect_len("sfnn backward nstm_l0_gradients", self.layout.l0_gradients_len(), self.nstm_l0_gradients.len())?;
        expect_len(
            "sfnn backward stm_l0_pre_gradients",
            self.layout.l0_gradients_len(),
            self.stm_l0_pre_gradients.len(),
        )?;
        expect_len(
            "sfnn backward nstm_l0_pre_gradients",
            self.layout.l0_gradients_len(),
            self.nstm_l0_pre_gradients.len(),
        )?;
        expect_len("sfnn backward l0w_gradients", self.layout.l0w_gradients_len(), self.l0w_gradients.len())?;
        expect_len("sfnn backward l0b_gradients", self.layout.l0b_gradients_len(), self.l0b_gradients.len())?;
        expect_len("sfnn backward l1w_gradients", self.layout.l1w_gradients_len(), self.l1w_gradients.len())?;
        expect_len("sfnn backward l1b_gradients", self.layout.l1b_gradients_len(), self.l1b_gradients.len())?;
        expect_len("sfnn backward l1fw_gradients", self.layout.l1fw_gradients_len(), self.l1fw_gradients.len())?;
        expect_len("sfnn backward l1fb_gradients", self.layout.l1fb_gradients_len(), self.l1fb_gradients.len())?;
        expect_len("sfnn backward l2w_gradients", self.layout.l2w_gradients_len(), self.l2w_gradients.len())?;
        expect_len("sfnn backward l2b_gradients", self.layout.l2b_gradients_len(), self.l2b_gradients.len())?;
        expect_len("sfnn backward l3w_gradients", self.layout.l3w_gradients_len(), self.l3w_gradients.len())?;
        expect_len("sfnn backward l3b_gradients", self.layout.l3b_gradients_len(), self.l3b_gradients.len())
    }

    pub fn download(&self, ctx: &Context) -> Result<SfnnBackwardReadback> {
        Ok(SfnnBackwardReadback {
            l2_gradients: self.l2_gradients.download(ctx)?,
            l1_gradients: self.l1_gradients.download(ctx)?,
            l2_input_gradients: self.l2_input_gradients.download(ctx)?,
            combined_gradients: self.combined_gradients.download(ctx)?,
            stm_l0_gradients: self.stm_l0_gradients.download(ctx)?,
            nstm_l0_gradients: self.nstm_l0_gradients.download(ctx)?,
            stm_l0_pre_gradients: self.stm_l0_pre_gradients.download(ctx)?,
            nstm_l0_pre_gradients: self.nstm_l0_pre_gradients.download(ctx)?,
            l0w_gradients: self.l0w_gradients.download(ctx)?,
            l0b_gradients: self.l0b_gradients.download(ctx)?,
            l1w_gradients: self.l1w_gradients.download(ctx)?,
            l1b_gradients: self.l1b_gradients.download(ctx)?,
            l1fw_gradients: self.l1fw_gradients.download(ctx)?,
            l1fb_gradients: self.l1fb_gradients.download(ctx)?,
            l2w_gradients: self.l2w_gradients.download(ctx)?,
            l2b_gradients: self.l2b_gradients.download(ctx)?,
            l3w_gradients: self.l3w_gradients.download(ctx)?,
            l3b_gradients: self.l3b_gradients.download(ctx)?,
        })
    }

    fn zero_parameter_gradients(&self, ctx: &Context) -> Result<()> {
        self.l0w_gradients.fill(ctx, 0.0)?;
        self.l0b_gradients.fill(ctx, 0.0)?;
        self.l1w_gradients.fill(ctx, 0.0)?;
        self.l1b_gradients.fill(ctx, 0.0)?;
        self.l1fw_gradients.fill(ctx, 0.0)?;
        self.l1fb_gradients.fill(ctx, 0.0)?;
        self.l2w_gradients.fill(ctx, 0.0)?;
        self.l2b_gradients.fill(ctx, 0.0)?;
        self.l3w_gradients.fill(ctx, 0.0)?;
        self.l3b_gradients.fill(ctx, 0.0)
    }
}

pub fn sfnn_backward_device(
    ctx: &Context,
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    forward: &SfnnForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &SfnnBackwardWorkspace,
) -> Result<()> {
    sfnn_backward_device_impl(ctx, batch, weights, forward, loss, backward, false)
}

pub fn sfnn_backward_train_device(
    ctx: &Context,
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    forward: &SfnnForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &SfnnBackwardWorkspace,
) -> Result<()> {
    sfnn_backward_device_impl(ctx, batch, weights, forward, loss, backward, true)
}

pub fn sfnn_backward_train_profile_device(
    ctx: &Context,
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    forward: &SfnnForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &SfnnBackwardWorkspace,
) -> Result<SfnnBackwardStageProfile> {
    batch.validate()?;
    weights.validate()?;
    forward.validate()?;
    loss.validate()?;
    backward.validate()?;
    let shape = weights.shape;
    if forward.layout.shape != shape || backward.layout.shape != shape {
        return Err(CudaCppError::message(format!(
            "SFNN backward shape mismatch: weights={shape:?} forward={:?} backward={:?}",
            forward.layout.shape, backward.layout.shape
        )));
    }
    if forward.layout.batch_size != batch.batch_size
        || loss.layout.batch_size != batch.batch_size
        || backward.layout.batch_size != batch.batch_size
    {
        return Err(CudaCppError::message(format!(
            "SFNN backward batch mismatch: batch={} forward={} loss={} backward={}",
            batch.batch_size, forward.layout.batch_size, loss.layout.batch_size, backward.layout.batch_size
        )));
    }
    if backward.layout.max_active != batch.max_active {
        return Err(CudaCppError::message(format!(
            "SFNN backward max_active mismatch: batch={} backward={}",
            batch.max_active, backward.layout.max_active
        )));
    }
    let (l1fw, has_l1f) = match (&weights.l1fw, &weights.l1fb) {
        (Some(l1fw), Some(_)) => (l1fw.as_ptr(), 1),
        (None, None) => (std::ptr::null_mut(), 0),
        _ => return Err(CudaCppError::message("SFNN factorized L1 state is partial")),
    };

    let mut profile_ms = [0.0f32; 7];
    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    check(unsafe {
        ffi::bulletou_cuda_cpp_sfnn_backward_train_profile_device(
            ctx.as_ptr(),
            shape.input_size,
            shape.ft_size,
            shape.l1_hidden,
            shape.l2_size,
            shape.num_stacks,
            shape.l1_group_count,
            shape.l1_common_size,
            shape.l1_shard_size,
            batch.batch_size,
            batch.max_active,
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            batch.buckets.as_ptr(),
            forward.stm_l0.as_ptr(),
            forward.nstm_l0.as_ptr(),
            forward.combined.as_ptr(),
            forward.l1.as_ptr(),
            forward.l2_input.as_ptr(),
            forward.l2.as_ptr(),
            weights.l1w.as_ptr(),
            l1fw,
            has_l1f,
            weights.l2w.as_ptr(),
            weights.l3w.as_ptr(),
            loss.mean_output_gradients.as_ptr(),
            backward.l2_gradients.as_ptr(),
            backward.l1_gradients.as_ptr(),
            backward.l2_input_gradients.as_ptr(),
            backward.combined_gradients.as_ptr(),
            backward.stm_l0_gradients.as_ptr(),
            backward.nstm_l0_gradients.as_ptr(),
            backward.stm_l0_pre_gradients.as_ptr(),
            backward.nstm_l0_pre_gradients.as_ptr(),
            backward.l0w_gradients.as_ptr(),
            backward.l0b_gradients.as_ptr(),
            backward.l1w_gradients.as_ptr(),
            backward.l1b_gradients.as_ptr(),
            backward.l1fw_gradients.as_ptr(),
            backward.l1fb_gradients.as_ptr(),
            backward.l2w_gradients.as_ptr(),
            backward.l2b_gradients.as_ptr(),
            backward.l3w_gradients.as_ptr(),
            backward.l3b_gradients.as_ptr(),
            1,
            profile_ms.as_mut_ptr(),
            profile_ms.len(),
        )
    })?;
    Ok(SfnnBackwardStageProfile {
        zero_ms: profile_ms[0],
        l3_ms: profile_ms[1],
        l2_ms: profile_ms[2],
        l2_input_ms: profile_ms[3],
        l1_ms: profile_ms[4],
        l0_ms: profile_ms[5],
        total_ms: profile_ms[6],
    })
}

fn sfnn_backward_device_impl(
    ctx: &Context,
    batch: &SfnnForwardDeviceBatch,
    weights: &SfnnForwardDeviceWeights,
    forward: &SfnnForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &SfnnBackwardWorkspace,
    use_train_entry: bool,
) -> Result<()> {
    batch.validate()?;
    weights.validate()?;
    forward.validate()?;
    loss.validate()?;
    backward.validate()?;
    let shape = weights.shape;
    if forward.layout.shape != shape || backward.layout.shape != shape {
        return Err(CudaCppError::message(format!(
            "SFNN backward shape mismatch: weights={shape:?} forward={:?} backward={:?}",
            forward.layout.shape, backward.layout.shape
        )));
    }
    if forward.layout.batch_size != batch.batch_size
        || loss.layout.batch_size != batch.batch_size
        || backward.layout.batch_size != batch.batch_size
    {
        return Err(CudaCppError::message(format!(
            "SFNN backward batch mismatch: batch={} forward={} loss={} backward={}",
            batch.batch_size, forward.layout.batch_size, loss.layout.batch_size, backward.layout.batch_size
        )));
    }
    if backward.layout.max_active != batch.max_active {
        return Err(CudaCppError::message(format!(
            "SFNN backward max_active mismatch: batch={} backward={}",
            batch.max_active, backward.layout.max_active
        )));
    }
    let (l1fw, has_l1f) = match (&weights.l1fw, &weights.l1fb) {
        (Some(l1fw), Some(_)) => (l1fw.as_ptr(), 1),
        (None, None) => (std::ptr::null_mut(), 0),
        _ => return Err(CudaCppError::message("SFNN factorized L1 state is partial")),
    };

    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    let rc = unsafe {
        if use_train_entry {
            ffi::bulletou_cuda_cpp_sfnn_backward_train_device(
                ctx.as_ptr(),
                shape.input_size,
                shape.ft_size,
                shape.l1_hidden,
                shape.l2_size,
                shape.num_stacks,
                shape.l1_group_count,
                shape.l1_common_size,
                shape.l1_shard_size,
                batch.batch_size,
                batch.max_active,
                batch.stm_indices.as_ptr(),
                batch.nstm_indices.as_ptr(),
                batch.buckets.as_ptr(),
                forward.stm_l0.as_ptr(),
                forward.nstm_l0.as_ptr(),
                forward.combined.as_ptr(),
                forward.l1.as_ptr(),
                forward.l2_input.as_ptr(),
                forward.l2.as_ptr(),
                weights.l1w.as_ptr(),
                l1fw,
                has_l1f,
                weights.l2w.as_ptr(),
                weights.l3w.as_ptr(),
                loss.mean_output_gradients.as_ptr(),
                backward.l2_gradients.as_ptr(),
                backward.l1_gradients.as_ptr(),
                backward.l2_input_gradients.as_ptr(),
                backward.combined_gradients.as_ptr(),
                backward.stm_l0_gradients.as_ptr(),
                backward.nstm_l0_gradients.as_ptr(),
                backward.stm_l0_pre_gradients.as_ptr(),
                backward.nstm_l0_pre_gradients.as_ptr(),
                backward.l0w_gradients.as_ptr(),
                backward.l0b_gradients.as_ptr(),
                backward.l1w_gradients.as_ptr(),
                backward.l1b_gradients.as_ptr(),
                backward.l1fw_gradients.as_ptr(),
                backward.l1fb_gradients.as_ptr(),
                backward.l2w_gradients.as_ptr(),
                backward.l2b_gradients.as_ptr(),
                backward.l3w_gradients.as_ptr(),
                backward.l3b_gradients.as_ptr(),
                0,
            )
        } else {
            ffi::bulletou_cuda_cpp_sfnn_backward_device(
                ctx.as_ptr(),
                shape.input_size,
                shape.ft_size,
                shape.l1_hidden,
                shape.l2_size,
                shape.num_stacks,
                shape.l1_group_count,
                shape.l1_common_size,
                shape.l1_shard_size,
                batch.batch_size,
                batch.max_active,
                batch.stm_indices.as_ptr(),
                batch.nstm_indices.as_ptr(),
                batch.buckets.as_ptr(),
                forward.stm_l0.as_ptr(),
                forward.nstm_l0.as_ptr(),
                forward.combined.as_ptr(),
                forward.l1.as_ptr(),
                forward.l2_input.as_ptr(),
                forward.l2.as_ptr(),
                weights.l1w.as_ptr(),
                l1fw,
                has_l1f,
                weights.l2w.as_ptr(),
                weights.l3w.as_ptr(),
                loss.mean_output_gradients.as_ptr(),
                backward.l2_gradients.as_ptr(),
                backward.l1_gradients.as_ptr(),
                backward.l2_input_gradients.as_ptr(),
                backward.combined_gradients.as_ptr(),
                backward.stm_l0_gradients.as_ptr(),
                backward.nstm_l0_gradients.as_ptr(),
                backward.stm_l0_pre_gradients.as_ptr(),
                backward.nstm_l0_pre_gradients.as_ptr(),
                backward.l0w_gradients.as_ptr(),
                backward.l0b_gradients.as_ptr(),
                backward.l1w_gradients.as_ptr(),
                backward.l1b_gradients.as_ptr(),
                backward.l1fw_gradients.as_ptr(),
                backward.l1fb_gradients.as_ptr(),
                backward.l2w_gradients.as_ptr(),
                backward.l2b_gradients.as_ptr(),
                backward.l3w_gradients.as_ptr(),
                backward.l3b_gradients.as_ptr(),
            )
        }
    };
    check(rc)
}

#[derive(Debug)]
pub struct RangerParamState {
    pub momentum: F32Buffer,
    pub velocity: F32Buffer,
    pub slow_params: F32Buffer,
}

#[derive(Debug, Clone, Copy)]
pub struct RangerParamHostState<'a> {
    pub momentum: &'a [f32],
    pub velocity: &'a [f32],
    pub slow_params: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangerParamStateReadback {
    pub momentum: Vec<f32>,
    pub velocity: Vec<f32>,
    pub slow_params: Vec<f32>,
}

impl RangerParamHostState<'_> {
    fn validate(self, len: usize, name: &'static str) -> Result<()> {
        expect_len(name, len, self.momentum.len())?;
        expect_len(name, len, self.velocity.len())?;
        expect_len(name, len, self.slow_params.len())
    }
}

impl RangerParamState {
    pub fn from_host_weights(ctx: &Context, weights: &[f32]) -> Result<Self> {
        let momentum = F32Buffer::new(ctx, weights.len())?;
        momentum.fill(ctx, 0.0)?;
        let velocity = F32Buffer::new(ctx, weights.len())?;
        velocity.fill(ctx, 0.0)?;
        let slow_params = F32Buffer::from_host(ctx, weights)?;
        Ok(Self { momentum, velocity, slow_params })
    }

    pub fn from_host_state(ctx: &Context, len: usize, state: RangerParamHostState<'_>) -> Result<Self> {
        state.validate(len, "optimizer state")?;
        Ok(Self {
            momentum: F32Buffer::from_host(ctx, state.momentum)?,
            velocity: F32Buffer::from_host(ctx, state.velocity)?,
            slow_params: F32Buffer::from_host(ctx, state.slow_params)?,
        })
    }

    pub fn download(&self, ctx: &Context) -> Result<RangerParamStateReadback> {
        Ok(RangerParamStateReadback {
            momentum: self.momentum.download(ctx)?,
            velocity: self.velocity.download(ctx)?,
            slow_params: self.slow_params.download(ctx)?,
        })
    }

    fn validate(&self, len: usize, name: &'static str) -> Result<()> {
        expect_len(name, len, self.momentum.len())?;
        expect_len(name, len, self.velocity.len())?;
        expect_len(name, len, self.slow_params.len())
    }
}

#[derive(Debug)]
pub struct NnueRangerOptimizerStates {
    pub l0w: RangerParamState,
    pub l0b: RangerParamState,
    pub l1w: RangerParamState,
    pub l1b: RangerParamState,
    pub l2w: RangerParamState,
    pub l2b: RangerParamState,
    pub outw: RangerParamState,
    pub outb: RangerParamState,
}

#[derive(Debug, Clone, Copy)]
pub struct NnueRangerOptimizerHostStates<'a> {
    pub l0w: RangerParamHostState<'a>,
    pub l0b: RangerParamHostState<'a>,
    pub l1w: RangerParamHostState<'a>,
    pub l1b: RangerParamHostState<'a>,
    pub l2w: RangerParamHostState<'a>,
    pub l2b: RangerParamHostState<'a>,
    pub outw: RangerParamHostState<'a>,
    pub outb: RangerParamHostState<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NnueRangerOptimizerStatesReadback {
    pub l0w: RangerParamStateReadback,
    pub l0b: RangerParamStateReadback,
    pub l1w: RangerParamStateReadback,
    pub l1b: RangerParamStateReadback,
    pub l2w: RangerParamStateReadback,
    pub l2b: RangerParamStateReadback,
    pub outw: RangerParamStateReadback,
    pub outb: RangerParamStateReadback,
}

impl NnueRangerOptimizerStates {
    pub fn from_host_weights(ctx: &Context, weights: NnueForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            l0w: RangerParamState::from_host_weights(ctx, weights.l0w)?,
            l0b: RangerParamState::from_host_weights(ctx, weights.l0b)?,
            l1w: RangerParamState::from_host_weights(ctx, weights.l1w)?,
            l1b: RangerParamState::from_host_weights(ctx, weights.l1b)?,
            l2w: RangerParamState::from_host_weights(ctx, weights.l2w)?,
            l2b: RangerParamState::from_host_weights(ctx, weights.l2b)?,
            outw: RangerParamState::from_host_weights(ctx, weights.outw)?,
            outb: RangerParamState::from_host_weights(ctx, weights.outb)?,
        })
    }

    pub fn from_host_states(
        ctx: &Context,
        shape: NnueForwardShape,
        states: NnueRangerOptimizerHostStates<'_>,
    ) -> Result<Self> {
        validate_nnue_shape(shape)?;
        let l0w_len = nnue_l0w_len(shape)?;
        let l1w_len = checked_product("l1w", &[shape.l1, 2, shape.l2])?;
        let l2w_len = checked_product("l2w", &[shape.l2, shape.l3])?;
        Ok(Self {
            l0w: RangerParamState::from_host_state(ctx, l0w_len, states.l0w)?,
            l0b: RangerParamState::from_host_state(ctx, shape.l1, states.l0b)?,
            l1w: RangerParamState::from_host_state(ctx, l1w_len, states.l1w)?,
            l1b: RangerParamState::from_host_state(ctx, shape.l2, states.l1b)?,
            l2w: RangerParamState::from_host_state(ctx, l2w_len, states.l2w)?,
            l2b: RangerParamState::from_host_state(ctx, shape.l3, states.l2b)?,
            outw: RangerParamState::from_host_state(ctx, shape.l3, states.outw)?,
            outb: RangerParamState::from_host_state(ctx, 1, states.outb)?,
        })
    }

    pub fn download(&self, ctx: &Context) -> Result<NnueRangerOptimizerStatesReadback> {
        Ok(NnueRangerOptimizerStatesReadback {
            l0w: self.l0w.download(ctx)?,
            l0b: self.l0b.download(ctx)?,
            l1w: self.l1w.download(ctx)?,
            l1b: self.l1b.download(ctx)?,
            l2w: self.l2w.download(ctx)?,
            l2b: self.l2b.download(ctx)?,
            outw: self.outw.download(ctx)?,
            outb: self.outb.download(ctx)?,
        })
    }

    fn validate(&self, shape: NnueForwardShape) -> Result<()> {
        self.l0w.validate(nnue_l0w_len(shape)?, "optimizer l0w")?;
        self.l0b.validate(shape.l1, "optimizer l0b")?;
        self.l1w.validate(checked_product("l1w", &[shape.l1, 2, shape.l2])?, "optimizer l1w")?;
        self.l1b.validate(shape.l2, "optimizer l1b")?;
        self.l2w.validate(checked_product("l2w", &[shape.l2, shape.l3])?, "optimizer l2w")?;
        self.l2b.validate(shape.l3, "optimizer l2b")?;
        self.outw.validate(shape.l3, "optimizer outw")?;
        self.outb.validate(1, "optimizer outb")
    }
}

#[derive(Debug)]
pub struct SfnnRangerOptimizerStates {
    pub l0w: RangerParamState,
    pub l0b: RangerParamState,
    pub l1w: RangerParamState,
    pub l1b: RangerParamState,
    pub l1fw: Option<RangerParamState>,
    pub l1fb: Option<RangerParamState>,
    pub l2w: RangerParamState,
    pub l2b: RangerParamState,
    pub l3w: RangerParamState,
    pub l3b: RangerParamState,
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnRangerOptimizerHostStates<'a> {
    pub l0w: RangerParamHostState<'a>,
    pub l0b: RangerParamHostState<'a>,
    pub l1w: RangerParamHostState<'a>,
    pub l1b: RangerParamHostState<'a>,
    pub l1fw: Option<RangerParamHostState<'a>>,
    pub l1fb: Option<RangerParamHostState<'a>>,
    pub l2w: RangerParamHostState<'a>,
    pub l2b: RangerParamHostState<'a>,
    pub l3w: RangerParamHostState<'a>,
    pub l3b: RangerParamHostState<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SfnnRangerOptimizerStatesReadback {
    pub l0w: RangerParamStateReadback,
    pub l0b: RangerParamStateReadback,
    pub l1w: RangerParamStateReadback,
    pub l1b: RangerParamStateReadback,
    pub l1fw: Option<RangerParamStateReadback>,
    pub l1fb: Option<RangerParamStateReadback>,
    pub l2w: RangerParamStateReadback,
    pub l2b: RangerParamStateReadback,
    pub l3w: RangerParamStateReadback,
    pub l3b: RangerParamStateReadback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KpptTableShape {
    pub input_size: usize,
}

impl KpptTableShape {
    pub fn table_w_len(self) -> usize {
        self.input_size
    }

    pub fn table_b_len(self) -> usize {
        1
    }

    pub fn outw_len(self) -> usize {
        2
    }

    pub fn outb_len(self) -> usize {
        1
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KpptTableForwardHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl KpptTableForwardHostBatch<'_> {
    pub fn validate(self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(CudaCppError::message("KPPT table batch_size must be greater than zero"));
        }
        if self.max_active == 0 {
            return Err(CudaCppError::message("KPPT table max_active must be greater than zero"));
        }
        let sparse_len = self
            .batch_size
            .checked_mul(self.max_active)
            .ok_or_else(|| CudaCppError::message("KPPT table sparse batch length overflow"))?;
        expect_len("kppt table stm_indices", sparse_len, self.stm_indices.len())?;
        expect_len("kppt table nstm_indices", sparse_len, self.nstm_indices.len())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KpptTableForwardHostWeights<'a> {
    pub shape: KpptTableShape,
    pub table_w: &'a [f32],
    pub table_b: &'a [f32],
    pub outw: &'a [f32],
    pub outb: &'a [f32],
}

impl KpptTableForwardHostWeights<'_> {
    pub fn validate(self) -> Result<()> {
        validate_kppt_table_shape(self.shape)?;
        expect_len("kppt table_w", self.shape.table_w_len(), self.table_w.len())?;
        expect_len("kppt table_b", self.shape.table_b_len(), self.table_b.len())?;
        expect_len("kppt outw", self.shape.outw_len(), self.outw.len())?;
        expect_len("kppt outb", self.shape.outb_len(), self.outb.len())
    }
}

#[derive(Debug)]
pub struct KpptTableForwardDeviceBatch {
    pub batch_size: usize,
    pub max_active: usize,
    pub stm_indices: I32Buffer,
    pub nstm_indices: I32Buffer,
}

impl KpptTableForwardDeviceBatch {
    pub fn from_host(ctx: &Context, batch: KpptTableForwardHostBatch<'_>) -> Result<Self> {
        batch.validate()?;
        Ok(Self {
            batch_size: batch.batch_size,
            max_active: batch.max_active,
            stm_indices: I32Buffer::from_host(ctx, batch.stm_indices)?,
            nstm_indices: I32Buffer::from_host(ctx, batch.nstm_indices)?,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(CudaCppError::message("KPPT table device batch_size must be greater than zero"));
        }
        if self.max_active == 0 {
            return Err(CudaCppError::message("KPPT table device max_active must be greater than zero"));
        }
        let sparse_len = self
            .batch_size
            .checked_mul(self.max_active)
            .ok_or_else(|| CudaCppError::message("KPPT table sparse batch length overflow"))?;
        expect_len("kppt table device stm_indices", sparse_len, self.stm_indices.len())?;
        expect_len("kppt table device nstm_indices", sparse_len, self.nstm_indices.len())
    }
}

#[derive(Debug)]
pub struct KpptTableForwardDeviceWeights {
    pub shape: KpptTableShape,
    pub table_w: F32Buffer,
    pub table_b: F32Buffer,
    pub outw: F32Buffer,
    pub outb: F32Buffer,
}

impl KpptTableForwardDeviceWeights {
    pub fn from_host(ctx: &Context, weights: KpptTableForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            shape: weights.shape,
            table_w: F32Buffer::from_host(ctx, weights.table_w)?,
            table_b: F32Buffer::from_host(ctx, weights.table_b)?,
            outw: F32Buffer::from_host(ctx, weights.outw)?,
            outb: F32Buffer::from_host(ctx, weights.outb)?,
        })
    }

    fn validate(&self) -> Result<()> {
        validate_kppt_table_shape(self.shape)?;
        expect_len("kppt device table_w", self.shape.table_w_len(), self.table_w.len())?;
        expect_len("kppt device table_b", self.shape.table_b_len(), self.table_b.len())?;
        expect_len("kppt device outw", self.shape.outw_len(), self.outw.len())?;
        expect_len("kppt device outb", self.shape.outb_len(), self.outb.len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KpptTableForwardWorkspaceLayout {
    pub shape: KpptTableShape,
    pub batch_size: usize,
}

impl KpptTableForwardWorkspaceLayout {
    pub fn new(shape: KpptTableShape, batch_size: usize) -> Self {
        Self { shape, batch_size }
    }

    pub fn stm_eval_len(self) -> usize {
        self.batch_size
    }

    pub fn nstm_eval_len(self) -> usize {
        self.batch_size
    }

    pub fn output_len(self) -> usize {
        self.batch_size
    }

    fn validate(self) -> Result<()> {
        validate_kppt_table_shape(self.shape)?;
        if self.batch_size == 0 {
            Err(CudaCppError::message("KPPT table workspace batch_size must be greater than zero"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct KpptTableForwardWorkspace {
    pub layout: KpptTableForwardWorkspaceLayout,
    pub stm_eval: F32Buffer,
    pub nstm_eval: F32Buffer,
    pub output: F32Buffer,
}

impl KpptTableForwardWorkspace {
    pub fn new(ctx: &Context, layout: KpptTableForwardWorkspaceLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            stm_eval: F32Buffer::new(ctx, layout.stm_eval_len())?,
            nstm_eval: F32Buffer::new(ctx, layout.nstm_eval_len())?,
            output: F32Buffer::new(ctx, layout.output_len())?,
        })
    }

    fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        expect_len("kppt workspace stm_eval", self.layout.stm_eval_len(), self.stm_eval.len())?;
        expect_len("kppt workspace nstm_eval", self.layout.nstm_eval_len(), self.nstm_eval.len())?;
        expect_len("kppt workspace output", self.layout.output_len(), self.output.len())
    }
}

pub fn kppt_table_forward_device(
    ctx: &Context,
    batch: &KpptTableForwardDeviceBatch,
    weights: &KpptTableForwardDeviceWeights,
    workspace: &KpptTableForwardWorkspace,
) -> Result<()> {
    batch.validate()?;
    weights.validate()?;
    workspace.validate()?;
    if workspace.layout.shape != weights.shape {
        return Err(CudaCppError::message(format!(
            "KPPT table workspace shape mismatch: workspace={:?} weights={:?}",
            workspace.layout.shape, weights.shape
        )));
    }
    if workspace.layout.batch_size != batch.batch_size {
        return Err(CudaCppError::message(format!(
            "KPPT table workspace batch mismatch: workspace={} batch={}",
            workspace.layout.batch_size, batch.batch_size
        )));
    }
    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    check(unsafe {
        ffi::bulletou_cuda_cpp_kppt_forward_device(
            ctx.as_ptr(),
            weights.shape.input_size,
            batch.batch_size,
            batch.max_active,
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            weights.table_w.as_ptr(),
            weights.table_b.as_ptr(),
            weights.outw.as_ptr(),
            weights.outb.as_ptr(),
            workspace.stm_eval.as_ptr(),
            workspace.nstm_eval.as_ptr(),
            workspace.output.as_ptr(),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KpptTableBackwardWorkspaceLayout {
    pub shape: KpptTableShape,
    pub batch_size: usize,
    pub max_active: usize,
}

impl KpptTableBackwardWorkspaceLayout {
    pub fn new(shape: KpptTableShape, batch_size: usize, max_active: usize) -> Self {
        Self { shape, batch_size, max_active }
    }

    pub fn table_w_gradients_len(self) -> usize {
        self.shape.table_w_len()
    }

    pub fn table_b_gradients_len(self) -> usize {
        self.shape.table_b_len()
    }

    pub fn outw_gradients_len(self) -> usize {
        self.shape.outw_len()
    }

    pub fn outb_gradients_len(self) -> usize {
        self.shape.outb_len()
    }

    fn validate(self) -> Result<()> {
        validate_kppt_table_shape(self.shape)?;
        if self.batch_size == 0 {
            Err(CudaCppError::message("KPPT table backward batch_size must be greater than zero"))
        } else if self.max_active == 0 {
            Err(CudaCppError::message("KPPT table backward max_active must be greater than zero"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct KpptTableBackwardWorkspace {
    pub layout: KpptTableBackwardWorkspaceLayout,
    pub table_w_gradients: F32Buffer,
    pub table_b_gradients: F32Buffer,
    pub outw_gradients: F32Buffer,
    pub outb_gradients: F32Buffer,
}

impl KpptTableBackwardWorkspace {
    pub fn new(ctx: &Context, layout: KpptTableBackwardWorkspaceLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            layout,
            table_w_gradients: F32Buffer::new(ctx, layout.table_w_gradients_len())?,
            table_b_gradients: F32Buffer::new(ctx, layout.table_b_gradients_len())?,
            outw_gradients: F32Buffer::new(ctx, layout.outw_gradients_len())?,
            outb_gradients: F32Buffer::new(ctx, layout.outb_gradients_len())?,
        })
    }

    fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        expect_len(
            "kppt backward table_w_gradients",
            self.layout.table_w_gradients_len(),
            self.table_w_gradients.len(),
        )?;
        expect_len(
            "kppt backward table_b_gradients",
            self.layout.table_b_gradients_len(),
            self.table_b_gradients.len(),
        )?;
        expect_len("kppt backward outw_gradients", self.layout.outw_gradients_len(), self.outw_gradients.len())?;
        expect_len("kppt backward outb_gradients", self.layout.outb_gradients_len(), self.outb_gradients.len())
    }
}

pub fn kppt_table_backward_device(
    ctx: &Context,
    batch: &KpptTableForwardDeviceBatch,
    weights: &KpptTableForwardDeviceWeights,
    forward: &KpptTableForwardWorkspace,
    loss: &ScalarLossWorkspace,
    backward: &KpptTableBackwardWorkspace,
) -> Result<()> {
    batch.validate()?;
    weights.validate()?;
    forward.validate()?;
    loss.validate()?;
    backward.validate()?;
    let shape = weights.shape;
    if forward.layout.shape != shape || backward.layout.shape != shape {
        return Err(CudaCppError::message(format!(
            "KPPT table backward shape mismatch: weights={shape:?} forward={:?} backward={:?}",
            forward.layout.shape, backward.layout.shape
        )));
    }
    if forward.layout.batch_size != batch.batch_size
        || loss.layout.batch_size != batch.batch_size
        || backward.layout.batch_size != batch.batch_size
    {
        return Err(CudaCppError::message(format!(
            "KPPT table backward batch mismatch: batch={} forward={} loss={} backward={}",
            batch.batch_size, forward.layout.batch_size, loss.layout.batch_size, backward.layout.batch_size
        )));
    }
    if backward.layout.max_active != batch.max_active {
        return Err(CudaCppError::message(format!(
            "KPPT table backward max_active mismatch: batch={} backward={}",
            batch.max_active, backward.layout.max_active
        )));
    }
    // SAFETY: all device buffers have been length-validated; backend validates device ownership.
    check(unsafe {
        ffi::bulletou_cuda_cpp_kppt_backward_device(
            ctx.as_ptr(),
            shape.input_size,
            batch.batch_size,
            batch.max_active,
            batch.stm_indices.as_ptr(),
            batch.nstm_indices.as_ptr(),
            forward.stm_eval.as_ptr(),
            forward.nstm_eval.as_ptr(),
            weights.outw.as_ptr(),
            loss.mean_output_gradients.as_ptr(),
            backward.table_w_gradients.as_ptr(),
            backward.table_b_gradients.as_ptr(),
            backward.outw_gradients.as_ptr(),
            backward.outb_gradients.as_ptr(),
        )
    })
}

#[derive(Debug, Clone, Copy)]
pub struct KpptTableTrainStepHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub targets: &'a [f32],
    pub entry_weights: &'a [f32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl<'a> KpptTableTrainStepHostBatch<'a> {
    fn forward_batch(self) -> KpptTableForwardHostBatch<'a> {
        KpptTableForwardHostBatch {
            stm_indices: self.stm_indices,
            nstm_indices: self.nstm_indices,
            batch_size: self.batch_size,
            max_active: self.max_active,
        }
    }

    pub fn validate(self) -> Result<()> {
        self.forward_batch().validate()?;
        expect_len("kppt train targets", self.batch_size, self.targets.len())?;
        expect_len("kppt train entry_weights", self.batch_size, self.entry_weights.len())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KpptTableTrainWeightsReadback {
    pub table_w: Vec<f32>,
    pub table_b: Vec<f32>,
    pub outw: Vec<f32>,
    pub outb: Vec<f32>,
}

#[derive(Debug)]
pub struct KpptTableRangerOptimizerStates {
    pub table_w: RangerParamState,
    pub table_b: RangerParamState,
    pub outw: RangerParamState,
    pub outb: RangerParamState,
}

#[derive(Debug, Clone, Copy)]
pub struct KpptTableRangerOptimizerHostStates<'a> {
    pub table_w: RangerParamHostState<'a>,
    pub table_b: RangerParamHostState<'a>,
    pub outw: RangerParamHostState<'a>,
    pub outb: RangerParamHostState<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KpptTableRangerOptimizerStatesReadback {
    pub table_w: RangerParamStateReadback,
    pub table_b: RangerParamStateReadback,
    pub outw: RangerParamStateReadback,
    pub outb: RangerParamStateReadback,
}

impl KpptTableRangerOptimizerStates {
    pub fn from_host_weights(ctx: &Context, weights: KpptTableForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            table_w: RangerParamState::from_host_weights(ctx, weights.table_w)?,
            table_b: RangerParamState::from_host_weights(ctx, weights.table_b)?,
            outw: RangerParamState::from_host_weights(ctx, weights.outw)?,
            outb: RangerParamState::from_host_weights(ctx, weights.outb)?,
        })
    }

    pub fn from_host_states(
        ctx: &Context,
        shape: KpptTableShape,
        states: KpptTableRangerOptimizerHostStates<'_>,
    ) -> Result<Self> {
        validate_kppt_table_shape(shape)?;
        Ok(Self {
            table_w: RangerParamState::from_host_state(ctx, shape.table_w_len(), states.table_w)?,
            table_b: RangerParamState::from_host_state(ctx, shape.table_b_len(), states.table_b)?,
            outw: RangerParamState::from_host_state(ctx, shape.outw_len(), states.outw)?,
            outb: RangerParamState::from_host_state(ctx, shape.outb_len(), states.outb)?,
        })
    }

    pub fn download(&self, ctx: &Context) -> Result<KpptTableRangerOptimizerStatesReadback> {
        Ok(KpptTableRangerOptimizerStatesReadback {
            table_w: self.table_w.download(ctx)?,
            table_b: self.table_b.download(ctx)?,
            outw: self.outw.download(ctx)?,
            outb: self.outb.download(ctx)?,
        })
    }

    fn validate(&self, shape: KpptTableShape) -> Result<()> {
        self.table_w.validate(shape.table_w_len(), "optimizer kppt table_w")?;
        self.table_b.validate(shape.table_b_len(), "optimizer kppt table_b")?;
        self.outw.validate(shape.outw_len(), "optimizer kppt outw")?;
        self.outb.validate(shape.outb_len(), "optimizer kppt outb")
    }
}

#[derive(Debug)]
pub struct KpptTableTrainStepRunner {
    pub shape: KpptTableShape,
    pub batch_size: usize,
    pub max_active: usize,
    pub device_batch: KpptTableForwardDeviceBatch,
    pub targets: F32Buffer,
    pub entry_weights: F32Buffer,
    pub weights: KpptTableForwardDeviceWeights,
    pub optimizer_states: KpptTableRangerOptimizerStates,
    pub forward_workspace: KpptTableForwardWorkspace,
    pub loss_workspace: ScalarLossWorkspace,
    pub backward_workspace: KpptTableBackwardWorkspace,
}

impl KpptTableTrainStepRunner {
    pub fn new(
        ctx: &Context,
        initial_weights: KpptTableForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        let optimizer_states = KpptTableRangerOptimizerStates::from_host_weights(ctx, initial_weights)?;
        Self::with_device_optimizer_states(ctx, initial_weights, optimizer_states, batch_size, max_active)
    }

    pub fn with_optimizer_states(
        ctx: &Context,
        initial_weights: KpptTableForwardHostWeights<'_>,
        optimizer_states: KpptTableRangerOptimizerHostStates<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        let optimizer_states =
            KpptTableRangerOptimizerStates::from_host_states(ctx, initial_weights.shape, optimizer_states)?;
        Self::with_device_optimizer_states(ctx, initial_weights, optimizer_states, batch_size, max_active)
    }

    fn with_device_optimizer_states(
        ctx: &Context,
        initial_weights: KpptTableForwardHostWeights<'_>,
        optimizer_states: KpptTableRangerOptimizerStates,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        initial_weights.validate()?;
        if batch_size == 0 {
            return Err(CudaCppError::message("KPPT table train-step batch_size must be greater than zero"));
        }
        if max_active == 0 {
            return Err(CudaCppError::message("KPPT table train-step max_active must be greater than zero"));
        }

        let shape = initial_weights.shape;
        let sparse_len = batch_size
            .checked_mul(max_active)
            .ok_or_else(|| CudaCppError::message("KPPT table train-step sparse length overflow"))?;
        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_batch: KpptTableForwardDeviceBatch {
                batch_size,
                max_active,
                stm_indices: I32Buffer::new(ctx, sparse_len)?,
                nstm_indices: I32Buffer::new(ctx, sparse_len)?,
            },
            targets: F32Buffer::new(ctx, batch_size)?,
            entry_weights: F32Buffer::new(ctx, batch_size)?,
            weights: KpptTableForwardDeviceWeights::from_host(ctx, initial_weights)?,
            optimizer_states,
            forward_workspace: KpptTableForwardWorkspace::new(
                ctx,
                KpptTableForwardWorkspaceLayout::new(shape, batch_size),
            )?,
            loss_workspace: ScalarLossWorkspace::new(ctx, ScalarLossWorkspaceLayout::new(batch_size))?,
            backward_workspace: KpptTableBackwardWorkspace::new(
                ctx,
                KpptTableBackwardWorkspaceLayout::new(shape, batch_size, max_active),
            )?,
        })
    }

    pub fn step(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: KpptTableTrainStepHostBatch<'_>,
    ) -> Result<ScalarLossReadback> {
        self.step_no_readback(ctx, params, loss_kind, output_inv_scale, batch)?;
        self.read_loss(ctx)
    }

    pub fn step_no_readback(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: KpptTableTrainStepHostBatch<'_>,
    ) -> Result<()> {
        self.step_no_readback_with_loss_finalize(ctx, params, loss_kind, output_inv_scale, batch, true)
    }

    pub fn step_no_readback_with_loss_finalize(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: KpptTableTrainStepHostBatch<'_>,
        finalize_loss: bool,
    ) -> Result<()> {
        self.validate()?;
        batch.validate()?;
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(CudaCppError::message(format!(
                "KPPT table train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }

        self.device_batch.stm_indices.upload(ctx, batch.stm_indices)?;
        self.device_batch.nstm_indices.upload(ctx, batch.nstm_indices)?;
        self.targets.upload(ctx, batch.targets)?;
        self.entry_weights.upload(ctx, batch.entry_weights)?;

        kppt_table_forward_device(ctx, &self.device_batch, &self.weights, &self.forward_workspace)?;
        scalar_loss_device_from_buffers_with_finalize(
            ctx,
            loss_kind,
            output_inv_scale,
            self.batch_size,
            &self.forward_workspace.output,
            &self.targets,
            &self.entry_weights,
            &self.loss_workspace,
            finalize_loss,
        )?;
        kppt_table_backward_device(
            ctx,
            &self.device_batch,
            &self.weights,
            &self.forward_workspace,
            &self.loss_workspace,
            &self.backward_workspace,
        )?;
        self.update_weights(ctx, params)
    }

    pub fn read_loss(&self, ctx: &Context) -> Result<ScalarLossReadback> {
        self.loss_workspace.download(ctx)
    }

    pub fn read_weights(&self, ctx: &Context) -> Result<KpptTableTrainWeightsReadback> {
        Ok(KpptTableTrainWeightsReadback {
            table_w: self.weights.table_w.download(ctx)?,
            table_b: self.weights.table_b.download(ctx)?,
            outw: self.weights.outw.download(ctx)?,
            outb: self.weights.outb.download(ctx)?,
        })
    }

    pub fn read_optimizer_states(&self, ctx: &Context) -> Result<KpptTableRangerOptimizerStatesReadback> {
        self.optimizer_states.download(ctx)
    }

    fn validate(&self) -> Result<()> {
        validate_kppt_table_shape(self.shape)?;
        self.device_batch.validate()?;
        self.weights.validate()?;
        self.optimizer_states.validate(self.shape)?;
        self.forward_workspace.validate()?;
        self.loss_workspace.validate()?;
        self.backward_workspace.validate()?;
        expect_len("kppt train targets", self.batch_size, self.targets.len())?;
        expect_len("kppt train entry_weights", self.batch_size, self.entry_weights.len())
    }

    fn update_weights(&mut self, ctx: &Context, params: RangerUpdateParams) -> Result<()> {
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.table_w_gradients,
            &self.weights.table_w,
            &self.optimizer_states.table_w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.table_b_gradients,
            &self.weights.table_b,
            &self.optimizer_states.table_b,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.outw_gradients,
            &self.weights.outw,
            &self.optimizer_states.outw,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.outb_gradients,
            &self.weights.outb,
            &self.optimizer_states.outb,
        )
    }
}

impl SfnnRangerOptimizerStates {
    pub fn from_host_weights(ctx: &Context, weights: SfnnForwardHostWeights<'_>) -> Result<Self> {
        weights.validate()?;
        Ok(Self {
            l0w: RangerParamState::from_host_weights(ctx, weights.l0w)?,
            l0b: RangerParamState::from_host_weights(ctx, weights.l0b)?,
            l1w: RangerParamState::from_host_weights(ctx, weights.l1w)?,
            l1b: RangerParamState::from_host_weights(ctx, weights.l1b)?,
            l1fw: weights.l1fw.map(|values| RangerParamState::from_host_weights(ctx, values)).transpose()?,
            l1fb: weights.l1fb.map(|values| RangerParamState::from_host_weights(ctx, values)).transpose()?,
            l2w: RangerParamState::from_host_weights(ctx, weights.l2w)?,
            l2b: RangerParamState::from_host_weights(ctx, weights.l2b)?,
            l3w: RangerParamState::from_host_weights(ctx, weights.l3w)?,
            l3b: RangerParamState::from_host_weights(ctx, weights.l3b)?,
        })
    }

    pub fn from_host_states(
        ctx: &Context,
        shape: SfnnForwardShape,
        states: SfnnRangerOptimizerHostStates<'_>,
    ) -> Result<Self> {
        validate_sfnn_shape(shape)?;
        let l0w_len = checked_product("sfnn l0w", &[shape.input_size, shape.ft_size])?;
        let l1w_len = shape.l1w_len()?;
        let l2w_len = checked_product("sfnn l2w", &[shape.num_stacks, shape.l2_size, shape.l2_in()])?;
        let l3w_len = checked_product("sfnn l3w", &[shape.num_stacks, shape.l2_size])?;
        let (l1fw, l1fb) = match (states.l1fw, states.l1fb) {
            (Some(l1fw), Some(l1fb)) => (
                {
                    if shape.has_compact_l1() {
                        return Err(CudaCppError::message(
                            "SFNN compact L1 does not support factorized optimizer state",
                        ));
                    }
                    Some(RangerParamState::from_host_state(
                        ctx,
                        checked_product("sfnn l1fw", &[shape.ft_size, shape.l1_out()])?,
                        l1fw,
                    )?)
                },
                Some(RangerParamState::from_host_state(ctx, shape.l1_out(), l1fb)?),
            ),
            (None, None) => (None, None),
            (Some(_), None) => return Err(CudaCppError::message("SFNN optimizer l1fw requires l1fb")),
            (None, Some(_)) => return Err(CudaCppError::message("SFNN optimizer l1fb requires l1fw")),
        };
        Ok(Self {
            l0w: RangerParamState::from_host_state(ctx, l0w_len, states.l0w)?,
            l0b: RangerParamState::from_host_state(ctx, shape.ft_size, states.l0b)?,
            l1w: RangerParamState::from_host_state(ctx, l1w_len, states.l1w)?,
            l1b: RangerParamState::from_host_state(ctx, shape.num_stacks * shape.l1_out(), states.l1b)?,
            l1fw,
            l1fb,
            l2w: RangerParamState::from_host_state(ctx, l2w_len, states.l2w)?,
            l2b: RangerParamState::from_host_state(ctx, shape.num_stacks * shape.l2_size, states.l2b)?,
            l3w: RangerParamState::from_host_state(ctx, l3w_len, states.l3w)?,
            l3b: RangerParamState::from_host_state(ctx, shape.num_stacks, states.l3b)?,
        })
    }

    pub fn download(&self, ctx: &Context) -> Result<SfnnRangerOptimizerStatesReadback> {
        Ok(SfnnRangerOptimizerStatesReadback {
            l0w: self.l0w.download(ctx)?,
            l0b: self.l0b.download(ctx)?,
            l1w: self.l1w.download(ctx)?,
            l1b: self.l1b.download(ctx)?,
            l1fw: self.l1fw.as_ref().map(|state| state.download(ctx)).transpose()?,
            l1fb: self.l1fb.as_ref().map(|state| state.download(ctx)).transpose()?,
            l2w: self.l2w.download(ctx)?,
            l2b: self.l2b.download(ctx)?,
            l3w: self.l3w.download(ctx)?,
            l3b: self.l3b.download(ctx)?,
        })
    }

    fn validate(&self, shape: SfnnForwardShape) -> Result<()> {
        self.l0w.validate(checked_product("sfnn l0w", &[shape.input_size, shape.ft_size])?, "optimizer sfnn l0w")?;
        self.l0b.validate(shape.ft_size, "optimizer sfnn l0b")?;
        self.l1w.validate(shape.l1w_len()?, "optimizer sfnn l1w")?;
        self.l1b.validate(shape.num_stacks * shape.l1_out(), "optimizer sfnn l1b")?;
        match (&self.l1fw, &self.l1fb) {
            (Some(l1fw), Some(l1fb)) => {
                if shape.has_compact_l1() {
                    return Err(CudaCppError::message("SFNN compact L1 does not support factorized optimizer state"));
                }
                l1fw.validate(checked_product("sfnn l1fw", &[shape.ft_size, shape.l1_out()])?, "optimizer sfnn l1fw")?;
                l1fb.validate(shape.l1_out(), "optimizer sfnn l1fb")?;
            }
            (None, None) => {}
            (Some(_), None) => return Err(CudaCppError::message("SFNN optimizer l1fw requires l1fb")),
            (None, Some(_)) => return Err(CudaCppError::message("SFNN optimizer l1fb requires l1fw")),
        }
        self.l2w.validate(
            checked_product("sfnn l2w", &[shape.num_stacks, shape.l2_size, shape.l2_in()])?,
            "optimizer sfnn l2w",
        )?;
        self.l2b.validate(shape.num_stacks * shape.l2_size, "optimizer sfnn l2b")?;
        self.l3w.validate(checked_product("sfnn l3w", &[shape.num_stacks, shape.l2_size])?, "optimizer sfnn l3w")?;
        self.l3b.validate(shape.num_stacks, "optimizer sfnn l3b")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NnueTrainStepHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub targets: &'a [f32],
    pub entry_weights: &'a [f32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl<'a> NnueTrainStepHostBatch<'a> {
    fn forward_batch(self) -> NnueForwardHostBatch<'a> {
        NnueForwardHostBatch {
            stm_indices: self.stm_indices,
            nstm_indices: self.nstm_indices,
            batch_size: self.batch_size,
            max_active: self.max_active,
        }
    }

    pub fn validate(self) -> Result<()> {
        self.forward_batch().validate()?;
        expect_len("train targets", self.batch_size, self.targets.len())?;
        expect_len("train entry_weights", self.batch_size, self.entry_weights.len())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NnueTrainWeightsReadback {
    pub l0w: Vec<f32>,
    pub l0b: Vec<f32>,
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub outw: Vec<f32>,
    pub outb: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NnueTrainStepProfile {
    pub upload_ms: f32,
    pub forward_ms: f32,
    pub loss_ms: f32,
    pub backward_ms: f32,
    pub update_ms: f32,
    pub total_ms: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SfnnBackwardStageProfile {
    pub zero_ms: f32,
    pub l3_ms: f32,
    pub l2_ms: f32,
    pub l2_input_ms: f32,
    pub l1_ms: f32,
    pub l0_ms: f32,
    pub total_ms: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SfnnTrainStepProfile {
    pub upload_ms: f32,
    pub forward_ms: f32,
    pub loss_ms: f32,
    pub backward_ms: f32,
    pub update_ms: f32,
    pub total_ms: f32,
    pub backward_stages: SfnnBackwardStageProfile,
}

#[derive(Debug)]
pub struct NnueTrainStepUploadSlot {
    pub device_batch: NnueForwardDeviceBatch,
    pub targets: F32Buffer,
    pub entry_weights: F32Buffer,
    pinned_stm_indices: I32PinnedBuffer,
    pinned_nstm_indices: I32PinnedBuffer,
    pinned_targets: F32PinnedBuffer,
    pinned_entry_weights: F32PinnedBuffer,
    upload_ready: Event,
    compute_done: Event,
    in_use: bool,
}

impl NnueTrainStepUploadSlot {
    pub fn new(ctx: &Context, batch_size: usize, max_active: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(CudaCppError::message("NNUE upload slot batch_size must be greater than zero"));
        }
        if max_active == 0 {
            return Err(CudaCppError::message("NNUE upload slot max_active must be greater than zero"));
        }
        let sparse_len = batch_size
            .checked_mul(max_active)
            .ok_or_else(|| CudaCppError::message("NNUE upload slot sparse length overflow"))?;
        Ok(Self {
            device_batch: NnueForwardDeviceBatch {
                batch_size,
                max_active,
                stm_indices: I32Buffer::new(ctx, sparse_len)?,
                nstm_indices: I32Buffer::new(ctx, sparse_len)?,
            },
            targets: F32Buffer::new(ctx, batch_size)?,
            entry_weights: F32Buffer::new(ctx, batch_size)?,
            pinned_stm_indices: I32PinnedBuffer::new(ctx, sparse_len)?,
            pinned_nstm_indices: I32PinnedBuffer::new(ctx, sparse_len)?,
            pinned_targets: F32PinnedBuffer::new(ctx, batch_size)?,
            pinned_entry_weights: F32PinnedBuffer::new(ctx, batch_size)?,
            upload_ready: Event::new(ctx)?,
            compute_done: Event::new(ctx)?,
            in_use: false,
        })
    }

    fn upload(&mut self, upload_ctx: &Context, batch: NnueTrainStepHostBatch<'_>) -> Result<()> {
        batch.validate()?;
        if batch.batch_size != self.device_batch.batch_size || batch.max_active != self.device_batch.max_active {
            return Err(CudaCppError::message(format!(
                "NNUE upload slot batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.device_batch.batch_size, self.device_batch.max_active
            )));
        }
        if self.in_use {
            self.compute_done.wait(upload_ctx)?;
        }
        self.pinned_stm_indices.upload_to_device(upload_ctx, &self.device_batch.stm_indices, batch.stm_indices)?;
        self.pinned_nstm_indices.upload_to_device(upload_ctx, &self.device_batch.nstm_indices, batch.nstm_indices)?;
        self.pinned_targets.upload_to_device(upload_ctx, &self.targets, batch.targets)?;
        self.pinned_entry_weights.upload_to_device(upload_ctx, &self.entry_weights, batch.entry_weights)?;
        self.upload_ready.record(upload_ctx)?;
        self.in_use = true;
        Ok(())
    }

    fn wait_upload_on(&self, compute_ctx: &Context) -> Result<()> {
        self.upload_ready.wait(compute_ctx)
    }

    fn record_compute_done(&self, compute_ctx: &Context) -> Result<()> {
        self.compute_done.record(compute_ctx)
    }

    fn validate(&self, batch_size: usize, max_active: usize) -> Result<()> {
        if self.device_batch.batch_size != batch_size || self.device_batch.max_active != max_active {
            return Err(CudaCppError::message(format!(
                "NNUE upload slot layout mismatch: slot batch_size={} max_active={}, expected batch_size={} max_active={}",
                self.device_batch.batch_size, self.device_batch.max_active, batch_size, max_active
            )));
        }
        self.device_batch.validate()?;
        expect_len("nnue upload slot targets", batch_size, self.targets.len())?;
        expect_len("nnue upload slot entry_weights", batch_size, self.entry_weights.len())?;
        expect_len("nnue upload slot pinned targets", batch_size, self.pinned_targets.len())?;
        expect_len("nnue upload slot pinned entry_weights", batch_size, self.pinned_entry_weights.len())?;
        let sparse_len = batch_size
            .checked_mul(max_active)
            .ok_or_else(|| CudaCppError::message("NNUE upload slot sparse length overflow"))?;
        expect_len("nnue upload slot pinned stm_indices", sparse_len, self.pinned_stm_indices.len())?;
        expect_len("nnue upload slot pinned nstm_indices", sparse_len, self.pinned_nstm_indices.len())
    }
}

#[derive(Debug)]
pub struct NnueTrainStepRunner {
    pub shape: NnueForwardShape,
    pub batch_size: usize,
    pub max_active: usize,
    pub device_batch: NnueForwardDeviceBatch,
    pub targets: F32Buffer,
    pub entry_weights: F32Buffer,
    pub weights: NnueForwardDeviceWeights,
    pub optimizer_states: NnueRangerOptimizerStates,
    pub forward_workspace: NnueForwardWorkspace,
    pub loss_workspace: ScalarLossWorkspace,
    pub backward_workspace: NnueBackwardWorkspace,
    pub upload_slots: Vec<NnueTrainStepUploadSlot>,
    pub next_upload_slot: usize,
}

impl NnueTrainStepRunner {
    pub fn new(
        ctx: &Context,
        initial_weights: NnueForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        let optimizer_states = NnueRangerOptimizerStates::from_host_weights(ctx, initial_weights)?;
        Self::with_device_optimizer_states(ctx, initial_weights, optimizer_states, batch_size, max_active)
    }

    pub fn with_optimizer_states(
        ctx: &Context,
        initial_weights: NnueForwardHostWeights<'_>,
        optimizer_states: NnueRangerOptimizerHostStates<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        let optimizer_states =
            NnueRangerOptimizerStates::from_host_states(ctx, initial_weights.shape, optimizer_states)?;
        Self::with_device_optimizer_states(ctx, initial_weights, optimizer_states, batch_size, max_active)
    }

    fn with_device_optimizer_states(
        ctx: &Context,
        initial_weights: NnueForwardHostWeights<'_>,
        optimizer_states: NnueRangerOptimizerStates,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        initial_weights.validate()?;
        if batch_size == 0 {
            return Err(CudaCppError::message("NNUE train-step batch_size must be greater than zero"));
        }
        if max_active == 0 {
            return Err(CudaCppError::message("NNUE train-step max_active must be greater than zero"));
        }

        let shape = initial_weights.shape;
        let sparse_len = batch_size
            .checked_mul(max_active)
            .ok_or_else(|| CudaCppError::message("NNUE train-step sparse length overflow"))?;
        let mut upload_slots = Vec::with_capacity(2);
        for _ in 0..2 {
            upload_slots.push(NnueTrainStepUploadSlot::new(ctx, batch_size, max_active)?);
        }
        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_batch: NnueForwardDeviceBatch {
                batch_size,
                max_active,
                stm_indices: I32Buffer::new(ctx, sparse_len)?,
                nstm_indices: I32Buffer::new(ctx, sparse_len)?,
            },
            targets: F32Buffer::new(ctx, batch_size)?,
            entry_weights: F32Buffer::new(ctx, batch_size)?,
            weights: NnueForwardDeviceWeights::from_host(ctx, initial_weights)?,
            optimizer_states,
            forward_workspace: NnueForwardWorkspace::new(ctx, NnueForwardWorkspaceLayout::new(shape, batch_size))?,
            loss_workspace: ScalarLossWorkspace::new(ctx, ScalarLossWorkspaceLayout::new(batch_size))?,
            backward_workspace: NnueBackwardWorkspace::new(
                ctx,
                NnueBackwardWorkspaceLayout::new(shape, batch_size, max_active),
            )?,
            upload_slots,
            next_upload_slot: 0,
        })
    }

    pub fn step(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: NnueTrainStepHostBatch<'_>,
    ) -> Result<ScalarLossReadback> {
        self.step_no_readback(ctx, params, loss_kind, output_inv_scale, batch)?;
        self.read_loss(ctx)
    }

    pub fn warmup(&self, ctx: &Context) -> Result<()> {
        self.validate()?;
        nnue_train_warmup_device(
            ctx,
            &self.weights,
            &self.forward_workspace,
            &self.loss_workspace,
            &self.backward_workspace,
        )
    }

    pub fn step_no_readback(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: NnueTrainStepHostBatch<'_>,
    ) -> Result<()> {
        self.step_no_readback_with_loss_finalize(ctx, params, loss_kind, output_inv_scale, batch, true)
    }

    pub fn step_no_readback_with_loss_finalize(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: NnueTrainStepHostBatch<'_>,
        finalize_loss: bool,
    ) -> Result<()> {
        self.validate()?;
        batch.validate()?;
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(CudaCppError::message(format!(
                "NNUE train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }

        self.device_batch.stm_indices.upload(ctx, batch.stm_indices)?;
        self.device_batch.nstm_indices.upload(ctx, batch.nstm_indices)?;
        self.targets.upload(ctx, batch.targets)?;
        self.entry_weights.upload(ctx, batch.entry_weights)?;

        nnue_forward_device(ctx, &self.device_batch, &self.weights, &self.forward_workspace)?;
        scalar_loss_device_from_buffers_with_finalize(
            ctx,
            loss_kind,
            output_inv_scale,
            self.batch_size,
            &self.forward_workspace.output,
            &self.targets,
            &self.entry_weights,
            &self.loss_workspace,
            finalize_loss,
        )?;
        nnue_backward_device_reusing_zeroed_l0_gradients(
            ctx,
            &self.device_batch,
            &self.weights,
            &self.forward_workspace,
            &self.loss_workspace,
            &self.backward_workspace,
        )?;
        self.update_weights(ctx, params)
    }

    pub fn step_pipelined_no_readback(
        &mut self,
        ctx: &Context,
        upload_ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: NnueTrainStepHostBatch<'_>,
    ) -> Result<()> {
        self.step_pipelined_no_readback_with_loss_finalize(
            ctx,
            upload_ctx,
            params,
            loss_kind,
            output_inv_scale,
            batch,
            true,
        )
    }

    pub fn step_pipelined_no_readback_with_loss_finalize(
        &mut self,
        ctx: &Context,
        upload_ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: NnueTrainStepHostBatch<'_>,
        finalize_loss: bool,
    ) -> Result<()> {
        self.validate()?;
        batch.validate()?;
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(CudaCppError::message(format!(
                "NNUE train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }
        if self.upload_slots.is_empty() {
            return Err(CudaCppError::message("NNUE train-step runner has no upload slots"));
        }

        let slot_idx = self.next_upload_slot;
        self.next_upload_slot = (self.next_upload_slot + 1) % self.upload_slots.len();
        {
            let slot = &mut self.upload_slots[slot_idx];
            slot.upload(upload_ctx, batch)?;
        }
        {
            let slot = &self.upload_slots[slot_idx];
            slot.wait_upload_on(ctx)?;
            nnue_forward_device(ctx, &slot.device_batch, &self.weights, &self.forward_workspace)?;
            scalar_loss_device_from_buffers_with_finalize(
                ctx,
                loss_kind,
                output_inv_scale,
                self.batch_size,
                &self.forward_workspace.output,
                &slot.targets,
                &slot.entry_weights,
                &self.loss_workspace,
                finalize_loss,
            )?;
            nnue_backward_device_reusing_zeroed_l0_gradients(
                ctx,
                &slot.device_batch,
                &self.weights,
                &self.forward_workspace,
                &self.loss_workspace,
                &self.backward_workspace,
            )?;
        }
        self.update_weights(ctx, params)?;
        self.upload_slots[slot_idx].record_compute_done(ctx)
    }

    pub fn step_profiled_no_readback(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: NnueTrainStepHostBatch<'_>,
    ) -> Result<NnueTrainStepProfile> {
        self.validate()?;
        batch.validate()?;
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(CudaCppError::message(format!(
                "NNUE train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }

        let start = Event::new(ctx)?;
        let after_upload = Event::new(ctx)?;
        let after_forward = Event::new(ctx)?;
        let after_loss = Event::new(ctx)?;
        let after_backward = Event::new(ctx)?;
        let stop = Event::new(ctx)?;

        start.record(ctx)?;
        self.device_batch.stm_indices.upload(ctx, batch.stm_indices)?;
        self.device_batch.nstm_indices.upload(ctx, batch.nstm_indices)?;
        self.targets.upload(ctx, batch.targets)?;
        self.entry_weights.upload(ctx, batch.entry_weights)?;
        after_upload.record(ctx)?;

        nnue_forward_device(ctx, &self.device_batch, &self.weights, &self.forward_workspace)?;
        after_forward.record(ctx)?;
        scalar_loss_device_from_buffers(
            ctx,
            loss_kind,
            output_inv_scale,
            self.batch_size,
            &self.forward_workspace.output,
            &self.targets,
            &self.entry_weights,
            &self.loss_workspace,
        )?;
        after_loss.record(ctx)?;
        nnue_backward_device_reusing_zeroed_l0_gradients(
            ctx,
            &self.device_batch,
            &self.weights,
            &self.forward_workspace,
            &self.loss_workspace,
            &self.backward_workspace,
        )?;
        after_backward.record(ctx)?;
        self.update_weights(ctx, params)?;
        stop.record(ctx)?;
        stop.synchronize()?;

        Ok(NnueTrainStepProfile {
            upload_ms: after_upload.elapsed_ms_since(&start)?,
            forward_ms: after_forward.elapsed_ms_since(&after_upload)?,
            loss_ms: after_loss.elapsed_ms_since(&after_forward)?,
            backward_ms: after_backward.elapsed_ms_since(&after_loss)?,
            update_ms: stop.elapsed_ms_since(&after_backward)?,
            total_ms: stop.elapsed_ms_since(&start)?,
        })
    }

    pub fn read_loss(&self, ctx: &Context) -> Result<ScalarLossReadback> {
        self.loss_workspace.download(ctx)
    }

    pub fn read_weights(&self, ctx: &Context) -> Result<NnueTrainWeightsReadback> {
        Ok(NnueTrainWeightsReadback {
            l0w: self.weights.l0w.download(ctx)?,
            l0b: self.weights.l0b.download(ctx)?,
            l1w: self.weights.l1w.download(ctx)?,
            l1b: self.weights.l1b.download(ctx)?,
            l2w: self.weights.l2w.download(ctx)?,
            l2b: self.weights.l2b.download(ctx)?,
            outw: self.weights.outw.download(ctx)?,
            outb: self.weights.outb.download(ctx)?,
        })
    }

    pub fn read_optimizer_states(&self, ctx: &Context) -> Result<NnueRangerOptimizerStatesReadback> {
        self.optimizer_states.download(ctx)
    }

    fn validate(&self) -> Result<()> {
        validate_nnue_shape(self.shape)?;
        self.device_batch.validate()?;
        self.weights.validate()?;
        self.optimizer_states.validate(self.shape)?;
        self.forward_workspace.validate()?;
        self.loss_workspace.validate()?;
        self.backward_workspace.validate()?;
        expect_len("train targets", self.batch_size, self.targets.len())?;
        expect_len("train entry_weights", self.batch_size, self.entry_weights.len())?;
        for slot in &self.upload_slots {
            slot.validate(self.batch_size, self.max_active)?;
        }
        Ok(())
    }

    fn update_weights(&mut self, ctx: &Context, params: RangerUpdateParams) -> Result<()> {
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l0w_gradients,
            &self.weights.l0w,
            &self.optimizer_states.l0w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l0b_gradients,
            &self.weights.l0b,
            &self.optimizer_states.l0b,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l1w_gradients,
            &self.weights.l1w,
            &self.optimizer_states.l1w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l1b_gradients,
            &self.weights.l1b,
            &self.optimizer_states.l1b,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l2w_gradients,
            &self.weights.l2w,
            &self.optimizer_states.l2w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l2b_gradients,
            &self.weights.l2b,
            &self.optimizer_states.l2b,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.outw_gradients,
            &self.weights.outw,
            &self.optimizer_states.outw,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.outb_gradients,
            &self.weights.outb,
            &self.optimizer_states.outb,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnTrainStepHostBatch<'a> {
    pub stm_indices: &'a [i32],
    pub nstm_indices: &'a [i32],
    pub buckets: &'a [i32],
    pub targets: &'a [f32],
    pub entry_weights: &'a [f32],
    pub batch_size: usize,
    pub max_active: usize,
}

impl<'a> SfnnTrainStepHostBatch<'a> {
    fn forward_batch(self) -> SfnnForwardHostBatch<'a> {
        SfnnForwardHostBatch {
            stm_indices: self.stm_indices,
            nstm_indices: self.nstm_indices,
            buckets: self.buckets,
            batch_size: self.batch_size,
            max_active: self.max_active,
        }
    }

    pub fn validate(self) -> Result<()> {
        self.forward_batch().validate()?;
        expect_len("sfnn train targets", self.batch_size, self.targets.len())?;
        expect_len("sfnn train entry_weights", self.batch_size, self.entry_weights.len())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SfnnTrainWeightsReadback {
    pub l0w: Vec<f32>,
    pub l0b: Vec<f32>,
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l1fw: Option<Vec<f32>>,
    pub l1fb: Option<Vec<f32>>,
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub l3w: Vec<f32>,
    pub l3b: Vec<f32>,
}

#[derive(Debug)]
pub struct SfnnTrainStepUploadSlot {
    pub device_batch: SfnnForwardDeviceBatch,
    pub targets: F32Buffer,
    pub entry_weights: F32Buffer,
    upload_ready: Event,
    compute_done: Event,
    in_use: bool,
}

impl SfnnTrainStepUploadSlot {
    pub fn new(ctx: &Context, batch_size: usize, max_active: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(CudaCppError::message("SFNN upload slot batch_size must be greater than zero"));
        }
        if max_active == 0 {
            return Err(CudaCppError::message("SFNN upload slot max_active must be greater than zero"));
        }
        let sparse_len = batch_size
            .checked_mul(max_active)
            .ok_or_else(|| CudaCppError::message("SFNN upload slot sparse length overflow"))?;
        Ok(Self {
            device_batch: SfnnForwardDeviceBatch {
                batch_size,
                max_active,
                stm_indices: I32Buffer::new(ctx, sparse_len)?,
                nstm_indices: I32Buffer::new(ctx, sparse_len)?,
                buckets: I32Buffer::new(ctx, batch_size)?,
            },
            targets: F32Buffer::new(ctx, batch_size)?,
            entry_weights: F32Buffer::new(ctx, batch_size)?,
            upload_ready: Event::new(ctx)?,
            compute_done: Event::new(ctx)?,
            in_use: false,
        })
    }

    fn upload(&mut self, upload_ctx: &Context, batch: SfnnTrainStepHostBatch<'_>) -> Result<()> {
        batch.validate()?;
        if batch.batch_size != self.device_batch.batch_size || batch.max_active != self.device_batch.max_active {
            return Err(CudaCppError::message(format!(
                "SFNN upload slot batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.device_batch.batch_size, self.device_batch.max_active
            )));
        }
        if self.in_use {
            self.compute_done.wait(upload_ctx)?;
        }
        self.device_batch.stm_indices.upload(upload_ctx, batch.stm_indices)?;
        self.device_batch.nstm_indices.upload(upload_ctx, batch.nstm_indices)?;
        self.device_batch.buckets.upload(upload_ctx, batch.buckets)?;
        self.targets.upload(upload_ctx, batch.targets)?;
        self.entry_weights.upload(upload_ctx, batch.entry_weights)?;
        self.upload_ready.record(upload_ctx)?;
        self.in_use = true;
        Ok(())
    }

    fn wait_upload_on(&self, compute_ctx: &Context) -> Result<()> {
        self.upload_ready.wait(compute_ctx)
    }

    fn record_compute_done(&self, compute_ctx: &Context) -> Result<()> {
        self.compute_done.record(compute_ctx)
    }

    fn validate(&self, batch_size: usize, max_active: usize) -> Result<()> {
        if self.device_batch.batch_size != batch_size || self.device_batch.max_active != max_active {
            return Err(CudaCppError::message(format!(
                "SFNN upload slot layout mismatch: slot batch_size={} max_active={}, expected batch_size={} max_active={}",
                self.device_batch.batch_size, self.device_batch.max_active, batch_size, max_active
            )));
        }
        self.device_batch.validate()?;
        expect_len("sfnn upload slot targets", batch_size, self.targets.len())?;
        expect_len("sfnn upload slot entry_weights", batch_size, self.entry_weights.len())
    }
}

#[derive(Debug)]
pub struct SfnnTrainStepRunner {
    pub shape: SfnnForwardShape,
    pub batch_size: usize,
    pub max_active: usize,
    pub device_batch: SfnnForwardDeviceBatch,
    pub targets: F32Buffer,
    pub entry_weights: F32Buffer,
    pub weights: SfnnForwardDeviceWeights,
    pub optimizer_states: SfnnRangerOptimizerStates,
    pub forward_workspace: SfnnForwardWorkspace,
    pub loss_workspace: ScalarLossWorkspace,
    pub backward_workspace: SfnnBackwardWorkspace,
    pub upload_slots: Vec<SfnnTrainStepUploadSlot>,
    pub next_upload_slot: usize,
}

impl SfnnTrainStepRunner {
    pub fn new(
        ctx: &Context,
        initial_weights: SfnnForwardHostWeights<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        let optimizer_states = SfnnRangerOptimizerStates::from_host_weights(ctx, initial_weights)?;
        Self::with_device_optimizer_states(ctx, initial_weights, optimizer_states, batch_size, max_active)
    }

    pub fn with_optimizer_states(
        ctx: &Context,
        initial_weights: SfnnForwardHostWeights<'_>,
        optimizer_states: SfnnRangerOptimizerHostStates<'_>,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        let optimizer_states =
            SfnnRangerOptimizerStates::from_host_states(ctx, initial_weights.shape, optimizer_states)?;
        Self::with_device_optimizer_states(ctx, initial_weights, optimizer_states, batch_size, max_active)
    }

    fn with_device_optimizer_states(
        ctx: &Context,
        initial_weights: SfnnForwardHostWeights<'_>,
        optimizer_states: SfnnRangerOptimizerStates,
        batch_size: usize,
        max_active: usize,
    ) -> Result<Self> {
        initial_weights.validate()?;
        if batch_size == 0 {
            return Err(CudaCppError::message("SFNN train-step batch_size must be greater than zero"));
        }
        if max_active == 0 {
            return Err(CudaCppError::message("SFNN train-step max_active must be greater than zero"));
        }

        let shape = initial_weights.shape;
        let sparse_len = batch_size
            .checked_mul(max_active)
            .ok_or_else(|| CudaCppError::message("SFNN train-step sparse length overflow"))?;
        let mut upload_slots = Vec::with_capacity(2);
        for _ in 0..2 {
            upload_slots.push(SfnnTrainStepUploadSlot::new(ctx, batch_size, max_active)?);
        }
        let backward_workspace =
            SfnnBackwardWorkspace::new(ctx, SfnnBackwardWorkspaceLayout::new(shape, batch_size, max_active))?;
        backward_workspace.zero_parameter_gradients(ctx)?;
        Ok(Self {
            shape,
            batch_size,
            max_active,
            device_batch: SfnnForwardDeviceBatch {
                batch_size,
                max_active,
                stm_indices: I32Buffer::new(ctx, sparse_len)?,
                nstm_indices: I32Buffer::new(ctx, sparse_len)?,
                buckets: I32Buffer::new(ctx, batch_size)?,
            },
            targets: F32Buffer::new(ctx, batch_size)?,
            entry_weights: F32Buffer::new(ctx, batch_size)?,
            weights: SfnnForwardDeviceWeights::from_host(ctx, initial_weights)?,
            optimizer_states,
            forward_workspace: SfnnForwardWorkspace::new(ctx, SfnnForwardWorkspaceLayout::new(shape, batch_size))?,
            loss_workspace: ScalarLossWorkspace::new(ctx, ScalarLossWorkspaceLayout::new(batch_size))?,
            backward_workspace,
            upload_slots,
            next_upload_slot: 0,
        })
    }

    pub fn step(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: SfnnTrainStepHostBatch<'_>,
    ) -> Result<ScalarLossReadback> {
        self.step_no_readback(ctx, params, loss_kind, output_inv_scale, batch)?;
        self.read_loss(ctx)
    }

    pub fn step_no_readback(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: SfnnTrainStepHostBatch<'_>,
    ) -> Result<()> {
        self.step_no_readback_with_loss_finalize(ctx, params, loss_kind, output_inv_scale, batch, true)
    }

    pub fn step_no_readback_with_loss_finalize(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: SfnnTrainStepHostBatch<'_>,
        finalize_loss: bool,
    ) -> Result<()> {
        self.validate()?;
        batch.validate()?;
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(CudaCppError::message(format!(
                "SFNN train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }

        self.device_batch.stm_indices.upload(ctx, batch.stm_indices)?;
        self.device_batch.nstm_indices.upload(ctx, batch.nstm_indices)?;
        self.device_batch.buckets.upload(ctx, batch.buckets)?;
        self.targets.upload(ctx, batch.targets)?;
        self.entry_weights.upload(ctx, batch.entry_weights)?;

        sfnn_forward_device(ctx, &self.device_batch, &self.weights, &self.forward_workspace)?;
        scalar_loss_device_from_buffers_with_finalize(
            ctx,
            loss_kind,
            output_inv_scale,
            self.batch_size,
            &self.forward_workspace.output,
            &self.targets,
            &self.entry_weights,
            &self.loss_workspace,
            finalize_loss,
        )?;
        sfnn_backward_train_device(
            ctx,
            &self.device_batch,
            &self.weights,
            &self.forward_workspace,
            &self.loss_workspace,
            &self.backward_workspace,
        )?;
        self.update_weights(ctx, params)
    }

    pub fn step_pipelined_no_readback(
        &mut self,
        ctx: &Context,
        upload_ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: SfnnTrainStepHostBatch<'_>,
    ) -> Result<()> {
        self.step_pipelined_no_readback_with_loss_finalize(
            ctx,
            upload_ctx,
            params,
            loss_kind,
            output_inv_scale,
            batch,
            true,
        )
    }

    pub fn step_pipelined_no_readback_with_loss_finalize(
        &mut self,
        ctx: &Context,
        upload_ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: SfnnTrainStepHostBatch<'_>,
        finalize_loss: bool,
    ) -> Result<()> {
        self.validate()?;
        batch.validate()?;
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(CudaCppError::message(format!(
                "SFNN train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }
        if self.upload_slots.is_empty() {
            return Err(CudaCppError::message("SFNN train-step runner has no upload slots"));
        }

        let slot_idx = self.next_upload_slot;
        self.next_upload_slot = (self.next_upload_slot + 1) % self.upload_slots.len();
        {
            let slot = &mut self.upload_slots[slot_idx];
            slot.upload(upload_ctx, batch)?;
        }
        {
            let slot = &self.upload_slots[slot_idx];
            slot.wait_upload_on(ctx)?;
            sfnn_forward_device(ctx, &slot.device_batch, &self.weights, &self.forward_workspace)?;
            scalar_loss_device_from_buffers_with_finalize(
                ctx,
                loss_kind,
                output_inv_scale,
                self.batch_size,
                &self.forward_workspace.output,
                &slot.targets,
                &slot.entry_weights,
                &self.loss_workspace,
                finalize_loss,
            )?;
            sfnn_backward_train_device(
                ctx,
                &slot.device_batch,
                &self.weights,
                &self.forward_workspace,
                &self.loss_workspace,
                &self.backward_workspace,
            )?;
        }
        self.update_weights(ctx, params)?;
        self.upload_slots[slot_idx].record_compute_done(ctx)
    }

    pub fn step_profiled_no_readback(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: SfnnTrainStepHostBatch<'_>,
    ) -> Result<SfnnTrainStepProfile> {
        self.validate()?;
        batch.validate()?;
        if batch.batch_size != self.batch_size || batch.max_active != self.max_active {
            return Err(CudaCppError::message(format!(
                "SFNN train-step batch layout mismatch: got batch_size={} max_active={}, expected batch_size={} max_active={}",
                batch.batch_size, batch.max_active, self.batch_size, self.max_active
            )));
        }

        let start = Event::new(ctx)?;
        let after_upload = Event::new(ctx)?;
        let after_forward = Event::new(ctx)?;
        let after_loss = Event::new(ctx)?;
        let after_backward = Event::new(ctx)?;
        let stop = Event::new(ctx)?;

        start.record(ctx)?;
        self.device_batch.stm_indices.upload(ctx, batch.stm_indices)?;
        self.device_batch.nstm_indices.upload(ctx, batch.nstm_indices)?;
        self.device_batch.buckets.upload(ctx, batch.buckets)?;
        self.targets.upload(ctx, batch.targets)?;
        self.entry_weights.upload(ctx, batch.entry_weights)?;
        after_upload.record(ctx)?;

        sfnn_forward_device(ctx, &self.device_batch, &self.weights, &self.forward_workspace)?;
        after_forward.record(ctx)?;
        scalar_loss_device_from_buffers(
            ctx,
            loss_kind,
            output_inv_scale,
            self.batch_size,
            &self.forward_workspace.output,
            &self.targets,
            &self.entry_weights,
            &self.loss_workspace,
        )?;
        after_loss.record(ctx)?;
        let backward_stages = sfnn_backward_train_profile_device(
            ctx,
            &self.device_batch,
            &self.weights,
            &self.forward_workspace,
            &self.loss_workspace,
            &self.backward_workspace,
        )?;
        after_backward.record(ctx)?;
        self.update_weights(ctx, params)?;
        stop.record(ctx)?;
        stop.synchronize()?;

        Ok(SfnnTrainStepProfile {
            upload_ms: after_upload.elapsed_ms_since(&start)?,
            forward_ms: after_forward.elapsed_ms_since(&after_upload)?,
            loss_ms: after_loss.elapsed_ms_since(&after_forward)?,
            backward_ms: after_backward.elapsed_ms_since(&after_loss)?,
            update_ms: stop.elapsed_ms_since(&after_backward)?,
            total_ms: stop.elapsed_ms_since(&start)?,
            backward_stages,
        })
    }

    pub fn read_loss(&self, ctx: &Context) -> Result<ScalarLossReadback> {
        self.loss_workspace.download(ctx)
    }

    pub fn read_weights(&self, ctx: &Context) -> Result<SfnnTrainWeightsReadback> {
        Ok(SfnnTrainWeightsReadback {
            l0w: self.weights.l0w.download(ctx)?,
            l0b: self.weights.l0b.download(ctx)?,
            l1w: self.weights.l1w.download(ctx)?,
            l1b: self.weights.l1b.download(ctx)?,
            l1fw: self.weights.l1fw.as_ref().map(|weights| weights.download(ctx)).transpose()?,
            l1fb: self.weights.l1fb.as_ref().map(|weights| weights.download(ctx)).transpose()?,
            l2w: self.weights.l2w.download(ctx)?,
            l2b: self.weights.l2b.download(ctx)?,
            l3w: self.weights.l3w.download(ctx)?,
            l3b: self.weights.l3b.download(ctx)?,
        })
    }

    pub fn read_optimizer_states(&self, ctx: &Context) -> Result<SfnnRangerOptimizerStatesReadback> {
        self.optimizer_states.download(ctx)
    }

    fn validate(&self) -> Result<()> {
        validate_sfnn_shape(self.shape)?;
        self.device_batch.validate()?;
        self.weights.validate()?;
        self.optimizer_states.validate(self.shape)?;
        self.validate_optional_l1f_state_matches_weights()?;
        self.forward_workspace.validate()?;
        self.loss_workspace.validate()?;
        self.backward_workspace.validate()?;
        expect_len("sfnn train targets", self.batch_size, self.targets.len())?;
        expect_len("sfnn train entry_weights", self.batch_size, self.entry_weights.len())?;
        if self.next_upload_slot >= self.upload_slots.len() {
            return Err(CudaCppError::message(format!(
                "SFNN next upload slot {} is out of range for {} slots",
                self.next_upload_slot,
                self.upload_slots.len()
            )));
        }
        for slot in &self.upload_slots {
            slot.validate(self.batch_size, self.max_active)?;
        }
        Ok(())
    }

    fn validate_optional_l1f_state_matches_weights(&self) -> Result<()> {
        match (
            self.weights.l1fw.is_some(),
            self.weights.l1fb.is_some(),
            self.optimizer_states.l1fw.is_some(),
            self.optimizer_states.l1fb.is_some(),
        ) {
            (true, true, true, true) | (false, false, false, false) => Ok(()),
            _ => Err(CudaCppError::message("SFNN weight/state l1fw/l1fb optional groups mismatch")),
        }
    }

    fn update_weights(&mut self, ctx: &Context, params: RangerUpdateParams) -> Result<()> {
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l0w_gradients,
            &self.weights.l0w,
            &self.optimizer_states.l0w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l0b_gradients,
            &self.weights.l0b,
            &self.optimizer_states.l0b,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l1w_gradients,
            &self.weights.l1w,
            &self.optimizer_states.l1w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l1b_gradients,
            &self.weights.l1b,
            &self.optimizer_states.l1b,
        )?;
        match (&self.weights.l1fw, &self.weights.l1fb, &self.optimizer_states.l1fw, &self.optimizer_states.l1fb) {
            (Some(l1fw), Some(l1fb), Some(l1fw_state), Some(l1fb_state)) => {
                update_param_group(ctx, params, &self.backward_workspace.l1fw_gradients, l1fw, l1fw_state)?;
                update_param_group(ctx, params, &self.backward_workspace.l1fb_gradients, l1fb, l1fb_state)?;
            }
            (None, None, None, None) => {}
            _ => return Err(CudaCppError::message("SFNN weight/state l1fw/l1fb optional groups mismatch")),
        }
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l2w_gradients,
            &self.weights.l2w,
            &self.optimizer_states.l2w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l2b_gradients,
            &self.weights.l2b,
            &self.optimizer_states.l2b,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l3w_gradients,
            &self.weights.l3w,
            &self.optimizer_states.l3w,
        )?;
        update_param_group(
            ctx,
            params,
            &self.backward_workspace.l3b_gradients,
            &self.weights.l3b,
            &self.optimizer_states.l3b,
        )
    }
}

fn update_param_group(
    ctx: &Context,
    params: RangerUpdateParams,
    gradients: &F32Buffer,
    weights: &F32Buffer,
    state: &RangerParamState,
) -> Result<()> {
    ranger_update_device(
        ctx,
        params,
        RangerDeviceStateMut {
            gradients,
            weights,
            momentum: &state.momentum,
            velocity: &state.velocity,
            slow_params: &state.slow_params,
        },
    )
}

fn validate_nnue_shape(shape: NnueForwardShape) -> Result<()> {
    if shape.input_size == 0 || shape.l1 == 0 || shape.l2 == 0 || shape.l3 == 0 {
        Err(CudaCppError::message(format!("NNUE shape dimensions must be non-zero: {shape:?}")))
    } else {
        Ok(())
    }
}

fn validate_sfnn_shape(shape: SfnnForwardShape) -> Result<()> {
    let has_common_shard_marker = shape.l1_common_size != 0 || shape.l1_shard_size != 0;
    if shape.input_size == 0
        || shape.ft_size == 0
        || shape.l1_hidden == 0
        || shape.l2_size == 0
        || shape.num_stacks == 0
        || shape.l1_group_count == 0
        || shape.ft_size % 2 != 0
    {
        Err(CudaCppError::message(format!("SFNN shape dimensions are invalid: {shape:?}")))
    } else if has_common_shard_marker
        && (shape.l1_shard_size == 0
            || shape.l1_group_count <= 1
            || shape.l1_common_size + shape.l1_shard_size * shape.l1_group_count != shape.ft_size
            || shape.l1_out() % shape.l1_group_count != 0
            || shape.l1_common_size % 64 != 0
            || shape.l1_shard_size % 64 != 0)
    {
        Err(CudaCppError::message(format!("SFNN common+shard-L1 shape dimensions are invalid: {shape:?}")))
    } else if shape.has_grouped_l1()
        && (shape.ft_size % shape.l1_group_count != 0 || shape.l1_out() % shape.l1_group_count != 0)
    {
        Err(CudaCppError::message(format!("SFNN grouped-L1 shape dimensions are invalid: {shape:?}")))
    } else {
        Ok(())
    }
}

fn validate_kppt_table_shape(shape: KpptTableShape) -> Result<()> {
    if shape.input_size == 0 {
        Err(CudaCppError::message(format!("KPPT table shape dimensions must be non-zero: {shape:?}")))
    } else {
        Ok(())
    }
}

fn checked_product(name: &'static str, values: &[usize]) -> Result<usize> {
    let mut out = 1usize;
    for &value in values {
        out = out.checked_mul(value).ok_or_else(|| CudaCppError::message(format!("NNUE {name} length overflow")))?;
    }
    Ok(out)
}

fn expect_len(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(CudaCppError::message(format!("{name} length mismatch: expected {expected}, got {actual}")))
    }
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

    #[repr(C)]
    pub struct BulletOuCudaCppI32Buffer {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct BulletOuCudaCppPinnedF32Buffer {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct BulletOuCudaCppPinnedI32Buffer {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct BulletOuCudaCppEvent {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct BulletOuCudaCppGraphExec {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn bulletou_cuda_cpp_last_error(out: *mut c_char, out_len: usize) -> i32;
        pub fn bulletou_cuda_cpp_device_name(device: i32, out: *mut c_char, out_len: usize) -> i32;
        pub fn bulletou_cuda_cpp_context_create(device: i32, out: *mut *mut BulletOuCudaCppContext) -> i32;
        pub fn bulletou_cuda_cpp_context_destroy(ctx: *mut BulletOuCudaCppContext) -> i32;
        pub fn bulletou_cuda_cpp_context_synchronize(ctx: *mut BulletOuCudaCppContext) -> i32;
        pub fn bulletou_cuda_cpp_event_create(
            ctx: *mut BulletOuCudaCppContext,
            out: *mut *mut BulletOuCudaCppEvent,
        ) -> i32;
        pub fn bulletou_cuda_cpp_event_destroy(event: *mut BulletOuCudaCppEvent) -> i32;
        pub fn bulletou_cuda_cpp_event_record(
            ctx: *mut BulletOuCudaCppContext,
            event: *mut BulletOuCudaCppEvent,
        ) -> i32;
        pub fn bulletou_cuda_cpp_event_wait(ctx: *mut BulletOuCudaCppContext, event: *mut BulletOuCudaCppEvent) -> i32;
        pub fn bulletou_cuda_cpp_event_synchronize(event: *mut BulletOuCudaCppEvent) -> i32;
        pub fn bulletou_cuda_cpp_event_elapsed_ms(
            start: *mut BulletOuCudaCppEvent,
            stop: *mut BulletOuCudaCppEvent,
            out_ms: *mut f32,
        ) -> i32;
        pub fn bulletou_cuda_cpp_graph_begin_capture(ctx: *mut BulletOuCudaCppContext) -> i32;
        pub fn bulletou_cuda_cpp_graph_end_capture(
            ctx: *mut BulletOuCudaCppContext,
            out: *mut *mut BulletOuCudaCppGraphExec,
        ) -> i32;
        pub fn bulletou_cuda_cpp_graph_destroy(graph: *mut BulletOuCudaCppGraphExec) -> i32;
        pub fn bulletou_cuda_cpp_graph_launch(
            ctx: *mut BulletOuCudaCppContext,
            graph: *mut BulletOuCudaCppGraphExec,
        ) -> i32;
        pub fn bulletou_cuda_cpp_f32_buffer_create(
            ctx: *mut BulletOuCudaCppContext,
            len: usize,
            out: *mut *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_f32_buffer_destroy(buffer: *mut BulletOuCudaCppF32Buffer) -> i32;
        pub fn bulletou_cuda_cpp_i32_buffer_create(
            ctx: *mut BulletOuCudaCppContext,
            len: usize,
            out: *mut *mut BulletOuCudaCppI32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_i32_buffer_destroy(buffer: *mut BulletOuCudaCppI32Buffer) -> i32;
        pub fn bulletou_cuda_cpp_pinned_f32_buffer_create(
            ctx: *mut BulletOuCudaCppContext,
            len: usize,
            out: *mut *mut BulletOuCudaCppPinnedF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_pinned_f32_buffer_destroy(buffer: *mut BulletOuCudaCppPinnedF32Buffer) -> i32;
        pub fn bulletou_cuda_cpp_pinned_i32_buffer_create(
            ctx: *mut BulletOuCudaCppContext,
            len: usize,
            out: *mut *mut BulletOuCudaCppPinnedI32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_pinned_i32_buffer_destroy(buffer: *mut BulletOuCudaCppPinnedI32Buffer) -> i32;
        pub fn bulletou_cuda_cpp_i32_upload(
            ctx: *mut BulletOuCudaCppContext,
            dst: *mut BulletOuCudaCppI32Buffer,
            src: *const i32,
            len: usize,
        ) -> i32;
        pub fn bulletou_cuda_cpp_i32_upload_staged(
            ctx: *mut BulletOuCudaCppContext,
            dst: *mut BulletOuCudaCppI32Buffer,
            staging: *mut BulletOuCudaCppPinnedI32Buffer,
            src: *const i32,
            len: usize,
        ) -> i32;
        pub fn bulletou_cuda_cpp_i32_download(
            ctx: *mut BulletOuCudaCppContext,
            src: *mut BulletOuCudaCppI32Buffer,
            dst: *mut i32,
            len: usize,
        ) -> i32;
        pub fn bulletou_cuda_cpp_f32_upload(
            ctx: *mut BulletOuCudaCppContext,
            dst: *mut BulletOuCudaCppF32Buffer,
            src: *const f32,
            len: usize,
        ) -> i32;
        pub fn bulletou_cuda_cpp_f32_upload_staged(
            ctx: *mut BulletOuCudaCppContext,
            dst: *mut BulletOuCudaCppF32Buffer,
            staging: *mut BulletOuCudaCppPinnedF32Buffer,
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
        pub fn bulletou_cuda_cpp_nnue_forward_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            l1: usize,
            l2: usize,
            l3: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            l0w: *mut BulletOuCudaCppF32Buffer,
            l0b: *mut BulletOuCudaCppF32Buffer,
            l1w: *mut BulletOuCudaCppF32Buffer,
            l1b: *mut BulletOuCudaCppF32Buffer,
            l2w: *mut BulletOuCudaCppF32Buffer,
            l2b: *mut BulletOuCudaCppF32Buffer,
            outw: *mut BulletOuCudaCppF32Buffer,
            outb: *mut BulletOuCudaCppF32Buffer,
            stm_l0: *mut BulletOuCudaCppF32Buffer,
            nstm_l0: *mut BulletOuCudaCppF32Buffer,
            combined: *mut BulletOuCudaCppF32Buffer,
            hidden1: *mut BulletOuCudaCppF32Buffer,
            hidden2: *mut BulletOuCudaCppF32Buffer,
            output: *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_nnue_forward_host(
            device: i32,
            input_size: usize,
            l1: usize,
            l2: usize,
            l3: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *const i32,
            nstm_indices: *const i32,
            l0w: *const f32,
            l0b: *const f32,
            l1w: *const f32,
            l1b: *const f32,
            l2w: *const f32,
            l2b: *const f32,
            outw: *const f32,
            outb: *const f32,
            output: *mut f32,
        ) -> i32;
        pub fn bulletou_cuda_cpp_scalar_loss_device_with_finalize(
            ctx: *mut BulletOuCudaCppContext,
            kind: i32,
            output_inv_scale: f32,
            batch: usize,
            outputs: *mut BulletOuCudaCppF32Buffer,
            targets: *mut BulletOuCudaCppF32Buffer,
            entry_weights: *mut BulletOuCudaCppF32Buffer,
            per_sample: *mut BulletOuCudaCppF32Buffer,
            mean_output_gradients: *mut BulletOuCudaCppF32Buffer,
            weighted_sum: *mut BulletOuCudaCppF32Buffer,
            mean: *mut BulletOuCudaCppF32Buffer,
            finalize_loss: i32,
        ) -> i32;
        pub fn bulletou_cuda_cpp_scalar_loss_host(
            device: i32,
            kind: i32,
            output_inv_scale: f32,
            batch: usize,
            outputs: *const f32,
            targets: *const f32,
            entry_weights: *const f32,
            per_sample: *mut f32,
            mean_output_gradients: *mut f32,
            weighted_sum: *mut f32,
            mean: *mut f32,
        ) -> i32;
        pub fn bulletou_cuda_cpp_kppt_forward_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            table_w: *mut BulletOuCudaCppF32Buffer,
            table_b: *mut BulletOuCudaCppF32Buffer,
            outw: *mut BulletOuCudaCppF32Buffer,
            outb: *mut BulletOuCudaCppF32Buffer,
            stm_eval: *mut BulletOuCudaCppF32Buffer,
            nstm_eval: *mut BulletOuCudaCppF32Buffer,
            outputs: *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_kppt_backward_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            stm_eval: *mut BulletOuCudaCppF32Buffer,
            nstm_eval: *mut BulletOuCudaCppF32Buffer,
            outw: *mut BulletOuCudaCppF32Buffer,
            mean_output_gradients: *mut BulletOuCudaCppF32Buffer,
            table_w_gradients: *mut BulletOuCudaCppF32Buffer,
            table_b_gradients: *mut BulletOuCudaCppF32Buffer,
            outw_gradients: *mut BulletOuCudaCppF32Buffer,
            outb_gradients: *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_nnue_backward_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            l1: usize,
            l2: usize,
            l3: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            combined: *mut BulletOuCudaCppF32Buffer,
            hidden1: *mut BulletOuCudaCppF32Buffer,
            hidden2: *mut BulletOuCudaCppF32Buffer,
            stm_l0: *mut BulletOuCudaCppF32Buffer,
            nstm_l0: *mut BulletOuCudaCppF32Buffer,
            l1w: *mut BulletOuCudaCppF32Buffer,
            l2w: *mut BulletOuCudaCppF32Buffer,
            outw: *mut BulletOuCudaCppF32Buffer,
            mean_output_gradients: *mut BulletOuCudaCppF32Buffer,
            hidden2_gradients: *mut BulletOuCudaCppF32Buffer,
            hidden1_gradients: *mut BulletOuCudaCppF32Buffer,
            combined_gradients: *mut BulletOuCudaCppF32Buffer,
            stm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            nstm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            l0w_gradients: *mut BulletOuCudaCppF32Buffer,
            l0b_gradients: *mut BulletOuCudaCppF32Buffer,
            l1w_gradients: *mut BulletOuCudaCppF32Buffer,
            l1b_gradients: *mut BulletOuCudaCppF32Buffer,
            l2w_gradients: *mut BulletOuCudaCppF32Buffer,
            l2b_gradients: *mut BulletOuCudaCppF32Buffer,
            outw_gradients: *mut BulletOuCudaCppF32Buffer,
            outb_gradients: *mut BulletOuCudaCppF32Buffer,
            zero_l0_gradients: i32,
        ) -> i32;
        pub fn bulletou_cuda_cpp_nnue_train_warmup_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            l1: usize,
            l2: usize,
            l3: usize,
            batch: usize,
            max_active: usize,
            combined: *mut BulletOuCudaCppF32Buffer,
            hidden1: *mut BulletOuCudaCppF32Buffer,
            hidden2: *mut BulletOuCudaCppF32Buffer,
            l1w: *mut BulletOuCudaCppF32Buffer,
            l2w: *mut BulletOuCudaCppF32Buffer,
            outw: *mut BulletOuCudaCppF32Buffer,
            mean_output_gradients: *mut BulletOuCudaCppF32Buffer,
            hidden2_gradients: *mut BulletOuCudaCppF32Buffer,
            hidden1_gradients: *mut BulletOuCudaCppF32Buffer,
            combined_gradients: *mut BulletOuCudaCppF32Buffer,
            l1w_gradients: *mut BulletOuCudaCppF32Buffer,
            l1b_gradients: *mut BulletOuCudaCppF32Buffer,
            l2w_gradients: *mut BulletOuCudaCppF32Buffer,
            l2b_gradients: *mut BulletOuCudaCppF32Buffer,
            outw_gradients: *mut BulletOuCudaCppF32Buffer,
            outb_gradients: *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_sfnn_forward_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            ft_size: usize,
            l1_hidden: usize,
            l2_size: usize,
            num_stacks: usize,
            l1_group_count: usize,
            l1_common_size: usize,
            l1_shard_size: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            buckets: *mut BulletOuCudaCppI32Buffer,
            l0w: *mut BulletOuCudaCppF32Buffer,
            l0b: *mut BulletOuCudaCppF32Buffer,
            l1w: *mut BulletOuCudaCppF32Buffer,
            l1b: *mut BulletOuCudaCppF32Buffer,
            l1fw: *mut BulletOuCudaCppF32Buffer,
            l1fb: *mut BulletOuCudaCppF32Buffer,
            has_l1f: i32,
            l2w: *mut BulletOuCudaCppF32Buffer,
            l2b: *mut BulletOuCudaCppF32Buffer,
            l3w: *mut BulletOuCudaCppF32Buffer,
            l3b: *mut BulletOuCudaCppF32Buffer,
            stm_l0: *mut BulletOuCudaCppF32Buffer,
            nstm_l0: *mut BulletOuCudaCppF32Buffer,
            combined: *mut BulletOuCudaCppF32Buffer,
            l1: *mut BulletOuCudaCppF32Buffer,
            l2_input: *mut BulletOuCudaCppF32Buffer,
            l2: *mut BulletOuCudaCppF32Buffer,
            output: *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_sfnn_backward_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            ft_size: usize,
            l1_hidden: usize,
            l2_size: usize,
            num_stacks: usize,
            l1_group_count: usize,
            l1_common_size: usize,
            l1_shard_size: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            buckets: *mut BulletOuCudaCppI32Buffer,
            stm_l0: *mut BulletOuCudaCppF32Buffer,
            nstm_l0: *mut BulletOuCudaCppF32Buffer,
            combined: *mut BulletOuCudaCppF32Buffer,
            l1: *mut BulletOuCudaCppF32Buffer,
            l2_input: *mut BulletOuCudaCppF32Buffer,
            l2: *mut BulletOuCudaCppF32Buffer,
            l1w: *mut BulletOuCudaCppF32Buffer,
            l1fw: *mut BulletOuCudaCppF32Buffer,
            has_l1f: i32,
            l2w: *mut BulletOuCudaCppF32Buffer,
            l3w: *mut BulletOuCudaCppF32Buffer,
            mean_output_gradients: *mut BulletOuCudaCppF32Buffer,
            l2_gradients: *mut BulletOuCudaCppF32Buffer,
            l1_gradients: *mut BulletOuCudaCppF32Buffer,
            l2_input_gradients: *mut BulletOuCudaCppF32Buffer,
            combined_gradients: *mut BulletOuCudaCppF32Buffer,
            stm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            nstm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            stm_l0_pre_gradients: *mut BulletOuCudaCppF32Buffer,
            nstm_l0_pre_gradients: *mut BulletOuCudaCppF32Buffer,
            l0w_gradients: *mut BulletOuCudaCppF32Buffer,
            l0b_gradients: *mut BulletOuCudaCppF32Buffer,
            l1w_gradients: *mut BulletOuCudaCppF32Buffer,
            l1b_gradients: *mut BulletOuCudaCppF32Buffer,
            l1fw_gradients: *mut BulletOuCudaCppF32Buffer,
            l1fb_gradients: *mut BulletOuCudaCppF32Buffer,
            l2w_gradients: *mut BulletOuCudaCppF32Buffer,
            l2b_gradients: *mut BulletOuCudaCppF32Buffer,
            l3w_gradients: *mut BulletOuCudaCppF32Buffer,
            l3b_gradients: *mut BulletOuCudaCppF32Buffer,
        ) -> i32;
        pub fn bulletou_cuda_cpp_sfnn_backward_train_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            ft_size: usize,
            l1_hidden: usize,
            l2_size: usize,
            num_stacks: usize,
            l1_group_count: usize,
            l1_common_size: usize,
            l1_shard_size: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            buckets: *mut BulletOuCudaCppI32Buffer,
            stm_l0: *mut BulletOuCudaCppF32Buffer,
            nstm_l0: *mut BulletOuCudaCppF32Buffer,
            combined: *mut BulletOuCudaCppF32Buffer,
            l1: *mut BulletOuCudaCppF32Buffer,
            l2_input: *mut BulletOuCudaCppF32Buffer,
            l2: *mut BulletOuCudaCppF32Buffer,
            l1w: *mut BulletOuCudaCppF32Buffer,
            l1fw: *mut BulletOuCudaCppF32Buffer,
            has_l1f: i32,
            l2w: *mut BulletOuCudaCppF32Buffer,
            l3w: *mut BulletOuCudaCppF32Buffer,
            mean_output_gradients: *mut BulletOuCudaCppF32Buffer,
            l2_gradients: *mut BulletOuCudaCppF32Buffer,
            l1_gradients: *mut BulletOuCudaCppF32Buffer,
            l2_input_gradients: *mut BulletOuCudaCppF32Buffer,
            combined_gradients: *mut BulletOuCudaCppF32Buffer,
            stm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            nstm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            stm_l0_pre_gradients: *mut BulletOuCudaCppF32Buffer,
            nstm_l0_pre_gradients: *mut BulletOuCudaCppF32Buffer,
            l0w_gradients: *mut BulletOuCudaCppF32Buffer,
            l0b_gradients: *mut BulletOuCudaCppF32Buffer,
            l1w_gradients: *mut BulletOuCudaCppF32Buffer,
            l1b_gradients: *mut BulletOuCudaCppF32Buffer,
            l1fw_gradients: *mut BulletOuCudaCppF32Buffer,
            l1fb_gradients: *mut BulletOuCudaCppF32Buffer,
            l2w_gradients: *mut BulletOuCudaCppF32Buffer,
            l2b_gradients: *mut BulletOuCudaCppF32Buffer,
            l3w_gradients: *mut BulletOuCudaCppF32Buffer,
            l3b_gradients: *mut BulletOuCudaCppF32Buffer,
            zero_parameter_gradients: i32,
        ) -> i32;
        pub fn bulletou_cuda_cpp_sfnn_backward_train_profile_device(
            ctx: *mut BulletOuCudaCppContext,
            input_size: usize,
            ft_size: usize,
            l1_hidden: usize,
            l2_size: usize,
            num_stacks: usize,
            l1_group_count: usize,
            l1_common_size: usize,
            l1_shard_size: usize,
            batch: usize,
            max_active: usize,
            stm_indices: *mut BulletOuCudaCppI32Buffer,
            nstm_indices: *mut BulletOuCudaCppI32Buffer,
            buckets: *mut BulletOuCudaCppI32Buffer,
            stm_l0: *mut BulletOuCudaCppF32Buffer,
            nstm_l0: *mut BulletOuCudaCppF32Buffer,
            combined: *mut BulletOuCudaCppF32Buffer,
            l1: *mut BulletOuCudaCppF32Buffer,
            l2_input: *mut BulletOuCudaCppF32Buffer,
            l2: *mut BulletOuCudaCppF32Buffer,
            l1w: *mut BulletOuCudaCppF32Buffer,
            l1fw: *mut BulletOuCudaCppF32Buffer,
            has_l1f: i32,
            l2w: *mut BulletOuCudaCppF32Buffer,
            l3w: *mut BulletOuCudaCppF32Buffer,
            mean_output_gradients: *mut BulletOuCudaCppF32Buffer,
            l2_gradients: *mut BulletOuCudaCppF32Buffer,
            l1_gradients: *mut BulletOuCudaCppF32Buffer,
            l2_input_gradients: *mut BulletOuCudaCppF32Buffer,
            combined_gradients: *mut BulletOuCudaCppF32Buffer,
            stm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            nstm_l0_gradients: *mut BulletOuCudaCppF32Buffer,
            stm_l0_pre_gradients: *mut BulletOuCudaCppF32Buffer,
            nstm_l0_pre_gradients: *mut BulletOuCudaCppF32Buffer,
            l0w_gradients: *mut BulletOuCudaCppF32Buffer,
            l0b_gradients: *mut BulletOuCudaCppF32Buffer,
            l1w_gradients: *mut BulletOuCudaCppF32Buffer,
            l1b_gradients: *mut BulletOuCudaCppF32Buffer,
            l1fw_gradients: *mut BulletOuCudaCppF32Buffer,
            l1fb_gradients: *mut BulletOuCudaCppF32Buffer,
            l2w_gradients: *mut BulletOuCudaCppF32Buffer,
            l2b_gradients: *mut BulletOuCudaCppF32Buffer,
            l3w_gradients: *mut BulletOuCudaCppF32Buffer,
            l3b_gradients: *mut BulletOuCudaCppF32Buffer,
            zero_parameter_gradients: i32,
            profile_ms: *mut f32,
            profile_ms_len: usize,
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
    fn radam_step_scale_matches_reference_points() {
        let step1 = RAdamUpdateParams { step: 1, ..Default::default() }.step_scale().unwrap();
        assert!(!step1.use_denom);
        assert!((step1.step_size - 10.0).abs() < 0.00001);

        let step6 = RAdamUpdateParams { step: 6, ..Default::default() }.step_scale().unwrap();
        assert!(step6.step_size.is_finite());
        assert!(step6.step_size > 0.0);
    }

    #[test]
    fn nnue_shape_validation_reports_weight_mismatch() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let weights = NnueForwardHostWeights {
            shape,
            l0w: &[0.0; 7],
            l0b: &[0.0; 2],
            l1w: &[0.0; 8],
            l1b: &[0.0; 2],
            l2w: &[0.0; 2],
            l2b: &[0.0; 1],
            outw: &[0.0; 1],
            outb: &[0.0; 1],
        };

        let err = weights.validate().unwrap_err();

        assert!(err.to_string().contains("l0w length mismatch"));
    }

    #[test]
    fn nnue_workspace_layout_counts_forward_activations() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 3, l3: 1 };
        let layout = NnueForwardWorkspaceLayout::new(shape, 5);

        assert_eq!(layout.l0_len(), 10);
        assert_eq!(layout.combined_len(), 20);
        assert_eq!(layout.hidden1_len(), 15);
        assert_eq!(layout.hidden2_len(), 5);
        assert_eq!(layout.output_len(), 5);
    }

    #[test]
    fn sfnn_workspace_layout_counts_forward_activations() {
        let shape = SfnnForwardShape {
            input_size: 4,
            ft_size: 4,
            l1_hidden: 2,
            l2_size: 3,
            num_stacks: 2,
            l1_group_count: 1,
            l1_common_size: 0,
            l1_shard_size: 0,
        };
        let layout = SfnnForwardWorkspaceLayout::new(shape, 5);

        assert_eq!(shape.l1_out(), 3);
        assert_eq!(shape.l2_in(), 4);
        assert_eq!(layout.l0_len(), 20);
        assert_eq!(layout.combined_len(), 20);
        assert_eq!(layout.l1_len(), 15);
        assert_eq!(layout.l2_input_len(), 20);
        assert_eq!(layout.l2_len(), 15);
        assert_eq!(layout.output_len(), 5);
    }

    #[test]
    fn sfnn_backward_workspace_layout_counts_gradients() {
        let shape = SfnnForwardShape {
            input_size: 4,
            ft_size: 4,
            l1_hidden: 2,
            l2_size: 3,
            num_stacks: 2,
            l1_group_count: 1,
            l1_common_size: 0,
            l1_shard_size: 0,
        };
        let layout = SfnnBackwardWorkspaceLayout::new(shape, 5, 3);

        assert_eq!(layout.l2_gradients_len(), 15);
        assert_eq!(layout.l1_gradients_len(), 15);
        assert_eq!(layout.l2_input_gradients_len(), 20);
        assert_eq!(layout.combined_gradients_len(), 20);
        assert_eq!(layout.l0_gradients_len(), 20);
        assert_eq!(layout.l0w_gradients_len(), 16);
        assert_eq!(layout.l0b_gradients_len(), 4);
        assert_eq!(layout.l1w_gradients_len(), 24);
        assert_eq!(layout.l1b_gradients_len(), 6);
        assert_eq!(layout.l1fw_gradients_len(), 12);
        assert_eq!(layout.l1fb_gradients_len(), 3);
        assert_eq!(layout.l2w_gradients_len(), 24);
        assert_eq!(layout.l2b_gradients_len(), 6);
        assert_eq!(layout.l3w_gradients_len(), 6);
        assert_eq!(layout.l3b_gradients_len(), 2);
    }

    #[test]
    fn sfnn_grouped_l1w_layout_is_compact() {
        let shape = SfnnForwardShape {
            input_size: 133578,
            ft_size: 8192,
            l1_hidden: 15,
            l2_size: 64,
            num_stacks: 9,
            l1_group_count: 16,
            l1_common_size: 0,
            l1_shard_size: 0,
        };
        let layout = SfnnBackwardWorkspaceLayout::new(shape, 5, 40);

        assert!(shape.has_grouped_l1());
        assert_eq!(shape.l1_out(), 16);
        assert_eq!(shape.l1_group_input(), 512);
        assert_eq!(shape.l1_group_output(), 1);
        assert_eq!(shape.l1w_len().unwrap(), 9 * 16 * 1 * 512);
        assert_eq!(layout.l1w_gradients_len(), 9 * 16 * 1 * 512);
        assert!(shape.l1w_len().unwrap() < shape.num_stacks * shape.l1_out() * shape.ft_size);
    }

    #[test]
    fn sfnn_common_shard_l1w_layout_is_compact() {
        let shape = SfnnForwardShape {
            input_size: 1791,
            ft_size: 3072,
            l1_hidden: 7,
            l2_size: 64,
            num_stacks: 9,
            l1_group_count: 8,
            l1_common_size: 1024,
            l1_shard_size: 256,
        };
        let layout = SfnnBackwardWorkspaceLayout::new(shape, 5, 40);

        assert!(shape.has_common_shard_l1());
        assert_eq!(shape.l1_out(), 8);
        assert_eq!(shape.l1_common_shard_input(), 1280);
        assert_eq!(shape.l1_group_output(), 1);
        assert_eq!(shape.l1w_len().unwrap(), 9 * 8 * 1280);
        assert_eq!(layout.l1w_gradients_len(), 9 * 8 * 1280);
        assert!(shape.l1w_len().unwrap() < shape.num_stacks * shape.l1_out() * shape.ft_size);

        let c0_shape = SfnnForwardShape {
            input_size: 133578,
            ft_size: 8192,
            l1_hidden: 7,
            l2_size: 64,
            num_stacks: 9,
            l1_group_count: 8,
            l1_common_size: 0,
            l1_shard_size: 1024,
        };
        let c0_layout = SfnnBackwardWorkspaceLayout::new(c0_shape, 5, 40);

        assert!(c0_shape.has_common_shard_l1());
        assert!(!c0_shape.has_grouped_l1());
        assert_eq!(c0_shape.l1_out(), 8);
        assert_eq!(c0_shape.l1_common_shard_input(), 1024);
        assert_eq!(c0_shape.l1_group_output(), 1);
        assert_eq!(c0_shape.l1w_len().unwrap(), 9 * 8 * 1024);
        assert_eq!(c0_layout.l1w_gradients_len(), 9 * 8 * 1024);
        assert_eq!(c0_shape.l1w_len().unwrap(), c0_shape.num_stacks * c0_shape.l1_out() * 1024);
    }

    #[test]
    fn scalar_loss_validation_reports_length_mismatch() {
        let batch = ScalarLossHostBatch { outputs: &[0.0], targets: &[], entry_weights: &[1.0] };

        let err = batch.validate().unwrap_err();

        assert!(err.to_string().contains("loss targets length mismatch"));
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn axpy_gpu_smoke() {
        let out = axpy_host(0, 2.0, &[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(out, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn nnue_tiny_forward_gpu_smoke() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let batch = NnueForwardHostBatch {
            stm_indices: &[0, 1, -1, 3, -1, -1],
            nstm_indices: &[2, -1, -1, 1, 2, -1],
            batch_size: 2,
            max_active: 3,
        };
        let weights = tiny_nnue_weights(shape);

        let out = nnue_forward_host(0, batch, weights).unwrap();

        assert_close_slice("nnue", &out, &[1.208, 1.1195], 1.0e-5);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn scalar_loss_gpu_smoke() {
        let outputs = [-2.0, 0.0, 2.0];
        let targets = [0.0, 0.5, 1.0];
        let entry_weights = [1.0, 0.5, 2.0];
        let batch = ScalarLossHostBatch { outputs: &outputs, targets: &targets, entry_weights: &entry_weights };

        let host = scalar_loss_host(0, ScalarLossKind::SigmoidMse, 1.0, batch).unwrap();

        assert_close_slice("per_sample", &host.per_sample, &[0.014209336, 0.0, 0.028418668], 1.0e-6);
        assert_close_slice(
            "mean_output_gradients",
            &host.mean_output_gradients,
            &[0.008343695, 0.0, -0.01668739],
            1.0e-6,
        );
        assert_close("weighted_sum", host.weighted_sum, 0.042628005, 1.0e-6);
        assert_close("mean", host.mean, 0.014209335, 1.0e-6);

        let ctx = Context::new(0).unwrap();
        let device_batch = ScalarLossDeviceBatch::from_host(&ctx, batch).unwrap();
        let workspace = ScalarLossWorkspace::new(&ctx, ScalarLossWorkspaceLayout::new(batch.batch_size())).unwrap();
        scalar_loss_device(&ctx, ScalarLossKind::SigmoidMse, 1.0, &device_batch, &workspace).unwrap();
        let device = workspace.download(&ctx).unwrap();

        assert_close_slice("device per_sample", &device.per_sample, &host.per_sample, 1.0e-6);
        assert_close_slice(
            "device mean_output_gradients",
            &device.mean_output_gradients,
            &host.mean_output_gradients,
            1.0e-6,
        );
        assert_close("device weighted_sum", device.weighted_sum, host.weighted_sum, 1.0e-6);
        assert_close("device mean", device.mean, host.mean, 1.0e-6);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn sfnn_tiny_forward_gpu_smoke() {
        let shape = tiny_sfnn_shape();
        let batch = SfnnForwardHostBatch {
            stm_indices: &[0, 1, -1, 2, -1, -1],
            nstm_indices: &[2, -1, -1, 0, 3, -1],
            buckets: &[0, 1],
            batch_size: 2,
            max_active: 3,
        };
        let weights = tiny_sfnn_weights(shape);
        let expected = tiny_sfnn_forward_cpu(batch, weights);

        let ctx = Context::new(0).unwrap();
        let device_batch = SfnnForwardDeviceBatch::from_host(&ctx, batch).unwrap();
        let device_weights = SfnnForwardDeviceWeights::from_host(&ctx, weights).unwrap();
        let workspace =
            SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(shape, batch.batch_size)).unwrap();
        sfnn_forward_device(&ctx, &device_batch, &device_weights, &workspace).unwrap();
        let actual = workspace.download_output(&ctx).unwrap();

        assert_close_slice("sfnn", &actual, &expected, 1.0e-5);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn sfnn_tiny_backward_gpu_smoke() {
        let shape = tiny_sfnn_shape();
        let batch = SfnnForwardHostBatch {
            stm_indices: &[0, 1, -1, 2, -1, -1],
            nstm_indices: &[2, -1, -1, 0, 3, -1],
            buckets: &[0, 1],
            batch_size: 2,
            max_active: 3,
        };
        let weights = tiny_sfnn_weights(shape);
        let targets = [0.25, 0.75];
        let entry_weights = [1.0, 0.5];
        let expected = tiny_sfnn_backward_cpu(batch, weights, &targets, &entry_weights);

        let ctx = Context::new(0).unwrap();
        let device_batch = SfnnForwardDeviceBatch::from_host(&ctx, batch).unwrap();
        let device_weights = SfnnForwardDeviceWeights::from_host(&ctx, weights).unwrap();
        let forward =
            SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(shape, batch.batch_size)).unwrap();
        sfnn_forward_device(&ctx, &device_batch, &device_weights, &forward).unwrap();

        let targets_dev = F32Buffer::from_host(&ctx, &targets).unwrap();
        let entry_weights_dev = F32Buffer::from_host(&ctx, &entry_weights).unwrap();
        let loss = ScalarLossWorkspace::new(&ctx, ScalarLossWorkspaceLayout::new(batch.batch_size)).unwrap();
        scalar_loss_device_from_buffers(
            &ctx,
            ScalarLossKind::SigmoidMse,
            1.0,
            batch.batch_size,
            &forward.output,
            &targets_dev,
            &entry_weights_dev,
            &loss,
        )
        .unwrap();

        let backward = SfnnBackwardWorkspace::new(
            &ctx,
            SfnnBackwardWorkspaceLayout::new(shape, batch.batch_size, batch.max_active),
        )
        .unwrap();
        sfnn_backward_device(&ctx, &device_batch, &device_weights, &forward, &loss, &backward).unwrap();
        let actual = backward.download(&ctx).unwrap();

        assert_close_slice("sfnn l2_grad", &actual.l2_gradients, &expected.l2_gradients, 1.0e-6);
        assert_close_slice("sfnn l1_grad", &actual.l1_gradients, &expected.l1_gradients, 1.0e-6);
        assert_close_slice("sfnn l2_input_grad", &actual.l2_input_gradients, &expected.l2_input_gradients, 1.0e-6);
        assert_close_slice("sfnn combined_grad", &actual.combined_gradients, &expected.combined_gradients, 1.0e-6);
        assert_close_slice("sfnn stm_l0_grad", &actual.stm_l0_gradients, &expected.stm_l0_gradients, 1.0e-6);
        assert_close_slice("sfnn nstm_l0_grad", &actual.nstm_l0_gradients, &expected.nstm_l0_gradients, 1.0e-6);
        assert_close_slice(
            "sfnn stm_l0_pre_grad",
            &actual.stm_l0_pre_gradients,
            &expected.stm_l0_pre_gradients,
            1.0e-6,
        );
        assert_close_slice(
            "sfnn nstm_l0_pre_grad",
            &actual.nstm_l0_pre_gradients,
            &expected.nstm_l0_pre_gradients,
            1.0e-6,
        );
        assert_close_slice("sfnn l0w_grad", &actual.l0w_gradients, &expected.l0w_gradients, 1.0e-6);
        assert_close_slice("sfnn l0b_grad", &actual.l0b_gradients, &expected.l0b_gradients, 1.0e-6);
        assert_close_slice("sfnn l1w_grad", &actual.l1w_gradients, &expected.l1w_gradients, 1.0e-6);
        assert_close_slice("sfnn l1b_grad", &actual.l1b_gradients, &expected.l1b_gradients, 1.0e-6);
        assert_close_slice("sfnn l1fw_grad", &actual.l1fw_gradients, &expected.l1fw_gradients, 1.0e-6);
        assert_close_slice("sfnn l1fb_grad", &actual.l1fb_gradients, &expected.l1fb_gradients, 1.0e-6);
        assert_close_slice("sfnn l2w_grad", &actual.l2w_gradients, &expected.l2w_gradients, 1.0e-6);
        assert_close_slice("sfnn l2b_grad", &actual.l2b_gradients, &expected.l2b_gradients, 1.0e-6);
        assert_close_slice("sfnn l3w_grad", &actual.l3w_gradients, &expected.l3w_gradients, 1.0e-6);
        assert_close_slice("sfnn l3b_grad", &actual.l3b_gradients, &expected.l3b_gradients, 1.0e-6);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn sfnn_tiny_train_step_runner_smoke() {
        let shape = tiny_sfnn_shape();
        let batch = SfnnForwardHostBatch {
            stm_indices: &[0, 1, -1, 2, -1, -1],
            nstm_indices: &[2, -1, -1, 0, 3, -1],
            buckets: &[0, 1],
            batch_size: 2,
            max_active: 3,
        };
        let weights = tiny_sfnn_weights(shape);
        let targets = [0.25, 0.75];
        let entry_weights = [1.0, 0.5];
        let params = RangerUpdateParams {
            radam: RAdamUpdateParams {
                step: 1,
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                min_weight: -1.98,
                max_weight: 1.98,
                ..RAdamUpdateParams::default()
            },
            lookahead_alpha: 0.5,
            lookahead_period: 6,
        };
        let expected_gradients = tiny_sfnn_backward_cpu(batch, weights, &targets, &entry_weights);
        let expected = host_ranger_updated_tiny_sfnn_weights(0, weights, &expected_gradients, params);

        let ctx = Context::new(0).unwrap();
        let mut runner = SfnnTrainStepRunner::new(&ctx, weights, batch.batch_size, batch.max_active).unwrap();
        let loss = runner
            .step(
                &ctx,
                params,
                ScalarLossKind::SigmoidMse,
                1.0,
                SfnnTrainStepHostBatch {
                    stm_indices: batch.stm_indices,
                    nstm_indices: batch.nstm_indices,
                    buckets: batch.buckets,
                    targets: &targets,
                    entry_weights: &entry_weights,
                    batch_size: batch.batch_size,
                    max_active: batch.max_active,
                },
            )
            .unwrap();
        assert!(loss.mean.is_finite());
        let actual = runner.read_weights(&ctx).unwrap();

        assert_close_slice("train sfnn l0w", &actual.l0w, &expected.l0w, 1.0e-6);
        assert_close_slice("train sfnn l0b", &actual.l0b, &expected.l0b, 1.0e-6);
        assert_close_slice("train sfnn l1w", &actual.l1w, &expected.l1w, 1.0e-6);
        assert_close_slice("train sfnn l1b", &actual.l1b, &expected.l1b, 1.0e-6);
        assert_close_slice(
            "train sfnn l1fw",
            actual.l1fw.as_deref().unwrap(),
            expected.l1fw.as_deref().unwrap(),
            1.0e-6,
        );
        assert_close_slice(
            "train sfnn l1fb",
            actual.l1fb.as_deref().unwrap(),
            expected.l1fb.as_deref().unwrap(),
            1.0e-6,
        );
        assert_close_slice("train sfnn l2w", &actual.l2w, &expected.l2w, 1.0e-6);
        assert_close_slice("train sfnn l2b", &actual.l2b, &expected.l2b, 1.0e-6);
        assert_close_slice("train sfnn l3w", &actual.l3w, &expected.l3w, 1.0e-6);
        assert_close_slice("train sfnn l3b", &actual.l3b, &expected.l3b, 1.0e-6);

        let upload_ctx = Context::new(0).unwrap();
        let mut pipelined_runner = SfnnTrainStepRunner::new(&ctx, weights, batch.batch_size, batch.max_active).unwrap();
        pipelined_runner
            .step_pipelined_no_readback(
                &ctx,
                &upload_ctx,
                params,
                ScalarLossKind::SigmoidMse,
                1.0,
                SfnnTrainStepHostBatch {
                    stm_indices: batch.stm_indices,
                    nstm_indices: batch.nstm_indices,
                    buckets: batch.buckets,
                    targets: &targets,
                    entry_weights: &entry_weights,
                    batch_size: batch.batch_size,
                    max_active: batch.max_active,
                },
            )
            .unwrap();
        let pipelined_loss = pipelined_runner.read_loss(&ctx).unwrap();
        assert!(pipelined_loss.mean.is_finite());
        let pipelined = pipelined_runner.read_weights(&ctx).unwrap();
        assert_close_slice("pipelined train sfnn l0w", &pipelined.l0w, &expected.l0w, 1.0e-6);
        assert_close_slice("pipelined train sfnn l0b", &pipelined.l0b, &expected.l0b, 1.0e-6);
        assert_close_slice("pipelined train sfnn l1w", &pipelined.l1w, &expected.l1w, 1.0e-6);
        assert_close_slice("pipelined train sfnn l1b", &pipelined.l1b, &expected.l1b, 1.0e-6);
        assert_close_slice(
            "pipelined train sfnn l1fw",
            pipelined.l1fw.as_deref().unwrap(),
            expected.l1fw.as_deref().unwrap(),
            1.0e-6,
        );
        assert_close_slice(
            "pipelined train sfnn l1fb",
            pipelined.l1fb.as_deref().unwrap(),
            expected.l1fb.as_deref().unwrap(),
            1.0e-6,
        );
        assert_close_slice("pipelined train sfnn l2w", &pipelined.l2w, &expected.l2w, 1.0e-6);
        assert_close_slice("pipelined train sfnn l2b", &pipelined.l2b, &expected.l2b, 1.0e-6);
        assert_close_slice("pipelined train sfnn l3w", &pipelined.l3w, &expected.l3w, 1.0e-6);
        assert_close_slice("pipelined train sfnn l3b", &pipelined.l3b, &expected.l3b, 1.0e-6);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn persistent_device_api_smoke() {
        let ctx = Context::new(0).unwrap();
        let x = F32Buffer::from_host(&ctx, &[1.0, 2.0, 3.0]).unwrap();
        let y = F32Buffer::from_host(&ctx, &[10.0, 20.0, 30.0]).unwrap();
        let out = F32Buffer::new(&ctx, 3).unwrap();
        let start = Event::new(&ctx).unwrap();
        let stop = Event::new(&ctx).unwrap();
        start.record(&ctx).unwrap();
        axpy_device(&ctx, 3, 2.0, &x, &y, &out).unwrap();
        stop.record(&ctx).unwrap();
        stop.synchronize().unwrap();
        assert!(stop.elapsed_ms_since(&start).unwrap() >= 0.0);
        assert_eq!(out.download(&ctx).unwrap(), vec![12.0, 24.0, 36.0]);

        let graph_out = F32Buffer::new(&ctx, 3).unwrap();
        ctx.begin_capture().unwrap();
        axpy_device(&ctx, 3, 2.0, &x, &y, &graph_out).unwrap();
        let graph = ctx.end_capture().unwrap();
        graph_out.fill(&ctx, 0.0).unwrap();
        graph.launch(&ctx).unwrap();
        graph.launch(&ctx).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(graph_out.download(&ctx).unwrap(), vec![12.0, 24.0, 36.0]);

        let upload_ctx = Context::new(0).unwrap();
        let upload_x = F32UploadSlot::new(&upload_ctx, 3).unwrap();
        let upload_y = F32UploadSlot::new(&upload_ctx, 3).unwrap();
        upload_x.upload(&upload_ctx, &[1.0, 2.0, 3.0]).unwrap();
        upload_y.upload(&upload_ctx, &[10.0, 20.0, 30.0]).unwrap();
        let upload_out = F32Buffer::new(&ctx, 3).unwrap();
        axpy_device(&ctx, 3, 2.0, upload_x.wait_on(&ctx).unwrap(), upload_y.wait_on(&ctx).unwrap(), &upload_out)
            .unwrap();
        assert_eq!(upload_out.download(&ctx).unwrap(), vec![12.0, 24.0, 36.0]);

        let upload_i = I32UploadSlot::new(&upload_ctx, 3).unwrap();
        upload_i.upload(&upload_ctx, &[7, 8, 9]).unwrap();
        assert_eq!(upload_i.wait_on(&ctx).unwrap().download(&ctx).unwrap(), vec![7, 8, 9]);

        let staged_x = F32Buffer::new(&ctx, 3).unwrap();
        let staged_x_host = F32PinnedBuffer::new(&upload_ctx, 3).unwrap();
        staged_x_host.upload_to_device(&upload_ctx, &staged_x, &[3.0, 4.0, 5.0]).unwrap();
        let staged_ready = Event::new(&upload_ctx).unwrap();
        staged_ready.record(&upload_ctx).unwrap();
        staged_ready.wait(&ctx).unwrap();
        assert_eq!(staged_x.download(&ctx).unwrap(), vec![3.0, 4.0, 5.0]);

        let staged_i = I32Buffer::new(&ctx, 3).unwrap();
        let staged_i_host = I32PinnedBuffer::new(&upload_ctx, 3).unwrap();
        staged_i_host.upload_to_device(&upload_ctx, &staged_i, &[3, 4, 5]).unwrap();
        let staged_i_ready = Event::new(&upload_ctx).unwrap();
        staged_i_ready.record(&upload_ctx).unwrap();
        staged_i_ready.wait(&ctx).unwrap();
        assert_eq!(staged_i.download(&ctx).unwrap(), vec![3, 4, 5]);
    }

    fn tiny_nnue_weights(shape: NnueForwardShape) -> NnueForwardHostWeights<'static> {
        assert_eq!(shape, NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 });
        NnueForwardHostWeights {
            shape,
            l0w: &[
                0.2, 0.3, // feature 0
                0.4, -0.1, // feature 1
                -0.3, 0.5, // feature 2
                0.7, 0.9, // feature 3
            ],
            l0b: &[0.1, 0.2],
            l1w: &[
                0.5, -0.2, // combined 0
                0.1, 0.3, // combined 1
                -0.4, 0.2, // combined 2
                0.6, 0.1, // combined 3
            ],
            l1b: &[0.05, 0.1],
            l2w: &[
                0.7,  // hidden1 0
                -0.2, // hidden1 1
            ],
            l2b: &[0.2],
            outw: &[1.5],
            outb: &[0.05],
        }
    }

    fn tiny_sfnn_shape() -> SfnnForwardShape {
        SfnnForwardShape {
            input_size: 4,
            ft_size: 4,
            l1_hidden: 2,
            l2_size: 2,
            num_stacks: 2,
            l1_group_count: 1,
            l1_common_size: 0,
            l1_shard_size: 0,
        }
    }

    fn tiny_sfnn_weights(shape: SfnnForwardShape) -> SfnnForwardHostWeights<'static> {
        assert_eq!(shape, tiny_sfnn_shape());
        SfnnForwardHostWeights {
            shape,
            l0w: &[
                0.2, 0.3, 0.4, 0.5, // feature 0
                0.1, -0.2, 0.3, -0.4, // feature 1
                -0.3, 0.2, 0.6, 0.1, // feature 2
                0.7, 0.4, -0.5, 0.2, // feature 3
            ],
            l0b: &[0.05, 0.1, 0.15, 0.2],
            l1w: &[
                0.4, -0.1, 0.2, 0.3, // stack 0, row 0
                0.1, 0.5, -0.2, 0.4, // stack 0, row 1
                -0.3, 0.2, 0.6, -0.1, // stack 0, row 2 (PSQT)
                0.2, 0.3, -0.4, 0.1, // stack 1, row 0
                -0.2, 0.4, 0.1, 0.5, // stack 1, row 1
                0.3, -0.5, 0.2, 0.1, // stack 1, row 2 (PSQT)
            ],
            l1b: &[0.01, 0.02, 0.03, -0.01, 0.04, -0.02],
            l1fw: Some(&[
                0.01, -0.02, 0.03, // input 0
                -0.01, 0.02, -0.03, // input 1
                0.04, 0.01, -0.02, // input 2
                0.02, -0.04, 0.01, // input 3
            ]),
            l1fb: Some(&[0.005, -0.006, 0.007]),
            l2w: &[
                0.3, -0.1, 0.2, 0.4, // stack 0, row 0
                -0.2, 0.5, 0.1, -0.3, // stack 0, row 1
                0.1, 0.2, -0.4, 0.3, // stack 1, row 0
                0.4, -0.2, 0.3, 0.1, // stack 1, row 1
            ],
            l2b: &[0.02, -0.03, 0.01, 0.04],
            l3w: &[0.6, -0.4, 0.5, 0.2],
            l3b: &[0.05, -0.02],
        }
    }

    fn tiny_sfnn_forward_cpu(batch: SfnnForwardHostBatch<'_>, weights: SfnnForwardHostWeights<'_>) -> Vec<f32> {
        let shape = weights.shape;
        let mut out = vec![0.0; batch.batch_size];
        for (sample, out_sample) in out.iter_mut().enumerate() {
            let stack = batch.buckets[sample] as usize;
            let mut stm_l0 = weights.l0b.to_vec();
            let mut nstm_l0 = weights.l0b.to_vec();
            add_sparse_l0(
                &mut stm_l0,
                weights.l0w,
                shape.ft_size,
                shape.input_size,
                &batch.stm_indices[sample * batch.max_active..(sample + 1) * batch.max_active],
            );
            add_sparse_l0(
                &mut nstm_l0,
                weights.l0w,
                shape.ft_size,
                shape.input_size,
                &batch.nstm_indices[sample * batch.max_active..(sample + 1) * batch.max_active],
            );
            stm_l0.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            nstm_l0.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));

            let mut combined = vec![0.0; shape.ft_size];
            for pair in 0..shape.pairwise_size() {
                combined[pair] = stm_l0[pair] * stm_l0[shape.pairwise_size() + pair] * (127.0 / 128.0);
                combined[shape.pairwise_size() + pair] =
                    nstm_l0[pair] * nstm_l0[shape.pairwise_size() + pair] * (127.0 / 128.0);
            }

            let mut l1 = stacked_affine_cpu(
                &combined,
                weights.l1w,
                weights.l1b,
                shape.ft_size,
                shape.l1_out(),
                shape.num_stacks,
                stack,
            );
            if let (Some(l1fw), Some(l1fb)) = (weights.l1fw, weights.l1fb) {
                for row in 0..shape.l1_out() {
                    l1[row] += l1fb[row];
                    for input in 0..shape.ft_size {
                        l1[row] += combined[input] * l1fw[input * shape.l1_out() + row];
                    }
                }
            }
            let psqt = l1[shape.l1_hidden];
            let mut l2_input = vec![0.0; shape.l2_in()];
            for col in 0..shape.l2_in() {
                let value = l1[col % shape.l1_hidden];
                l2_input[col] = if col < shape.l1_hidden {
                    (value.abs() * value.abs() * (127.0 / 128.0)).clamp(0.0, 1.0)
                } else {
                    value.clamp(0.0, 1.0)
                };
            }

            let mut l2 = stacked_affine_cpu(
                &l2_input,
                weights.l2w,
                weights.l2b,
                shape.l2_in(),
                shape.l2_size,
                shape.num_stacks,
                stack,
            );
            l2.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            let mut value = weights.l3b[stack] + psqt;
            for input in 0..shape.l2_size {
                value += l2[input] * weights.l3w[stack * shape.l2_size + input];
            }
            *out_sample = value;
        }
        out
    }

    struct SfnnCpuTrace {
        stm_l0: Vec<f32>,
        nstm_l0: Vec<f32>,
        combined: Vec<f32>,
        l1: Vec<f32>,
        l2_input: Vec<f32>,
        l2: Vec<f32>,
        outputs: Vec<f32>,
    }

    struct SfnnCpuBackward {
        l2_gradients: Vec<f32>,
        l1_gradients: Vec<f32>,
        l2_input_gradients: Vec<f32>,
        combined_gradients: Vec<f32>,
        stm_l0_gradients: Vec<f32>,
        nstm_l0_gradients: Vec<f32>,
        stm_l0_pre_gradients: Vec<f32>,
        nstm_l0_pre_gradients: Vec<f32>,
        l0w_gradients: Vec<f32>,
        l0b_gradients: Vec<f32>,
        l1w_gradients: Vec<f32>,
        l1b_gradients: Vec<f32>,
        l1fw_gradients: Vec<f32>,
        l1fb_gradients: Vec<f32>,
        l2w_gradients: Vec<f32>,
        l2b_gradients: Vec<f32>,
        l3w_gradients: Vec<f32>,
        l3b_gradients: Vec<f32>,
    }

    struct SfnnTinyWeightsOwned {
        l0w: Vec<f32>,
        l0b: Vec<f32>,
        l1w: Vec<f32>,
        l1b: Vec<f32>,
        l1fw: Option<Vec<f32>>,
        l1fb: Option<Vec<f32>>,
        l2w: Vec<f32>,
        l2b: Vec<f32>,
        l3w: Vec<f32>,
        l3b: Vec<f32>,
    }

    fn tiny_sfnn_forward_trace_cpu(
        batch: SfnnForwardHostBatch<'_>,
        weights: SfnnForwardHostWeights<'_>,
    ) -> SfnnCpuTrace {
        let shape = weights.shape;
        let mut trace = SfnnCpuTrace {
            stm_l0: vec![0.0; batch.batch_size * shape.ft_size],
            nstm_l0: vec![0.0; batch.batch_size * shape.ft_size],
            combined: vec![0.0; batch.batch_size * shape.ft_size],
            l1: vec![0.0; batch.batch_size * shape.l1_out()],
            l2_input: vec![0.0; batch.batch_size * shape.l2_in()],
            l2: vec![0.0; batch.batch_size * shape.l2_size],
            outputs: vec![0.0; batch.batch_size],
        };

        for sample in 0..batch.batch_size {
            let stack = batch.buckets[sample] as usize;
            let sparse_base = sample * batch.max_active;
            let l0_base = sample * shape.ft_size;
            trace.stm_l0[l0_base..l0_base + shape.ft_size].copy_from_slice(weights.l0b);
            trace.nstm_l0[l0_base..l0_base + shape.ft_size].copy_from_slice(weights.l0b);
            add_sparse_l0(
                &mut trace.stm_l0[l0_base..l0_base + shape.ft_size],
                weights.l0w,
                shape.ft_size,
                shape.input_size,
                &batch.stm_indices[sparse_base..sparse_base + batch.max_active],
            );
            add_sparse_l0(
                &mut trace.nstm_l0[l0_base..l0_base + shape.ft_size],
                weights.l0w,
                shape.ft_size,
                shape.input_size,
                &batch.nstm_indices[sparse_base..sparse_base + batch.max_active],
            );
            trace.stm_l0[l0_base..l0_base + shape.ft_size].iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            trace.nstm_l0[l0_base..l0_base + shape.ft_size].iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));

            let combined_base = sample * shape.ft_size;
            for pair in 0..shape.pairwise_size() {
                trace.combined[combined_base + pair] = trace.stm_l0[l0_base + pair]
                    * trace.stm_l0[l0_base + shape.pairwise_size() + pair]
                    * (127.0 / 128.0);
                trace.combined[combined_base + shape.pairwise_size() + pair] = trace.nstm_l0[l0_base + pair]
                    * trace.nstm_l0[l0_base + shape.pairwise_size() + pair]
                    * (127.0 / 128.0);
            }

            let l1 = stacked_affine_cpu(
                &trace.combined[combined_base..combined_base + shape.ft_size],
                weights.l1w,
                weights.l1b,
                shape.ft_size,
                shape.l1_out(),
                shape.num_stacks,
                stack,
            );
            let l1_base = sample * shape.l1_out();
            trace.l1[l1_base..l1_base + shape.l1_out()].copy_from_slice(&l1);
            if let (Some(l1fw), Some(l1fb)) = (weights.l1fw, weights.l1fb) {
                for row in 0..shape.l1_out() {
                    trace.l1[l1_base + row] += l1fb[row];
                    for input in 0..shape.ft_size {
                        trace.l1[l1_base + row] +=
                            trace.combined[combined_base + input] * l1fw[input * shape.l1_out() + row];
                    }
                }
            }

            let l2_input_base = sample * shape.l2_in();
            for col in 0..shape.l2_in() {
                let value = trace.l1[l1_base + col % shape.l1_hidden];
                trace.l2_input[l2_input_base + col] = if col < shape.l1_hidden {
                    (value.abs() * value.abs() * (127.0 / 128.0)).clamp(0.0, 1.0)
                } else {
                    value.clamp(0.0, 1.0)
                };
            }

            let mut l2 = stacked_affine_cpu(
                &trace.l2_input[l2_input_base..l2_input_base + shape.l2_in()],
                weights.l2w,
                weights.l2b,
                shape.l2_in(),
                shape.l2_size,
                shape.num_stacks,
                stack,
            );
            l2.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            let l2_base = sample * shape.l2_size;
            trace.l2[l2_base..l2_base + shape.l2_size].copy_from_slice(&l2);
            let mut value = weights.l3b[stack] + trace.l1[l1_base + shape.l1_hidden];
            for input in 0..shape.l2_size {
                value += trace.l2[l2_base + input] * weights.l3w[stack * shape.l2_size + input];
            }
            trace.outputs[sample] = value;
        }

        trace
    }

    fn tiny_sfnn_backward_cpu(
        batch: SfnnForwardHostBatch<'_>,
        weights: SfnnForwardHostWeights<'_>,
        targets: &[f32],
        entry_weights: &[f32],
    ) -> SfnnCpuBackward {
        let shape = weights.shape;
        let trace = tiny_sfnn_forward_trace_cpu(batch, weights);
        let mut output_gradients = vec![0.0; batch.batch_size];
        for sample in 0..batch.batch_size {
            let prediction = sigmoid_cpu(trace.outputs[sample]);
            let error = prediction - targets[sample];
            output_gradients[sample] =
                entry_weights[sample] * 2.0 * error * prediction * (1.0 - prediction) / batch.batch_size as f32;
        }

        let mut l2_gradients = vec![0.0; batch.batch_size * shape.l2_size];
        let mut l1_gradients = vec![0.0; batch.batch_size * shape.l1_out()];
        let mut l3w_gradients = vec![0.0; shape.num_stacks * shape.l2_size];
        let mut l3b_gradients = vec![0.0; shape.num_stacks];
        for sample in 0..batch.batch_size {
            let stack = batch.buckets[sample] as usize;
            let output_gradient = output_gradients[sample];
            l3b_gradients[stack] += output_gradient;
            l1_gradients[sample * shape.l1_out() + shape.l1_hidden] = output_gradient;
            for row in 0..shape.l2_size {
                l2_gradients[sample * shape.l2_size + row] = output_gradient * weights.l3w[stack * shape.l2_size + row];
                l3w_gradients[stack * shape.l2_size + row] += output_gradient * trace.l2[sample * shape.l2_size + row];
            }
        }

        let mut l2_input_gradients = vec![0.0; batch.batch_size * shape.l2_in()];
        let mut l2w_gradients = vec![0.0; shape.num_stacks * shape.l2_size * shape.l2_in()];
        let mut l2b_gradients = vec![0.0; shape.num_stacks * shape.l2_size];
        for sample in 0..batch.batch_size {
            let stack = batch.buckets[sample] as usize;
            for out_col in 0..shape.l2_size {
                let out_idx = sample * shape.l2_size + out_col;
                let grad = crelu_pre_gradient_cpu(trace.l2[out_idx], l2_gradients[out_idx]);
                l2b_gradients[stack * shape.l2_size + out_col] += grad;
                for in_col in 0..shape.l2_in() {
                    let input_idx = sample * shape.l2_in() + in_col;
                    let weight_idx = stack * shape.l2_size * shape.l2_in() + out_col * shape.l2_in() + in_col;
                    l2_input_gradients[input_idx] += grad * weights.l2w[weight_idx];
                    l2w_gradients[weight_idx] += grad * trace.l2_input[input_idx];
                }
            }
        }

        for sample in 0..batch.batch_size {
            for col in 0..shape.l1_hidden {
                let l1_idx = sample * shape.l1_out() + col;
                let l2_input_base = sample * shape.l2_in();
                let square_idx = l2_input_base + col;
                let linear_idx = l2_input_base + shape.l1_hidden + col;
                let value = trace.l1[l1_idx];
                let square_grad = crelu_pre_gradient_cpu(trace.l2_input[square_idx], l2_input_gradients[square_idx])
                    * (2.0 * value * (127.0 / 128.0));
                let linear_grad = crelu_pre_gradient_cpu(trace.l2_input[linear_idx], l2_input_gradients[linear_idx]);
                l1_gradients[l1_idx] += square_grad + linear_grad;
            }
        }

        let mut combined_gradients = vec![0.0; batch.batch_size * shape.ft_size];
        let mut l1w_gradients = vec![0.0; shape.num_stacks * shape.l1_out() * shape.ft_size];
        let mut l1b_gradients = vec![0.0; shape.num_stacks * shape.l1_out()];
        let mut l1fw_gradients = vec![0.0; shape.ft_size * shape.l1_out()];
        let mut l1fb_gradients = vec![0.0; shape.l1_out()];
        for sample in 0..batch.batch_size {
            let stack = batch.buckets[sample] as usize;
            for out_col in 0..shape.l1_out() {
                let grad = l1_gradients[sample * shape.l1_out() + out_col];
                l1b_gradients[stack * shape.l1_out() + out_col] += grad;
                if weights.l1fb.is_some() {
                    l1fb_gradients[out_col] += grad;
                }
                for in_col in 0..shape.ft_size {
                    let input_idx = sample * shape.ft_size + in_col;
                    let weight_idx = stack * shape.l1_out() * shape.ft_size + out_col * shape.ft_size + in_col;
                    let input = trace.combined[input_idx];
                    let mut weight = weights.l1w[weight_idx];
                    l1w_gradients[weight_idx] += grad * input;
                    if let Some(l1fw) = weights.l1fw {
                        let shared_idx = in_col * shape.l1_out() + out_col;
                        weight += l1fw[shared_idx];
                        l1fw_gradients[shared_idx] += grad * input;
                    }
                    combined_gradients[input_idx] += grad * weight;
                }
            }
        }

        let mut stm_l0_gradients = vec![0.0; batch.batch_size * shape.ft_size];
        let mut nstm_l0_gradients = vec![0.0; batch.batch_size * shape.ft_size];
        for sample in 0..batch.batch_size {
            let l0_base = sample * shape.ft_size;
            for col in 0..shape.ft_size {
                let pair = col % shape.pairwise_size();
                let mate_col = if col < shape.pairwise_size() { shape.pairwise_size() + pair } else { pair };
                stm_l0_gradients[l0_base + col] =
                    combined_gradients[l0_base + pair] * trace.stm_l0[l0_base + mate_col] * (127.0 / 128.0);
                nstm_l0_gradients[l0_base + col] = combined_gradients[l0_base + shape.pairwise_size() + pair]
                    * trace.nstm_l0[l0_base + mate_col]
                    * (127.0 / 128.0);
            }
        }

        let mut stm_l0_pre_gradients = vec![0.0; batch.batch_size * shape.ft_size];
        let mut nstm_l0_pre_gradients = vec![0.0; batch.batch_size * shape.ft_size];
        let mut l0w_gradients = vec![0.0; shape.input_size * shape.ft_size];
        let mut l0b_gradients = vec![0.0; shape.ft_size];
        for sample in 0..batch.batch_size {
            let sparse_base = sample * batch.max_active;
            for row in 0..shape.ft_size {
                let idx = sample * shape.ft_size + row;
                let stm_grad = crelu_pre_gradient_cpu(trace.stm_l0[idx], stm_l0_gradients[idx]);
                let nstm_grad = crelu_pre_gradient_cpu(trace.nstm_l0[idx], nstm_l0_gradients[idx]);
                stm_l0_pre_gradients[idx] = stm_grad;
                nstm_l0_pre_gradients[idx] = nstm_grad;
                l0b_gradients[row] += stm_grad + nstm_grad;
                for slot in 0..batch.max_active {
                    let stm_feature = batch.stm_indices[sparse_base + slot];
                    if stm_feature >= 0 && (stm_feature as usize) < shape.input_size {
                        add_sfnn_l0w_gradient(
                            &mut l0w_gradients,
                            stm_feature as usize,
                            shape.input_size,
                            shape.ft_size,
                            row,
                            stm_grad,
                        );
                    }
                    let nstm_feature = batch.nstm_indices[sparse_base + slot];
                    if nstm_feature >= 0 && (nstm_feature as usize) < shape.input_size {
                        add_sfnn_l0w_gradient(
                            &mut l0w_gradients,
                            nstm_feature as usize,
                            shape.input_size,
                            shape.ft_size,
                            row,
                            nstm_grad,
                        );
                    }
                }
            }
        }

        SfnnCpuBackward {
            l2_gradients,
            l1_gradients,
            l2_input_gradients,
            combined_gradients,
            stm_l0_gradients,
            nstm_l0_gradients,
            stm_l0_pre_gradients,
            nstm_l0_pre_gradients,
            l0w_gradients,
            l0b_gradients,
            l1w_gradients,
            l1b_gradients,
            l1fw_gradients,
            l1fb_gradients,
            l2w_gradients,
            l2b_gradients,
            l3w_gradients,
            l3b_gradients,
        }
    }

    fn host_ranger_updated_tiny_sfnn_weights(
        device: i32,
        weights: SfnnForwardHostWeights<'_>,
        gradients: &SfnnCpuBackward,
        params: RangerUpdateParams,
    ) -> SfnnTinyWeightsOwned {
        let mut out = SfnnTinyWeightsOwned {
            l0w: weights.l0w.to_vec(),
            l0b: weights.l0b.to_vec(),
            l1w: weights.l1w.to_vec(),
            l1b: weights.l1b.to_vec(),
            l1fw: weights.l1fw.map(|values| values.to_vec()),
            l1fb: weights.l1fb.map(|values| values.to_vec()),
            l2w: weights.l2w.to_vec(),
            l2b: weights.l2b.to_vec(),
            l3w: weights.l3w.to_vec(),
            l3b: weights.l3b.to_vec(),
        };
        apply_host_ranger_to_group(device, params, &gradients.l0w_gradients, &mut out.l0w);
        apply_host_ranger_to_group(device, params, &gradients.l0b_gradients, &mut out.l0b);
        apply_host_ranger_to_group(device, params, &gradients.l1w_gradients, &mut out.l1w);
        apply_host_ranger_to_group(device, params, &gradients.l1b_gradients, &mut out.l1b);
        if let Some(l1fw) = &mut out.l1fw {
            apply_host_ranger_to_group(device, params, &gradients.l1fw_gradients, l1fw);
        }
        if let Some(l1fb) = &mut out.l1fb {
            apply_host_ranger_to_group(device, params, &gradients.l1fb_gradients, l1fb);
        }
        apply_host_ranger_to_group(device, params, &gradients.l2w_gradients, &mut out.l2w);
        apply_host_ranger_to_group(device, params, &gradients.l2b_gradients, &mut out.l2b);
        apply_host_ranger_to_group(device, params, &gradients.l3w_gradients, &mut out.l3w);
        apply_host_ranger_to_group(device, params, &gradients.l3b_gradients, &mut out.l3b);
        out
    }

    fn apply_host_ranger_to_group(device: i32, params: RangerUpdateParams, gradients: &[f32], weights: &mut [f32]) {
        let mut gradients = gradients.to_vec();
        let mut momentum = vec![0.0; gradients.len()];
        let mut velocity = vec![0.0; gradients.len()];
        let mut slow_params = weights.to_vec();
        ranger_update_host(
            device,
            params,
            RangerStateMut {
                gradients: &mut gradients,
                weights,
                momentum: &mut momentum,
                velocity: &mut velocity,
                slow_params: &mut slow_params,
            },
        )
        .unwrap();
    }

    fn add_sfnn_l0w_gradient(
        gradients: &mut [f32],
        feature: usize,
        input_size: usize,
        rows: usize,
        row: usize,
        value: f32,
    ) {
        gradients[feature * rows + row] += value;
        if let Some(virtual_feature) = sfnn_factorized_virtual_feature_cpu(feature, input_size) {
            gradients[virtual_feature * rows + row] += value;
        }
    }

    fn sfnn_factorized_virtual_feature_cpu(feature: usize, input_size: usize) -> Option<usize> {
        const BASE_INPUT_SIZE: usize = 131_949;
        const PIECE_INPUTS: usize = 1_629;
        if input_size == BASE_INPUT_SIZE + PIECE_INPUTS && feature < BASE_INPUT_SIZE {
            Some(BASE_INPUT_SIZE + feature % PIECE_INPUTS)
        } else {
            None
        }
    }

    fn crelu_pre_gradient_cpu(activation: f32, output_gradient: f32) -> f32 {
        if activation > 0.0 && activation < 1.0 { output_gradient } else { 0.0 }
    }

    fn sigmoid_cpu(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn add_sparse_l0(out: &mut [f32], weights: &[f32], rows: usize, input_size: usize, indices: &[i32]) {
        for &feature in indices {
            if feature >= 0 && (feature as usize) < input_size {
                let base = feature as usize * rows;
                for row in 0..rows {
                    out[row] += weights[base + row];
                }
            }
        }
    }

    fn stacked_affine_cpu(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        input_dim: usize,
        output_dim: usize,
        _num_stacks: usize,
        stack: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; output_dim];
        let stack_base = stack * output_dim * input_dim;
        for row in 0..output_dim {
            out[row] = bias[stack * output_dim + row];
            for col in 0..input_dim {
                out[row] += input[col] * weights[stack_base + row * input_dim + col];
            }
        }
        out
    }

    fn assert_close_slice(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let abs_diff = (actual - expected).abs();
            assert!(
                abs_diff <= tolerance,
                "{name}[{idx}] mismatch: expected {expected}, got {actual}, abs_diff={abs_diff}"
            );
        }
    }

    fn assert_close(name: &str, actual: f32, expected: f32, tolerance: f32) {
        let abs_diff = (actual - expected).abs();
        assert!(abs_diff <= tolerance, "{name}: expected {expected}, got {actual}, abs_diff={abs_diff}");
    }
}
