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
pub struct F32UploadSlot {
    buffer: F32Buffer,
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
        expect_len("l0w", checked_product("l0w", &[shape.input_size, shape.l1])?, self.l0w.len())?;
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
        expect_len("device l0w", checked_product("l0w", &[shape.input_size, shape.l1])?, self.l0w.len())?;
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
        ffi::bulletou_cuda_cpp_scalar_loss_device(
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
        self.shape.input_size.saturating_mul(self.shape.l1)
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
        Ok(Self {
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
        })
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
        )
    })
}

#[derive(Debug)]
pub struct RangerParamState {
    pub momentum: F32Buffer,
    pub velocity: F32Buffer,
    pub slow_params: F32Buffer,
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

    fn validate(&self, shape: NnueForwardShape) -> Result<()> {
        self.l0w.validate(checked_product("l0w", &[shape.input_size, shape.l1])?, "optimizer l0w")?;
        self.l0b.validate(shape.l1, "optimizer l0b")?;
        self.l1w.validate(checked_product("l1w", &[shape.l1, 2, shape.l2])?, "optimizer l1w")?;
        self.l1b.validate(shape.l2, "optimizer l1b")?;
        self.l2w.validate(checked_product("l2w", &[shape.l2, shape.l3])?, "optimizer l2w")?;
        self.l2b.validate(shape.l3, "optimizer l2b")?;
        self.outw.validate(shape.l3, "optimizer outw")?;
        self.outb.validate(1, "optimizer outb")
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
}

impl NnueTrainStepRunner {
    pub fn new(
        ctx: &Context,
        initial_weights: NnueForwardHostWeights<'_>,
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
            optimizer_states: NnueRangerOptimizerStates::from_host_weights(ctx, initial_weights)?,
            forward_workspace: NnueForwardWorkspace::new(ctx, NnueForwardWorkspaceLayout::new(shape, batch_size))?,
            loss_workspace: ScalarLossWorkspace::new(ctx, ScalarLossWorkspaceLayout::new(batch_size))?,
            backward_workspace: NnueBackwardWorkspace::new(
                ctx,
                NnueBackwardWorkspaceLayout::new(shape, batch_size, max_active),
            )?,
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

    pub fn step_no_readback(
        &mut self,
        ctx: &Context,
        params: RangerUpdateParams,
        loss_kind: ScalarLossKind,
        output_inv_scale: f32,
        batch: NnueTrainStepHostBatch<'_>,
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
        nnue_backward_device(
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

    fn validate(&self) -> Result<()> {
        validate_nnue_shape(self.shape)?;
        self.device_batch.validate()?;
        self.weights.validate()?;
        self.optimizer_states.validate(self.shape)?;
        self.forward_workspace.validate()?;
        self.loss_workspace.validate()?;
        self.backward_workspace.validate()?;
        expect_len("train targets", self.batch_size, self.targets.len())?;
        expect_len("train entry_weights", self.batch_size, self.entry_weights.len())
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
        pub fn bulletou_cuda_cpp_i32_upload(
            ctx: *mut BulletOuCudaCppContext,
            dst: *mut BulletOuCudaCppI32Buffer,
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
        pub fn bulletou_cuda_cpp_scalar_loss_device(
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
