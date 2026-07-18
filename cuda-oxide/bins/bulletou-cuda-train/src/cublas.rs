//! Minimal cuBLAS wrapper for NNUE dense layers.

use std::{ffi::c_int, ptr::NonNull};

use bulletou_cuda_oxide_runtime::{CudaStream, DeviceBuffer, Error, Result};

#[repr(C)]
struct CublasContext {
    _opaque: [u8; 0],
}

type CublasHandleT = *mut CublasContext;
type CublasStatusT = c_int;
type CublasOperationT = c_int;
type CublasMathT = u32;

const CUBLAS_STATUS_SUCCESS: CublasStatusT = 0;
const CUBLAS_OP_N: CublasOperationT = 0;
const CUBLAS_OP_T: CublasOperationT = 1;
const CUBLAS_DEFAULT_MATH: CublasMathT = 0;
const CUBLAS_TF32_TENSOR_OP_MATH: CublasMathT = 3;

#[link(name = "cublas", kind = "dylib")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut CublasHandleT) -> CublasStatusT;
    fn cublasDestroy_v2(handle: CublasHandleT) -> CublasStatusT;
    fn cublasSetStream_v2(handle: CublasHandleT, stream_id: cuda_core::sys::CUstream) -> CublasStatusT;
    fn cublasSetMathMode(handle: CublasHandleT, mode: CublasMathT) -> CublasStatusT;
    fn cublasSgemm_v2(
        handle: CublasHandleT,
        transa: CublasOperationT,
        transb: CublasOperationT,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: c_int,
    ) -> CublasStatusT;
}

pub(crate) struct CublasHandle {
    handle: NonNull<CublasContext>,
}

// SAFETY: cuBLAS handles are opaque CUDA-side resources. We bind the handle to
// the train stream at creation and only use it from the owning trainer path.
unsafe impl Send for CublasHandle {}

impl CublasHandle {
    pub(crate) fn new(stream: &CudaStream, enable_tf32: bool) -> Result<Self> {
        let mut raw: CublasHandleT = std::ptr::null_mut();
        check_status("cublasCreate_v2", unsafe { cublasCreate_v2(&mut raw) })?;
        let Some(handle) = NonNull::new(raw) else {
            return Err(Error::Smoke("cublasCreate_v2 returned a null handle".to_string()));
        };

        let stream_status = unsafe { cublasSetStream_v2(handle.as_ptr(), stream.cu_stream()) };
        if let Err(err) = check_status("cublasSetStream_v2", stream_status) {
            unsafe {
                cublasDestroy_v2(handle.as_ptr());
            }
            return Err(err);
        }

        let math_mode = if enable_tf32 { CUBLAS_TF32_TENSOR_OP_MATH } else { CUBLAS_DEFAULT_MATH };
        let math_status = unsafe { cublasSetMathMode(handle.as_ptr(), math_mode) };
        if let Err(err) = check_status("cublasSetMathMode", math_status) {
            unsafe {
                cublasDestroy_v2(handle.as_ptr());
            }
            return Err(err);
        }

        Ok(Self { handle })
    }

    /// Row-major C[M, N] = X[M, K] @ Y[N, K]^T.
    pub(crate) fn sgemm_x_yt_rowmajor(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: &DeviceBuffer<f32>,
        y: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let (m, n, k) = checked_dims(m, n, k)?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle.as_ptr(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                y.cu_deviceptr() as *const f32,
                k,
                x.cu_deviceptr() as *const f32,
                k,
                &beta,
                c.cu_deviceptr() as *mut f32,
                n,
            )
        };
        check_status("cublasSgemm_v2(x_yt)", status)
    }

    /// Row-major C[M, N] = X[K, M]^T @ Y[K, N].
    pub(crate) fn sgemm_xt_y_rowmajor(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: &DeviceBuffer<f32>,
        y: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let (m, n, k) = checked_dims(m, n, k)?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle.as_ptr(),
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n,
                m,
                k,
                &alpha,
                y.cu_deviceptr() as *const f32,
                n,
                x.cu_deviceptr() as *const f32,
                m,
                &beta,
                c.cu_deviceptr() as *mut f32,
                n,
            )
        };
        check_status("cublasSgemm_v2(xt_y)", status)
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle.as_ptr());
        }
    }
}

fn checked_dims(m: usize, n: usize, k: usize) -> Result<(c_int, c_int, c_int)> {
    fn one(name: &str, value: usize) -> Result<c_int> {
        c_int::try_from(value).map_err(|_| Error::Smoke(format!("cuBLAS dimension {name} is too large: {value}")))
    }
    Ok((one("m", m)?, one("n", n)?, one("k", k)?))
}

fn check_status(label: &str, status: CublasStatusT) -> Result<()> {
    if status == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Smoke(format!("{label} failed: status={status}")))
    }
}
