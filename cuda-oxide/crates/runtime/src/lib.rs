//! Host-side boundary for the future BulletOu cuda-oxide backend.
//!
//! This crate is intentionally isolated from the root BulletOu workspace.
//! CO-004 starts with PTX module loading and host/device smoke checks without
//! touching the existing generic Bullet backend.

pub mod loss;
pub mod nnue;
pub mod sfnn;

#[cfg(feature = "cuda")]
use std::{path::Path, sync::Arc};

#[cfg(feature = "cuda")]
pub use cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatus {
    PtxSmokeReady,
}

pub fn backend_status() -> BackendStatus {
    BackendStatus::PtxSmokeReady
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    LossLayout(#[from] loss::LossLayoutError),
    #[error(transparent)]
    NnueLayout(#[from] nnue::NnueLayoutError),
    #[error(transparent)]
    SfnnLayout(#[from] sfnn::SfnnLayoutError),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] DriverError),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Ltoir(#[from] cuda_host::LtoirError),
    #[cfg(feature = "cuda")]
    #[error("failed reading CUDA artifact metadata {path}: {source}")]
    ArtifactMetadata {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[cfg(feature = "cuda")]
    #[error("invalid CUDA artifact metadata {path}: {message}")]
    InvalidArtifactMetadata { path: std::path::PathBuf, message: String },
    #[cfg(feature = "cuda")]
    #[error("path is not valid UTF-8: {0}")]
    NonUtf8Path(String),
    #[cfg(feature = "cuda")]
    #[error("kernel symbol `{kernel}` was not found in PTX module")]
    MissingKernel {
        kernel: String,
        #[source]
        source: DriverError,
    },
    #[error("{0}")]
    Smoke(String),
    #[error("cuda feature is disabled; rebuild with `--features cuda`")]
    CudaFeatureDisabled,
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "cuda")]
pub fn load_ptx_module(ctx: &Arc<CudaContext>, ptx_path: &Path) -> Result<Arc<CudaModule>> {
    if is_nvvm_ir_path(ptx_path) {
        let arch = nvvm_ir_target_arch(ptx_path)?;
        let cubin_path = cuda_host::ltoir::build_cubin_from_ll(ptx_path, &arch)?;
        let path = cubin_path.to_str().ok_or_else(|| Error::NonUtf8Path(cubin_path.display().to_string()))?;
        return Ok(ctx.load_module_from_file(path)?);
    }

    let path = ptx_path.to_str().ok_or_else(|| Error::NonUtf8Path(ptx_path.display().to_string()))?;
    Ok(ctx.load_module_from_file(path)?)
}

#[cfg(feature = "cuda")]
fn is_nvvm_ir_path(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("ll"))
}

#[cfg(feature = "cuda")]
fn nvvm_ir_target_arch(ll_path: &Path) -> Result<String> {
    if let Ok(target) = std::env::var("CUDA_OXIDE_TARGET") {
        return Ok(target);
    }

    let target_path = ll_path.with_extension("target");
    let contents = std::fs::read_to_string(&target_path)
        .map_err(|source| Error::ArtifactMetadata { path: target_path.clone(), source })?;
    let target =
        contents.lines().map(str::trim).find(|line| !line.is_empty() && !line.contains('=')).ok_or_else(|| {
            Error::InvalidArtifactMetadata {
                path: target_path.clone(),
                message: "missing CUDA target line such as sm_89".to_string(),
            }
        })?;

    Ok(target.to_string())
}

#[cfg(feature = "cuda")]
pub fn resolve_kernel(module: &Arc<CudaModule>, kernel: &str) -> Result<CudaFunction> {
    module.load_function(kernel).map_err(|source| Error::MissingKernel { kernel: kernel.to_string(), source })
}

#[cfg(feature = "cuda")]
pub fn host_device_roundtrip(ctx: &Arc<CudaContext>, len: usize) -> Result<bool> {
    let stream = ctx.default_stream();
    let host: Vec<f32> = (0..len).map(|idx| idx as f32).collect();
    let device = DeviceBuffer::from_host(&stream, &host)?;
    let restored = device.to_host_vec(&stream)?;
    Ok(restored == host)
}

#[cfg(feature = "cuda")]
pub fn launch_zero_arg_kernel(ctx: &Arc<CudaContext>, func: &CudaFunction) -> Result<()> {
    let stream = ctx.default_stream();
    let config = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    let mut kernel_params: [*mut std::ffi::c_void; 0] = [];

    // SAFETY: this smoke launch uses a repository-local zero-argument PTX
    // kernel. The module and stream belong to the same context, and the empty
    // parameter list matches `.entry noop()`.
    unsafe {
        cuda_core::launch_kernel_on_stream(
            func,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            &mut kernel_params,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

pub fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}
