#[cfg(feature = "cuda")]
mod kernels;

#[cfg(feature = "cuda")]
mod nnue_forward;

#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(crate) use kernels::*;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(2);
    }
}

#[cfg(feature = "cuda")]
fn run() -> bulletou_cuda_oxide_runtime::Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    match args.mode {
        SmokeMode::Ptx => run_ptx_smoke(args),
        SmokeMode::NnueForward => run_nnue_forward_smoke(args),
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct Args {
    mode: SmokeMode,
    ptx: Option<std::path::PathBuf>,
    kernel: String,
    device: usize,
    tolerance: f32,
    debug_readback: bool,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmokeMode {
    Ptx,
    NnueForward,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse(raw_args: impl Iterator<Item = String>) -> bulletou_cuda_oxide_runtime::Result<Self> {
        let mut args = raw_args;
        let mut parsed = Self {
            mode: SmokeMode::Ptx,
            ptx: None,
            kernel: String::from("noop"),
            device: 0,
            tolerance: 1.0e-5,
            debug_readback: false,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--nnue-forward-smoke" => parsed.mode = SmokeMode::NnueForward,
                "--ptx" => parsed.ptx = Some(required_path_arg(&mut args, "--ptx")?),
                "--kernel" => parsed.kernel = required_arg(&mut args, "--kernel")?,
                "--device" => {
                    parsed.device = parse_usize_arg(required_arg(&mut args, "--device")?, "--device")?;
                }
                "--tolerance" => {
                    parsed.tolerance = parse_f32_arg(required_arg(&mut args, "--tolerance")?, "--tolerance")?;
                }
                "--debug-readback" => parsed.debug_readback = true,
                "--help" | "-h" => usage_success(),
                _ => usage_error(format!("unknown argument: {arg}"))?,
            }
        }

        if !(parsed.tolerance.is_finite() && parsed.tolerance >= 0.0) {
            return usage_error("--tolerance must be a finite non-negative number");
        }

        Ok(parsed)
    }
}

#[cfg(feature = "cuda")]
fn run_ptx_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    let ptx = args.ptx.unwrap_or_else(default_smoke_ptx);
    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let kernel_func = bulletou_cuda_oxide_runtime::resolve_kernel(&module, &args.kernel)?;
    bulletou_cuda_oxide_runtime::launch_zero_arg_kernel(&ctx, &kernel_func)?;
    let roundtrip_ok = bulletou_cuda_oxide_runtime::host_device_roundtrip(&ctx, 1024)?;

    println!("bulletou-cuda-train PTX smoke");
    println!("  ptx       : {}", ptx.display());
    println!("  kernel    : {}", args.kernel);
    println!("  launch    : ok");
    println!("  roundtrip : {roundtrip_ok}");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_nnue_forward_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::nnue::{
        NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardHostBatch, NnueForwardHostWeights,
        NnueForwardShape, NnueForwardWorkspace, NnueForwardWorkspaceLayout,
    };

    let case = TinyNnueForwardCase::new();
    let cpu_trace = case.cpu_forward_trace();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let shape =
        NnueForwardShape { input_size: case.shape.input_size, l1: case.shape.l1, l2: case.shape.l2, l3: case.shape.l3 };
    let host_batch = NnueForwardHostBatch {
        stm_indices: &case.stm,
        nstm_indices: &case.nstm,
        batch_size: case.batch_size,
        max_active: case.max_active,
    };
    let host_weights = NnueForwardHostWeights {
        shape,
        l0w: &case.l0w,
        l0b: &case.l0b,
        l1w: &case.l1w,
        l1b: &case.l1b,
        l2w: &case.l2w,
        l2b: &case.l2b,
        outw: &case.outw,
        outb: &case.outb,
    };
    let device_batch = NnueForwardDeviceBatch::from_host(&stream, &host_batch)?;
    let device_weights = NnueForwardDeviceWeights::from_host(&stream, &host_weights)?;
    let layout = NnueForwardWorkspaceLayout::new(shape, case.batch_size);
    let mut workspace = NnueForwardWorkspace::new(&stream, layout)?;

    nnue_forward::launch_nnue_forward(&stream, &module, &device_batch, &device_weights, &mut workspace)?;
    stream.synchronize()?;

    let gpu_outputs = workspace.output.to_host_vec(&stream)?;
    let output_cmp = compare_slices("output", &cpu_trace.outputs, &gpu_outputs, args.tolerance)?;

    println!("bulletou-cuda-train NNUE forward smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  shape        : input={} l1={} l2={} l3={}", shape.input_size, shape.l1, shape.l2, shape.l3);
    println!("  batch        : {} samples, max_active={}", case.batch_size, case.max_active);
    println!("  tolerance    : {}", args.tolerance);
    println!(
        "  output diff  : max_abs={} at {}, mean_abs={}",
        output_cmp.max_abs_diff, output_cmp.max_abs_index, output_cmp.mean_abs_diff
    );

    if args.debug_readback {
        let stm_l0 = workspace.stm_l0.to_host_vec(&stream)?;
        let nstm_l0 = workspace.nstm_l0.to_host_vec(&stream)?;
        let combined = workspace.combined.to_host_vec(&stream)?;
        let hidden1 = workspace.hidden1.to_host_vec(&stream)?;
        let hidden2 = workspace.hidden2.to_host_vec(&stream)?;
        for cmp in [
            compare_slices("stm_l0", &cpu_trace.stm_l0, &stm_l0, args.tolerance)?,
            compare_slices("nstm_l0", &cpu_trace.nstm_l0, &nstm_l0, args.tolerance)?,
            compare_slices("combined", &cpu_trace.combined, &combined, args.tolerance)?,
            compare_slices("hidden1", &cpu_trace.hidden1, &hidden1, args.tolerance)?,
            compare_slices("hidden2", &cpu_trace.hidden2, &hidden2, args.tolerance)?,
        ] {
            println!(
                "  {:<11}: max_abs={} at {}, mean_abs={}",
                cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
            );
        }
    }

    println!("  compare      : ok");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run() -> bulletou_cuda_oxide_runtime::Result<()> {
    let _ = bulletou_cuda_oxide_runtime::backend_status();
    eprintln!(
        "bulletou-cuda-train was built without CUDA support.\n\
         Rebuild with:\n  cargo run -p bulletou-cuda-train --features cuda -- [--ptx <PATH>] [--kernel <NAME>]\n\
         NNUE smoke:\n  cargo run -p bulletou-cuda-train --features cuda -- --nnue-forward-smoke [--ptx <PATH>]"
    );
    Err(bulletou_cuda_oxide_runtime::Error::CudaFeatureDisabled)
}

#[cfg(feature = "cuda")]
fn default_smoke_ptx() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../smoke/noop.ptx")
}

#[cfg(feature = "cuda")]
fn default_nnue_ptx() -> bulletou_cuda_oxide_runtime::Result<std::path::PathBuf> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().and_then(|p| p.parent()).ok_or_else(|| {
        bulletou_cuda_oxide_runtime::Error::Smoke("cannot resolve cuda-oxide workspace root".to_string())
    })?;
    let candidates = [
        manifest_dir.join("bulletou-cuda-train.ptx"),
        workspace_root.join("bulletou-cuda-train.ptx"),
        manifest_dir.join("bulletou_cuda_train.ptx"),
        workspace_root.join("bulletou_cuda_train.ptx"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
        "NNUE forward PTX not found. Run cargo-oxide for the binary crate, then pass the generated PTX with --ptx.\n\
         Probed:\n  {}",
        [
            manifest_dir.join("bulletou-cuda-train.ptx"),
            workspace_root.join("bulletou-cuda-train.ptx"),
            manifest_dir.join("bulletou_cuda_train.ptx"),
            workspace_root.join("bulletou_cuda_train.ptx"),
        ]
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n  ")
    )))
}

#[cfg(feature = "cuda")]
fn usage_success() -> ! {
    println!("{}", usage());
    std::process::exit(0);
}

#[cfg(feature = "cuda")]
fn usage_error<T>(message: impl Into<String>) -> bulletou_cuda_oxide_runtime::Result<T> {
    Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!("{}\n\n{}", message.into(), usage())))
}

#[cfg(feature = "cuda")]
fn required_arg(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<String> {
    args.next()
        .ok_or_else(|| bulletou_cuda_oxide_runtime::Error::Smoke(format!("{option} requires a value\n\n{}", usage())))
}

#[cfg(feature = "cuda")]
fn required_path_arg(
    args: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<std::path::PathBuf> {
    Ok(std::path::PathBuf::from(required_arg(args, option)?))
}

#[cfg(feature = "cuda")]
fn parse_usize_arg(value: String, option: &'static str) -> bulletou_cuda_oxide_runtime::Result<usize> {
    value.parse().map_err(|_| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!("{option} must be a non-negative integer\n\n{}", usage()))
    })
}

#[cfg(feature = "cuda")]
fn parse_f32_arg(value: String, option: &'static str) -> bulletou_cuda_oxide_runtime::Result<f32> {
    value
        .parse()
        .map_err(|_| bulletou_cuda_oxide_runtime::Error::Smoke(format!("{option} must be a number\n\n{}", usage())))
}

#[cfg(feature = "cuda")]
fn usage() -> &'static str {
    "Usage:\n\
       bulletou-cuda-train [--ptx <PATH>] [--kernel <NAME>] [--device <ID>]\n\
       bulletou-cuda-train --nnue-forward-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
     \n\
     CO-004 smoke command: load a PTX module, resolve a kernel symbol, launch a\n\
     zero-argument kernel, and verify a host-device-host buffer round trip. If\n\
     --ptx is omitted, cuda-oxide/smoke/noop.ptx is used.\n\
     \n\
     CO-006 NNUE forward smoke: build a tiny fixed NNUE batch, compare the GPU\n\
     launch_nnue_forward output against a CPU scalar golden, and fail if any\n\
     output differs by more than --tolerance (default 1e-5)."
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct TinyShape {
    input_size: usize,
    l1: usize,
    l2: usize,
    l3: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct TinyNnueForwardCase {
    shape: TinyShape,
    batch_size: usize,
    max_active: usize,
    stm: Vec<i32>,
    nstm: Vec<i32>,
    l0w: Vec<f32>,
    l0b: Vec<f32>,
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    outw: Vec<f32>,
    outb: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct TinyNnueForwardTrace {
    stm_l0: Vec<f32>,
    nstm_l0: Vec<f32>,
    combined: Vec<f32>,
    hidden1: Vec<f32>,
    hidden2: Vec<f32>,
    outputs: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl TinyNnueForwardCase {
    fn new() -> Self {
        Self {
            shape: TinyShape { input_size: 4, l1: 2, l2: 2, l3: 1 },
            batch_size: 2,
            max_active: 3,
            stm: vec![0, 1, -1, 3, -1, -1],
            nstm: vec![2, -1, -1, 1, 2, -1],
            l0w: vec![
                0.2, 0.3, // feature 0
                0.4, -0.1, // feature 1
                -0.3, 0.5, // feature 2
                0.7, 0.9, // feature 3
            ],
            l0b: vec![0.1, 0.2],
            l1w: vec![
                0.5, -0.2, // combined 0
                0.1, 0.3, // combined 1
                -0.4, 0.2, // combined 2
                0.6, 0.1, // combined 3
            ],
            l1b: vec![0.05, 0.1],
            l2w: vec![
                0.7,  // hidden1 0
                -0.2, // hidden1 1
            ],
            l2b: vec![0.2],
            outw: vec![1.5],
            outb: vec![0.05],
        }
    }

    fn cpu_forward_trace(&self) -> TinyNnueForwardTrace {
        let l0_len = self.batch_size * self.shape.l1;
        let combined_len = self.batch_size * self.shape.l1 * 2;
        let hidden1_len = self.batch_size * self.shape.l2;
        let hidden2_len = self.batch_size * self.shape.l3;
        let mut trace = TinyNnueForwardTrace {
            stm_l0: vec![0.0; l0_len],
            nstm_l0: vec![0.0; l0_len],
            combined: vec![0.0; combined_len],
            hidden1: vec![0.0; hidden1_len],
            hidden2: vec![0.0; hidden2_len],
            outputs: vec![0.0; self.batch_size],
        };

        for sample in 0..self.batch_size {
            let l0_start = sample * self.shape.l1;
            let l0_end = l0_start + self.shape.l1;
            let combined_start = sample * self.shape.l1 * 2;
            let combined_mid = combined_start + self.shape.l1;
            let combined_end = combined_start + self.shape.l1 * 2;
            let hidden1_start = sample * self.shape.l2;
            let hidden1_end = hidden1_start + self.shape.l2;
            let hidden2_start = sample * self.shape.l3;
            let hidden2_end = hidden2_start + self.shape.l3;
            let sparse_start = sample * self.max_active;
            let sparse_end = sparse_start + self.max_active;

            affine_sparse_padded(
                &self.l0w,
                &self.l0b,
                self.shape.l1,
                self.shape.input_size,
                &self.stm[sparse_start..sparse_end],
                &mut trace.stm_l0[l0_start..l0_end],
            );
            affine_sparse_padded(
                &self.l0w,
                &self.l0b,
                self.shape.l1,
                self.shape.input_size,
                &self.nstm[sparse_start..sparse_end],
                &mut trace.nstm_l0[l0_start..l0_end],
            );
            crelu_in_place(&mut trace.stm_l0[l0_start..l0_end]);
            crelu_in_place(&mut trace.nstm_l0[l0_start..l0_end]);

            trace.combined[combined_start..combined_mid].copy_from_slice(&trace.stm_l0[l0_start..l0_end]);
            trace.combined[combined_mid..combined_end].copy_from_slice(&trace.nstm_l0[l0_start..l0_end]);

            affine_dense(
                &self.l1w,
                &self.l1b,
                &trace.combined[combined_start..combined_end],
                self.shape.l2,
                &mut trace.hidden1[hidden1_start..hidden1_end],
            );
            crelu_in_place(&mut trace.hidden1[hidden1_start..hidden1_end]);

            affine_dense(
                &self.l2w,
                &self.l2b,
                &trace.hidden1[hidden1_start..hidden1_end],
                self.shape.l3,
                &mut trace.hidden2[hidden2_start..hidden2_end],
            );
            crelu_in_place(&mut trace.hidden2[hidden2_start..hidden2_end]);

            trace.outputs[sample] = dot(&self.outw, &trace.hidden2[hidden2_start..hidden2_end]) + self.outb[0];
        }

        trace
    }
}

#[cfg(feature = "cuda")]
fn affine_sparse_padded(weights: &[f32], bias: &[f32], rows: usize, cols: usize, active: &[i32], out: &mut [f32]) {
    out.copy_from_slice(&bias[..rows]);
    for &feature in active {
        if feature < 0 || feature as usize >= cols {
            continue;
        }
        let base = feature as usize * rows;
        for row in 0..rows {
            out[row] += weights[base + row];
        }
    }
}

#[cfg(feature = "cuda")]
fn affine_dense(weights: &[f32], bias: &[f32], input: &[f32], rows: usize, out: &mut [f32]) {
    out.copy_from_slice(&bias[..rows]);
    for (input_idx, &value) in input.iter().enumerate() {
        if value == 0.0 {
            continue;
        }
        let base = input_idx * rows;
        for row in 0..rows {
            out[row] += weights[base + row] * value;
        }
    }
}

#[cfg(feature = "cuda")]
fn crelu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.clamp(0.0, 1.0);
    }
}

#[cfg(feature = "cuda")]
fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&lhs, &rhs)| lhs * rhs).sum()
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct SliceComparison {
    name: &'static str,
    max_abs_diff: f32,
    max_abs_index: usize,
    mean_abs_diff: f32,
}

#[cfg(feature = "cuda")]
fn compare_slices(
    name: &'static str,
    expected: &[f32],
    actual: &[f32],
    tolerance: f32,
) -> bulletou_cuda_oxide_runtime::Result<SliceComparison> {
    if expected.len() != actual.len() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "{name} length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        )));
    }

    let mut max_abs_diff = 0.0_f32;
    let mut max_abs_index = 0usize;
    let mut sum_abs_diff = 0.0_f32;
    for (idx, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let abs_diff = (expected - actual).abs();
        sum_abs_diff += abs_diff;
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            max_abs_index = idx;
        }
        if abs_diff > tolerance {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "{name}[{idx}] mismatch: expected {expected}, got {actual}, abs_diff={abs_diff}, tolerance={tolerance}"
            )));
        }
    }

    let mean_abs_diff = if expected.is_empty() { 0.0 } else { sum_abs_diff / expected.len() as f32 };

    Ok(SliceComparison { name, max_abs_diff, max_abs_index, mean_abs_diff })
}
