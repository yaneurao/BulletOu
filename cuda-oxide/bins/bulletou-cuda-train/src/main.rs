#[cfg(feature = "cuda")]
mod kernels;

#[cfg(feature = "cuda")]
mod nnue_forward;

#[cfg(feature = "cuda")]
mod sfnn_forward;

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
        SmokeMode::SfnnForward => run_sfnn_forward_smoke(args),
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
    nnue_case: NnueForwardCaseKind,
    sfnn_case: SfnnForwardCaseKind,
    nnue_forward_fixture: Option<std::path::PathBuf>,
    write_nnue_forward_fixture: Option<std::path::PathBuf>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmokeMode {
    Ptx,
    NnueForward,
    SfnnForward,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NnueForwardCaseKind {
    Tiny,
    Halfkp,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SfnnForwardCaseKind {
    Tiny,
    Halfka2,
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
            nnue_case: NnueForwardCaseKind::Tiny,
            sfnn_case: SfnnForwardCaseKind::Tiny,
            nnue_forward_fixture: None,
            write_nnue_forward_fixture: None,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--nnue-forward-smoke" => parsed.mode = SmokeMode::NnueForward,
                "--sfnn-forward-smoke" => parsed.mode = SmokeMode::SfnnForward,
                "--ptx" => parsed.ptx = Some(required_path_arg(&mut args, "--ptx")?),
                "--kernel" => parsed.kernel = required_arg(&mut args, "--kernel")?,
                "--device" => {
                    parsed.device = parse_usize_arg(required_arg(&mut args, "--device")?, "--device")?;
                }
                "--tolerance" => {
                    parsed.tolerance = parse_f32_arg(required_arg(&mut args, "--tolerance")?, "--tolerance")?;
                }
                "--nnue-forward-case" => {
                    parsed.nnue_case = parse_nnue_forward_case(required_arg(&mut args, "--nnue-forward-case")?)?;
                }
                "--sfnn-forward-case" => {
                    parsed.sfnn_case = parse_sfnn_forward_case(required_arg(&mut args, "--sfnn-forward-case")?)?;
                }
                "--nnue-forward-fixture" => {
                    parsed.nnue_forward_fixture = Some(required_path_arg(&mut args, "--nnue-forward-fixture")?);
                }
                "--write-nnue-forward-fixture" => {
                    parsed.write_nnue_forward_fixture =
                        Some(required_path_arg(&mut args, "--write-nnue-forward-fixture")?);
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
        NnueForwardWorkspace, NnueForwardWorkspaceLayout,
    };

    let case = match &args.nnue_forward_fixture {
        Some(path) => NnueForwardCase::read_fixture(path)?,
        None => NnueForwardCase::new(args.nnue_case),
    };
    if let Some(path) = &args.write_nnue_forward_fixture {
        case.write_fixture(path)?;
    }
    let cpu_trace = case.cpu_forward_trace();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let shape = case.shape;
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
    println!("  case         : {}", case.label);
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

#[cfg(feature = "cuda")]
fn run_sfnn_forward_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::sfnn::{
        SfnnForwardDeviceBatch, SfnnForwardDeviceWeights, SfnnForwardHostBatch, SfnnForwardHostWeights,
        SfnnForwardWorkspace, SfnnForwardWorkspaceLayout,
    };

    let case = SfnnForwardCase::new(args.sfnn_case);
    let cpu_trace = case.cpu_forward_trace();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let shape = case.shape;
    let host_batch = SfnnForwardHostBatch {
        stm_indices: &case.stm,
        nstm_indices: &case.nstm,
        buckets: &case.buckets,
        batch_size: case.batch_size,
        max_active: case.max_active,
    };
    let host_weights = SfnnForwardHostWeights {
        shape,
        l0w: &case.l0w,
        l0b: &case.l0b,
        l1w: &case.l1w,
        l1b: &case.l1b,
        l2w: &case.l2w,
        l2b: &case.l2b,
        l3w: &case.l3w,
        l3b: &case.l3b,
    };
    let device_batch = SfnnForwardDeviceBatch::from_host(&stream, &host_batch)?;
    let device_weights = SfnnForwardDeviceWeights::from_host(&stream, &host_weights)?;
    let layout = SfnnForwardWorkspaceLayout::new(shape, case.batch_size);
    let mut workspace = SfnnForwardWorkspace::new(&stream, layout)?;

    sfnn_forward::launch_sfnn_forward(&stream, &module, &device_batch, &device_weights, &mut workspace)?;
    stream.synchronize()?;

    let gpu_outputs = workspace.output.to_host_vec(&stream)?;
    let output_cmp = compare_slices("output", &cpu_trace.outputs, &gpu_outputs, args.tolerance)?;

    println!("bulletou-cuda-train SFNN forward smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!(
        "  shape        : input={} ft={} l1_hidden={} l2={} stacks={}",
        shape.input_size, shape.ft_size, shape.l1_hidden, shape.l2_size, shape.num_stacks
    );
    println!(
        "  batch        : {} samples, max_active={}, buckets={:?}",
        case.batch_size, case.max_active, case.buckets
    );
    println!("  tolerance    : {}", args.tolerance);
    println!(
        "  output diff  : max_abs={} at {}, mean_abs={}",
        output_cmp.max_abs_diff, output_cmp.max_abs_index, output_cmp.mean_abs_diff
    );

    if args.debug_readback {
        let stm_l0 = workspace.stm_l0.to_host_vec(&stream)?;
        let nstm_l0 = workspace.nstm_l0.to_host_vec(&stream)?;
        let combined = workspace.combined.to_host_vec(&stream)?;
        let l1 = workspace.l1.to_host_vec(&stream)?;
        let l2_input = workspace.l2_input.to_host_vec(&stream)?;
        let l2 = workspace.l2.to_host_vec(&stream)?;
        for cmp in [
            compare_slices("stm_l0", &cpu_trace.stm_l0, &stm_l0, args.tolerance)?,
            compare_slices("nstm_l0", &cpu_trace.nstm_l0, &nstm_l0, args.tolerance)?,
            compare_slices("combined", &cpu_trace.combined, &combined, args.tolerance)?,
            compare_slices("l1", &cpu_trace.l1, &l1, args.tolerance)?,
            compare_slices("l2_input", &cpu_trace.l2_input, &l2_input, args.tolerance)?,
            compare_slices("l2", &cpu_trace.l2, &l2, args.tolerance)?,
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
fn parse_nnue_forward_case(value: String) -> bulletou_cuda_oxide_runtime::Result<NnueForwardCaseKind> {
    match value.as_str() {
        "tiny" => Ok(NnueForwardCaseKind::Tiny),
        "halfkp" | "halfkp-256x2-32-32" | "NNUE_HALFKP_256x2_32_32" => Ok(NnueForwardCaseKind::Halfkp),
        _ => usage_error(format!("--nnue-forward-case must be one of: tiny, halfkp (got {value})")),
    }
}

#[cfg(feature = "cuda")]
fn parse_sfnn_forward_case(value: String) -> bulletou_cuda_oxide_runtime::Result<SfnnForwardCaseKind> {
    match value.as_str() {
        "tiny" => Ok(SfnnForwardCaseKind::Tiny),
        "halfka2" | "halfka2-1024-7-64-k3k3" | "SFNN_halfka2_1024_7_64_k3k3" => {
            Ok(SfnnForwardCaseKind::Halfka2)
        }
        _ => usage_error(format!("--sfnn-forward-case must be one of: tiny, halfka2 (got {value})")),
    }
}

#[cfg(feature = "cuda")]
fn usage() -> &'static str {
    "Usage:\n\
       bulletou-cuda-train [--ptx <PATH>] [--kernel <NAME>] [--device <ID>]\n\
       bulletou-cuda-train --nnue-forward-smoke [--nnue-forward-case tiny|halfkp] [--nnue-forward-fixture <PATH>] [--write-nnue-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --sfnn-forward-smoke [--sfnn-forward-case tiny|halfka2] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
     \n\
     CO-004 smoke command: load a PTX module, resolve a kernel symbol, launch a\n\
     zero-argument kernel, and verify a host-device-host buffer round trip. If\n\
     --ptx is omitted, cuda-oxide/smoke/noop.ptx is used.\n\
     \n\
     CO-006 NNUE forward smoke: build a fixed NNUE batch, compare the GPU\n\
     launch_nnue_forward output against a CPU scalar golden, and fail if any\n\
     output differs by more than --tolerance (default 1e-5). The default case\n\
     is tiny; use --nnue-forward-case halfkp for NNUE_HALFKP_256x2_32_32.\n\
     --nnue-forward-fixture loads the same buffers from a simple binary fixture;\n\
     --write-nnue-forward-fixture writes the selected/generated case in that format.\n\
     \n\
     CO-007 SFNN forward smoke: build a fixed SFNN batch, compare the GPU\n\
     launch_sfnn_forward output against a CPU scalar golden, and fail if any\n\
     output differs by more than --tolerance. The default case is tiny; use\n\
     --sfnn-forward-case halfka2 for SFNN_halfka2_1024_7_64_k3k3."
}

#[cfg(feature = "cuda")]
const NNUE_FORWARD_FIXTURE_MAGIC: &[u8; 8] = b"BOUNFWD1";

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct NnueForwardCase {
    label: &'static str,
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
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
struct NnueForwardTrace {
    stm_l0: Vec<f32>,
    nstm_l0: Vec<f32>,
    combined: Vec<f32>,
    hidden1: Vec<f32>,
    hidden2: Vec<f32>,
    outputs: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct SfnnForwardCase {
    label: &'static str,
    shape: bulletou_cuda_oxide_runtime::sfnn::SfnnForwardShape,
    batch_size: usize,
    max_active: usize,
    stm: Vec<i32>,
    nstm: Vec<i32>,
    buckets: Vec<i32>,
    l0w: Vec<f32>,
    l0b: Vec<f32>,
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    l3w: Vec<f32>,
    l3b: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct SfnnForwardTrace {
    stm_l0: Vec<f32>,
    nstm_l0: Vec<f32>,
    combined: Vec<f32>,
    l1: Vec<f32>,
    l2_input: Vec<f32>,
    l2: Vec<f32>,
    outputs: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl NnueForwardCase {
    fn new(kind: NnueForwardCaseKind) -> Self {
        match kind {
            NnueForwardCaseKind::Tiny => Self::tiny(),
            NnueForwardCaseKind::Halfkp => Self::halfkp_256x2_32_32(),
        }
    }

    fn tiny() -> Self {
        Self {
            label: "tiny",
            shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 },
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

    fn halfkp_256x2_32_32() -> Self {
        let shape = bulletou_cuda_oxide_runtime::nnue::NNUE_HALFKP_256X2_32_32;
        let layout = bulletou_cuda_oxide_runtime::nnue::NnueForwardWeightLayout::new(shape);
        let batch_size = 2;
        let max_active = 38;
        let (stm, nstm) = deterministic_sparse_batch(batch_size, max_active, shape.input_size);

        Self {
            label: "halfkp-256x2-32-32",
            shape,
            batch_size,
            max_active,
            stm,
            nstm,
            l0w: deterministic_f32_vec(layout.l0w_len(), 0x4B1D_5EED, 0.006, 0.0),
            l0b: deterministic_f32_vec(layout.l0b_len(), 0x10B1_A5ED, 0.025, 0.12),
            l1w: deterministic_f32_vec(layout.l1w_len(), 0xC1A5_51C1, 0.0015, 0.0),
            l1b: deterministic_f32_vec(layout.l1b_len(), 0xB1A5_0010, 0.006, 0.03),
            l2w: deterministic_f32_vec(layout.l2w_len(), 0xD2A5_E002, 0.003, 0.0),
            l2b: deterministic_f32_vec(layout.l2b_len(), 0xB2A5_0020, 0.004, 0.02),
            outw: deterministic_f32_vec(layout.outw_len(), 0x0A17_0003, 0.02, 0.0),
            outb: deterministic_f32_vec(layout.outb_len(), 0x0B17_0004, 0.002, 0.01),
        }
    }

    fn cpu_forward_trace(&self) -> NnueForwardTrace {
        let l0_len = self.batch_size * self.shape.l1;
        let combined_len = self.batch_size * self.shape.l1 * 2;
        let hidden1_len = self.batch_size * self.shape.l2;
        let hidden2_len = self.batch_size * self.shape.l3;
        let mut trace = NnueForwardTrace {
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

    fn read_fixture(path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<Self> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(path).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to open NNUE forward fixture {}: {err}",
                path.display()
            ))
        })?);

        let mut magic = [0_u8; 8];
        read_exact(&mut reader, &mut magic, "fixture magic")?;
        if &magic != NNUE_FORWARD_FIXTURE_MAGIC {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "invalid NNUE forward fixture magic in {}",
                path.display()
            )));
        }

        let shape = bulletou_cuda_oxide_runtime::nnue::NnueForwardShape {
            input_size: read_usize(&mut reader, "shape.input_size")?,
            l1: read_usize(&mut reader, "shape.l1")?,
            l2: read_usize(&mut reader, "shape.l2")?,
            l3: read_usize(&mut reader, "shape.l3")?,
        };
        let batch_size = read_usize(&mut reader, "batch_size")?;
        let max_active = read_usize(&mut reader, "max_active")?;
        let layout = bulletou_cuda_oxide_runtime::nnue::NnueForwardWeightLayout::new(shape);
        let sparse_len = batch_size.saturating_mul(max_active);

        let case = Self {
            label: "fixture",
            shape,
            batch_size,
            max_active,
            stm: read_i32_vec(&mut reader, sparse_len, "stm")?,
            nstm: read_i32_vec(&mut reader, sparse_len, "nstm")?,
            l0w: read_f32_vec(&mut reader, layout.l0w_len(), "l0w")?,
            l0b: read_f32_vec(&mut reader, layout.l0b_len(), "l0b")?,
            l1w: read_f32_vec(&mut reader, layout.l1w_len(), "l1w")?,
            l1b: read_f32_vec(&mut reader, layout.l1b_len(), "l1b")?,
            l2w: read_f32_vec(&mut reader, layout.l2w_len(), "l2w")?,
            l2b: read_f32_vec(&mut reader, layout.l2b_len(), "l2b")?,
            outw: read_f32_vec(&mut reader, layout.outw_len(), "outw")?,
            outb: read_f32_vec(&mut reader, layout.outb_len(), "outb")?,
        };

        let mut trailing = [0_u8; 1];
        match std::io::Read::read(&mut reader, &mut trailing) {
            Ok(0) => Ok(case),
            Ok(_) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "NNUE forward fixture {} has trailing bytes",
                path.display()
            ))),
            Err(err) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to read NNUE forward fixture {}: {err}",
                path.display()
            ))),
        }
    }

    fn write_fixture(&self, path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<()> {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to create NNUE forward fixture {}: {err}",
                path.display()
            ))
        })?);

        write_all(&mut writer, NNUE_FORWARD_FIXTURE_MAGIC, "fixture magic")?;
        for value in
            [self.shape.input_size, self.shape.l1, self.shape.l2, self.shape.l3, self.batch_size, self.max_active]
        {
            write_u64(&mut writer, value as u64)?;
        }
        write_i32_vec(&mut writer, &self.stm, "stm")?;
        write_i32_vec(&mut writer, &self.nstm, "nstm")?;
        write_f32_vec(&mut writer, &self.l0w, "l0w")?;
        write_f32_vec(&mut writer, &self.l0b, "l0b")?;
        write_f32_vec(&mut writer, &self.l1w, "l1w")?;
        write_f32_vec(&mut writer, &self.l1b, "l1b")?;
        write_f32_vec(&mut writer, &self.l2w, "l2w")?;
        write_f32_vec(&mut writer, &self.l2b, "l2b")?;
        write_f32_vec(&mut writer, &self.outw, "outw")?;
        write_f32_vec(&mut writer, &self.outb, "outb")?;
        std::io::Write::flush(&mut writer).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to flush NNUE forward fixture {}: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl SfnnForwardCase {
    fn new(kind: SfnnForwardCaseKind) -> Self {
        match kind {
            SfnnForwardCaseKind::Tiny => Self::tiny(),
            SfnnForwardCaseKind::Halfka2 => Self::halfka2_1024_7_64_k3k3(),
        }
    }

    fn tiny() -> Self {
        Self {
            label: "tiny",
            shape: bulletou_cuda_oxide_runtime::sfnn::SfnnForwardShape {
                input_size: 4,
                ft_size: 4,
                l1_hidden: 2,
                l2_size: 2,
                num_stacks: 2,
            },
            batch_size: 2,
            max_active: 3,
            stm: vec![0, 1, -1, 3, -1, -1],
            nstm: vec![2, -1, -1, 0, 2, -1],
            buckets: vec![0, 1],
            l0w: vec![
                0.2, 0.1, -0.1, 0.0, // feature 0
                -0.1, 0.2, 0.1, 0.2, // feature 1
                0.0, -0.2, 0.2, 0.1, // feature 2
                0.3, 0.0, -0.3, 0.2, // feature 3
            ],
            l0b: vec![0.1, 0.2, 0.3, 0.4],
            l1w: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, // combined 0
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // combined 1
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // combined 2
                0.0, 0.0, 1.0, 0.0, 1.0, 0.0, // combined 3
            ],
            l1b: vec![0.0; 6],
            l2w: vec![
                1.0, 0.0, 0.0, 0.0, // l2 input 0
                0.0, 1.0, 0.0, 0.0, // l2 input 1
                1.0, 0.0, 1.0, 0.0, // l2 input 2
                0.0, 1.0, 0.0, 1.0, // l2 input 3
            ],
            l2b: vec![0.0; 4],
            l3w: vec![
                2.0, -0.5, // l2 output 0
                -1.0, 0.8, // l2 output 1
            ],
            l3b: vec![0.1, -0.02],
        }
    }

    fn halfka2_1024_7_64_k3k3() -> Self {
        let shape = bulletou_cuda_oxide_runtime::sfnn::SFNN_HALFKA2_1024_7_64_K3K3;
        let layout = bulletou_cuda_oxide_runtime::sfnn::SfnnForwardWeightLayout::new(shape);
        let batch_size = 2;
        let max_active = 40;
        let (stm, nstm) = deterministic_sparse_batch(batch_size, max_active, shape.input_size);

        Self {
            label: "halfka2-1024-7-64-k3k3",
            shape,
            batch_size,
            max_active,
            stm,
            nstm,
            buckets: vec![0, 8],
            l0w: deterministic_f32_vec(layout.l0w_len(), 0x5F23_4CB8, 0.004, 0.0),
            l0b: deterministic_f32_vec(layout.l0b_len(), 0x10B1_5F23, 0.02, 0.10),
            l1w: deterministic_f32_vec(layout.l1w_len(), 0xC1A5_5F23, 0.002, 0.0),
            l1b: deterministic_f32_vec(layout.l1b_len(), 0xB1A5_5F23, 0.004, 0.02),
            l2w: deterministic_f32_vec(layout.l2w_len(), 0xD2A5_5F23, 0.003, 0.0),
            l2b: deterministic_f32_vec(layout.l2b_len(), 0xB2A5_5F23, 0.004, 0.02),
            l3w: deterministic_f32_vec(layout.l3w_len(), 0xD3A5_5F23, 0.02, 0.0),
            l3b: deterministic_f32_vec(layout.l3b_len(), 0xB3A5_5F23, 0.002, 0.01),
        }
    }

    fn cpu_forward_trace(&self) -> SfnnForwardTrace {
        let l0_len = self.batch_size * self.shape.ft_size;
        let combined_len = self.batch_size * self.shape.ft_size;
        let l1_len = self.batch_size * self.shape.l1_out();
        let l2_input_len = self.batch_size * self.shape.l2_in();
        let l2_len = self.batch_size * self.shape.l2_size;
        let mut trace = SfnnForwardTrace {
            stm_l0: vec![0.0; l0_len],
            nstm_l0: vec![0.0; l0_len],
            combined: vec![0.0; combined_len],
            l1: vec![0.0; l1_len],
            l2_input: vec![0.0; l2_input_len],
            l2: vec![0.0; l2_len],
            outputs: vec![0.0; self.batch_size],
        };

        for sample in 0..self.batch_size {
            let stack = self.buckets[sample] as usize;
            let l0_start = sample * self.shape.ft_size;
            let l0_end = l0_start + self.shape.ft_size;
            let pairwise = self.shape.pairwise_size();
            let combined_start = sample * self.shape.ft_size;
            let combined_mid = combined_start + pairwise;
            let combined_end = combined_start + self.shape.ft_size;
            let l1_start = sample * self.shape.l1_out();
            let l1_end = l1_start + self.shape.l1_out();
            let l2_input_start = sample * self.shape.l2_in();
            let l2_input_end = l2_input_start + self.shape.l2_in();
            let l2_start = sample * self.shape.l2_size;
            let l2_end = l2_start + self.shape.l2_size;
            let sparse_start = sample * self.max_active;
            let sparse_end = sparse_start + self.max_active;

            affine_sparse_padded(
                &self.l0w,
                &self.l0b,
                self.shape.ft_size,
                self.shape.input_size,
                &self.stm[sparse_start..sparse_end],
                &mut trace.stm_l0[l0_start..l0_end],
            );
            affine_sparse_padded(
                &self.l0w,
                &self.l0b,
                self.shape.ft_size,
                self.shape.input_size,
                &self.nstm[sparse_start..sparse_end],
                &mut trace.nstm_l0[l0_start..l0_end],
            );
            crelu_in_place(&mut trace.stm_l0[l0_start..l0_end]);
            crelu_in_place(&mut trace.nstm_l0[l0_start..l0_end]);
            pairwise_mul_scaled(&trace.stm_l0[l0_start..l0_end], &mut trace.combined[combined_start..combined_mid]);
            pairwise_mul_scaled(&trace.nstm_l0[l0_start..l0_end], &mut trace.combined[combined_mid..combined_end]);

            affine_stacked(
                &self.l1w,
                &self.l1b,
                &trace.combined[combined_start..combined_end],
                self.shape.l1_out(),
                self.shape.num_stacks,
                stack,
                &mut trace.l1[l1_start..l1_end],
            );
            fill_sfnn_l2_input(
                &trace.l1[l1_start..l1_end],
                self.shape.l1_hidden,
                &mut trace.l2_input[l2_input_start..l2_input_end],
            );
            affine_stacked(
                &self.l2w,
                &self.l2b,
                &trace.l2_input[l2_input_start..l2_input_end],
                self.shape.l2_size,
                self.shape.num_stacks,
                stack,
                &mut trace.l2[l2_start..l2_end],
            );
            crelu_in_place(&mut trace.l2[l2_start..l2_end]);

            trace.outputs[sample] = affine_stacked_scalar(
                &self.l3w,
                &self.l3b,
                &trace.l2[l2_start..l2_end],
                self.shape.num_stacks,
                stack,
            ) + trace.l1[l1_start + self.shape.l1_hidden];
        }

        trace
    }
}

#[cfg(feature = "cuda")]
fn read_usize(reader: &mut impl std::io::Read, name: &'static str) -> bulletou_cuda_oxide_runtime::Result<usize> {
    let value = read_u64(reader, name)?;
    usize::try_from(value)
        .map_err(|_| bulletou_cuda_oxide_runtime::Error::Smoke(format!("{name} value {value} does not fit in usize")))
}

#[cfg(feature = "cuda")]
fn read_u64(reader: &mut impl std::io::Read, name: &'static str) -> bulletou_cuda_oxide_runtime::Result<u64> {
    let mut bytes = [0_u8; 8];
    read_exact(reader, &mut bytes, name)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(feature = "cuda")]
fn read_i32_vec(
    reader: &mut impl std::io::Read,
    len: usize,
    name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<Vec<i32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let mut bytes = [0_u8; 4];
        read_exact(reader, &mut bytes, name)?;
        values.push(i32::from_le_bytes(bytes));
    }
    Ok(values)
}

#[cfg(feature = "cuda")]
fn read_f32_vec(
    reader: &mut impl std::io::Read,
    len: usize,
    name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        let mut bytes = [0_u8; 4];
        read_exact(reader, &mut bytes, name)?;
        values.push(f32::from_le_bytes(bytes));
    }
    Ok(values)
}

#[cfg(feature = "cuda")]
fn read_exact(
    reader: &mut impl std::io::Read,
    bytes: &mut [u8],
    name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    std::io::Read::read_exact(reader, bytes)
        .map_err(|err| bulletou_cuda_oxide_runtime::Error::Smoke(format!("failed to read {name}: {err}")))
}

#[cfg(feature = "cuda")]
fn write_u64(writer: &mut impl std::io::Write, value: u64) -> bulletou_cuda_oxide_runtime::Result<()> {
    write_all(writer, &value.to_le_bytes(), "u64")
}

#[cfg(feature = "cuda")]
fn write_i32_vec(
    writer: &mut impl std::io::Write,
    values: &[i32],
    name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    for &value in values {
        write_all(writer, &value.to_le_bytes(), name)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn write_f32_vec(
    writer: &mut impl std::io::Write,
    values: &[f32],
    name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    for &value in values {
        write_all(writer, &value.to_le_bytes(), name)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn write_all(
    writer: &mut impl std::io::Write,
    bytes: &[u8],
    name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    std::io::Write::write_all(writer, bytes)
        .map_err(|err| bulletou_cuda_oxide_runtime::Error::Smoke(format!("failed to write {name}: {err}")))
}

#[cfg(feature = "cuda")]
fn deterministic_sparse_batch(batch_size: usize, max_active: usize, input_size: usize) -> (Vec<i32>, Vec<i32>) {
    let mut stm = Vec::with_capacity(batch_size * max_active);
    let mut nstm = Vec::with_capacity(batch_size * max_active);
    for sample in 0..batch_size {
        let active = max_active.saturating_sub(sample % 5);
        let nstm_active = max_active.saturating_sub((sample + 2) % 7);
        for slot in 0..max_active {
            stm.push(if slot < active {
                deterministic_feature_index(sample, slot, input_size, 0x1357_2468) as i32
            } else {
                -1
            });
            nstm.push(if slot < nstm_active {
                deterministic_feature_index(sample, slot, input_size, 0x2468_1357) as i32
            } else {
                -1
            });
        }
    }
    (stm, nstm)
}

#[cfg(feature = "cuda")]
fn deterministic_feature_index(sample: usize, slot: usize, input_size: usize, seed: u64) -> usize {
    let mixed = mix_u64(seed ^ ((sample as u64) << 32) ^ slot as u64);
    (mixed as usize) % input_size
}

#[cfg(feature = "cuda")]
fn deterministic_f32_vec(len: usize, seed: u64, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let mixed = mix_u64(seed ^ idx as u64);
            let centered = (mixed % 2001) as i32 - 1000;
            bias + centered as f32 * (scale / 1000.0)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn mix_u64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
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
fn affine_stacked(
    weights: &[f32],
    bias: &[f32],
    input: &[f32],
    rows: usize,
    num_stacks: usize,
    stack: usize,
    out: &mut [f32],
) {
    let bias_base = stack * rows;
    out.copy_from_slice(&bias[bias_base..bias_base + rows]);
    let stack_stride = num_stacks * rows;
    for (input_idx, &value) in input.iter().enumerate() {
        if value == 0.0 {
            continue;
        }
        let base = input_idx * stack_stride + stack * rows;
        for row in 0..rows {
            out[row] += weights[base + row] * value;
        }
    }
}

#[cfg(feature = "cuda")]
fn affine_stacked_scalar(weights: &[f32], bias: &[f32], input: &[f32], num_stacks: usize, stack: usize) -> f32 {
    let mut out = bias[stack];
    for (input_idx, &value) in input.iter().enumerate() {
        if value != 0.0 {
            out += weights[input_idx * num_stacks + stack] * value;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn pairwise_mul_scaled(input: &[f32], out: &mut [f32]) {
    const SCALE: f32 = 127.0 / 128.0;
    for (idx, pair) in input.chunks_exact(2).enumerate() {
        out[idx] = pair[0] * pair[1] * SCALE;
    }
}

#[cfg(feature = "cuda")]
fn fill_sfnn_l2_input(l1: &[f32], l1_hidden: usize, out: &mut [f32]) {
    const SCALE: f32 = 127.0 / 128.0;
    for row in 0..l1_hidden {
        out[row] = (l1[row].abs() * l1[row].abs() * SCALE).clamp(0.0, 1.0);
        out[l1_hidden + row] = l1[row].clamp(0.0, 1.0);
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
