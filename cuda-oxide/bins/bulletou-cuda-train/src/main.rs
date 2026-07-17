#![cfg_attr(feature = "cuda", allow(internal_features))]
#![cfg_attr(feature = "cuda", feature(core_intrinsics))]

#[cfg(feature = "cuda")]
mod dense_backward;

#[cfg(feature = "cuda")]
mod kernels;

#[cfg(feature = "cuda")]
mod loss_forward;

#[cfg(feature = "cuda")]
mod nnue_forward;

#[cfg(feature = "cuda")]
mod optimizer_update;

#[cfg(feature = "cuda")]
mod sfnn_backward;

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
        SmokeMode::Loss => run_loss_smoke(args),
        SmokeMode::DenseCReluBackward => run_dense_crelu_backward_smoke(args),
        SmokeMode::DenseOutputBackward => run_dense_output_backward_smoke(args),
        SmokeMode::NnueDenseBackward => run_nnue_dense_backward_smoke(args),
        SmokeMode::NnueForward => run_nnue_forward_smoke(args),
        SmokeMode::AdamWUpdate => run_adamw_update_smoke(args),
        SmokeMode::SfnnOutputBackward => run_sfnn_output_backward_smoke(args),
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
    loss_kind: LossKind,
    loss_case: LossCaseKind,
    nnue_case: NnueForwardCaseKind,
    sfnn_case: SfnnForwardCaseKind,
    nnue_forward_fixture: Option<std::path::PathBuf>,
    write_nnue_forward_fixture: Option<std::path::PathBuf>,
    sfnn_forward_fixture: Option<std::path::PathBuf>,
    write_sfnn_forward_fixture: Option<std::path::PathBuf>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmokeMode {
    Ptx,
    Loss,
    DenseCReluBackward,
    DenseOutputBackward,
    NnueDenseBackward,
    NnueForward,
    AdamWUpdate,
    SfnnOutputBackward,
    SfnnForward,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LossKind {
    SigmoidMse,
    NnuePytorchWrm,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LossCaseKind {
    Tiny,
    Weighted,
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
            loss_kind: LossKind::SigmoidMse,
            loss_case: LossCaseKind::Tiny,
            nnue_case: NnueForwardCaseKind::Tiny,
            sfnn_case: SfnnForwardCaseKind::Tiny,
            nnue_forward_fixture: None,
            write_nnue_forward_fixture: None,
            sfnn_forward_fixture: None,
            write_sfnn_forward_fixture: None,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--loss-smoke" => parsed.mode = SmokeMode::Loss,
                "--dense-crelu-backward-smoke" => parsed.mode = SmokeMode::DenseCReluBackward,
                "--dense-output-backward-smoke" => parsed.mode = SmokeMode::DenseOutputBackward,
                "--nnue-dense-backward-smoke" => parsed.mode = SmokeMode::NnueDenseBackward,
                "--nnue-forward-smoke" => parsed.mode = SmokeMode::NnueForward,
                "--adamw-update-smoke" => parsed.mode = SmokeMode::AdamWUpdate,
                "--sfnn-dense-backward-smoke" => parsed.mode = SmokeMode::SfnnOutputBackward,
                "--sfnn-output-backward-smoke" => parsed.mode = SmokeMode::SfnnOutputBackward,
                "--sfnn-forward-smoke" => parsed.mode = SmokeMode::SfnnForward,
                "--ptx" => parsed.ptx = Some(required_path_arg(&mut args, "--ptx")?),
                "--kernel" => parsed.kernel = required_arg(&mut args, "--kernel")?,
                "--device" => {
                    parsed.device = parse_usize_arg(required_arg(&mut args, "--device")?, "--device")?;
                }
                "--tolerance" => {
                    parsed.tolerance = parse_f32_arg(required_arg(&mut args, "--tolerance")?, "--tolerance")?;
                }
                "--loss-kind" => {
                    parsed.loss_kind = parse_loss_kind(required_arg(&mut args, "--loss-kind")?)?;
                }
                "--loss-case" => {
                    parsed.loss_case = parse_loss_case(required_arg(&mut args, "--loss-case")?)?;
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
                "--sfnn-forward-fixture" => {
                    parsed.sfnn_forward_fixture = Some(required_path_arg(&mut args, "--sfnn-forward-fixture")?);
                }
                "--write-sfnn-forward-fixture" => {
                    parsed.write_sfnn_forward_fixture =
                        Some(required_path_arg(&mut args, "--write-sfnn-forward-fixture")?);
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
fn run_dense_crelu_backward_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{backward::DenseCReluBackwardLayout, DeviceBuffer};

    let case = DenseCReluBackwardCase::tiny();
    let cpu_trace = case.cpu_backward_trace();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let layout = DenseCReluBackwardLayout::new(case.batch_size, case.input_dim, case.output_dim);
    let inputs = DeviceBuffer::from_host(&stream, &case.inputs)?;
    let activations = DeviceBuffer::from_host(&stream, &case.activations)?;
    let output_gradients = DeviceBuffer::from_host(&stream, &case.output_gradients)?;
    let weights = DeviceBuffer::from_host(&stream, &case.weights)?;
    let mut input_gradients = DeviceBuffer::<f32>::zeroed(&stream, layout.input_gradients_len())?;
    let mut weight_gradients = DeviceBuffer::<f32>::zeroed(&stream, layout.weight_len())?;
    let mut bias_gradients = DeviceBuffer::<f32>::zeroed(&stream, layout.bias_len())?;

    dense_backward::launch_dense_crelu_backward(
        &stream,
        &module,
        layout,
        &inputs,
        &activations,
        &output_gradients,
        &weights,
        &mut input_gradients,
        &mut weight_gradients,
        &mut bias_gradients,
    )?;
    stream.synchronize()?;

    let gpu_input_gradients = input_gradients.to_host_vec(&stream)?;
    let gpu_weight_gradients = weight_gradients.to_host_vec(&stream)?;
    let gpu_bias_gradients = bias_gradients.to_host_vec(&stream)?;
    let input_cmp = compare_slices("input_grad", &cpu_trace.input_gradients, &gpu_input_gradients, args.tolerance)?;
    let weight_cmp = compare_slices("weight_grad", &cpu_trace.weight_gradients, &gpu_weight_gradients, args.tolerance)?;
    let bias_cmp = compare_slices("bias_grad", &cpu_trace.bias_gradients, &gpu_bias_gradients, args.tolerance)?;

    println!("bulletou-cuda-train dense CReLU backward smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!("  batch        : {} samples", case.batch_size);
    println!("  input_dim    : {}", case.input_dim);
    println!("  output_dim   : {}", case.output_dim);
    println!("  tolerance    : {}", args.tolerance);
    println!(
        "  {:<11}: max_abs={} at {}, mean_abs={}",
        input_cmp.name, input_cmp.max_abs_diff, input_cmp.max_abs_index, input_cmp.mean_abs_diff
    );
    println!(
        "  {:<11}: max_abs={} at {}, mean_abs={}",
        weight_cmp.name, weight_cmp.max_abs_diff, weight_cmp.max_abs_index, weight_cmp.mean_abs_diff
    );
    println!(
        "  {:<11}: max_abs={} at {}, mean_abs={}",
        bias_cmp.name, bias_cmp.max_abs_diff, bias_cmp.max_abs_index, bias_cmp.mean_abs_diff
    );
    println!("  compare      : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_dense_output_backward_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{backward::DenseOutputBackwardLayout, DeviceBuffer};

    let case = DenseOutputBackwardCase::tiny();
    let cpu_trace = case.cpu_backward_trace();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let layout = DenseOutputBackwardLayout::new(case.batch_size, case.input_len);
    let inputs = DeviceBuffer::from_host(&stream, &case.inputs)?;
    let output_gradients = DeviceBuffer::from_host(&stream, &case.output_gradients)?;
    let weights = DeviceBuffer::from_host(&stream, &case.weights)?;
    let mut input_gradients = DeviceBuffer::<f32>::zeroed(&stream, layout.input_gradients_len())?;
    let mut weight_gradients = DeviceBuffer::<f32>::zeroed(&stream, layout.weight_len())?;
    let mut bias_gradient = DeviceBuffer::<f32>::zeroed(&stream, layout.bias_len())?;

    dense_backward::launch_dense_output_backward(
        &stream,
        &module,
        layout,
        &inputs,
        &output_gradients,
        &weights,
        &mut input_gradients,
        &mut weight_gradients,
        &mut bias_gradient,
    )?;
    stream.synchronize()?;

    let gpu_input_gradients = input_gradients.to_host_vec(&stream)?;
    let gpu_weight_gradients = weight_gradients.to_host_vec(&stream)?;
    let gpu_bias_gradient = bias_gradient.to_host_vec(&stream)?;
    let input_cmp = compare_slices("input_grad", &cpu_trace.input_gradients, &gpu_input_gradients, args.tolerance)?;
    let weight_cmp = compare_slices("weight_grad", &cpu_trace.weight_gradients, &gpu_weight_gradients, args.tolerance)?;
    let bias_cmp = compare_slices("bias_grad", &[cpu_trace.bias_gradient], &gpu_bias_gradient, args.tolerance)?;

    println!("bulletou-cuda-train dense output backward smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!("  batch        : {} samples", case.batch_size);
    println!("  input_len    : {}", case.input_len);
    println!("  tolerance    : {}", args.tolerance);
    println!(
        "  {:<11}: max_abs={} at {}, mean_abs={}",
        input_cmp.name, input_cmp.max_abs_diff, input_cmp.max_abs_index, input_cmp.mean_abs_diff
    );
    println!(
        "  {:<11}: max_abs={} at {}, mean_abs={}",
        weight_cmp.name, weight_cmp.max_abs_diff, weight_cmp.max_abs_index, weight_cmp.mean_abs_diff
    );
    println!(
        "  {:<11}: max_abs={} at {}, mean_abs={}",
        bias_cmp.name, bias_cmp.max_abs_diff, bias_cmp.max_abs_index, bias_cmp.mean_abs_diff
    );
    println!("  compare      : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_nnue_dense_backward_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{
        backward::{
            DenseCReluBackwardLayout, DenseOutputBackwardLayout, NnueL0CReluBackwardLayout, NnueL0SparseBackwardLayout,
        },
        nnue::{
            NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardHostBatch, NnueForwardHostWeights,
            NnueForwardWorkspace, NnueForwardWorkspaceLayout,
        },
        DeviceBuffer,
    };
    let case = match args.nnue_forward_fixture.as_deref() {
        Some(path) => NnueForwardCase::read_fixture(path)?,
        None => NnueForwardCase::new(args.nnue_case),
    };
    let cpu_forward_trace = case.cpu_forward_trace();
    let cpu_trace = case.cpu_dense_backward_trace(&cpu_forward_trace);
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let host_batch = NnueForwardHostBatch {
        stm_indices: &case.stm,
        nstm_indices: &case.nstm,
        batch_size: case.batch_size,
        max_active: case.max_active,
    };
    let host_weights = NnueForwardHostWeights {
        shape: case.shape,
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
    let forward_layout = NnueForwardWorkspaceLayout::new(case.shape, case.batch_size);
    let mut forward_workspace = NnueForwardWorkspace::new(&stream, forward_layout)?;

    nnue_forward::launch_nnue_forward(&stream, &module, &device_batch, &device_weights, &mut forward_workspace)?;

    let output_gradients = DeviceBuffer::from_host(&stream, &cpu_trace.output_gradients)?;

    let output_layout = DenseOutputBackwardLayout::new(case.batch_size, case.shape.l3);
    let mut hidden2_gradients = DeviceBuffer::<f32>::zeroed(&stream, output_layout.input_gradients_len())?;
    let mut outw_gradients = DeviceBuffer::<f32>::zeroed(&stream, output_layout.weight_len())?;
    let mut outb_gradient = DeviceBuffer::<f32>::zeroed(&stream, output_layout.bias_len())?;
    dense_backward::launch_dense_output_backward(
        &stream,
        &module,
        output_layout,
        &forward_workspace.hidden2,
        &output_gradients,
        &device_weights.outw,
        &mut hidden2_gradients,
        &mut outw_gradients,
        &mut outb_gradient,
    )?;

    let l2_layout = DenseCReluBackwardLayout::new(case.batch_size, case.shape.l2, case.shape.l3);
    let mut hidden1_gradients = DeviceBuffer::<f32>::zeroed(&stream, l2_layout.input_gradients_len())?;
    let mut l2w_gradients = DeviceBuffer::<f32>::zeroed(&stream, l2_layout.weight_len())?;
    let mut l2b_gradients = DeviceBuffer::<f32>::zeroed(&stream, l2_layout.bias_len())?;
    dense_backward::launch_dense_crelu_backward(
        &stream,
        &module,
        l2_layout,
        &forward_workspace.hidden1,
        &forward_workspace.hidden2,
        &hidden2_gradients,
        &device_weights.l2w,
        &mut hidden1_gradients,
        &mut l2w_gradients,
        &mut l2b_gradients,
    )?;

    let l1_layout = DenseCReluBackwardLayout::new(case.batch_size, case.shape.l1 * 2, case.shape.l2);
    let mut combined_gradients = DeviceBuffer::<f32>::zeroed(&stream, l1_layout.input_gradients_len())?;
    let mut l1w_gradients = DeviceBuffer::<f32>::zeroed(&stream, l1_layout.weight_len())?;
    let mut l1b_gradients = DeviceBuffer::<f32>::zeroed(&stream, l1_layout.bias_len())?;
    dense_backward::launch_dense_crelu_backward(
        &stream,
        &module,
        l1_layout,
        &forward_workspace.combined,
        &forward_workspace.hidden1,
        &hidden1_gradients,
        &device_weights.l1w,
        &mut combined_gradients,
        &mut l1w_gradients,
        &mut l1b_gradients,
    )?;

    let l0_layout = NnueL0CReluBackwardLayout::new(case.batch_size, case.shape.l1);
    let mut stm_l0_gradients = DeviceBuffer::<f32>::zeroed(&stream, l0_layout.per_perspective_len())?;
    let mut nstm_l0_gradients = DeviceBuffer::<f32>::zeroed(&stream, l0_layout.per_perspective_len())?;
    dense_backward::launch_nnue_l0_crelu_backward(
        &stream,
        &module,
        l0_layout,
        &combined_gradients,
        &forward_workspace.stm_l0,
        &forward_workspace.nstm_l0,
        &mut stm_l0_gradients,
        &mut nstm_l0_gradients,
    )?;

    let sparse_l0_layout =
        NnueL0SparseBackwardLayout::new(case.batch_size, case.max_active, case.shape.input_size, case.shape.l1);
    let mut l0w_gradients = DeviceBuffer::<f32>::zeroed(&stream, sparse_l0_layout.weight_len())?;
    let mut l0b_gradients = DeviceBuffer::<f32>::zeroed(&stream, sparse_l0_layout.bias_len())?;
    dense_backward::launch_nnue_l0_sparse_backward(
        &stream,
        &module,
        sparse_l0_layout,
        &device_batch.stm_indices,
        &device_batch.nstm_indices,
        &stm_l0_gradients,
        &nstm_l0_gradients,
        &mut l0w_gradients,
        &mut l0b_gradients,
    )?;
    stream.synchronize()?;

    let gpu_hidden2_gradients = hidden2_gradients.to_host_vec(&stream)?;
    let gpu_hidden1_gradients = hidden1_gradients.to_host_vec(&stream)?;
    let gpu_combined_gradients = combined_gradients.to_host_vec(&stream)?;
    let gpu_stm_l0_gradients = stm_l0_gradients.to_host_vec(&stream)?;
    let gpu_nstm_l0_gradients = nstm_l0_gradients.to_host_vec(&stream)?;
    let gpu_l0w_gradients = l0w_gradients.to_host_vec(&stream)?;
    let gpu_l0b_gradients = l0b_gradients.to_host_vec(&stream)?;
    let gpu_outw_gradients = outw_gradients.to_host_vec(&stream)?;
    let gpu_outb_gradient = outb_gradient.to_host_vec(&stream)?;
    let gpu_l2w_gradients = l2w_gradients.to_host_vec(&stream)?;
    let gpu_l2b_gradients = l2b_gradients.to_host_vec(&stream)?;
    let gpu_l1w_gradients = l1w_gradients.to_host_vec(&stream)?;
    let gpu_l1b_gradients = l1b_gradients.to_host_vec(&stream)?;

    let comparisons = [
        compare_slices("hidden2_grad", &cpu_trace.hidden2_gradients, &gpu_hidden2_gradients, args.tolerance)?,
        compare_slices("hidden1_grad", &cpu_trace.hidden1_gradients, &gpu_hidden1_gradients, args.tolerance)?,
        compare_slices("combined_grad", &cpu_trace.combined_gradients, &gpu_combined_gradients, args.tolerance)?,
        compare_slices("stm_l0_grad", &cpu_trace.stm_l0_gradients, &gpu_stm_l0_gradients, args.tolerance)?,
        compare_slices("nstm_l0_grad", &cpu_trace.nstm_l0_gradients, &gpu_nstm_l0_gradients, args.tolerance)?,
        compare_slices("l0w_grad", &cpu_trace.l0w_gradients, &gpu_l0w_gradients, args.tolerance)?,
        compare_slices("l0b_grad", &cpu_trace.l0b_gradients, &gpu_l0b_gradients, args.tolerance)?,
        compare_slices("outw_grad", &cpu_trace.outw_gradients, &gpu_outw_gradients, args.tolerance)?,
        compare_slices("outb_grad", &[cpu_trace.outb_gradient], &gpu_outb_gradient, args.tolerance)?,
        compare_slices("l2w_grad", &cpu_trace.l2w_gradients, &gpu_l2w_gradients, args.tolerance)?,
        compare_slices("l2b_grad", &cpu_trace.l2b_gradients, &gpu_l2b_gradients, args.tolerance)?,
        compare_slices("l1w_grad", &cpu_trace.l1w_gradients, &gpu_l1w_gradients, args.tolerance)?,
        compare_slices("l1b_grad", &cpu_trace.l1b_gradients, &gpu_l1b_gradients, args.tolerance)?,
    ];

    println!("bulletou-cuda-train NNUE dense backward smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!("  batch        : {} samples", case.batch_size);
    println!(
        "  shape        : input={} l1={} l2={} l3={}",
        case.shape.input_size, case.shape.l1, case.shape.l2, case.shape.l3
    );
    println!("  tolerance    : {}", args.tolerance);
    for cmp in comparisons {
        println!(
            "  {:<13}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
    }
    println!("  compare      : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_loss_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::loss::{
        ScalarLossDeviceBatch, ScalarLossHostBatch, ScalarLossLayout, ScalarLossWorkspace,
    };

    let case = LossSmokeCase::new(args.loss_case);
    let cpu_trace = case.cpu_loss_trace(args.loss_kind);
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let host_batch = ScalarLossHostBatch {
        outputs: &case.outputs,
        targets: &case.targets,
        entry_weights: &case.entry_weights,
        batch_size: case.outputs.len(),
    };
    let device_batch = ScalarLossDeviceBatch::from_host(&stream, &host_batch)?;
    let layout = ScalarLossLayout::new(case.outputs.len());
    let mut workspace = ScalarLossWorkspace::new(&stream, layout)?;

    match args.loss_kind {
        LossKind::SigmoidMse => loss_forward::launch_sigmoid_mse_loss(&stream, &module, &device_batch, &mut workspace)?,
        LossKind::NnuePytorchWrm => {
            loss_forward::launch_nnue_pytorch_wrm_loss(&stream, &module, &device_batch, &mut workspace)?
        }
    }
    stream.synchronize()?;

    let gpu_sum = workspace.weighted_sum.to_host_vec(&stream)?;
    let gpu_mean = workspace.mean.to_host_vec(&stream)?;
    let sum_cmp = compare_slices("weighted_sum", &[cpu_trace.weighted_sum], &gpu_sum, args.tolerance)?;
    let mean_cmp = compare_slices("mean", &[cpu_trace.mean], &gpu_mean, args.tolerance)?;

    println!("bulletou-cuda-train scalar loss smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  loss_kind    : {}", loss_kind_label(args.loss_kind));
    println!("  case         : {}", case.label);
    println!("  batch        : {} samples", case.outputs.len());
    println!("  tolerance    : {}", args.tolerance);
    println!(
        "  sum diff     : max_abs={} at {}, mean_abs={}",
        sum_cmp.max_abs_diff, sum_cmp.max_abs_index, sum_cmp.mean_abs_diff
    );
    println!(
        "  mean diff    : max_abs={} at {}, mean_abs={}",
        mean_cmp.max_abs_diff, mean_cmp.max_abs_index, mean_cmp.mean_abs_diff
    );

    if args.debug_readback {
        let gpu_per_sample = workspace.per_sample.to_host_vec(&stream)?;
        let cmp = compare_slices("per_sample", &cpu_trace.per_sample, &gpu_per_sample, args.tolerance)?;
        println!(
            "  {:<11}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
        let gpu_gradients = workspace.mean_output_gradients.to_host_vec(&stream)?;
        let cmp = compare_slices("mean_grad", &cpu_trace.mean_output_gradients, &gpu_gradients, args.tolerance)?;
        println!(
            "  {:<11}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
    }

    println!("  compare      : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_adamw_update_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{DeviceBuffer, optimizer::AdamWUpdateLayout};

    let case = AdamWUpdateCase::tiny();
    let cpu_trace = case.cpu_update_trace();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let layout = AdamWUpdateLayout::new(case.weights.len());
    let gradients = DeviceBuffer::from_host(&stream, &case.gradients)?;
    let mut weights = DeviceBuffer::from_host(&stream, &case.weights)?;
    let mut momentum = DeviceBuffer::from_host(&stream, &case.momentum)?;
    let mut velocity = DeviceBuffer::from_host(&stream, &case.velocity)?;

    optimizer_update::launch_adamw_update(
        &stream,
        &module,
        layout,
        case.params,
        &gradients,
        &mut weights,
        &mut momentum,
        &mut velocity,
    )?;
    stream.synchronize()?;

    let gpu_weights = weights.to_host_vec(&stream)?;
    let gpu_momentum = momentum.to_host_vec(&stream)?;
    let gpu_velocity = velocity.to_host_vec(&stream)?;
    let comparisons = [
        compare_slices("weights", &cpu_trace.weights, &gpu_weights, args.tolerance)?,
        compare_slices("momentum", &cpu_trace.momentum, &gpu_momentum, args.tolerance)?,
        compare_slices("velocity", &cpu_trace.velocity, &gpu_velocity, args.tolerance)?,
    ];

    println!("bulletou-cuda-train AdamW update smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!("  len          : {}", case.weights.len());
    println!("  tolerance    : {}", args.tolerance);
    println!(
        "  params       : lr={} decay={} beta1={} beta2={} eps={} clamp=[{}, {}] grad_factor={}",
        case.params.learning_rate,
        case.params.decay,
        case.params.beta1,
        case.params.beta2,
        case.params.epsilon,
        case.params.min_weight,
        case.params.max_weight,
        case.params.gradient_factor
    );
    for cmp in comparisons {
        println!(
            "  {:<11}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
    }
    if args.debug_readback {
        println!("  cpu weights  : {:?}", cpu_trace.weights);
        println!("  gpu weights  : {:?}", gpu_weights);
    }
    println!("  compare      : ok");

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

    let case = match &args.sfnn_forward_fixture {
        Some(path) => SfnnForwardCase::read_fixture(path)?,
        None => SfnnForwardCase::new(args.sfnn_case),
    };
    if let Some(path) = &args.write_sfnn_forward_fixture {
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

#[cfg(feature = "cuda")]
fn run_sfnn_output_backward_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{
        backward::{
            SfnnBackwardWorkspace, SfnnBackwardWorkspaceLayout, SfnnL0SparseBackwardLayout, SfnnL2InputBackwardLayout,
            SfnnPairwiseBackwardLayout, SfnnStackedAffineBackwardLayout, SfnnStackedCReluBackwardLayout,
            SfnnStackedL3BackwardLayout,
        },
        sfnn::{
            SfnnForwardDeviceBatch, SfnnForwardDeviceWeights, SfnnForwardHostBatch, SfnnForwardHostWeights,
            SfnnForwardWorkspace, SfnnForwardWorkspaceLayout,
        },
        DeviceBuffer,
    };

    let case = match &args.sfnn_forward_fixture {
        Some(path) => SfnnForwardCase::read_fixture(path)?,
        None => SfnnForwardCase::new(args.sfnn_case),
    };
    let cpu_forward_trace = case.cpu_forward_trace();
    let cpu_trace = case.cpu_output_backward_trace(&cpu_forward_trace);
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
    let forward_layout = SfnnForwardWorkspaceLayout::new(shape, case.batch_size);
    let mut forward_workspace = SfnnForwardWorkspace::new(&stream, forward_layout)?;
    let backward_layout = SfnnBackwardWorkspaceLayout::new(shape, case.batch_size, case.max_active);
    let mut backward_workspace = SfnnBackwardWorkspace::new(&stream, backward_layout)?;

    sfnn_forward::launch_sfnn_forward(&stream, &module, &device_batch, &device_weights, &mut forward_workspace)?;

    let layout = SfnnStackedL3BackwardLayout::new(case.batch_size, shape.l2_size, shape.l1_out(), shape.num_stacks);
    let output_gradients = DeviceBuffer::from_host(&stream, &cpu_trace.output_gradients)?;
    sfnn_backward::launch_sfnn_stacked_l3_backward(
        &stream,
        &module,
        layout,
        &forward_workspace.l2,
        &output_gradients,
        &device_weights.l3w,
        &device_batch.buckets,
        &mut backward_workspace.l2_gradients,
        &mut backward_workspace.l1_gradients,
        &mut backward_workspace.l3w_gradients,
        &mut backward_workspace.l3b_gradients,
    )?;

    let l2_layout =
        SfnnStackedCReluBackwardLayout::new(case.batch_size, shape.l2_in(), shape.l2_size, shape.num_stacks);
    sfnn_backward::launch_sfnn_stacked_crelu_backward(
        &stream,
        &module,
        l2_layout,
        &forward_workspace.l2_input,
        &forward_workspace.l2,
        &backward_workspace.l2_gradients,
        &device_weights.l2w,
        &device_batch.buckets,
        &mut backward_workspace.l2_input_gradients,
        &mut backward_workspace.l2w_gradients,
        &mut backward_workspace.l2b_gradients,
    )?;

    let l2_input_layout = SfnnL2InputBackwardLayout::new(case.batch_size, shape.l1_hidden);
    sfnn_backward::launch_sfnn_l2_input_backward(
        &stream,
        &module,
        l2_input_layout,
        &forward_workspace.l1,
        &forward_workspace.l2_input,
        &backward_workspace.l2_input_gradients,
        &mut backward_workspace.l1_gradients,
    )?;

    let l1_layout =
        SfnnStackedAffineBackwardLayout::new(case.batch_size, shape.ft_size, shape.l1_out(), shape.num_stacks);
    sfnn_backward::launch_sfnn_stacked_affine_backward(
        &stream,
        &module,
        l1_layout,
        &forward_workspace.combined,
        &backward_workspace.l1_gradients,
        &device_weights.l1w,
        &device_batch.buckets,
        &mut backward_workspace.combined_gradients,
        &mut backward_workspace.l1w_gradients,
        &mut backward_workspace.l1b_gradients,
    )?;

    let pairwise_layout = SfnnPairwiseBackwardLayout::new(case.batch_size, shape.ft_size);
    sfnn_backward::launch_sfnn_pairwise_backward(
        &stream,
        &module,
        pairwise_layout,
        &forward_workspace.stm_l0,
        &forward_workspace.nstm_l0,
        &backward_workspace.combined_gradients,
        &mut backward_workspace.stm_l0_gradients,
        &mut backward_workspace.nstm_l0_gradients,
    )?;

    let l0_layout = SfnnL0SparseBackwardLayout::new(case.batch_size, case.max_active, shape.input_size, shape.ft_size);
    sfnn_backward::launch_sfnn_l0_sparse_backward(
        &stream,
        &module,
        l0_layout,
        &device_batch.stm_indices,
        &device_batch.nstm_indices,
        &forward_workspace.stm_l0,
        &forward_workspace.nstm_l0,
        &backward_workspace.stm_l0_gradients,
        &backward_workspace.nstm_l0_gradients,
        &mut backward_workspace.stm_l0_pre_gradients,
        &mut backward_workspace.nstm_l0_pre_gradients,
        &mut backward_workspace.l0w_gradients,
        &mut backward_workspace.l0b_gradients,
    )?;
    stream.synchronize()?;

    let gpu_l2_gradients = backward_workspace.l2_gradients.to_host_vec(&stream)?;
    let gpu_l1_gradients = backward_workspace.l1_gradients.to_host_vec(&stream)?;
    let gpu_l3w_gradients = backward_workspace.l3w_gradients.to_host_vec(&stream)?;
    let gpu_l3b_gradients = backward_workspace.l3b_gradients.to_host_vec(&stream)?;
    let gpu_l2_input_gradients = backward_workspace.l2_input_gradients.to_host_vec(&stream)?;
    let gpu_l2w_gradients = backward_workspace.l2w_gradients.to_host_vec(&stream)?;
    let gpu_l2b_gradients = backward_workspace.l2b_gradients.to_host_vec(&stream)?;
    let gpu_combined_gradients = backward_workspace.combined_gradients.to_host_vec(&stream)?;
    let gpu_l1w_gradients = backward_workspace.l1w_gradients.to_host_vec(&stream)?;
    let gpu_l1b_gradients = backward_workspace.l1b_gradients.to_host_vec(&stream)?;
    let gpu_stm_l0_gradients = backward_workspace.stm_l0_gradients.to_host_vec(&stream)?;
    let gpu_nstm_l0_gradients = backward_workspace.nstm_l0_gradients.to_host_vec(&stream)?;
    let gpu_stm_l0_pre_gradients = backward_workspace.stm_l0_pre_gradients.to_host_vec(&stream)?;
    let gpu_nstm_l0_pre_gradients = backward_workspace.nstm_l0_pre_gradients.to_host_vec(&stream)?;
    let gpu_l0w_gradients = backward_workspace.l0w_gradients.to_host_vec(&stream)?;
    let gpu_l0b_gradients = backward_workspace.l0b_gradients.to_host_vec(&stream)?;
    let comparisons = [
        compare_slices("l2_grad", &cpu_trace.l2_gradients, &gpu_l2_gradients, args.tolerance)?,
        compare_slices("l1_grad", &cpu_trace.l1_gradients, &gpu_l1_gradients, args.tolerance)?,
        compare_slices("l3w_grad", &cpu_trace.l3w_gradients, &gpu_l3w_gradients, args.tolerance)?,
        compare_slices("l3b_grad", &cpu_trace.l3b_gradients, &gpu_l3b_gradients, args.tolerance)?,
        compare_slices("l2_in_grad", &cpu_trace.l2_input_gradients, &gpu_l2_input_gradients, args.tolerance)?,
        compare_slices("l2w_grad", &cpu_trace.l2w_gradients, &gpu_l2w_gradients, args.tolerance)?,
        compare_slices("l2b_grad", &cpu_trace.l2b_gradients, &gpu_l2b_gradients, args.tolerance)?,
        compare_slices("comb_grad", &cpu_trace.combined_gradients, &gpu_combined_gradients, args.tolerance)?,
        compare_slices("l1w_grad", &cpu_trace.l1w_gradients, &gpu_l1w_gradients, args.tolerance)?,
        compare_slices("l1b_grad", &cpu_trace.l1b_gradients, &gpu_l1b_gradients, args.tolerance)?,
        compare_slices("stm_l0_grad", &cpu_trace.stm_l0_gradients, &gpu_stm_l0_gradients, args.tolerance)?,
        compare_slices("nstm_l0_grad", &cpu_trace.nstm_l0_gradients, &gpu_nstm_l0_gradients, args.tolerance)?,
        compare_slices("stm_l0_pre", &cpu_trace.stm_l0_pre_gradients, &gpu_stm_l0_pre_gradients, args.tolerance)?,
        compare_slices("nstm_l0_pre", &cpu_trace.nstm_l0_pre_gradients, &gpu_nstm_l0_pre_gradients, args.tolerance)?,
        compare_slices("l0w_grad", &cpu_trace.l0w_gradients, &gpu_l0w_gradients, args.tolerance)?,
        compare_slices("l0b_grad", &cpu_trace.l0b_gradients, &gpu_l0b_gradients, args.tolerance)?,
    ];

    println!("bulletou-cuda-train SFNN dense backward smoke");
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
    for cmp in comparisons {
        println!(
            "  {:<11}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
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
        manifest_dir.join("bulletou-cuda-train.ll"),
        workspace_root.join("bulletou-cuda-train.ll"),
        manifest_dir.join("bulletou_cuda_train.ll"),
        workspace_root.join("bulletou_cuda_train.ll"),
        manifest_dir.join("bulletou-cuda-train.ptx"),
        workspace_root.join("bulletou-cuda-train.ptx"),
        manifest_dir.join("bulletou_cuda_train.ptx"),
        workspace_root.join("bulletou_cuda_train.ptx"),
        manifest_dir.join("bulletou-cuda-train.cubin"),
        workspace_root.join("bulletou-cuda-train.cubin"),
        manifest_dir.join("bulletou_cuda_train.cubin"),
        workspace_root.join("bulletou_cuda_train.cubin"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
        "CUDA kernel artifact not found. Run cargo-oxide for the binary crate, then pass the generated artifact with --ptx.\n\
         Probed:\n  {}",
        candidates
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
fn parse_loss_kind(value: String) -> bulletou_cuda_oxide_runtime::Result<LossKind> {
    match value.as_str() {
        "sigmoid-mse" | "sigmoid_mse" => Ok(LossKind::SigmoidMse),
        "wrm" | "nnue-pytorch-wrm" | "nnue_pytorch_wrm" => Ok(LossKind::NnuePytorchWrm),
        _ => usage_error(format!("--loss-kind must be one of: sigmoid-mse, wrm (got {value})")),
    }
}

#[cfg(feature = "cuda")]
fn parse_loss_case(value: String) -> bulletou_cuda_oxide_runtime::Result<LossCaseKind> {
    match value.as_str() {
        "tiny" => Ok(LossCaseKind::Tiny),
        "weighted" => Ok(LossCaseKind::Weighted),
        _ => usage_error(format!("--loss-case must be one of: tiny, weighted (got {value})")),
    }
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
        "halfka2" | "halfka2-1024-7-64-k3k3" | "SFNN_halfka2_1024_7_64_k3k3" => Ok(SfnnForwardCaseKind::Halfka2),
        _ => usage_error(format!("--sfnn-forward-case must be one of: tiny, halfka2 (got {value})")),
    }
}

#[cfg(feature = "cuda")]
fn usage() -> &'static str {
    "Usage:\n\
       bulletou-cuda-train [--ptx <PATH>] [--kernel <NAME>] [--device <ID>]\n\
       bulletou-cuda-train --loss-smoke [--loss-kind sigmoid-mse|wrm] [--loss-case tiny|weighted] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --dense-crelu-backward-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>]\n\
       bulletou-cuda-train --dense-output-backward-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>]\n\
       bulletou-cuda-train --nnue-dense-backward-smoke [--nnue-forward-case tiny|halfkp] [--nnue-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>]\n\
       bulletou-cuda-train --nnue-forward-smoke [--nnue-forward-case tiny|halfkp] [--nnue-forward-fixture <PATH>] [--write-nnue-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --adamw-update-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --sfnn-dense-backward-smoke [--sfnn-forward-case tiny|halfka2] [--sfnn-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>]\n\
       bulletou-cuda-train --sfnn-output-backward-smoke [alias of --sfnn-dense-backward-smoke]\n\
       bulletou-cuda-train --sfnn-forward-smoke [--sfnn-forward-case tiny|halfka2] [--sfnn-forward-fixture <PATH>] [--write-sfnn-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
     \n\
     CO-004 smoke command: load a PTX module, resolve a kernel symbol, launch a\n\
     zero-argument kernel, and verify a host-device-host buffer round trip. If\n\
     --ptx is omitted, cuda-oxide/smoke/noop.ptx is used.\n\
     \n\
     CO-008 loss smoke: compare the GPU sigmoid-MSE weighted loss reduction\n\
     against a CPU scalar golden. The kernel writes per-sample weighted loss,\n\
     weighted_sum, and mean (= weighted_sum / batch_size).\n\
     \n\
     CO-009 dense output backward smoke: compare scalar-output affine backward\n\
     gradients for input, weight, and bias against a CPU scalar golden.\n\
     CO-009 dense CReLU backward smoke: compare CReLU-gated dense layer\n\
     gradients for input, weight, and bias against a CPU scalar golden.\n\
     CO-009 NNUE dense backward smoke: run NNUE forward, then chain output,\n\
     hidden2, hidden1, L0 CReLU split, and sparse L0 weight/bias backward\n\
     kernels against a CPU golden.\n\
     CO-009 SFNN dense backward smoke: run SFNN forward, then chain stacked\n\
     L3 output backward, L2 CReLU backward, L2-input transform backward,\n\
     stacked L1 backward, pairwise backward, and sparse L0 CReLU backward\n\
     kernels against a CPU scalar golden.\n\
     \n\
     CO-010 AdamW update smoke: compare one fused weight/momentum/velocity\n\
     update pass against a CPU scalar golden, including decoupled weight decay\n\
     and weight clamping.\n\
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
     --sfnn-forward-case halfka2 for SFNN_halfka2_1024_7_64_k3k3.\n\
     --sfnn-forward-fixture loads the same buffers from a simple binary fixture;\n\
     --write-sfnn-forward-fixture writes the selected/generated case in that format."
}

#[cfg(feature = "cuda")]
const NNUE_FORWARD_FIXTURE_MAGIC: &[u8; 8] = b"BOUNFWD1";

#[cfg(feature = "cuda")]
const SFNN_FORWARD_FIXTURE_MAGIC: &[u8; 8] = b"BOUSFWD1";

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct AdamWUpdateCase {
    label: &'static str,
    gradients: Vec<f32>,
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    params: bulletou_cuda_oxide_runtime::optimizer::AdamWUpdateParams,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct AdamWUpdateTrace {
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl AdamWUpdateCase {
    fn tiny() -> Self {
        Self {
            label: "tiny-clamped",
            gradients: vec![0.1, -0.2, 5.0, -5.0, 0.0, 0.3, -0.4],
            weights: vec![0.5, -0.25, 1.97, -1.97, 0.0, 1.2, -1.2],
            momentum: vec![0.0, 0.01, -0.02, 0.03, 0.0, 0.2, -0.1],
            velocity: vec![0.0, 0.0004, 0.0009, 0.0016, 0.0, 0.04, 0.09],
            params: bulletou_cuda_oxide_runtime::optimizer::AdamWUpdateParams {
                gradient_factor: 0.25,
                learning_rate: 0.01,
                decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1.0e-8,
                min_weight: -1.0,
                max_weight: 1.0,
            },
        }
    }

    fn cpu_update_trace(&self) -> AdamWUpdateTrace {
        let mut weights = self.weights.clone();
        let mut momentum = self.momentum.clone();
        let mut velocity = self.velocity.clone();

        for idx in 0..weights.len() {
            let grad = self.params.gradient_factor * self.gradients[idx];
            weights[idx] *= 1.0 - self.params.decay * self.params.learning_rate;
            momentum[idx] = self.params.beta1 * momentum[idx] + (1.0 - self.params.beta1) * grad;
            velocity[idx] = self.params.beta2 * velocity[idx] + (1.0 - self.params.beta2) * grad * grad;
            weights[idx] -= self.params.learning_rate * momentum[idx] / (velocity[idx].sqrt() + self.params.epsilon);
            weights[idx] = weights[idx].clamp(self.params.min_weight, self.params.max_weight);
        }

        AdamWUpdateTrace { weights, momentum, velocity }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct DenseCReluBackwardCase {
    label: &'static str,
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    inputs: Vec<f32>,
    activations: Vec<f32>,
    output_gradients: Vec<f32>,
    weights: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct DenseCReluBackwardTrace {
    input_gradients: Vec<f32>,
    weight_gradients: Vec<f32>,
    bias_gradients: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct DenseOutputBackwardCase {
    label: &'static str,
    batch_size: usize,
    input_len: usize,
    inputs: Vec<f32>,
    output_gradients: Vec<f32>,
    weights: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct DenseOutputBackwardTrace {
    input_gradients: Vec<f32>,
    weight_gradients: Vec<f32>,
    bias_gradient: f32,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct LossSmokeCase {
    label: &'static str,
    outputs: Vec<f32>,
    targets: Vec<f32>,
    entry_weights: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct LossSmokeTrace {
    per_sample: Vec<f32>,
    mean_output_gradients: Vec<f32>,
    weighted_sum: f32,
    mean: f32,
}

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
struct NnueDenseBackwardTrace {
    output_gradients: Vec<f32>,
    hidden2_gradients: Vec<f32>,
    hidden1_gradients: Vec<f32>,
    combined_gradients: Vec<f32>,
    stm_l0_gradients: Vec<f32>,
    nstm_l0_gradients: Vec<f32>,
    l0w_gradients: Vec<f32>,
    l0b_gradients: Vec<f32>,
    outw_gradients: Vec<f32>,
    outb_gradient: f32,
    l2w_gradients: Vec<f32>,
    l2b_gradients: Vec<f32>,
    l1w_gradients: Vec<f32>,
    l1b_gradients: Vec<f32>,
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
#[derive(Debug, Clone)]
struct SfnnOutputBackwardTrace {
    output_gradients: Vec<f32>,
    l2_gradients: Vec<f32>,
    l1_gradients: Vec<f32>,
    l3w_gradients: Vec<f32>,
    l3b_gradients: Vec<f32>,
    l2_input_gradients: Vec<f32>,
    l2w_gradients: Vec<f32>,
    l2b_gradients: Vec<f32>,
    combined_gradients: Vec<f32>,
    l1w_gradients: Vec<f32>,
    l1b_gradients: Vec<f32>,
    stm_l0_gradients: Vec<f32>,
    nstm_l0_gradients: Vec<f32>,
    stm_l0_pre_gradients: Vec<f32>,
    nstm_l0_pre_gradients: Vec<f32>,
    l0w_gradients: Vec<f32>,
    l0b_gradients: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl DenseCReluBackwardCase {
    fn tiny() -> Self {
        Self {
            label: "tiny",
            batch_size: 3,
            input_dim: 4,
            output_dim: 3,
            inputs: vec![
                0.25, -0.5, 1.0, 2.0, //
                -1.5, 0.0, 0.75, -0.25, //
                3.0, -2.0, 0.5, 1.25,
            ],
            // Post-CReLU activations. Values at 0 and 1 intentionally gate
            // gradients off; interior values pass them through.
            activations: vec![
                0.2, 0.0, 0.8, //
                1.0, 0.4, 0.6, //
                0.7, 1.0, 0.0,
            ],
            output_gradients: vec![
                0.1, -0.5, 0.25, //
                -0.2, 0.3, -0.1, //
                0.05, 0.4, -0.35,
            ],
            weights: vec![
                0.5, -1.0, 0.25, //
                1.5, 0.75, -0.5, //
                -0.25, 0.6, 1.25, //
                2.0, -1.5, 0.1,
            ],
        }
    }

    fn cpu_backward_trace(&self) -> DenseCReluBackwardTrace {
        let mut input_gradients = vec![0.0_f32; self.batch_size * self.input_dim];
        let mut weight_gradients = vec![0.0_f32; self.input_dim * self.output_dim];
        let mut bias_gradients = vec![0.0_f32; self.output_dim];

        for sample in 0..self.batch_size {
            for out_col in 0..self.output_dim {
                let pre_grad = self.crelu_pre_gradient(sample, out_col);
                bias_gradients[out_col] += pre_grad;
                for in_col in 0..self.input_dim {
                    input_gradients[sample * self.input_dim + in_col] +=
                        pre_grad * self.weights[in_col * self.output_dim + out_col];
                    weight_gradients[in_col * self.output_dim + out_col] +=
                        pre_grad * self.inputs[sample * self.input_dim + in_col];
                }
            }
        }

        DenseCReluBackwardTrace { input_gradients, weight_gradients, bias_gradients }
    }

    fn crelu_pre_gradient(&self, sample: usize, out_col: usize) -> f32 {
        let idx = sample * self.output_dim + out_col;
        let activation = self.activations[idx];
        if activation > 0.0 && activation < 1.0 {
            self.output_gradients[idx]
        } else {
            0.0
        }
    }
}

#[cfg(feature = "cuda")]
impl DenseOutputBackwardCase {
    fn tiny() -> Self {
        Self {
            label: "tiny",
            batch_size: 3,
            input_len: 4,
            inputs: vec![
                0.25, -0.5, 1.0, 2.0, //
                -1.5, 0.0, 0.75, -0.25, //
                3.0, -2.0, 0.5, 1.25,
            ],
            output_gradients: vec![0.1, -0.2, 0.05],
            weights: vec![0.5, -1.0, 0.25, 2.0],
        }
    }

    fn cpu_backward_trace(&self) -> DenseOutputBackwardTrace {
        let mut input_gradients = vec![0.0_f32; self.batch_size * self.input_len];
        let mut weight_gradients = vec![0.0_f32; self.input_len];
        let mut bias_gradient = 0.0_f32;

        for sample in 0..self.batch_size {
            let out_grad = self.output_gradients[sample];
            bias_gradient += out_grad;
            for row in 0..self.input_len {
                input_gradients[sample * self.input_len + row] = out_grad * self.weights[row];
                weight_gradients[row] += out_grad * self.inputs[sample * self.input_len + row];
            }
        }

        DenseOutputBackwardTrace { input_gradients, weight_gradients, bias_gradient }
    }
}

#[cfg(feature = "cuda")]
impl LossSmokeCase {
    fn new(kind: LossCaseKind) -> Self {
        match kind {
            LossCaseKind::Tiny => Self {
                label: "tiny",
                outputs: vec![-2.0, 0.0, 2.0],
                targets: vec![0.0, 0.5, 1.0],
                entry_weights: vec![1.0, 0.5, 2.0],
            },
            LossCaseKind::Weighted => Self {
                label: "weighted",
                outputs: vec![-4.0, -1.25, 0.0, 1.5, 4.0],
                targets: vec![0.0, 0.25, 0.5, 0.75, 1.0],
                entry_weights: vec![1.0, 0.0, 0.5, 2.0, 0.25],
            },
        }
    }

    fn cpu_loss_trace(&self, kind: LossKind) -> LossSmokeTrace {
        let mut per_sample = Vec::with_capacity(self.outputs.len());
        let mut mean_output_gradients = Vec::with_capacity(self.outputs.len());
        let mut weighted_sum = 0.0_f32;
        let inv_batch = 1.0_f32 / self.outputs.len() as f32;
        for ((&output, &target), &entry_weight) in self.outputs.iter().zip(&self.targets).zip(&self.entry_weights) {
            let (loss, output_gradient) = loss_value_and_gradient(kind, output, target);
            let weighted = entry_weight * loss;
            per_sample.push(weighted);
            mean_output_gradients.push(entry_weight * output_gradient * inv_batch);
            weighted_sum += weighted;
        }
        let mean = weighted_sum / self.outputs.len() as f32;
        LossSmokeTrace { per_sample, mean_output_gradients, weighted_sum, mean }
    }
}

#[cfg(feature = "cuda")]
fn loss_kind_label(kind: LossKind) -> &'static str {
    match kind {
        LossKind::SigmoidMse => "sigmoid-mse",
        LossKind::NnuePytorchWrm => "wrm",
    }
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

    fn cpu_dense_backward_trace(&self, forward: &NnueForwardTrace) -> NnueDenseBackwardTrace {
        let output_gradients = deterministic_f32_vec(self.batch_size, 0xD15E_A5ED, 0.35, 0.0);
        let (hidden2_gradients, outw_gradients, outb_gradient) = dense_output_backward_trace(
            &forward.hidden2,
            &output_gradients,
            &self.outw,
            self.batch_size,
            self.shape.l3,
        );
        let (hidden1_gradients, l2w_gradients, l2b_gradients) = dense_crelu_backward_trace(
            &forward.hidden1,
            &forward.hidden2,
            &hidden2_gradients,
            &self.l2w,
            self.batch_size,
            self.shape.l2,
            self.shape.l3,
        );
        let (combined_gradients, l1w_gradients, l1b_gradients) = dense_crelu_backward_trace(
            &forward.combined,
            &forward.hidden1,
            &hidden1_gradients,
            &self.l1w,
            self.batch_size,
            self.shape.l1 * 2,
            self.shape.l2,
        );
        let (stm_l0_gradients, nstm_l0_gradients) = nnue_l0_crelu_backward_trace(
            &combined_gradients,
            &forward.stm_l0,
            &forward.nstm_l0,
            self.batch_size,
            self.shape.l1,
        );
        let (l0w_gradients, l0b_gradients) = nnue_l0_sparse_backward_trace(
            &self.stm,
            &self.nstm,
            &stm_l0_gradients,
            &nstm_l0_gradients,
            self.batch_size,
            self.max_active,
            self.shape.input_size,
            self.shape.l1,
        );

        NnueDenseBackwardTrace {
            output_gradients,
            hidden2_gradients,
            hidden1_gradients,
            combined_gradients,
            stm_l0_gradients,
            nstm_l0_gradients,
            l0w_gradients,
            l0b_gradients,
            outw_gradients,
            outb_gradient,
            l2w_gradients,
            l2b_gradients,
            l1w_gradients,
            l1b_gradients,
        }
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

            trace.outputs[sample] =
                affine_stacked_scalar(&self.l3w, &self.l3b, &trace.l2[l2_start..l2_end], self.shape.num_stacks, stack)
                    + trace.l1[l1_start + self.shape.l1_hidden];
        }

        trace
    }

    fn cpu_output_backward_trace(&self, forward: &SfnnForwardTrace) -> SfnnOutputBackwardTrace {
        let output_gradients = deterministic_f32_vec(self.batch_size, 0x5F33_BA6D, 0.35, 0.0);
        let (l2_gradients, l1_gradients, l3w_gradients, l3b_gradients) = sfnn_stacked_l3_backward_trace(
            &forward.l2,
            &output_gradients,
            &self.l3w,
            &self.buckets,
            self.batch_size,
            self.shape.l2_size,
            self.shape.l1_out(),
            self.shape.num_stacks,
        );
        let (l2_input_gradients, l2w_gradients, l2b_gradients) = sfnn_stacked_crelu_backward_trace(
            &forward.l2_input,
            &forward.l2,
            &l2_gradients,
            &self.l2w,
            &self.buckets,
            self.batch_size,
            self.shape.l2_in(),
            self.shape.l2_size,
            self.shape.num_stacks,
        );
        let l1_gradients = sfnn_l2_input_backward_trace(
            &forward.l1,
            &forward.l2_input,
            &l2_input_gradients,
            l1_gradients,
            self.batch_size,
            self.shape.l1_hidden,
        );
        let (combined_gradients, l1w_gradients, l1b_gradients) = sfnn_stacked_affine_backward_trace(
            &forward.combined,
            &l1_gradients,
            &self.l1w,
            &self.buckets,
            self.batch_size,
            self.shape.ft_size,
            self.shape.l1_out(),
            self.shape.num_stacks,
        );
        let (stm_l0_gradients, nstm_l0_gradients) = sfnn_pairwise_backward_trace(
            &forward.stm_l0,
            &forward.nstm_l0,
            &combined_gradients,
            self.batch_size,
            self.shape.ft_size,
        );
        let (stm_l0_pre_gradients, nstm_l0_pre_gradients, l0w_gradients, l0b_gradients) = sfnn_l0_sparse_backward_trace(
            &self.stm,
            &self.nstm,
            &forward.stm_l0,
            &forward.nstm_l0,
            &stm_l0_gradients,
            &nstm_l0_gradients,
            self.batch_size,
            self.max_active,
            self.shape.input_size,
            self.shape.ft_size,
        );

        SfnnOutputBackwardTrace {
            output_gradients,
            l2_gradients,
            l1_gradients,
            l3w_gradients,
            l3b_gradients,
            l2_input_gradients,
            l2w_gradients,
            l2b_gradients,
            combined_gradients,
            l1w_gradients,
            l1b_gradients,
            stm_l0_gradients,
            nstm_l0_gradients,
            stm_l0_pre_gradients,
            nstm_l0_pre_gradients,
            l0w_gradients,
            l0b_gradients,
        }
    }

    fn read_fixture(path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<Self> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(path).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to open SFNN forward fixture {}: {err}",
                path.display()
            ))
        })?);

        let mut magic = [0_u8; 8];
        read_exact(&mut reader, &mut magic, "fixture magic")?;
        if &magic != SFNN_FORWARD_FIXTURE_MAGIC {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "invalid SFNN forward fixture magic in {}",
                path.display()
            )));
        }

        let shape = bulletou_cuda_oxide_runtime::sfnn::SfnnForwardShape {
            input_size: read_usize(&mut reader, "shape.input_size")?,
            ft_size: read_usize(&mut reader, "shape.ft_size")?,
            l1_hidden: read_usize(&mut reader, "shape.l1_hidden")?,
            l2_size: read_usize(&mut reader, "shape.l2_size")?,
            num_stacks: read_usize(&mut reader, "shape.num_stacks")?,
        };
        let batch_size = read_usize(&mut reader, "batch_size")?;
        let max_active = read_usize(&mut reader, "max_active")?;
        let layout = bulletou_cuda_oxide_runtime::sfnn::SfnnForwardWeightLayout::new(shape);
        let sparse_len = batch_size.saturating_mul(max_active);

        let case = Self {
            label: "fixture",
            shape,
            batch_size,
            max_active,
            stm: read_i32_vec(&mut reader, sparse_len, "stm")?,
            nstm: read_i32_vec(&mut reader, sparse_len, "nstm")?,
            buckets: read_i32_vec(&mut reader, batch_size, "buckets")?,
            l0w: read_f32_vec(&mut reader, layout.l0w_len(), "l0w")?,
            l0b: read_f32_vec(&mut reader, layout.l0b_len(), "l0b")?,
            l1w: read_f32_vec(&mut reader, layout.l1w_len(), "l1w")?,
            l1b: read_f32_vec(&mut reader, layout.l1b_len(), "l1b")?,
            l2w: read_f32_vec(&mut reader, layout.l2w_len(), "l2w")?,
            l2b: read_f32_vec(&mut reader, layout.l2b_len(), "l2b")?,
            l3w: read_f32_vec(&mut reader, layout.l3w_len(), "l3w")?,
            l3b: read_f32_vec(&mut reader, layout.l3b_len(), "l3b")?,
        };

        let mut trailing = [0_u8; 1];
        match std::io::Read::read(&mut reader, &mut trailing) {
            Ok(0) => Ok(case),
            Ok(_) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "SFNN forward fixture {} has trailing bytes",
                path.display()
            ))),
            Err(err) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to read SFNN forward fixture {}: {err}",
                path.display()
            ))),
        }
    }

    fn write_fixture(&self, path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<()> {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to create SFNN forward fixture {}: {err}",
                path.display()
            ))
        })?);

        write_all(&mut writer, SFNN_FORWARD_FIXTURE_MAGIC, "fixture magic")?;
        for value in [
            self.shape.input_size,
            self.shape.ft_size,
            self.shape.l1_hidden,
            self.shape.l2_size,
            self.shape.num_stacks,
            self.batch_size,
            self.max_active,
        ] {
            write_u64(&mut writer, value as u64)?;
        }
        write_i32_vec(&mut writer, &self.stm, "stm")?;
        write_i32_vec(&mut writer, &self.nstm, "nstm")?;
        write_i32_vec(&mut writer, &self.buckets, "buckets")?;
        write_f32_vec(&mut writer, &self.l0w, "l0w")?;
        write_f32_vec(&mut writer, &self.l0b, "l0b")?;
        write_f32_vec(&mut writer, &self.l1w, "l1w")?;
        write_f32_vec(&mut writer, &self.l1b, "l1b")?;
        write_f32_vec(&mut writer, &self.l2w, "l2w")?;
        write_f32_vec(&mut writer, &self.l2b, "l2b")?;
        write_f32_vec(&mut writer, &self.l3w, "l3w")?;
        write_f32_vec(&mut writer, &self.l3b, "l3b")?;
        std::io::Write::flush(&mut writer).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to flush SFNN forward fixture {}: {err}",
                path.display()
            ))
        })?;
        Ok(())
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
fn dense_output_backward_trace(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    batch_size: usize,
    input_len: usize,
) -> (Vec<f32>, Vec<f32>, f32) {
    let mut input_gradients = vec![0.0_f32; batch_size * input_len];
    let mut weight_gradients = vec![0.0_f32; input_len];
    let mut bias_gradient = 0.0_f32;

    for sample in 0..batch_size {
        let out_grad = output_gradients[sample];
        bias_gradient += out_grad;
        for row in 0..input_len {
            input_gradients[sample * input_len + row] = out_grad * weights[row];
            weight_gradients[row] += out_grad * inputs[sample * input_len + row];
        }
    }

    (input_gradients, weight_gradients, bias_gradient)
}

#[cfg(feature = "cuda")]
fn sfnn_stacked_l3_backward_trace(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    buckets: &[i32],
    batch_size: usize,
    input_dim: usize,
    l1_out: usize,
    num_stacks: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut input_gradients = vec![0.0_f32; batch_size * input_dim];
    let mut l1_gradients = vec![0.0_f32; batch_size * l1_out];
    let mut weight_gradients = vec![0.0_f32; input_dim * num_stacks];
    let mut bias_gradients = vec![0.0_f32; num_stacks];

    for sample in 0..batch_size {
        let stack_i32 = buckets[sample];
        if stack_i32 < 0 || stack_i32 as usize >= num_stacks {
            continue;
        }
        let stack = stack_i32 as usize;
        let out_grad = output_gradients[sample];
        bias_gradients[stack] += out_grad;
        l1_gradients[sample * l1_out + l1_out - 1] = out_grad;
        for row in 0..input_dim {
            input_gradients[sample * input_dim + row] = out_grad * weights[row * num_stacks + stack];
            weight_gradients[row * num_stacks + stack] += out_grad * inputs[sample * input_dim + row];
        }
    }

    (input_gradients, l1_gradients, weight_gradients, bias_gradients)
}

#[cfg(feature = "cuda")]
fn sfnn_stacked_crelu_backward_trace(
    inputs: &[f32],
    activations: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    buckets: &[i32],
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    num_stacks: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut input_gradients = vec![0.0_f32; batch_size * input_dim];
    let stack_stride = num_stacks * output_dim;
    let mut weight_gradients = vec![0.0_f32; input_dim * stack_stride];
    let mut bias_gradients = vec![0.0_f32; stack_stride];

    for sample in 0..batch_size {
        let stack_i32 = buckets[sample];
        if stack_i32 < 0 || stack_i32 as usize >= num_stacks {
            continue;
        }
        let stack = stack_i32 as usize;
        let sample_input_start = sample * input_dim;
        let sample_output_start = sample * output_dim;

        for out_col in 0..output_dim {
            let activation_idx = sample_output_start + out_col;
            let pre_grad =
                crelu_pre_gradient_from_activation(activations[activation_idx], output_gradients[activation_idx]);
            let stacked_out_col = stack * output_dim + out_col;
            bias_gradients[stacked_out_col] += pre_grad;
            for in_col in 0..input_dim {
                let input_value = inputs[sample_input_start + in_col];
                let weight_idx = in_col * stack_stride + stacked_out_col;
                input_gradients[sample_input_start + in_col] += pre_grad * weights[weight_idx];
                weight_gradients[weight_idx] += pre_grad * input_value;
            }
        }
    }

    (input_gradients, weight_gradients, bias_gradients)
}

#[cfg(feature = "cuda")]
fn sfnn_l2_input_backward_trace(
    l1: &[f32],
    l2_input: &[f32],
    l2_input_gradients: &[f32],
    mut l1_gradients: Vec<f32>,
    batch_size: usize,
    l1_hidden: usize,
) -> Vec<f32> {
    const SCALE: f32 = 127.0 / 128.0;
    let l1_out = l1_hidden + 1;
    let l2_input_dim = l1_hidden * 2;

    for sample in 0..batch_size {
        let l1_base = sample * l1_out;
        let l2_base = sample * l2_input_dim;
        for row in 0..l1_hidden {
            let value = l1[l1_base + row];
            let square_idx = l2_base + row;
            let linear_idx = l2_base + l1_hidden + row;
            let square_grad = crelu_pre_gradient_from_activation(l2_input[square_idx], l2_input_gradients[square_idx])
                * (2.0 * value * SCALE);
            let linear_grad = crelu_pre_gradient_from_activation(l2_input[linear_idx], l2_input_gradients[linear_idx]);
            l1_gradients[l1_base + row] += square_grad + linear_grad;
        }
    }

    l1_gradients
}

#[cfg(feature = "cuda")]
fn sfnn_stacked_affine_backward_trace(
    inputs: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    buckets: &[i32],
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    num_stacks: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut input_gradients = vec![0.0_f32; batch_size * input_dim];
    let stack_stride = num_stacks * output_dim;
    let mut weight_gradients = vec![0.0_f32; input_dim * stack_stride];
    let mut bias_gradients = vec![0.0_f32; stack_stride];

    for sample in 0..batch_size {
        let stack_i32 = buckets[sample];
        if stack_i32 < 0 || stack_i32 as usize >= num_stacks {
            continue;
        }
        let stack = stack_i32 as usize;
        let sample_input_start = sample * input_dim;
        let sample_output_start = sample * output_dim;

        for out_col in 0..output_dim {
            let grad = output_gradients[sample_output_start + out_col];
            let stacked_out_col = stack * output_dim + out_col;
            bias_gradients[stacked_out_col] += grad;
            for in_col in 0..input_dim {
                let input_value = inputs[sample_input_start + in_col];
                let weight_idx = in_col * stack_stride + stacked_out_col;
                input_gradients[sample_input_start + in_col] += grad * weights[weight_idx];
                weight_gradients[weight_idx] += grad * input_value;
            }
        }
    }

    (input_gradients, weight_gradients, bias_gradients)
}

#[cfg(feature = "cuda")]
fn sfnn_pairwise_backward_trace(
    stm_l0: &[f32],
    nstm_l0: &[f32],
    combined_gradients: &[f32],
    batch_size: usize,
    ft_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    const SCALE: f32 = 127.0 / 128.0;
    let pairwise = ft_size / 2;
    let mut stm_gradients = vec![0.0_f32; batch_size * ft_size];
    let mut nstm_gradients = vec![0.0_f32; batch_size * ft_size];

    for sample in 0..batch_size {
        let l0_base = sample * ft_size;
        let combined_base = sample * ft_size;
        for col in 0..ft_size {
            let pair = col / 2;
            let mate_col = pair * 2 + (1 - (col - pair * 2));
            stm_gradients[l0_base + col] =
                combined_gradients[combined_base + pair] * stm_l0[l0_base + mate_col] * SCALE;
            nstm_gradients[l0_base + col] =
                combined_gradients[combined_base + pairwise + pair] * nstm_l0[l0_base + mate_col] * SCALE;
        }
    }

    (stm_gradients, nstm_gradients)
}

#[cfg(feature = "cuda")]
fn sfnn_l0_sparse_backward_trace(
    stm_indices: &[i32],
    nstm_indices: &[i32],
    stm_activations: &[f32],
    nstm_activations: &[f32],
    stm_output_gradients: &[f32],
    nstm_output_gradients: &[f32],
    batch_size: usize,
    max_active: usize,
    input_size: usize,
    ft_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut stm_pre_gradients = vec![0.0_f32; batch_size * ft_size];
    let mut nstm_pre_gradients = vec![0.0_f32; batch_size * ft_size];
    let mut weight_gradients = vec![0.0_f32; input_size * ft_size];
    let mut bias_gradients = vec![0.0_f32; ft_size];

    for sample in 0..batch_size {
        let sparse_start = sample * max_active;
        let gradient_start = sample * ft_size;
        for row in 0..ft_size {
            let idx = gradient_start + row;
            stm_pre_gradients[idx] =
                crelu_pre_gradient_from_activation(stm_activations[idx], stm_output_gradients[idx]);
            nstm_pre_gradients[idx] =
                crelu_pre_gradient_from_activation(nstm_activations[idx], nstm_output_gradients[idx]);
            bias_gradients[row] += stm_pre_gradients[idx] + nstm_pre_gradients[idx];
        }

        accumulate_sparse_l0_weight_gradients(
            &stm_indices[sparse_start..sparse_start + max_active],
            &stm_pre_gradients[gradient_start..gradient_start + ft_size],
            ft_size,
            input_size,
            &mut weight_gradients,
        );
        accumulate_sparse_l0_weight_gradients(
            &nstm_indices[sparse_start..sparse_start + max_active],
            &nstm_pre_gradients[gradient_start..gradient_start + ft_size],
            ft_size,
            input_size,
            &mut weight_gradients,
        );
    }

    (stm_pre_gradients, nstm_pre_gradients, weight_gradients, bias_gradients)
}

#[cfg(feature = "cuda")]
fn dense_crelu_backward_trace(
    inputs: &[f32],
    activations: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut input_gradients = vec![0.0_f32; batch_size * input_dim];
    let mut weight_gradients = vec![0.0_f32; input_dim * output_dim];
    let mut bias_gradients = vec![0.0_f32; output_dim];

    for sample in 0..batch_size {
        for out_col in 0..output_dim {
            let idx = sample * output_dim + out_col;
            let pre_grad = crelu_pre_gradient_from_activation(activations[idx], output_gradients[idx]);
            bias_gradients[out_col] += pre_grad;
            for in_col in 0..input_dim {
                input_gradients[sample * input_dim + in_col] += pre_grad * weights[in_col * output_dim + out_col];
                weight_gradients[in_col * output_dim + out_col] += pre_grad * inputs[sample * input_dim + in_col];
            }
        }
    }

    (input_gradients, weight_gradients, bias_gradients)
}

#[cfg(feature = "cuda")]
fn nnue_l0_crelu_backward_trace(
    combined_gradients: &[f32],
    stm_activations: &[f32],
    nstm_activations: &[f32],
    batch_size: usize,
    l1: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut stm_gradients = vec![0.0_f32; batch_size * l1];
    let mut nstm_gradients = vec![0.0_f32; batch_size * l1];

    for sample in 0..batch_size {
        let perspective_start = sample * l1;
        let combined_start = sample * l1 * 2;
        for row in 0..l1 {
            let perspective_idx = perspective_start + row;
            stm_gradients[perspective_idx] = crelu_pre_gradient_from_activation(
                stm_activations[perspective_idx],
                combined_gradients[combined_start + row],
            );
            nstm_gradients[perspective_idx] = crelu_pre_gradient_from_activation(
                nstm_activations[perspective_idx],
                combined_gradients[combined_start + l1 + row],
            );
        }
    }

    (stm_gradients, nstm_gradients)
}

#[cfg(feature = "cuda")]
fn nnue_l0_sparse_backward_trace(
    stm_indices: &[i32],
    nstm_indices: &[i32],
    stm_gradients: &[f32],
    nstm_gradients: &[f32],
    batch_size: usize,
    max_active: usize,
    input_size: usize,
    l1: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut weight_gradients = vec![0.0_f32; input_size * l1];
    let mut bias_gradients = vec![0.0_f32; l1];

    for sample in 0..batch_size {
        let sparse_start = sample * max_active;
        let gradient_start = sample * l1;
        for row in 0..l1 {
            let stm_grad = stm_gradients[gradient_start + row];
            let nstm_grad = nstm_gradients[gradient_start + row];
            bias_gradients[row] += stm_grad + nstm_grad;
        }

        accumulate_sparse_l0_weight_gradients(
            &stm_indices[sparse_start..sparse_start + max_active],
            &stm_gradients[gradient_start..gradient_start + l1],
            l1,
            input_size,
            &mut weight_gradients,
        );
        accumulate_sparse_l0_weight_gradients(
            &nstm_indices[sparse_start..sparse_start + max_active],
            &nstm_gradients[gradient_start..gradient_start + l1],
            l1,
            input_size,
            &mut weight_gradients,
        );
    }

    (weight_gradients, bias_gradients)
}

#[cfg(feature = "cuda")]
fn accumulate_sparse_l0_weight_gradients(
    indices: &[i32],
    gradients: &[f32],
    l1: usize,
    input_size: usize,
    weight_gradients: &mut [f32],
) {
    for &feature in indices {
        if feature < 0 || feature as usize >= input_size {
            continue;
        }
        let weight_start = feature as usize * l1;
        for row in 0..l1 {
            weight_gradients[weight_start + row] += gradients[row];
        }
    }
}

#[cfg(feature = "cuda")]
fn crelu_pre_gradient_from_activation(activation: f32, output_gradient: f32) -> f32 {
    if activation > 0.0 && activation < 1.0 {
        output_gradient
    } else {
        0.0
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
fn loss_value_and_gradient(kind: LossKind, output: f32, target: f32) -> (f32, f32) {
    match kind {
        LossKind::SigmoidMse => {
            let prediction = sigmoid(output);
            let error = prediction - target;
            let loss = error * error;
            let gradient = 2.0 * error * prediction * (1.0 - prediction);
            (loss, gradient)
        }
        LossKind::NnuePytorchWrm => nnue_pytorch_wrm_loss_and_gradient(output, target),
    }
}

#[cfg(feature = "cuda")]
fn nnue_pytorch_wrm_loss_and_gradient(output: f32, target: f32) -> (f32, f32) {
    const NNUE2SCORE: f32 = 600.0;
    const IN_OFFSET: f32 = 270.0;
    const IN_SCALING: f32 = 340.0;
    const POW_EXP: f32 = 2.5;

    let scorenet = output * NNUE2SCORE;
    let q = sigmoid((scorenet - IN_OFFSET) / IN_SCALING);
    let qm = sigmoid((-scorenet - IN_OFFSET) / IN_SCALING);
    let prediction = (1.0 + q - qm) * 0.5;
    let error = prediction - target;
    let abs_error = error.abs();
    let loss = abs_error.powf(POW_EXP);
    let q_prime = q * (1.0 - q);
    let qm_prime = qm * (1.0 - qm);
    let prediction_gradient = 0.5 * (NNUE2SCORE / IN_SCALING) * (q_prime + qm_prime);
    let loss_gradient = POW_EXP * error.signum() * abs_error.powf(POW_EXP - 1.0);
    (loss, loss_gradient * prediction_gradient)
}

#[cfg(feature = "cuda")]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
