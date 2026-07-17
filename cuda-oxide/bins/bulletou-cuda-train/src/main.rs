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
mod nnue_train_step;

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
        SmokeMode::NnueFixtureTrain => run_nnue_fixture_train(args),
        SmokeMode::NnueForward => run_nnue_forward_smoke(args),
        SmokeMode::NnueLossRangerStep => run_nnue_loss_ranger_step_smoke(args),
        SmokeMode::NnueRangerStep => run_nnue_ranger_step_smoke(args),
        #[cfg(feature = "root-loader")]
        SmokeMode::NnueTeacherTrain => run_nnue_teacher_train(args),
        SmokeMode::AdamWUpdate => run_adamw_update_smoke(args),
        SmokeMode::RAdamUpdate => run_radam_update_smoke(args),
        SmokeMode::RangerLookahead => run_ranger_lookahead_smoke(args),
        SmokeMode::RangerUpdate => run_ranger_update_smoke(args),
        SmokeMode::SfnnOutputBackward => run_sfnn_output_backward_smoke(args),
        SmokeMode::SfnnForward => run_sfnn_forward_smoke(args),
        SmokeMode::SfnnRangerStep => run_sfnn_ranger_step_smoke(args),
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
enum NnueTrainFixtureArg {
    Full(std::path::PathBuf),
    Batch(std::path::PathBuf),
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
    nnue_train_state_fixture: Option<std::path::PathBuf>,
    nnue_train_state_bin: Option<std::path::PathBuf>,
    nnue_train_fixture_args: Vec<NnueTrainFixtureArg>,
    weights_bin: Option<std::path::PathBuf>,
    teacher: Option<String>,
    test_teacher: Option<String>,
    test_positions: usize,
    test_batch_size: usize,
    test_seed: u64,
    output: Option<std::path::PathBuf>,
    train_steps: usize,
    save_rate: usize,
    batches_per_superbatch: usize,
    superbatches_per_epoch: usize,
    lr_schedule: TrainLrScheduleKind,
    learning_rate: f32,
    lr_min: f32,
    lr_step_gamma: f32,
    lr_step_positions: Option<u64>,
    lr_period_positions: u64,
    optimizer_weight_decay: f32,
    optimizer_epsilon: f32,
    optimizer_beta1: f32,
    optimizer_beta2: f32,
    batch_size: usize,
    buffer_mb: usize,
    loader_threads: usize,
    threads: usize,
    score_drop_abs: u16,
    write_nnue_forward_fixture: Option<std::path::PathBuf>,
    write_nnue_trained_forward_fixture: Option<std::path::PathBuf>,
    write_nnue_train_state_fixture: Option<std::path::PathBuf>,
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
    NnueFixtureTrain,
    NnueForward,
    NnueLossRangerStep,
    NnueRangerStep,
    #[cfg(feature = "root-loader")]
    NnueTeacherTrain,
    AdamWUpdate,
    RAdamUpdate,
    RangerLookahead,
    RangerUpdate,
    SfnnOutputBackward,
    SfnnForward,
    SfnnRangerStep,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LossKind {
    SigmoidMse,
    NnuePytorchWrm,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainLrScheduleKind {
    Fixed,
    Step,
    Geometric,
    Cos,
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
            nnue_train_state_fixture: None,
            nnue_train_state_bin: None,
            nnue_train_fixture_args: Vec::new(),
            weights_bin: None,
            teacher: None,
            test_teacher: None,
            test_positions: 100_000,
            test_batch_size: 1024,
            test_seed: 0,
            output: None,
            train_steps: 1,
            save_rate: 0,
            batches_per_superbatch: 1,
            superbatches_per_epoch: 0,
            lr_schedule: TrainLrScheduleKind::Fixed,
            learning_rate: 0.01,
            lr_min: 0.01,
            lr_step_gamma: 1.0,
            lr_step_positions: None,
            lr_period_positions: 0,
            optimizer_weight_decay: 0.01,
            optimizer_epsilon: 1.0e-8,
            optimizer_beta1: 0.99,
            optimizer_beta2: 0.999,
            batch_size: 2,
            buffer_mb: 1,
            loader_threads: 1,
            threads: 1,
            score_drop_abs: 32_000,
            write_nnue_forward_fixture: None,
            write_nnue_trained_forward_fixture: None,
            write_nnue_train_state_fixture: None,
            sfnn_forward_fixture: None,
            write_sfnn_forward_fixture: None,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--loss-smoke" => parsed.mode = SmokeMode::Loss,
                "--dense-crelu-backward-smoke" => parsed.mode = SmokeMode::DenseCReluBackward,
                "--dense-output-backward-smoke" => parsed.mode = SmokeMode::DenseOutputBackward,
                "--nnue-dense-backward-smoke" => parsed.mode = SmokeMode::NnueDenseBackward,
                "--nnue-fixture-train" => parsed.mode = SmokeMode::NnueFixtureTrain,
                "--nnue-forward-smoke" => parsed.mode = SmokeMode::NnueForward,
                "--nnue-loss-ranger-step-smoke" => parsed.mode = SmokeMode::NnueLossRangerStep,
                "--nnue-ranger-step-smoke" => parsed.mode = SmokeMode::NnueRangerStep,
                "--nnue-teacher-train" => {
                    #[cfg(feature = "root-loader")]
                    {
                        parsed.mode = SmokeMode::NnueTeacherTrain;
                    }
                    #[cfg(not(feature = "root-loader"))]
                    {
                        return usage_error("--nnue-teacher-train requires building with --features cuda,root-loader");
                    }
                }
                "--adamw-update-smoke" => parsed.mode = SmokeMode::AdamWUpdate,
                "--radam-update-smoke" => parsed.mode = SmokeMode::RAdamUpdate,
                "--ranger-lookahead-smoke" => parsed.mode = SmokeMode::RangerLookahead,
                "--ranger-update-smoke" => parsed.mode = SmokeMode::RangerUpdate,
                "--sfnn-dense-backward-smoke" => parsed.mode = SmokeMode::SfnnOutputBackward,
                "--sfnn-output-backward-smoke" => parsed.mode = SmokeMode::SfnnOutputBackward,
                "--sfnn-forward-smoke" => parsed.mode = SmokeMode::SfnnForward,
                "--sfnn-ranger-step-smoke" => parsed.mode = SmokeMode::SfnnRangerStep,
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
                "--nnue-train-state-fixture" => {
                    parsed.nnue_train_state_fixture = Some(required_path_arg(&mut args, "--nnue-train-state-fixture")?);
                }
                "--nnue-train-state-bin" => {
                    parsed.nnue_train_state_bin = Some(required_path_arg(&mut args, "--nnue-train-state-bin")?);
                }
                "--nnue-train-fixture" => {
                    parsed
                        .nnue_train_fixture_args
                        .push(NnueTrainFixtureArg::Full(required_path_arg(&mut args, "--nnue-train-fixture")?));
                }
                "--nnue-train-batch-fixture" => {
                    parsed
                        .nnue_train_fixture_args
                        .push(NnueTrainFixtureArg::Batch(required_path_arg(&mut args, "--nnue-train-batch-fixture")?));
                }
                "--weights-bin" => parsed.weights_bin = Some(required_path_arg(&mut args, "--weights-bin")?),
                "--teacher" => parsed.teacher = Some(required_arg(&mut args, "--teacher")?),
                "--test-teacher" => parsed.test_teacher = Some(required_arg(&mut args, "--test-teacher")?),
                "--test-positions" => {
                    parsed.test_positions =
                        parse_usize_arg(required_arg(&mut args, "--test-positions")?, "--test-positions")?;
                }
                "--test-batch-size" => {
                    parsed.test_batch_size =
                        parse_usize_arg(required_arg(&mut args, "--test-batch-size")?, "--test-batch-size")?;
                }
                "--test-seed" => {
                    parsed.test_seed = parse_u64_arg(required_arg(&mut args, "--test-seed")?, "--test-seed")?;
                }
                "--output" => parsed.output = Some(required_path_arg(&mut args, "--output")?),
                "--train-steps" => {
                    parsed.train_steps = parse_usize_arg(required_arg(&mut args, "--train-steps")?, "--train-steps")?;
                }
                "--save-rate" => {
                    parsed.save_rate = parse_usize_arg(required_arg(&mut args, "--save-rate")?, "--save-rate")?;
                }
                "--batches-per-superbatch" => {
                    parsed.batches_per_superbatch =
                        parse_usize_arg(required_arg(&mut args, "--batches-per-superbatch")?, "--batches-per-superbatch")?;
                }
                "--superbatches-per-epoch" => {
                    parsed.superbatches_per_epoch =
                        parse_usize_arg(required_arg(&mut args, "--superbatches-per-epoch")?, "--superbatches-per-epoch")?;
                }
                "--lr-schedule" => {
                    parsed.lr_schedule = parse_train_lr_schedule(required_arg(&mut args, "--lr-schedule")?)?;
                }
                "--learning-rate" | "--lr" => {
                    parsed.learning_rate = parse_f32_arg(required_arg(&mut args, "--learning-rate")?, "--learning-rate")?;
                }
                "--lr-min" => {
                    parsed.lr_min = parse_f32_arg(required_arg(&mut args, "--lr-min")?, "--lr-min")?;
                }
                "--lr-step-gamma" => {
                    parsed.lr_step_gamma = parse_f32_arg(required_arg(&mut args, "--lr-step-gamma")?, "--lr-step-gamma")?;
                }
                "--lr-step-positions" => {
                    parsed.lr_step_positions =
                        Some(parse_u64_arg(required_arg(&mut args, "--lr-step-positions")?, "--lr-step-positions")?);
                }
                "--lr-period-positions" => {
                    parsed.lr_period_positions =
                        parse_u64_arg(required_arg(&mut args, "--lr-period-positions")?, "--lr-period-positions")?;
                }
                "--optimizer-weight-decay" => {
                    parsed.optimizer_weight_decay =
                        parse_f32_arg(required_arg(&mut args, "--optimizer-weight-decay")?, "--optimizer-weight-decay")?;
                }
                "--optimizer-epsilon" => {
                    parsed.optimizer_epsilon =
                        parse_f32_arg(required_arg(&mut args, "--optimizer-epsilon")?, "--optimizer-epsilon")?;
                }
                "--optimizer-beta1" => {
                    parsed.optimizer_beta1 =
                        parse_f32_arg(required_arg(&mut args, "--optimizer-beta1")?, "--optimizer-beta1")?;
                }
                "--optimizer-beta2" => {
                    parsed.optimizer_beta2 =
                        parse_f32_arg(required_arg(&mut args, "--optimizer-beta2")?, "--optimizer-beta2")?;
                }
                "--batch-size" => {
                    parsed.batch_size = parse_usize_arg(required_arg(&mut args, "--batch-size")?, "--batch-size")?;
                }
                "--buffer-mb" => {
                    parsed.buffer_mb = parse_usize_arg(required_arg(&mut args, "--buffer-mb")?, "--buffer-mb")?;
                }
                "--loader-threads" => {
                    parsed.loader_threads =
                        parse_usize_arg(required_arg(&mut args, "--loader-threads")?, "--loader-threads")?;
                }
                "--threads" => {
                    parsed.threads = parse_usize_arg(required_arg(&mut args, "--threads")?, "--threads")?;
                }
                "--score-drop-abs" => {
                    parsed.score_drop_abs =
                        parse_u16_arg(required_arg(&mut args, "--score-drop-abs")?, "--score-drop-abs")?;
                }
                "--write-nnue-forward-fixture" => {
                    parsed.write_nnue_forward_fixture =
                        Some(required_path_arg(&mut args, "--write-nnue-forward-fixture")?);
                }
                "--write-nnue-trained-forward-fixture" => {
                    parsed.write_nnue_trained_forward_fixture =
                        Some(required_path_arg(&mut args, "--write-nnue-trained-forward-fixture")?);
                }
                "--write-nnue-train-state-fixture" => {
                    parsed.write_nnue_train_state_fixture =
                        Some(required_path_arg(&mut args, "--write-nnue-train-state-fixture")?);
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
            DenseCReluBackwardLayout, DenseOutputBackwardLayout, NnueBackwardWorkspace, NnueBackwardWorkspaceLayout,
            NnueL0CReluBackwardLayout, NnueL0SparseBackwardLayout,
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
    let backward_layout = NnueBackwardWorkspaceLayout::new(case.shape, case.batch_size, case.max_active);
    let mut backward_workspace = NnueBackwardWorkspace::new(&stream, backward_layout)?;

    let output_layout = DenseOutputBackwardLayout::new(case.batch_size, case.shape.l3);
    dense_backward::launch_dense_output_backward(
        &stream,
        &module,
        output_layout,
        &forward_workspace.hidden2,
        &output_gradients,
        &device_weights.outw,
        &mut backward_workspace.hidden2_gradients,
        &mut backward_workspace.outw_gradients,
        &mut backward_workspace.outb_gradients,
    )?;

    let l2_layout = DenseCReluBackwardLayout::new(case.batch_size, case.shape.l2, case.shape.l3);
    dense_backward::launch_dense_crelu_backward(
        &stream,
        &module,
        l2_layout,
        &forward_workspace.hidden1,
        &forward_workspace.hidden2,
        &backward_workspace.hidden2_gradients,
        &device_weights.l2w,
        &mut backward_workspace.hidden1_gradients,
        &mut backward_workspace.l2w_gradients,
        &mut backward_workspace.l2b_gradients,
    )?;

    let l1_layout = DenseCReluBackwardLayout::new(case.batch_size, case.shape.l1 * 2, case.shape.l2);
    dense_backward::launch_dense_crelu_backward(
        &stream,
        &module,
        l1_layout,
        &forward_workspace.combined,
        &forward_workspace.hidden1,
        &backward_workspace.hidden1_gradients,
        &device_weights.l1w,
        &mut backward_workspace.combined_gradients,
        &mut backward_workspace.l1w_gradients,
        &mut backward_workspace.l1b_gradients,
    )?;

    let l0_layout = NnueL0CReluBackwardLayout::new(case.batch_size, case.shape.l1);
    dense_backward::launch_nnue_l0_crelu_backward(
        &stream,
        &module,
        l0_layout,
        &backward_workspace.combined_gradients,
        &forward_workspace.stm_l0,
        &forward_workspace.nstm_l0,
        &mut backward_workspace.stm_l0_gradients,
        &mut backward_workspace.nstm_l0_gradients,
    )?;

    let sparse_l0_layout =
        NnueL0SparseBackwardLayout::new(case.batch_size, case.max_active, case.shape.input_size, case.shape.l1);
    dense_backward::launch_nnue_l0_sparse_backward(
        &stream,
        &module,
        sparse_l0_layout,
        &device_batch.stm_indices,
        &device_batch.nstm_indices,
        &backward_workspace.stm_l0_gradients,
        &backward_workspace.nstm_l0_gradients,
        &mut backward_workspace.l0w_gradients,
        &mut backward_workspace.l0b_gradients,
    )?;
    stream.synchronize()?;

    let gpu_hidden2_gradients = backward_workspace.hidden2_gradients.to_host_vec(&stream)?;
    let gpu_hidden1_gradients = backward_workspace.hidden1_gradients.to_host_vec(&stream)?;
    let gpu_combined_gradients = backward_workspace.combined_gradients.to_host_vec(&stream)?;
    let gpu_stm_l0_gradients = backward_workspace.stm_l0_gradients.to_host_vec(&stream)?;
    let gpu_nstm_l0_gradients = backward_workspace.nstm_l0_gradients.to_host_vec(&stream)?;
    let gpu_l0w_gradients = backward_workspace.l0w_gradients.to_host_vec(&stream)?;
    let gpu_l0b_gradients = backward_workspace.l0b_gradients.to_host_vec(&stream)?;
    let gpu_outw_gradients = backward_workspace.outw_gradients.to_host_vec(&stream)?;
    let gpu_outb_gradient = backward_workspace.outb_gradients.to_host_vec(&stream)?;
    let gpu_l2w_gradients = backward_workspace.l2w_gradients.to_host_vec(&stream)?;
    let gpu_l2b_gradients = backward_workspace.l2b_gradients.to_host_vec(&stream)?;
    let gpu_l1w_gradients = backward_workspace.l1w_gradients.to_host_vec(&stream)?;
    let gpu_l1b_gradients = backward_workspace.l1b_gradients.to_host_vec(&stream)?;

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
fn run_nnue_ranger_step_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{
        backward::{
            DenseCReluBackwardLayout, DenseOutputBackwardLayout, NnueBackwardWorkspace, NnueBackwardWorkspaceLayout,
            NnueL0CReluBackwardLayout, NnueL0SparseBackwardLayout,
        },
        nnue::{
            NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardHostBatch, NnueForwardHostWeights,
            NnueForwardWorkspace, NnueForwardWorkspaceLayout,
        },
        optimizer::NnueRangerOptimizerStates,
        DeviceBuffer,
    };

    let case = match args.nnue_forward_fixture.as_deref() {
        Some(path) => NnueForwardCase::read_fixture(path)?,
        None => NnueForwardCase::new(args.nnue_case),
    };
    let cpu_forward_trace = case.cpu_forward_trace();
    let cpu_trace = case.cpu_dense_backward_trace(&cpu_forward_trace);
    let params = grouped_ranger_step_params();
    let ptx = match args.ptx.as_ref() {
        Some(ptx) => ptx.clone(),
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
    let mut device_weights = NnueForwardDeviceWeights::from_host(&stream, &host_weights)?;
    let mut optimizer_states = NnueRangerOptimizerStates::from_host_weights(&stream, &host_weights)?;
    let forward_layout = NnueForwardWorkspaceLayout::new(case.shape, case.batch_size);
    let mut forward_workspace = NnueForwardWorkspace::new(&stream, forward_layout)?;

    nnue_forward::launch_nnue_forward(&stream, &module, &device_batch, &device_weights, &mut forward_workspace)?;

    let output_gradients = DeviceBuffer::from_host(&stream, &cpu_trace.output_gradients)?;
    let backward_layout = NnueBackwardWorkspaceLayout::new(case.shape, case.batch_size, case.max_active);
    let mut backward_workspace = NnueBackwardWorkspace::new(&stream, backward_layout)?;

    let output_layout = DenseOutputBackwardLayout::new(case.batch_size, case.shape.l3);
    dense_backward::launch_dense_output_backward(
        &stream,
        &module,
        output_layout,
        &forward_workspace.hidden2,
        &output_gradients,
        &device_weights.outw,
        &mut backward_workspace.hidden2_gradients,
        &mut backward_workspace.outw_gradients,
        &mut backward_workspace.outb_gradients,
    )?;

    let l2_layout = DenseCReluBackwardLayout::new(case.batch_size, case.shape.l2, case.shape.l3);
    dense_backward::launch_dense_crelu_backward(
        &stream,
        &module,
        l2_layout,
        &forward_workspace.hidden1,
        &forward_workspace.hidden2,
        &backward_workspace.hidden2_gradients,
        &device_weights.l2w,
        &mut backward_workspace.hidden1_gradients,
        &mut backward_workspace.l2w_gradients,
        &mut backward_workspace.l2b_gradients,
    )?;

    let l1_layout = DenseCReluBackwardLayout::new(case.batch_size, case.shape.l1 * 2, case.shape.l2);
    dense_backward::launch_dense_crelu_backward(
        &stream,
        &module,
        l1_layout,
        &forward_workspace.combined,
        &forward_workspace.hidden1,
        &backward_workspace.hidden1_gradients,
        &device_weights.l1w,
        &mut backward_workspace.combined_gradients,
        &mut backward_workspace.l1w_gradients,
        &mut backward_workspace.l1b_gradients,
    )?;

    let l0_layout = NnueL0CReluBackwardLayout::new(case.batch_size, case.shape.l1);
    dense_backward::launch_nnue_l0_crelu_backward(
        &stream,
        &module,
        l0_layout,
        &backward_workspace.combined_gradients,
        &forward_workspace.stm_l0,
        &forward_workspace.nstm_l0,
        &mut backward_workspace.stm_l0_gradients,
        &mut backward_workspace.nstm_l0_gradients,
    )?;

    let sparse_l0_layout =
        NnueL0SparseBackwardLayout::new(case.batch_size, case.max_active, case.shape.input_size, case.shape.l1);
    dense_backward::launch_nnue_l0_sparse_backward(
        &stream,
        &module,
        sparse_l0_layout,
        &device_batch.stm_indices,
        &device_batch.nstm_indices,
        &backward_workspace.stm_l0_gradients,
        &backward_workspace.nstm_l0_gradients,
        &mut backward_workspace.l0w_gradients,
        &mut backward_workspace.l0b_gradients,
    )?;

    optimizer_update::launch_nnue_ranger_update(
        &stream,
        &module,
        params,
        &mut device_weights,
        &backward_workspace,
        &mut optimizer_states,
    )?;
    stream.synchronize()?;

    let mut comparisons = Vec::new();
    macro_rules! compare_group {
        ($field:ident, $weights:expr, $gradients:expr) => {{
            let expected = cpu_single_ranger_update_trace($weights, $gradients, params)?;
            let gpu_weights = device_weights.$field.to_host_vec(&stream)?;
            let gpu_momentum = optimizer_states.$field.momentum.to_host_vec(&stream)?;
            let gpu_velocity = optimizer_states.$field.velocity.to_host_vec(&stream)?;
            let gpu_slow_params = optimizer_states.$field.slow_params.to_host_vec(&stream)?;
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_weights"),
                &expected.weights,
                &gpu_weights,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_momentum"),
                &expected.momentum,
                &gpu_momentum,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_velocity"),
                &expected.velocity,
                &gpu_velocity,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_slow"),
                &expected.slow_params,
                &gpu_slow_params,
                args.tolerance,
            )?);
        }};
    }

    compare_group!(l0w, &case.l0w, &cpu_trace.l0w_gradients);
    compare_group!(l0b, &case.l0b, &cpu_trace.l0b_gradients);
    compare_group!(l1w, &case.l1w, &cpu_trace.l1w_gradients);
    compare_group!(l1b, &case.l1b, &cpu_trace.l1b_gradients);
    compare_group!(l2w, &case.l2w, &cpu_trace.l2w_gradients);
    compare_group!(l2b, &case.l2b, &cpu_trace.l2b_gradients);
    compare_group!(outw, &case.outw, &cpu_trace.outw_gradients);
    compare_group!(outb, &case.outb, &[cpu_trace.outb_gradient]);

    println!("bulletou-cuda-train NNUE Ranger step smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!("  batch        : {} samples", case.batch_size);
    println!(
        "  shape        : input={} l1={} l2={} l3={}",
        case.shape.input_size, case.shape.l1, case.shape.l2, case.shape.l3
    );
    println!("  tolerance    : {}", args.tolerance);
    println!("  params       : step={} k={} alpha={}", params.radam.step, params.k, params.lookahead.alpha);
    for cmp in comparisons {
        println!(
            "  {:<14}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
    }
    println!("  compare      : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
fn load_nnue_train_fixture_sequence(
    args: &Args,
    mode_name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<(NnueTrainCase, Vec<NnueTrainBatchCase>)> {
    if args.nnue_train_fixture_args.is_empty() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "{mode_name} requires at least one --nnue-train-fixture <PATH>"
        )));
    }

    let first_train_case = match &args.nnue_train_fixture_args[0] {
        NnueTrainFixtureArg::Full(path) => NnueTrainCase::read_fixture(path)?,
        NnueTrainFixtureArg::Batch(path) => {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "first NNUE train input must be --nnue-train-fixture with weights, got batch-only fixture {}",
                path.display()
            )));
        }
    };
    let mut train_batches = Vec::with_capacity(args.nnue_train_fixture_args.len());
    train_batches.push(NnueTrainBatchCase::from_train_case(&first_train_case));
    for input in args.nnue_train_fixture_args.iter().skip(1) {
        match input {
            NnueTrainFixtureArg::Full(path) => {
                train_batches.push(NnueTrainBatchCase::from_train_case(&NnueTrainCase::read_fixture(path)?));
            }
            NnueTrainFixtureArg::Batch(path) => {
                train_batches.push(NnueTrainBatchCase::read_fixture(path)?);
            }
        }
    }
    ensure_compatible_nnue_train_batches(first_train_case.forward.shape, &train_batches)?;

    Ok((first_train_case, train_batches))
}

#[cfg(feature = "cuda")]
fn load_nnue_train_batches_from_args(
    args: &Args,
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
    mode_name: &'static str,
) -> bulletou_cuda_oxide_runtime::Result<Vec<NnueTrainBatchCase>> {
    if args.nnue_train_fixture_args.is_empty() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "{mode_name} with --nnue-train-state-fixture requires at least one --nnue-train-fixture or --nnue-train-batch-fixture"
        )));
    }

    let mut train_batches = Vec::with_capacity(args.nnue_train_fixture_args.len());
    for input in &args.nnue_train_fixture_args {
        match input {
            NnueTrainFixtureArg::Full(path) => {
                train_batches.push(NnueTrainBatchCase::from_train_case(&NnueTrainCase::read_fixture(path)?));
            }
            NnueTrainFixtureArg::Batch(path) => {
                train_batches.push(NnueTrainBatchCase::read_fixture(path)?);
            }
        }
    }
    ensure_compatible_nnue_train_batches(shape, &train_batches)?;
    Ok(train_batches)
}

#[cfg(feature = "cuda")]
fn run_nnue_fixture_train(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use nnue_train_step::{NnueLossRangerStepRunner, NnueTrainLossKind, NnueTrainStepHostBatch};

    let ptx = match args.ptx.as_ref() {
        Some(ptx) => ptx.clone(),
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let (shape, completed_step_offset, train_batches, mut runner) = if let Some(path) = &args.nnue_train_state_fixture {
        let state_case = NnueTrainStateCase::read_fixture(path)?;
        let train_batches = load_nnue_train_batches_from_args(&args, state_case.shape, "--nnue-fixture-train")?;
        let first_batch = train_batches.first().expect("validated non-empty train batch sequence");
        let runner = NnueLossRangerStepRunner::with_optimizer_state(
            &stream,
            &state_case.host_weights(),
            state_case.host_optimizer_states(),
            first_batch.batch_size,
            first_batch.max_active,
        )?;
        (state_case.shape, state_case.completed_steps, train_batches, runner)
    } else {
        let (first_train_case, train_batches) = load_nnue_train_fixture_sequence(&args, "--nnue-fixture-train")?;
        let first_case = &first_train_case.forward;
        let host_weights = bulletou_cuda_oxide_runtime::nnue::NnueForwardHostWeights {
            shape: first_case.shape,
            l0w: &first_case.l0w,
            l0b: &first_case.l0b,
            l1w: &first_case.l1w,
            l1b: &first_case.l1b,
            l2w: &first_case.l2w,
            l2b: &first_case.l2b,
            outw: &first_case.outw,
            outb: &first_case.outb,
        };
        let runner =
            NnueLossRangerStepRunner::new(&stream, &host_weights, first_case.batch_size, first_case.max_active)?;
        (first_case.shape, 0, train_batches, runner)
    };
    let train_loss_kind = match args.loss_kind {
        LossKind::SigmoidMse => NnueTrainLossKind::SigmoidMse,
        LossKind::NnuePytorchWrm => NnueTrainLossKind::NnuePytorchWrm,
    };

    let mut losses = Vec::with_capacity(train_batches.len());
    for (step_idx, train_case) in train_batches.iter().enumerate() {
        let step = completed_step_offset + step_idx + 1;
        let params = grouped_ranger_step_params_for_step(step);
        let host_batch = NnueTrainStepHostBatch {
            stm_indices: &train_case.stm,
            nstm_indices: &train_case.nstm,
            targets: &train_case.targets,
            entry_weights: &train_case.entry_weights,
            batch_size: train_case.batch_size,
            max_active: train_case.max_active,
        };
        runner.step(&stream, &module, params, train_loss_kind, host_batch)?;
        stream.synchronize()?;

        let loss = runner.read_loss(&stream, args.debug_readback)?;
        losses.push((step, loss.weighted_sum[0], loss.mean[0]));
    }

    if let Some(path) = &args.write_nnue_trained_forward_fixture {
        let weights = runner.read_weights(&stream)?;
        let last_batch = train_batches.last().expect("validated non-empty train batch sequence");
        let trained_forward = NnueForwardCase {
            label: "trained-forward-fixture",
            shape,
            batch_size: last_batch.batch_size,
            max_active: last_batch.max_active,
            stm: last_batch.stm.clone(),
            nstm: last_batch.nstm.clone(),
            l0w: weights.l0w,
            l0b: weights.l0b,
            l1w: weights.l1w,
            l1b: weights.l1b,
            l2w: weights.l2w,
            l2b: weights.l2b,
            outw: weights.outw,
            outb: weights.outb,
        };
        trained_forward.write_fixture(path)?;
    }
    if let Some(path) = &args.write_nnue_train_state_fixture {
        let state = runner.read_state(&stream)?;
        write_nnue_train_state_fixture(path, shape, completed_step_offset + train_batches.len(), &state)?;
    }

    println!("bulletou-cuda-train NNUE fixture train");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  fixtures     : {}", args.nnue_train_fixture_args.len());
    for (idx, fixture) in args.nnue_train_fixture_args.iter().enumerate() {
        match fixture {
            NnueTrainFixtureArg::Full(path) => println!("    [{}] full  {}", idx + 1, path.display()),
            NnueTrainFixtureArg::Batch(path) => println!("    [{}] batch {}", idx + 1, path.display()),
        }
    }
    println!("  loss_kind    : {}", loss_kind_label(args.loss_kind));
    println!("  batch        : {} samples", train_batches[0].batch_size);
    println!(
        "  shape        : input={} l1={} l2={} l3={}",
        shape.input_size, shape.l1, shape.l2, shape.l3
    );
    println!("  start_step   : {}", completed_step_offset + 1);
    println!("  steps        : {}", losses.len());
    for (step, weighted_sum, mean) in losses {
        println!("  step{step}_loss  : weighted_sum={weighted_sum} mean={mean}");
    }
    if let Some(path) = &args.write_nnue_trained_forward_fixture {
        println!("  wrote        : {}", path.display());
    }
    if let Some(path) = &args.write_nnue_train_state_fixture {
        println!("  wrote state  : {}", path.display());
    }
    println!("  train        : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
#[cfg(feature = "root-loader")]
fn run_nnue_teacher_train(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use nnue_train_step::{NnueLossRangerStepRunner, NnueTrainLossKind, NnueTrainStepHostBatch};

    let teacher = args.teacher.as_deref().ok_or_else(|| {
        bulletou_cuda_oxide_runtime::Error::Smoke("--nnue-teacher-train requires --teacher <PATH>".to_string())
    })?;
    if args.train_steps == 0 {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--nnue-teacher-train requires --train-steps > 0".to_string(),
        ));
    }
    if args.batch_size == 0 {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--nnue-teacher-train requires --batch-size > 0".to_string(),
        ));
    }
    if args.batches_per_superbatch == 0 {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--nnue-teacher-train requires --batches-per-superbatch > 0".to_string(),
        ));
    }
    if args.test_teacher.is_some() && args.test_batch_size == 0 {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--test-batch-size must be > 0".to_string(),
        ));
    }
    validate_nnue_teacher_schedule_args(&args)?;
    let save_interval_steps = nnue_teacher_save_interval_steps(&args)?;

    let ptx = match args.ptx.as_ref() {
        Some(ptx) => ptx.clone(),
        None => default_nnue_ptx()?,
    };

    if args.nnue_train_state_fixture.is_some() && args.nnue_train_state_bin.is_some() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--nnue-train-state-fixture and --nnue-train-state-bin cannot be used together".to_string(),
        ));
    }

    let output_resume_checkpoint = if args.nnue_train_state_fixture.is_none() && args.nnue_train_state_bin.is_none() {
        match args.output.as_deref() {
            Some(output_dir) => find_latest_bridge_checkpoint(output_dir)?,
            None => None,
        }
    } else {
        None
    };
    let train_state_source = if let Some(path) = args.nnue_train_state_fixture.as_deref() {
        Some(NnueTrainStateSource::BoungFixture(path))
    } else if let Some(path) = args.nnue_train_state_bin.as_deref() {
        Some(NnueTrainStateSource::RootStateBin(path))
    } else {
        output_resume_checkpoint.as_ref().map(|checkpoint| checkpoint.state_source())
    };
    let resume_checkpoint_teacher = output_resume_checkpoint.as_ref().and_then(|checkpoint| checkpoint.teacher_spec.as_deref());
    let resume_teacher_matches = resume_checkpoint_teacher.is_some_and(|stored| stored == teacher);
    let legacy_resume_checkpoint = output_resume_checkpoint.is_some() && resume_checkpoint_teacher.is_none();
    let dataloader_resume_pos = if resume_teacher_matches || legacy_resume_checkpoint {
        match output_resume_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.dataloader_pos_path.as_deref())
        {
            Some(path) => parse_bridge_dataloader_pos(path)?,
            None => None,
        }
    } else {
        None
    };

    if args.weights_bin.is_some() && train_state_source.is_some() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--weights-bin cannot be used together with a restored train state; the train-state fixture already contains weights".to_string(),
        ));
    }
    let initial_case = match args.weights_bin.as_deref() {
        Some(path) => load_root_halfkp_weights_as_nnue_case(path)?,
        None => NnueForwardCase::new(NnueForwardCaseKind::Halfkp),
    };
    let restored_state = match train_state_source {
        Some(NnueTrainStateSource::BoungFixture(path)) => Some(NnueTrainStateCase::read_fixture(path)?),
        Some(NnueTrainStateSource::RootStateBin(path)) => Some(load_root_halfkp_train_state_case(path)?),
        None => None,
    };
    let shape = restored_state.as_ref().map(|state| state.shape).unwrap_or(initial_case.shape);
    if shape != initial_case.shape {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "--nnue-teacher-train currently supports only HalfKP shape input={} l1={} l2={} l3={}, but state has input={} l1={} l2={} l3={}",
            initial_case.shape.input_size,
            initial_case.shape.l1,
            initial_case.shape.l2,
            initial_case.shape.l3,
            shape.input_size,
            shape.l1,
            shape.l2,
            shape.l3
        )));
    }
    let completed_step_offset = restored_state.as_ref().map(|state| state.completed_steps).unwrap_or(0);

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let train_loss_kind = match args.loss_kind {
        LossKind::SigmoidMse => NnueTrainLossKind::SigmoidMse,
        LossKind::NnuePytorchWrm => NnueTrainLossKind::NnuePytorchWrm,
    };
    let test_cache = load_nnue_bridge_test_cache(&args)?;
    let log_context = NnueBridgeLogContext {
        batch_size: args.batch_size,
        batches_per_superbatch: args.batches_per_superbatch,
        superbatches_per_epoch: args.superbatches_per_epoch,
        lambda: 1.0,
    };

    let mut runner = None;
    let mut losses = Vec::with_capacity(args.train_steps);
    let mut learning_rates = Vec::with_capacity(args.train_steps);
    let mut sources = Vec::with_capacity(args.train_steps);
    let mut dataloader_positions = Vec::with_capacity(args.train_steps);
    let mut checkpoint_losses = Vec::new();
    let mut checkpoint_learning_rates = Vec::new();
    let mut checkpoint_sources = Vec::new();
    let mut checkpoint_dataloader_positions = Vec::new();
    let mut bridge_checkpoints = Vec::new();
    let mut last_batch = None;
    let mut run_steps = 0usize;
    let teacher_batch_index = if dataloader_resume_pos.is_some()
        || resume_teacher_matches
        || legacy_resume_checkpoint
        || output_resume_checkpoint.is_none()
    {
        completed_step_offset
    } else {
        0
    };
    let teacher_batch_config = bulletou_lib::value::HalfkpTeacherBatchConfig {
        teacher,
        batch_size: args.batch_size,
        batch_index: teacher_batch_index,
        dataloader_resume_pos,
        buffer_mb: args.buffer_mb,
        loader_threads: args.loader_threads,
        threads: args.threads,
        lambda: 1.0,
        scale: 400.0,
        nnue_pytorch_wrm_loss: matches!(args.loss_kind, LossKind::NnuePytorchWrm),
        score_drop_abs: (args.score_drop_abs > 0).then_some(args.score_drop_abs),
    };
    bulletou_lib::value::for_each_halfkp_teacher_fast_batch(
        &teacher_batch_config,
        args.train_steps,
        |loaded| -> bulletou_cuda_oxide_runtime::Result<()> {
            let source = loaded.source;
            let dataloader_pos = loaded.dataloader_pos;
            let train_batch = nnue_train_batch_from_root_fast_batch(shape.input_size, loaded.batch)?;
            if runner.is_none() {
                runner = Some(match restored_state.as_ref() {
                    Some(state) => NnueLossRangerStepRunner::with_optimizer_state(
                        &stream,
                        &state.host_weights(),
                        state.host_optimizer_states(),
                        train_batch.batch_size,
                        train_batch.max_active,
                    )?,
                    None => {
                        let host_weights = bulletou_cuda_oxide_runtime::nnue::NnueForwardHostWeights {
                            shape: initial_case.shape,
                            l0w: &initial_case.l0w,
                            l0b: &initial_case.l0b,
                            l1w: &initial_case.l1w,
                            l1b: &initial_case.l1b,
                            l2w: &initial_case.l2w,
                            l2b: &initial_case.l2b,
                            outw: &initial_case.outw,
                            outb: &initial_case.outb,
                        };
                        NnueLossRangerStepRunner::new(
                            &stream,
                            &host_weights,
                            train_batch.batch_size,
                            train_batch.max_active,
                        )?
                    }
                });
            }

            let runner_ref = runner.as_mut().expect("runner is initialized");
            let step = completed_step_offset + run_steps + 1;
            let learning_rate = nnue_teacher_learning_rate_for_step(&args, step);
            let params = grouped_ranger_step_params_for_step_with_hyperparams(
                step,
                learning_rate,
                args.optimizer_weight_decay,
                args.optimizer_beta1,
                args.optimizer_beta2,
                args.optimizer_epsilon,
            );
            let host_batch = NnueTrainStepHostBatch {
                stm_indices: &train_batch.stm,
                nstm_indices: &train_batch.nstm,
                targets: &train_batch.targets,
                entry_weights: &train_batch.entry_weights,
                batch_size: train_batch.batch_size,
                max_active: train_batch.max_active,
            };
            runner_ref.step(&stream, &module, params, train_loss_kind, host_batch)?;
            stream.synchronize()?;

            let loss = runner_ref.read_loss(&stream, args.debug_readback)?;
            let loss_entry = (step, loss.weighted_sum[0], loss.mean[0]);
            losses.push(loss_entry);
            learning_rates.push(learning_rate);
            sources.push(source.clone());
            dataloader_positions.push(dataloader_pos);
            checkpoint_losses.push(loss_entry);
            checkpoint_learning_rates.push(learning_rate);
            checkpoint_sources.push(source);
            checkpoint_dataloader_positions.push(dataloader_pos);
            run_steps += 1;
            if args.output.is_some()
                && save_interval_steps.map(|interval| run_steps % interval == 0).unwrap_or(false)
            {
                let weights = runner_ref.read_weights(&stream)?;
                let state = runner_ref.read_state(&stream)?;
                let test_metrics = match &test_cache {
                    Some(cache) => Some(run_nnue_bridge_test_pass(shape, &weights, cache, &args)?),
                    None => None,
                };
                bridge_checkpoints.push(write_nnue_bridge_checkpoint(
                    args.output.as_ref().expect("checked output exists"),
                    shape,
                    completed_step_offset + run_steps,
                    weights,
                    &state,
                    &train_batch,
                    teacher,
                    &checkpoint_losses,
                    &checkpoint_learning_rates,
                    &checkpoint_sources,
                    &checkpoint_dataloader_positions,
                    test_metrics,
                    log_context,
                )?);
                checkpoint_losses.clear();
                checkpoint_learning_rates.clear();
                checkpoint_sources.clear();
                checkpoint_dataloader_positions.clear();
            }
            last_batch = Some(train_batch);
            Ok(())
        },
    )
    .map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to stream NNUE teacher batches from {teacher}: {err}"
        ))
    })?;

    let runner = runner.as_ref().expect("validated non-empty teacher train steps");
    let last_batch = last_batch.as_ref().expect("validated non-empty teacher train steps");
    if let Some(path) = &args.write_nnue_trained_forward_fixture {
        let weights = runner.read_weights(&stream)?;
        let trained_forward = NnueForwardCase {
            label: "teacher-trained-forward-fixture",
            shape,
            batch_size: last_batch.batch_size,
            max_active: last_batch.max_active,
            stm: last_batch.stm.clone(),
            nstm: last_batch.nstm.clone(),
            l0w: weights.l0w,
            l0b: weights.l0b,
            l1w: weights.l1w,
            l1b: weights.l1b,
            l2w: weights.l2w,
            l2b: weights.l2b,
            outw: weights.outw,
            outb: weights.outb,
        };
        trained_forward.write_fixture(path)?;
    }
    if let Some(path) = &args.write_nnue_train_state_fixture {
        let state = runner.read_state(&stream)?;
        write_nnue_train_state_fixture(path, shape, completed_step_offset + run_steps, &state)?;
    }
    if let Some(output_dir) = &args.output {
        let should_write_final_bridge_checkpoint =
            save_interval_steps.is_none() || !checkpoint_losses.is_empty() || bridge_checkpoints.is_empty();
        if should_write_final_bridge_checkpoint {
            let checkpoint_loss_entries = if save_interval_steps.is_none() {
                losses.as_slice()
            } else {
                checkpoint_losses.as_slice()
            };
            let checkpoint_lr_entries = if save_interval_steps.is_none() {
                learning_rates.as_slice()
            } else {
                checkpoint_learning_rates.as_slice()
            };
            let checkpoint_source_entries = if save_interval_steps.is_none() {
                sources.as_slice()
            } else {
                checkpoint_sources.as_slice()
            };
            let checkpoint_dataloader_pos_entries = if save_interval_steps.is_none() {
                dataloader_positions.as_slice()
            } else {
                checkpoint_dataloader_positions.as_slice()
            };

            let weights = runner.read_weights(&stream)?;
            let state = runner.read_state(&stream)?;
            let test_metrics = match &test_cache {
                Some(cache) => Some(run_nnue_bridge_test_pass(shape, &weights, cache, &args)?),
                None => None,
            };
            bridge_checkpoints.push(write_nnue_bridge_checkpoint(
                output_dir,
                shape,
                completed_step_offset + run_steps,
                weights,
                &state,
                last_batch,
                teacher,
                checkpoint_loss_entries,
                checkpoint_lr_entries,
                checkpoint_source_entries,
                checkpoint_dataloader_pos_entries,
                test_metrics,
                log_context,
            )?);
        }
    }

    println!("bulletou-cuda-train NNUE teacher train");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  teacher      : {teacher}");
    if let Some(source) = train_state_source {
        println!("  resume_state : {} ({})", source.path().display(), source.label());
    }
    if let Some(pos) = dataloader_resume_pos {
        println!("  resume_data  : byte_offset={}, plies={}", pos.byte_offset, pos.plies);
    } else if resume_checkpoint_teacher.is_some_and(|stored_teacher| stored_teacher != teacher) {
        let stored_teacher = resume_checkpoint_teacher.expect("checked resume checkpoint teacher");
        println!("  resume_data  : teacher changed; starting teacher stream at batch 0");
        println!("    previous   : {stored_teacher}");
    }
    println!("  batches      : {}", sources.len());
    for (idx, source) in sources.iter().enumerate() {
        println!("    [{}] {}", idx + 1, source);
    }
    println!("  loss_kind    : {}", loss_kind_label(args.loss_kind));
    println!("  batch        : {} samples", last_batch.batch_size);
    println!(
        "  shape        : input={} l1={} l2={} l3={}",
        shape.input_size, shape.l1, shape.l2, shape.l3
    );
    println!("  start_step   : {}", completed_step_offset + 1);
    println!("  steps        : {}", losses.len());
    println!("  batches/sb   : {}", args.batches_per_superbatch);
    println!("  lr_schedule  : {}", train_lr_schedule_label(args.lr_schedule));
    if let Some(first_lr) = learning_rates.first() {
        println!("  lr_start     : {first_lr}");
    }
    if let Some(last_lr) = learning_rates.last() {
        println!("  lr_last      : {last_lr}");
    }
    if args.output.is_some() {
        if let Some(interval) = save_interval_steps {
            if args.batches_per_superbatch == 1 {
                println!("  save_rate    : {} batches", args.save_rate);
            } else {
                println!("  save_rate    : {} superbatches ({interval} batches)", args.save_rate);
            }
        } else {
            println!("  save_rate    : final");
        }
    }
    for (step, weighted_sum, mean) in &losses {
        println!("  step{step}_loss  : weighted_sum={weighted_sum} mean={mean}");
    }
    if let Some(path) = &args.write_nnue_trained_forward_fixture {
        println!("  wrote        : {}", path.display());
    }
    if let Some(path) = &args.write_nnue_train_state_fixture {
        println!("  wrote state  : {}", path.display());
    }
    for checkpoint in &bridge_checkpoints {
        println!("  output       : {}", checkpoint.dir.display());
        println!("  checkpoint   : {:04}", checkpoint.index);
        println!("    forward    : {}", checkpoint.forward_path.display());
        println!("    nn_bin     : {}", checkpoint.nn_bin_path.display());
        println!("    state      : {}", checkpoint.state_path.display());
        println!("    state_bin  : {}", checkpoint.state_bin_path.display());
        println!("    teacher    : {}", checkpoint.teacher_spec_path.display());
        if let Some(path) = &checkpoint.dataloader_pos_path {
            println!("    loader_pos : {}", path.display());
        }
        println!("    learn_log  : {}", checkpoint.learn_log_path.display());
        println!("    summary    : {}", checkpoint.summary_log_path.display());
    }
    println!("  train        : ok");

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
struct NnueBridgeCheckpointWrite {
    index: usize,
    dir: std::path::PathBuf,
    forward_path: std::path::PathBuf,
    nn_bin_path: std::path::PathBuf,
    state_path: std::path::PathBuf,
    state_bin_path: std::path::PathBuf,
    teacher_spec_path: std::path::PathBuf,
    dataloader_pos_path: Option<std::path::PathBuf>,
    learn_log_path: std::path::PathBuf,
    summary_log_path: std::path::PathBuf,
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
const BRIDGE_TEACHER_SPEC_NAME: &str = "teacher.txt";

#[cfg(all(feature = "cuda", feature = "root-loader"))]
const BRIDGE_DATALOADER_POS_NAME: &str = "dataloader_pos.txt";

#[cfg(all(feature = "cuda", feature = "root-loader"))]
#[derive(Debug, Clone, Copy)]
struct NnueBridgeTestMetrics {
    accuracy: f32,
    loss: f32,
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
#[derive(Clone)]
struct NnueBridgeTestCache {
    positions: Vec<bulletou_lib::shogi::PackedSfenValue>,
    teacher_scores: Vec<i16>,
    teacher_results: Vec<i8>,
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
#[derive(Debug, Clone, Copy)]
struct NnueBridgeLogContext {
    batch_size: usize,
    batches_per_superbatch: usize,
    superbatches_per_epoch: usize,
    lambda: f32,
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn load_nnue_bridge_test_cache(args: &Args) -> bulletou_cuda_oxide_runtime::Result<Option<NnueBridgeTestCache>> {
    let Some(path) = args.test_teacher.as_deref() else {
        return Ok(None);
    };
    let positions = bulletou_lib::validate::read_random_hcpe_positions(path, args.test_positions, args.test_seed)
        .map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!("failed to read --test-teacher {path}: {err}"))
        })?;
    let teacher_scores = positions.iter().map(|p| p.score()).collect();
    let teacher_results = positions.iter().map(|p| p.game_result()).collect();
    eprintln!("  cuda-oxide validation: loaded {} test positions from {path}", positions.len());
    Ok(Some(NnueBridgeTestCache { positions, teacher_scores, teacher_results }))
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn run_nnue_bridge_test_pass(
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
    weights: &nnue_train_step::NnueTrainWeightsReadback,
    cache: &NnueBridgeTestCache,
    args: &Args,
) -> bulletou_cuda_oxide_runtime::Result<NnueBridgeTestMetrics> {
    use bulletou_lib::{
        game::inputs::ShogiHalfKP,
        validate::{compute_sign_accuracy_with_loss, ValidationLossKind},
        value::{
            loader::{DefaultDataLoader, DirectSequentialDataLoader},
            FastBatchHost, NoOutputBuckets, NnueForwardWeights,
        },
    };

    let root_shape = bulletou_lib::value::NnueForwardShape {
        input_size: shape.input_size,
        l1: shape.l1,
        l2: shape.l2,
        l3: shape.l3,
    };
    let forward_weights = NnueForwardWeights {
        shape: root_shape,
        l0w: &weights.l0w,
        l0b: &weights.l0b,
        l1w: &weights.l1w,
        l1b: &weights.l1b,
        l2w: &weights.l2w,
        l2b: &weights.l2b,
        outw: &weights.outw,
        outb: &weights.outb,
    };
    forward_weights.validate().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!("validation forward weight shape mismatch: {err}"))
    })?;

    let empty_files: [&str; 0] = [];
    let dataloader = DefaultDataLoader::new(
        ShogiHalfKP,
        NoOutputBuckets,
        (|_, blend| blend) as fn(&bulletou_lib::shogi::PackedSfenValue, f32) -> f32,
        None,
        matches!(args.loss_kind, LossKind::NnuePytorchWrm),
        false,
        400.0,
        (args.score_drop_abs > 0).then_some(args.score_drop_abs),
        DirectSequentialDataLoader::new(&empty_files),
    );

    let mut outputs = Vec::with_capacity(cache.positions.len());
    for chunk in cache.positions.chunks(args.test_batch_size.max(1)) {
        let prepared = dataloader.prepare(chunk, args.threads.max(1), 0.0);
        let batch = FastBatchHost::from(prepared);
        let mut chunk_outputs = forward_weights.forward_batch(&batch).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!("validation forward failed: {err}"))
        })?;
        outputs.append(&mut chunk_outputs);
    }

    let loss_kind = match args.loss_kind {
        LossKind::SigmoidMse => ValidationLossKind::SigmoidMse,
        LossKind::NnuePytorchWrm => ValidationLossKind::NnuePytorchWrm,
    };
    let report = compute_sign_accuracy_with_loss(
        &outputs,
        &cache.teacher_scores,
        &cache.teacher_results,
        (args.score_drop_abs > 0).then_some(args.score_drop_abs),
        1.0,
        400.0,
        loss_kind,
    );
    let accuracy = if report.compared == 0 { f32::NAN } else { report.accuracy() };
    let loss = report.test_loss.unwrap_or(f32::NAN);
    eprintln!(
        "  cuda-oxide validation: accuracy={:.4}% ({}/{} decisive; draws={} excluded; mate={} filtered), loss={:.6} (n={})",
        accuracy * 100.0,
        report.sign_matches,
        report.compared,
        report.drawn_games,
        report.filtered_by_score_cap,
        loss,
        report.loss_sampled,
    );
    Ok(NnueBridgeTestMetrics { accuracy, loss })
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_nnue_bridge_checkpoint(
    output_dir: &std::path::Path,
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
    completed_steps: usize,
    weights: nnue_train_step::NnueTrainWeightsReadback,
    state: &nnue_train_step::NnueTrainStateReadback,
    last_batch: &NnueTrainBatchCase,
    teacher_spec: &str,
    losses: &[(usize, f32, f32)],
    learning_rates: &[f32],
    sources: &[String],
    dataloader_positions: &[Option<bulletou_lib::value::TeacherDataloaderPos>],
    test_metrics: Option<NnueBridgeTestMetrics>,
    log_context: NnueBridgeLogContext,
) -> bulletou_cuda_oxide_runtime::Result<NnueBridgeCheckpointWrite> {
    if losses.len() != sources.len() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "internal bridge checkpoint log mismatch: losses={} sources={}",
            losses.len(),
            sources.len()
        )));
    }
    if losses.len() != learning_rates.len() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "internal bridge checkpoint LR log mismatch: losses={} learning_rates={}",
            losses.len(),
            learning_rates.len()
        )));
    }
    if losses.len() != dataloader_positions.len() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "internal bridge checkpoint dataloader position mismatch: losses={} dataloader_positions={}",
            losses.len(),
            dataloader_positions.len()
        )));
    }

    let (index, dir) = create_next_numbered_checkpoint_dir(output_dir)?;
    let forward_path = dir.join("trained-forward.nnuef");
    let nn_bin_path = dir.join("nn.bin");
    let state_path = dir.join("state.boung");
    let state_bin_path = dir.join("state.bin");
    let teacher_spec_path = dir.join(BRIDGE_TEACHER_SPEC_NAME);
    let dataloader_pos_path = dir.join(BRIDGE_DATALOADER_POS_NAME);
    let learn_log_path = dir.join("learn.log");
    let summary_log_path = output_dir.join("summary-learn.log");

    let trained_forward = NnueForwardCase {
        label: "teacher-trained-bridge-checkpoint",
        shape,
        batch_size: last_batch.batch_size,
        max_active: last_batch.max_active,
        stm: last_batch.stm.clone(),
        nstm: last_batch.nstm.clone(),
        l0w: weights.l0w,
        l0b: weights.l0b,
        l1w: weights.l1w,
        l1b: weights.l1b,
        l2w: weights.l2w,
        l2b: weights.l2b,
        outw: weights.outw,
        outb: weights.outb,
    };
    trained_forward.write_fixture(&forward_path)?;
    write_nnue_halfkp_nn_bin(&nn_bin_path, shape, state)?;
    write_nnue_train_state_fixture(&state_path, shape, completed_steps, state)?;
    write_nnue_root_state_bin(&state_bin_path, completed_steps, state)?;
    write_bridge_teacher_spec(&teacher_spec_path, teacher_spec)?;
    let dataloader_pos_path = write_nnue_bridge_dataloader_pos(&dataloader_pos_path, dataloader_positions)?;
    write_nnue_bridge_learn_log(&learn_log_path, losses, learning_rates, sources, test_metrics, log_context)?;
    append_nnue_bridge_summary_log(
        &summary_log_path,
        index,
        completed_steps,
        losses,
        learning_rates,
        sources,
        test_metrics,
        log_context,
    )?;

    Ok(NnueBridgeCheckpointWrite {
        index,
        dir,
        forward_path,
        nn_bin_path,
        state_path,
        state_bin_path,
        teacher_spec_path,
        dataloader_pos_path,
        learn_log_path,
        summary_log_path,
    })
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_nnue_halfkp_nn_bin(
    path: &std::path::Path,
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
    state: &nnue_train_step::NnueTrainStateReadback,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_lib::value::nnue_save::{
        header_bytes, ft_hash_bytes, l1_bias_scale, network_layer_hash_bytes, pad_weights_for_simd, Activation,
        NnueFeatureSet,
    };
    use std::io::Write as _;

    let feature_set = NnueFeatureSet::HalfKp;
    if shape.input_size != feature_set.input_size() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "cannot write HalfKP nn.bin for input_size={}, expected {}",
            shape.input_size,
            feature_set.input_size()
        )));
    }

    let qa: i16 = 127;
    let qb: i16 = 64;
    let l1_input_dim = shape.l1 * 2;
    let l1_bias = l1_bias_scale(Activation::Crelu, false, qa, qb);

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!("failed to create NNUE nn.bin {}: {err}", path.display()))
    })?);

    writer
        .write_all(&header_bytes(feature_set, shape.l1, shape.l2, shape.l3))
        .and_then(|_| writer.write_all(&ft_hash_bytes(feature_set, shape.l1)))
        .map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to write NNUE nn.bin header {}: {err}",
                path.display()
            ))
        })?;
    write_quantized_i16(&mut writer, path, "l0b", &state.l0b.weights, qa)?;
    write_quantized_i16(&mut writer, path, "l0w", &state.l0w.weights, qa)?;
    writer
        .write_all(&network_layer_hash_bytes(shape.l1, shape.l2, shape.l3))
        .map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to write NNUE nn.bin network hash {}: {err}",
                path.display()
            ))
        })?;

    write_quantized_i32(&mut writer, path, "l1b", &state.l1b.weights, l1_bias)?;
    let l1w = transpose_input_major_dense_weights(&state.l1w.weights, l1_input_dim, shape.l2)?;
    let l1w = pad_weights_for_simd(&l1w, shape.l2, l1_input_dim);
    write_quantized_i8(&mut writer, path, "l1w", &l1w, qb)?;

    write_quantized_i32(&mut writer, path, "l2b", &state.l2b.weights, 127 * i32::from(qb))?;
    let l2w = transpose_input_major_dense_weights(&state.l2w.weights, shape.l2, shape.l3)?;
    let l2w = pad_weights_for_simd(&l2w, shape.l3, shape.l2);
    write_quantized_i8(&mut writer, path, "l2w", &l2w, qb)?;

    write_quantized_i32(&mut writer, path, "outb", &state.outb.weights, 127 * i32::from(qb))?;
    let outw = transpose_input_major_dense_weights(&state.outw.weights, shape.l3, 1)?;
    let outw = pad_weights_for_simd(&outw, 1, shape.l3);
    write_quantized_i8(&mut writer, path, "outw", &outw, qb)?;

    writer.flush().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!("failed to flush NNUE nn.bin {}: {err}", path.display()))
    })?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn transpose_input_major_dense_weights(
    weights: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> bulletou_cuda_oxide_runtime::Result<Vec<f32>> {
    let expected = input_dim.checked_mul(output_dim).ok_or_else(|| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "dense weight shape overflow: input_dim={input_dim} output_dim={output_dim}"
        ))
    })?;
    if weights.len() != expected {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "dense weight length mismatch: got {}, expected input_dim({input_dim}) * output_dim({output_dim}) = {expected}",
            weights.len()
        )));
    }
    let mut transposed = vec![0.0_f32; weights.len()];
    for input in 0..input_dim {
        for output in 0..output_dim {
            transposed[output * input_dim + input] = weights[input * output_dim + output];
        }
    }
    Ok(transposed)
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_quantized_i8(
    writer: &mut impl std::io::Write,
    path: &std::path::Path,
    name: &'static str,
    values: &[f32],
    scale: i16,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    let mut bytes = Vec::with_capacity(values.len());
    for &value in values {
        let qf = (f64::from(scale) * f64::from(value)).round();
        let q = qf as i8;
        if qf != f64::from(q) {
            return Err(quantization_error(path, name, "i8", value, qf));
        }
        bytes.extend_from_slice(&q.to_le_bytes());
    }
    write_nnue_bin_chunk(writer, path, name, &bytes)
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_quantized_i16(
    writer: &mut impl std::io::Write,
    path: &std::path::Path,
    name: &'static str,
    values: &[f32],
    scale: i16,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i16>());
    for &value in values {
        let qf = (f64::from(scale) * f64::from(value)).round();
        let q = qf as i16;
        if qf != f64::from(q) {
            return Err(quantization_error(path, name, "i16", value, qf));
        }
        bytes.extend_from_slice(&q.to_le_bytes());
    }
    write_nnue_bin_chunk(writer, path, name, &bytes)
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_quantized_i32(
    writer: &mut impl std::io::Write,
    path: &std::path::Path,
    name: &'static str,
    values: &[f32],
    scale: i32,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i32>());
    for &value in values {
        let qf = (f64::from(scale) * f64::from(value)).round();
        let q = qf as i32;
        if qf != f64::from(q) {
            return Err(quantization_error(path, name, "i32", value, qf));
        }
        bytes.extend_from_slice(&q.to_le_bytes());
    }
    write_nnue_bin_chunk(writer, path, name, &bytes)
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_nnue_bin_chunk(
    writer: &mut impl std::io::Write,
    path: &std::path::Path,
    name: &'static str,
    bytes: &[u8],
) -> bulletou_cuda_oxide_runtime::Result<()> {
    writer.write_all(bytes).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to write NNUE nn.bin {name} chunk {}: {err}",
            path.display()
        ))
    })
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn quantization_error(
    path: &std::path::Path,
    name: &'static str,
    target: &'static str,
    value: f32,
    quantized: f64,
) -> bulletou_cuda_oxide_runtime::Error {
    bulletou_cuda_oxide_runtime::Error::Smoke(format!(
        "failed to quantize NNUE nn.bin {name} value {value} -> {quantized} as {target} for {}",
        path.display()
    ))
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_nnue_root_state_bin(
    path: &std::path::Path,
    completed_steps: usize,
    state: &nnue_train_step::NnueTrainStateReadback,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    use std::io::Write as _;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to create NNUE root state.bin {}: {err}",
            path.display()
        ))
    })?);

    let marker = bulletou_lib::value::yaneuraou_kppt::write_state_backend_marker(
        bulletou_lib::value::yaneuraou_kppt::STATE_BACKEND_CUDA_OXIDE,
    );
    writer.write_all(&marker).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to write NNUE root state.bin backend marker {}: {err}",
            path.display()
        ))
    })?;

    macro_rules! write_section {
        ($section:literal, $field:ident) => {{
            let mut records: Vec<(String, &[f32])> = vec![
                (format!("nnue/{}/l0w", $section), state.l0w.$field.as_slice()),
                (format!("nnue/{}/l0b", $section), state.l0b.$field.as_slice()),
                (format!("nnue/{}/l1w", $section), state.l1w.$field.as_slice()),
                (format!("nnue/{}/l1b", $section), state.l1b.$field.as_slice()),
                (format!("nnue/{}/l2w", $section), state.l2w.$field.as_slice()),
                (format!("nnue/{}/l2b", $section), state.l2b.$field.as_slice()),
                (format!("nnue/{}/outw", $section), state.outw.$field.as_slice()),
                (format!("nnue/{}/outb", $section), state.outb.$field.as_slice()),
            ];
            records.sort_by(|a, b| a.0.cmp(&b.0));
            let chunk =
                bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(
                    records.iter().map(|(id, values)| (id.as_str(), *values)),
                );
            writer.write_all(&chunk).map_err(|err| {
                bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                    "failed to write NNUE root state.bin {}: {err}",
                    path.display()
                ))
            })?;
        }};
    }

    write_section!("weights", weights);
    write_section!("momentum", momentum);
    write_section!("velocity", velocity);
    write_section!("slow", slow_params);

    let step = completed_steps as f32;
    let mut step_records: Vec<(String, Vec<f32>)> = ["l0w", "l0b", "l1w", "l1b", "l2w", "l2b", "outw", "outb"]
        .iter()
        .map(|id| (format!("nnue/step_ranger/{id}"), vec![step]))
        .collect();
    step_records.sort_by(|a, b| a.0.cmp(&b.0));
    let chunk = bulletou_lib::value::yaneuraou_kppt::write_model_weights_bin(
        step_records.iter().map(|(id, values)| (id.as_str(), values.as_slice())),
    );
    writer.write_all(&chunk).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to write NNUE root state.bin {}: {err}",
            path.display()
        ))
    })?;

    writer.flush().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to flush NNUE root state.bin {}: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn create_next_numbered_checkpoint_dir(
    output_dir: &std::path::Path,
) -> bulletou_cuda_oxide_runtime::Result<(usize, std::path::PathBuf)> {
    std::fs::create_dir_all(output_dir).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to create checkpoint output dir {}: {err}",
            output_dir.display()
        ))
    })?;

    let mut max_index = 0usize;
    for entry in std::fs::read_dir(output_dir).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to read checkpoint output dir {}: {err}",
            output_dir.display()
        ))
    })? {
        let entry = entry.map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to read checkpoint output entry in {}: {err}",
                output_dir.display()
            ))
        })?;
        if let Some(index) = numbered_checkpoint_dir_index(&entry)? {
            max_index = max_index.max(index);
        }
    }

    let index = max_index + 1;
    let dir = output_dir.join(format!("{index:04}"));
    std::fs::create_dir(&dir).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to create checkpoint dir {}: {err}",
            dir.display()
        ))
    })?;
    Ok((index, dir))
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
#[derive(Debug, Clone, Copy)]
enum NnueTrainStateSource<'a> {
    BoungFixture(&'a std::path::Path),
    RootStateBin(&'a std::path::Path),
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
impl<'a> NnueTrainStateSource<'a> {
    fn path(self) -> &'a std::path::Path {
        match self {
            Self::BoungFixture(path) | Self::RootStateBin(path) => path,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::BoungFixture(_) => "state.boung",
            Self::RootStateBin(_) => "state.bin",
        }
    }
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BridgeResumeStateKind {
    BoungFixture,
    RootStateBin,
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
struct BridgeResumeCheckpoint {
    state_path: std::path::PathBuf,
    state_kind: BridgeResumeStateKind,
    teacher_spec: Option<String>,
    dataloader_pos_path: Option<std::path::PathBuf>,
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
impl BridgeResumeCheckpoint {
    fn state_source(&self) -> NnueTrainStateSource<'_> {
        match self.state_kind {
            BridgeResumeStateKind::BoungFixture => NnueTrainStateSource::BoungFixture(&self.state_path),
            BridgeResumeStateKind::RootStateBin => NnueTrainStateSource::RootStateBin(&self.state_path),
        }
    }
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn find_latest_bridge_checkpoint(
    output_dir: &std::path::Path,
) -> bulletou_cuda_oxide_runtime::Result<Option<BridgeResumeCheckpoint>> {
    if !output_dir.exists() {
        return Ok(None);
    }

    let mut latest = None;
    for entry in std::fs::read_dir(output_dir).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to read checkpoint output dir {}: {err}",
            output_dir.display()
        ))
    })? {
        let entry = entry.map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to read checkpoint output entry in {}: {err}",
                output_dir.display()
            ))
        })?;
        let Some(index) = numbered_checkpoint_dir_index(&entry)? else {
            continue;
        };
        let boung_path = entry.path().join("state.boung");
        let state_bin_path = entry.path().join("state.bin");
        let state = if boung_path.is_file() {
            Some((boung_path, BridgeResumeStateKind::BoungFixture))
        } else if state_bin_path.is_file() {
            Some((state_bin_path, BridgeResumeStateKind::RootStateBin))
        } else {
            None
        };
        if let Some((state_path, state_kind)) = state
            && latest.as_ref().map_or(true, |(latest_index, _)| index > *latest_index)
        {
            let teacher_spec_path = entry.path().join(BRIDGE_TEACHER_SPEC_NAME);
            let teacher_spec = if teacher_spec_path.is_file() {
                let text = std::fs::read_to_string(&teacher_spec_path).map_err(|err| {
                    bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                        "failed to read bridge checkpoint teacher spec {}: {err}",
                        teacher_spec_path.display()
                    ))
                })?;
                Some(text.trim_end_matches(|ch| ch == '\r' || ch == '\n').to_string())
            } else {
                None
            };
            let dataloader_pos_path = entry.path().join(BRIDGE_DATALOADER_POS_NAME);
            let dataloader_pos_path = dataloader_pos_path.is_file().then_some(dataloader_pos_path);
            latest = Some((index, BridgeResumeCheckpoint { state_path, state_kind, teacher_spec, dataloader_pos_path }));
        }
    }

    Ok(latest.map(|(_, checkpoint)| checkpoint))
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn numbered_checkpoint_dir_index(entry: &std::fs::DirEntry) -> bulletou_cuda_oxide_runtime::Result<Option<usize>> {
    let file_type = entry.file_type().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to read checkpoint output entry type {}: {err}",
            entry.path().display()
        ))
    })?;
    if !file_type.is_dir() {
        return Ok(None);
    }
    let name = entry.file_name();
    let Some(name) = name.to_str() else {
        return Ok(None);
    };
    if name.is_empty() || !name.chars().all(|ch| ch.is_ascii_digit()) {
        return Ok(None);
    }
    let index = name.parse::<usize>().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to parse checkpoint dir index {}: {err}",
            entry.path().display()
        ))
    })?;
    Ok(Some(index))
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn parse_bridge_dataloader_pos(
    path: &std::path::Path,
) -> bulletou_cuda_oxide_runtime::Result<Option<bulletou_lib::value::TeacherDataloaderPos>> {
    let text = std::fs::read_to_string(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to read bridge checkpoint dataloader_pos.txt {}: {err}",
            path.display()
        ))
    })?;
    let line = text.lines().find(|line| !line.trim().is_empty());
    let Some(line) = line else {
        return Ok(None);
    };
    let Some((offset, plies)) = line.trim().split_once(',') else {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "invalid dataloader_pos.txt {}: expected `<byte_offset>,<plies>`",
            path.display()
        )));
    };
    let offset = offset.trim().parse::<u64>().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "invalid byte offset in dataloader_pos.txt {}: {err}",
            path.display()
        ))
    })?;
    let plies = plies.trim().parse::<usize>().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "invalid plies in dataloader_pos.txt {}: {err}",
            path.display()
        ))
    })?;
    Ok(Some(bulletou_lib::value::TeacherDataloaderPos { byte_offset: offset, plies }))
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
#[derive(Debug, Clone, Copy)]
struct NnueBridgeDisplayStep {
    epoch: usize,
    superbatch: usize,
    curr_batch: usize,
    positions: u64,
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn bridge_log_display_step(step: usize, context: NnueBridgeLogContext) -> NnueBridgeDisplayStep {
    let batches_per_superbatch = context.batches_per_superbatch.max(1);
    let global_superbatch = (step.saturating_sub(1) / batches_per_superbatch) + 1;
    let curr_batch = ((step.saturating_sub(1) % batches_per_superbatch) + 1).min(batches_per_superbatch);
    let (epoch, superbatch) = if context.superbatches_per_epoch == 0 {
        (1, global_superbatch)
    } else {
        (
            (global_superbatch.saturating_sub(1) / context.superbatches_per_epoch) + 1,
            ((global_superbatch.saturating_sub(1) % context.superbatches_per_epoch) + 1),
        )
    };
    let positions = (step as u128).saturating_mul(context.batch_size as u128).min(u64::MAX as u128) as u64;
    NnueBridgeDisplayStep { epoch, superbatch, curr_batch, positions }
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_nnue_bridge_learn_log(
    path: &std::path::Path,
    losses: &[(usize, f32, f32)],
    learning_rates: &[f32],
    sources: &[String],
    test_metrics: Option<NnueBridgeTestMetrics>,
    context: NnueBridgeLogContext,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    use std::io::Write as _;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to create bridge checkpoint learn.log {}: {err}",
            path.display()
        ))
    })?);

    writeln!(
        writer,
        "eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher"
    )
    .map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to write bridge checkpoint learn.log {}: {err}",
            path.display()
        ))
    })?;
    for (idx, (((step, _weighted_sum, mean), learning_rate), source)) in
        losses.iter().zip(learning_rates.iter()).zip(sources.iter()).enumerate()
    {
        let display = bridge_log_display_step(*step, context);
        let is_last = idx + 1 == losses.len();
        let test_accuracy = if is_last {
            test_metrics.map(|m| format!("{:.6}", m.accuracy)).unwrap_or_else(|| "-".to_string())
        } else {
            "-".to_string()
        };
        let test_loss = if is_last {
            test_metrics.map(|m| format!("{:.6}", m.loss)).unwrap_or_else(|| "-".to_string())
        } else {
            "-".to_string()
        };
        writeln!(
            writer,
            "nnue,{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{},{}",
            display.epoch,
            display.superbatch,
            display.curr_batch,
            test_accuracy,
            test_loss,
            mean,
            learning_rate,
            learning_rate,
            context.lambda,
            display.positions,
            csv_escape(source),
        )
        .map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to write bridge checkpoint learn.log {}: {err}",
                path.display()
            ))
        })?;
    }
    writer.flush().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to flush bridge checkpoint learn.log {}: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[cfg(feature = "root-loader")]
fn nnue_train_batch_from_root_fast_batch(
    input_size: usize,
    batch: bulletou_lib::value::FastBatchHost,
) -> bulletou_cuda_oxide_runtime::Result<NnueTrainBatchCase> {
    if batch.layout.output_size != 1 {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "NNUE teacher train requires output_size=1, got {}",
            batch.layout.output_size
        )));
    }
    if batch.layout.hand_count_dim != 0 || batch.hand_count.is_some() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "NNUE teacher train does not accept hand_count side inputs".to_string(),
        ));
    }

    Ok(NnueTrainBatchCase {
        label: "teacher-batch",
        input_size,
        batch_size: batch.layout.batch_size,
        max_active: batch.layout.max_active,
        stm: batch.stm,
        nstm: batch.nstm,
        targets: batch.targets,
        entry_weights: batch.weights,
    })
}

#[cfg(feature = "cuda")]
#[cfg(feature = "root-loader")]
fn load_root_halfkp_weights_as_nnue_case(path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<NnueForwardCase> {
    let bytes = std::fs::read(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!("failed to read root NNUE weights {}: {err}", path.display()))
    })?;
    let records = bulletou_lib::value::yaneuraou_kppt::parse_model_weights_bin(&bytes).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to parse root NNUE weights {}: {err}",
            path.display()
        ))
    })?;
    let weights = if records.contains_key("l0w") {
        records
    } else {
        bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "weights")
    };
    let root_weights =
        bulletou_lib::value::NnueForwardOwnedWeights::from_weight_map(
            bulletou_lib::value::NNUE_HALFKP_256X2_32_32,
            &weights,
        )
        .map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to load HalfKP NNUE weights {}: {err}",
                path.display()
            ))
        })?;

    Ok(NnueForwardCase {
        label: "root-weights",
        shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape {
            input_size: root_weights.shape.input_size,
            l1: root_weights.shape.l1,
            l2: root_weights.shape.l2,
            l3: root_weights.shape.l3,
        },
        batch_size: 0,
        max_active: 0,
        stm: Vec::new(),
        nstm: Vec::new(),
        l0w: root_weights.l0w,
        l0b: root_weights.l0b,
        l1w: root_weights.l1w,
        l1b: root_weights.l1b,
        l2w: root_weights.l2w,
        l2b: root_weights.l2b,
        outw: root_weights.outw,
        outb: root_weights.outb,
    })
}

#[cfg(feature = "cuda")]
#[cfg(feature = "root-loader")]
fn load_root_halfkp_train_state_case(path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<NnueTrainStateCase> {
    let bytes = std::fs::read(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to read root NNUE train state {}: {err}",
            path.display()
        ))
    })?;
    let records = bulletou_lib::value::yaneuraou_kppt::parse_model_weights_bin(&bytes).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to parse root NNUE train state {}: {err}",
            path.display()
        ))
    })?;
    match bulletou_lib::value::yaneuraou_kppt::detect_state_backend(&records).as_deref() {
        Some(bulletou_lib::value::yaneuraou_kppt::STATE_BACKEND_BULLET)
        | Some(bulletou_lib::value::yaneuraou_kppt::STATE_BACKEND_CUDA_OXIDE)
        | None => {}
        Some(backend) => {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "unsupported root NNUE train state backend marker {backend:?} in {}",
                path.display()
            )));
        }
    }

    let root_shape = bulletou_lib::value::NNUE_HALFKP_256X2_32_32;
    let shape = bulletou_cuda_oxide_runtime::nnue::NnueForwardShape {
        input_size: root_shape.input_size,
        l1: root_shape.l1,
        l2: root_shape.l2,
        l3: root_shape.l3,
    };
    let layout = bulletou_cuda_oxide_runtime::nnue::NnueForwardWeightLayout::new(shape);
    let weights = bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "weights");
    let momentum = bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "momentum");
    let velocity = bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "velocity");
    let slow = bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "slow");
    let slow = if slow.is_empty() { &weights } else { &slow };
    let steps = bulletou_lib::value::yaneuraou_kppt::extract_component_section(&records, "nnue", "step_ranger");
    let completed_steps = read_root_state_completed_steps(path, &steps)?;

    fn take_record(
        path: &std::path::Path,
        section: &'static str,
        map: &std::collections::BTreeMap<String, Vec<f32>>,
        id: &'static str,
        expected_len: usize,
    ) -> bulletou_cuda_oxide_runtime::Result<Vec<f32>> {
        let values = map.get(id).ok_or_else(|| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "root NNUE train state {} is missing nnue/{section}/{id}",
                path.display()
            ))
        })?;
        if values.len() != expected_len {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "root NNUE train state {} record nnue/{section}/{id} has len {}, expected {expected_len}",
                path.display(),
                values.len()
            )));
        }
        Ok(values.clone())
    }

    let group = |id: &'static str, expected_len: usize| -> bulletou_cuda_oxide_runtime::Result<NnueTrainStateGroupCase> {
        Ok(NnueTrainStateGroupCase {
            weights: take_record(path, "weights", &weights, id, expected_len)?,
            momentum: take_record(path, "momentum", &momentum, id, expected_len)?,
            velocity: take_record(path, "velocity", &velocity, id, expected_len)?,
            slow_params: take_record(path, "slow", slow, id, expected_len)?,
        })
    };

    Ok(NnueTrainStateCase {
        shape,
        completed_steps,
        l0w: group("l0w", layout.l0w_len())?,
        l0b: group("l0b", layout.l0b_len())?,
        l1w: group("l1w", layout.l1w_len())?,
        l1b: group("l1b", layout.l1b_len())?,
        l2w: group("l2w", layout.l2w_len())?,
        l2b: group("l2b", layout.l2b_len())?,
        outw: group("outw", layout.outw_len())?,
        outb: group("outb", layout.outb_len())?,
    })
}

#[cfg(feature = "cuda")]
#[cfg(feature = "root-loader")]
fn read_root_state_completed_steps(
    path: &std::path::Path,
    steps: &std::collections::BTreeMap<String, Vec<f32>>,
) -> bulletou_cuda_oxide_runtime::Result<usize> {
    if steps.is_empty() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "root NNUE train state {} has no nnue/step_ranger/* records",
            path.display()
        )));
    }

    let mut completed_steps = None;
    for (id, values) in steps {
        if values.len() != 1 {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "root NNUE train state {} record nnue/step_ranger/{id} has len {}, expected 1",
                path.display(),
                values.len()
            )));
        }
        let value = values[0];
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f32 {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "root NNUE train state {} record nnue/step_ranger/{id} has invalid step value {value}",
                path.display()
            )));
        }
        let step = value as usize;
        match completed_steps {
            Some(existing) if existing != step => {
                return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                    "root NNUE train state {} has inconsistent step_ranger values: {existing} and {step}",
                    path.display()
                )));
            }
            Some(_) => {}
            None => completed_steps = Some(step),
        }
    }

    completed_steps.ok_or_else(|| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "root NNUE train state {} has no usable nnue/step_ranger/* records",
            path.display()
        ))
    })
}

#[cfg(feature = "cuda")]
fn run_nnue_loss_ranger_step_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::nnue::NnueForwardHostWeights;
    use nnue_train_step::{NnueLossRangerStepRunner, NnueTrainLossKind, NnueTrainStepHostBatch};

    let (first_train_case, train_batches) =
        load_nnue_train_fixture_sequence(&args, "--nnue-loss-ranger-step-smoke")?;

    let first_case = &first_train_case.forward;
    let mut cpu_case = first_case.clone();
    macro_rules! init_cpu_state {
        ($field:ident, $momentum:ident, $velocity:ident, $slow_params:ident) => {
            let mut $momentum = vec![0.0_f32; cpu_case.$field.len()];
            let mut $velocity = vec![0.0_f32; cpu_case.$field.len()];
            let mut $slow_params = cpu_case.$field.clone();
        };
    }
    init_cpu_state!(l0w, cpu_l0w_momentum, cpu_l0w_velocity, cpu_l0w_slow);
    init_cpu_state!(l0b, cpu_l0b_momentum, cpu_l0b_velocity, cpu_l0b_slow);
    init_cpu_state!(l1w, cpu_l1w_momentum, cpu_l1w_velocity, cpu_l1w_slow);
    init_cpu_state!(l1b, cpu_l1b_momentum, cpu_l1b_velocity, cpu_l1b_slow);
    init_cpu_state!(l2w, cpu_l2w_momentum, cpu_l2w_velocity, cpu_l2w_slow);
    init_cpu_state!(l2b, cpu_l2b_momentum, cpu_l2b_velocity, cpu_l2b_slow);
    init_cpu_state!(outw, cpu_outw_momentum, cpu_outw_velocity, cpu_outw_slow);
    init_cpu_state!(outb, cpu_outb_momentum, cpu_outb_velocity, cpu_outb_slow);

    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let host_weights = NnueForwardHostWeights {
        shape: first_case.shape,
        l0w: &first_case.l0w,
        l0b: &first_case.l0b,
        l1w: &first_case.l1w,
        l1b: &first_case.l1b,
        l2w: &first_case.l2w,
        l2b: &first_case.l2b,
        outw: &first_case.outw,
        outb: &first_case.outb,
    };
    let mut runner =
        NnueLossRangerStepRunner::new(&stream, &host_weights, first_case.batch_size, first_case.max_active)?;
    let train_loss_kind = match args.loss_kind {
        LossKind::SigmoidMse => NnueTrainLossKind::SigmoidMse,
        LossKind::NnuePytorchWrm => NnueTrainLossKind::NnuePytorchWrm,
    };
    let mut comparisons = Vec::new();

    for (step_idx, train_case) in train_batches.iter().enumerate() {
        let step = step_idx + 1;
        cpu_case.batch_size = train_case.batch_size;
        cpu_case.max_active = train_case.max_active;
        cpu_case.stm.clone_from(&train_case.stm);
        cpu_case.nstm.clone_from(&train_case.nstm);

        let cpu_forward_trace = cpu_case.cpu_forward_trace();
        let cpu_loss_case = LossSmokeCase {
            label: train_case.label,
            outputs: cpu_forward_trace.outputs.clone(),
            targets: train_case.targets.clone(),
            entry_weights: train_case.entry_weights.clone(),
        };
        let cpu_loss_trace = cpu_loss_case.cpu_loss_trace(args.loss_kind);
        let cpu_trace = cpu_case.cpu_dense_backward_trace_with_output_gradients(
            &cpu_forward_trace,
            cpu_loss_trace.mean_output_gradients.clone(),
        );
        let params = grouped_ranger_step_params_for_step(step);

        let host_batch = NnueTrainStepHostBatch {
            stm_indices: &train_case.stm,
            nstm_indices: &train_case.nstm,
            targets: &train_case.targets,
            entry_weights: &train_case.entry_weights,
            batch_size: train_case.batch_size,
            max_active: train_case.max_active,
        };
        runner.step(&stream, &module, params, train_loss_kind, host_batch)?;
        stream.synchronize()?;

        let gpu_loss = runner.read_loss(&stream, args.debug_readback)?;
        comparisons.push(compare_slices(
            format!("step{step}_weighted_sum"),
            &[cpu_loss_trace.weighted_sum],
            &gpu_loss.weighted_sum,
            args.tolerance,
        )?);
        comparisons.push(compare_slices(
            format!("step{step}_loss_mean"),
            &[cpu_loss_trace.mean],
            &gpu_loss.mean,
            args.tolerance,
        )?);
        if args.debug_readback {
            let gpu_per_sample = gpu_loss.per_sample.as_ref().expect("debug loss readback requested");
            let gpu_output_gradients = gpu_loss
                .mean_output_gradients
                .as_ref()
                .expect("debug loss gradient readback requested");
            comparisons.push(compare_slices(
                format!("step{step}_per_sample"),
                &cpu_loss_trace.per_sample,
                gpu_per_sample,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                format!("step{step}_loss_grad"),
                &cpu_loss_trace.mean_output_gradients,
                gpu_output_gradients,
                args.tolerance,
            )?);
        }

        macro_rules! update_cpu_group {
            ($field:ident, $momentum:ident, $velocity:ident, $slow_params:ident, $gradients:expr) => {{
                let expected = cpu_single_ranger_update_trace_from_state(
                    &cpu_case.$field,
                    &$momentum,
                    &$velocity,
                    &$slow_params,
                    $gradients,
                    params,
                )?;
                cpu_case.$field = expected.weights;
                $momentum = expected.momentum;
                $velocity = expected.velocity;
                $slow_params = expected.slow_params;
            }};
        }

        update_cpu_group!(l0w, cpu_l0w_momentum, cpu_l0w_velocity, cpu_l0w_slow, &cpu_trace.l0w_gradients);
        update_cpu_group!(l0b, cpu_l0b_momentum, cpu_l0b_velocity, cpu_l0b_slow, &cpu_trace.l0b_gradients);
        update_cpu_group!(l1w, cpu_l1w_momentum, cpu_l1w_velocity, cpu_l1w_slow, &cpu_trace.l1w_gradients);
        update_cpu_group!(l1b, cpu_l1b_momentum, cpu_l1b_velocity, cpu_l1b_slow, &cpu_trace.l1b_gradients);
        update_cpu_group!(l2w, cpu_l2w_momentum, cpu_l2w_velocity, cpu_l2w_slow, &cpu_trace.l2w_gradients);
        update_cpu_group!(l2b, cpu_l2b_momentum, cpu_l2b_velocity, cpu_l2b_slow, &cpu_trace.l2b_gradients);
        update_cpu_group!(outw, cpu_outw_momentum, cpu_outw_velocity, cpu_outw_slow, &cpu_trace.outw_gradients);
        update_cpu_group!(outb, cpu_outb_momentum, cpu_outb_velocity, cpu_outb_slow, &[cpu_trace.outb_gradient]);
    }

    let gpu_state = runner.read_state(&stream)?;
    macro_rules! compare_group {
        ($group:expr, $field:ident, $momentum:ident, $velocity:ident, $slow_params:ident) => {{
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_weights"),
                &cpu_case.$field,
                &$group.weights,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_momentum"),
                &$momentum,
                &$group.momentum,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_velocity"),
                &$velocity,
                &$group.velocity,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_slow"),
                &$slow_params,
                &$group.slow_params,
                args.tolerance,
            )?);
        }};
    }

    compare_group!(gpu_state.l0w, l0w, cpu_l0w_momentum, cpu_l0w_velocity, cpu_l0w_slow);
    compare_group!(gpu_state.l0b, l0b, cpu_l0b_momentum, cpu_l0b_velocity, cpu_l0b_slow);
    compare_group!(gpu_state.l1w, l1w, cpu_l1w_momentum, cpu_l1w_velocity, cpu_l1w_slow);
    compare_group!(gpu_state.l1b, l1b, cpu_l1b_momentum, cpu_l1b_velocity, cpu_l1b_slow);
    compare_group!(gpu_state.l2w, l2w, cpu_l2w_momentum, cpu_l2w_velocity, cpu_l2w_slow);
    compare_group!(gpu_state.l2b, l2b, cpu_l2b_momentum, cpu_l2b_velocity, cpu_l2b_slow);
    compare_group!(gpu_state.outw, outw, cpu_outw_momentum, cpu_outw_velocity, cpu_outw_slow);
    compare_group!(gpu_state.outb, outb, cpu_outb_momentum, cpu_outb_velocity, cpu_outb_slow);

    println!("bulletou-cuda-train NNUE loss Ranger step smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  fixtures     : {}", args.nnue_train_fixture_args.len());
    for (idx, fixture) in args.nnue_train_fixture_args.iter().enumerate() {
        match fixture {
            NnueTrainFixtureArg::Full(path) => println!("    [{}] full  {}", idx + 1, path.display()),
            NnueTrainFixtureArg::Batch(path) => println!("    [{}] batch {}", idx + 1, path.display()),
        }
    }
    println!("  loss_kind    : {}", loss_kind_label(args.loss_kind));
    println!("  batch        : {} samples", first_case.batch_size);
    println!(
        "  shape        : input={} l1={} l2={} l3={}",
        first_case.shape.input_size, first_case.shape.l1, first_case.shape.l2, first_case.shape.l3
    );
    println!("  tolerance    : {}", args.tolerance);
    println!(
        "  params       : steps=1..{} k={} alpha={}",
        train_batches.len(),
        grouped_ranger_step_params().k,
        grouped_ranger_step_params().lookahead.alpha
    );
    for cmp in comparisons {
        println!(
            "  {:<14}: max_abs={} at {}, mean_abs={}",
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
fn run_radam_update_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{optimizer::RAdamUpdateLayout, DeviceBuffer};

    let cases = RAdamUpdateCase::cases();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;

    println!("bulletou-cuda-train RAdam update smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  tolerance    : {}", args.tolerance);

    for case in cases {
        let cpu_trace = case.cpu_update_trace()?;
        let layout = RAdamUpdateLayout::new(case.weights.len());
        let gradients = DeviceBuffer::from_host(&stream, &case.gradients)?;
        let mut weights = DeviceBuffer::from_host(&stream, &case.weights)?;
        let mut momentum = DeviceBuffer::from_host(&stream, &case.momentum)?;
        let mut velocity = DeviceBuffer::from_host(&stream, &case.velocity)?;

        optimizer_update::launch_radam_update(
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

        println!("  case         : {}", case.label);
        println!("    len        : {}", case.weights.len());
        println!(
            "    params     : step={} lr={} decay={} beta1={} beta2={} eps={} clamp=[{}, {}] grad_factor={}",
            case.params.step,
            case.params.learning_rate,
            case.params.decay,
            case.params.beta1,
            case.params.beta2,
            case.params.epsilon,
            case.params.min_weight,
            case.params.max_weight,
            case.params.gradient_factor
        );
        println!(
            "    step_scale : step_size={} use_denom={}",
            cpu_trace.step_scale.step_size, cpu_trace.step_scale.use_denom
        );
        for cmp in comparisons {
            println!(
                "    {:<9}: max_abs={} at {}, mean_abs={}",
                cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
            );
        }
        if args.debug_readback {
            println!("    cpu weights: {:?}", cpu_trace.weights);
            println!("    gpu weights: {:?}", gpu_weights);
        }
    }

    println!("  compare      : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_ranger_lookahead_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{optimizer::RangerLookaheadLayout, DeviceBuffer};

    let case = RangerLookaheadCase::tiny();
    let cpu_trace = case.cpu_lookahead_trace();
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let layout = RangerLookaheadLayout::new(case.weights.len());
    let mut weights = DeviceBuffer::from_host(&stream, &case.weights)?;
    let mut slow_params = DeviceBuffer::from_host(&stream, &case.slow_params)?;

    optimizer_update::launch_ranger_lookahead(
        &stream,
        &module,
        layout,
        case.params,
        &mut weights,
        &mut slow_params,
    )?;
    stream.synchronize()?;

    let gpu_weights = weights.to_host_vec(&stream)?;
    let gpu_slow_params = slow_params.to_host_vec(&stream)?;
    let comparisons = [
        compare_slices("weights", &cpu_trace.weights, &gpu_weights, args.tolerance)?,
        compare_slices("slow_params", &cpu_trace.slow_params, &gpu_slow_params, args.tolerance)?,
    ];

    println!("bulletou-cuda-train Ranger lookahead smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!("  len          : {}", case.weights.len());
    println!("  tolerance    : {}", args.tolerance);
    println!("  params       : alpha={}", case.params.alpha);
    for cmp in comparisons {
        println!(
            "  {:<11}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
    }
    if args.debug_readback {
        println!("  cpu weights  : {:?}", cpu_trace.weights);
        println!("  gpu weights  : {:?}", gpu_weights);
        println!("  gpu slow     : {:?}", gpu_slow_params);
    }
    println!("  compare      : ok");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_ranger_update_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{
        optimizer::{OptimizerStateLayout, RangerOptimizerHostState, RangerOptimizerState, RangerUpdateLayout},
        DeviceBuffer,
    };

    let case = RangerUpdateCase::tiny();
    let cpu_trace = case.cpu_update_trace()?;
    let ptx = match args.ptx {
        Some(ptx) => ptx,
        None => default_nnue_ptx()?,
    };

    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(args.device)?;
    let stream = ctx.default_stream();
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let layout = RangerUpdateLayout::new(case.weights.len());
    let mut weights = DeviceBuffer::from_host(&stream, &case.weights)?;
    let state_layout = OptimizerStateLayout::new(case.weights.len());
    let host_state = RangerOptimizerHostState {
        momentum: &case.momentum,
        velocity: &case.velocity,
        slow_params: &case.slow_params,
    };
    let mut optimizer_state = RangerOptimizerState::from_host(&stream, state_layout, host_state)?;

    for step_idx in 0..case.gradients_by_step.len() {
        let gradients = DeviceBuffer::from_host(&stream, &case.gradients_by_step[step_idx])?;
        optimizer_update::launch_ranger_update(
            &stream,
            &module,
            layout,
            case.params_for_step(step_idx + 1),
            &gradients,
            &mut weights,
            &mut optimizer_state,
        )?;
    }
    stream.synchronize()?;

    let gpu_weights = weights.to_host_vec(&stream)?;
    let gpu_momentum = optimizer_state.momentum.to_host_vec(&stream)?;
    let gpu_velocity = optimizer_state.velocity.to_host_vec(&stream)?;
    let gpu_slow_params = optimizer_state.slow_params.to_host_vec(&stream)?;
    let comparisons = [
        compare_slices("weights", &cpu_trace.weights, &gpu_weights, args.tolerance)?,
        compare_slices("momentum", &cpu_trace.momentum, &gpu_momentum, args.tolerance)?,
        compare_slices("velocity", &cpu_trace.velocity, &gpu_velocity, args.tolerance)?,
        compare_slices("slow_params", &cpu_trace.slow_params, &gpu_slow_params, args.tolerance)?,
    ];

    println!("bulletou-cuda-train Ranger update chain smoke");
    println!("  ptx          : {}", ptx.display());
    println!("  device       : {}", args.device);
    println!("  case         : {}", case.label);
    println!("  len          : {}", case.weights.len());
    println!("  steps        : {}", case.gradients_by_step.len());
    println!("  tolerance    : {}", args.tolerance);
    println!("  params       : k={} alpha={}", case.k, case.lookahead.alpha);
    for cmp in comparisons {
        println!(
            "  {:<11}: max_abs={} at {}, mean_abs={}",
            cmp.name, cmp.max_abs_diff, cmp.max_abs_index, cmp.mean_abs_diff
        );
    }
    if args.debug_readback {
        println!("  lookahead at : {:?}", cpu_trace.lookahead_steps);
        println!("  cpu weights  : {:?}", cpu_trace.weights);
        println!("  gpu weights  : {:?}", gpu_weights);
        println!("  gpu slow     : {:?}", gpu_slow_params);
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

#[cfg(feature = "cuda")]
fn run_sfnn_ranger_step_smoke(args: Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    use bulletou_cuda_oxide_runtime::{
        backward::{
            SfnnBackwardWorkspace, SfnnBackwardWorkspaceLayout, SfnnL0SparseBackwardLayout, SfnnL2InputBackwardLayout,
            SfnnPairwiseBackwardLayout, SfnnStackedAffineBackwardLayout, SfnnStackedCReluBackwardLayout,
            SfnnStackedL3BackwardLayout,
        },
        optimizer::SfnnRangerOptimizerStates,
        sfnn::{
            SfnnForwardDeviceBatch, SfnnForwardDeviceWeights, SfnnForwardHostBatch, SfnnForwardHostWeights,
            SfnnForwardWorkspace, SfnnForwardWorkspaceLayout,
        },
        DeviceBuffer,
    };

    let case = match args.sfnn_forward_fixture.as_deref() {
        Some(path) => SfnnForwardCase::read_fixture(path)?,
        None => SfnnForwardCase::new(args.sfnn_case),
    };
    let cpu_forward_trace = case.cpu_forward_trace();
    let cpu_trace = case.cpu_output_backward_trace(&cpu_forward_trace);
    let params = grouped_ranger_step_params();
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
    let mut device_weights = SfnnForwardDeviceWeights::from_host(&stream, &host_weights)?;
    let mut optimizer_states = SfnnRangerOptimizerStates::from_host_weights(&stream, &host_weights)?;
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

    optimizer_update::launch_sfnn_ranger_update(
        &stream,
        &module,
        params,
        &mut device_weights,
        &backward_workspace,
        &mut optimizer_states,
    )?;
    stream.synchronize()?;

    let mut comparisons = Vec::new();
    macro_rules! compare_group {
        ($field:ident, $weights:expr, $gradients:expr) => {{
            let expected = cpu_single_ranger_update_trace($weights, $gradients, params)?;
            let gpu_weights = device_weights.$field.to_host_vec(&stream)?;
            let gpu_momentum = optimizer_states.$field.momentum.to_host_vec(&stream)?;
            let gpu_velocity = optimizer_states.$field.velocity.to_host_vec(&stream)?;
            let gpu_slow_params = optimizer_states.$field.slow_params.to_host_vec(&stream)?;
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_weights"),
                &expected.weights,
                &gpu_weights,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_momentum"),
                &expected.momentum,
                &gpu_momentum,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_velocity"),
                &expected.velocity,
                &gpu_velocity,
                args.tolerance,
            )?);
            comparisons.push(compare_slices(
                concat!(stringify!($field), "_slow"),
                &expected.slow_params,
                &gpu_slow_params,
                args.tolerance,
            )?);
        }};
    }

    compare_group!(l0w, &case.l0w, &cpu_trace.l0w_gradients);
    compare_group!(l0b, &case.l0b, &cpu_trace.l0b_gradients);
    compare_group!(l1w, &case.l1w, &cpu_trace.l1w_gradients);
    compare_group!(l1b, &case.l1b, &cpu_trace.l1b_gradients);
    compare_group!(l2w, &case.l2w, &cpu_trace.l2w_gradients);
    compare_group!(l2b, &case.l2b, &cpu_trace.l2b_gradients);
    compare_group!(l3w, &case.l3w, &cpu_trace.l3w_gradients);
    compare_group!(l3b, &case.l3b, &cpu_trace.l3b_gradients);

    println!("bulletou-cuda-train SFNN Ranger step smoke");
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
    println!("  params       : step={} k={} alpha={}", params.radam.step, params.k, params.lookahead.alpha);
    for cmp in comparisons {
        println!(
            "  {:<14}: max_abs={} at {}, mean_abs={}",
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
fn parse_u64_arg(value: String, option: &'static str) -> bulletou_cuda_oxide_runtime::Result<u64> {
    value.parse().map_err(|_| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!("{option} must be a non-negative integer\n\n{}", usage()))
    })
}

#[cfg(feature = "cuda")]
fn parse_u16_arg(value: String, option: &'static str) -> bulletou_cuda_oxide_runtime::Result<u16> {
    value.parse().map_err(|_| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "{option} must be an integer in [0, 65535]\n\n{}",
            usage()
        ))
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
fn parse_train_lr_schedule(value: String) -> bulletou_cuda_oxide_runtime::Result<TrainLrScheduleKind> {
    match value.as_str() {
        "fixed" => Ok(TrainLrScheduleKind::Fixed),
        "step" => Ok(TrainLrScheduleKind::Step),
        "geometric" => Ok(TrainLrScheduleKind::Geometric),
        "cos" | "cosine" => Ok(TrainLrScheduleKind::Cos),
        _ => usage_error(format!("--lr-schedule must be one of: fixed, step, geometric, cos (got {value})")),
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
       bulletou-cuda-train --nnue-fixture-train [--nnue-train-state-fixture <PATH>] --nnue-train-fixture <PATH>|--nnue-train-batch-fixture <PATH> [--nnue-train-fixture <PATH> | --nnue-train-batch-fixture <PATH> ...] [--write-nnue-trained-forward-fixture <PATH>] [--write-nnue-train-state-fixture <PATH>] [--loss-kind sigmoid-mse|wrm] [--ptx <PATH>] [--device <ID>] [--debug-readback]\n\
       bulletou-cuda-train --nnue-teacher-train --teacher <PATH> [--weights-bin <PATH>] [--nnue-train-state-fixture <PATH>] [--nnue-train-state-bin <PATH>] [--output <DIR>] [--train-steps <N>] [--save-rate <N>] [--batches-per-superbatch <N>] [--batch-size <N>] [--lr-schedule fixed|step|geometric|cos] [--learning-rate <F32>] [--lr-min <F32>] [--lr-step-gamma <F32>] [--lr-step-positions <N>] [--lr-period-positions <N>] [--optimizer-weight-decay <F32>] [--optimizer-epsilon <F32>] [--optimizer-beta1 <F32>] [--optimizer-beta2 <F32>] [--buffer-mb <N>] [--loader-threads <N>] [--threads <N>] [--score-drop-abs <N>] [--write-nnue-trained-forward-fixture <PATH>] [--write-nnue-train-state-fixture <PATH>] [--loss-kind sigmoid-mse|wrm] [--ptx <PATH>] [--device <ID>] [--debug-readback]\n\
       bulletou-cuda-train --nnue-forward-smoke [--nnue-forward-case tiny|halfkp] [--nnue-forward-fixture <PATH>] [--write-nnue-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --nnue-loss-ranger-step-smoke --nnue-train-fixture <PATH> [--nnue-train-fixture <PATH> | --nnue-train-batch-fixture <PATH> ...] [--loss-kind sigmoid-mse|wrm] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --nnue-ranger-step-smoke [--nnue-forward-case tiny|halfkp] [--nnue-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>]\n\
       bulletou-cuda-train --adamw-update-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --radam-update-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --ranger-lookahead-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --ranger-update-smoke [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --sfnn-dense-backward-smoke [--sfnn-forward-case tiny|halfka2] [--sfnn-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>]\n\
       bulletou-cuda-train --sfnn-output-backward-smoke [alias of --sfnn-dense-backward-smoke]\n\
       bulletou-cuda-train --sfnn-forward-smoke [--sfnn-forward-case tiny|halfka2] [--sfnn-forward-fixture <PATH>] [--write-sfnn-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>] [--debug-readback]\n\
       bulletou-cuda-train --sfnn-ranger-step-smoke [--sfnn-forward-case tiny|halfka2] [--sfnn-forward-fixture <PATH>] [--ptx <PATH>] [--device <ID>] [--tolerance <F32>]\n\
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
     CO-010 RAdam update smoke: compare the fused RAdam weight/momentum/\n\
     velocity update against CPU scalar goldens for both the warmup and\n\
     rectified denominator branches.\n\
     CO-010 Ranger lookahead smoke: compare the fast/slow parameter\n\
     interpolation kernel against a CPU scalar golden.\n\
     CO-010 Ranger update smoke: chain RAdam updates with conditional\n\
     Lookahead steps and compare all optimizer state buffers against a CPU\n\
     scalar golden.\n\
     CO-010 NNUE Ranger step smoke: run NNUE forward/backward, then update all\n\
     NNUE parameter groups with the Ranger launcher and compare weights plus\n\
     optimizer state buffers against CPU scalar goldens.\n\
     CO-010 NNUE loss Ranger step smoke: load an initial BOUNTRN1 fixture\n\
     with weights plus one or more BOUNTRN1/BOUNBCH1 batch fixtures, run NNUE\n\
     forward, scalar value loss, backward, and grouped Ranger updates while\n\
     carrying weights/optimizer state across fixtures, then compare against\n\
     CPU scalar goldens.\n\
     CO-010 NNUE fixture train: load the same initial BOUNTRN1 plus optional\n\
     BOUNTRN1/BOUNBCH1 batch sequence and run the NNUE loss/Ranger runner\n\
     without CPU-golden comparison, printing GPU loss readbacks for each step.\n\
     --write-nnue-trained-forward-fixture writes the final trained weights\n\
     plus the last batch layout as a BOUNFWD1 fixture for follow-up forward\n\
     validation.\n\
     --write-nnue-train-state-fixture writes BOUNRNG1 with final weights,\n\
     Ranger momentum/velocity/slow state, and completed step count. This is\n\
     the checkpoint/resume bridge format. --nnue-train-state-fixture restores\n\
     from BOUNRNG1, then applies the supplied later batch fixtures starting at\n\
     completed_steps + 1.\n\
     CO-010 NNUE teacher train: when built with --features cuda,root-loader,\n\
     read real teacher batches through bulletou_lib and feed them directly to\n\
     the NNUE loss/Ranger runner without writing BOUNTRN1/BOUNBCH1 fixtures.\n\
     It uses deterministic HalfKP initial weights by default, or --weights-bin\n\
     to load root weights.bin / bundled state.bin weights. It also supports\n\
     --nnue-train-state-fixture resume, --nnue-train-state-bin root-format\n\
     state.bin resume, and the same output fixture/state write flags as\n\
     --nnue-fixture-train. --output writes numbered cuda-oxide bridge\n\
     checkpoints with nn.bin, trained-forward.nnuef, state.boung, root state.bin,\n\
     teacher.txt, dataloader_pos.txt, and learn.log. --save-rate N writes every N superbatches\n\
     when --batches-per-superbatch is supplied (default 1, preserving the old\n\
     every-N-batches smoke behavior); the default 0 writes only the final batch.\n\
     LR schedule flags set the Ranger learning rate per batch. Later runs auto-resume the latest\n\
     numbered state.boung and HCPE/HCPE3/pack dataloader_pos.txt when the teacher spec matches.\n\
     CO-010 SFNN Ranger step smoke: run SFNN forward/backward, then update all\n\
     SFNN parameter groups with the Ranger launcher and compare weights plus\n\
     optimizer state buffers against CPU scalar goldens.\n\
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
const NNUE_TRAIN_FIXTURE_MAGIC: &[u8; 8] = b"BOUNTRN1";

#[cfg(feature = "cuda")]
const NNUE_TRAIN_BATCH_FIXTURE_MAGIC: &[u8; 8] = b"BOUNBCH1";

#[cfg(feature = "cuda")]
const NNUE_TRAIN_STATE_FIXTURE_MAGIC: &[u8; 8] = b"BOUNRNG1";

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
struct RAdamUpdateCase {
    label: &'static str,
    gradients: Vec<f32>,
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    params: bulletou_cuda_oxide_runtime::optimizer::RAdamUpdateParams,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RAdamUpdateTrace {
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    step_scale: bulletou_cuda_oxide_runtime::optimizer::RAdamStepScale,
}

#[cfg(feature = "cuda")]
impl RAdamUpdateCase {
    fn cases() -> [Self; 2] {
        [
            Self::new("warmup-no-denom", 1),
            Self::new("rectified-denom", 6),
        ]
    }

    fn new(label: &'static str, step: usize) -> Self {
        Self {
            label,
            gradients: vec![0.1, -0.2, 5.0, -5.0, 0.0, 0.3, -0.4],
            weights: vec![0.5, -0.25, 1.97, -1.97, 0.0, 1.2, -1.2],
            momentum: vec![0.0, 0.01, -0.02, 0.03, 0.0, 0.2, -0.1],
            velocity: vec![0.0, 0.0004, 0.0009, 0.0016, 0.0, 0.04, 0.09],
            params: bulletou_cuda_oxide_runtime::optimizer::RAdamUpdateParams {
                gradient_factor: 0.25,
                learning_rate: 0.01,
                step,
                decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                n_sma_threshold: 5.0,
                epsilon: 1.0e-8,
                min_weight: -1.0,
                max_weight: 1.0,
            },
        }
    }

    fn cpu_update_trace(&self) -> bulletou_cuda_oxide_runtime::Result<RAdamUpdateTrace> {
        let step_scale = self.params.step_scale()?;
        let rate = self.params.learning_rate * step_scale.step_size;
        let mut weights = self.weights.clone();
        let mut momentum = self.momentum.clone();
        let mut velocity = self.velocity.clone();

        for idx in 0..weights.len() {
            let grad = self.params.gradient_factor * self.gradients[idx];
            weights[idx] *= 1.0 - self.params.decay * rate;
            momentum[idx] = self.params.beta1 * momentum[idx] + (1.0 - self.params.beta1) * grad;
            velocity[idx] = self.params.beta2 * velocity[idx] + (1.0 - self.params.beta2) * grad * grad;
            let mut value = momentum[idx];
            if step_scale.use_denom {
                value /= velocity[idx].sqrt() + self.params.epsilon;
            }
            weights[idx] -= rate * value;
            weights[idx] = weights[idx].clamp(self.params.min_weight, self.params.max_weight);
        }

        Ok(RAdamUpdateTrace { weights, momentum, velocity, step_scale })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RangerLookaheadCase {
    label: &'static str,
    weights: Vec<f32>,
    slow_params: Vec<f32>,
    params: bulletou_cuda_oxide_runtime::optimizer::RangerLookaheadParams,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RangerLookaheadTrace {
    weights: Vec<f32>,
    slow_params: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl RangerLookaheadCase {
    fn tiny() -> Self {
        Self {
            label: "tiny-alpha-035",
            weights: vec![0.5, -0.25, 1.0, -1.0, 0.0, 0.75, -0.75],
            slow_params: vec![0.0, -0.5, 0.25, -0.25, 1.0, -1.0, 0.2],
            params: bulletou_cuda_oxide_runtime::optimizer::RangerLookaheadParams { alpha: 0.35 },
        }
    }

    fn cpu_lookahead_trace(&self) -> RangerLookaheadTrace {
        let mut updated = Vec::with_capacity(self.weights.len());
        for (&weight, &slow) in self.weights.iter().zip(&self.slow_params) {
            updated.push(self.params.alpha * weight + (1.0 - self.params.alpha) * slow);
        }

        RangerLookaheadTrace { weights: updated.clone(), slow_params: updated }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RangerUpdateCase {
    label: &'static str,
    gradients_by_step: Vec<Vec<f32>>,
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    slow_params: Vec<f32>,
    radam: bulletou_cuda_oxide_runtime::optimizer::RAdamUpdateParams,
    lookahead: bulletou_cuda_oxide_runtime::optimizer::RangerLookaheadParams,
    k: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RangerUpdateTrace {
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    slow_params: Vec<f32>,
    lookahead_steps: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl RangerUpdateCase {
    fn tiny() -> Self {
        Self {
            label: "tiny-radam-lookahead-k3",
            gradients_by_step: vec![
                vec![0.10, -0.20, 5.00, -5.00, 0.00, 0.30, -0.40],
                vec![0.20, -0.10, 4.00, -3.00, 0.05, -0.20, 0.10],
                vec![-0.15, 0.25, 3.00, -2.50, -0.05, 0.10, -0.30],
                vec![0.05, 0.15, -2.00, 2.00, 0.10, -0.15, 0.25],
                vec![-0.10, -0.05, 1.50, -1.50, -0.10, 0.05, -0.20],
                vec![0.30, -0.30, 0.75, -0.75, 0.20, -0.10, 0.15],
            ],
            weights: vec![0.5, -0.25, 1.2, -1.2, 0.0, 0.75, -0.75],
            momentum: vec![0.0; 7],
            velocity: vec![0.0; 7],
            slow_params: vec![0.0; 7],
            radam: bulletou_cuda_oxide_runtime::optimizer::RAdamUpdateParams {
                gradient_factor: 0.25,
                learning_rate: 0.01,
                step: 1,
                decay: 0.01,
                beta1: 0.99,
                beta2: 0.999,
                n_sma_threshold: 5.0,
                epsilon: 1.0e-8,
                min_weight: -1.0,
                max_weight: 1.0,
            },
            lookahead: bulletou_cuda_oxide_runtime::optimizer::RangerLookaheadParams { alpha: 0.5 },
            k: 3,
        }
    }

    fn params_for_step(&self, step: usize) -> bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams {
        bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams {
            radam: bulletou_cuda_oxide_runtime::optimizer::RAdamUpdateParams { step, ..self.radam },
            lookahead: self.lookahead,
            k: self.k,
        }
    }

    fn cpu_update_trace(&self) -> bulletou_cuda_oxide_runtime::Result<RangerUpdateTrace> {
        let mut weights = self.weights.clone();
        let mut momentum = self.momentum.clone();
        let mut velocity = self.velocity.clone();
        let mut slow_params = self.slow_params.clone();
        let mut lookahead_steps = Vec::new();

        for (step_idx, gradients) in self.gradients_by_step.iter().enumerate() {
            let step = step_idx + 1;
            let params = self.params_for_step(step);
            let step_scale = params.radam.step_scale()?;
            let rate = params.radam.learning_rate * step_scale.step_size;

            for idx in 0..weights.len() {
                let grad = params.radam.gradient_factor * gradients[idx];
                weights[idx] *= 1.0 - params.radam.decay * rate;
                momentum[idx] = params.radam.beta1 * momentum[idx] + (1.0 - params.radam.beta1) * grad;
                velocity[idx] = params.radam.beta2 * velocity[idx] + (1.0 - params.radam.beta2) * grad * grad;
                let mut value = momentum[idx];
                if step_scale.use_denom {
                    value /= velocity[idx].sqrt() + params.radam.epsilon;
                }
                weights[idx] -= rate * value;
                weights[idx] = weights[idx].clamp(params.radam.min_weight, params.radam.max_weight);
            }

            if params.should_lookahead()? {
                lookahead_steps.push(step);
                for idx in 0..weights.len() {
                    let new_weight =
                        params.lookahead.alpha * weights[idx] + (1.0 - params.lookahead.alpha) * slow_params[idx];
                    weights[idx] = new_weight;
                    slow_params[idx] = new_weight;
                }
            }
        }

        Ok(RangerUpdateTrace { weights, momentum, velocity, slow_params, lookahead_steps })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct SingleRangerUpdateTrace {
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    slow_params: Vec<f32>,
}

#[cfg(feature = "cuda")]
fn validate_nnue_teacher_schedule_args(args: &Args) -> bulletou_cuda_oxide_runtime::Result<()> {
    if !(args.learning_rate.is_finite() && args.learning_rate > 0.0) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--learning-rate must be finite and > 0".to_string(),
        ));
    }
    if !(args.lr_min.is_finite() && args.lr_min >= 0.0) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--lr-min must be finite and >= 0".to_string(),
        ));
    }
    if matches!(args.lr_schedule, TrainLrScheduleKind::Step | TrainLrScheduleKind::Geometric) && args.lr_min <= 0.0 {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--lr-min must be > 0 for --lr-schedule step/geometric".to_string(),
        ));
    }
    if args.lr_min > args.learning_rate {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--lr-min must be <= --learning-rate".to_string(),
        ));
    }
    if !(args.lr_step_gamma.is_finite() && args.lr_step_gamma > 0.0 && args.lr_step_gamma <= 1.0) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--lr-step-gamma must satisfy 0 < gamma <= 1".to_string(),
        ));
    }
    if matches!(args.lr_step_positions, Some(0)) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--lr-step-positions must be > 0".to_string(),
        ));
    }
    if !(args.optimizer_weight_decay.is_finite() && args.optimizer_weight_decay >= 0.0) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--optimizer-weight-decay must be finite and >= 0".to_string(),
        ));
    }
    if !(args.optimizer_epsilon.is_finite() && args.optimizer_epsilon > 0.0) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--optimizer-epsilon must be finite and > 0".to_string(),
        ));
    }
    if !(args.optimizer_beta1.is_finite() && args.optimizer_beta1 > 0.0 && args.optimizer_beta1 < 1.0) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--optimizer-beta1 must satisfy 0 < beta1 < 1".to_string(),
        ));
    }
    if !(args.optimizer_beta2.is_finite() && args.optimizer_beta2 > 0.0 && args.optimizer_beta2 < 1.0) {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "--optimizer-beta2 must satisfy 0 < beta2 < 1".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn nnue_teacher_save_interval_steps(args: &Args) -> bulletou_cuda_oxide_runtime::Result<Option<usize>> {
    if args.save_rate == 0 {
        return Ok(None);
    }
    args.save_rate
        .checked_mul(args.batches_per_superbatch)
        .filter(|&interval| interval > 0)
        .map(Some)
        .ok_or_else(|| {
            bulletou_cuda_oxide_runtime::Error::Smoke(
                "--save-rate * --batches-per-superbatch overflowed".to_string(),
            )
        })
}

#[cfg(feature = "cuda")]
fn nnue_teacher_learning_rate_for_step(args: &Args, step: usize) -> f32 {
    let total_positions = ((step.saturating_sub(1) as u128).saturating_mul(args.batch_size as u128))
        .min(u64::MAX as u128) as u64;
    match args.lr_schedule {
        TrainLrScheduleKind::Fixed => args.learning_rate,
        TrainLrScheduleKind::Step => {
            let step_positions = args.lr_step_positions.unwrap_or_else(|| {
                (args.batch_size as u64).saturating_mul(args.batches_per_superbatch as u64).max(1)
            });
            let epoch_pos = if args.lr_period_positions == 0 {
                total_positions
            } else {
                total_positions % args.lr_period_positions
            };
            let drops = epoch_pos / step_positions;
            let lr = (args.learning_rate as f64) * (args.lr_step_gamma as f64).powf(drops as f64);
            (lr as f32).max(args.lr_min)
        }
        TrainLrScheduleKind::Geometric => {
            if args.lr_period_positions == 0 {
                return args.learning_rate;
            }
            let t = (total_positions % args.lr_period_positions) as f64 / args.lr_period_positions as f64;
            let start = args.learning_rate as f64;
            let min = (args.lr_min as f64).max(1.0e-12);
            (start * (min / start).powf(t)) as f32
        }
        TrainLrScheduleKind::Cos => {
            if args.lr_period_positions == 0 {
                return args.learning_rate;
            }
            let t = (total_positions % args.lr_period_positions) as f64 / args.lr_period_positions as f64;
            let cos_val = (std::f64::consts::PI * t).cos();
            (args.lr_min as f64 + 0.5 * (args.learning_rate - args.lr_min) as f64 * (1.0 + cos_val)) as f32
        }
    }
}

#[cfg(feature = "cuda")]
fn train_lr_schedule_label(kind: TrainLrScheduleKind) -> &'static str {
    match kind {
        TrainLrScheduleKind::Fixed => "fixed",
        TrainLrScheduleKind::Step => "step",
        TrainLrScheduleKind::Geometric => "geometric",
        TrainLrScheduleKind::Cos => "cos",
    }
}

#[cfg(feature = "cuda")]
fn grouped_ranger_step_params() -> bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams {
    grouped_ranger_step_params_for_step(1)
}

#[cfg(feature = "cuda")]
fn grouped_ranger_step_params_for_step(step: usize) -> bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams {
    grouped_ranger_step_params_for_step_with_hyperparams(step, 0.01, 0.01, 0.99, 0.999, 1.0e-8)
}

#[cfg(feature = "cuda")]
fn grouped_ranger_step_params_for_step_with_hyperparams(
    step: usize,
    learning_rate: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams {
    bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams {
        radam: bulletou_cuda_oxide_runtime::optimizer::RAdamUpdateParams {
            gradient_factor: 0.25,
            learning_rate,
            step,
            decay,
            beta1,
            beta2,
            n_sma_threshold: 5.0,
            epsilon,
            min_weight: -1.98,
            max_weight: 1.98,
        },
        lookahead: bulletou_cuda_oxide_runtime::optimizer::RangerLookaheadParams { alpha: 0.5 },
        k: 1,
    }
}

#[cfg(feature = "cuda")]
fn cpu_single_ranger_update_trace(
    initial_weights: &[f32],
    gradients: &[f32],
    params: bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams,
) -> bulletou_cuda_oxide_runtime::Result<SingleRangerUpdateTrace> {
    let momentum = vec![0.0; initial_weights.len()];
    let velocity = vec![0.0; initial_weights.len()];
    let slow_params = initial_weights.to_vec();
    cpu_single_ranger_update_trace_from_state(initial_weights, &momentum, &velocity, &slow_params, gradients, params)
}

#[cfg(feature = "cuda")]
fn cpu_single_ranger_update_trace_from_state(
    initial_weights: &[f32],
    initial_momentum: &[f32],
    initial_velocity: &[f32],
    initial_slow_params: &[f32],
    gradients: &[f32],
    params: bulletou_cuda_oxide_runtime::optimizer::RangerUpdateParams,
) -> bulletou_cuda_oxide_runtime::Result<SingleRangerUpdateTrace> {
    if initial_weights.len() != gradients.len() {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "Ranger CPU trace length mismatch: weights={}, gradients={}",
            initial_weights.len(),
            gradients.len()
        )));
    }
    if initial_weights.len() != initial_momentum.len()
        || initial_weights.len() != initial_velocity.len()
        || initial_weights.len() != initial_slow_params.len()
    {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "Ranger CPU state length mismatch: weights={}, momentum={}, velocity={}, slow_params={}",
            initial_weights.len(),
            initial_momentum.len(),
            initial_velocity.len(),
            initial_slow_params.len()
        )));
    }

    params.validate()?;
    let step_scale = params.radam.step_scale()?;
    let rate = params.radam.learning_rate * step_scale.step_size;
    let mut weights = initial_weights.to_vec();
    let mut momentum = initial_momentum.to_vec();
    let mut velocity = initial_velocity.to_vec();
    let mut slow_params = initial_slow_params.to_vec();

    for idx in 0..weights.len() {
        let grad = params.radam.gradient_factor * gradients[idx];
        weights[idx] *= 1.0 - params.radam.decay * rate;
        momentum[idx] = params.radam.beta1 * momentum[idx] + (1.0 - params.radam.beta1) * grad;
        velocity[idx] = params.radam.beta2 * velocity[idx] + (1.0 - params.radam.beta2) * grad * grad;
        let mut value = momentum[idx];
        if step_scale.use_denom {
            value /= velocity[idx].sqrt() + params.radam.epsilon;
        }
        weights[idx] -= rate * value;
        weights[idx] = weights[idx].clamp(params.radam.min_weight, params.radam.max_weight);
    }

    if params.should_lookahead()? {
        for idx in 0..weights.len() {
            let new_weight = params.lookahead.alpha * weights[idx] + (1.0 - params.lookahead.alpha) * slow_params[idx];
            weights[idx] = new_weight;
            slow_params[idx] = new_weight;
        }
    }

    Ok(SingleRangerUpdateTrace { weights, momentum, velocity, slow_params })
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
struct NnueTrainCase {
    forward: NnueForwardCase,
    targets: Vec<f32>,
    entry_weights: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct NnueTrainStateGroupCase {
    weights: Vec<f32>,
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    slow_params: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct NnueTrainStateCase {
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
    completed_steps: usize,
    l0w: NnueTrainStateGroupCase,
    l0b: NnueTrainStateGroupCase,
    l1w: NnueTrainStateGroupCase,
    l1b: NnueTrainStateGroupCase,
    l2w: NnueTrainStateGroupCase,
    l2b: NnueTrainStateGroupCase,
    outw: NnueTrainStateGroupCase,
    outb: NnueTrainStateGroupCase,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct NnueTrainBatchCase {
    label: &'static str,
    input_size: usize,
    batch_size: usize,
    max_active: usize,
    stm: Vec<i32>,
    nstm: Vec<i32>,
    targets: Vec<f32>,
    entry_weights: Vec<f32>,
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
        self.cpu_dense_backward_trace_with_output_gradients(forward, output_gradients)
    }

    fn cpu_dense_backward_trace_with_output_gradients(
        &self,
        forward: &NnueForwardTrace,
        output_gradients: Vec<f32>,
    ) -> NnueDenseBackwardTrace {
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
impl NnueTrainCase {
    fn read_fixture(path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<Self> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(path).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to open NNUE train fixture {}: {err}",
                path.display()
            ))
        })?);

        let mut magic = [0_u8; 8];
        read_exact(&mut reader, &mut magic, "fixture magic")?;
        if &magic != NNUE_TRAIN_FIXTURE_MAGIC {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "invalid NNUE train fixture magic in {}",
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

        let forward = NnueForwardCase {
            label: "train-fixture",
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
        let targets = read_f32_vec(&mut reader, batch_size, "targets")?;
        let entry_weights = read_f32_vec(&mut reader, batch_size, "entry_weights")?;
        let case = Self { forward, targets, entry_weights };

        let mut trailing = [0_u8; 1];
        match std::io::Read::read(&mut reader, &mut trailing) {
            Ok(0) => Ok(case),
            Ok(_) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "NNUE train fixture {} has trailing bytes",
                path.display()
            ))),
            Err(err) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to read NNUE train fixture {}: {err}",
                path.display()
            ))),
        }
    }
}

#[cfg(feature = "cuda")]
impl NnueTrainStateCase {
    fn read_fixture(path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<Self> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(path).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to open NNUE train state fixture {}: {err}",
                path.display()
            ))
        })?);

        let mut magic = [0_u8; 8];
        read_exact(&mut reader, &mut magic, "fixture magic")?;
        if &magic != NNUE_TRAIN_STATE_FIXTURE_MAGIC {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "invalid NNUE train state fixture magic in {}",
                path.display()
            )));
        }

        let shape = bulletou_cuda_oxide_runtime::nnue::NnueForwardShape {
            input_size: read_usize(&mut reader, "shape.input_size")?,
            l1: read_usize(&mut reader, "shape.l1")?,
            l2: read_usize(&mut reader, "shape.l2")?,
            l3: read_usize(&mut reader, "shape.l3")?,
        };
        let completed_steps = read_usize(&mut reader, "completed_steps")?;
        let layout =
            bulletou_cuda_oxide_runtime::nnue::NnueForwardWeightLayout::new(shape);

        fn read_group(
            reader: &mut impl std::io::Read,
            len: usize,
            name: &'static str,
        ) -> bulletou_cuda_oxide_runtime::Result<NnueTrainStateGroupCase> {
            Ok(NnueTrainStateGroupCase {
                weights: read_f32_vec(reader, len, name)?,
                momentum: read_f32_vec(reader, len, name)?,
                velocity: read_f32_vec(reader, len, name)?,
                slow_params: read_f32_vec(reader, len, name)?,
            })
        }

        let case = Self {
            shape,
            completed_steps,
            l0w: read_group(&mut reader, layout.l0w_len(), "l0w")?,
            l0b: read_group(&mut reader, layout.l0b_len(), "l0b")?,
            l1w: read_group(&mut reader, layout.l1w_len(), "l1w")?,
            l1b: read_group(&mut reader, layout.l1b_len(), "l1b")?,
            l2w: read_group(&mut reader, layout.l2w_len(), "l2w")?,
            l2b: read_group(&mut reader, layout.l2b_len(), "l2b")?,
            outw: read_group(&mut reader, layout.outw_len(), "outw")?,
            outb: read_group(&mut reader, layout.outb_len(), "outb")?,
        };

        let mut trailing = [0_u8; 1];
        match std::io::Read::read(&mut reader, &mut trailing) {
            Ok(0) => Ok(case),
            Ok(_) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "NNUE train state fixture {} has trailing bytes",
                path.display()
            ))),
            Err(err) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to read NNUE train state fixture {}: {err}",
                path.display()
            ))),
        }
    }

    fn host_weights(&self) -> bulletou_cuda_oxide_runtime::nnue::NnueForwardHostWeights<'_> {
        bulletou_cuda_oxide_runtime::nnue::NnueForwardHostWeights {
            shape: self.shape,
            l0w: &self.l0w.weights,
            l0b: &self.l0b.weights,
            l1w: &self.l1w.weights,
            l1b: &self.l1b.weights,
            l2w: &self.l2w.weights,
            l2b: &self.l2b.weights,
            outw: &self.outw.weights,
            outb: &self.outb.weights,
        }
    }

    fn host_optimizer_states(
        &self,
    ) -> bulletou_cuda_oxide_runtime::optimizer::NnueRangerOptimizerHostStates<'_> {
        use bulletou_cuda_oxide_runtime::optimizer::{NnueRangerOptimizerHostStates, RangerOptimizerHostState};

        macro_rules! group {
            ($field:ident) => {
                RangerOptimizerHostState {
                    momentum: &self.$field.momentum,
                    velocity: &self.$field.velocity,
                    slow_params: &self.$field.slow_params,
                }
            };
        }

        NnueRangerOptimizerHostStates {
            l0w: group!(l0w),
            l0b: group!(l0b),
            l1w: group!(l1w),
            l1b: group!(l1b),
            l2w: group!(l2w),
            l2b: group!(l2b),
            outw: group!(outw),
            outb: group!(outb),
        }
    }
}

#[cfg(feature = "cuda")]
impl NnueTrainBatchCase {
    fn from_train_case(case: &NnueTrainCase) -> Self {
        Self {
            label: case.forward.label,
            input_size: case.forward.shape.input_size,
            batch_size: case.forward.batch_size,
            max_active: case.forward.max_active,
            stm: case.forward.stm.clone(),
            nstm: case.forward.nstm.clone(),
            targets: case.targets.clone(),
            entry_weights: case.entry_weights.clone(),
        }
    }

    fn read_fixture(path: &std::path::Path) -> bulletou_cuda_oxide_runtime::Result<Self> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(path).map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to open NNUE train batch fixture {}: {err}",
                path.display()
            ))
        })?);

        let mut magic = [0_u8; 8];
        read_exact(&mut reader, &mut magic, "fixture magic")?;
        if &magic != NNUE_TRAIN_BATCH_FIXTURE_MAGIC {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "invalid NNUE train batch fixture magic in {}",
                path.display()
            )));
        }

        let input_size = read_usize(&mut reader, "shape.input_size")?;
        let batch_size = read_usize(&mut reader, "batch_size")?;
        let max_active = read_usize(&mut reader, "max_active")?;
        let sparse_len = batch_size.saturating_mul(max_active);

        let case = Self {
            label: "train-batch-fixture",
            input_size,
            batch_size,
            max_active,
            stm: read_i32_vec(&mut reader, sparse_len, "stm")?,
            nstm: read_i32_vec(&mut reader, sparse_len, "nstm")?,
            targets: read_f32_vec(&mut reader, batch_size, "targets")?,
            entry_weights: read_f32_vec(&mut reader, batch_size, "entry_weights")?,
        };

        let mut trailing = [0_u8; 1];
        match std::io::Read::read(&mut reader, &mut trailing) {
            Ok(0) => Ok(case),
            Ok(_) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "NNUE train batch fixture {} has trailing bytes",
                path.display()
            ))),
            Err(err) => Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to read NNUE train batch fixture {}: {err}",
                path.display()
            ))),
        }
    }
}

#[cfg(feature = "cuda")]
fn write_nnue_train_state_fixture(
    path: &std::path::Path,
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
    completed_steps: usize,
    state: &nnue_train_step::NnueTrainStateReadback,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to create NNUE train state fixture {}: {err}",
            path.display()
        ))
    })?);

    write_all(&mut writer, NNUE_TRAIN_STATE_FIXTURE_MAGIC, "fixture magic")?;
    for value in [shape.input_size, shape.l1, shape.l2, shape.l3, completed_steps] {
        write_u64(&mut writer, value as u64)?;
    }

    macro_rules! write_group {
        ($group:expr, $name:literal) => {{
            write_f32_vec(&mut writer, &$group.weights, concat!($name, ".weights"))?;
            write_f32_vec(&mut writer, &$group.momentum, concat!($name, ".momentum"))?;
            write_f32_vec(&mut writer, &$group.velocity, concat!($name, ".velocity"))?;
            write_f32_vec(&mut writer, &$group.slow_params, concat!($name, ".slow_params"))?;
        }};
    }

    write_group!(state.l0w, "l0w");
    write_group!(state.l0b, "l0b");
    write_group!(state.l1w, "l1w");
    write_group!(state.l1b, "l1b");
    write_group!(state.l2w, "l2w");
    write_group!(state.l2b, "l2b");
    write_group!(state.outw, "outw");
    write_group!(state.outb, "outb");

    std::io::Write::flush(&mut writer).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to flush NNUE train state fixture {}: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_bridge_teacher_spec(path: &std::path::Path, teacher_spec: &str) -> bulletou_cuda_oxide_runtime::Result<()> {
    use std::io::Write as _;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to create bridge checkpoint teacher spec {}: {err}",
            path.display()
        ))
    })?);
    writeln!(writer, "{teacher_spec}").map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to write bridge checkpoint teacher spec {}: {err}",
            path.display()
        ))
    })?;
    writer.flush().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to flush bridge checkpoint teacher spec {}: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn write_nnue_bridge_dataloader_pos(
    path: &std::path::Path,
    dataloader_positions: &[Option<bulletou_lib::value::TeacherDataloaderPos>],
) -> bulletou_cuda_oxide_runtime::Result<Option<std::path::PathBuf>> {
    use std::io::Write as _;

    let Some(pos) = dataloader_positions.iter().rev().find_map(|pos| *pos) else {
        return Ok(None);
    };

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to create bridge checkpoint dataloader_pos.txt {}: {err}",
            path.display()
        ))
    })?);
    writeln!(writer, "{},{}", pos.byte_offset, pos.plies).map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to write bridge checkpoint dataloader_pos.txt {}: {err}",
            path.display()
        ))
    })?;
    writer.flush().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to flush bridge checkpoint dataloader_pos.txt {}: {err}",
            path.display()
        ))
    })?;
    Ok(Some(path.to_path_buf()))
}

#[cfg(all(feature = "cuda", feature = "root-loader"))]
fn append_nnue_bridge_summary_log(
    path: &std::path::Path,
    checkpoint_index: usize,
    completed_steps: usize,
    losses: &[(usize, f32, f32)],
    learning_rates: &[f32],
    sources: &[String],
    test_metrics: Option<NnueBridgeTestMetrics>,
    context: NnueBridgeLogContext,
) -> bulletou_cuda_oxide_runtime::Result<()> {
    use std::io::Write as _;

    let write_header = match std::fs::metadata(path) {
        Ok(meta) => meta.len() == 0,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
        Err(err) => {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to stat bridge summary log {}: {err}",
                path.display()
            )));
        }
    };
    let mut writer = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|err| {
                bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                    "failed to open bridge summary log {}: {err}",
                    path.display()
                ))
            })?,
    );

    if write_header {
        writeln!(
            writer,
            "eval,epoch,superbatch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher"
        )
        .map_err(|err| {
            bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "failed to write bridge summary log header {}: {err}",
                path.display()
            ))
        })?;
    }

    let Some((last_step, _weighted_sum, mean)) = losses.last().copied() else {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "cannot append bridge summary log without losses".to_string(),
        ));
    };
    let learning_rate = learning_rates.last().copied().unwrap_or(0.0);
    let source = sources.last().map(String::as_str).unwrap_or("");
    let display = bridge_log_display_step(last_step, context);
    let test_accuracy = test_metrics.map(|m| format!("{:.6}", m.accuracy)).unwrap_or_else(|| "-".to_string());
    let test_loss = test_metrics.map(|m| format!("{:.6}", m.loss)).unwrap_or_else(|| "-".to_string());
    let _ = (checkpoint_index, completed_steps);
    writeln!(
        writer,
        "nnue,{},{},{},{},{:.9},{:.9},{:.9},{:.9},{},{}",
        display.epoch,
        display.superbatch,
        test_accuracy,
        test_loss,
        mean,
        learning_rate,
        learning_rate,
        context.lambda,
        display.positions,
        csv_escape(source),
    )
    .map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to append bridge summary log {}: {err}",
            path.display()
        ))
    })?;
    writer.flush().map_err(|err| {
        bulletou_cuda_oxide_runtime::Error::Smoke(format!(
            "failed to flush bridge summary log {}: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn ensure_compatible_nnue_train_batches(
    shape: bulletou_cuda_oxide_runtime::nnue::NnueForwardShape,
    cases: &[NnueTrainBatchCase],
) -> bulletou_cuda_oxide_runtime::Result<()> {
    let Some(first) = cases.first() else {
        return Err(bulletou_cuda_oxide_runtime::Error::Smoke(
            "NNUE train smoke requires at least one batch".to_string(),
        ));
    };
    for (idx, case) in cases.iter().enumerate() {
        if case.input_size != shape.input_size || case.batch_size != first.batch_size || case.max_active != first.max_active {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "NNUE train batch #{} has incompatible layout: input={} batch={} max_active={}, expected input={} batch={} max_active={}",
                idx + 1,
                case.input_size,
                case.batch_size,
                case.max_active,
                shape.input_size,
                first.batch_size,
                first.max_active
            )));
        }
        if case.targets.len() != case.batch_size || case.entry_weights.len() != case.batch_size {
            return Err(bulletou_cuda_oxide_runtime::Error::Smoke(format!(
                "NNUE train batch #{} target/weight length mismatch: targets={} entry_weights={} batch={}",
                idx + 1,
                case.targets.len(),
                case.entry_weights.len(),
                case.batch_size
            )));
        }
    }
    Ok(())
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
#[derive(Debug, Clone)]
struct SliceComparison {
    name: String,
    max_abs_diff: f32,
    max_abs_index: usize,
    mean_abs_diff: f32,
}

#[cfg(feature = "cuda")]
fn compare_slices(
    name: impl Into<String>,
    expected: &[f32],
    actual: &[f32],
    tolerance: f32,
) -> bulletou_cuda_oxide_runtime::Result<SliceComparison> {
    let name = name.into();
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

#[cfg(all(test, feature = "cuda", feature = "root-loader"))]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("bulletou_cuda_train_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_bridge_dataloader_pos_accepts_nonzero_plies() {
        let dir = tmp_dir("parse_bridge_dataloader_pos_accepts_nonzero_plies");
        let path = dir.join(BRIDGE_DATALOADER_POS_NAME);
        std::fs::write(&path, "123,45\n").unwrap();

        let pos = parse_bridge_dataloader_pos(&path).unwrap().unwrap();
        assert_eq!(pos, bulletou_lib::value::TeacherDataloaderPos { byte_offset: 123, plies: 45 });

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn write_nnue_bridge_dataloader_pos_uses_last_available_position() {
        let dir = tmp_dir("write_nnue_bridge_dataloader_pos_uses_last_available_position");
        let path = dir.join(BRIDGE_DATALOADER_POS_NAME);
        let positions = [
            None,
            Some(bulletou_lib::value::TeacherDataloaderPos { byte_offset: 11, plies: 2 }),
            None,
            Some(bulletou_lib::value::TeacherDataloaderPos { byte_offset: 99, plies: 7 }),
        ];

        let written = write_nnue_bridge_dataloader_pos(&path, &positions).unwrap().unwrap();
        assert_eq!(written, path);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "99,7\n");

        std::fs::remove_dir_all(dir).unwrap();
    }
}
