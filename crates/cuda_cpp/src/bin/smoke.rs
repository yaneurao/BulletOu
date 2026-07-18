use bulletou_cuda_cpp::{
    Context, Event, F32Buffer, F32UploadSlot, NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardHostBatch,
    NnueForwardHostWeights, NnueForwardShape, NnueForwardWorkspace, NnueForwardWorkspaceLayout, RAdamUpdateParams,
    RangerDeviceStateMut, RangerStateMut, RangerUpdateParams,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = 0;
    let name = bulletou_cuda_cpp::device_name(device)?;
    println!("bulletou-cuda-cpp smoke");
    println!("  device : {device}");
    println!("  name   : {name}");

    let axpy = bulletou_cuda_cpp::axpy_host(device, 2.0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])?;
    println!("  axpy   : {axpy:?}");
    assert_eq!(axpy, vec![12.0, 24.0, 36.0, 48.0]);

    let ctx = Context::new(device)?;
    let x_dev = F32Buffer::from_host(&ctx, &[1.0, 2.0, 3.0, 4.0])?;
    let y_dev = F32Buffer::from_host(&ctx, &[10.0, 20.0, 30.0, 40.0])?;
    let out_dev = F32Buffer::new(&ctx, 4)?;
    let start = Event::new(&ctx)?;
    let stop = Event::new(&ctx)?;
    start.record(&ctx)?;
    bulletou_cuda_cpp::axpy_device(&ctx, 4, 2.0, &x_dev, &y_dev, &out_dev)?;
    stop.record(&ctx)?;
    stop.synchronize()?;
    let axpy_ms = stop.elapsed_ms_since(&start)?;
    let axpy_device = out_dev.download(&ctx)?;
    println!("  axpy_d : {axpy_device:?} ({axpy_ms:.3} ms)");
    assert_eq!(axpy_device, vec![12.0, 24.0, 36.0, 48.0]);

    let graph_out = F32Buffer::new(&ctx, 4)?;
    ctx.begin_capture()?;
    bulletou_cuda_cpp::axpy_device(&ctx, 4, 2.0, &x_dev, &y_dev, &graph_out)?;
    let graph = ctx.end_capture()?;
    graph_out.fill(&ctx, 0.0)?;
    graph.launch(&ctx)?;
    graph.launch(&ctx)?;
    ctx.synchronize()?;
    let graph_axpy = graph_out.download(&ctx)?;
    println!("  graph  : {graph_axpy:?}");
    assert_eq!(graph_axpy, vec![12.0, 24.0, 36.0, 48.0]);

    let upload_ctx = Context::new(device)?;
    let upload_x = F32UploadSlot::new(&upload_ctx, 4)?;
    let upload_y = F32UploadSlot::new(&upload_ctx, 4)?;
    upload_x.upload(&upload_ctx, &[1.0, 2.0, 3.0, 4.0])?;
    upload_y.upload(&upload_ctx, &[10.0, 20.0, 30.0, 40.0])?;
    let upload_out = F32Buffer::new(&ctx, 4)?;
    let ready_x = upload_x.wait_on(&ctx)?;
    let ready_y = upload_y.wait_on(&ctx)?;
    bulletou_cuda_cpp::axpy_device(&ctx, 4, 2.0, ready_x, ready_y, &upload_out)?;
    let upload_axpy = upload_out.download(&ctx)?;
    println!("  upload : {upload_axpy:?}");
    assert_eq!(upload_axpy, vec![12.0, 24.0, 36.0, 48.0]);

    let shape = tiny_nnue_shape();
    let batch = tiny_nnue_batch();
    let weights = tiny_nnue_weights(shape);
    let nnue_host = bulletou_cuda_cpp::nnue_forward_host(device, batch, weights)?;
    println!("  nnue_h: {nnue_host:?}");
    assert_close_slice("nnue_h", &nnue_host, &[1.208, 1.1195], 1.0e-5);

    let device_batch = NnueForwardDeviceBatch::from_host(&ctx, batch)?;
    let device_weights = NnueForwardDeviceWeights::from_host(&ctx, weights)?;
    let workspace = NnueForwardWorkspace::new(&ctx, NnueForwardWorkspaceLayout::new(shape, batch.batch_size))?;
    bulletou_cuda_cpp::nnue_forward_device(&ctx, &device_batch, &device_weights, &workspace)?;
    let nnue_device = workspace.download_output(&ctx)?;
    println!("  nnue_d: {nnue_device:?}");
    assert_close_slice("nnue_d", &nnue_device, &[1.208, 1.1195], 1.0e-5);

    let mut gradients = vec![0.25, -0.5, 1.0, -1.5];
    let mut weights = vec![0.1, -0.2, 0.3, -0.4];
    let mut momentum = vec![0.0; gradients.len()];
    let mut velocity = vec![0.0; gradients.len()];
    let mut slow_params = weights.clone();
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
    bulletou_cuda_cpp::ranger_update_host(
        device,
        params,
        RangerStateMut {
            gradients: &mut gradients,
            weights: &mut weights,
            momentum: &mut momentum,
            velocity: &mut velocity,
            slow_params: &mut slow_params,
        },
    )?;
    println!("  ranger : weights={weights:?} gradients={gradients:?}");
    assert!(gradients.iter().all(|&g| g == 0.0));

    let gradients_dev = F32Buffer::from_host(&ctx, &[0.25, -0.5, 1.0, -1.5])?;
    let weights_dev = F32Buffer::from_host(&ctx, &[0.1, -0.2, 0.3, -0.4])?;
    let momentum_dev = F32Buffer::from_host(&ctx, &[0.0; 4])?;
    let velocity_dev = F32Buffer::from_host(&ctx, &[0.0; 4])?;
    let slow_dev = F32Buffer::from_host(&ctx, &[0.1, -0.2, 0.3, -0.4])?;
    bulletou_cuda_cpp::ranger_update_device(
        &ctx,
        params,
        RangerDeviceStateMut {
            gradients: &gradients_dev,
            weights: &weights_dev,
            momentum: &momentum_dev,
            velocity: &velocity_dev,
            slow_params: &slow_dev,
        },
    )?;
    ctx.synchronize()?;
    let gradients_device = gradients_dev.download(&ctx)?;
    let weights_device = weights_dev.download(&ctx)?;
    println!("  ranger_d: weights={weights_device:?} gradients={gradients_device:?}");
    assert!(gradients_device.iter().all(|&g| g == 0.0));
    assert_eq!(weights_device, weights);

    println!("  result : ok");
    Ok(())
}

fn tiny_nnue_shape() -> NnueForwardShape {
    NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 }
}

fn tiny_nnue_batch() -> NnueForwardHostBatch<'static> {
    NnueForwardHostBatch {
        stm_indices: &[0, 1, -1, 3, -1, -1],
        nstm_indices: &[2, -1, -1, 1, 2, -1],
        batch_size: 2,
        max_active: 3,
    }
}

fn tiny_nnue_weights(shape: NnueForwardShape) -> NnueForwardHostWeights<'static> {
    assert_eq!(shape, tiny_nnue_shape());
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
