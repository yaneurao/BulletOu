use bulletou_cuda_cpp::{
    Context, Event, F32Buffer, F32UploadSlot, NnueForwardDeviceBatch, NnueForwardDeviceWeights, NnueForwardHostBatch,
    NnueForwardHostWeights, NnueForwardShape, NnueForwardWorkspace, NnueForwardWorkspaceLayout, RAdamUpdateParams,
    RangerDeviceStateMut, RangerStateMut, RangerUpdateParams, ScalarLossDeviceBatch, ScalarLossHostBatch,
    ScalarLossKind, ScalarLossWorkspace, ScalarLossWorkspaceLayout, SfnnForwardDeviceBatch, SfnnForwardDeviceWeights,
    SfnnForwardHostBatch, SfnnForwardHostWeights, SfnnForwardShape, SfnnForwardWorkspace, SfnnForwardWorkspaceLayout,
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

    let sfnn_shape = tiny_sfnn_shape();
    let sfnn_batch = tiny_sfnn_batch();
    let sfnn_weights = tiny_sfnn_weights(sfnn_shape);
    let sfnn_expected = cpu_tiny_sfnn_forward(sfnn_batch, sfnn_weights);
    let sfnn_device_batch = SfnnForwardDeviceBatch::from_host(&ctx, sfnn_batch)?;
    let sfnn_device_weights = SfnnForwardDeviceWeights::from_host(&ctx, sfnn_weights)?;
    let sfnn_workspace =
        SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(sfnn_shape, sfnn_batch.batch_size))?;
    bulletou_cuda_cpp::sfnn_forward_device(&ctx, &sfnn_device_batch, &sfnn_device_weights, &sfnn_workspace)?;
    let sfnn_device = sfnn_workspace.download_output(&ctx)?;
    println!("  sfnn_d: {sfnn_device:?}");
    assert_close_slice("sfnn_d", &sfnn_device, &sfnn_expected, 1.0e-5);

    let loss_batch = tiny_loss_batch();
    let loss_host = bulletou_cuda_cpp::scalar_loss_host(device, ScalarLossKind::SigmoidMse, 1.0, loss_batch)?;
    println!("  loss_h: mean={} weighted_sum={}", loss_host.mean, loss_host.weighted_sum);
    assert_close_slice("loss_h per_sample", &loss_host.per_sample, &[0.014209336, 0.0, 0.028418668], 1.0e-6);
    assert_close_slice(
        "loss_h mean_output_gradients",
        &loss_host.mean_output_gradients,
        &[0.008343695, 0.0, -0.01668739],
        1.0e-6,
    );
    assert_close_scalar("loss_h weighted_sum", loss_host.weighted_sum, 0.042628005, 1.0e-6);
    assert_close_scalar("loss_h mean", loss_host.mean, 0.014209335, 1.0e-6);

    let loss_device_batch = ScalarLossDeviceBatch::from_host(&ctx, loss_batch)?;
    let loss_workspace = ScalarLossWorkspace::new(&ctx, ScalarLossWorkspaceLayout::new(loss_batch.batch_size()))?;
    bulletou_cuda_cpp::scalar_loss_device(&ctx, ScalarLossKind::SigmoidMse, 1.0, &loss_device_batch, &loss_workspace)?;
    let loss_device = loss_workspace.download(&ctx)?;
    println!("  loss_d: mean={} weighted_sum={}", loss_device.mean, loss_device.weighted_sum);
    assert_close_slice("loss_d per_sample", &loss_device.per_sample, &loss_host.per_sample, 1.0e-6);
    assert_close_slice(
        "loss_d mean_output_gradients",
        &loss_device.mean_output_gradients,
        &loss_host.mean_output_gradients,
        1.0e-6,
    );
    assert_close_scalar("loss_d weighted_sum", loss_device.weighted_sum, loss_host.weighted_sum, 1.0e-6);
    assert_close_scalar("loss_d mean", loss_device.mean, loss_host.mean, 1.0e-6);

    let nnue_targets = [0.25, 0.75];
    let nnue_entry_weights = [1.0, 0.5];
    let nnue_targets_dev = F32Buffer::from_host(&ctx, &nnue_targets)?;
    let nnue_entry_weights_dev = F32Buffer::from_host(&ctx, &nnue_entry_weights)?;
    let nnue_loss_workspace = ScalarLossWorkspace::new(&ctx, ScalarLossWorkspaceLayout::new(batch.batch_size))?;
    bulletou_cuda_cpp::scalar_loss_device_from_buffers(
        &ctx,
        ScalarLossKind::SigmoidMse,
        1.0,
        batch.batch_size,
        &workspace.output,
        &nnue_targets_dev,
        &nnue_entry_weights_dev,
        &nnue_loss_workspace,
    )?;
    let backward_workspace = bulletou_cuda_cpp::NnueBackwardWorkspace::new(
        &ctx,
        bulletou_cuda_cpp::NnueBackwardWorkspaceLayout::new(shape, batch.batch_size, batch.max_active),
    )?;
    bulletou_cuda_cpp::nnue_backward_device(
        &ctx,
        &device_batch,
        &device_weights,
        &workspace,
        &nnue_loss_workspace,
        &backward_workspace,
    )?;
    let backward_device = backward_workspace.download(&ctx)?;
    let backward_expected = cpu_tiny_nnue_backward(batch, weights, &nnue_targets, &nnue_entry_weights);
    println!("  bwd_d : outb_grad={:?} l0b_grad={:?}", backward_device.outb_gradients, backward_device.l0b_gradients);
    assert_close_slice("bwd hidden2", &backward_device.hidden2_gradients, &backward_expected.hidden2_gradients, 1.0e-6);
    assert_close_slice("bwd hidden1", &backward_device.hidden1_gradients, &backward_expected.hidden1_gradients, 1.0e-6);
    assert_close_slice(
        "bwd combined",
        &backward_device.combined_gradients,
        &backward_expected.combined_gradients,
        1.0e-6,
    );
    assert_close_slice("bwd stm_l0", &backward_device.stm_l0_gradients, &backward_expected.stm_l0_gradients, 1.0e-6);
    assert_close_slice("bwd nstm_l0", &backward_device.nstm_l0_gradients, &backward_expected.nstm_l0_gradients, 1.0e-6);
    assert_close_slice("bwd l0w", &backward_device.l0w_gradients, &backward_expected.l0w_gradients, 1.0e-6);
    assert_close_slice("bwd l0b", &backward_device.l0b_gradients, &backward_expected.l0b_gradients, 1.0e-6);
    assert_close_slice("bwd l1w", &backward_device.l1w_gradients, &backward_expected.l1w_gradients, 1.0e-6);
    assert_close_slice("bwd l1b", &backward_device.l1b_gradients, &backward_expected.l1b_gradients, 1.0e-6);
    assert_close_slice("bwd l2w", &backward_device.l2w_gradients, &backward_expected.l2w_gradients, 1.0e-6);
    assert_close_slice("bwd l2b", &backward_device.l2b_gradients, &backward_expected.l2b_gradients, 1.0e-6);
    assert_close_slice("bwd outw", &backward_device.outw_gradients, &backward_expected.outw_gradients, 1.0e-6);
    assert_close_slice("bwd outb", &backward_device.outb_gradients, &backward_expected.outb_gradients, 1.0e-6);

    let train_params = RangerUpdateParams {
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
    let mut train_runner =
        bulletou_cuda_cpp::NnueTrainStepRunner::new(&ctx, weights, batch.batch_size, batch.max_active)?;
    let train_loss = train_runner.step(
        &ctx,
        train_params,
        ScalarLossKind::SigmoidMse,
        1.0,
        bulletou_cuda_cpp::NnueTrainStepHostBatch {
            stm_indices: batch.stm_indices,
            nstm_indices: batch.nstm_indices,
            targets: &nnue_targets,
            entry_weights: &nnue_entry_weights,
            batch_size: batch.batch_size,
            max_active: batch.max_active,
        },
    )?;
    let nnue_loss = nnue_loss_workspace.download(&ctx)?;
    assert_close_scalar("train loss mean", train_loss.mean, nnue_loss.mean, 1.0e-6);
    assert_close_scalar("train loss weighted_sum", train_loss.weighted_sum, nnue_loss.weighted_sum, 1.0e-6);
    let trained_weights = train_runner.read_weights(&ctx)?;
    let expected_trained_weights = host_ranger_updated_tiny_weights(device, weights, &backward_expected, train_params)?;
    println!("  train : loss_mean={} outb={:?}", train_loss.mean, trained_weights.outb);
    assert_close_slice("train l0w", &trained_weights.l0w, &expected_trained_weights.l0w, 1.0e-6);
    assert_close_slice("train l0b", &trained_weights.l0b, &expected_trained_weights.l0b, 1.0e-6);
    assert_close_slice("train l1w", &trained_weights.l1w, &expected_trained_weights.l1w, 1.0e-6);
    assert_close_slice("train l1b", &trained_weights.l1b, &expected_trained_weights.l1b, 1.0e-6);
    assert_close_slice("train l2w", &trained_weights.l2w, &expected_trained_weights.l2w, 1.0e-6);
    assert_close_slice("train l2b", &trained_weights.l2b, &expected_trained_weights.l2b, 1.0e-6);
    assert_close_slice("train outw", &trained_weights.outw, &expected_trained_weights.outw, 1.0e-6);
    assert_close_slice("train outb", &trained_weights.outb, &expected_trained_weights.outb, 1.0e-6);

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

fn tiny_sfnn_batch() -> SfnnForwardHostBatch<'static> {
    SfnnForwardHostBatch {
        stm_indices: &[0, 1, -1, 2, -1, -1],
        nstm_indices: &[2, -1, -1, 0, 3, -1],
        buckets: &[0, 1],
        batch_size: 2,
        max_active: 3,
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
        l2fw: Some(&[
            0.01, -0.02, 0.03, -0.04, // shared row 0
            -0.03, 0.02, -0.01, 0.04, // shared row 1
        ]),
        l2fb: Some(&[0.006, -0.008]),
        l3w: &[0.6, -0.4, 0.5, 0.2],
        l3b: &[0.05, -0.02],
        l3fw: Some(&[0.03, -0.02]),
        l3fb: Some(&[0.004]),
    }
}

fn cpu_tiny_sfnn_forward(batch: SfnnForwardHostBatch<'_>, weights: SfnnForwardHostWeights<'_>) -> Vec<f32> {
    let shape = weights.shape;
    let mut out = vec![0.0; batch.batch_size];
    for (sample, out_sample) in out.iter_mut().enumerate() {
        let stack = batch.buckets[sample] as usize;
        let sparse_base = sample * batch.max_active;
        let mut stm_l0 = weights.l0b.to_vec();
        let mut nstm_l0 = weights.l0b.to_vec();
        add_sparse_l0_sfnn(
            &mut stm_l0,
            weights.l0w,
            shape.ft_size,
            shape.input_size,
            &batch.stm_indices[sparse_base..sparse_base + batch.max_active],
        );
        add_sparse_l0_sfnn(
            &mut nstm_l0,
            weights.l0w,
            shape.ft_size,
            shape.input_size,
            &batch.nstm_indices[sparse_base..sparse_base + batch.max_active],
        );
        stm_l0.iter_mut().for_each(|value| *value = value.clamp(0.0, 1.0));
        nstm_l0.iter_mut().for_each(|value| *value = value.clamp(0.0, 1.0));

        let mut combined = vec![0.0; shape.ft_size];
        for pair in 0..shape.pairwise_size() {
            combined[pair] = stm_l0[pair] * stm_l0[shape.pairwise_size() + pair] * (127.0 / 128.0);
            combined[shape.pairwise_size() + pair] =
                nstm_l0[pair] * nstm_l0[shape.pairwise_size() + pair] * (127.0 / 128.0);
        }

        let mut l1 = stacked_affine_sfnn_cpu(&combined, weights.l1w, weights.l1b, shape.ft_size, shape.l1_out(), stack);
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

        let mut l2 = stacked_affine_sfnn_cpu(&l2_input, weights.l2w, weights.l2b, shape.l2_in(), shape.l2_size, stack);
        if let (Some(l2fw), Some(l2fb)) = (weights.l2fw, weights.l2fb) {
            for row in 0..shape.l2_size {
                l2[row] += l2fb[row];
                for input in 0..shape.l2_in() {
                    l2[row] += l2_input[input] * l2fw[row * shape.l2_in() + input];
                }
            }
        }
        l2.iter_mut().for_each(|value| *value = value.clamp(0.0, 1.0));

        let mut value = weights.l3b[stack] + psqt;
        if let Some(l3fb) = weights.l3fb {
            value += l3fb[0];
        }
        for input in 0..shape.l2_size {
            let mut weight = weights.l3w[stack * shape.l2_size + input];
            if let Some(l3fw) = weights.l3fw {
                weight += l3fw[input];
            }
            value += l2[input] * weight;
        }
        *out_sample = value;
    }
    out
}

fn add_sparse_l0_sfnn(out: &mut [f32], weights: &[f32], rows: usize, input_size: usize, indices: &[i32]) {
    for &feature in indices {
        if feature >= 0 && (feature as usize) < input_size {
            let base = feature as usize * rows;
            for row in 0..rows {
                out[row] += weights[base + row];
            }
        }
    }
}

fn stacked_affine_sfnn_cpu(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    input_dim: usize,
    output_dim: usize,
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

fn tiny_loss_batch() -> ScalarLossHostBatch<'static> {
    ScalarLossHostBatch { outputs: &[-2.0, 0.0, 2.0], targets: &[0.0, 0.5, 1.0], entry_weights: &[1.0, 0.5, 2.0] }
}

struct CpuBackward {
    hidden2_gradients: Vec<f32>,
    hidden1_gradients: Vec<f32>,
    combined_gradients: Vec<f32>,
    stm_l0_gradients: Vec<f32>,
    nstm_l0_gradients: Vec<f32>,
    l0w_gradients: Vec<f32>,
    l0b_gradients: Vec<f32>,
    l1w_gradients: Vec<f32>,
    l1b_gradients: Vec<f32>,
    l2w_gradients: Vec<f32>,
    l2b_gradients: Vec<f32>,
    outw_gradients: Vec<f32>,
    outb_gradients: Vec<f32>,
}

struct TinyWeightsOwned {
    l0w: Vec<f32>,
    l0b: Vec<f32>,
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    outw: Vec<f32>,
    outb: Vec<f32>,
}

fn host_ranger_updated_tiny_weights(
    device: i32,
    weights: NnueForwardHostWeights<'_>,
    gradients: &CpuBackward,
    params: RangerUpdateParams,
) -> Result<TinyWeightsOwned, Box<dyn std::error::Error>> {
    let mut out = TinyWeightsOwned {
        l0w: weights.l0w.to_vec(),
        l0b: weights.l0b.to_vec(),
        l1w: weights.l1w.to_vec(),
        l1b: weights.l1b.to_vec(),
        l2w: weights.l2w.to_vec(),
        l2b: weights.l2b.to_vec(),
        outw: weights.outw.to_vec(),
        outb: weights.outb.to_vec(),
    };
    apply_host_ranger(device, params, &gradients.l0w_gradients, &mut out.l0w)?;
    apply_host_ranger(device, params, &gradients.l0b_gradients, &mut out.l0b)?;
    apply_host_ranger(device, params, &gradients.l1w_gradients, &mut out.l1w)?;
    apply_host_ranger(device, params, &gradients.l1b_gradients, &mut out.l1b)?;
    apply_host_ranger(device, params, &gradients.l2w_gradients, &mut out.l2w)?;
    apply_host_ranger(device, params, &gradients.l2b_gradients, &mut out.l2b)?;
    apply_host_ranger(device, params, &gradients.outw_gradients, &mut out.outw)?;
    apply_host_ranger(device, params, &gradients.outb_gradients, &mut out.outb)?;
    Ok(out)
}

fn apply_host_ranger(
    device: i32,
    params: RangerUpdateParams,
    gradients: &[f32],
    weights: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut gradients = gradients.to_vec();
    let mut momentum = vec![0.0; gradients.len()];
    let mut velocity = vec![0.0; gradients.len()];
    let mut slow_params = weights.to_vec();
    bulletou_cuda_cpp::ranger_update_host(
        device,
        params,
        RangerStateMut {
            gradients: &mut gradients,
            weights,
            momentum: &mut momentum,
            velocity: &mut velocity,
            slow_params: &mut slow_params,
        },
    )?;
    Ok(())
}

fn cpu_tiny_nnue_backward(
    batch: NnueForwardHostBatch<'_>,
    weights: NnueForwardHostWeights<'_>,
    targets: &[f32],
    entry_weights: &[f32],
) -> CpuBackward {
    let shape = weights.shape;
    let trace = cpu_forward_trace(batch, weights);
    let mut output_gradients = vec![0.0; batch.batch_size];
    for sample in 0..batch.batch_size {
        let prediction = sigmoid(trace.outputs[sample]);
        let error = prediction - targets[sample];
        output_gradients[sample] =
            entry_weights[sample] * 2.0 * error * prediction * (1.0 - prediction) / batch.batch_size as f32;
    }

    let mut hidden2_gradients = vec![0.0; batch.batch_size * shape.l3];
    let mut outw_gradients = vec![0.0; shape.l3];
    let mut outb_gradients = vec![0.0; 1];
    for sample in 0..batch.batch_size {
        outb_gradients[0] += output_gradients[sample];
        for row in 0..shape.l3 {
            hidden2_gradients[sample * shape.l3 + row] = output_gradients[sample] * weights.outw[row];
            outw_gradients[row] += output_gradients[sample] * trace.hidden2[sample * shape.l3 + row];
        }
    }

    let (hidden1_gradients, l2w_gradients, l2b_gradients) = dense_crelu_backward_cpu(
        &trace.hidden1,
        &trace.hidden2,
        &hidden2_gradients,
        weights.l2w,
        batch.batch_size,
        shape.l2,
        shape.l3,
    );
    let (combined_gradients, l1w_gradients, l1b_gradients) = dense_crelu_backward_cpu(
        &trace.combined,
        &trace.hidden1,
        &hidden1_gradients,
        weights.l1w,
        batch.batch_size,
        shape.l1 * 2,
        shape.l2,
    );

    let mut stm_l0_gradients = vec![0.0; batch.batch_size * shape.l1];
    let mut nstm_l0_gradients = vec![0.0; batch.batch_size * shape.l1];
    for sample in 0..batch.batch_size {
        for row in 0..shape.l1 {
            let combined_base = sample * shape.l1 * 2;
            let perspective_idx = sample * shape.l1 + row;
            stm_l0_gradients[perspective_idx] =
                crelu_pre_gradient(trace.stm_l0[perspective_idx], combined_gradients[combined_base + row]);
            nstm_l0_gradients[perspective_idx] =
                crelu_pre_gradient(trace.nstm_l0[perspective_idx], combined_gradients[combined_base + shape.l1 + row]);
        }
    }

    let mut l0w_gradients = vec![0.0; shape.input_size * shape.l1];
    let mut l0b_gradients = vec![0.0; shape.l1];
    for sample in 0..batch.batch_size {
        for row in 0..shape.l1 {
            let perspective_idx = sample * shape.l1 + row;
            l0b_gradients[row] += stm_l0_gradients[perspective_idx] + nstm_l0_gradients[perspective_idx];
        }
        let sparse_base = sample * batch.max_active;
        for slot in 0..batch.max_active {
            let stm_feature = batch.stm_indices[sparse_base + slot];
            if stm_feature >= 0 && (stm_feature as usize) < shape.input_size {
                for row in 0..shape.l1 {
                    l0w_gradients[stm_feature as usize * shape.l1 + row] += stm_l0_gradients[sample * shape.l1 + row];
                }
            }
            let nstm_feature = batch.nstm_indices[sparse_base + slot];
            if nstm_feature >= 0 && (nstm_feature as usize) < shape.input_size {
                for row in 0..shape.l1 {
                    l0w_gradients[nstm_feature as usize * shape.l1 + row] += nstm_l0_gradients[sample * shape.l1 + row];
                }
            }
        }
    }

    CpuBackward {
        hidden2_gradients,
        hidden1_gradients,
        combined_gradients,
        stm_l0_gradients,
        nstm_l0_gradients,
        l0w_gradients,
        l0b_gradients,
        l1w_gradients,
        l1b_gradients,
        l2w_gradients,
        l2b_gradients,
        outw_gradients,
        outb_gradients,
    }
}

struct CpuTrace {
    stm_l0: Vec<f32>,
    nstm_l0: Vec<f32>,
    combined: Vec<f32>,
    hidden1: Vec<f32>,
    hidden2: Vec<f32>,
    outputs: Vec<f32>,
}

fn cpu_forward_trace(batch: NnueForwardHostBatch<'_>, weights: NnueForwardHostWeights<'_>) -> CpuTrace {
    let shape = weights.shape;
    let mut trace = CpuTrace {
        stm_l0: vec![0.0; batch.batch_size * shape.l1],
        nstm_l0: vec![0.0; batch.batch_size * shape.l1],
        combined: vec![0.0; batch.batch_size * shape.l1 * 2],
        hidden1: vec![0.0; batch.batch_size * shape.l2],
        hidden2: vec![0.0; batch.batch_size * shape.l3],
        outputs: vec![0.0; batch.batch_size],
    };
    for sample in 0..batch.batch_size {
        let sparse_base = sample * batch.max_active;
        let l0_base = sample * shape.l1;
        sparse_l0_cpu(
            weights.l0w,
            weights.l0b,
            shape.input_size,
            shape.l1,
            &batch.stm_indices[sparse_base..sparse_base + batch.max_active],
            &mut trace.stm_l0[l0_base..l0_base + shape.l1],
        );
        sparse_l0_cpu(
            weights.l0w,
            weights.l0b,
            shape.input_size,
            shape.l1,
            &batch.nstm_indices[sparse_base..sparse_base + batch.max_active],
            &mut trace.nstm_l0[l0_base..l0_base + shape.l1],
        );
        let combined_base = sample * shape.l1 * 2;
        trace.combined[combined_base..combined_base + shape.l1]
            .copy_from_slice(&trace.stm_l0[l0_base..l0_base + shape.l1]);
        trace.combined[combined_base + shape.l1..combined_base + shape.l1 * 2]
            .copy_from_slice(&trace.nstm_l0[l0_base..l0_base + shape.l1]);
        dense_crelu_cpu(
            weights.l1w,
            weights.l1b,
            shape.l1 * 2,
            shape.l2,
            &trace.combined[combined_base..combined_base + shape.l1 * 2],
            &mut trace.hidden1[sample * shape.l2..(sample + 1) * shape.l2],
        );
        dense_crelu_cpu(
            weights.l2w,
            weights.l2b,
            shape.l2,
            shape.l3,
            &trace.hidden1[sample * shape.l2..(sample + 1) * shape.l2],
            &mut trace.hidden2[sample * shape.l3..(sample + 1) * shape.l3],
        );
        let mut sum = weights.outb[0];
        for row in 0..shape.l3 {
            sum += trace.hidden2[sample * shape.l3 + row] * weights.outw[row];
        }
        trace.outputs[sample] = sum;
    }
    trace
}

fn sparse_l0_cpu(weights: &[f32], bias: &[f32], input_size: usize, rows: usize, active: &[i32], out: &mut [f32]) {
    out.copy_from_slice(&bias[..rows]);
    for &feature in active {
        if feature >= 0 && (feature as usize) < input_size {
            let base = feature as usize * rows;
            for row in 0..rows {
                out[row] += weights[base + row];
            }
        }
    }
    for value in out {
        *value = value.clamp(0.0, 1.0);
    }
}

fn dense_crelu_cpu(weights: &[f32], bias: &[f32], input_dim: usize, output_dim: usize, input: &[f32], out: &mut [f32]) {
    out.copy_from_slice(&bias[..output_dim]);
    for in_col in 0..input_dim {
        for out_col in 0..output_dim {
            out[out_col] += input[in_col] * weights[in_col * output_dim + out_col];
        }
    }
    for value in out {
        *value = value.clamp(0.0, 1.0);
    }
}

fn dense_crelu_backward_cpu(
    inputs: &[f32],
    activations: &[f32],
    output_gradients: &[f32],
    weights: &[f32],
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut input_gradients = vec![0.0; batch_size * input_dim];
    let mut weight_gradients = vec![0.0; input_dim * output_dim];
    let mut bias_gradients = vec![0.0; output_dim];
    for sample in 0..batch_size {
        for out_col in 0..output_dim {
            let out_idx = sample * output_dim + out_col;
            let grad = crelu_pre_gradient(activations[out_idx], output_gradients[out_idx]);
            bias_gradients[out_col] += grad;
            for in_col in 0..input_dim {
                input_gradients[sample * input_dim + in_col] += grad * weights[in_col * output_dim + out_col];
                weight_gradients[in_col * output_dim + out_col] += grad * inputs[sample * input_dim + in_col];
            }
        }
    }
    (input_gradients, weight_gradients, bias_gradients)
}

fn crelu_pre_gradient(activation: f32, output_gradient: f32) -> f32 {
    if activation > 0.0 && activation < 1.0 { output_gradient } else { 0.0 }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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

fn assert_close_scalar(name: &str, actual: f32, expected: f32, tolerance: f32) {
    let abs_diff = (actual - expected).abs();
    assert!(abs_diff <= tolerance, "{name}: expected {expected}, got {actual}, abs_diff={abs_diff}");
}
