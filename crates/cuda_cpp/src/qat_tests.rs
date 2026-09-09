use super::*;

fn batch() -> SfnnTrainStepHostBatch<'static> {
    SfnnTrainStepHostBatch {
        stm_indices: &[0, 1, -1, 2, -1, -1, 1, 3, -1, 3, 2, -1],
        nstm_indices: &[2, -1, -1, 0, 3, -1, 0, -1, -1, 1, -1, -1],
        buckets: &[0, 1, 0, 1],
        targets: &[0.25, 0.75, 0.6, 0.1],
        entry_weights: &[1.0, 0.5, 0.75, 1.25],
        batch_size: 4,
        max_active: 3,
    }
}

fn host_batch() -> SfnnForwardHostBatch<'static> {
    let b = batch();
    SfnnForwardHostBatch {
        stm_indices: b.stm_indices,
        nstm_indices: b.nstm_indices,
        buckets: b.buckets,
        batch_size: b.batch_size,
        max_active: b.max_active,
    }
}

fn qdq(x: f32, scale: f32, low: f32, high: f32) -> f32 {
    (x * scale).round().clamp(low, high) / scale
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn sfnn_qat_l1_forward_ste_and_export_parity() {
    let ctx = Context::new(0).unwrap();
    for skip in [false, true] {
        for factors in [false, true] {
            let mut host = tiny_sfnn_weights(tiny_sfnn_shape());
            let old_out = host.shape.l1_out();
            host.shape.l1_skip = skip;
            host.shape.factorizer_progress_axis = factors;
            let s = host.shape;
            let out = s.l1_out();
            let mut w: Vec<_> = (0..s.num_stacks)
                .flat_map(|b| {
                    host.l1w[b * old_out * s.ft_size..b * old_out * s.ft_size + out * s.ft_size].iter().copied()
                })
                .collect();
            // Exercise ties away from zero and both saturation edges.
            w[..4].copy_from_slice(&[-0.5 / 64.0, 0.5 / 64.0, -8.0, 8.0]);
            let b: Vec<_> =
                (0..s.num_stacks).flat_map(|b| host.l1b[b * old_out..b * old_out + out].iter().copied()).collect();
            let shared: Vec<_> = (0..s.ft_size * out).map(|i| 0.01 * (i as f32 - 3.0)).collect();
            let shared_b = vec![0.002; out];
            let axis = vec![0.013; s.factorizer_axis_count() * s.ft_size * out];
            let axis_b = vec![0.003; s.factorizer_axis_count() * out];
            host.l1w = &w;
            host.l1b = &b;
            host.l1fw = factors.then_some(&shared);
            host.l1fb = factors.then_some(&shared_b);
            host.l1axw = factors.then_some(&axis);
            host.l1axb = factors.then_some(&axis_b);
            host.l2fw = None;
            host.l2fb = None;
            host.l3fw = None;
            host.l3fb = None;
            let active = SfnnFactorizerActive::from_host(host);
            let alpha = SfnnFactorizerAlpha { shared: 0.7, progress_axis: 1.3, ..SfnnFactorizerAlpha::ONE };
            let mut runner = SfnnTrainStepRunner::new_with_factorizer(&ctx, host, 4, 3, active, alpha).unwrap();
            if factors {
                runner.set_residual_count_gates_by_stack(&ctx, Some(&[0.5, 0.25])).unwrap();
                runner.set_factorizer_axis_confidences(&ctx, Some(&[0.6, 0.8])).unwrap();
            }
            let mut qw = Vec::new();
            let mut qb = Vec::new();
            for stack in 0..2 {
                let gate = if factors { [0.5, 0.25][stack] } else { 1.0 };
                let axis_alpha = alpha.progress_axis * [0.6, 0.8][stack];
                for o in 0..out {
                    let mut bias = gate * b[stack * out + o];
                    if factors {
                        bias += alpha.shared * shared_b[o];
                        bias += axis_alpha * axis_b[stack * out + o];
                    }
                    qb.push(qdq(bias, 8128.0, -2147483648.0, 2147483647.0));
                    for i in 0..s.ft_size {
                        let mut value = gate * w[(stack * out + o) * s.ft_size + i];
                        if factors {
                            value += alpha.shared * shared[i * out + o];
                            value += axis_alpha * axis[(stack * s.ft_size + i) * out + o];
                        }
                        qw.push(qdq(value, 64.0, -128.0, 127.0));
                    }
                }
            }
            let fake =
                SfnnForwardHostWeights { l1w: &qw, l1b: &qb, l1fw: None, l1fb: None, l1axw: None, l1axb: None, ..host };
            runner
                .step_no_readback_with_loss_finalize_update_and_lr_multipliers(
                    &ctx,
                    RangerUpdateParams::default(),
                    ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                    1.0,
                    batch(),
                    true,
                    false,
                    SfnnLayerLrMultipliers { qat_l1: true, ..Default::default() },
                )
                .unwrap();
            let qat = runner.forward_workspace.qat_l1.as_ref().unwrap();
            assert_close_slice("fold then round", &qat.weights.download(&ctx).unwrap(), &qw, 1e-7);
            assert_close_slice("bias QAT", &qat.bias.download(&ctx).unwrap(), &qb, 1e-7);
            assert!(!qat.refresh.get());
            assert_eq!(runner.read_weights(&ctx).unwrap().l1w, w, "FP32 masters must not be rounded in-place");
            assert_close_slice(
                "QAT forward CPU",
                &runner.forward_workspace.download_output(&ctx).unwrap(),
                &tiny_sfnn_forward_cpu(host_batch(), fake),
                1e-6,
            );
            let expected = tiny_sfnn_backward_cpu(host_batch(), fake, batch().targets, batch().entry_weights);
            let actual = runner.backward_workspace.download(&ctx).unwrap();
            assert_close_slice("QAT FT gradient uses rounded L1", &actual.l0w_gradients, &expected.l0w_gradients, 1e-6);
            assert_close_slice("QAT input gradient", &actual.combined_gradients, &expected.combined_gradients, 1e-6);
            assert_close_slice("identity STE L1", &actual.l1w_gradients, &expected.l1w_gradients, 1e-6);
            assert_close_slice("identity STE bias", &actual.l1b_gradients, &expected.l1b_gradients, 1e-6);
            assert_close_slice("L2 gradient", &actual.l2w_gradients, &expected.l2w_gradients, 1e-6);
            if factors {
                let mut shared_g = vec![0.0; shared.len()];
                let mut axis_g = vec![0.0; axis.len()];
                for stack in 0..2 {
                    for o in 0..out {
                        for i in 0..s.ft_size {
                            let g = expected.l1w_gradients[(stack * out + o) * s.ft_size + i];
                            shared_g[i * out + o] += alpha.shared * g;
                            axis_g[(stack * s.ft_size + i) * out + o] += alpha.progress_axis * [0.6, 0.8][stack] * g;
                        }
                    }
                }
                assert_close_slice("STE shared chain rule", &actual.l1fw_gradients, &shared_g, 1e-6);
                assert_close_slice("STE axis chain rule", &actual.l1axw_gradients, &axis_g, 1e-6);
                runner.apply_residual_count_gates_to_gradients(&ctx, SfnnLayerLrMultipliers::default(), None).unwrap();
                let gated = runner.backward_workspace.l1w_gradients.download(&ctx).unwrap();
                let expected_g: Vec<_> = expected
                    .l1w_gradients
                    .iter()
                    .enumerate()
                    .map(|(i, g)| g * [0.5, 0.25][i / (out * s.ft_size)])
                    .collect();
                assert_close_slice("STE residual chain rule", &gated, &expected_g, 1e-6);
            }
            // GPU qvalid/export quantizer and QAT use the same folded values.
            let proxy =
                SfnnForwardDeviceWeights::new_dense(&ctx, SfnnForwardShape { factorizer_progress_axis: false, ..s })
                    .unwrap();
            sfnn_build_quantized_proxy_device(
                &ctx,
                s.input_size,
                0,
                &runner.weights,
                &proxy,
                active,
                alpha,
                runner.residual_count_gates(),
                runner.factorizer_axis_confidences(),
            )
            .unwrap();
            assert_eq!(proxy.l1w.download(&ctx).unwrap(), qw);
            assert_close_slice("export bias", &proxy.l1b.download(&ctx).unwrap(), &qb, 1e-7);
            // test_value_* is still FP32, not the training fake-quantized forward.
            let val = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(s, 4)).unwrap();
            let dev_batch = SfnnForwardDeviceBatch::from_host(&ctx, host_batch()).unwrap();
            runner.forward_current_weights(&ctx, &dev_batch, &val).unwrap();
            let raw_output = val.download_output(&ctx).unwrap();
            runner.forward_current_weights(&ctx, &dev_batch, &runner.forward_workspace).unwrap();
            assert!(runner.forward_workspace.qat_l1_backward_weights().is_null());
            assert_eq!(raw_output, runner.forward_workspace.download_output(&ctx).unwrap());
            runner.prepare_l1_qat(&ctx, false).unwrap();
            assert!(runner.forward_workspace.qat_l1.is_none());
            runner
                .step_no_readback_with_loss_finalize_update_and_lr_multipliers(
                    &ctx,
                    RangerUpdateParams::default(),
                    ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                    1.0,
                    batch(),
                    true,
                    false,
                    SfnnLayerLrMultipliers::default(),
                )
                .unwrap();
            assert_eq!(raw_output, runner.forward_workspace.download_output(&ctx).unwrap());
        }
    }
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn sfnn_qat_l1_update_restore_pipeline_and_freeze() {
    let ctx = Context::new(0).unwrap();
    let upload = Context::new(0).unwrap();
    let host = tiny_sfnn_weights(tiny_sfnn_shape());
    let mut runner = SfnnTrainStepRunner::new(&ctx, host, 4, 3).unwrap();
    let original = runner.snapshot_device(&ctx).unwrap();
    let policy = SfnnLayerLrMultipliers { qat_l1: true, ..Default::default() };
    let mut params = RangerUpdateParams::default();
    params.radam.learning_rate = 0.01;
    params.radam.step = 1;
    for step in 0..4 {
        runner
            .step_pipelined_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                &ctx,
                &upload,
                params,
                ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                1.0,
                batch(),
                true,
                step == 3,
                policy,
                Some(&[0, 1]),
            )
            .unwrap();
        assert_eq!(runner.forward_workspace.qat_l1.as_ref().unwrap().refresh.get(), step == 3);
        if step < 3 {
            assert_eq!(runner.read_weights(&ctx).unwrap().l1w, host.l1w);
        }
    }
    assert_ne!(runner.read_weights(&ctx).unwrap().l1w, host.l1w);
    // Restore must invalidate the rounded cache too (worker trial A/B).
    runner.copy_state_from_device(&ctx, &original).unwrap();
    assert!(runner.forward_workspace.qat_l1.as_ref().unwrap().refresh.get());
    let frozen = SfnnLayerLrMultipliers { l1: 0.0, ..policy };
    runner
        .step_profiled_no_readback_with_update_and_lr_multipliers(
            &ctx,
            params,
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            batch(),
            true,
            frozen,
        )
        .unwrap();
    let trained = runner.read_weights(&ctx).unwrap();
    assert_eq!(trained.l1w, host.l1w);
    assert_eq!(trained.l1fw.as_deref(), host.l1fw);
    assert_ne!(trained.l0w, host.l0w, "freeze L1 must not freeze FT gradients");
    runner.copy_state_from_device(&ctx, &original).unwrap();
    runner.prepare_l1_qat(&ctx, true).unwrap();
    runner.forward_workspace.qat_l1.as_ref().unwrap().refresh.set(false);
    runner
        .set_factorizer_config(runner.factorizer, SfnnFactorizerAlpha { shared: 0.5, ..SfnnFactorizerAlpha::ONE })
        .unwrap();
    assert!(runner.forward_workspace.qat_l1.as_ref().unwrap().refresh.get());
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn sfnn_qat_l1_many_bucket_backward() {
    // Exercise the atomic/scatter path, not only the <=16-stack dense reducer.
    let ctx = Context::new(0).unwrap();
    let mut host = tiny_sfnn_weights(tiny_sfnn_shape());
    host.shape.num_stacks = 256;
    host.shape.factorizer_progress_axis = true;
    let s = host.shape;
    let w = host.l1w.repeat(128);
    let b = host.l1b.repeat(128);
    let w2 = host.l2w.repeat(128);
    let b2 = host.l2b.repeat(128);
    let w3 = host.l3w.repeat(128);
    let b3 = host.l3b.repeat(128);
    let aw = vec![0.013; s.factorizer_axis_count() * s.ft_size * s.l1_out()];
    let ab = vec![0.003; s.factorizer_axis_count() * s.l1_out()];
    host.l1w = &w;
    host.l1b = &b;
    host.l2w = &w2;
    host.l2b = &b2;
    host.l3w = &w3;
    host.l3b = &b3;
    host.l1axw = Some(&aw);
    host.l1axb = Some(&ab);
    host.l2fw = None;
    host.l2fb = None;
    host.l3fw = None;
    host.l3fb = None;
    let mut runner = SfnnTrainStepRunner::new(&ctx, host, 4, 3).unwrap();
    runner
        .step_no_readback_with_loss_finalize_update_and_lr_multipliers(
            &ctx,
            RangerUpdateParams::default(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            batch(),
            true,
            false,
            SfnnLayerLrMultipliers { qat_l1: true, ..Default::default() },
        )
        .unwrap();
    let qat = runner.forward_workspace.qat_l1.as_ref().unwrap();
    let qw = qat.weights.download(&ctx).unwrap();
    let qb = qat.bias.download(&ctx).unwrap();
    let fake = SfnnForwardHostWeights { l1w: &qw, l1b: &qb, l1fw: None, l1fb: None, l1axw: None, l1axb: None, ..host };
    let expected = tiny_sfnn_backward_cpu(host_batch(), fake, batch().targets, batch().entry_weights);
    let actual = runner.backward_workspace.download(&ctx).unwrap();
    assert_close_slice("large-stack QAT FT gradients", &actual.l0w_gradients, &expected.l0w_gradients, 1e-6);
    assert_close_slice("large-stack QAT STE", &actual.l1w_gradients, &expected.l1w_gradients, 1e-6);
    let mut sg = vec![0.0; s.ft_size * s.l1_out()];
    for stack in 0..s.num_stacks {
        for o in 0..s.l1_out() {
            for i in 0..s.ft_size {
                sg[i * s.l1_out() + o] += expected.l1w_gradients[(stack * s.l1_out() + o) * s.ft_size + i];
            }
        }
    }
    assert_close_slice("large-stack QAT shared gradients", &actual.l1fw_gradients, &sg, 1e-6);
}
