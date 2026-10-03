use super::*;

#[test]
#[ignore = "requires CUDA; compares compact axis reduction to the original general reducer"]
fn compact_axis_backward_matches_reference_with_qat_and_bpu() {
    unsafe extern "C" {
        fn bulletou_bn_reference_mode(enabled: i32);
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            unsafe {
                bulletou_bn_reference_mode(0);
            }
        }
    }
    let _reset = Reset;
    let ctx = Context::new(0).unwrap();
    let upload = Context::new(0).unwrap();
    const N: usize = 1031;
    let indices: Vec<i32> = (0..N).map(|i| (i % 4) as i32).collect();
    let targets: Vec<f32> = (0..N).map(|i| (i % 11) as f32 / 10.0).collect();
    let entries: Vec<f32> = (0..N).map(|i| if i % 5 == 0 { 0.0 } else { 1.0 }).collect();
    for (stacks, king, hand, progress, pairs) in
        [(9, 3, 0, false, false), (8, 0, 0, true, false), (16, 2, 2, false, true)]
    {
        let shape = SfnnForwardShape {
            input_size: 4,
            ft_size: 1024,
            l1_hidden: 7,
            l1_skip: true,
            l2_size: 4,
            num_stacks: stacks,
            factorizer_king_axis_dim: king,
            factorizer_hand_axis_dim: hand,
            factorizer_progress_axis: progress,
            factorizer_king_hand_pair: pairs,
            ..tiny_sfnn_shape()
        };
        let values = |n, scale| (0..n).map(|i| ((i * 17 % 101) as f32 - 50.0) * scale).collect::<Vec<_>>();
        let w0 = values(4 * 1024, 0.002);
        let b0 = vec![0.5; 1024];
        let w1 = values(stacks * 1024 * 8, 0.0002);
        let b1 = vec![0.25; stacks * 8];
        let fw = values(1024 * 8, 0.0001);
        let fb = vec![0.03; 8];
        let aw = values(shape.factorizer_axis_count() * 1024 * 8, 0.0003);
        let ab = values(shape.factorizer_axis_count() * 8, 0.0002);
        let w2 = values(stacks * 14 * 4, 0.01);
        let b2 = vec![0.3; stacks * 4];
        let w3 = values(stacks * 4, 0.1);
        let b3 = vec![0.0; stacks];
        let host = SfnnForwardHostWeights {
            shape,
            l0w: &w0,
            l0b: &b0,
            l1w: &w1,
            l1b: &b1,
            l1fw: Some(&fw),
            l1fb: Some(&fb),
            l1axw: Some(&aw),
            l1axb: Some(&ab),
            l2w: &w2,
            l2b: &b2,
            l3w: &w3,
            l3b: &b3,
            l2fw: None,
            l2fb: None,
            l3fw: None,
            l3fb: None,
            ..tiny_sfnn_weights(tiny_sfnn_shape())
        };
        let buckets: Vec<i32> = (0..N).map(|i| (i % (stacks - 1)) as i32).collect();
        let batch = SfnnTrainStepHostBatch {
            stm_indices: &indices,
            nstm_indices: &indices,
            buckets: &buckets,
            targets: &targets,
            entry_weights: &entries,
            batch_size: N,
            max_active: 1,
        };
        for qat in [false, true] {
            for shared in [false, true] {
                let mut fast = SfnnTrainStepRunner::new(&ctx, host, N, 1).unwrap();
                let mut reference = SfnnTrainStepRunner::new(&ctx, host, N, 1).unwrap();
                for r in [&mut fast, &mut reference] {
                    let active = SfnnFactorizerActive { shared, ..r.factorizer };
                    r.set_factorizer_config(
                        active,
                        SfnnFactorizerAlpha {
                            shared: 0.7,
                            king_axis: 0.6,
                            hand_axis: 0.8,
                            progress_axis: 0.9,
                            pair: 0.4,
                            ..SfnnFactorizerAlpha::ONE
                        },
                    )
                    .unwrap();
                    let confidence: Vec<f32> =
                        (0..shape.factorizer_axis_count()).map(|i| 0.2 + (i % 5) as f32 * 0.15).collect();
                    r.set_factorizer_axis_confidences(&ctx, Some(&confidence)).unwrap();
                    let gates: Vec<f32> = (0..stacks).map(|i| if i == 0 { 0.0 } else { 0.6 }).collect();
                    r.set_residual_count_gates_by_stack(&ctx, Some(&gates)).unwrap();
                }
                let policy = SfnnLayerLrMultipliers { qat_l1: qat, ..Default::default() };
                for micro in 1..=4 {
                    for (r, old) in [(&mut fast, false), (&mut reference, true)] {
                        unsafe {
                            bulletou_bn_reference_mode(i32::from(old));
                        }
                        r.step_pipelined_no_readback_with_loss_finalize_update_and_lr_multipliers(
                            &ctx,
                            &upload,
                            RangerUpdateParams::default(),
                            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                            1.0,
                            batch,
                            true,
                            false,
                            policy,
                        )
                        .unwrap();
                    }
                    let a = fast.backward_workspace.download(&ctx).unwrap();
                    let b = reference.backward_workspace.download(&ctx).unwrap();
                    // Include axis gradients: a double-counted BPU contribution must fail.
                    for (name, x, y) in [
                        ("FT", &a.l0w_gradients, &b.l0w_gradients),
                        ("L1", &a.l1w_gradients, &b.l1w_gradients),
                        ("bias", &a.l1b_gradients, &b.l1b_gradients),
                        ("shared", &a.l1fw_gradients, &b.l1fw_gradients),
                        ("shared bias", &a.l1fb_gradients, &b.l1fb_gradients),
                    ] {
                        for (j, (&x, &y)) in x.iter().zip(y).enumerate() {
                            assert!(
                                (x - y).abs() < 2e-6 + y.abs() * 3e-4,
                                "{name} stacks={stacks} qat={qat} shared={shared} micro={micro} j={j}: {x} vs {y}"
                            );
                        }
                    }
                    for (x, y) in [
                        (&fast.backward_workspace.l1axw_gradients, &reference.backward_workspace.l1axw_gradients),
                        (&fast.backward_workspace.l1axb_gradients, &reference.backward_workspace.l1axb_gradients),
                    ] {
                        let x = x.download(&ctx).unwrap();
                        let y = y.download(&ctx).unwrap();
                        assert!(y.iter().any(|v| v.abs() > 1e-8), "vacuous axis gradient");
                        for (&x, &y) in x.iter().zip(&y) {
                            assert!((x - y).abs() < 2e-6 + y.abs() * 3e-4, "axis {x} vs {y}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; checks large-batch Ka2 and wide L1 fast paths"]
fn ka2_parallel_backward_matches_cpu_with_qat_and_accumulation() {
    let ctx = Context::new(0).unwrap();
    let upload = Context::new(0).unwrap();
    const BATCH: usize = 1031; // Partial row tile, skewed occurrence lists, empty features/bucket.
    let a: Vec<i32> = (0..BATCH).flat_map(|i| [0, 1 + (i % 23) as i32, -1]).collect();
    let b: Vec<i32> = (0..BATCH).flat_map(|i| [0, if i % 13 == 0 { 1789 } else { -1 }, 0]).collect();
    let targets: Vec<f32> = (0..BATCH).map(|i| 0.1 + (i % 9) as f32 * 0.1).collect();
    let ew: Vec<f32> = (0..BATCH).map(|i| if i % 5 == 0 { 0.0 } else { 1.0 }).collect();
    for (ft, hidden, skip, stacks) in [(1026, 8, false, 8), (1024, 7, true, 9), (2048, 8, true, 16)] {
        let buckets: Vec<i32> = (0..BATCH).map(|i| (i % (stacks - 1)) as i32).collect();
        let shape = SfnnForwardShape {
            input_size: 1791,
            ft_size: ft,
            l1_hidden: hidden,
            l1_skip: skip,
            l2_size: 4,
            num_stacks: stacks,
            ..tiny_sfnn_shape()
        };
        // All masters are exactly representable at export precision. Thus the
        // same independent CPU reference checks both ordinary and all-layer QAT.
        let pattern = |len, divisor| (0..len).map(|i| ((i % 7) as f32 - 3.0) / divisor).collect::<Vec<_>>();
        let w0 = pattern(shape.input_size * ft, 127.0);
        let b0 = vec![64.0 / 127.0; ft];
        let w1 = pattern(shape.num_stacks * shape.l1_out() * ft, 64.0);
        let b1 = vec![0.25; shape.num_stacks * shape.l1_out()];
        let fw = pattern(ft * shape.l1_out(), 64.0);
        let fb = vec![0.125; shape.l1_out()];
        let w2 = vec![1.0 / 64.0; shape.num_stacks * shape.l2_in() * shape.l2_size];
        let b2 = vec![0.125; shape.num_stacks * shape.l2_size];
        let w3 = vec![0.25; shape.num_stacks * shape.l2_size];
        let b3 = vec![0.125; shape.num_stacks];
        for shared in [false, true] {
            let host = SfnnForwardHostWeights {
                shape,
                l0w: &w0,
                l0b: &b0,
                l1w: &w1,
                l1b: &b1,
                l1fw: shared.then_some(fw.as_slice()),
                l1fb: shared.then_some(fb.as_slice()),
                l2w: &w2,
                l2b: &b2,
                l3w: &w3,
                l3b: &b3,
                l2fw: None,
                l2fb: None,
                l3fw: None,
                l3fb: None,
                ..tiny_sfnn_weights(tiny_sfnn_shape())
            };
            let batch = SfnnForwardHostBatch {
                stm_indices: &a,
                nstm_indices: &b,
                buckets: &buckets,
                batch_size: BATCH,
                max_active: 3,
            };
            let expected = tiny_sfnn_backward_cpu(batch, host, &targets, &ew);
            let train = SfnnTrainStepHostBatch {
                stm_indices: &a,
                nstm_indices: &b,
                buckets: &buckets,
                batch_size: BATCH,
                max_active: 3,
                targets: &targets,
                entry_weights: &ew,
            };
            for qat in [false, true] {
                let mut r = SfnnTrainStepRunner::new(&ctx, host, BATCH, 3).unwrap();
                let policy =
                    SfnnLayerLrMultipliers { qat_ft: qat, qat_l1: qat, qat_l2: qat, qat_l3: qat, ..Default::default() };
                for micro in 1..=4 {
                    r.step_pipelined_no_readback_with_loss_finalize_update_and_lr_multipliers(
                        &ctx,
                        &upload,
                        RangerUpdateParams::default(),
                        ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                        1.0,
                        train,
                        true,
                        false,
                        policy,
                    )
                    .unwrap();
                    for (name, actual, reference) in [
                        ("FT weight", &r.backward_workspace.l0w_gradients, &expected.l0w_gradients),
                        ("FT bias", &r.backward_workspace.l0b_gradients, &expected.l0b_gradients),
                        ("L1 weight", &r.backward_workspace.l1w_gradients, &expected.l1w_gradients),
                        ("L1 bias", &r.backward_workspace.l1b_gradients, &expected.l1b_gradients),
                        ("L1 shared weight", &r.backward_workspace.l1fw_gradients, &expected.l1fw_gradients),
                        ("L1 shared bias", &r.backward_workspace.l1fb_gradients, &expected.l1fb_gradients),
                    ] {
                        let actual = actual.download(&ctx).unwrap();
                        if actual.is_empty() {
                            continue;
                        }
                        assert_eq!(actual.len(), reference.len());
                        if shared || !name.contains("shared") {
                            assert!(reference.iter().any(|v| v.abs() > 1e-8), "vacuous {name}");
                        }
                        for (i, (&x, &y)) in actual.iter().zip(reference).enumerate() {
                            let y = y * micro as f32;
                            assert!(
                                (x - y).abs() < 2e-6 + y.abs() * 2e-4,
                                "{name} ft={ft} shared={shared} qat={qat} micro={micro} index={i}: {x} vs {y}"
                            );
                        }
                    }
                }
            }
        }
    }
}
