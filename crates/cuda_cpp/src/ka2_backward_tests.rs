use super::*;

#[test]
#[ignore = "requires CUDA; checks large-batch Ka2 and wide L1 fast paths"]
fn ka2_parallel_backward_matches_cpu_with_qat_and_accumulation() {
    let ctx = Context::new(0).unwrap();
    let upload = Context::new(0).unwrap();
    const BATCH: usize = 1031; // Partial row tile, skewed occurrence lists, empty features/bucket.
    let a: Vec<i32> = (0..BATCH).flat_map(|i| [0, 1 + (i % 23) as i32, -1]).collect();
    let b: Vec<i32> = (0..BATCH).flat_map(|i| [0, if i % 13 == 0 { 1789 } else { -1 }, 0]).collect();
    let buckets: Vec<i32> = (0..BATCH).map(|i| (i % 7) as i32).collect();
    let targets: Vec<f32> = (0..BATCH).map(|i| 0.1 + (i % 9) as f32 * 0.1).collect();
    let ew: Vec<f32> = (0..BATCH).map(|i| if i % 5 == 0 { 0.0 } else { 1.0 }).collect();
    for (ft, skip) in [(1026, false), (2048, true)] {
        let shape = SfnnForwardShape {
            input_size: 1791,
            ft_size: ft,
            l1_hidden: 8,
            l1_skip: skip,
            l2_size: 4,
            num_stacks: 8,
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
