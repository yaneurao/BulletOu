use super::*;

#[test]
#[ignore = "requires CUDA; checks large king/hand/progress axis and pair indexing"]
fn large_king_hand_progress_axis_pair_quantization_matches_reference() {
    let ctx=Context::new(0).unwrap();
    for hand_dim in [16usize,32] {
        let k=9usize; let h=hand_dim*hand_dim; let p=8usize; let nk=k*k; let stacks=nk*h*p;
        let shape=SfnnForwardShape {input_size:4,ft_size:2,l1_hidden:1,l1_skip:true,l2_size:1,num_stacks:stacks,
          factorizer_king_axis_dim:k,factorizer_hand_axis_dim:hand_dim,factorizer_progress_axis:true,
          factorizer_king_hand_pair:true,factorizer_king_progress_pair:true,factorizer_hand_progress_pair:true,..tiny_sfnn_shape()};
        let axis_count=shape.factorizer_axis_count(); let out=shape.l1_out();
        let l0w=vec![0.1;8]; let l0b=vec![0.5;2]; let l1w=vec![0.0;stacks*2*out]; let l1b=vec![0.0;stacks*out];
        let l1fw=vec![0.0;2*out]; let l1fb=vec![0.0;out]; let aw=vec![0.0;axis_count*2*out];
        let ab:Vec<f32>=(0..axis_count*out).map(|i|((i/out)%97) as f32/8128.0).collect();
        let l2w=vec![0.0;stacks*2]; let l2b=vec![0.0;stacks]; let l3w=vec![0.0;stacks]; let l3b=vec![0.0;stacks];
        let weights=SfnnForwardHostWeights {shape,l0w:&l0w,l0b:&l0b,l1w:&l1w,l1b:&l1b,l1fw:Some(&l1fw),l1fb:Some(&l1fb),
          l1axw:Some(&aw),l1axb:Some(&ab),l2w:&l2w,l2b:&l2b,l3w:&l3w,l3b:&l3b,
          l2fw:None,l2fb:None,l2axw:None,l2axb:None,l3fw:None,l3fb:None,l3axw:None,l3axb:None};
        let source=SfnnForwardDeviceWeights::from_host(&ctx,weights).unwrap();
        let ps=SfnnForwardShape {factorizer_king_axis_dim:0,factorizer_hand_axis_dim:0,factorizer_progress_axis:false,
          factorizer_king_hand_pair:false,factorizer_king_progress_pair:false,factorizer_hand_progress_pair:false,..shape};
        let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,ps).unwrap();
        let active=SfnnFactorizerActive {shared:true,king_axis:true,hand_axis:true,progress_axis:true,
          king_hand_pair:true,king_progress_pair:true,hand_progress_pair:true};
        sfnn_build_quantized_proxy_device(&ctx,4,0,&source,&proxy,active,SfnnFactorizerAlpha::ONE,None,None).unwrap();
        let actual=proxy.l1b.download(&ctx).unwrap();
        for &stack in &[0,1,7,8,647,648,stacks/2,stacks-2,stacks-1] {
            let progress=stack%p; let king=(stack/p)%nk; let hand=(stack/p/nk)%h; let base=2*k+2*hand_dim;
            let ids=[king/k,k+king%k,2*k+hand/hand_dim,2*k+hand_dim+hand%hand_dim,
              base+hand*nk+king,base+h*nk+progress*nk+king,
              base+h*nk+p*nk+progress*h+hand,base+h*nk+p*nk+p*h+progress];
            let expected: f32=ids.iter().map(|&id|ab[id*out]).sum();
            assert!((actual[stack*out]-expected).abs()<1e-6,"hand_dim={hand_dim} stack={stack}");
        }
    }
}

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
    for (stacks, king, hand, progress, pairs, hidden) in
        [(9, 3, 0, false, false, 7), (8, 0, 0, true, false, 7), (16, 2, 2, false, true, 7),
         (8, 0, 0, true, false, 15), (9, 3, 0, false, false, 15),
         (8, 0, 0, false, false, 15)]
    {
        let shape = SfnnForwardShape {
            input_size: 4,
            ft_size: 1024,
            l1_hidden: hidden,
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
        let outputs = hidden + 1;
        let w1 = values(stacks * 1024 * outputs, 0.0002);
        let b1 = vec![0.25; stacks * outputs];
        let fw = values(1024 * outputs, 0.0001);
        let fb = vec![0.03; outputs];
        let aw = values(shape.factorizer_axis_count() * 1024 * outputs, 0.0003);
        let ab = values(shape.factorizer_axis_count() * outputs, 0.0002);
        let w2 = values(stacks * hidden * 2 * 4, 0.01);
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
            l1axw: (king != 0 || hand != 0 || progress).then_some(&aw),
            l1axb: (king != 0 || hand != 0 || progress).then_some(&ab),
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
                    if king != 0 || hand != 0 || progress {
                        r.set_factorizer_axis_confidences(&ctx, Some(&confidence)).unwrap();
                    }
                    let gates: Vec<f32> = (0..stacks).map(|i| if i == 0 { 0.0 } else { 0.6 }).collect();
                    if king != 0 || hand != 0 || progress {
                        r.set_residual_count_gates_by_stack(&ctx, Some(&gates)).unwrap();
                    }
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
                    let fast_output = fast.forward_workspace.download_output(&ctx).unwrap();
                    let reference_output = reference.forward_workspace.download_output(&ctx).unwrap();
                    for (&x, &y) in fast_output.iter().zip(&reference_output) {
                        assert!((x - y).abs() < 1e-6 + y.abs() * 1e-5,
                            "forward stacks={stacks} qat={qat} shared={shared} micro={micro}: {x} vs {y}");
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
                        if king != 0 || hand != 0 || progress {
                            assert!(y.iter().any(|v| v.abs() > 1e-8), "vacuous axis gradient");
                        }
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
#[ignore = "requires CUDA; compares compact axis reduction to the original general reducer"]
fn many_bucket_l1_gather_matches_reference_and_accumulates() {
    unsafe extern "C" {
        fn bulletou_bn_reference_mode(enabled: i32);
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            unsafe {
                bulletou_bn_reference_mode(0);
                std::env::remove_var("BULLETOU_DISABLE_FP32_L1_FOLD");
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
    for (stacks, king, hand, progress, pairs, hidden) in
        [(648, 9, 0, true, false, 15), (81, 9, 0, false, false, 15), (1024, 0, 32, false, false, 15)]
    {
        let shape = SfnnForwardShape {
            input_size: 4,
            ft_size: 64,
            l1_hidden: hidden,
            l1_skip: true,
            l2_size: 4,
            num_stacks: stacks,
            factorizer_king_axis_dim: king,
            factorizer_hand_axis_dim: hand,
            factorizer_progress_axis: progress,
            factorizer_king_hand_pair: pairs,
            factorizer_king_progress_pair: king != 0 && progress,
            ..tiny_sfnn_shape()
        };
        let values = |n, scale| (0..n).map(|i| ((i * 17 % 101) as f32 - 50.0) * scale).collect::<Vec<_>>();
        let w0 = values(4 * 64, 0.002);
        let b0 = vec![0.5; 64];
        let outputs = hidden + 1;
        let w1 = values(stacks * 64 * outputs, 0.0002);
        let b1 = vec![0.25; stacks * outputs];
        let fw = values(64 * outputs, 0.0001);
        let fb = vec![0.03; outputs];
        let aw = values(shape.factorizer_axis_count() * 64 * outputs, 0.0003);
        let ab = values(shape.factorizer_axis_count() * outputs, 0.0002);
        let w2 = values(stacks * hidden * 2 * 4, 0.01);
        let b2 = vec![0.3; stacks * 4];
        let w3 = values(stacks * 4, 0.1);
        let b3 = vec![0.0; stacks];
        let f2w = values(hidden * 2 * 4, 0.0001);
        let f2b = vec![0.01; 4];
        let a2w = values(shape.factorizer_axis_count() * hidden * 2 * 4, 0.0001);
        let a2b = values(shape.factorizer_axis_count() * 4, 0.0001);
        let f3w = values(4, 0.001);
        let f3b = vec![0.01];
        let a3w = values(shape.factorizer_axis_count() * 4, 0.0001);
        let a3b = values(shape.factorizer_axis_count(), 0.0001);
        let host = SfnnForwardHostWeights {
            shape,
            l0w: &w0,
            l0b: &b0,
            l1w: &w1,
            l1b: &b1,
            l1fw: Some(&fw),
            l1fb: Some(&fb),
            l1axw: (king != 0 || hand != 0 || progress).then_some(&aw),
            l1axb: (king != 0 || hand != 0 || progress).then_some(&ab),
            l2w: &w2,
            l2b: &b2,
            l3w: &w3,
            l3b: &b3,
            l2fw: Some(&f2w),
            l2fb: Some(&f2b),
            l2axw: Some(&a2w),
            l2axb: Some(&a2b),
            l3fw: Some(&f3w),
            l3fb: Some(&f3b),
            l3axw: Some(&a3w),
            l3axb: Some(&a3b),
            ..tiny_sfnn_weights(tiny_sfnn_shape())
        };
        let buckets: Vec<i32> = (0..N).map(|i| if i % 2 == 0 { 42 } else { (i % (stacks - 1)) as i32 }).collect();
        let batch = SfnnTrainStepHostBatch {
            stm_indices: &indices,
            nstm_indices: &indices,
            buckets: &buckets,
            targets: &targets,
            entry_weights: &entries,
            batch_size: N,
            max_active: 1,
        };
        for qat in [true, false] {
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
                    if king != 0 || hand != 0 || progress {
                        r.set_factorizer_axis_confidences(&ctx, Some(&confidence)).unwrap();
                    }
                    let gates: Vec<f32> = (0..stacks).map(|i| if i == 0 { 0.0 } else { 0.6 }).collect();
                    if qat && (king != 0 || hand != 0 || progress) {
                        r.set_residual_count_gates_by_stack(&ctx, Some(&gates)).unwrap();
                    }
                }
                let policy = SfnnLayerLrMultipliers { qat_l1: qat, ..Default::default() };
                for micro in 1..=1 {
                    for (r, old) in [(&mut fast, false), (&mut reference, true)] {
                        unsafe {
                            bulletou_bn_reference_mode(i32::from(old));
                            if !qat && old { std::env::set_var("BULLETOU_DISABLE_FP32_L1_FOLD", "1"); }
                            else { std::env::remove_var("BULLETOU_DISABLE_FP32_L1_FOLD"); }
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
                    let fast_output = fast.forward_workspace.download_output(&ctx).unwrap();
                    let reference_output = reference.forward_workspace.download_output(&ctx).unwrap();
                    if stacks == 648 && !qat {
                        assert!(!fast.forward_workspace.qat_l1.as_ref().expect("FP32 folded scratch").quantized);
                        assert!(reference.forward_workspace.qat_l1.is_none());
                    }
                    for (&x, &y) in fast_output.iter().zip(&reference_output) {
                        assert!((x - y).abs() < 1e-6 + y.abs() * 1e-5,
                            "forward stacks={stacks} qat={qat} shared={shared} micro={micro}: {x} vs {y}");
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
                        if king != 0 || hand != 0 || progress {
                            assert!(y.iter().any(|v| v.abs() > 1e-8), "vacuous axis gradient");
                        }
                        for (&x, &y) in x.iter().zip(&y) {
                            assert!((x - y).abs() < 2e-6 + y.abs() * 3e-4, "axis {x} vs {y}");
                        }
                    }
                    // The same microbatch twice without an optimizer update must
                    // double each gradient, including shared and pair axes.
                    unsafe { bulletou_bn_reference_mode(0); std::env::remove_var("BULLETOU_DISABLE_FP32_L1_FOLD"); }
                    fast.step_pipelined_no_readback_with_loss_finalize_update_and_lr_multipliers(
                        &ctx, &upload, RangerUpdateParams::default(),
                        ScalarLossKind::SigmoidPow { pow_exp: 2.0 }, 1.0,
                        batch, true, false, policy,
                    ).unwrap();
                    let doubled = fast.backward_workspace.download(&ctx).unwrap();
                    for (name, first, next) in [
                        ("FT", &a.l0w_gradients, &doubled.l0w_gradients),
                        ("L1", &a.l1w_gradients, &doubled.l1w_gradients),
                        ("bias", &a.l1b_gradients, &doubled.l1b_gradients),
                        ("shared", &a.l1fw_gradients, &doubled.l1fw_gradients),
                        ("shared bias", &a.l1fb_gradients, &doubled.l1fb_gradients),
                        ("L2 shared weight", &a.l2fw_gradients, &doubled.l2fw_gradients),
                        ("L2 shared bias", &a.l2fb_gradients, &doubled.l2fb_gradients),
                        ("L2 axis weight", &a.l2axw_gradients, &doubled.l2axw_gradients),
                        ("L2 axis bias", &a.l2axb_gradients, &doubled.l2axb_gradients),
                        ("L3 shared weight", &a.l3fw_gradients, &doubled.l3fw_gradients),
                        ("L3 shared bias", &a.l3fb_gradients, &doubled.l3fb_gradients),
                        ("L3 axis weight", &a.l3axw_gradients, &doubled.l3axw_gradients),
                        ("L3 axis bias", &a.l3axb_gradients, &doubled.l3axb_gradients),
                    ] {
                        for (&x, &y) in first.iter().zip(next) {
                            assert!((2.0*x-y).abs() < 2e-6 + x.abs()*6e-4, "{name}: {x} vs {y}");
                        }
                    }
                    for (x, y) in [
                        (&reference.backward_workspace.l1axw_gradients, &fast.backward_workspace.l1axw_gradients),
                        (&reference.backward_workspace.l1axb_gradients, &fast.backward_workspace.l1axb_gradients),
                    ] {
                        for (&first, &next) in x.download(&ctx).unwrap().iter().zip(y.download(&ctx).unwrap().iter()) {
                            assert!((2.0*first-next).abs() < 2e-6 + first.abs()*6e-4, "axis: {first} vs {next}");
                        }
                    }
                    if stacks == 648 && !qat {
                        unsafe { bulletou_bn_reference_mode(1); std::env::set_var("BULLETOU_DISABLE_FP32_L1_FOLD", "1"); }
                        reference.step_pipelined_no_readback_with_loss_finalize_update_and_lr_multipliers(
                            &ctx, &upload, RangerUpdateParams::default(), ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                            1.0, batch, true, false, policy,
                        ).unwrap();
                        let mut params=RangerUpdateParams::default();
                        params.radam.step=1; params.radam.learning_rate=0.0001;
                        fast.update_weights_with_lr_multipliers_and_dirty_buckets(&ctx,params,policy,None).unwrap();
                        reference.update_weights_with_lr_multipliers_and_dirty_buckets(&ctx,params,policy,None).unwrap();
                        assert!(fast.forward_workspace.qat_l1.as_ref().unwrap().refresh.get());
                        for (name,x,y) in [
                            ("L1",&fast.weights.l1w,&reference.weights.l1w),
                            ("L1 bias",&fast.weights.l1b,&reference.weights.l1b),
                            ("axis",fast.weights.l1axw.as_ref().unwrap(),reference.weights.l1axw.as_ref().unwrap()),
                        ] {
                            for (a,b) in x.download(&ctx).unwrap().iter().zip(y.download(&ctx).unwrap()) {
                                assert!((a-b).abs()<2e-6+2e-4*b.abs(),"updated {name}: {a} vs {b}");
                            }
                        }
                        unsafe { bulletou_bn_reference_mode(0); std::env::remove_var("BULLETOU_DISABLE_FP32_L1_FOLD"); }
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
        // same independent CPU reference checks both ordinary and L1-only QAT.
        // Full dense QAT has integer activations and is tested separately.
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
                let policy = SfnnLayerLrMultipliers { qat_l1: qat, ..Default::default() };
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
