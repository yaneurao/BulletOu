//! Non-BN layer QAT. When L1/L2/L3 are all selected, dense forward uses
//! integer arithmetic with tatara-style continuous activation STEs.
use super::*;

#[derive(Debug, Default)]
pub(crate) struct State {
    ft: Option<(F32Buffer, F32Buffer)>,
    l2: Option<(F32Buffer, F32Buffer)>,
    l3: Option<(F32Buffer, F32Buffer)>,
    signature: Option<(bool, bool, bool, usize, u32)>,
}

pub(crate) fn enabled(lr: SfnnLayerLrMultipliers) -> bool {
    lr.qat_ft || lr.qat_l2 || lr.qat_l3
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn integer_dense_expected(ctx:&Context, fw:&SfnnForwardWorkspace, proxy:&SfnnForwardDeviceWeights,
        shape:SfnnForwardShape, buckets:&[i32]) -> Vec<f32> {
        let combined=fw.combined.download(ctx).unwrap();
        let l1w=proxy.l1w.download(ctx).unwrap(); let l1b=proxy.l1b.download(ctx).unwrap();
        let l2w=proxy.l2w.download(ctx).unwrap(); let l2b=proxy.l2b.download(ctx).unwrap();
        let l3w=proxy.l3w.download(ctx).unwrap(); let l3b=proxy.l3b.download(ctx).unwrap();
        let l1out=shape.l1_out(); let l2in=shape.l2_in();
        let qbias=|v:f32| (f64::from(v)*8128.0).round() as i32;
        let qweight=|v:f32| (v*64.0).round() as i32;
        let mut out=Vec::new();
        for (sample,&bucket) in buckets.iter().enumerate() {
            let stack=bucket as usize;
            let x:Vec<i32>=combined[sample*shape.ft_size..(sample+1)*shape.ft_size].iter()
                .map(|&v| (f64::from(v)*127.0).clamp(0.0,127.0).floor() as i32).collect();
            let mut r1=vec![0i32;l1out];
            for u in 0..l1out {
                let row=stack*l1out+u;
                let sum=x.iter().enumerate().fold(i64::from(qbias(l1b[row])),|s,(j,&v)|
                    s+i64::from(v)*i64::from(qweight(l1w[row*shape.ft_size+j])));
                r1[u]=sum as i32;
            }
            let mut x2=vec![0i32;l2in];
            for u in 0..shape.l1_hidden {
                x2[u]=((i64::from(r1[u]).pow(2)>>19).min(127)) as i32;
                x2[shape.l1_hidden+u]=(r1[u]>>6).clamp(0,127);
            }
            let mut x3=vec![0i32;shape.l2_size];
            for u in 0..shape.l2_size {
                let row=stack*shape.l2_size+u;
                let sum=x2.iter().enumerate().fold(i64::from(qbias(l2b[row])),|s,(j,&v)|
                    s+i64::from(v)*i64::from(qweight(l2w[row*l2in+j])));
                x3[u]=((sum as i32)>>6).clamp(0,127);
            }
            let sum=x3.iter().enumerate().fold(i64::from(qbias(l3b[stack])),|s,(j,&v)|
                s+i64::from(v)*i64::from(qweight(l3w[stack*shape.l2_size+j])));
            let raw=(sum+if shape.l1_skip {i64::from(r1[shape.l1_hidden])} else {0}) as i32;
            out.push((f64::from(raw)/8128.0) as f32);
        }
        out
    }
    #[test]
    #[ignore = "requires CUDA; integer dense QAT with pair and compact L1"]
    fn dense_qat_pair_and_compact_match_integer_reference() {
        let ctx=Context::new(0).unwrap();
        for compact in [false,true] {
            let shape=SfnnForwardShape { input_size:4,ft_size:if compact {192}else{8},l1_hidden:3,l1_skip:true,
                l2_size:4,num_stacks:648,l1_group_count:if compact {2}else{1},
                l1_common_size:if compact {64}else{0},l1_shard_size:if compact {64}else{0},
                factorizer_king_axis_dim:9,factorizer_hand_axis_dim:0,factorizer_progress_axis:true,
                factorizer_king_hand_pair:false,factorizer_king_progress_pair:true,factorizer_hand_progress_pair:false };
            let n=shape.l1_out(); let ax=shape.factorizer_axis_count();
            let ftw=vec![0.03125;shape.input_size*shape.ft_size]; let ftb=vec![0.25;shape.ft_size];
            let l1w=vec![0.021;shape.l1w_len().unwrap()]; let l1b=vec![0.031;shape.num_stacks*n];
            let l1fw=vec![0.003;shape.ft_size*n]; let l1fb=vec![0.002;n];
            let l1aw=vec![0.001;ax*shape.ft_size*n]; let l1ab=vec![0.001;ax*n];
            let l2w=vec![0.13;shape.num_stacks*shape.l2_size*shape.l2_in()]; let l2b=vec![0.04;shape.num_stacks*shape.l2_size];
            let l3w=vec![0.12;shape.num_stacks*shape.l2_size]; let l3b=vec![0.03;shape.num_stacks];
            let host=SfnnForwardHostWeights {shape,l0w:&ftw,l0b:&ftb,l1w:&l1w,l1b:&l1b,
                l1fw:Some(&l1fw),l1fb:Some(&l1fb),l1axw:Some(&l1aw),l1axb:Some(&l1ab),
                l2w:&l2w,l2b:&l2b,l2fw:None,l2fb:None,l2axw:None,l2axb:None,
                l3w:&l3w,l3b:&l3b,l3fw:None,l3fb:None,l3axw:None,l3axb:None};
            let mut r=SfnnTrainStepRunner::new(&ctx,host,4,1).unwrap();
            r.set_factorizer_axis_confidences(&ctx,Some(&vec![0.75;ax])).unwrap();
            let ps=SfnnForwardShape {l1_group_count:1,l1_common_size:0,l1_shard_size:0,
                factorizer_king_axis_dim:0,factorizer_hand_axis_dim:0,factorizer_progress_axis:false,
                factorizer_king_hand_pair:false,factorizer_king_progress_pair:false,factorizer_hand_progress_pair:false,..shape};
            let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,ps).unwrap();
            r.build_quantized_proxy(&ctx,shape.input_size,0,&proxy).unwrap();
            let hb=SfnnForwardHostBatch {stm_indices:&[0,1,2,3],nstm_indices:&[3,2,1,0],
                buckets:&[0,1,646,647],batch_size:4,max_active:1};
            let db=SfnnForwardDeviceBatch::from_host(&ctx,hb).unwrap();
            let fw=SfnnForwardWorkspace::new(&ctx,SfnnForwardWorkspaceLayout::new(ps,4)).unwrap();
            sfnn_forward_device(&ctx,&db,&proxy,&fw).unwrap();
            let expected=integer_dense_expected(&ctx,&fw,&proxy,ps,hb.buckets);
            let batch=SfnnTrainStepHostBatch {stm_indices:hb.stm_indices,nstm_indices:hb.nstm_indices,buckets:hb.buckets,
                targets:&[0.1,0.3,0.7,0.9],entry_weights:&[1.0;4],batch_size:4,max_active:1};
            let lr=SfnnLayerLrMultipliers {qat_ft:true,qat_l1:true,qat_l2:true,qat_l3:true,..Default::default()};
            r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,RangerUpdateParams::default(),
                ScalarLossKind::BceWithLogits,1.0,batch,true,false,lr).unwrap();
            let actual=r.forward_workspace.download_output(&ctx).unwrap();
            for (a,e) in actual.iter().zip(expected){assert!((a-e).abs()<1e-6,"compact={compact}: {a} vs {e}");}
            assert_eq!(r.weights.l1w.download(&ctx).unwrap(),l1w);
            let first=r.backward_workspace.l1w_gradients.download(&ctx).unwrap();
            assert!(first.iter().all(|v|v.is_finite()));
            assert!(first.iter().any(|v|v.abs()>1e-8));
            r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,RangerUpdateParams::default(),
                ScalarLossKind::BceWithLogits,1.0,batch,true,false,lr).unwrap();
            let second=r.backward_workspace.l1w_gradients.download(&ctx).unwrap();
            for (a,b) in first.iter().zip(second) {
                assert!((2.0*a-b).abs()<1e-6+1e-4*a.abs(),"compact={compact}: {a} vs {b}");
            }
        }
    }
    #[test]
    #[ignore = "requires CUDA; verifies both STEs when integer activations round to zero"]
    fn dense_qat_zero_lattice_values_keep_positive_ste_gradients() {
        let ctx=Context::new(0).unwrap();
        let mut w=crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        let shape=w.shape; let n=shape.l1_out(); let l2in=shape.l2_in();
        let l1w=vec![0.0;shape.num_stacks*n*shape.ft_size];
        let mut l1b=vec![0.0;shape.num_stacks*n]; l1b[0]=1.0/8128.0;
        let mut l2w=vec![0.0;shape.num_stacks*shape.l2_size*l2in]; l2w[shape.l1_hidden]=1.0;
        let mut l2b=vec![0.0;shape.num_stacks*shape.l2_size]; l2b[0]=1.0/8128.0;
        let mut l3w=vec![0.0;shape.num_stacks*shape.l2_size]; l3w[0]=1.0;
        let l3b=vec![0.0;shape.num_stacks];
        w.l1w=&l1w; w.l1b=&l1b; w.l1fw=None; w.l1fb=None;
        w.l2w=&l2w; w.l2b=&l2b; w.l2fw=None; w.l2fb=None;
        w.l3w=&l3w; w.l3b=&l3b; w.l3fw=None; w.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,w,1,1).unwrap();
        let batch=SfnnTrainStepHostBatch {stm_indices:&[0],nstm_indices:&[1],buckets:&[0],
            targets:&[1.0],entry_weights:&[1.0],batch_size:1,max_active:1};
        let lr=SfnnLayerLrMultipliers {qat_l1:true,qat_l2:true,qat_l3:true,..Default::default()};
        r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,RangerUpdateParams::default(),
            ScalarLossKind::BceWithLogits,1.0,batch,true,false,lr).unwrap();
        assert_eq!(r.forward_workspace.l2.download(&ctx).unwrap()[0],0.0);
        assert!(r.backward_workspace.l2b_gradients.download(&ctx).unwrap()[0].abs()>0.1);
        assert!(r.backward_workspace.l1b_gradients.download(&ctx).unwrap()[0].abs()>0.1);
    }
    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn rejects_only_all_zero_quantized_ft() {
        let ctx = Context::new(0).unwrap();
        let w = F32Buffer::from_host(&ctx, &[0.002, -0.002]).unwrap();
        let b = F32Buffer::from_host(&ctx, &[0.0]).unwrap();
        let mut pair = None;
        prepare(&ctx, &mut pair, &w, &b, true, true, 0, 0, 1, 1.0).unwrap();
        let (qw, qb) = pair.as_ref().unwrap();
        assert!(reject_zero_ft(&ctx, qw, qb).unwrap_err().to_string().contains("rounded every"));
        let nonzero = F32Buffer::from_host(&ctx, &[1.0 / 127.0]).unwrap();
        assert!(reject_zero_ft(&ctx, qw, &nonzero).is_ok());
        assert!(reject_zero_ft(&ctx, &nonzero, qb).is_ok());
        assert_eq!(w.download(&ctx).unwrap(), vec![0.002, -0.002]);
    }
    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn all_layer_qat_matches_quantized_proxy_with_ft_factorizer() {
        const BASE: usize = 131949;
        const VR: usize = 1629;
        let ctx = Context::new(0).unwrap();
        let mut w = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        w.l2fw = None;
        w.l2fb = None;
        w.l3fw = None;
        w.l3fb = None;
        w.shape.input_size = BASE + VR;
        let ft: Vec<f32> = (0..w.shape.input_size * w.shape.ft_size).map(|i| 0.003 * ((i % 11) as f32 - 4.0)).collect();
        w.l0w = &ft;
        let mut r = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
        r.factorizer_alpha.ft = 0.7;
        let shape = SfnnForwardShape { input_size: BASE, ..w.shape };
        let proxy = SfnnForwardDeviceWeights::new_dense(&ctx, shape).unwrap();
        r.build_quantized_proxy(&ctx, BASE, VR, &proxy).unwrap();
        let hb = SfnnForwardHostBatch {
            stm_indices: &[0, 1629, 1, 1630],
            nstm_indices: &[2, 1631, 3, 1632],
            buckets: &[0, 1, 0, 1],
            batch_size: 4,
            max_active: 1,
        };
        let batch = SfnnForwardDeviceBatch::from_host(&ctx, hb).unwrap();
        let fw = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(shape, 4)).unwrap();
        sfnn_forward_device(&ctx, &batch, &proxy, &fw).unwrap();
        let expected = integer_dense_expected(&ctx, &fw, &proxy, shape, hb.buckets);
        let lr = SfnnLayerLrMultipliers {
            qat_ft: true,
            qat_l1: true,
            qat_l2: true,
            qat_l3: true,
            qat_ft_virtual_rows: VR,
            ..Default::default()
        };
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,
            RangerUpdateParams::default(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            SfnnTrainStepHostBatch {
                stm_indices: hb.stm_indices,
                nstm_indices: hb.nstm_indices,
                buckets: hb.buckets,
                targets: &[0.2, 0.8, 0.3, 0.7],
                entry_weights: &[1.0; 4],
                batch_size: 4,
                max_active: 1,
            },
            true,
            false,
            lr,
            None,
        )
        .unwrap();
        let actual = r.forward_workspace.download_output(&ctx).unwrap();
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{a} vs {e}");
        }
        let g = r.backward_workspace.l0w_gradients.download(&ctx).unwrap();
        assert!(g.iter().any(|v| v.abs() > 1e-10));
        for piece in 0..4 {
            for u in 0..w.shape.ft_size {
                let sum: f32 = (0..81).map(|k| g[(k * VR + piece) * w.shape.ft_size + u]).sum();
                assert!((g[(BASE + piece) * w.shape.ft_size + u] - 0.7 * sum).abs() < 1e-6);
            }
        }
    }
    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn halfkahm2_all_layer_qat_matches_quantized_proxy_with_ft_factorizer() {
        const BASE: usize = 73305;
        const VR: usize = 1629;
        let ctx = Context::new(0).unwrap();
        let mut w = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        w.l2fw = None;
        w.l2fb = None;
        w.l3fw = None;
        w.l3fb = None;
        w.shape.input_size = BASE + VR;
        let ft: Vec<f32> = (0..w.shape.input_size * w.shape.ft_size).map(|i| 0.003 * ((i % 11) as f32 - 4.0)).collect();
        w.l0w = &ft;
        let mut r = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
        r.factorizer_alpha.ft = 0.7;
        let shape = SfnnForwardShape { input_size: BASE, ..w.shape };
        let proxy = SfnnForwardDeviceWeights::new_dense(&ctx, shape).unwrap();
        r.build_quantized_proxy(&ctx, BASE, VR, &proxy).unwrap();
        let hb = SfnnForwardHostBatch {
            stm_indices: &[0, 1629, 1, 1630],
            nstm_indices: &[2, 1631, 3, 1632],
            buckets: &[0, 1, 0, 1],
            batch_size: 4,
            max_active: 1,
        };
        let batch = SfnnForwardDeviceBatch::from_host(&ctx, hb).unwrap();
        let fw = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(shape, 4)).unwrap();
        sfnn_forward_device(&ctx, &batch, &proxy, &fw).unwrap();
        let expected = integer_dense_expected(&ctx, &fw, &proxy, shape, hb.buckets);
        let lr = SfnnLayerLrMultipliers {
            qat_ft: true,
            qat_l1: true,
            qat_l2: true,
            qat_l3: true,
            qat_ft_virtual_rows: VR,
            ..Default::default()
        };
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,
            RangerUpdateParams::default(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            SfnnTrainStepHostBatch {
                stm_indices: hb.stm_indices,
                nstm_indices: hb.nstm_indices,
                buckets: hb.buckets,
                targets: &[0.2, 0.8, 0.3, 0.7],
                entry_weights: &[1.0; 4],
                batch_size: 4,
                max_active: 1,
            },
            true,
            false,
            lr,
            None,
        )
        .unwrap();
        let actual = r.forward_workspace.download_output(&ctx).unwrap();
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{a} vs {e}");
        }
        let g = r.backward_workspace.l0w_gradients.download(&ctx).unwrap();
        assert!(g.iter().any(|v| v.abs() > 1e-10));
        for piece in 0..4 {
            for u in 0..w.shape.ft_size {
                let sum: f32 = (0..45).map(|k| g[(k * VR + piece) * w.shape.ft_size + u]).sum();
                assert!((g[(BASE + piece) * w.shape.ft_size + u] - 0.7 * sum).abs() < 1e-6);
            }
        }
    }
    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn qat_rounding_matches_f64_export_at_half_boundaries() {
        let ctx = Context::new(0).unwrap();
        let w = F32Buffer::from_host(&ctx, &[-0.0669291317_f32]).unwrap();
        let b = F32Buffer::from_host(&ctx, &[-1.96770418_f32, 1032.0859375_f32, -1032.0859375_f32]).unwrap();
        let mut q = None;
        prepare(&ctx, &mut q, &w, &b, true, true, 0, 0, 1, 1.0).unwrap();
        let actual = q.as_ref().unwrap().0.download(&ctx).unwrap()[0];
        let expected = ((f64::from(-0.0669291317_f32) * 127.0).round() as f32) / 127.0;
        assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
        let mut q = None;
        prepare(&ctx, &mut q, &w, &b, true, false, 0, 0, 1, 1.0).unwrap();
        let actual = q.as_ref().unwrap().1.download(&ctx).unwrap()[0];
        let expected = ((f64::from(-1.96770418_f32) * 8128.0).round() as f32) / 8128.0;
        assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
        for (actual, source) in q.as_ref().unwrap().1.download(&ctx).unwrap().into_iter()
            .zip([-1.96770418_f32, 1032.0859375_f32, -1032.0859375_f32]) {
            let expected = ((f64::from(source) * 8128.0).round() as f32) / 8128.0;
            assert_eq!(actual, expected);
        }
    }
    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn selected_qat_rounding_and_ft_fold() {
        let ctx = Context::new(0).unwrap();
        let w = F32Buffer::from_host(&ctx, &[0.101, -0.205, 1.25, -1.5, 0.033, -0.071]).unwrap();
        let b = F32Buffer::from_host(&ctx, &[0.0123, -0.1234]).unwrap();
        let mut q = None;
        prepare(&ctx, &mut q, &w, &b, true, true, 2, 1, 2, 0.5).unwrap();
        let (qw, qb) = q.as_ref().unwrap();
        let actual = qw.download(&ctx).unwrap();
        let source = w.download(&ctx).unwrap();
        for j in 0..4 {
            let v = source[j] + 0.5 * source[4 + j % 2];
            assert!((actual[j] - (v * 127.0).round() / 127.0).abs() < 1e-6);
        }
        assert_eq!(&actual[4..], &[0.0, 0.0]);
        assert!((qb.download(&ctx).unwrap()[0] - (0.0123f32 * 127.0).round() / 127.0).abs() < 1e-6);
        let w = F32Buffer::from_host(&ctx, &[-3.0, -0.101, 0.205, 3.0]).unwrap();
        q = None;
        prepare(&ctx, &mut q, &w, &b, true, false, 0, 0, 1, 1.0).unwrap();
        let actual = q.as_ref().unwrap().0.download(&ctx).unwrap();
        assert_eq!(actual, vec![-2.0, -6.0 / 64.0, 13.0 / 64.0, 127.0 / 64.0]);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn selected_qat_steps_restore_masters_and_update_with_bpu() {
        let ctx = Context::new(0).unwrap();
        let upload_ctx = Context::new(0).unwrap();
        for bits in 1..8 {
            let shape = crate::tests::tiny_sfnn_shape();
            let mut w = crate::tests::tiny_sfnn_weights(shape);
            w.l2fw = None;
            w.l2fb = None;
            w.l3fw = None;
            w.l3fb = None;
            let mut r = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
            let lr = SfnnLayerLrMultipliers {
                qat_ft: bits & 1 != 0,
                qat_l2: bits & 2 != 0,
                qat_l3: bits & 4 != 0,
                qat_l1: true,
                ..Default::default()
            };
            let mut params = RangerUpdateParams::default();
            params.radam.step = 1;
            params.radam.learning_rate = 0.001;
            let batch = SfnnTrainStepHostBatch {
                stm_indices: &[0, 1, 2, 3],
                nstm_indices: &[2, 3, 0, 1],
                buckets: &[0, 1, 0, 1],
                targets: &[0.2, 0.8, 0.3, 0.7],
                entry_weights: &[1.0; 4],
                batch_size: 4,
                max_active: 1,
            };
            r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                &ctx,
                params,
                ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                1.0,
                batch,
                true,
                false,
                lr,
                None,
            )
            .unwrap();
            assert_eq!(r.weights.l0w.download(&ctx).unwrap(), w.l0w);
            assert_eq!(r.weights.l2w.download(&ctx).unwrap(), w.l2w);
            assert_eq!(r.weights.l3w.download(&ctx).unwrap(), w.l3w);
            assert_eq!(r.pending_gradient_batches, 1);
            let batch = SfnnTrainStepHostBatch {
                stm_indices: &[0, 1, 2, 3],
                nstm_indices: &[2, 3, 0, 1],
                buckets: &[0, 1, 0, 1],
                targets: &[0.2, 0.8, 0.3, 0.7],
                entry_weights: &[1.0; 4],
                batch_size: 4,
                max_active: 1,
            };
            r.step_profiled_no_readback_with_update_lr_multipliers_and_dirty_buckets(
                &ctx,
                params,
                ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                1.0,
                batch,
                true,
                lr,
                None,
            )
            .unwrap();
            assert_eq!(r.pending_gradient_batches, 0);
            let updated = r.weights.l3w.download(&ctx).unwrap();
            assert!(updated.iter().all(|v| v.is_finite()));
            assert_ne!(updated, w.l3w);
            let batch = SfnnTrainStepHostBatch {
                stm_indices: &[0, 1, 2, 3],
                nstm_indices: &[2, 3, 0, 1],
                buckets: &[0, 1, 0, 1],
                targets: &[0.2, 0.8, 0.3, 0.7],
                entry_weights: &[1.0; 4],
                batch_size: 4,
                max_active: 1,
            };
            params.radam.step = 2;
            r.step_pipelined_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                &ctx,
                &upload_ctx,
                params,
                ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                1.0,
                batch,
                true,
                true,
                lr,
                None,
            )
            .unwrap();
            assert!(r.read_loss(&ctx).unwrap().mean.is_finite());
            let before = r.weights.l0w.download(&ctx).unwrap();
            let fail: Result<()> =
                step(&mut r, &ctx, params, lr, false, None, |_, _| Err(CudaCppError::message("test failure")));
            assert!(fail.is_err());
            assert_eq!(r.weights.l0w.download(&ctx).unwrap(), before);
        }
    }
}

unsafe extern "C" {
    fn bulletou_dense_qat_begin(ctx: *mut ffi::BulletOuCudaCppContext,
        l2_bias: *mut ffi::BulletOuCudaCppF32Buffer, l3_bias: *mut ffi::BulletOuCudaCppF32Buffer,
        batch: usize, l1_out: usize, l2_size: usize, stacks: usize) -> i32;
    fn bulletou_dense_qat_end(ctx: *mut ffi::BulletOuCudaCppContext);
    fn bulletou_layer_qat_nonzero(ctx: *mut ffi::BulletOuCudaCppContext,
        w: *mut ffi::BulletOuCudaCppF32Buffer, b: *mut ffi::BulletOuCudaCppF32Buffer,
        flag: *mut ffi::BulletOuCudaCppF32Buffer) -> i32;
    fn bulletou_layer_qat(
        ctx: *mut ffi::BulletOuCudaCppContext,
        src: *mut ffi::BulletOuCudaCppF32Buffer,
        dst: *mut ffi::BulletOuCudaCppF32Buffer,
        scale: f32,
        lo: f32,
        hi: f32,
        base: usize,
        virtual_rows: usize,
        width: usize,
        alpha: f32,
    ) -> i32;
}

fn prepare(
    ctx: &Context,
    pair: &mut Option<(F32Buffer, F32Buffer)>,
    w: &F32Buffer,
    b: &F32Buffer,
    on: bool,
    ft: bool,
    base: usize,
    vr: usize,
    width: usize,
    alpha: f32,
) -> Result<()> {
    if !on {
        *pair = None;
        return Ok(());
    }
    if pair.is_none() {
        *pair = Some((F32Buffer::new(ctx, w.len())?, F32Buffer::new(ctx, b.len())?));
    }
    let (qw, qb) = pair.as_ref().unwrap();
    for (src, dst, scale, lo, hi, rows) in [
        (
            w,
            qw,
            if ft { 127.0 } else { 64.0 },
            if ft { -32768.0 } else { -128.0 },
            if ft { 32767.0 } else { 127.0 },
            vr,
        ),
        (
            b,
            qb,
            if ft { 127.0 } else { 8128.0 },
            if ft { -32768.0 } else { -2147483648.0 },
            if ft { 32767.0 } else { 2147483647.0 },
            0,
        ),
    ] {
        check(unsafe {
            bulletou_layer_qat(
                ctx.as_ptr(),
                src.as_ptr(),
                dst.as_ptr(),
                scale,
                lo,
                hi,
                if rows > 0 { base } else { 0 },
                rows,
                width,
                alpha,
            )
        })?;
    }
    Ok(())
}

fn reject_zero_ft(ctx: &Context, w: &F32Buffer, b: &F32Buffer) -> Result<()> {
    let flag = F32Buffer::from_host(ctx, &[0.0])?;
    check(unsafe { bulletou_layer_qat_nonzero(ctx.as_ptr(), w.as_ptr(), b.as_ptr(), flag.as_ptr()) })?;
    if flag.download(ctx)?[0] == 0.0 {
        let message = "FT QAT rounded every effective FT weight and bias to zero (scale=127); pairwise FT gradients vanish. Training stopped. Use --sfnn-init-ft-glorot for scratch initialization, or disable --sfnn-qat-ft until a trained checkpoint is available. No options were automatically changed.";
        use std::io::IsTerminal;
        let color = std::env::var("BULLETOU_COLOR").unwrap_or_default();
        if std::env::var_os("NO_COLOR").is_none() && !color.eq_ignore_ascii_case("never")
            && (color.eq_ignore_ascii_case("always") || std::io::stderr().is_terminal()) {
            eprintln!("\x1b[1;33mWARNING: {message}\x1b[0m");
        } else {
            eprintln!("WARNING: {message}");
        }
        return Err(CudaCppError::message(message));
    }
    Ok(())
}

impl State {
    pub(crate) fn invalidate(&mut self) {
        // Keep large proxy allocations across worker restores.
        self.signature = None;
    }
    fn swap(&mut self, w: &mut SfnnForwardDeviceWeights) {
        for (pair, ww, bb) in [
            (&mut self.ft, &mut w.l0w, &mut w.l0b),
            (&mut self.l2, &mut w.l2w, &mut w.l2b),
            (&mut self.l3, &mut w.l3w, &mut w.l3b),
        ] {
            if let Some((qw, qb)) = pair {
                std::mem::swap(qw, ww);
                std::mem::swap(qb, bb);
            }
        }
    }
}

pub(crate) fn step<T>(
    r: &mut SfnnTrainStepRunner,
    ctx: &Context,
    params: RangerUpdateParams,
    lr: SfnnLayerLrMultipliers,
    update: bool,
    dirty: Option<&[i32]>,
    run: impl FnOnce(&mut SfnnTrainStepRunner, SfnnLayerLrMultipliers) -> Result<T>,
) -> Result<T> {
    if r.batch_norm.is_some() {
        return Err(CudaCppError::message("Layer QAT requires no BN; use BN QAT instead"));
    }
    if (lr.qat_l2 && (r.weights.l2fw.is_some() || r.weights.l2axw.is_some()))
        || (lr.qat_l3 && (r.weights.l3fw.is_some() || r.weights.l3axw.is_some()))
    {
        return Err(CudaCppError::message(
            "Layer QAT does not support legacy L2/L3 factorizer tensors; fold them first",
        ));
    }
    let vr = lr.qat_ft_virtual_rows;
    let base = r.shape.input_size.checked_sub(vr).ok_or_else(|| CudaCppError::message("invalid QAT FT layout"))?;
    if vr > 0 && (base == 0 || base % vr != 0) {
        return Err(CudaCppError::message("invalid QAT FT factorizer layout"));
    }
    let mut q = std::mem::take(&mut r.layer_qat);
    let result = (|| {
        let signature = (lr.qat_ft, lr.qat_l2, lr.qat_l3, vr, r.factorizer_alpha.ft.to_bits());
        // Masters are unchanged within a gradient-accumulation group. Never
        // reuse across an update or restore (both reset pending batches).
        let refresh = r.pending_gradient_batches == 0 || q.signature != Some(signature);
        if refresh {
            prepare(
                ctx,
                &mut q.ft,
                &r.weights.l0w,
                &r.weights.l0b,
                lr.qat_ft,
                true,
                base,
                vr,
                r.shape.ft_size,
                r.factorizer_alpha.ft,
            )?;
            // One scalar readback only on activation/layout change/restore,
            // never on every batch. Check actual folded, quantized tensors.
            if q.signature != Some(signature) {
                if let Some((w, b)) = &q.ft { reject_zero_ft(ctx, w, b)?; }
            }
            prepare(ctx, &mut q.l2, &r.weights.l2w, &r.weights.l2b, lr.qat_l2, false, 0, 0, 1, 1.0)?;
            prepare(ctx, &mut q.l3, &r.weights.l3w, &r.weights.l3b, lr.qat_l3, false, 0, 0, 1, 1.0)?;
            q.signature = Some(signature);
        }
        let dense = lr.qat_l1 && lr.qat_l2 && lr.qat_l3;
        if dense {
            if std::env::var_os("BULLETOU_EXPERIMENT_RECYCLE_STEPS").is_some() {
                return Err(CudaCppError::message("dense QAT does not support the experimental recycle path"));
            }
            // SAFETY: master bias buffers live through the nested training step.
            check(unsafe { bulletou_dense_qat_begin(ctx.as_ptr(), r.weights.l2b.as_ptr(), r.weights.l3b.as_ptr(),
                r.batch_size, r.shape.l1_out(), r.shape.l2_size, r.shape.num_stacks) })?;
        }
        q.swap(&mut r.weights);
        let inner = SfnnLayerLrMultipliers { qat_ft: false, qat_l2: false, qat_l3: false, ..lr };
        let value = run(r, inner);
        if dense {
            // SAFETY: this only clears context-local mode and borrowed pointers.
            unsafe { bulletou_dense_qat_end(ctx.as_ptr()) };
        }
        // Restore masters even after a failed forward/backward; optimizers and
        // serialization never see fake-quantized tensors.
        q.swap(&mut r.weights);
        let value = value?;
        if update {
            r.update_weights_with_lr_multipliers_and_dirty_buckets(ctx, params, lr, dirty)?;
        }
        Ok(value)
    })();
    r.layer_qat = q;
    result
}
