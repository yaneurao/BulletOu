//! BN QAT: train-mode BN with running-scale weight fake quantization, or an
//! explicitly selected frozen-running-stat inference-fold fine-tuning mode.
use super::*;

#[derive(Debug)]
pub struct State {
    proxy: SfnnForwardDeviceWeights,
    base: usize,
    virtual_rows: usize,
    freeze_stats: bool,
    l2_effective_weight_clip: bool,
}

impl SfnnTrainStepRunner {
    pub fn configure_bn_qat(&mut self, ctx: &Context, enabled: bool, base: usize, virtual_rows: usize) -> Result<()> {
        self.configure_bn_qat_mode(ctx, enabled, base, virtual_rows, false)
    }
    pub fn configure_bn_qat_mode(&mut self, ctx: &Context, enabled: bool, base: usize, virtual_rows: usize, freeze_stats: bool) -> Result<()> {
        if self.pending_gradient_batches != 0 {
            return Err(CudaCppError::message("cannot switch BN QAT with pending accumulated gradients"));
        }
        if !enabled {
            self.bn_qat = None;
            return Ok(());
        }
        let bn = self.batch_norm.as_ref().ok_or_else(|| CudaCppError::message("BN QAT requires BN"))?;
        if freeze_stats { for (l, _) in bn.layers.iter().flatten() {
            let s = l.read_state(ctx)?;
            let c = s.width * s.groups;
            if !s.running[2 * c..].iter().any(|x| *x == 1.0) {
                return Err(CudaCppError::message("frozen BN QAT requires calibrated statistics; disable --sfnn-bn-qat-freeze-stats to train from scratch"));
            }
        } }
        if base == 0
            || !(self.shape.input_size == base
                || (virtual_rows > 0 && base % virtual_rows == 0 && self.shape.input_size == base + virtual_rows))
        {
            return Err(CudaCppError::message("invalid BN QAT FT factorizer layout"));
        }
        let mut proxy = SfnnForwardDeviceWeights::new_dense(ctx, self.shape)?;
        // Preserve optional tensor shapes for runner/optimizer validation. They
        // are inactive during the proxy pass and never optimized.
        for (dst, src) in [
            (&mut proxy.l1fw, &self.weights.l1fw),
            (&mut proxy.l1fb, &self.weights.l1fb),
            (&mut proxy.l2fw, &self.weights.l2fw),
            (&mut proxy.l2fb, &self.weights.l2fb),
            (&mut proxy.l3fw, &self.weights.l3fw),
            (&mut proxy.l3fb, &self.weights.l3fb),
        ] {
            if let Some(src) = src {
                let b = F32Buffer::new(ctx, src.len())?;
                b.fill(ctx, 0.0)?;
                *dst = Some(b);
            }
        }
        self.bn_qat =
            Some(State { proxy, base, virtual_rows: if self.shape.input_size == base { 0 } else { virtual_rows }, freeze_stats, l2_effective_weight_clip: false });
        Ok(())
    }

    /// Project L2 master/Lookahead weights in BN-folded coordinates. Moments are retained.
    pub fn configure_bn_l2_effective_weight_clip(&mut self, ctx: &Context, enabled: bool) -> Result<()> {
        if self.pending_gradient_batches != 0 {
            return Err(CudaCppError::message("cannot change BN L2 clipping with accumulated gradients"));
        }
        if enabled && (!self.bn_qat.as_ref().is_some_and(|q| q.freeze_stats)
            || !self.batch_norm.as_ref().is_some_and(|bn| bn.layers[2].is_some())) {
            return Err(CudaCppError::message("BN L2 effective weight clipping requires L2 BN and frozen-stat BN QAT"));
        }
        if enabled && self.factorizer.shared && self.weights.l2fw.is_some() {
            return Err(CudaCppError::message("BN L2 effective weight clipping does not support legacy L2 shared weights"));
        }
        if let Some(q) = self.bn_qat.as_mut() { q.l2_effective_weight_clip = enabled; }
        if enabled { project_l2(self, ctx)?; }
        Ok(())
    }
}

fn project_l2(r: &SfnnTrainStepRunner, ctx: &Context) -> Result<()> {
    let _bind = r.batch_norm.as_ref().unwrap().bind(ctx, 1, false)?;
    check(unsafe { bulletou_bn_qat_project_l2(ctx.as_ptr(), r.weights.l2w.as_ptr(),
        r.optimizer_states.l2w.slow_params.as_ptr(), r.shape.l2_in(), r.shape.l2_size * r.shape.num_stacks) })
}

unsafe extern "C" {
    fn bulletou_bn_qat_project_l2(ctx: *mut ffi::BulletOuCudaCppContext,
        w: *mut ffi::BulletOuCudaCppF32Buffer, slow: *mut ffi::BulletOuCudaCppF32Buffer,
        input: usize, channels: usize) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bn_l2_effective_projection_preserves_other_state() {
        let ctx=Context::new(0).unwrap();
        let shape=crate::tests::tiny_sfnn_shape();
        let mut r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(shape),4,1).unwrap();
        assert!(r.configure_bn_l2_effective_weight_clip(&ctx,true).is_err());
        calibrated(&mut r,&ctx);
        let (layer,_)=r.batch_norm.as_ref().unwrap().layers[2].as_ref().unwrap();
        let c=layer.width*layer.groups;
        let mut params=layer.affine.download(&ctx).unwrap();
        for (i,v) in params[..c].iter_mut().enumerate() {*v=match i%3 {0=>0.0,1=>-2.0,_=>3.0};}
        layer.affine.upload(&ctx,&params).unwrap();
        r.configure_bn_qat_mode(&ctx,true,shape.input_size,0,true).unwrap();
        // Tiny legacy fixtures contain L2 shared weights; production L2 does not.
        assert!(r.configure_bn_l2_effective_weight_clip(&ctx,true).is_err());
        r.factorizer.shared=false;
        r.weights.l2w.fill(&ctx,100.0).unwrap();
        r.optimizer_states.l2w.slow_params.fill(&ctx,-100.0).unwrap();
        r.optimizer_states.l2w.momentum.fill(&ctx,0.3).unwrap();
        let bn=r.read_batch_norm_state(&ctx).unwrap();
        let bias=r.weights.l2b.download(&ctx).unwrap();
        let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,shape).unwrap();
        r.build_quantized_proxy(&ctx,shape.input_size,0,&proxy).unwrap();
        let quantized_before=proxy.l2w.download(&ctx).unwrap();
        r.configure_bn_l2_effective_weight_clip(&ctx,true).unwrap();
        r.build_quantized_proxy(&ctx,shape.input_size,0,&proxy).unwrap();
        assert_eq!(quantized_before,proxy.l2w.download(&ctx).unwrap());
        let w=r.weights.l2w.download(&ctx).unwrap();
        let slow=r.optimizer_states.l2w.slow_params.download(&ctx).unwrap();
        for i in 0..w.len() {
            let ch=i/shape.l2_in();let scale=params[ch]/(0.21f32+1e-5).sqrt();
            if scale==0.0 {assert_eq!(w[i],100.0);assert_eq!(slow[i],-100.0);}
            else {assert!((-2.0..=127.0/64.0).contains(&(w[i]*scale)));assert!((-2.0..=127.0/64.0).contains(&(slow[i]*scale)));}
        }
        assert_eq!(bn,r.read_batch_norm_state(&ctx).unwrap());
        assert_eq!(bias,r.weights.l2b.download(&ctx).unwrap());
        assert!(r.optimizer_states.l2w.momentum.download(&ctx).unwrap().iter().all(|&x|x==0.3));
        project_l2(&r,&ctx).unwrap();
        assert_eq!(w,r.weights.l2w.download(&ctx).unwrap());
        let batch=SfnnTrainStepHostBatch {stm_indices:&[0,1,2,3],nstm_indices:&[3,2,1,0],
            buckets:&[0,0,1,1],targets:&[0.1,0.3,0.7,0.9],entry_weights:&[1.0;4],batch_size:4,max_active:1};
        // BPU=2: projection must follow the real update, including Lookahead.
        for step in 0..12 {
            r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,Default::default(),
                ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,batch,true,step%2==1,Default::default()).unwrap();
            if step%2==1 {
                let states=r.read_batch_norm_state(&ctx).unwrap();
                let (scale,_)=states.0[2].as_ref().unwrap().inference_affine().unwrap();
                for values in [r.weights.l2w.download(&ctx).unwrap(),r.optimizer_states.l2w.slow_params.download(&ctx).unwrap()] {
                    for (i,v) in values.iter().enumerate() {
                        assert!((-2.000001..=1.984376).contains(&(v*scale[i/shape.l2_in()])));
                    }
                }
            }
        }
        r.configure_bn_l2_effective_weight_clip(&ctx,false).unwrap();
        assert!(!r.bn_qat.as_ref().unwrap().l2_effective_weight_clip);
    }
    #[test]
    fn bn_qat_toggle_preserves_trained_state() {
        let ctx=Context::new(0).unwrap();
        let shape=crate::tests::tiny_sfnn_shape();
        let mut r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(shape),4,1).unwrap();
        r.configure_batch_norm(&ctx,[true;3],Default::default(),&Default::default()).unwrap();
        let batch=SfnnTrainStepHostBatch {stm_indices:&[0,1,2,3],nstm_indices:&[3,2,1,0],buckets:&[0,0,1,1],
            targets:&[0.1,0.3,0.7,0.9],entry_weights:&[1.0;4],batch_size:4,max_active:1};
        for enabled in [false,true,false,true] {
            let weights=r.read_weights(&ctx).unwrap();
            let optimizer=r.read_optimizer_states(&ctx).unwrap();
            r.configure_bn_qat(&ctx,enabled,shape.input_size,0).unwrap();
            assert_eq!(weights,r.read_weights(&ctx).unwrap());
            assert_eq!(optimizer,r.read_optimizer_states(&ctx).unwrap());
            r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,Default::default(),
                ScalarLossKind::BceWithLogits,1.0,batch,true,true,Default::default()).unwrap();
            assert!(r.forward_workspace.output.download(&ctx).unwrap().iter().all(|v|v.is_finite()));
        }
    }
    #[test]
    fn bn_qat_fused_ft_matches_separate_quantize_unfold() {
        let ctx=Context::new(0).unwrap();
        let shape=crate::tests::tiny_sfnn_shape();
        let mut r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(shape),4,1).unwrap();
        calibrated(&mut r,&ctx);
        for enabled in [true,false] {
        for vr in [0,2] {
            let base=65536;let width=shape.ft_size;
            // Exercise half-integer boundaries, adjacent f32 values, int16
            // endpoints, both signs and out-of-range values.
            let values:Vec<_>=(0..(base+vr)*width).map(|i| {
                let v=((i%70000) as f32-35000.0+0.5)/127.0;
                f32::from_bits(v.to_bits().wrapping_add((i%3) as u32).wrapping_sub(1))
            }).collect();
            let src=F32Buffer::from_host(&ctx,&values).unwrap();
            let bias=F32Buffer::from_host(&ctx,&[0.1,-0.2,0.3,-0.4]).unwrap();
            let old=F32Buffer::new(&ctx,values.len()).unwrap();
            let new=F32Buffer::new(&ctx,values.len()).unwrap();
            let ob=F32Buffer::from_host(&ctx,&[7.0;4]).unwrap();
            let nb=F32Buffer::from_host(&ctx,&[7.0;4]).unwrap();
            let l=&r.batch_norm.as_ref().unwrap().layers[0].as_ref().unwrap().0;
            let mut a=l.affine.download(&ctx).unwrap();a[0]=0.0;a[1]=-0.75;a[2]=0.00001;l.affine.upload(&ctx,&a).unwrap();
            let mut stats=l.running.download(&ctx).unwrap();stats[2*width+3]=0.0;l.running.upload(&ctx,&stats).unwrap();
            let _binding=enabled.then(||r.batch_norm.as_ref().unwrap().bind(&ctx,1,false).unwrap());
            for alpha in [0.0,0.3,1.0] {
                check(unsafe {bulletou_bn_qat_ft(ctx.as_ptr(),src.as_ptr(),old.as_ptr(),base,vr,width,alpha)}).unwrap();
                check(unsafe {bulletou_bn_qat_unfold_training(ctx.as_ptr(),src.as_ptr(),bias.as_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),
                    old.as_ptr(),ob.as_ptr(),base,width,1,0,vr,alpha)}).unwrap();
                check(unsafe {bulletou_bn_qat_ft_training(ctx.as_ptr(),src.as_ptr(),new.as_ptr(),bias.as_ptr(),nb.as_ptr(),base,vr,width,alpha)}).unwrap();
                assert_eq!(old.download(&ctx).unwrap(),new.download(&ctx).unwrap());
                assert_eq!(ob.download(&ctx).unwrap(),nb.download(&ctx).unwrap());
            }
        }}
    }
    #[test]
    fn bn_qat_scratch_virtual_ft_and_unseen_bucket_calibration() {
        let ctx=Context::new(0).unwrap();
        let original=crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        let mut shape=original.shape;shape.input_size=6;
        // Metadata exists although no axis factorizer tensors are active.
        shape.factorizer_progress_axis=true;
        let mut l0=original.l0w.to_vec();l0.extend([0.03;8]);
        let initial=SfnnForwardHostWeights {shape,l0w:&l0,..original};
        let mut r=SfnnTrainStepRunner::new(&ctx,initial,4,1).unwrap();
        r.configure_batch_norm(&ctx,[true;3],Default::default(),&Default::default()).unwrap();
        r.configure_bn_qat(&ctx,true,4,2).unwrap();
        for bucket in [0,1] {
            let buckets=[bucket;4];
            let batch=SfnnTrainStepHostBatch {stm_indices:&[0,1,2,3],nstm_indices:&[3,2,1,0],buckets:&buckets,
                targets:&[0.1,0.3,0.7,0.9],entry_weights:&[1.0;4],batch_size:4,max_active:1};
            r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,Default::default(),
                ScalarLossKind::BceWithLogits,1.0,batch,true,true,Default::default()).unwrap();
            let state=r.read_batch_norm_state(&ctx).unwrap();
            for s in state.0[1..].iter().flatten() {
                let flags=&s.running[2*s.width*s.groups..];
                assert!(flags[..s.width].iter().all(|v|*v==1.0));
                assert!(flags[s.width..].iter().all(|v|*v==bucket as f32));
            }
            assert!(r.weights.l0w.download(&ctx).unwrap().iter().all(|v|v.is_finite()));
        }
    }
    #[test]
    fn bn_qat_from_scratch_updates_stats_and_accumulates_raw_gradients() {
        let ctx=Context::new(0).unwrap();
        let shape=crate::tests::tiny_sfnn_shape();
        let initial=crate::tests::tiny_sfnn_weights(shape);
        let batch=SfnnTrainStepHostBatch {stm_indices:&[0,1,2,3],nstm_indices:&[3,2,1,0],
            buckets:&[0,0,1,1],targets:&[0.1,0.3,0.7,0.9],entry_weights:&[1.0;4],batch_size:4,max_active:1};
        for mask in 1..8 {
            let enabled=[mask&1!=0,mask&2!=0,mask&4!=0];
            let mut r=SfnnTrainStepRunner::new(&ctx,initial,4,1).unwrap();
            r.configure_batch_norm(&ctx,enabled,Default::default(),&Default::default()).unwrap();
            r.configure_bn_qat(&ctx,true,shape.input_size,0).unwrap();
            let before=r.read_batch_norm_state(&ctx).unwrap();
            let mut expected=vec![0.0;shape.l1w_len().unwrap()];
            for _ in 0..2 {
                // Same master weights, fresh gradient buffers, same pre-batch statistics.
                let mut reference=SfnnTrainStepRunner::new(&ctx,initial,4,1).unwrap();
                reference.configure_batch_norm(&ctx,enabled,Default::default(),&r.read_batch_norm_state(&ctx).unwrap()).unwrap();
                reference.configure_bn_qat(&ctx,true,shape.input_size,0).unwrap();
                for runner in [&mut reference,&mut r] {
                    runner.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,Default::default(),
                        ScalarLossKind::BceWithLogits,1.0,batch,true,false,Default::default()).unwrap();
                }
                for (sum,g) in expected.iter_mut().zip(reference.backward_workspace.l1w_gradients.download(&ctx).unwrap()) {*sum+=g;}
                for (a,b) in expected.iter().zip(r.backward_workspace.l1w_gradients.download(&ctx).unwrap()) {close(*a,b);}
                for (a,b) in reference.forward_workspace.output.download(&ctx).unwrap().iter().zip(r.forward_workspace.output.download(&ctx).unwrap()) {close(*a,b);}
            }
            let calibrated=r.read_batch_norm_state(&ctx).unwrap();
            assert_ne!(before,calibrated);
            for s in calibrated.0.iter().flatten() {
                assert!(s.running[2*s.width*s.groups..].iter().all(|v|*v==1.0));
            }
            let original=r.weights.l1w.download(&ctx).unwrap();
            r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,Default::default(),
                ScalarLossKind::BceWithLogits,1.0,batch,true,true,Default::default()).unwrap();
            assert_ne!(original,r.weights.l1w.download(&ctx).unwrap());
            assert_eq!(r.pending_gradient_batches,0);
            for s in r.read_batch_norm_state(&ctx).unwrap().0.iter().flatten() {s.validate().unwrap();}
            let state=r.snapshot_device(&ctx).unwrap();
            let saved=r.read_batch_norm_state(&ctx).unwrap();
            r.copy_state_from_device(&ctx,&state).unwrap();
            assert_eq!(saved,r.read_batch_norm_state(&ctx).unwrap());
        }
    }

    #[test]
    fn bn_qat_train_unfold_matches_fold_and_handles_zero_gamma() {
        let ctx=Context::new(0).unwrap();
        let shape=crate::tests::tiny_sfnn_shape();
        let mut r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(shape),4,1).unwrap();
        calibrated(&mut r,&ctx);
        r.configure_bn_qat(&ctx,true,shape.input_size,0).unwrap();
        let q=r.bn_qat.take().unwrap();
        // Proxy tensors include optional shapes, so use a plain destination here.
        let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,shape).unwrap();
        r.build_quantized_proxy(&ctx,shape.input_size,0,&proxy).unwrap();
        let q=State {proxy,..q};
        let saved=r.read_batch_norm_state(&ctx).unwrap();
        let folded=q.proxy.l1w.download(&ctx).unwrap();
        let guard=r.batch_norm.as_ref().unwrap().bind(&ctx,1,false).unwrap();
        unfold_training(&r,&ctx,&q,true).unwrap();
        let s=saved.0[1].as_ref().unwrap();let (scale,_)=s.inference_affine().unwrap();
        for (j,w) in q.proxy.l1w.download(&ctx).unwrap().iter().enumerate() {
            let ch=j/shape.ft_size;let u=ch%shape.l1_out();
            let factor=if u<s.width {scale[(ch/shape.l1_out())*s.width+u]} else {1.0};
            close(*w*factor,folded[j]);
        }
        drop(guard);
        // Zero gamma must not divide by zero, and absent buckets still calibrate.
        for (l,_) in r.batch_norm.as_ref().unwrap().layers.iter().flatten() {
            let mut a=l.affine.download(&ctx).unwrap();a[0]=0.0;l.affine.upload(&ctx,&a).unwrap();
        }
        drop(q);
        r.configure_bn_qat(&ctx,true,shape.input_size,0).unwrap();
        let batch=SfnnTrainStepHostBatch {stm_indices:&[0,1,2,3],nstm_indices:&[3,2,1,0],buckets:&[0,0,0,0],
            targets:&[0.1,0.3,0.7,0.9],entry_weights:&[1.0;4],batch_size:4,max_active:1};
        let upload=Context::new(0).unwrap();
        r.step_pipelined_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,&upload,Default::default(),
            ScalarLossKind::BceWithLogits,1.0,batch,true,true,Default::default()).unwrap();
        r.step_profiled_no_readback_with_update_and_lr_multipliers(&ctx,Default::default(),
            ScalarLossKind::BceWithLogits,1.0,batch,true,Default::default()).unwrap();
        assert!(r.forward_workspace.output.download(&ctx).unwrap().iter().all(|v|v.is_finite()));
        for s in r.read_batch_norm_state(&ctx).unwrap().0.iter().flatten() {s.validate().unwrap();}
    }
    fn calibrated(r: &mut SfnnTrainStepRunner, ctx: &Context) {
        r.configure_batch_norm(ctx, [true; 3], Default::default(), &Default::default()).unwrap();
        for (l, _) in r.batch_norm.as_ref().unwrap().layers.iter().flatten() {
            let c = l.width * l.groups;
            let mut v = vec![0.13; 3 * c];
            v[c..2 * c].fill(0.21);
            v[2 * c..].fill(1.0);
            l.running.upload(ctx, &v).unwrap();
        }
    }
    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 2e-5 * (1.0 + a.abs() + b.abs()), "{a} != {b}");
    }
    #[test]
    fn bn_qat_forward_export_accumulation_update_and_resume() {
        let ctx = Context::new(0).unwrap();
        let shape = crate::tests::tiny_sfnn_shape();
        let initial = crate::tests::tiny_sfnn_weights(shape);
        let mut r = SfnnTrainStepRunner::new(&ctx, initial, 4, 1).unwrap();
        r.configure_batch_norm(&ctx, [true; 3], Default::default(), &Default::default()).unwrap();
        assert!(r.configure_bn_qat_mode(&ctx, true, shape.input_size, 0, true).is_err());
        calibrated(&mut r, &ctx);
        r.factorizer_alpha.ft = 0.3;
        r.factorizer_alpha.shared = 0.7;
        r.configure_bn_qat_mode(&ctx, true, shape.input_size, 0, true).unwrap();
        let saved = r.read_batch_norm_state(&ctx).unwrap();
        let raw = r.weights.l1w.download(&ctx).unwrap();
        let batch = SfnnTrainStepHostBatch {
            stm_indices: &[0, 1, 2, 3],
            nstm_indices: &[3, 2, 1, 0],
            buckets: &[0, 0, 1, 1],
            targets: &[0.1, 0.3, 0.7, 0.9],
            entry_weights: &[1.0; 4],
            batch_size: 4,
            max_active: 1,
        };
        let proxy = SfnnForwardDeviceWeights::new_dense(&ctx, shape).unwrap();
        r.build_quantized_proxy(&ctx, shape.input_size, 0, &proxy).unwrap();
        let mut first = Vec::new();
        for i in 0..2 {
            r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(
                &ctx,
                Default::default(),
                ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                1.0,
                batch,
                true,
                false,
                Default::default(),
            )
            .unwrap();
            let out = r.forward_workspace.output.download(&ctx).unwrap();
            sfnn_forward_device_with_factorizer(
                &ctx,
                &r.device_batch,
                &proxy,
                &r.forward_workspace,
                SfnnFactorizerActive::NONE,
            )
            .unwrap();
            for (a, b) in out.iter().zip(r.forward_workspace.output.download(&ctx).unwrap()) {
                close(*a, b);
            }
            let g = r.backward_workspace.l1w_gradients.download(&ctx).unwrap();
            if i == 0 {
                first = g;
            } else {
                for (a, b) in g.iter().zip(&first) {
                    close(*a, 2.0 * b);
                }
            }
            assert_eq!(raw, r.weights.l1w.download(&ctx).unwrap());
            assert_eq!(saved, r.read_batch_norm_state(&ctx).unwrap());
        }
        assert!(r.configure_bn_qat(&ctx, false, shape.input_size, 0).is_err());
        // Third minibatch performs the one BPU update; running stats stay fixed.
        r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(
            &ctx,
            Default::default(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            batch,
            true,
            true,
            Default::default(),
        )
        .unwrap();
        let after = r.read_batch_norm_state(&ctx).unwrap();
        for (a, b) in saved.0.iter().flatten().zip(after.0.iter().flatten()) {
            assert_eq!(a.running, b.running);
        }
        assert_ne!(raw, r.weights.l1w.download(&ctx).unwrap());
        assert!(saved.0.iter().flatten().zip(after.0.iter().flatten()).any(|(a, b)| a.affine != b.affine));
        assert_eq!(r.pending_gradient_batches, 0);
        let snapshot = r.snapshot_device(&ctx).unwrap();
        // Checkpoint BN tensors keep the same format, including optimizer moments.
        for s in after.0.iter().flatten() {
            assert_eq!(batch_norm::State::decode(&s.encode().unwrap()).unwrap(), *s);
        }
        r.copy_state_from_device(&ctx, &snapshot).unwrap();
        assert_eq!(after, r.read_batch_norm_state(&ctx).unwrap());
        let upload = Context::new(0).unwrap();
        r.step_pipelined_no_readback_with_loss_finalize_update_and_lr_multipliers(
            &ctx,
            &upload,
            Default::default(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            batch,
            true,
            true,
            Default::default(),
        )
        .unwrap();
        r.step_profiled_no_readback_with_update_and_lr_multipliers(
            &ctx,
            Default::default(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            batch,
            true,
            Default::default(),
        )
        .unwrap();
        for (a, b) in saved.0.iter().flatten().zip(r.read_batch_norm_state(&ctx).unwrap().0.iter().flatten()) {
            assert_eq!(a.running, b.running);
        }
    }

    #[test]
    fn bn_qat_fold_pullback_matches_chain_rule_zero_negative_gamma_and_virtual_ft() {
        let ctx = Context::new(0).unwrap();
        let mut shape = crate::tests::tiny_sfnn_shape();
        shape.input_size = 6;
        let l0 = vec![0.4; shape.input_size * shape.ft_size];
        let original = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        let initial = SfnnForwardHostWeights { shape, l0w: &l0, ..original };
        let mut r = SfnnTrainStepRunner::new(&ctx, initial, 4, 1).unwrap();
        calibrated(&mut r, &ctx);
        for (l, _) in r.batch_norm.as_ref().unwrap().layers.iter().flatten() {
            let c = l.width * l.groups;
            let mut a = vec![0.0; 2 * c];
            for i in 0..c {
                a[i] = if i % 2 == 0 { 0.0 } else { -0.7 };
                a[c + i] = 0.4;
            }
            l.affine.upload(&ctx, &a).unwrap();
        }
        r.configure_bn_qat_mode(&ctx, true, 4, 2, true).unwrap();
        // Factorized FT proxy equals the export model exactly (including zero
        // gamma), with virtual feature rows removed instead of quantized twice.
        let mut export_shape = shape;
        export_shape.input_size = 4;
        let export = SfnnForwardDeviceWeights::new_dense(&ctx, export_shape).unwrap();
        r.build_quantized_proxy(&ctx, 4, 2, &export).unwrap();
        {
            let _bind = r.batch_norm.as_ref().unwrap().bind(&ctx, 1, false).unwrap();
            let dst = &r.bn_qat.as_ref().unwrap().proxy.l0w;
            check(unsafe {
                bulletou_bn_qat_ft(
                    ctx.as_ptr(),
                    r.weights.l0w.as_ptr(),
                    dst.as_ptr(),
                    4,
                    2,
                    shape.ft_size,
                    r.factorizer_alpha.ft,
                )
            })
            .unwrap();
            let v = dst.download(&ctx).unwrap();
            assert_eq!(&v[..4 * shape.ft_size], export.l0w.download(&ctx).unwrap());
            assert!(v[4 * shape.ft_size..].iter().all(|x| *x == 0.0));
        }
        let state = r.read_batch_norm_state(&ctx).unwrap();
        let w = &r.weights;
        let g = &r.backward_workspace;
        for x in [
            &g.l0w_gradients,
            &g.l0b_gradients,
            &g.l1w_gradients,
            &g.l1b_gradients,
            &g.l2w_gradients,
            &g.l2b_gradients,
            &g.l3w_gradients,
            &g.l3b_gradients,
        ] {
            x.fill(&ctx, 0.2).unwrap();
        }
        let weights = [w.l0w.download(&ctx).unwrap(), w.l1w.download(&ctx).unwrap(), w.l2w.download(&ctx).unwrap()];
        let biases = [w.l0b.download(&ctx).unwrap(), w.l1b.download(&ctx).unwrap(), w.l2b.download(&ctx).unwrap()];
        pullback(&r, &ctx, r.bn_qat.as_ref().unwrap()).unwrap();
        for (i, (l, _)) in r.batch_norm.as_ref().unwrap().layers.iter().flatten().enumerate() {
            let s = state.0[i].as_ref().unwrap();
            let c = s.width * s.groups;
            let grads = l.gradients.download(&ctx).unwrap();
            let input = match i {
                0 => 4,
                1 => shape.ft_size,
                _ => shape.l2_in(),
            };
            let output = if i == 1 { shape.l1_out() } else { s.width };
            for ch in 0..c {
                let group = ch / s.width;
                let u = ch % s.width;
                let mut dot = 0.0;
                for k in 0..input {
                    let j = if i == 0 { k * output + u } else { (group * output + u) * input + k };
                    let mut value = weights[i][j];
                    if i == 0 {
                        value += r.factorizer_alpha.ft * weights[i][(4 + k % 2) * output + u];
                    }
                    if i == 1 && r.factorizer.shared {
                        value += r.factorizer_alpha.shared
                            * w.l1fw.as_ref().unwrap().download(&ctx).unwrap()[k * output + u];
                    }
                    dot += 0.2 * value;
                }
                let inv = 1.0 / (s.running[c + ch] + s.config.epsilon).sqrt();
                let mut bias = biases[i][group * output + u];
                if i == 1 && r.factorizer.shared {
                    bias += r.factorizer_alpha.shared * w.l1fb.as_ref().unwrap().download(&ctx).unwrap()[u];
                }
                close(grads[ch], (dot + 0.2 * (bias - s.running[ch])) * inv);
                close(grads[c + ch], 0.2);
            }
        }
        let g0 = g.l0w_gradients.download(&ctx).unwrap();
        let shared_grad = g.l1fw_gradients.download(&ctx).unwrap();
        let shared_bias_grad = g.l1fb_gradients.download(&ctx).unwrap();
        let s1 = state.0[1].as_ref().unwrap();
        for u in 0..shape.l1_out() {
            let expected = (0..shape.num_stacks)
                .map(|b| {
                    let scale = if u < s1.width {
                        let ch = b * s1.width + u;
                        s1.affine[ch] / (s1.running[s1.width * s1.groups + ch] + s1.config.epsilon).sqrt()
                    } else {
                        1.0
                    };
                    0.2 * scale * r.factorizer_alpha.shared
                })
                .sum::<f32>();
            close(shared_bias_grad[u], expected);
            for k in 0..shape.ft_size {
                close(shared_grad[k * shape.l1_out() + u], expected);
            }
        }
        let s = state.0[0].as_ref().unwrap();
        for f in 0..6 {
            for u in 0..shape.ft_size {
                let scale = s.affine[u] / (s.running[s.width + u] + s.config.epsilon).sqrt();
                close(g0[f * shape.ft_size + u], 0.2 * scale * if f < 4 { 1.0 } else { 2.0 * r.factorizer_alpha.ft });
            }
        }
    }
}

unsafe extern "C" {
    fn bulletou_bn_qat_unfold_training(
        ctx: *mut ffi::BulletOuCudaCppContext,
        w: *mut ffi::BulletOuCudaCppF32Buffer, b: *mut ffi::BulletOuCudaCppF32Buffer,
        sw: *mut ffi::BulletOuCudaCppF32Buffer, sb: *mut ffi::BulletOuCudaCppF32Buffer,
        qw: *mut ffi::BulletOuCudaCppF32Buffer, qb: *mut ffi::BulletOuCudaCppF32Buffer,
        input: usize, output: usize, groups: usize, layer: i32, virtual_rows: usize, alpha: f32,
    ) -> i32;
    fn bulletou_bn_qat_ft(
        ctx: *mut ffi::BulletOuCudaCppContext,
        src: *mut ffi::BulletOuCudaCppF32Buffer,
        dst: *mut ffi::BulletOuCudaCppF32Buffer,
        base: usize,
        virtual_rows: usize,
        width: usize,
        alpha: f32,
    ) -> i32;
    fn bulletou_bn_qat_ft_training(
        ctx: *mut ffi::BulletOuCudaCppContext, src: *mut ffi::BulletOuCudaCppF32Buffer,
        dst: *mut ffi::BulletOuCudaCppF32Buffer, bias: *mut ffi::BulletOuCudaCppF32Buffer,
        out_bias: *mut ffi::BulletOuCudaCppF32Buffer,
        base: usize, virtual_rows: usize, width: usize, alpha: f32,
    ) -> i32;
    fn bulletou_bn_qat_pullback(
        ctx: *mut ffi::BulletOuCudaCppContext,
        w: *mut ffi::BulletOuCudaCppF32Buffer,
        b: *mut ffi::BulletOuCudaCppF32Buffer,
        shared: *mut ffi::BulletOuCudaCppF32Buffer,
        shared_b: *mut ffi::BulletOuCudaCppF32Buffer,
        gw: *mut ffi::BulletOuCudaCppF32Buffer,
        gb: *mut ffi::BulletOuCudaCppF32Buffer,
        gs: *mut ffi::BulletOuCudaCppF32Buffer,
        gsb: *mut ffi::BulletOuCudaCppF32Buffer,
        input: usize,
        output: usize,
        groups: usize,
        layer: i32,
        virtual_rows: usize,
        alpha: f32,
    ) -> i32;
}
fn ptr(b: Option<&F32Buffer>) -> *mut ffi::BulletOuCudaCppF32Buffer {
    b.map_or(std::ptr::null_mut(), F32Buffer::as_ptr)
}

/// Keep the normal optimizer/state in raw coordinates, even on a failed step.
/// Train-mode gradients accumulate in pre-BN coordinates; frozen mode uses
/// folded coordinates. Factorizer pullback happens at the actual BPU update.
pub(super) fn step<T>(
    r: &mut SfnnTrainStepRunner,
    ctx: &Context,
    params: RangerUpdateParams,
    lr: SfnnLayerLrMultipliers,
    update: bool,
    dirty: Option<&[i32]>,
    run: impl FnOnce(&mut SfnnTrainStepRunner) -> Result<T>,
) -> Result<T> {
    if r.batch_norm.is_none() {
        return Err(CudaCppError::message("BN QAT requires the configured BN state"));
    }
    if lr.l1_center
        || lr.l2_l3_center
        || lr.qat_l1
        || lr.l1_effective_weight_clip
        || lr.saturation_penalty != 0.0
        || lr.ft_saturation_penalty != 0.0
        || lr.update_scope != SfnnUpdateScope::All
        || lr.l0 != 1.0
        || lr.l1 != 1.0
        || lr.l2 != 1.0
        || lr.l3 != 1.0
    {
        return Err(CudaCppError::message(
            "BN QAT requires no centering, L1-only QAT, penalties, effective clipping or layer freezing/LR multipliers",
        ));
    }
    let mut q = r.bn_qat.take().unwrap();
    let result = (|| {
        // Training statistics change every microbatch, so refresh its proxy.
        // Frozen-stat mode can reuse the proxy throughout a BPU group.
        if !q.freeze_stats || r.pending_gradient_batches == 0 {
            let optional = (
                q.proxy.l1fw.take(),
                q.proxy.l1fb.take(),
                q.proxy.l2fw.take(),
                q.proxy.l2fb.take(),
                q.proxy.l3fw.take(),
                q.proxy.l3fb.take(),
            );
            // Export proxy has no axes, even when the architecture carries
            // inactive king/progress axis metadata (factorizer none/shared).
            let training_shape=q.proxy.shape;
            q.proxy.shape.factorizer_king_axis_dim=0;
            q.proxy.shape.factorizer_hand_axis_dim=0;
            q.proxy.shape.factorizer_progress_axis=false;
            q.proxy.shape.factorizer_king_hand_pair=false;
            q.proxy.shape.factorizer_king_progress_pair=false;
            q.proxy.shape.factorizer_hand_progress_pair=false;
            // The specialized FT kernel below replaces generic FT quantization.
            // Avoid writing the full FT matrix twice on every microbatch.
            let built = (|| {
                let _bind = r.batch_norm.as_ref().unwrap().bind(ctx, 1, false)?;
                sfnn_build_quantized_proxy_device_impl(ctx, r.shape.input_size, 0, &r.weights, &q.proxy,
                    r.factorizer, r.factorizer_alpha,
                    r.residual_count_gates_enabled.then_some(&r.residual_count_gates_by_stack),
                    r.factorizer_axis_confidences_enabled.then_some(&r.factorizer_axis_confidences), true)
            })();
            q.proxy.shape=training_shape;
            (q.proxy.l1fw, q.proxy.l1fb, q.proxy.l2fw, q.proxy.l2fb, q.proxy.l3fw, q.proxy.l3fb) = optional;
            built?;
            let _bind = r.batch_norm.as_ref().unwrap().bind(ctx, 1, false)?;
            if q.freeze_stats {
            check(unsafe {
                bulletou_bn_qat_ft(
                    ctx.as_ptr(),
                    r.weights.l0w.as_ptr(),
                    q.proxy.l0w.as_ptr(),
                    q.base,
                    q.virtual_rows,
                    r.shape.ft_size,
                    r.factorizer_alpha.ft,
                )
            })?;
            } else {
                check(unsafe { bulletou_bn_qat_ft_training(ctx.as_ptr(),r.weights.l0w.as_ptr(),q.proxy.l0w.as_ptr(),
                    r.weights.l0b.as_ptr(),q.proxy.l0b.as_ptr(),q.base,q.virtual_rows,r.shape.ft_size,r.factorizer_alpha.ft) })?;
                unfold_training(r, ctx, &q, false)?;
            }
        }
        let bn = if q.freeze_stats { r.batch_norm.take() } else { None };
        let factorizer = r.factorizer;
        r.factorizer = SfnnFactorizerActive::NONE;
        std::mem::swap(&mut r.weights, &mut q.proxy);
        let result = run(r);
        std::mem::swap(&mut r.weights, &mut q.proxy);
        r.factorizer = factorizer;
        if q.freeze_stats { r.batch_norm = bn; }
        let value = result?;
        if update {
            pullback(r, ctx, &q)?;
            r.update_weights_with_lr_multipliers_and_dirty_buckets(ctx, params, lr, dirty)?;
            // After both Ranger/Lookahead and BN gamma updates: use the new fold scale.
            if q.l2_effective_weight_clip { project_l2(r, ctx)?; }
        }
        Ok(value)
    })();
    r.bn_qat = Some(q);
    result
}

fn unfold_training(r: &SfnnTrainStepRunner, ctx: &Context, q: &State, include_ft: bool) -> Result<()> {
    let w=&r.weights; let p=&q.proxy; let s=r.shape;
    for (i, w, b, sw, sb, qw, qb, input, output, groups) in [
        (0,&w.l0w,&w.l0b,None,None,&p.l0w,&p.l0b,q.base,s.ft_size,1),
        (1,&w.l1w,&w.l1b,w.l1fw.as_ref(),w.l1fb.as_ref(),&p.l1w,&p.l1b,s.ft_size,s.l1_out(),s.num_stacks),
        (2,&w.l2w,&w.l2b,None,None,&p.l2w,&p.l2b,s.l2_in(),s.l2_size,s.num_stacks),
    ] {
        if i==0 && !include_ft { continue; }
        let shared=r.factorizer.shared && i==1;
        check(unsafe { bulletou_bn_qat_unfold_training(ctx.as_ptr(),w.as_ptr(),b.as_ptr(),
            ptr(sw.filter(|_|shared)),ptr(sb.filter(|_|shared)),qw.as_ptr(),qb.as_ptr(),
            input,output,groups,i,if i==0 {q.virtual_rows} else {0},
            if i==0 {r.factorizer_alpha.ft} else {r.factorizer_alpha.shared}) })?;
    }
    Ok(())
}

fn pullback(r: &SfnnTrainStepRunner, ctx: &Context, q: &State) -> Result<()> {
    // Train-mode BN already differentiated mean/variance and gamma/beta.
    // Its raw-affine gradients need only the FT/shared factorizer chain rule.
    let _bind = if q.freeze_stats { Some(r.batch_norm.as_ref().unwrap().bind(ctx, 1, false)?) } else { None };
    let w = &r.weights;
    let g = &r.backward_workspace;
    let s = r.shape;
    for (i, ww, b, sw, sb, gw, gb, gs, gsb, input, output, groups) in [
        (0, &w.l0w, &w.l0b, None, None, &g.l0w_gradients, &g.l0b_gradients, None, None, q.base, s.ft_size, 1),
        (
            1,
            &w.l1w,
            &w.l1b,
            w.l1fw.as_ref(),
            w.l1fb.as_ref(),
            &g.l1w_gradients,
            &g.l1b_gradients,
            Some(&g.l1fw_gradients),
            Some(&g.l1fb_gradients),
            s.ft_size,
            s.l1_out(),
            s.num_stacks,
        ),
        (
            2,
            &w.l2w,
            &w.l2b,
            None,
            None,
            &g.l2w_gradients,
            &g.l2b_gradients,
            None,
            None,
            s.l2_in(),
            s.l2_size,
            s.num_stacks,
        ),
        (3, &w.l3w, &w.l3b, None, None, &g.l3w_gradients, &g.l3b_gradients, None, None, s.l2_size, 1, s.num_stacks),
    ] {
        let shared = r.factorizer.shared && i == 1;
        check(unsafe {
            bulletou_bn_qat_pullback(
                ctx.as_ptr(),
                ww.as_ptr(),
                b.as_ptr(),
                ptr(sw.filter(|_| shared)),
                ptr(sb.filter(|_| shared)),
                gw.as_ptr(),
                gb.as_ptr(),
                ptr(gs.filter(|_| shared)),
                ptr(gsb.filter(|_| shared)),
                input,
                output,
                groups,
                i,
                if i == 0 { q.virtual_rows } else { 0 },
                if i == 0 { r.factorizer_alpha.ft } else { r.factorizer_alpha.shared },
            )
        })?;
    }
    Ok(())
}
