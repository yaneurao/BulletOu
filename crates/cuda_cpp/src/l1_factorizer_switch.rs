//! Fold removed axis terms at an optimizer boundary; keep base/shared moments.
use super::*;

pub fn fold_axes(shape: SfnnForwardShape, w: &mut [f32], b: &mut [f32],
    aw: &[f32], ab: &[f32], coefficients: &[Vec<(usize, f32)>]) -> Result<()> {
    let n = shape.l1_out();
    expect_len("L1 fold weights", shape.num_stacks * shape.ft_size * n, w.len())?;
    expect_len("L1 fold bias", shape.num_stacks * n, b.len())?;
    expect_len("L1 fold axes", shape.factorizer_axis_count()*shape.ft_size*n, aw.len())?;
    expect_len("L1 fold axis bias", shape.factorizer_axis_count()*n, ab.len())?;
    expect_len("L1 fold coefficients", shape.num_stacks, coefficients.len())?;
    for (stack, axes) in coefficients.iter().enumerate() {
        for &(axis, c) in axes {
            if axis >= shape.factorizer_axis_count() || !c.is_finite() {
                return Err(CudaCppError::message("invalid L1 axis fold coefficient"));
            }
            for u in 0..n {
                b[stack*n+u] += c*ab[axis*n+u];
                for i in 0..shape.ft_size {
                    w[(stack*n+u)*shape.ft_size+i] += c*aw[(axis*shape.ft_size+i)*n+u];
                }
            }
        }
    }
    if w.iter().chain(b.iter()).any(|x| !x.is_finite()) {
        return Err(CudaCppError::message("non-finite folded L1 parameters"));
    }
    Ok(())
}

impl SfnnTrainStepRunner {
    pub fn fold_l1_axes_to_shared(&mut self, ctx: &Context, coefficients: &[Vec<(usize, f32)>]) -> Result<()> {
        if self.pending_gradient_batches != 0 || self.batch_norm.is_some() || self.shape.has_compact_l1()
            || !self.factorizer.shared {
            return Err(CudaCppError::message("L1 axis fold requires non-BN dense shared L1 at an optimizer-update boundary"));
        }
        if !self.factorizer.any_axis() { return Ok(()); }
        ctx.synchronize()?;
        let aw = self.weights.l1axw.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis weights"))?.download(ctx)?;
        let ab = self.weights.l1axb.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis bias"))?.download(ctx)?;
        let mut w = self.weights.l1w.download(ctx)?;
        let mut b = self.weights.l1b.download(ctx)?;
        fold_axes(self.shape, &mut w, &mut b, &aw, &ab, coefficients)?;
        let mut sw = self.optimizer_states.l1w.slow_params.download(ctx)?;
        let mut sb = self.optimizer_states.l1b.slow_params.download(ctx)?;
        let saw = self.optimizer_states.l1axw.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis optimizer"))?.slow_params.download(ctx)?;
        let sab = self.optimizer_states.l1axb.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis bias optimizer"))?.slow_params.download(ctx)?;
        fold_axes(self.shape, &mut sw, &mut sb, &saw, &sab, coefficients)?;
        self.weights.l1w.upload(ctx, &w)?;
        self.weights.l1b.upload(ctx, &b)?;
        self.optimizer_states.l1w.slow_params.upload(ctx, &sw)?;
        self.optimizer_states.l1b.slow_params.upload(ctx, &sb)?;
        self.weights.l1axw = None; self.weights.l1axb = None;
        self.optimizer_states.l1axw = None; self.optimizer_states.l1axb = None;
        self.factorizer_axis_confidences_enabled = false;
        self.layer_qat.invalidate();
        self.set_factorizer_config(SfnnFactorizerActive::SHARED, self.factorizer_alpha)?;
        Ok(())
    }

    pub fn enable_l1_axes_zero(&mut self, ctx: &Context, shape: SfnnForwardShape,
        active: SfnnFactorizerActive, alpha: SfnnFactorizerAlpha) -> Result<()> {
        if self.pending_gradient_batches != 0 || self.batch_norm.is_some() || self.shape.has_compact_l1()
            || self.factorizer != SfnnFactorizerActive::SHARED || self.weights.l1axw.is_some() {
            return Err(CudaCppError::message("L1 axis enable requires non-BN dense shared L1 at an optimizer-update boundary"));
        }
        let mut expected = self.shape;
        expected.factorizer_progress_axis = shape.factorizer_progress_axis;
        if expected != shape || !active.shared || !active.any_axis() {
            return Err(CudaCppError::message("incompatible L1 axis enable shape"));
        }
        active.validate_for_shape(shape)?;
        alpha.validate()?;
        ctx.synchronize()?;
        let w = vec![0.0; shape.factorizer_axis_count()*shape.ft_size*shape.l1_out()];
        let b = vec![0.0; shape.factorizer_axis_count()*shape.l1_out()];
        self.weights.l1axw = Some(F32Buffer::from_host(ctx, &w)?);
        self.weights.l1axb = Some(F32Buffer::from_host(ctx, &b)?);
        self.optimizer_states.l1axw = Some(RangerParamState::from_host_weights(ctx, &w)?);
        self.optimizer_states.l1axb = Some(RangerParamState::from_host_weights(ctx, &b)?);
        let mut layout = self.backward_workspace.layout;
        layout.shape = shape;
        for (dst, len) in [
            (&mut self.backward_workspace.l1axw_gradients, layout.l1axw_gradients_len()),
            (&mut self.backward_workspace.l1axb_gradients, layout.l1axb_gradients_len()),
            (&mut self.backward_workspace.l2axw_gradients, layout.l2axw_gradients_len()),
            (&mut self.backward_workspace.l2axb_gradients, layout.l2axb_gradients_len()),
            (&mut self.backward_workspace.l3axw_gradients, layout.l3axw_gradients_len()),
            (&mut self.backward_workspace.l3axb_gradients, layout.l3axb_gradients_len()),
        ] {
            *dst = F32Buffer::new(ctx, len)?;
            dst.fill(ctx, 0.0)?;
        }
        self.shape = shape; self.weights.shape = shape;
        self.forward_workspace.layout.shape = shape;
        self.backward_workspace.layout = layout;
        self.factorizer_axis_confidences = F32Buffer::new(ctx, shape.factorizer_axis_count())?;
        self.factorizer_axis_confidences.fill(ctx, 1.0)?;
        self.factorizer_axis_confidences_enabled = false;
        self.layer_qat.invalidate();
        self.set_factorizer_config(active, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA; tiny synthetic model"]
    fn bidirectional_fold_preserves_output_proxy_and_optimizer() {
        let ctx = Context::new(0).unwrap();
        let mut host = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        host.l2fw=None; host.l2fb=None; host.l3fw=None; host.l3fb=None;
        let mut r = SfnnTrainStepRunner::new(&ctx, host, 4, 2).unwrap();
        let mut shape = host.shape; shape.factorizer_progress_axis = true;
        let active = SfnnFactorizerActive { progress_axis: true, ..SfnnFactorizerActive::SHARED };
        let alpha = SfnnFactorizerAlpha { progress_axis: 0.7, ..SfnnFactorizerAlpha::ONE };
        r.optimizer_states.l1w.momentum.fill(&ctx, 0.123).unwrap();
        r.optimizer_states.l1w.velocity.fill(&ctx, 0.456).unwrap();
        let base = r.weights.l1w.download(&ctx).unwrap();
        r.enable_l1_axes_zero(&ctx, shape, active, alpha).unwrap();
        assert_eq!(r.weights.l1w.download(&ctx).unwrap(), base);
        assert!(r.optimizer_states.l1axw.as_ref().unwrap().momentum.download(&ctx).unwrap().iter().all(|x|*x==0.0));
        let aw: Vec<_> = (0..shape.factorizer_axis_count()*shape.ft_size*shape.l1_out()).map(|i|0.01*(i as f32-7.)).collect();
        let ab = vec![0.02; shape.factorizer_axis_count()*shape.l1_out()];
        r.weights.l1axw.as_ref().unwrap().upload(&ctx,&aw).unwrap();
        r.weights.l1axb.as_ref().unwrap().upload(&ctx,&ab).unwrap();
        r.optimizer_states.l1axw.as_ref().unwrap().slow_params.upload(&ctx,&aw).unwrap();
        r.optimizer_states.l1axb.as_ref().unwrap().slow_params.upload(&ctx,&ab).unwrap();
        r.set_residual_count_gates_by_stack(&ctx,Some(&[0.5,0.25])).unwrap();
        r.set_factorizer_axis_confidences(&ctx,Some(&[0.8,0.6])).unwrap();
        let coefficients=vec![vec![(0,0.7*0.8/0.5)],vec![(1,0.7*0.6/0.25)]];
        let batch=SfnnForwardDeviceBatch::from_host(&ctx,SfnnForwardHostBatch {
            stm_indices:&[0,1,1,2,2,3,0,3],nstm_indices:&[3,2,2,1,1,0,3,0],buckets:&[0,1,0,1],batch_size:4,max_active:2}).unwrap();
        let workspace=SfnnForwardWorkspace::new(&ctx,SfnnForwardWorkspaceLayout::new(shape,4)).unwrap();
        r.forward_current_weights(&ctx,&batch,&workspace).unwrap();
        let before=workspace.download_output(&ctx).unwrap();
        let proxy_shape=SfnnForwardShape {factorizer_progress_axis:false,..shape};
        let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,proxy_shape).unwrap();
        r.build_quantized_proxy(&ctx,shape.input_size,0,&proxy).unwrap();
        let before_q=proxy.l1w.download(&ctx).unwrap();
        r.fold_l1_axes_to_shared(&ctx,&coefficients).unwrap();
        r.forward_current_weights(&ctx,&batch,&workspace).unwrap();
        for (a,b) in before.iter().zip(workspace.download_output(&ctx).unwrap()) { assert!((a-b).abs()<1e-5); }
        r.build_quantized_proxy(&ctx,shape.input_size,0,&proxy).unwrap();
        assert_eq!(before_q,proxy.l1w.download(&ctx).unwrap());
        assert!(r.optimizer_states.l1w.momentum.download(&ctx).unwrap().iter().all(|x|*x==0.123));
        assert!(r.optimizer_states.l1w.velocity.download(&ctx).unwrap().iter().all(|x|*x==0.456));
        assert_eq!(r.weights.l1w.download(&ctx).unwrap(),r.optimizer_states.l1w.slow_params.download(&ctx).unwrap());
        let folded=r.weights.l1w.download(&ctx).unwrap();
        r.fold_l1_axes_to_shared(&ctx,&coefficients).unwrap(); // no double fold
        r.enable_l1_axes_zero(&ctx,shape,active,alpha).unwrap();
        assert_eq!(r.weights.l1w.download(&ctx).unwrap(),folded);
        assert!(r.weights.l1axw.as_ref().unwrap().download(&ctx).unwrap().iter().all(|x|*x==0.0));
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,RangerUpdateParams::default(),ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,
            SfnnTrainStepHostBatch{stm_indices:&[0,1,1,2,2,3,0,3],nstm_indices:&[3,2,2,1,1,0,3,0],
            buckets:&[0,1,0,1],targets:&[0.2,0.8,0.3,0.7],entry_weights:&[1.;4],batch_size:4,max_active:2},
            true,true,SfnnLayerLrMultipliers::default(),None).unwrap();
        assert!(r.forward_workspace.download_output(&ctx).unwrap().iter().all(|x|x.is_finite()));
    }
}
