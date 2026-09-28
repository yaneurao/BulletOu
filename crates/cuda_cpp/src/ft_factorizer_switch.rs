//! One-way FT fold. Base moments survive; shared moments are discarded.
use super::*;

pub fn fold_rows(values: &mut Vec<f32>, base: usize, virtual_rows: usize, width: usize, alpha: f32) -> Result<()> {
    if virtual_rows == 0 || base == 0 || width == 0 || !alpha.is_finite() || alpha < 0.0 {
        return Err(CudaCppError::message("invalid FT fold dimensions/coefficient"));
    }
    let rows = base.checked_add(virtual_rows).ok_or_else(|| CudaCppError::message("FT fold overflow"))?;
    expect_len("FT fold weights", checked_product("FT fold", &[rows, width])?, values.len())?;
    for row in 0..base {
        for u in 0..width {
            let value = values[row*width+u] + alpha*values[(base+row%virtual_rows)*width+u];
            if !value.is_finite() { return Err(CudaCppError::message("non-finite FT fold result")); }
            values[row*width+u] = value;
        }
    }
    values.truncate(base*width);
    Ok(())
}

impl SfnnTrainStepRunner {
    /// At an update boundary, remove the virtual FT rows without rebuilding other layers.
    pub fn disable_ft_factorizer(&mut self, ctx: &Context, base: usize, virtual_rows: usize) -> Result<()> {
        if self.pending_gradient_batches != 0 || self.batch_norm.is_some() {
            return Err(CudaCppError::message("FT factorizer switch requires non-BN and an optimizer-update boundary"));
        }
        if self.shape.input_size == base { return Ok(()); }
        if self.shape.input_size != base + virtual_rows {
            return Err(CudaCppError::message("FT factorizer switch: incompatible input layout"));
        }
        ctx.synchronize()?;
        let mut weights = self.weights.l0w.download(ctx)?;
        let mut state = self.optimizer_states.l0w.download(ctx)?;
        fold_rows(&mut weights, base, virtual_rows, self.shape.ft_size, self.factorizer_alpha.ft)?;
        fold_rows(&mut state.slow_params, base, virtual_rows, self.shape.ft_size, self.factorizer_alpha.ft)?;
        state.momentum.truncate(weights.len());
        state.velocity.truncate(weights.len());
        // Release QAT storage and replace one FT buffer at a time. No second model on GPU.
        self.layer_qat = Default::default();
        self.bn_qat = None;
        fn replace(ctx: &Context, dst: &mut F32Buffer, values: &[f32]) -> Result<()> {
            drop(std::mem::replace(dst, F32Buffer::new(ctx, 0)?));
            *dst = F32Buffer::from_host(ctx, values)?;
            Ok(())
        }
        replace(ctx, &mut self.weights.l0w, &weights)?;
        replace(ctx, &mut self.optimizer_states.l0w.momentum, &state.momentum)?;
        replace(ctx, &mut self.optimizer_states.l0w.velocity, &state.velocity)?;
        replace(ctx, &mut self.optimizer_states.l0w.slow_params, &state.slow_params)?;
        drop(std::mem::replace(&mut self.backward_workspace.l0w_gradients, F32Buffer::new(ctx, 0)?));
        self.backward_workspace.l0w_gradients = F32Buffer::new(ctx, weights.len())?;
        self.backward_workspace.l0w_gradients.fill(ctx, 0.0)?;
        self.shape.input_size = base;
        self.weights.shape = self.shape;
        self.forward_workspace.layout.shape = self.shape;
        self.backward_workspace.layout.shape = self.shape;
        self.factorizer_alpha.ft = 1.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_fold_keeps_effective_rows() {
        let mut v = vec![1.,2.,3.,4.,5.,6.,7.,8.,10.,20.,30.,40.];
        fold_rows(&mut v,4,2,2,0.5).unwrap();
        assert_eq!(v,vec![6.,12.,18.,24.,10.,16.,22.,28.]);
        assert!(fold_rows(&mut v,4,2,2,1.0).is_err());
    }

    #[test]
    #[ignore = "requires CUDA; tiny model, no teacher data"]
    fn gpu_fold_preserves_proxy_slow_and_base_moments_and_trains() {
        const BASE: usize = 131949;
        const VR: usize = 1629;
        let ctx = Context::new(0).unwrap();
        let mut host = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        host.l2fw=None; host.l2fb=None; host.l3fw=None; host.l3fb=None;
        host.shape.input_size=BASE+VR;
        let ft:Vec<_>=(0..host.shape.input_size*host.shape.ft_size).map(|i|0.002*((i%11) as f32-4.)).collect();
        host.l0w=&ft;
        let mut r=SfnnTrainStepRunner::new(&ctx,host,4,2).unwrap();
        r.factorizer_alpha.ft=0.5;
        r.optimizer_states.l0w.momentum.fill(&ctx,0.123).unwrap();
        r.optimizer_states.l0w.velocity.fill(&ctx,0.456).unwrap();
        let mut slow:Vec<_>=ft.iter().map(|v|v*0.7).collect();
        r.optimizer_states.l0w.slow_params.upload(&ctx,&slow).unwrap();
        fold_rows(&mut slow,BASE,VR,host.shape.ft_size,0.5).unwrap();
        let other=r.weights.l1w.download(&ctx).unwrap();
        let shape=SfnnForwardShape { input_size:BASE,..host.shape };
        let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,shape).unwrap();
        r.build_quantized_proxy(&ctx,BASE,VR,&proxy).unwrap();
        let before=proxy.l0w.download(&ctx).unwrap();
        r.disable_ft_factorizer(&ctx,BASE,VR).unwrap();
        assert_eq!(r.shape,shape);
        assert_eq!(r.optimizer_states.l0w.slow_params.download(&ctx).unwrap(),slow);
        assert!(r.optimizer_states.l0w.momentum.download(&ctx).unwrap().iter().all(|&v|v==0.123));
        assert!(r.optimizer_states.l0w.velocity.download(&ctx).unwrap().iter().all(|&v|v==0.456));
        assert_eq!(r.weights.l1w.download(&ctx).unwrap(),other);
        r.build_quantized_proxy(&ctx,BASE,VR,&proxy).unwrap();
        assert_eq!(proxy.l0w.download(&ctx).unwrap(),before);
        r.disable_ft_factorizer(&ctx,BASE,VR).unwrap(); // no double fold
        let lr=SfnnLayerLrMultipliers {qat_ft:true,qat_l1:true,qat_l2:true,qat_l3:true,qat_ft_virtual_rows:0,..Default::default()};
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,RangerUpdateParams::default(),ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,
            SfnnTrainStepHostBatch {stm_indices:&[0,BASE as i32,1,BASE as i32+1,2,BASE as i32+2,3,BASE as i32+3],
                nstm_indices:&[3,BASE as i32+3,2,BASE as i32+2,1,BASE as i32+1,0,BASE as i32],
                buckets:&[0,1,0,1],targets:&[0.2,0.8,0.3,0.7],entry_weights:&[1.;4],batch_size:4,max_active:2},
            true,true,lr,None).unwrap();
        assert!(r.forward_workspace.download_output(&ctx).unwrap().iter().all(|v|v.is_finite()));
    }
}
