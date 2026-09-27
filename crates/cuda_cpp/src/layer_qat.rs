//! Non-BN, selected-layer weight/bias fake quantization. Identity STE.
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
mod tests {
    use super::*;
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
        let expected = fw.download_output(&ctx).unwrap();
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

impl State {
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
            prepare(ctx, &mut q.l2, &r.weights.l2w, &r.weights.l2b, lr.qat_l2, false, 0, 0, 1, 1.0)?;
            prepare(ctx, &mut q.l3, &r.weights.l3w, &r.weights.l3b, lr.qat_l3, false, 0, 0, 1, 1.0)?;
            q.signature = Some(signature);
        }
        q.swap(&mut r.weights);
        let inner = SfnnLayerLrMultipliers { qat_ft: false, qat_l2: false, qat_l3: false, ..lr };
        let value = run(r, inner);
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
