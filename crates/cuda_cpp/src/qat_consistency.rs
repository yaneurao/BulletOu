//! Stop-gradient FP32 teacher on the current training batch; no extra activation workspace.
use super::*;

#[derive(Debug, Default)]
pub(crate) struct State {
    teacher: Option<F32Buffer>,
    strength: f32,
    scale: f32,
    offset: f32,
}

pub(crate) fn prepare(
    r: &mut SfnnTrainStepRunner,
    ctx: &Context,
    batch: SfnnTrainStepHostBatch<'_>,
    lr: SfnnLayerLrMultipliers,
    kind: ScalarLossKind,
    scale: f32,
) -> Result<()> {
    r.qat_consistency.strength = 0.0;
    lr.validate()?;
    if lr.qat_consistency == 0.0 {
        return Ok(());
    }
    if !(lr.qat_ft || lr.qat_l1 || lr.qat_l2 || lr.qat_l3) || r.batch_norm.is_some() {
        return Err(CudaCppError::message("QAT consistency requires non-BN SFNN with at least one QAT layer"));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(CudaCppError::message("QAT consistency requires a finite positive output scale"));
    }
    batch.validate()?;
    if batch.batch_size != r.batch_size || batch.max_active != r.max_active {
        return Err(CudaCppError::message("QAT consistency batch layout mismatch"));
    }
    r.device_batch.stm_indices.upload(ctx, batch.stm_indices)?;
    r.device_batch.nstm_indices.upload(ctx, batch.nstm_indices)?;
    r.device_batch.buckets.upload(ctx, batch.buckets)?;
    // Inference mode bypasses QAT but retains effective factors/count confidence.
    // Never use the FT gradient buffer as scratch: bpu may have pending gradients.
    sfnn_forward_device_with_factorizer_impl(
        ctx,
        &r.device_batch,
        &r.weights,
        &r.forward_workspace,
        r.factorizer,
        r.factorizer_alpha,
        None,
        r.residual_count_gates(),
        r.factorizer_axis_confidences(),
        false,
    )?;
    if r.qat_consistency.teacher.is_none() {
        r.qat_consistency.teacher = Some(F32Buffer::new(ctx, r.batch_size)?);
    }
    r.qat_consistency.teacher.as_ref().unwrap().copy_from(ctx, &r.forward_workspace.output)?;
    r.qat_consistency.scale = scale;
    r.qat_consistency.offset = match kind {
        ScalarLossKind::WinRateModel { in_offset_over_scaling, .. } => in_offset_over_scaling,
        _ => 0.0,
    };
    r.qat_consistency.strength = lr.qat_consistency;
    Ok(())
}

pub(crate) fn clear(r: &mut SfnnTrainStepRunner) {
    r.qat_consistency.strength = 0.0;
}

pub(crate) fn apply(r: &SfnnTrainStepRunner, ctx: &Context, entries: &F32Buffer) -> Result<()> {
    let s = &r.qat_consistency;
    if s.strength == 0.0 {
        return Ok(());
    }
    check(unsafe {
        bulletou_qat_consistency(
            ctx.as_ptr(),
            r.forward_workspace.output.as_ptr(),
            s.teacher.as_ref().unwrap().as_ptr(),
            entries.as_ptr(),
            r.loss_workspace.mean_output_gradients.as_ptr(),
            s.strength,
            s.scale,
            s.offset,
            r.batch_size,
        )
    })
}

unsafe extern "C" {
    fn bulletou_qat_consistency(
        ctx: *mut ffi::BulletOuCudaCppContext,
        student: *mut ffi::BulletOuCudaCppF32Buffer,
        teacher: *mut ffi::BulletOuCudaCppF32Buffer,
        entries: *mut ffi::BulletOuCudaCppF32Buffer,
        gradient: *mut ffi::BulletOuCudaCppF32Buffer,
        strength: f32,
        scale: f32,
        offset: f32,
        batch: usize,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn strength_validation() {
        for strength in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(SfnnLayerLrMultipliers { qat_consistency: strength, ..Default::default() }.validate().is_err());
        }
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn qat_consistency_gradient_matches_kl_finite_difference_and_respects_mask() {
        let ctx = Context::new(0).unwrap();
        let student_values = [0.2, -0.7, 2.0, f32::NAN];
        let teacher_values = [-0.4, 0.3, -1.0, f32::NAN];
        let mask = [1.0, 0.5, 2.0, 0.0];
        let student = F32Buffer::from_host(&ctx, &student_values).unwrap();
        let teacher = F32Buffer::from_host(&ctx, &teacher_values).unwrap();
        let entries = F32Buffer::from_host(&ctx, &mask).unwrap();
        let grad = F32Buffer::from_host(&ctx, &[0.13; 4]).unwrap();
        let strength = 0.7;
        let scale = 1.3;
        let offset = 0.2;
        check(unsafe {
            bulletou_qat_consistency(
                ctx.as_ptr(),
                student.as_ptr(),
                teacher.as_ptr(),
                entries.as_ptr(),
                grad.as_ptr(),
                strength,
                scale,
                offset,
                4,
            )
        })
        .unwrap();
        let result = grad.download(&ctx).unwrap();
        for i in 0..3 {
            let t = sigmoid(teacher_values[i] * scale - offset) as f64;
            let kl = |x: f64| {
                let p = 1.0 / (1.0 + (-(x * scale as f64 - offset as f64)).exp());
                (t * (t / p).ln() + (1.0 - t) * ((1.0 - t) / (1.0 - p)).ln()) * strength as f64 * mask[i] as f64 / 4.0
            };
            let x = student_values[i] as f64;
            let finite_difference = (kl(x + 1e-4) - kl(x - 1e-4)) / 2e-4;
            assert!((result[i] as f64 - 0.13 - finite_difference).abs() < 1e-6);
            assert!(kl(x - 0.1 * finite_difference) < kl(x));
        }
        assert_eq!(result[3], 0.13);
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn qat_consistency_teacher_and_gradients_match_across_step_paths_and_bpu() {
        let ctx = Context::new(0).unwrap();
        let upload = Context::new(0).unwrap();
        let mut w = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        w.l2fw = None;
        w.l2fb = None;
        w.l3fw = None;
        w.l3fb = None;
        for all_layers in [false, true] {
            for path in 0..3 {
                for kind in [
                    ScalarLossKind::BceWithLogits,
                    ScalarLossKind::BceErrorWeighted { k: 0.5 },
                    ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
                    ScalarLossKind::WinRateModel { pow_exp: 2.0, in_offset_over_scaling: 0.2 },
                ] {
                    let mut r = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
                    let mut control = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
                    let policy = SfnnLayerLrMultipliers {
                        qat_l1: true,
                        qat_ft: all_layers,
                        qat_l2: all_layers,
                        qat_l3: all_layers,
                        qat_consistency: 0.7,
                        ..Default::default()
                    };
                    let offset = match kind {
                        ScalarLossKind::WinRateModel { in_offset_over_scaling, .. } => in_offset_over_scaling,
                        _ => 0.0,
                    };
                    let scale = 1.3;
                    for iteration in 0..2 {
                        let stm = if iteration == 0 { [0, 1, 2, 3] } else { [3, 2, 1, 0] };
                        let batch = SfnnTrainStepHostBatch {
                            stm_indices: &stm,
                            nstm_indices: &[2, 3, 0, 1],
                            buckets: &[0, 1, 0, 1],
                            targets: &[0.2, 0.8, 0.3, 0.7],
                            entry_weights: &[1.0, 0.5, 0.0, 1.0],
                            batch_size: 4,
                            max_active: 1,
                        };
                        // Reference teacher: ordinary FP32 inference with original masters.
                        r.device_batch.stm_indices.upload(&ctx, batch.stm_indices).unwrap();
                        r.device_batch.nstm_indices.upload(&ctx, batch.nstm_indices).unwrap();
                        r.device_batch.buckets.upload(&ctx, batch.buckets).unwrap();
                        sfnn_forward_device_with_factorizer_and_alpha(
                            &ctx,
                            &r.device_batch,
                            &r.weights,
                            &r.forward_workspace,
                            r.factorizer,
                            r.factorizer_alpha,
                        )
                        .unwrap();
                        let teacher = r.forward_workspace.output.download(&ctx).unwrap();
                        control
                            .step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                                &ctx,
                                RangerUpdateParams::default(),
                                kind,
                                scale,
                                batch,
                                true,
                                false,
                                SfnnLayerLrMultipliers { qat_consistency: 0.0, ..policy },
                                None,
                            )
                            .unwrap();
                        match path {
                            0 => r
                                .step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                                    &ctx,
                                    RangerUpdateParams::default(),
                                    kind,
                                    scale,
                                    batch,
                                    true,
                                    false,
                                    policy,
                                    None,
                                )
                                .unwrap(),
                            1 => r
                                .step_pipelined_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                                    &ctx,
                                    &upload,
                                    RangerUpdateParams::default(),
                                    kind,
                                    scale,
                                    batch,
                                    false,
                                    false,
                                    policy,
                                    None,
                                )
                                .unwrap(),
                            _ => {
                                r.step_profiled_no_readback_with_update_lr_multipliers_and_dirty_buckets(
                                    &ctx,
                                    RangerUpdateParams::default(),
                                    kind,
                                    scale,
                                    batch,
                                    false,
                                    policy,
                                    None,
                                )
                                .unwrap();
                            }
                        }
                        let saved = r.qat_consistency.teacher.as_ref().unwrap().download(&ctx).unwrap();
                        let student = r.forward_workspace.output.download(&ctx).unwrap();
                        let gradient = r.loss_workspace.mean_output_gradients.download(&ctx).unwrap();
                        let base = control.loss_workspace.mean_output_gradients.download(&ctx).unwrap();
                        assert_eq!(saved, teacher);
                        assert_eq!(student, control.forward_workspace.output.download(&ctx).unwrap());
                        for i in 0..4 {
                            let expected = base[i]
                                + 0.7
                                    * batch.entry_weights[i]
                                    * (sigmoid(student[i] * scale - offset) - sigmoid(teacher[i] * scale - offset))
                                    * scale
                                    / 4.0;
                            assert!(
                                (gradient[i] - expected).abs() < 2e-7,
                                "path={path} sample={i}: {} != {expected}",
                                gradient[i]
                            );
                        }
                        assert_eq!(r.weights.l0w.download(&ctx).unwrap(), w.l0w);
                        assert_eq!(r.weights.l3w.download(&ctx).unwrap(), w.l3w);
                        assert_eq!(r.qat_consistency.strength, 0.0);
                        assert_eq!(r.pending_gradient_batches, iteration + 1);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn qat_consistency_error_restore_update_and_disable() {
        let ctx = Context::new(0).unwrap();
        let mut w = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        w.l2fw = None;
        w.l2fb = None;
        w.l3fw = None;
        w.l3fb = None;
        let mut r = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
        let batch = SfnnTrainStepHostBatch {
            stm_indices: &[0, 1, 2, 3],
            nstm_indices: &[2, 3, 0, 1],
            buckets: &[0, 1, 0, 1],
            targets: &[0.2, 0.8, 0.3, 0.7],
            entry_weights: &[1.0; 4],
            batch_size: 4,
            max_active: 1,
        };
        let policy = SfnnLayerLrMultipliers {
            qat_l1: true,
            qat_ft: true,
            qat_l2: true,
            qat_l3: true,
            qat_consistency: 0.5,
            ..Default::default()
        };
        let mut params = RangerUpdateParams::default();
        params.radam.learning_rate = 0.01;
        let fail: Result<()> = layer_qat::step(
            &mut r,
            &ctx,
            params,
            policy,
            false,
            None,
            batch,
            ScalarLossKind::BceWithLogits,
            1.0,
            |_, _| Err(CudaCppError::message("injected error")),
        );
        assert!(fail.is_err());
        assert_eq!(r.qat_consistency.strength, 0.0);
        assert_eq!(r.pending_gradient_batches, 0);
        assert_eq!(r.weights.l0w.download(&ctx).unwrap(), w.l0w);
        assert_eq!(r.weights.l3w.download(&ctx).unwrap(), w.l3w);
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,
            params,
            ScalarLossKind::BceWithLogits,
            1.0,
            batch,
            true,
            true,
            policy,
            None,
        )
        .unwrap();
        assert_eq!(r.pending_gradient_batches, 0);
        assert_eq!(r.qat_consistency.strength, 0.0);
        assert_ne!(r.weights.l3w.download(&ctx).unwrap(), w.l3w);
        // Turning consistency off must not reuse the last teacher or gradient.
        let off = SfnnLayerLrMultipliers { qat_consistency: 0.0, ..policy };
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,
            params,
            ScalarLossKind::BceWithLogits,
            1.0,
            batch,
            true,
            false,
            off,
            None,
        )
        .unwrap();
        let outputs = r.forward_workspace.output.download(&ctx).unwrap();
        let gradients = r.loss_workspace.mean_output_gradients.download(&ctx).unwrap();
        for i in 0..4 {
            assert!((gradients[i] - (sigmoid(outputs[i]) - batch.targets[i]) / 4.0).abs() < 1e-6);
        }
    }
    #[test]
    #[ignore = "requires a CUDA-capable NVIDIA GPU"]
    fn qat_consistency_folded_ft_teacher_uses_both_real_and_virtual_features() {
        let ctx = Context::new(0).unwrap();
        let mut w = crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        w.l2fw = None;
        w.l2fb = None;
        w.l3fw = None;
        w.l3fb = None;
        const BASE: usize = 131949;
        const VR: usize = 1629;
        w.shape.input_size = BASE + VR;
        let ft: Vec<f32> = (0..w.shape.input_size * w.shape.ft_size).map(|i| 0.02 * ((i % 11) as f32 - 4.0)).collect();
        w.l0w = &ft;
        let mut r = SfnnTrainStepRunner::new(&ctx, w, 4, 2).unwrap();
        r.factorizer_alpha.ft = 0.7;
        r.factorizer_alpha.shared = 0.6;
        let stm = [0, BASE as i32, 1629, BASE as i32, 1, (BASE + 1) as i32, 1630, (BASE + 1) as i32];
        let nstm = [2, (BASE + 2) as i32, 1631, (BASE + 2) as i32, 3, (BASE + 3) as i32, 1632, (BASE + 3) as i32];
        let batch = SfnnTrainStepHostBatch {
            stm_indices: &stm,
            nstm_indices: &nstm,
            buckets: &[0, 1, 0, 1],
            targets: &[0.2, 0.8, 0.3, 0.7],
            entry_weights: &[1.0; 4],
            batch_size: 4,
            max_active: 2,
        };
        r.device_batch.stm_indices.upload(&ctx, &stm).unwrap();
        r.device_batch.nstm_indices.upload(&ctx, &nstm).unwrap();
        r.device_batch.buckets.upload(&ctx, batch.buckets).unwrap();
        sfnn_forward_device_with_factorizer_and_alpha(
            &ctx,
            &r.device_batch,
            &r.weights,
            &r.forward_workspace,
            r.factorizer,
            r.factorizer_alpha,
        )
        .unwrap();
        let expected = r.forward_workspace.output.download(&ctx).unwrap();
        let policy = SfnnLayerLrMultipliers {
            qat_ft: true,
            qat_l1: true,
            qat_l2: true,
            qat_l3: true,
            qat_ft_virtual_rows: VR,
            qat_consistency: 0.5,
            ..Default::default()
        };
        // Repeat to exercise cached QAT weights while the FP32 teacher is recalculated.
        for _ in 0..2 {
            r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                &ctx,
                RangerUpdateParams::default(),
                ScalarLossKind::BceWithLogits,
                1.0,
                batch,
                true,
                false,
                policy,
                None,
            )
            .unwrap();
            assert_eq!(r.qat_consistency.teacher.as_ref().unwrap().download(&ctx).unwrap(), expected);
            assert_eq!(r.weights.l0w.download(&ctx).unwrap(), ft);
        }
        let gradients = r.backward_workspace.l0w_gradients.download(&ctx).unwrap();
        for piece in 0..4 {
            for unit in 0..w.shape.ft_size {
                let sum: f32 = (0..81).map(|k| gradients[(k * VR + piece) * w.shape.ft_size + unit]).sum();
                assert!((gradients[(BASE + piece) * w.shape.ft_size + unit] - 0.7 * sum).abs() < 1e-6);
            }
        }
    }
}
