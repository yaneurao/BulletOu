//! Optional L1 optimizer input centering. Forward/export remain folded affine.
//! The same global mean is used for residual and shared weights, not per-bucket means.
use super::*;
use experimental_output_center::{mean_into, scale_into};

pub(super) fn validate_effective_clip(r: &SfnnTrainStepRunner, lr: SfnnLayerLrMultipliers) -> Result<()> {
    if r.shape.has_compact_l1() || r.residual_count_gates_enabled || r.factorizer_axis_confidences_enabled
        || lr.update_scope != SfnnUpdateScope::All {
        return Err(CudaCppError::message("L1 effective weight clip requires dense L1, no count gates or axis confidences, update-scope=all"));
    }
    Ok(())
}

pub(super) fn clip_effective_weights(r: &SfnnTrainStepRunner, ctx: &Context, sgd: bool) -> Result<()> {
    let (shared, slow) = if r.factorizer.shared {
        (r.weights.l1fw.as_ref().unwrap().as_ptr(), r.optimizer_states.l1fw.as_ref().unwrap().slow_params.as_ptr())
    } else { (std::ptr::null_mut(), std::ptr::null_mut()) };
    let remainder = if sgd { r.optimizer_states.l1w.velocity.as_ptr() } else { std::ptr::null_mut() };
    if r.factorizer.any_axis() {
        let f=r.factorizer; let a=r.factorizer_alpha; let shape=r.shape;
        return check(unsafe { ffi::bulletou_clip_effective_l1_axis(ctx.as_ptr(),r.weights.l1w.as_ptr(),
            r.optimizer_states.l1w.slow_params.as_ptr(),shared,slow,
            r.weights.l1axw.as_ref().unwrap().as_ptr(),r.optimizer_states.l1axw.as_ref().unwrap().slow_params.as_ptr(),remainder,
            shape.ft_size,shape.l1_out(),shape.num_stacks,shape.factorizer_king_axis_dim,shape.factorizer_hand_axis_dim,
            f.king_axis as i32,f.hand_axis as i32,f.progress_axis as i32,f.king_hand_pair as i32,
            f.king_progress_pair as i32,f.hand_progress_pair as i32,
            a.shared,a.king_axis,a.hand_axis,a.progress_axis,a.pair) });
    }
    check(unsafe { ffi::bulletou_clip_effective_l1(ctx.as_ptr(), r.weights.l1w.as_ptr(),
        r.optimizer_states.l1w.slow_params.as_ptr(), shared, slow,remainder,
        r.shape.ft_size, r.shape.l1_out(), r.factorizer_alpha.shared) })
}

pub(super) fn accumulate(r: &mut SfnnTrainStepRunner, ctx: &Context, enabled: bool) -> Result<()> {
    if !enabled {
        r.l1_center_batches = 0;
        return Ok(());
    }
    let d = r.shape.ft_size;
    if r.l1_center_buffers.is_none() {
        r.l1_center_buffers = Some((F32Buffer::new(ctx, 32 * d)?, F32Buffer::new(ctx, d)?));
    }
    let (c, sum) = r.l1_center_buffers.as_ref().unwrap();
    if r.l1_center_batches == 1 {
        scale_into(ctx, c, sum, d, 1.0)?;
    }
    mean_into(ctx, &r.forward_workspace.combined, r.batch_size, d, c)?;
    if r.l1_center_batches > 0 {
        axpy_device(ctx, d, 1.0, c, sum, sum)?;
    }
    r.l1_center_batches += 1;
    Ok(())
}

pub(super) fn prepare(
    r: &SfnnTrainStepRunner,
    ctx: &Context,
    p: RangerUpdateParams,
    lr: SfnnLayerLrMultipliers,
) -> Result<()> {
    if !lr.l1_center {
        return Ok(());
    }
    if r.shape.has_compact_l1()
        || r.factorizer.any_axis()
        || r.residual_count_gates_enabled
        || lr.update_scope != SfnnUpdateScope::All
        || lr.tatara_weight_clip
        || p.radam.decay != 0.0
        || lr.norm_loss_strength != 0.0
        || lr.saturation_penalty != 0.0
        || lr.factorizer_residual_decay != 0.0
        || p.radam.min_weight > -1e20
        || p.radam.max_weight < 1e20
        || std::env::var_os("BULLETOU_EXPERIMENT_L1_CENTER").is_some()
        || std::env::var_os("BULLETOU_EXPERIMENT_L1_PROJECT_UPDATE").is_some()
    {
        return Err(CudaCppError::message("L1 centering requires dense L1, none/shared, update-scope=all, no gates/clip/penalties/legacy L1 experiments"));
    }
    let (c, sum) = r
        .l1_center_buffers
        .as_ref()
        .ok_or_else(|| CudaCppError::message("L1 centering requires accumulated input means before update"))?;
    if r.l1_center_batches == 0 {
        return Err(CudaCppError::message("L1 centering has no accumulated batches"));
    }
    if r.l1_center_batches > 1 {
        scale_into(ctx, sum, c, r.shape.ft_size, 1.0 / r.l1_center_batches as f32)?;
    }
    Ok(())
}

pub(super) fn transform(r: &SfnnTrainStepRunner, ctx: &Context, before: bool) -> Result<()> {
    let c = &r.l1_center_buffers.as_ref().unwrap().0;
    let shared = |v: Option<&F32Buffer>| if r.factorizer.shared { v.unwrap().as_ptr() } else { std::ptr::null_mut() };
    check(unsafe {
        ffi::bulletou_center_l1_groups(
            ctx.as_ptr(),
            r.weights.l1w.as_ptr(),
            r.weights.l1b.as_ptr(),
            r.optimizer_states.l1w.slow_params.as_ptr(),
            r.optimizer_states.l1b.slow_params.as_ptr(),
            r.backward_workspace.l1w_gradients.as_ptr(),
            r.backward_workspace.l1b_gradients.as_ptr(),
            shared(r.weights.l1fw.as_ref()),
            shared(r.weights.l1fb.as_ref()),
            shared(r.optimizer_states.l1fw.as_ref().map(|s| &s.slow_params)),
            shared(r.optimizer_states.l1fb.as_ref().map(|s| &s.slow_params)),
            shared(Some(&r.backward_workspace.l1fw_gradients)),
            shared(Some(&r.backward_workspace.l1fb_gradients)),
            c.as_ptr(),
            r.shape.ft_size,
            before as i32,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA"]
    fn effective_clip_pair_projects_fast_and_slow_without_changing_factors() {
        let ctx=Context::new(0).unwrap();
        let (stacks,cols,outputs)=(36usize,4usize,3usize);
        let cells=cols*outputs; let axes=46;
        let initial:Vec<f32>=(0..stacks*cells).map(|i|((i*17%31) as f32-15.0)*0.18).collect();
        let slow_initial:Vec<f32>=initial.iter().map(|x|-x*0.8).collect();
        let shared_values:Vec<f32>=(0..cells).map(|i|(i as f32-5.0)*0.13).collect();
        let axis_values:Vec<f32>=(0..axes*cells).map(|i|((i*7%23) as f32-11.0)*0.07).collect();
        let slow_shared:Vec<f32>=shared_values.iter().map(|x|x*0.5).collect();
        let slow_axis:Vec<f32>=axis_values.iter().map(|x|-x*0.6).collect();
        let w=F32Buffer::from_host(&ctx,&initial).unwrap();
        let slow=F32Buffer::from_host(&ctx,&slow_initial).unwrap();
        let sh=F32Buffer::from_host(&ctx,&shared_values).unwrap();
        let ss=F32Buffer::from_host(&ctx,&slow_shared).unwrap();
        let ax=F32Buffer::from_host(&ctx,&axis_values).unwrap();
        let sa=F32Buffer::from_host(&ctx,&slow_axis).unwrap();
        let remainder=F32Buffer::from_host(&ctx,&vec![0.25;stacks*cells]).unwrap();
        check(unsafe{ffi::bulletou_clip_effective_l1_axis(ctx.as_ptr(),w.as_ptr(),slow.as_ptr(),
            sh.as_ptr(),ss.as_ptr(),ax.as_ptr(),sa.as_ptr(),remainder.as_ptr(),cols,outputs,stacks,3,0,
            1,0,1,0,1,0,0.7,0.5,1.0,0.9,0.4)}).unwrap();
        let remains=remainder.download(&ctx).unwrap();
        for (result,base,shared,axis) in [(w.download(&ctx).unwrap(),&initial,&shared_values,&axis_values),
            (slow.download(&ctx).unwrap(),&slow_initial,&slow_shared,&slow_axis)] {
            for stack in 0..stacks {
                let k=stack/4; let p=stack%4; let ids=[k/3,3+k%3,6+p*9+k,42+p];
                let alpha=[0.5,0.5,0.4,0.9];
                for out in 0..outputs { for input in 0..cols {
                    let idx=stack*cells+out*cols+input; let f=input*outputs+out;
                    let effective=|v:f32| {let mut x=v+0.7*shared[f];
                        for j in 0..4 {x+=alpha[j]*axis[ids[j]*cells+f];} x};
                    let original=effective(base[idx]); let actual=effective(result[idx]);
                    assert!((actual-original.clamp(-2.0,127.0/64.0)).abs()<1e-5,
                        "stack={stack} out={out} input={input} original={original} actual={actual}");
                    if (-2.0..=127.0/64.0).contains(&original) {assert_eq!(result[idx],base[idx]);}
                    if std::ptr::eq(base,&initial) {
                        assert_eq!(remains[idx], if (-2.0..=127.0/64.0).contains(&original) {0.25} else {0.0});
                    }
                }}
            }
        }
        assert_eq!(sh.download(&ctx).unwrap(),shared_values);
        assert_eq!(ax.download(&ctx).unwrap(),axis_values);
    }
    #[test]
    #[ignore = "requires CUDA"]
    fn effective_clip_fast_slow_shared_alpha_and_noop() {
        let ctx=Context::new(0).unwrap();
        for enabled_shared in [false,true] {
            for alpha in [0.0,0.5,2.0] {
                let values=vec![-4.0,0.125,4.0,0.0,3.0,-3.0,1.0,-1.0];
                let shared_values=vec![0.25,-0.5,0.75,0.125];
                let w=F32Buffer::from_host(&ctx,&values).unwrap();
                let slow=F32Buffer::from_host(&ctx,&values.iter().map(|x|-x).collect::<Vec<_>>()).unwrap();
                let shared=F32Buffer::from_host(&ctx,&shared_values).unwrap();
                let ss=F32Buffer::from_host(&ctx,&shared_values.iter().map(|x|2.0*x).collect::<Vec<_>>()).unwrap();
                let ptr=|x:&F32Buffer| if enabled_shared{x.as_ptr()}else{std::ptr::null_mut()};
                check(unsafe{ffi::bulletou_clip_effective_l1(ctx.as_ptr(),w.as_ptr(),slow.as_ptr(),ptr(&shared),ptr(&ss),std::ptr::null_mut(),2,2,alpha)}).unwrap();
                for (got,sign,factor) in [(w.download(&ctx).unwrap(),1.0,1.0),(slow.download(&ctx).unwrap(),-1.0,2.0)] {
                    for i in 0..8 {
                        let f=if enabled_shared{alpha*factor*shared_values[(i%2)*2+(i/2)%2]}else{0.0};
                        assert_eq!(got[i]+f,(sign*values[i]+f).clamp(-2.0,127.0/64.0));
                        if (-2.0..=127.0/64.0).contains(&(sign*values[i]+f)){assert_eq!(got[i],sign*values[i]);}
                    }
                }
                assert_eq!(shared.download(&ctx).unwrap(),shared_values);
            }
        }
    }
    #[test]
    fn fused_groups_match_independent_host_reference() {
        let ctx = Context::new(0).unwrap();
        let shape = crate::tests::tiny_sfnn_shape();
        let w = crate::tests::tiny_sfnn_weights(shape);
        for shared in [false, true] {
            let mut r = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
            let mut reference = SfnnTrainStepRunner::new(&ctx, w, 4, 1).unwrap();
            r.factorizer.shared = shared;
            reference.factorizer.shared = shared;
            let c: Vec<f32> = (0..shape.ft_size).map(|j| 0.2 + 0.07 * j as f32).collect();
            r.l1_center_buffers =
                Some((F32Buffer::from_host(&ctx, &c).unwrap(), F32Buffer::new(&ctx, shape.ft_size).unwrap()));
            for runner in [&r, &reference] {
                runner.backward_workspace.l1w_gradients.fill(&ctx, 0.04).unwrap();
                runner.backward_workspace.l1b_gradients.fill(&ctx, 0.03).unwrap();
                runner.backward_workspace.l1fw_gradients.fill(&ctx, 0.02).unwrap();
                runner.backward_workspace.l1fb_gradients.fill(&ctx, 0.01).unwrap();
            }
            for before in [true, false] {
                transform(&r, &ctx, before).unwrap();
                experimental_l1_center::transform(&reference, &ctx, &c, before).unwrap();
                for (a, b) in [
                    (&r.weights.l1b, &reference.weights.l1b),
                    (&r.optimizer_states.l1b.slow_params, &reference.optimizer_states.l1b.slow_params),
                    (&r.backward_workspace.l1w_gradients, &reference.backward_workspace.l1w_gradients),
                    (r.weights.l1fb.as_ref().unwrap(), reference.weights.l1fb.as_ref().unwrap()),
                    (&r.backward_workspace.l1fw_gradients, &reference.backward_workspace.l1fw_gradients),
                ] {
                    for (x, y) in a.download(&ctx).unwrap().iter().zip(b.download(&ctx).unwrap()) {
                        assert!((x - y).abs() < 1e-6);
                    }
                }
            }
        }
    }
}
