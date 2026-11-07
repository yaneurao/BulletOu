use super::super::*;

#[test]
fn norm_loss_scope() {
    let policy = SfnnLayerLrMultipliers { norm_loss_strength: 1e-4, ..Default::default() };
    for layer in [SfnnUpdateLayer::L0, SfnnUpdateLayer::L1, SfnnUpdateLayer::L2, SfnnUpdateLayer::L3] {
        for kind in [SfnnUpdateParamKind::Weight, SfnnUpdateParamKind::Bias] {
            let p = policy.clip_params(RangerUpdateParams::default(), layer, kind);
            let excluded = layer == SfnnUpdateLayer::L0 && kind == SfnnUpdateParamKind::Weight;
            assert_eq!(p.norm_loss_strength, if excluded { 0.0 } else { 1e-4 });
            assert_eq!(p.norm_loss_before, layer == SfnnUpdateLayer::L3 && kind == SfnnUpdateParamKind::Bias);
        }
    }
    for strength in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(SfnnLayerLrMultipliers { norm_loss_strength: strength, ..Default::default() }.validate().is_err());
    }
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn norm_loss_gpu_matches_reference() {
    let ctx = Context::new(0).unwrap();
    for stride in [3, 4] {
        for before in [false, true] {
            for dirty in [false, true] {
                for step in [1, 6] {
                    for strength in [0.0, 0.1] {
                        let len = 2 * stride;
                        let initial: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { 0.1 } else { 2.0 }).collect();
                        let p = RangerUpdateParams {
                            norm_loss_strength: strength, norm_loss_before: before,
                            radam: RAdamUpdateParams { step, learning_rate: 0.1, decay: 0.0,
                                min_weight: -0.5, max_weight: 0.5, ..Default::default() },
                            clip_after_lookahead: false, ..Default::default()
                        };
                        let mut expected = initial.clone();
                        let t = step as f32;
                        let lr = if before { 0.1 } else { 0.1 * (1.0-p.radam.beta2.powf(t)).sqrt()
                            / ((1.0-p.radam.beta1.powf(t))*((1.0+p.radam.beta2).powi(2)+p.radam.beta2.powi(2)).sqrt()) };
                        let apply = |values: &mut Vec<f32>| {
                            let norm = values.iter().map(|v| v*v).sum::<f32>().sqrt();
                            let scale = 1.0 - 2.0*strength*lr*(1.0-1.0/(norm+1e-7));
                            for v in values { *v *= scale; }
                        };
                        if before { apply(&mut expected); }
                        for (i, w) in expected.iter_mut().enumerate() {
                            if !dirty || i >= stride { *w = w.clamp(-0.5, 0.5); }
                        }
                        if !before { apply(&mut expected); }
                        if step == 6 {
                            for (i, w) in expected.iter_mut().enumerate() {
                                if !dirty || i >= stride { *w = 0.5 * (*w + initial[i]); }
                            }
                        }
                        let w = F32Buffer::from_host(&ctx, &initial).unwrap();
                        let g = F32Buffer::from_host(&ctx, &vec![0.0; len]).unwrap();
                        let m = F32Buffer::from_host(&ctx, &vec![0.0; len]).unwrap();
                        let v = F32Buffer::from_host(&ctx, &vec![0.0; len]).unwrap();
                        let s = F32Buffer::from_host(&ctx, &initial).unwrap();
                        if dirty {
                            let buckets = I32Buffer::from_host(&ctx, &[1]).unwrap();
                            ranger_update_stacked_dirty_device(&ctx, p, RangerStackedDirtyDeviceStateMut {
                                gradients: &g, weights: &w, momentum: &m, velocity: &v, slow_params: &s,
                                dirty_buckets: &buckets, dirty_count: 1, stride,
                            }).unwrap();
                        } else {
                            ranger_update_device(&ctx, p, RangerDeviceStateMut {
                                gradients: &g, weights: &w, momentum: &m, velocity: &v, slow_params: &s,
                            }).unwrap();
                        }
                        for (got, want) in w.download(&ctx).unwrap().iter().zip(&expected) {
                            assert!((got-want).abs() < 2e-6, "{got} != {want}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn norm_loss_gpu_handles_small_and_large_tensor_boundaries() {
    let ctx = Context::new(0).unwrap();
    // Exercise both paths and reuse the same context after a large tensor.
    for len in [65_537usize, 1, 255, 256, 257, 1_536, 2_048, 2_049, 3] {
        let initial: Vec<f32> = (0..len).map(|i| (i % 29) as f32 * 0.01 + 0.1).collect();
        let norm = initial.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt() as f32;
        let scale = 1.0 - 2.0 * 0.1 * 0.01 * (1.0 - 1.0 / (norm + 1.0e-7));
        let p = RangerUpdateParams {
            norm_loss_strength: 0.1, norm_loss_before: true,
            radam: RAdamUpdateParams { step: 1, learning_rate: 0.01, decay: 0.0,
                min_weight: -10.0, max_weight: 10.0, ..Default::default() },
            ..Default::default()
        };
        let w = F32Buffer::from_host(&ctx, &initial).unwrap();
        let g = F32Buffer::from_host(&ctx, &vec![0.0; len]).unwrap();
        let m = F32Buffer::from_host(&ctx, &vec![0.0; len]).unwrap();
        let v = F32Buffer::from_host(&ctx, &vec![0.0; len]).unwrap();
        let s = F32Buffer::from_host(&ctx, &initial).unwrap();
        ranger_update_device(&ctx, p, RangerDeviceStateMut {
            gradients: &g, weights: &w, momentum: &m, velocity: &v, slow_params: &s,
        }).unwrap();
        for (&got, &value) in w.download(&ctx).unwrap().iter().zip(&initial) {
            assert!((got - value * scale).abs() < 2.0e-6, "len={len}: {got} != {}", value * scale);
        }
    }
}
