//! Compact L1 revival: structural zeros, optimizer isolation and dense export parity.
use super::*;

fn shape(common: usize, shard: usize, groups: usize) -> SfnnForwardShape {
    SfnnForwardShape {
        input_size: 4,
        ft_size: if shard > 0 { common + shard * groups } else { 8 },
        l1_hidden: 3,
        l1_skip: true,
        l2_size: 2,
        num_stacks: 2,
        l1_group_count: groups,
        l1_common_size: common,
        l1_shard_size: shard,
        factorizer_king_axis_dim: 0,
        factorizer_hand_axis_dim: 0,
        factorizer_progress_axis: false,
        factorizer_king_hand_pair: false,
        factorizer_king_progress_pair: false,
        factorizer_hand_progress_pair: false,
    }
}

fn cases() -> [SfnnForwardShape; 5] {
    [
        shape(64, 64, 2),
        shape(0, 64, 2),
        shape(0, 0, 2),
        shape(0, 64, 4),
        SfnnForwardShape {
            ft_size: 3072,
            l1_hidden: 15,
            l2_size: 64,
            l1_common_size: 1024,
            l1_shard_size: 1024,
            ..shape(64, 64, 2)
        },
    ]
}

fn runner(ctx: &Context, shape: SfnnForwardShape) -> SfnnTrainStepRunner {
    let l0w: Vec<f32> = (0..shape.input_size * shape.ft_size).map(|i| 0.03125 * (i % 5) as f32).collect();
    let l0b = vec![0.125; shape.ft_size];
    let l1w: Vec<f32> = (0..shape.l1w_len().unwrap()).map(|i| 0.02 + 0.001 * (i % 101) as f32).collect();
    let l1b = vec![0.03; shape.num_stacks * shape.l1_out()];
    let l2w = vec![0.1; shape.num_stacks * shape.l2_size * shape.l2_in()];
    let l2b = vec![0.02; shape.num_stacks * shape.l2_size];
    let l3w = vec![0.2; shape.num_stacks * shape.l2_size];
    let l3b = vec![0.03; shape.num_stacks];
    let weights = SfnnForwardHostWeights {
        shape,
        l0w: &l0w,
        l0b: &l0b,
        l1w: &l1w,
        l1b: &l1b,
        l1fw: None,
        l1fb: None,
        l1axw: None,
        l1axb: None,
        l2w: &l2w,
        l2b: &l2b,
        l2fw: None,
        l2fb: None,
        l2axw: None,
        l2axb: None,
        l3w: &l3w,
        l3b: &l3b,
        l3fw: None,
        l3fb: None,
        l3axw: None,
        l3axb: None,
    };
    SfnnTrainStepRunner::new(ctx, weights, 4, 1).unwrap()
}

fn expand(s: SfnnForwardShape, compact: &[f32]) -> Vec<f32> {
    // Independent reference expansion: enumerate each physical row's explicit input list.
    let mut dense = vec![0.0; s.num_stacks * s.l1_out() * s.ft_size];
    for bucket in 0..s.num_stacks {
        for out in 0..s.l1_out() {
            let group = out / (s.l1_out() / s.l1_group_count);
            let inputs: Vec<usize> = if s.l1_shard_size != 0 {
                (0..s.l1_common_size)
                    .chain(s.l1_common_size + group * s.l1_shard_size..s.l1_common_size + (group + 1) * s.l1_shard_size)
                    .collect()
            } else if s.l1_group_count > 1 {
                (group * s.ft_size / s.l1_group_count..(group + 1) * s.ft_size / s.l1_group_count).collect()
            } else {
                (0..s.ft_size).collect()
            };
            for (local, &col) in inputs.iter().enumerate() {
                dense[(bucket * s.l1_out() + out) * s.ft_size + col] =
                    compact[(bucket * s.l1_out() + out) * inputs.len() + local];
            }
        }
    }
    dense
}

#[test]
fn compact_revival_connection_map_matches_expansion() {
    for s in [shape(64, 64, 2), shape(0, 64, 2), shape(0, 0, 2), shape(0, 0, 1)] {
        let compact: Vec<f32> = (0..s.l1w_len().unwrap()).map(|i| (i + 1) as f32).collect();
        let dense = expand(s, &compact);
        for b in 0..s.num_stacks {
            for out in 0..s.l1_out() {
                for col in 0..s.ft_size {
                    let value = revive_common::weight_index(s, b, out, col).map_or(0.0, |i| compact[i]);
                    assert_eq!(value, dense[(b * s.l1_out() + out) * s.ft_size + col]);
                }
                for local in 0..revive_common::row_width(s) {
                    let col = revive_common::input_column(s, out, local);
                    assert_eq!(
                        revive_common::weight_index(s, b, out, col),
                        Some((b * s.l1_out() + out) * revive_common::row_width(s) + local)
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn compact_revival_gpu_proxy_matches_export_and_preserves_structural_zeros() {
    let ctx = Context::new(0).unwrap();
    for s in cases() {
        let r = runner(&ctx, s);
        let proxy_shape = SfnnForwardShape { l1_group_count: 1, l1_common_size: 0, l1_shard_size: 0, ..s };
        let proxy = SfnnForwardDeviceWeights::new_dense(&ctx, proxy_shape).unwrap();
        r.build_quantized_proxy(&ctx, s.input_size, 0, &proxy).unwrap();
        let compact = r.weights.l1w.download(&ctx).unwrap();
        let expected: Vec<f32> =
            expand(s, &compact).into_iter().map(|x| (x * 64.0).round().clamp(-128.0, 127.0) / 64.0).collect();
        assert_eq!(proxy.l1w.download(&ctx).unwrap(), expected);
        assert_eq!(r.weights.l1w.download(&ctx).unwrap(), compact);
        // FT contribution must count only existing outgoing connections, including skip.
        let mut c = ft_revive::Calibration::new(s.ft_size, s.num_stacks);
        for mad in [false, true] {
            c.add(&[0, 1], &vec![0.25; 2 * s.ft_size], &[1.0; 2], mad).unwrap();
            c.add(&[0, 1], &vec![0.75; 2 * s.ft_size], &[1.0; 2], mad).unwrap();
            if !mad {
                c.finish_means();
            }
        }
        c.finish(&proxy.l1w.download(&ctx).unwrap(), s.l1_out()).unwrap();
        assert!(c.utility.iter().all(|x| *x > 0.0));
        // Quantization of only the weights does not alter the compact master layout.
        assert_eq!(r.weights.l1w.len(), s.l1w_len().unwrap());
    }
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn compact_revival_ft_resets_only_connected_columns_and_preserves_mean() {
    let ctx = Context::new(0).unwrap();
    for s in cases() {
        let mut r = runner(&ctx, s);
        r.optimizer_states.l1w.momentum.fill(&ctx, 0.123).unwrap();
        r.optimizer_states.l1w.velocity.fill(&ctx, 0.456).unwrap();
        r.optimizer_states.l1w.slow_params.fill(&ctx, 0.3).unwrap();
        r.optimizer_states.l1b.momentum.fill(&ctx, 0.11).unwrap();
        r.optimizer_states.l1b.velocity.fill(&ctx, 0.22).unwrap();
        let old = r.read_weights(&ctx).unwrap();
        let opt = r.read_optimizer_states(&ctx).unwrap();
        let mut c = ft_revive::Calibration::new(s.ft_size, s.num_stacks);
        c.counts.fill(1024);
        c.means.fill(0.25);
        let selected = if s.l1_group_count == 4 { vec![0] } else { vec![0, 2] };
        r.revive_ft(&ctx, &c, &selected, s.input_size).unwrap();
        c.means.fill(0.4);
        r.compensate_ft_revival_mean(&ctx, &selected, &c).unwrap();
        let new = r.read_weights(&ctx).unwrap();
        let now = r.read_optimizer_states(&ctx).unwrap();
        for row in 0..s.input_size {
            for col in 0..s.ft_size {
                let i = row * s.ft_size + col;
                if !selected.contains(&(col % (s.ft_size / 2))) {
                    assert_eq!(old.l0w[i], new.l0w[i]);
                }
            }
        }
        for bucket in 0..s.num_stacks {
            for out in 0..s.l1_out() {
                let row = bucket * s.l1_out() + out;
                let mut mean = old.l1b[row];
                let mut slow = opt.l1b.slow_params[row];
                let mut touched = false;
                for local in 0..revive_common::row_width(s) {
                    let col = revive_common::input_column(s, out, local);
                    let i = row * revive_common::row_width(s) + local;
                    if selected.contains(&(col % (s.ft_size / 2))) {
                        touched = true;
                        assert_eq!(new.l1w[i].abs(), 1.0 / 64.0);
                        assert_eq!(now.l1w.slow_params[i], new.l1w[i]);
                        assert_eq!(now.l1w.momentum[i], 0.0);
                        assert_eq!(now.l1w.velocity[i], 0.0);
                        mean += old.l1w[i] * 0.25 - new.l1w[i] * 0.4;
                        slow += opt.l1w.slow_params[i] * 0.25 - now.l1w.slow_params[i] * 0.4;
                    } else {
                        assert_eq!(new.l1w[i], old.l1w[i]);
                        assert_eq!(now.l1w.momentum[i], 0.123);
                        assert_eq!(now.l1w.velocity[i], 0.456);
                        assert_eq!(now.l1w.slow_params[i], 0.3);
                    }
                }
                assert!((new.l1b[row] - mean).abs() < 1e-6);
                assert!((now.l1b.slow_params[row] - slow).abs() < 1e-6);
                assert_eq!(now.l1b.momentum[row], if touched { 0.0 } else { 0.11 });
                assert_eq!(now.l1b.velocity[row], if touched { 0.0 } else { 0.22 });
            }
        }
        assert_eq!(old.l2w, new.l2w);
        assert_eq!(old.l3w, new.l3w);
    }
}

#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn compact_revival_l1_l2_mean_compensation_and_backward() {
    let ctx = Context::new(0).unwrap();
    for s in cases() {
        let mut r = runner(&ctx, s);
        r.optimizer_states.l1w.momentum.fill(&ctx, 0.123).unwrap();
        r.optimizer_states.l1w.velocity.fill(&ctx, 0.456).unwrap();
        let old = r.read_weights(&ctx).unwrap();
        let opt = r.read_optimizer_states(&ctx).unwrap();
        let mut c = l1_revive::Calibration::new(s.ft_size, s.l1_hidden, s.num_stacks);
        // A compact group-1 unit receives non-contiguous common + private inputs.
        let unit = s.l1_group_output();
        for _ in 0..1024 {
            c.add(&[0, 1], &vec![0.25; 2 * s.ft_size], &vec![0.5; 2 * s.l2_in()]).unwrap();
        }
        c.contribution.relative = vec![1.0; s.num_stacks * s.l1_hidden];
        c.contribution.relative[unit] = 0.0;
        assert_eq!(r.revive_l1_selected(&ctx, &c, true, false).unwrap(), vec![unit]);
        let new = r.read_weights(&ctx).unwrap();
        let now = r.read_optimizer_states(&ctx).unwrap();
        for bucket in 0..s.num_stacks {
            for out in 0..s.l1_out() {
                let row = bucket * s.l1_out() + out;
                let range = row * revive_common::row_width(s)..(row + 1) * revive_common::row_width(s);
                if bucket == 0 && out == unit {
                    let mean: f32 = new.l1w[range.clone()].iter().sum::<f32>() * 0.25 + new.l1b[row];
                    assert!((mean - 0.5).abs() < 1e-6);
                    assert_eq!(&new.l1w[range.clone()], &now.l1w.slow_params[range.clone()]);
                    assert!(now.l1w.momentum[range.clone()].iter().all(|&x| x == 0.0));
                } else {
                    assert_eq!(&new.l1w[range.clone()], &old.l1w[range.clone()]);
                    assert_eq!(&now.l1w.momentum[range.clone()], &opt.l1w.momentum[range]);
                    assert_eq!(new.l1b[row], old.l1b[row]);
                }
            }
        }
        // Replay measured branches at 0.3: compensate their new mean at L2.
        let mut replay = l1_revive::Calibration::new(s.ft_size, s.l1_hidden, s.num_stacks);
        for _ in 0..1024 {
            replay.add(&[0, 1], &vec![0.25; 2 * s.ft_size], &vec![0.3; 2 * s.l2_in()]).unwrap();
        }
        r.compensate_l1_revival_mean(&ctx, &[unit], &replay).unwrap();
        let compensated = r.read_weights(&ctx).unwrap();
        for out in 0..s.l2_size {
            let row = out;
            let expected = old.l2b[row]
                + 0.5 * (old.l2w[row * s.l2_in() + unit] + old.l2w[row * s.l2_in() + s.l1_hidden + unit])
                - 0.3
                    * (compensated.l2w[row * s.l2_in() + unit] + compensated.l2w[row * s.l2_in() + s.l1_hidden + unit]);
            assert!((compensated.l2b[row] - expected).abs() < 1e-6);
        }
        let upstream = r.read_weights(&ctx).unwrap().l1w;
        let mut c = l2_revive::Calibration::new(s.l2_in(), s.l2_size, s.num_stacks);
        c.counts.fill(1024);
        c.contribution.relative = vec![1.0; s.num_stacks * s.l2_size];
        c.contribution.relative[0] = 0.0;
        assert_eq!(r.revive_l2_selected(&ctx, &c, true, false).unwrap(), vec![0]);
        assert_eq!(r.read_weights(&ctx).unwrap().l1w, upstream);
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,
            RangerUpdateParams::default(),
            ScalarLossKind::BceWithLogits,
            1.0,
            SfnnTrainStepHostBatch {
                stm_indices: &[0, 1, 2, 3],
                nstm_indices: &[3, 2, 1, 0],
                buckets: &[0, 1, 0, 1],
                targets: &[0.2, 0.8, 0.3, 0.7],
                entry_weights: &[1.0; 4],
                batch_size: 4,
                max_active: 1,
            },
            true,
            true,
            SfnnLayerLrMultipliers { qat_ft: true, qat_l2: true, qat_l3: true, ..Default::default() },
            None,
        )
        .unwrap();
        let trained = r.read_weights(&ctx).unwrap();
        assert!(trained.l1w.iter().all(|x| x.is_finite()));
        assert_eq!(trained.l1w.len(), s.l1w_len().unwrap());
    }
}

#[test]
#[ignore = "requires CUDA; compact factors and QAT"]
fn compact_factorizer_switch_qat_and_gradients() {
    let ctx = Context::new(0).unwrap();
    for mut s in [shape(64, 64, 2), shape(0, 64, 2), shape(0, 0, 2)] {
        s.num_stacks = 8;
        s.factorizer_king_axis_dim = 2;
        let mut r = runner(&ctx, s);
        let shared = vec![0.002; s.ft_size * s.l1_out()];
        let bias = vec![0.003; s.l1_out()];
        r.weights.l1fw = Some(F32Buffer::from_host(&ctx, &shared).unwrap());
        r.weights.l1fb = Some(F32Buffer::from_host(&ctx, &bias).unwrap());
        r.optimizer_states.l1fw = Some(RangerParamState::from_host_weights(&ctx, &shared).unwrap());
        r.optimizer_states.l1fb = Some(RangerParamState::from_host_weights(&ctx, &bias).unwrap());
        r.set_factorizer_config(SfnnFactorizerActive::SHARED, SfnnFactorizerAlpha::ONE).unwrap();
        let hb = SfnnForwardHostBatch {
            stm_indices: &[0, 1, 2, 3],
            nstm_indices: &[3, 2, 1, 0],
            buckets: &[0, 1, 6, 7],
            batch_size: 4,
            max_active: 1,
        };
        let batch = SfnnForwardDeviceBatch::from_host(&ctx, hb).unwrap();
        for mode in ["pair", "axis", "shared", "pair", "shared"] {
            let old = r.shape;
            let active = r.factorizer;
            let before_fw = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(old, 4)).unwrap();
            r.forward_current_weights(&ctx, &batch, &before_fw).unwrap();
            let before = before_fw.download_output(&ctx).unwrap();
            let mut coefficients = vec![vec![]; 8];
            for (b, terms) in coefficients.iter_mut().enumerate() {
                if active.king_axis {
                    terms.extend([(b / 4, 1.0), (2 + (b / 2) % 2, 1.0)]);
                }
                if active.king_progress_pair {
                    terms.push((4 + (b % 2) * 4 + b / 2, 1.0));
                }
                if active.progress_axis {
                    terms.push((4 + old.factorizer_pair_count() + b % 2, 1.0));
                }
            }
            let active = if mode == "shared" {
                SfnnFactorizerActive::SHARED
            } else {
                SfnnFactorizerActive {
                    king_axis: true,
                    progress_axis: true,
                    king_progress_pair: mode == "pair",
                    ..SfnnFactorizerActive::SHARED
                }
            };
            let next = SfnnForwardShape {
                factorizer_progress_axis: active.progress_axis,
                factorizer_king_progress_pair: active.king_progress_pair,
                ..s
            };
            r.switch_l1_factorizer(&ctx, next, active, SfnnFactorizerAlpha::ONE, &coefficients).unwrap();
            let fw = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(next, 4)).unwrap();
            r.forward_current_weights(&ctx, &batch, &fw).unwrap();
            for (a, b) in before.iter().zip(fw.download_output(&ctx).unwrap()) {
                assert!((a - b).abs() < 1e-5, "switch {mode}: {a} {b}");
            }
            if active.any_axis() {
                r.weights.l1axw.as_ref().unwrap().fill(&ctx, 0.001).unwrap();
                r.weights.l1axb.as_ref().unwrap().fill(&ctx, 0.002).unwrap();
                r.optimizer_states.l1axw.as_ref().unwrap().slow_params.fill(&ctx, 0.001).unwrap();
                r.optimizer_states.l1axb.as_ref().unwrap().slow_params.fill(&ctx, 0.002).unwrap();
            }
            let ps = SfnnForwardShape {
                l1_group_count: 1,
                l1_common_size: 0,
                l1_shard_size: 0,
                factorizer_king_axis_dim: 0,
                factorizer_progress_axis: false,
                factorizer_king_progress_pair: false,
                ..next
            };
            let proxy = SfnnForwardDeviceWeights::new_dense(&ctx, ps).unwrap();
            r.build_quantized_proxy(&ctx, s.input_size, 0, &proxy).unwrap();
            let qfw = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(ps, 4)).unwrap();
            sfnn_forward_device(&ctx, &batch, &proxy, &qfw).unwrap();
            let expected = qfw.download_output(&ctx).unwrap();
            let masters = r.weights.l1w.download(&ctx).unwrap();
            r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                &ctx,
                RangerUpdateParams::default(),
                ScalarLossKind::BceWithLogits,
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
                SfnnLayerLrMultipliers { qat_l1: true, qat_ft: true, qat_l2: true, qat_l3: true, ..Default::default() },
                None,
            )
            .unwrap();
            for (a, b) in expected.iter().zip(r.forward_workspace.download_output(&ctx).unwrap()) {
                assert!((a - b).abs() < 1e-5, "QAT {mode}: {a} {b}");
            }
            assert_eq!(masters, r.weights.l1w.download(&ctx).unwrap());
            let g = r.backward_workspace.l1fw_gradients.download(&ctx).unwrap();
            // Disconnected coordinates of a factor must never accumulate gradients.
            assert!(g.iter().any(|x| x.abs() > 1e-9));
            for col in 0..s.ft_size {
                for out in 0..s.l1_out() {
                    if revive_common::weight_index(s, 0, out, col).is_none() {
                        assert_eq!(g[col * s.l1_out() + out], 0.0);
                    }
                }
            }
            r.pending_gradient_batches = 0;
        }
        let before = r.weights.l1fw.as_ref().unwrap().download(&ctx).unwrap();
        let mut params = RangerUpdateParams::default();
        params.radam.learning_rate = 0.001;
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,
            params,
            ScalarLossKind::BceWithLogits,
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
            true,
            SfnnLayerLrMultipliers { qat_l1: true, ..Default::default() },
            None,
        )
        .unwrap();
        assert_ne!(before, r.weights.l1fw.as_ref().unwrap().download(&ctx).unwrap());
        assert!(
            r.optimizer_states.l1fw.as_ref().unwrap().momentum.download(&ctx).unwrap().iter().any(|x| x.abs() > 1e-12)
        );
    }
}

#[test]
#[ignore = "requires CUDA; compact gradient finite differences"]
fn compact_shared_backward_matches_finite_differences_and_rebase() {
    let ctx = Context::new(0).unwrap();
    for s in [shape(64, 64, 2), shape(0, 64, 2), shape(0, 0, 2)] {
        let mut r = runner(&ctx, s);
        let shared = vec![0.002; s.ft_size * s.l1_out()];
        let bias = vec![0.003; s.l1_out()];
        r.weights.l1fw = Some(F32Buffer::from_host(&ctx, &shared).unwrap());
        r.weights.l1fb = Some(F32Buffer::from_host(&ctx, &bias).unwrap());
        r.optimizer_states.l1fw = Some(RangerParamState::from_host_weights(&ctx, &shared).unwrap());
        r.optimizer_states.l1fb = Some(RangerParamState::from_host_weights(&ctx, &bias).unwrap());
        r.set_factorizer_config(SfnnFactorizerActive::SHARED, SfnnFactorizerAlpha::ONE).unwrap();
        let hb = SfnnForwardHostBatch {
            stm_indices: &[0, 1, 2, 3],
            nstm_indices: &[3, 2, 1, 0],
            buckets: &[0, 1, 0, 1],
            batch_size: 4,
            max_active: 1,
        };
        let batch = SfnnForwardDeviceBatch::from_host(&ctx, hb).unwrap();
        let fw = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(s, 4)).unwrap();
        r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
            &ctx,
            RangerUpdateParams::default(),
            ScalarLossKind::BceWithLogits,
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
            SfnnLayerLrMultipliers::default(),
            None,
        )
        .unwrap();
        let loss = || {
            r.forward_current_weights(&ctx, &batch, &fw).unwrap();
            fw.download_output(&ctx)
                .unwrap()
                .iter()
                .zip([0.2f64, 0.8, 0.3, 0.7])
                .map(|(&x, t)| {
                    let x = x as f64;
                    x.max(0.0) - t * x + (-x.abs()).exp().ln_1p()
                })
                .sum::<f64>()
                / 4.0
        };
        for (w, g) in [
            (&r.weights.l1w, &r.backward_workspace.l1w_gradients),
            (r.weights.l1fw.as_ref().unwrap(), &r.backward_workspace.l1fw_gradients),
            (&r.weights.l0w, &r.backward_workspace.l0w_gradients),
        ] {
            let original = w.download(&ctx).unwrap();
            let analytic = g.download(&ctx).unwrap()[0] as f64;
            let mut perturbed = original.clone();
            perturbed[0] += 0.001;
            w.upload(&ctx, &perturbed).unwrap();
            let plus = loss();
            perturbed[0] -= 0.002;
            w.upload(&ctx, &perturbed).unwrap();
            let minus = loss();
            w.upload(&ctx, &original).unwrap();
            let numeric = (plus - minus) / 0.002;
            assert!((analytic - numeric).abs() < 2e-5 + numeric.abs() * 0.01, "analytic={analytic} numeric={numeric}");
        }
        r.pending_gradient_batches = 0;
        r.forward_current_weights(&ctx, &batch, &fw).unwrap();
        let before = fw.download_output(&ctx).unwrap();
        r.set_residual_count_gates_by_stack(&ctx, Some(&[0.5, 0.25])).unwrap();
        r.forward_current_weights(&ctx, &batch, &fw).unwrap();
        let gated = fw.download_output(&ctx).unwrap();
        r.rebase_shared_factorizer_terms(&ctx, 1.0, 1.0, 1.0, 0.5).unwrap();
        r.set_factorizer_config(
            SfnnFactorizerActive::SHARED,
            SfnnFactorizerAlpha { shared: 0.5, ..SfnnFactorizerAlpha::ONE },
        )
        .unwrap();
        r.forward_current_weights(&ctx, &batch, &fw).unwrap();
        for (a, b) in gated.iter().zip(fw.download_output(&ctx).unwrap()) {
            assert!((a - b).abs() < 1e-5);
        }
        r.rebase_shared_factorizer_terms(&ctx, 1.0, 1.0, 0.5, 0.0).unwrap();
        r.set_factorizer_config(
            SfnnFactorizerActive::SHARED,
            SfnnFactorizerAlpha { shared: 0.0, ..SfnnFactorizerAlpha::ONE },
        )
        .unwrap();
        r.forward_current_weights(&ctx, &batch, &fw).unwrap();
        for (a, b) in gated.iter().zip(fw.download_output(&ctx).unwrap()) {
            assert!((a - b).abs() < 1e-5);
        }
        assert!(before.iter().all(|x| x.is_finite()));
    }
}
