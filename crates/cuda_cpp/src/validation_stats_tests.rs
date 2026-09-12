use super::*;

#[test]
#[ignore = "CUDA diagnostic timing only; about 550 MiB VRAM, no training"]
fn gpu_validation_stats_timing_1024() {
    let ctx = Context::new(0).unwrap();
    let shape = SfnnForwardShape { ft_size: 1024, l1_hidden: 7, l2_size: 64, ..tiny_sfnn_shape() };
    let w = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(shape, 65536)).unwrap();
    let scratch = F32Buffer::new(&ctx, 1280).unwrap();
    w.stm_l0.fill(&ctx, 1.0).unwrap();
    w.nstm_l0.fill(&ctx, 0.5).unwrap();
    w.l2_input.fill(&ctx, 1.0).unwrap();
    w.l2.fill(&ctx, 0.0).unwrap();
    w.output.fill(&ctx, 0.25).unwrap();
    for _ in 0..3 {
        w.validation_stats(&ctx, &scratch).unwrap();
    }
    let start = std::time::Instant::now();
    for _ in 0..30 {
        w.validation_stats(&ctx, &scratch).unwrap();
    }
    eprintln!(
        "qstats reduction + partial readback: {:.3} ms/chunk; {:.3} ms/15 chunks (synthetic buffers, not full qvalid)",
        start.elapsed().as_secs_f64() * 1000.0 / 30.0,
        start.elapsed().as_secs_f64() * 1000.0 / 2.0
    );
}

// Forward-buffer diagnostics only: no optimizer update or teacher loading.
#[test]
#[ignore = "requires a CUDA-capable NVIDIA GPU"]
fn gpu_validation_stats_match_cpu_and_reuse_scratch() {
    let ctx = Context::new(0).unwrap();
    let scratch = F32Buffer::new(&ctx, 1280).unwrap();
    for skip in [false, true] {
        for batch in [1, 17, 257] {
            let shape = SfnnForwardShape { l1_skip: skip, ..tiny_sfnn_shape() };
            let w = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(shape, batch)).unwrap();
            let pattern = |n: usize| (0..n).map(|i| [-2.0, 0.0, 0.999, 1.0, 2.0][i % 5]).collect::<Vec<_>>();
            let stm = pattern(w.stm_l0.len());
            let nstm = stm.iter().rev().copied().collect::<Vec<_>>();
            let input = pattern(w.l2_input.len());
            let l2 = pattern(w.l2.len());
            let output = pattern(batch);
            w.stm_l0.upload(&ctx, &stm).unwrap();
            w.nstm_l0.upload(&ctx, &nstm).unwrap();
            w.l2_input.upload(&ctx, &input).unwrap();
            w.l2.upload(&ctx, &l2).unwrap();
            w.output.upload(&ctx, &output).unwrap();
            let upper = |a: &[f32]| a.iter().filter(|&&v| v >= 1.0).count() as f64;
            let normal = input.chunks(2 * shape.l1_hidden).map(|row| upper(&row[shape.l1_hidden..])).sum::<f64>();
            let square = input.chunks(2 * shape.l1_hidden).map(|row| upper(&row[..shape.l1_hidden])).sum::<f64>();
            let expected = [
                upper(&stm) + upper(&nstm),
                normal,
                square,
                upper(&l2),
                output.iter().map(|&v| f64::from(v) * f64::from(v)).sum(),
            ];
            for _ in 0..2 {
                let actual = w.validation_stats(&ctx, &scratch).unwrap();
                for i in 0..5 {
                    assert!((actual[i] - expected[i]).abs() < 1e-4, "metric {i}: {actual:?} != {expected:?}");
                }
                assert_eq!(w.download_output(&ctx).unwrap(), output);
            }
        }
    }
}
