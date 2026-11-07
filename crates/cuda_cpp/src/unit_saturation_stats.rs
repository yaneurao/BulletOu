use super::*;

/// Whole-validation unit counts, not a maximum of per-chunk rates.
/// GPU counts reset each chunk; host u64 totals avoid whole-dataset i32 overflow.
pub struct SfnnUnitSaturationStats {
    shape: SfnnForwardShape,
    device: I32Buffer,
    host: Vec<i32>,
    totals: Vec<u64>,
    positions: u64,
}

impl SfnnUnitSaturationStats {
    pub fn new(ctx: &Context, shape: SfnnForwardShape) -> Result<Self> {
        let stride = checked_product("unit stats hidden", &[2, shape.l1_hidden])?
            .checked_add(shape.l2_size)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| CudaCppError::message("unit stats stride overflow"))?;
        let len = checked_product("unit stats buckets", &[shape.num_stacks, stride])?
            .checked_add(shape.ft_size)
            .ok_or_else(|| CudaCppError::message("unit stats length overflow"))?;
        Ok(Self { shape, device: I32Buffer::new(ctx, len)?, host: vec![0; len], totals: vec![0; len], positions: 0 })
    }

    pub fn reset(&mut self) {
        self.totals.fill(0);
        self.positions = 0;
    }

    /// Counts are shared across both perspectives and independent of L1 buckets.
    pub fn ft_upper_counts(&self) -> (&[u64], u64) {
        (&self.totals[..self.shape.ft_size], 2 * self.positions)
    }

    pub fn accumulate(&mut self, ctx: &Context, w: &SfnnForwardWorkspace, buckets: &I32Buffer) -> Result<()> {
        w.validate()?;
        if w.layout.shape != self.shape {
            return Err(CudaCppError::message("unit stats shape mismatch"));
        }
        let s = self.shape;
        // SAFETY: owned buffers; backend validates dimensions and buffer lengths.
        check(unsafe {
            ffi::bulletou_cuda_cpp_sfnn_unit_stats(
                ctx.as_ptr(),
                w.stm_l0.as_ptr(),
                w.nstm_l0.as_ptr(),
                w.l2_input.as_ptr(),
                w.l2.as_ptr(),
                buckets.as_ptr(),
                self.device.as_ptr(),
                w.layout.batch_size,
                s.ft_size,
                s.l1_hidden,
                s.l2_size,
                s.num_stacks,
            )
        })?;
        self.device.download_prefix(ctx, &mut self.host)?;
        for (total, &value) in self.totals.iter_mut().zip(&self.host) {
            if value < 0 {
                return Err(CudaCppError::message("negative unit saturation count"));
            }
            *total += value as u64;
        }
        self.positions += w.layout.batch_size as u64;
        Ok(())
    }

    pub fn summary(&self) -> String {
        let s = self.shape;
        let ft = maximum((0..s.ft_size).map(|u| (self.totals[u], 2 * self.positions, None, u)));
        let stride = 1 + 2 * s.l1_hidden + s.l2_size;
        let dense = |offset: usize, width: usize| {
            maximum((0..s.num_stacks).flat_map(|b| {
                let base = s.ft_size + b * stride;
                (0..width).map(move |u| (self.totals[base + 1 + offset + u], self.totals[base], Some(b), u))
            }))
        };
        format!("ft_unit_upper_max={ft} l1_unit_upper_max={} l1_square_unit_upper_max={} l2_unit_upper_max={} l3_unit_upper_max=n/a(no-clamp)",
            dense(0,s.l1_hidden), dense(s.l1_hidden,s.l1_hidden), dense(2*s.l1_hidden,s.l2_size))
    }
}

fn maximum(values: impl Iterator<Item = (u64, u64, Option<usize>, usize)>) -> String {
    // Stable tie: first bucket/unit. Unseen buckets have no defined rate.
    let mut best: Option<(u64, u64, Option<usize>, usize)> = None;
    for v in values.filter(|v| v.1 > 0) {
        if best.is_none_or(|b| u128::from(v.0) * u128::from(b.1) > u128::from(b.0) * u128::from(v.1)) {
            best = Some(v);
        }
    }
    match best {
        None => "n/a(no-samples)".into(),
        Some((count, n, b, u)) => format!(
            "{:.4}%(bucket={},unit={u},hits={count},n={n})",
            100.0 * count as f64 / n as f64,
            b.map_or_else(|| "shared".into(), |v| v.to_string())
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maxima_keep_rare_buckets_and_ignore_unseen() {
        assert_eq!(
            maximum([(0, 0, Some(0), 0), (80, 100, Some(1), 1), (1, 1, Some(2), 2)].into_iter()),
            "100.0000%(bucket=2,unit=2,hits=1,n=1)"
        );
        assert_eq!(maximum(std::iter::empty()), "n/a(no-samples)");
    }

    #[test]
    #[ignore = "requires CUDA; synthetic activations only"]
    fn gpu_counts_match_cpu_and_reset() {
        let ctx = Context::new(0).unwrap();
        for skip in [false, true] {
            let s = SfnnForwardShape { l1_skip: skip, num_stacks: 3, ..crate::tests::tiny_sfnn_shape() };
            let mut stats = SfnnUnitSaturationStats::new(&ctx, s).unwrap();
            let mut expected = vec![0u64; stats.totals.len()];
            for n in [1, 17, 257] {
                let w = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(s, n)).unwrap();
                let pattern = |len| (0..len).map(|i| [-0.5, 0.0, 0.999, 1.0, 2.0][i % 5]).collect::<Vec<f32>>();
                let stm = pattern(w.stm_l0.len());
                let nstm = stm.iter().rev().copied().collect::<Vec<_>>();
                let input = pattern(w.l2_input.len());
                let l2 = pattern(w.l2.len());
                let buckets = (0..n).map(|p| (p % 2) as i32).collect::<Vec<_>>();
                w.stm_l0.upload(&ctx, &stm).unwrap();
                w.nstm_l0.upload(&ctx, &nstm).unwrap();
                w.l2_input.upload(&ctx, &input).unwrap();
                w.l2.upload(&ctx, &l2).unwrap();
                for p in 0..n {
                    for u in 0..s.ft_size {
                        expected[u] +=
                            u64::from(stm[p * s.ft_size + u] >= 1.0) + u64::from(nstm[p * s.ft_size + u] >= 1.0);
                    }
                    let b = s.ft_size + buckets[p] as usize * (1 + 2 * s.l1_hidden + s.l2_size);
                    expected[b] += 1;
                    for u in 0..s.l1_hidden {
                        expected[b + 1 + u] += u64::from(input[p * 2 * s.l1_hidden + s.l1_hidden + u] >= 1.0);
                        expected[b + 1 + s.l1_hidden + u] += u64::from(input[p * 2 * s.l1_hidden + u] >= 1.0);
                    }
                    for u in 0..s.l2_size {
                        expected[b + 1 + 2 * s.l1_hidden + u] += u64::from(l2[p * s.l2_size + u] >= 1.0);
                    }
                }
                stats.accumulate(&ctx, &w, &I32Buffer::from_host(&ctx, &buckets).unwrap()).unwrap();
                assert_eq!(stats.totals, expected);
                assert_eq!(w.l2.download(&ctx).unwrap(), l2);
            }
            assert_eq!(stats.positions, 275);
            stats.reset();
            assert!(stats.totals.iter().all(|&v| v == 0));
            assert!(stats.summary().contains("n/a(no-samples)"));
        }
    }

    #[test]
    #[ignore = "CUDA timing, about 550 MiB VRAM; no training"]
    fn gpu_timing() {
        let ctx = Context::new(0).unwrap();
        let s = SfnnForwardShape {
            ft_size: 1024,
            l1_hidden: 7,
            l2_size: 64,
            num_stacks: 9,
            ..crate::tests::tiny_sfnn_shape()
        };
        let w = SfnnForwardWorkspace::new(&ctx, SfnnForwardWorkspaceLayout::new(s, 65536)).unwrap();
        w.stm_l0.fill(&ctx, 1.0).unwrap();
        w.nstm_l0.fill(&ctx, 1.0).unwrap();
        w.l2_input.fill(&ctx, 1.0).unwrap();
        w.l2.fill(&ctx, 1.0).unwrap();
        let buckets = I32Buffer::from_host(&ctx, &vec![8; 65536]).unwrap();
        let mut stats = SfnnUnitSaturationStats::new(&ctx, s).unwrap();
        stats.accumulate(&ctx, &w, &buckets).unwrap();
        let start = std::time::Instant::now();
        for _ in 0..15 {
            stats.accumulate(&ctx, &w, &buckets).unwrap();
        }
        eprintln!("unit stats 15 chunks, all saturated/single bucket: {:.3}s", start.elapsed().as_secs_f64());
        assert!(stats.summary().contains("100.0000%"));
    }
}
