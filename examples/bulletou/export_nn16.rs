//! Explicit high-precision export; ordinary training and nn.bin stay i8.
use super::*;

pub(super) const VERSION: u32 = 0x7AF32F17;

#[derive(Parser, Debug)]
#[command(name = "bulletou export-nn16", about = "Export FP32 SFNN state.bin to nn16.bin (L1/L2/L3 int16, CPU only)")]
pub(super) struct ExportNn16Args {
    #[arg(long)]
    arch: NnueArch,
    /// FP32 checkpoint, not the already rounded nn.bin.
    #[arg(long)]
    state_bin: PathBuf,
    /// Defaults to bulletou-settings.json next to state.bin. Must describe
    /// the saved factorizer/count settings, not a new training configuration.
    #[arg(long)]
    settings_file: Option<PathBuf>,
    /// Existing files are never overwritten.
    #[arg(long, default_value = "nn16.bin")]
    output: PathBuf,
    /// L1/L2/L3 weight multiplier QB; power of two in 64..16384.
    #[arg(long, default_value_t = 4096)]
    weight_scale: u32,
}

pub(super) fn validate_scale(qb: u32) -> Result<(), String> {
    if !(64..=16384).contains(&qb) || !qb.is_power_of_two() {
        return Err("--weight-scale must be a power of two in 64..16384".into());
    }
    Ok(())
}

pub(super) fn validate_quantization(name: &str, values: &[f32], scale: f32, min: f64, max: f64) -> Result<(), String> {
    for (index, &value) in values.iter().enumerate() {
        let q = (f64::from(value) * f64::from(scale)).round();
        if !q.is_finite() || q < min || q > max {
            return Err(format!(
                "nn16 {name}[{index}]={value}: rounded value {q} outside [{min}, {max}]; use a smaller --weight-scale"
            ));
        }
    }
    Ok(())
}

pub(super) fn run(args: &ExportNn16Args) -> Result<(), String> {
    validate_scale(args.weight_scale)?;
    let feature = cuda_cpp_sfnn_feature_kind_from_arch(args.arch)?;
    if args.output.exists() {
        return Err(format!("{} already exists; choose a different output (no overwrite)", args.output.display()));
    }
    let settings =
        args.settings_file.clone().unwrap_or_else(|| args.state_bin.with_file_name("bulletou-settings.json"));
    let mut raw = vec![std::ffi::OsString::from("bulletou")];
    raw.extend(bulletou_settings_json_args(&settings)?);
    let train_args = Args::try_parse_from(raw).map_err(|e| e.to_string())?;
    if train_args.arch().cli_name() != args.arch.cli_name() {
        return Err("--arch disagrees with checkpoint settings".into());
    }
    let spec = effective_sfnn_factorizer_spec(&train_args);
    let mut alpha = effective_sfnn_factorizer_alpha(&train_args);
    let stack = train_args.effective_layerstack().unwrap_or(LayerStackMode::Kingrank3by3);
    let (ft_size, l1_hidden, l2_size) = args.arch.dims();
    let shape = bulletou_cuda_cpp::SfnnForwardShape {
        input_size: feature.input_size_for_args(&train_args),
        ft_size,
        l1_hidden,
        l1_skip: args.arch.sfnn_l1_skip(),
        l2_size,
        num_stacks: stack.num_stacks(),
        l1_group_count: args.arch.sfnn_l1_group_count(),
        l1_common_size: args.arch.sfnn_l1_common_size(),
        l1_shard_size: args.arch.sfnn_l1_shard_size(),
        factorizer_king_axis_dim: stack.factorizer_king_axis_dim(),
        factorizer_hand_axis_dim: stack.factorizer_hand_axis_dim(),
        factorizer_progress_axis: spec.progress_axis,
        factorizer_king_hand_pair: spec.king_hand_pair,
        factorizer_king_progress_pair: spec.king_progress_pair,
        factorizer_hand_progress_pair: spec.hand_progress_pair,
    };
    eprintln!("export-nn16: reading FP32 weights (no GPU or optimizer allocation)");
    eprintln!("  settings = {}", settings.display());
    let mut sections = load_cuda_cpp_component_state_sections(&args.state_bin, "nnue", &["weights", "train"], true)?;
    let records = sections.remove("weights").unwrap_or_default();
    validate_sfnn_l1_only_factorizer_checkpoint(&records)?;
    validate_ft_factorizer_checkpoint(
        &records,
        feature.base_input_size(),
        feature.virtual_rows(),
        ft_size,
        train_args.no_ft_factorize,
    )?;
    // Export is not a resume: never migrate/rebase/extract or create new factors.
    if records.contains_key("l1fw") != (spec.shared && !shape.has_compact_l1())
        || records.contains_key("l1axw") != (spec.any_axis() && !shape.has_compact_l1())
    {
        return Err(
            "checkpoint factorizer tensors disagree with settings; supply the settings used at save time".into()
        );
    }
    if let Some(saved) = load_sfnn_shared_coefficients(&sections.remove("train").unwrap_or_default())? {
        alpha.ft = saved[0];
        alpha.shared = saved[1];
    }
    let (weights, appended) = load_cuda_cpp_sfnn_weights_from_records(feature, shape, &records)?;
    if appended {
        return Err("export-nn16 cannot append missing progress factorizer terms".into());
    }
    drop(records);
    let progress = read_sfnn_progress_params_from_state_bin(&args.state_bin, args.arch, stack)?;
    if stack.progress_bucket_count() > 1 && progress.is_none() {
        return Err("progress architecture requires checkpoint progress parameters".into());
    }
    let count = WorkerSfnnSession::compute_count_settings(&train_args, shape)?;
    let readback = sfnn_initial_weights_into_readback(weights);
    for (name, values) in [("l0w", &readback.l0w), ("l0b", &readback.l0b)] {
        if values.iter().any(|x| !x.is_finite()) {
            return Err(format!("non-finite {name} in checkpoint"));
        }
    }
    let parent = args.output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let nonce =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
    let temp = parent.join(format!(".nn16-export-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&temp).map_err(|e| e.to_string())?;
    // Only this newly created private directory is cleaned up on error.
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(temp.clone());
    let temporary_output = temp.join("nn16.bin");
    write_cuda_cpp_sfnn_nn_bin_format(
        &temporary_output,
        feature,
        shape,
        &readback,
        spec,
        alpha,
        count.residual_count_gates.as_deref(),
        count.factorizer_axis_confidences.as_deref(),
        progress.as_ref(),
        Some(args.weight_scale),
    )?;
    if progress.is_some() {
        let source = temp.join("progress.bin");
        let destination = parent.join("progress.bin");
        if destination.exists() {
            if std::fs::read(&destination).map_err(|e| e.to_string())?
                != std::fs::read(&source).map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "{} contains a different classifier; export to another folder",
                    destination.display()
                ));
            }
        } else {
            std::fs::rename(source, &destination).map_err(|e| e.to_string())?;
        }
    }
    // A hard link publishes the complete file atomically and fails if the name exists.
    std::fs::hard_link(&temporary_output, &args.output)
        .map_err(|e| format!("cannot publish {}: {e}", args.output.display()))?;
    println!("export-nn16 complete:");
    println!("  source = {}", args.state_bin.display());
    println!("  output = {}", args.output.display());
    println!("  arch = {}", args.arch.cli_name());
    println!(
        "  version = 0x{VERSION:08X}, QA=127, QB={}, FC bias scale={}",
        args.weight_scale,
        127 * args.weight_scale
    );
    println!(
        "  shift = {}, raw divisor for network_output = {}",
        args.weight_scale.trailing_zeros(),
        127 * args.weight_scale
    );
    println!("  factorizer alpha = {} (FT/shared from state when recorded)", alpha.config_string());
    println!("  L1/L2/L3 = int16 weights + int32 biases; FT unchanged; FC range overflow is an error");
    Ok(())
}

struct Nn16Weights {
    ft: QuantizedSfnnWeights,
    qb: u32,
    l1w: Vec<i16>,
    l2w: Vec<i16>,
    l3w: Vec<i16>,
}

fn read_i16s(bytes: &[u8], pos: &mut usize, count: usize) -> Result<Vec<i16>, String> {
    let end = count.checked_mul(2).and_then(|n| pos.checked_add(n)).ok_or("nn16 length overflow")?;
    let data = bytes.get(*pos..end).ok_or("truncated nn16 weights")?;
    *pos = end;
    Ok(data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect())
}

fn read_weights(path: &Path, arch: NnueArch, layerstack: LayerStackMode) -> Result<Nn16Weights, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if read_u32_le(&bytes, 0, "version")? != VERSION || read_u32_le(&bytes, 4, "model hash")? != KHASH_SFNN {
        return Err("nn16 version/model hash mismatch".into());
    }
    let end = 12 + read_u32_le(&bytes, 8, "desc length")? as usize;
    let arch_desc = std::str::from_utf8(bytes.get(12..end).ok_or("truncated nn16 description")?)
        .map_err(|e| e.to_string())?
        .to_owned();
    let qb = read_u32_le(&bytes, end, "QB")?;
    validate_scale(qb)?;
    if read_u32_le(&bytes, end + 4, "FT hash")? != FT_HASH_SFNN {
        return Err("nn16 FT hash mismatch".into());
    }
    let feature_kind = cuda_cpp_sfnn_feature_kind_from_arch(arch)?;
    let input_size = feature_kind.base_input_size();
    let (ft_size, l1_hidden, l2_size) = arch.dims();
    let num_stacks = layerstack.num_stacks();
    let l1_skip = arch.sfnn_l1_skip();
    let expected_desc = format!(
        "ModelType=SFNNWithoutPsqt;Features={}[{}->{}x2],Network=SFNN-{}{{LayerStack={}}};FCWeightBits=16;QB={};H1={};Skip={};H2={}",
        feature_kind.feature_set().display_name(),
        input_size,
        ft_size,
        ft_size,
        num_stacks,
        qb,
        l1_hidden,
        u8::from(l1_skip),
        l2_size
    );
    if arch_desc != expected_desc {
        return Err("nn16 architecture description disagrees with --arch".into());
    }
    let l1_out = l1_hidden + usize::from(l1_skip);
    let l1_pad_in = nnue_pad32(ft_size);
    let l2_pad_in = nnue_pad32(l1_hidden * 2);
    let l3_pad_in = nnue_pad32(l2_size);
    let (l0b, pos) = read_sfnn_leb128_i16_chunk(&bytes, end + 8, ft_size, "FT bias")?;
    let (l0w, mut pos) = read_sfnn_leb128_i16_chunk(&bytes, pos, input_size * ft_size, "FT weights")?;
    let mut l1b = Vec::new();
    let mut l2b = Vec::new();
    let mut l3b = Vec::new();
    let mut l1w = Vec::new();
    let mut l2w = Vec::new();
    let mut l3w = Vec::new();
    for _ in 0..num_stacks {
        if read_u32_le(&bytes, pos, "network hash")? != NETWORK_HASH_SFNN {
            return Err("nn16 network hash mismatch".into());
        }
        pos += 4;
        let (b, next) = read_i32_vec_le(&bytes, pos, l1_out, "L1 bias")?;
        l1b.extend(b);
        pos = next;
        l1w.extend(read_i16s(&bytes, &mut pos, l1_out * l1_pad_in)?);
        let (b, next) = read_i32_vec_le(&bytes, pos, l2_size, "L2 bias")?;
        l2b.extend(b);
        pos = next;
        l2w.extend(read_i16s(&bytes, &mut pos, l2_size * l2_pad_in)?);
        let (b, next) = read_i32_vec_le(&bytes, pos, 1, "L3 bias")?;
        l3b.extend(b);
        pos = next;
        l3w.extend(read_i16s(&bytes, &mut pos, l3_pad_in)?);
    }
    if pos != bytes.len() {
        return Err("nn16 has trailing bytes / mismatched architecture".into());
    }
    let progress_params = read_sfnn_progress_sidecar(path, arch, layerstack)?;
    if layerstack.progress_bucket_count() > 1 && progress_params.is_none() {
        return Err("nn16 requires progress.bin sidecar".into());
    }
    Ok(Nn16Weights {
        qb,
        l1w,
        l2w,
        l3w,
        ft: QuantizedSfnnWeights {
            arch_desc,
            feature_kind,
            layerstack,
            input_size,
            ft_size,
            l1_hidden,
            l1_skip,
            l2_size,
            num_stacks,
            l1_pad_in,
            l2_pad_in,
            l3_pad_in,
            l0b,
            l0w,
            progress_params,
            l1b,
            l2b,
            l3b,
            l1w: Vec::new(),
            l2w: Vec::new(),
            l3w: Vec::new(),
        },
    })
}

fn crelu(value: i64, shift: u32, round: QuantizedRoundMode) -> u8 {
    quantized_shift_right_nonnegative(value.max(0), shift, round).min(127) as u8
}

fn sqrcrelu(value: i64, shift: u32, round: QuantizedRoundMode) -> u8 {
    // Squared activation is symmetric (including negative inputs), like the
    // existing SFNN path. i128 avoids overflow before the saturation clamp.
    let square = i128::from(value) * i128::from(value);
    let bits = 2 * shift + 7;
    let rounded = match round {
        QuantizedRoundMode::Floor => square,
        QuantizedRoundMode::Nearest => square + (1i128 << (bits - 1)),
    };
    (rounded >> bits).min(127) as u8
}

fn forward_tail(w: &Nn16Weights, ft: &[u8], stack: usize, args: &QuantizedTestArgs) -> i64 {
    let m = &w.ft;
    let shift = w.qb.trailing_zeros();
    let mut l1 = vec![0i64; m.l1_out()];
    for (out, sum) in l1.iter_mut().enumerate() {
        *sum = i64::from(m.l1b[stack * m.l1_out() + out]);
        let row = (stack * m.l1_out() + out) * m.l1_pad_in;
        for (i, &x) in ft.iter().enumerate() {
            *sum += i64::from(x) * i64::from(w.l1w[row + i]);
        }
    }
    let mut input = vec![0u8; m.l1_hidden * 2];
    for i in 0..m.l1_hidden {
        input[i] = sqrcrelu(l1[i], shift, args.quant_sqrcrelu_round);
        input[m.l1_hidden + i] = crelu(l1[i], shift, args.quant_crelu_round);
    }
    let mut output = i64::from(m.l3b[stack]);
    for out in 0..m.l2_size {
        let row = (stack * m.l2_size + out) * m.l2_pad_in;
        let sum = i64::from(m.l2b[stack * m.l2_size + out])
            + input.iter().enumerate().map(|(i, &x)| i64::from(x) * i64::from(w.l2w[row + i])).sum::<i64>();
        output += i64::from(crelu(sum, shift, args.quant_crelu_round)) * i64::from(w.l3w[stack * m.l3_pad_in + out]);
    }
    if m.l1_skip {
        output += l1[m.l1_hidden];
    }
    output
}

pub(super) fn test(args: &QuantizedTestArgs, verbose: bool) -> Result<QuantizedTestReport, String> {
    if !matches!(args.mode, QuantizedTestMode::CpuExact) {
        return Err("nn16 currently supports --mode cpu-exact only".into());
    }
    let w = read_weights(&args.nn_bin, args.arch, args.effective_layerstack())?;
    if verbose {
        eprintln!(
            "quantized-test nn16: mode=cpu-exact, QB={}, raw/(127*QB) = raw/{}; int64 accumulation",
            w.qb,
            127 * w.qb
        );
        eprintln!(
            "  FV_SCALE={} is in the original QB=64 units (raw16 normalized before engine division)",
            args.fv_scale
        );
    }
    let teacher = args.test_teacher.to_str().ok_or("non UTF-8 teacher path")?;
    let positions = match args.test_positions.and_then(ValidationPositionCount::limit) {
        None => read_all_teacher_positions(teacher),
        Some(n) => match args.test_sample {
            TestSampleMode::Sequential => read_teacher_positions_prefix(teacher, n),
            TestSampleMode::Random => read_random_teacher_positions(teacher, n, args.test_seed),
        },
    }
    .map_err(|e| e.to_string())?;
    if positions.is_empty() {
        return Err("empty test teacher".into());
    }
    let scores: Vec<_> = positions.iter().map(|p| p.score()).collect();
    let results: Vec<_> = positions.iter().map(|p| p.game_result()).collect();
    let mask =
        build_validation_sample_mask(&scores, &results, (args.score_drop_abs > 0).then_some(args.score_drop_abs));
    let started = std::time::Instant::now();
    let mut raw = Vec::with_capacity(positions.len());
    for chunk in positions.chunks(65536) {
        let batch =
            build_sfnn_validation_fast_batch(w.ft.feature_kind, w.ft.layerstack, chunk, w.ft.progress_params.as_ref())?;
        let out: Result<Vec<i64>, String> = (0..chunk.len())
            .into_par_iter()
            .map_init(
                || vec![0u8; w.ft.ft_size],
                |ft, sample| {
                    let stack = quantized_sfnn_ft_forward_sample(
                        &w.ft,
                        &batch,
                        sample,
                        args.sfnn_ft_shift,
                        args.quant_ft_round,
                        ft,
                    )?;
                    Ok(forward_tail(&w, ft, stack, args))
                },
            )
            .collect();
        raw.extend(out?);
        if verbose {
            eprintln!("  [nn16 test] {}/{} positions", raw.len(), positions.len());
        }
    }
    let normalized: Vec<_> = raw.iter().map(|&v| (v as f64 / (127.0 * f64::from(w.qb))) as f32).collect();
    let divisor = i64::from(args.fv_scale) * i64::from(w.qb / 64);
    let engine: Vec<_> = raw
        .iter()
        .map(|&v| {
            let v = match args.quant_final_div_round {
                QuantizedRoundMode::Floor => v,
                QuantizedRoundMode::Nearest => {
                    if v >= 0 {
                        v + divisor / 2
                    } else {
                        v - divisor / 2
                    }
                }
            };
            (v / divisor) as f32 + args.engine_score_offset
        })
        .collect();
    let train_scale = compute_sign_accuracy_with_loss_masked(
        &normalized,
        &scores,
        &results,
        &mask,
        args.lambda,
        args.scale as f32,
        quantized_train_scale_model_output_scale(args),
        quantized_train_scale_loss_kind(args),
    );
    let engine_scale = compute_sign_accuracy_with_loss_masked(
        &engine,
        &scores,
        &results,
        &mask,
        args.lambda,
        args.scale as f32,
        quantized_engine_scale_model_output_scale(args),
        quantized_engine_scale_loss_kind(args),
    );
    Ok(QuantizedTestReport { records: positions.len(), train_scale, engine_scale, elapsed: started.elapsed() })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nn16_roundtrip_matches_i8_at_same_scale_and_preserves_folding() {
        let train = Args::try_parse_from([
            "bulletou",
            "--teacher",
            "-",
            "--arch",
            "SFNN_halfka2_32_7_32_k3k3",
            "--sfnn-factorizer",
            "axis",
        ])
        .unwrap();
        let feature = CudaCppSfnnFeatureKind::Halfka2;
        let mut w = build_sfnn_initial_weights_for_cuda_cpp(&train, feature).unwrap();
        let shape = w.shape;
        w.l1w[0] = 0.12345;
        w.l1fw.as_mut().unwrap().fill(0.003);
        w.l1fb.as_mut().unwrap().fill(0.004);
        w.l1axw.as_mut().unwrap().fill(0.005);
        w.l1axb.as_mut().unwrap().fill(0.006);
        // FT shared rows are also nonzero, testing actual FT folding.
        w.l0w[feature.base_input_size() * shape.ft_size..].fill(0.007);
        let w = sfnn_initial_weights_into_readback(w);
        let alpha = SfnnFactorizerAlphaSpec { ft: 0.5, shared: 0.75, ..SfnnFactorizerAlphaSpec::ONE };
        let gates = vec![0.5; shape.num_stacks];
        let multipliers = vec![0.25; shape.factorizer_axis_count()];
        let dir = std::env::temp_dir().join(format!("bulletou-nn16-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let old = dir.join("nn.bin");
        let new = dir.join("nn16.bin");
        let spec = effective_sfnn_factorizer_spec(&train);
        write_cuda_cpp_sfnn_nn_bin(&old, feature, shape, &w, spec, alpha, Some(&gates), Some(&multipliers), None)
            .unwrap();
        write_cuda_cpp_sfnn_nn_bin_format(
            &new,
            feature,
            shape,
            &w,
            spec,
            alpha,
            Some(&gates),
            Some(&multipliers),
            None,
            Some(64),
        )
        .unwrap();
        let a = parse_quantized_sfnn_nn_bin(&old, train.arch(), LayerStackMode::Kingrank3by3).unwrap();
        let b = read_weights(&new, train.arch(), LayerStackMode::Kingrank3by3).unwrap();
        assert_eq!(a.l0w, b.ft.l0w);
        assert_eq!(a.l0b, b.ft.l0b);
        assert_eq!(a.l1b, b.ft.l1b);
        assert_eq!(a.l2b, b.ft.l2b);
        assert_eq!(a.l3b, b.ft.l3b);
        assert_eq!(a.l1w.iter().copied().map(i16::from).collect::<Vec<_>>(), b.l1w);
        assert_eq!(a.l2w.iter().copied().map(i16::from).collect::<Vec<_>>(), b.l2w);
        assert_eq!(a.l3w.iter().copied().map(i16::from).collect::<Vec<_>>(), b.l3w);
        // Compare the whole integer path, including skip and negative square activation.
        let test = QuantizedTestArgs::try_parse_from([
            "bulletou",
            "--arch",
            "SFNN_halfka2_32_7_32_k3k3",
            "--nn-bin",
            "-",
            "--test-teacher",
            "-",
        ])
        .unwrap();
        let batch = bulletou_lib::value::FastBatchHost {
            layout: bulletou_lib::value::FastBatchLayout {
                batch_size: 1,
                max_active: 1,
                output_size: 1,
                hand_count_dim: 0,
            },
            stm: vec![-1],
            nstm: vec![-1],
            buckets: vec![0],
            targets: vec![0.0],
            weights: vec![1.0],
            hand_count: None,
            progress: None,
        };
        let mut state = QuantizedSfnnThreadState::new(&a);
        let output = quantized_sfnn_forward_sample(
            &a,
            &batch,
            0,
            7,
            QuantizedRoundMode::Floor,
            QuantizedRoundMode::Floor,
            QuantizedRoundMode::Floor,
            &mut state,
        )
        .unwrap();
        assert_eq!(i64::from(output.raw), forward_tail(&b, &state.ft, 0, &test));
        write_cuda_cpp_sfnn_nn_bin_format(
            &new,
            feature,
            shape,
            &w,
            spec,
            alpha,
            Some(&gates),
            Some(&multipliers),
            None,
            Some(4096),
        )
        .unwrap();
        let b = read_weights(&new, train.arch(), LayerStackMode::Kingrank3by3).unwrap();
        assert_eq!(b.qb, 4096);
        assert_eq!(a.l0w, b.ft.l0w);
        assert!(b.l1w.iter().any(|x| x.unsigned_abs() > 127));
        assert!(parse_quantized_sfnn_nn_bin(&new, train.arch(), LayerStackMode::Kingrank3by3).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn nn16_activations_and_i16_decode() {
        assert_eq!(crelu(4096 * 128, 12, QuantizedRoundMode::Floor), 127);
        assert_eq!(crelu(-1, 12, QuantizedRoundMode::Floor), 0);
        assert_eq!(sqrcrelu(-4096 * 128, 12, QuantizedRoundMode::Floor), 127);
        assert_eq!(sqrcrelu(i64::MAX, 12, QuantizedRoundMode::Floor), 127);
        let mut pos = 0;
        assert_eq!(read_i16s(&[0, 128, 255, 127], &mut pos, 2).unwrap(), vec![i16::MIN, i16::MAX]);
        assert!(read_i16s(&[0], &mut 0, 1).is_err());
    }
    #[test]
    fn nn16_scale_and_rounding() {
        for n in [64, 256, 4096, 16384] {
            validate_scale(n).unwrap();
        }
        for n in [0, 32, 1000, 32768] {
            assert!(validate_scale(n).is_err());
        }
        assert_eq!(sfnn_quantise_i16(0.5 / 4096.0, 4096.0), 1);
        assert_eq!(sfnn_quantise_i16(-0.5 / 4096.0, 4096.0), -1);
        for value in [-0.333, 0.111, 1.9] {
            assert!((value - f32::from(sfnn_quantise_i16(value, 4096.0)) / 4096.0).abs() <= 0.5 / 4096.0);
        }
        assert!(validate_quantization("w", &[8.0], 4096.0, -32768.0, 32767.0).is_err());
        assert!(validate_quantization("w", &[f32::NAN], 4096.0, -32768.0, 32767.0).is_err());
        validate_quantization("w", &[-8.0, 7.999], 4096.0, -32768.0, 32767.0).unwrap();
    }
}
