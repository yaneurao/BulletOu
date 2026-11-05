//! Layer-wise GPTQ primitives. Calibration and export integration are separate.

use super::*;

#[derive(Parser, Debug)]
#[command(name = "bulletou export-gptq", about = "Export SFNN with L2/L3 GPTQ using training calibration positions")]
pub(super) struct ExportArgs {
    #[arg(long)] arch: NnueArch,
    #[arg(long)] state_bin: PathBuf,
    #[arg(long)] settings_file: PathBuf,
    /// Epoch of the checkpoint, used to resolve scheduled factorizer settings.
    #[arg(long)] epoch: usize,
    /// A representative training teacher file, separate from evaluation data.
    #[arg(long)] calibration_teacher: PathBuf,
    #[arg(long, default_value_t = 65536)] calibration_positions: usize,
    #[arg(long, default_value_t = 128)] min_bucket_positions: usize,
    #[arg(long, default_value_t = 0.01)] damping: f64,
    #[arg(long, default_value_t = 0)] cuda_cpp_device: i32,
    /// New output path; existing files are never overwritten.
    #[arg(long)] output: PathBuf,
    /// Count file produced by bucket-count; enables removal of rare L1 residuals.
    #[arg(long, requires = "residual_max_count")] bucket_counts: Option<PathBuf>,
    /// Remove the bucket-specific L1 residual when its count is at most this value.
    #[arg(long, requires = "bucket_counts")] residual_max_count: Option<u32>,
    /// Also export the non-GPTQ network after residual removal.
    #[arg(long)] normal_output: Option<PathBuf>,
    /// Check that this progress classifier is the one saved with the checkpoint.
    #[arg(long)] progress_bin: Option<PathBuf>,
    /// Measure and print GPTQ qacc/qloss with the checkpoint's test settings.
    #[arg(long)] validate: bool,
}

fn calibration_teacher(args: &Args) -> Result<PathBuf, String> {
    let path=args.sfnn_gptq_calibration_teacher.clone().unwrap_or_else(||PathBuf::from(&args.teacher));
    if path.is_file() { return Ok(path); }
    if !path.is_dir() { return Err(format!("GPTQ calibration teacher does not exist: {}",path.display())); }
    let mut files=Vec::new();
    for entry in std::fs::read_dir(&path).map_err(|e|format!("cannot read {}: {e}",path.display()))? {
        let entry=entry.map_err(|e|e.to_string())?; let p=entry.path();
        if p.is_file() && p.extension().and_then(|v|v.to_str()).is_some_and(|v|["bin","hcpe","psv"].iter().any(|x|v.eq_ignore_ascii_case(x))) { files.push(p); }
    }
    files.sort();
    files.into_iter().next().ok_or_else(||format!("no supported GPTQ calibration file in {}",path.display()))
}

pub(super) fn run_automatic(args:&Args, checkpoint:&Path, epoch:usize, superbatch:usize) -> Result<(),String> {
    let teacher=calibration_teacher(args)?;
    let settings=checkpoint.join("bulletou-settings.json");
    if !settings.is_file() {return Err("automatic GPTQ requires a checkpoint settings file (--settings-file)".into());}
    let output=checkpoint.join("nn-gptq.bin");
    eprintln!("  [GPTQ] START epoch={epoch} teacher={} positions={} output={}",teacher.display(),args.sfnn_gptq_calibration_positions,output.display());
    run(&ExportArgs {arch:args.arch(),state_bin:checkpoint.join("state.bin"),settings_file:settings,
        epoch,calibration_teacher:teacher,calibration_positions:args.sfnn_gptq_calibration_positions,
        min_bucket_positions:128,damping:0.01,cuda_cpp_device:args.cuda_cpp_device,output,
        bucket_counts:None,residual_max_count:None,normal_output:None,progress_bin:None,validate:false})?;
    eprintln!("  [GPTQ] DONE {}",checkpoint.join("nn-gptq.bin").display());
    if let Err(err)=validate_automatic(args,checkpoint,epoch,superbatch) {
        eprintln!("  WARNING: GPTQ validation failed for {}: {err}; nn-gptq.bin was saved",checkpoint.display());
    }
    Ok(())
}

fn validate_automatic(args:&Args, checkpoint:&Path, epoch:usize, superbatch:usize) -> Result<(),String> {
    let Some(mut test_args)=quantized_test_args_from_training_args(args,checkpoint.join("nn-gptq.bin"))? else {
        return Err("GPTQ validation requires --test-teacher and a supported SFNN arch".into());
    };
    // Exact CPU inference avoids a second full-model GPU allocation while the
    // training runner and its validation caches remain resident.
    test_args.mode=QuantizedTestMode::CpuExact;
    let report=run_quantized_test_impl(&test_args,false)?;
    let accuracy=report.train_scale.accuracy();
    let loss=report.train_scale.test_loss.ok_or("GPTQ validation returned no loss")?;
    if !accuracy.is_finite() || !loss.is_finite() {return Err("GPTQ validation returned non-finite metrics".into());}
    let record=serde_json::json!({"version":1,"epoch":epoch,"superbatch":superbatch,
        "gptq_qacc":accuracy,"gptq_qloss":loss,"mode":"cpu-exact","positions":report.records});
    let tmp=checkpoint.join(format!(".gptq-validation-{}.tmp",std::process::id()));
    let dest=checkpoint.join("gptq-validation.json");
    let bytes=serde_json::to_vec_pretty(&record).map_err(|e|e.to_string())?;
    std::fs::write(&tmp,bytes).map_err(|e|e.to_string())?;
    if let Err(e)=std::fs::rename(&tmp,&dest) {let _=std::fs::remove_file(&tmp);return Err(e.to_string());}
    eprintln!("  [GPTQ VALID] epoch={epoch} sb={superbatch} qacc={accuracy:.7} qloss={loss:.8} mode=cpu-exact positions={} elapsed={:.3}s",report.records,report.elapsed.as_secs_f64());
    Ok(())
}

pub(super) fn run(args: &ExportArgs) -> Result<(), String> {
    if args.epoch == 0 || args.calibration_positions == 0 || args.min_bucket_positions == 0
        || !args.damping.is_finite() || args.damping <= 0.0 { return Err("invalid epoch, calibration size, bucket minimum or damping".into()); }
    if args.output.exists() { return Err("output already exists; choose a new file".into()); }
    let validation_path = args.output.with_file_name("gptq-validation.json");
    if args.validate && validation_path.exists() { return Err("GPTQ validation output already exists; choose a new directory".into()); }
    let train=training_args_for_export(&args.settings_file,args.epoch)?;
    if train.arch().cli_name()!=args.arch.cli_name() { return Err("--arch disagrees with settings".into()); }
    if train.ft_factorizer { return Err("export-gptq currently requires a checkpoint with FT factorizer off".into()); }
    // This first implementation does not silently ignore per-axis gates or BN.
    if train.sfnn_bucket_counts.is_some() { return Err("export-gptq does not yet support count gates".into()); }
    let alpha=effective_sfnn_factorizer_alpha(&train);
    if [alpha.ft,alpha.shared,alpha.king_axis,alpha.hand_axis,alpha.progress_axis,
        alpha.king_hand_pair,alpha.king_progress_pair,alpha.hand_progress_pair].iter().any(|&v|v!=1.0) {
        return Err("export-gptq currently requires unit factorizer coefficients".into());
    }
    let feature=cuda_cpp_sfnn_feature_kind_from_arch(args.arch)?;
    let stack=train.effective_layerstack().ok_or("SFNN layerstack required")?;
    eprintln!("export-gptq: loading checkpoint and calibration positions; L2/L3 only");
    let mut state=load_cuda_cpp_sfnn_initial_state(&args.state_bin,&train,feature)?;
    let shape=state.weights.shape;
    if state.weights.batch_norm.0.iter().any(Option::is_some)
        || state.weights.l2fw.is_some() || state.weights.l2axw.is_some()
        || state.weights.l3fw.is_some() || state.weights.l3axw.is_some() {
        return Err("export-gptq requires non-BN weights without L2/L3 factorizers".into());
    }
    let progress=cuda_cpp_sfnn_progress_params_for_state(state.progress.as_ref())?;
    if stack.progress_bucket_count()>1 && progress.is_none() {return Err("checkpoint lacks progress classifier".into());}
    if let Some(path) = &args.progress_bin {
        let checkpoint_progress = args.state_bin.parent().ok_or("checkpoint has no parent")?.join("progress.bin");
        if std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?
            != std::fs::read(&checkpoint_progress).map_err(|e| format!("cannot read {}: {e}", checkpoint_progress.display()))? {
            return Err("--progress-bin differs from the checkpoint progress.bin".into());
        }
    }
    if let Some(path) = &args.bucket_counts {
        if !effective_sfnn_factorizer_spec(&train).any_axis() && !effective_sfnn_factorizer_spec(&train).shared {
            return Err("residual removal requires an active L1 factorizer".into());
        }
        let counts = SfnnBucketCounts::read_from_path(path)?;
        if counts.arch != args.arch.cli_name() { return Err("bucket-count arch differs from checkpoint arch".into()); }
        let removed = zero_rare_l1_residuals(&mut state.weights, &counts.counts, args.residual_max_count.unwrap())?;
        eprintln!("  residual removal: {removed}/{} buckets (count <= {})", shape.num_stacks, args.residual_max_count.unwrap());
    }
    if let Some(path) = &args.normal_output {
        if path.exists() || path == &args.output { return Err("normal output already exists or matches GPTQ output".into()); }
        if path.parent() != args.output.parent() { return Err("normal and GPTQ output must be in the same directory".into()); }
        if progress.is_some() && path.with_file_name("progress.bin").exists() {
            return Err("normal output directory already contains progress.bin; choose a new output directory".into());
        }
    }
    let path=args.calibration_teacher.to_str().ok_or("invalid calibration path")?;
    let positions=read_teacher_positions_prefix(path,args.calibration_positions).map_err(|e|e.to_string())?;
    if positions.is_empty() {return Err("empty calibration teacher".into());}
    let ctx=bulletou_cuda_cpp::Context::new(args.cuda_cpp_device).map_err(|e|e.to_string())?;
    let device=bulletou_cuda_cpp::SfnnForwardDeviceWeights::from_host(&ctx,state.weights.as_host()).map_err(|e|e.to_string())?;
    let mut gram2=vec![0.0f64;shape.num_stacks*shape.l2_in()*shape.l2_in()];
    let mut gram3=vec![0.0f64;shape.num_stacks*shape.l2_size*shape.l2_size];
    let mut counts=vec![0usize;shape.num_stacks];
    for chunk in positions.chunks(1024) {
        let b=build_sfnn_validation_fast_batch(feature,stack,chunk,progress.as_ref())?;
        let batch=bulletou_cuda_cpp::SfnnForwardDeviceBatch::from_host(&ctx,bulletou_cuda_cpp::SfnnForwardHostBatch {
            stm_indices:&b.stm,nstm_indices:&b.nstm,buckets:&b.buckets,batch_size:b.layout.batch_size,max_active:b.layout.max_active
        }).map_err(|e|e.to_string())?;
        let ws=bulletou_cuda_cpp::SfnnForwardWorkspace::new(&ctx,bulletou_cuda_cpp::SfnnForwardWorkspaceLayout::new(shape,b.layout.batch_size)).map_err(|e|e.to_string())?;
        bulletou_cuda_cpp::sfnn_forward_device_with_factorizer_and_alpha(&ctx,&batch,&device,&ws,
            cuda_cpp_sfnn_factorizer_active(&train),cuda_cpp_sfnn_factorizer_alpha(&train)).map_err(|e|e.to_string())?;
        let x2=ws.l2_input.download(&ctx).map_err(|e|e.to_string())?;
        let x3=ws.l2.download(&ctx).map_err(|e|e.to_string())?;
        for (i,&bucket) in b.buckets.iter().enumerate() {
            let s=usize::try_from(bucket).map_err(|_|"invalid calibration bucket")?;
            if s>=shape.num_stacks {return Err("calibration bucket out of range".into());}
            counts[s]+=1;
            for (gram,input,n) in [(&mut gram2,&x2,shape.l2_in()),(&mut gram3,&x3,shape.l2_size)] {
                let x=&input[i*n..(i+1)*n]; let g=&mut gram[s*n*n..(s+1)*n*n];
                for row in 0..n { for col in 0..=row { g[row*n+col]+=f64::from(x[row])*f64::from(x[col]); } }
            }
        }
    }
    drop(device);
    let mut weights=sfnn_initial_weights_into_readback(state.weights);
    if let Some(path) = &args.normal_output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) { std::fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
        write_cuda_cpp_sfnn_nn_bin(path,feature,shape,&weights,effective_sfnn_factorizer_spec(&train),alpha,None,None,progress.as_ref())?;
        println!("normal export complete: {}",path.display());
    }
    let mut optimized=0usize;
    for (s,&count) in counts.iter().enumerate() {
        if count<args.min_bucket_positions {continue;}
        let mut replacements=Vec::new();
        for (gram,weights,rows,n) in [(&mut gram2,&weights.l2w,shape.l2_size,shape.l2_in()),(&mut gram3,&weights.l3w,1,shape.l2_size)] {
            let g=&mut gram[s*n*n..(s+1)*n*n];
            for i in 0..n {for j in 0..i {g[j*n+i]=g[i*n+j];}}
            let offset=s*rows*n;
            replacements.push(quantize_rows(&weights[offset..offset+rows*n],rows,n,g,args.damping));
        }
        if let [Ok(q2),Ok(q3)] = replacements.as_slice() {
            let o2=s*shape.l2_size*shape.l2_in(); let o3=s*shape.l2_size;
            weights.l2w[o2..o2+q2.len()].copy_from_slice(q2);
            weights.l3w[o3..o3+q3.len()].copy_from_slice(q3);
            optimized+=1;
        }
    }
    eprintln!("  calibration positions={} buckets={} optimized={} (other layers/buckets use ordinary rounding)",positions.len(),shape.num_stacks,optimized);
    if optimized==0 {return Err("no buckets had sufficient usable calibration data".into());}
    let parent=args.output.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e|e.to_string())?;
    let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e|e.to_string())?.as_nanos();
    let temp=parent.join(format!(".gptq-{}-{nonce}",std::process::id()));
    std::fs::create_dir(&temp).map_err(|e|e.to_string())?;
    struct Cleanup(PathBuf); impl Drop for Cleanup {fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
    let _cleanup=Cleanup(temp.clone());
    let out=temp.join("nn.bin");
    write_cuda_cpp_sfnn_nn_bin(&out,feature,shape,&weights,effective_sfnn_factorizer_spec(&train),alpha,None,None,progress.as_ref())?;
    if progress.is_some() {
        let src=temp.join("progress.bin"); let dst=parent.join("progress.bin");
        if dst.exists() {if std::fs::read(&src).map_err(|e|e.to_string())? != std::fs::read(&dst).map_err(|e|e.to_string())? {return Err("output folder has a different progress.bin".into());}}
        else {std::fs::hard_link(src,dst).map_err(|e|e.to_string())?;}
    }
    std::fs::hard_link(&out,&args.output).map_err(|e|e.to_string())?;
    println!("export-gptq complete: {}",args.output.display());
    if args.validate {
        let Some(mut test_args) = quantized_test_args_from_training_args(&train, args.output.clone())? else {
            return Err("GPTQ validation requires test_teacher in settings".into());
        };
        test_args.mode = QuantizedTestMode::CpuExact;
        let report = run_quantized_test_impl(&test_args, false)?;
        let loss = report.train_scale.test_loss.ok_or("GPTQ validation returned no loss")?;
        let accuracy = report.train_scale.accuracy();
        if !accuracy.is_finite() || !loss.is_finite() { return Err("GPTQ validation returned non-finite metrics".into()); }
        let record = serde_json::json!({"version":1, "epoch":args.epoch, "gptq_qacc":accuracy,
            "gptq_qloss":loss, "mode":"cpu-exact", "positions":report.records});
        let bytes = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
        std::fs::write(&validation_path, bytes).map_err(|e| format!("cannot write {}: {e}", validation_path.display()))?;
        println!("GPTQ validation: qacc={accuracy:.7} qloss={loss:.8} positions={} mode=cpu-exact file={}", report.records, validation_path.display());
    }
    Ok(())
}

fn zero_rare_l1_residuals(weights: &mut CudaCppSfnnInitialWeights, counts: &[u32], max_count: u32) -> Result<usize, String> {
    zero_rare_l1_residual_slices(weights.shape.num_stacks, &mut weights.l1w, &mut weights.l1b, counts, max_count)
}

fn zero_rare_l1_residual_slices(stacks: usize, l1w: &mut [f32], l1b: &mut [f32], counts: &[u32], max_count: u32) -> Result<usize, String> {
    if stacks == 0 || counts.len() != stacks || l1w.len() % stacks != 0 || l1b.len() % stacks != 0 {
        return Err("bucket counts or L1 tensor shape do not match checkpoint".into());
    }
    let w_stride = l1w.len() / stacks;
    let b_stride = l1b.len() / stacks;
    let mut removed = 0;
    for (stack, &count) in counts.iter().enumerate() {
        if count <= max_count {
            l1w[stack*w_stride..(stack+1)*w_stride].fill(0.0);
            l1b[stack*b_stride..(stack+1)*b_stride].fill(0.0);
            removed += 1;
        }
    }
    Ok(removed)
}

fn training_args_for_export(settings_file: &Path, epoch: usize) -> Result<Args, String> {
    let raw = vec![
        std::ffi::OsString::from("bulletou"),
        std::ffi::OsString::from("--settings-file"),
        settings_file.as_os_str().to_owned(),
    ];
    let parsed = Args::try_parse_from(expand_settings_file_args(raw)?).map_err(|e| e.to_string())?;
    args_at_epoch(&parsed, epoch)
}

/// Quantize rows of a matrix using a shared input Gram matrix and the fixed
/// NNUE i8 scale. `gram` is row-major X^T X, not its inverse.
pub(crate) fn quantize_rows(weights: &[f32], rows: usize, columns: usize, gram: &[f64], damping: f64) -> Result<Vec<f32>, String> {
    if rows == 0 || columns == 0 || weights.len() != rows.checked_mul(columns).ok_or("GPTQ dimensions overflow")?
        || gram.len() != columns.checked_mul(columns).ok_or("GPTQ dimensions overflow")?
        || !damping.is_finite() || damping <= 0.0 || weights.iter().any(|x| !x.is_finite())
        || gram.iter().any(|x| !x.is_finite()) {
        return Err("invalid GPTQ weights, Gram matrix, or damping".into());
    }
    let mut h = gram.to_vec();
    let mean_diag = (0..columns).map(|i| h[i*columns+i]).sum::<f64>() / columns as f64;
    if !mean_diag.is_finite() || mean_diag <= 0.0 { return Err("GPTQ calibration has no input energy".into()); }
    for i in 0..columns { h[i*columns+i] += damping * mean_diag; }
    let l = cholesky(&h, columns)?;
    let mut inv = vec![0.0; columns*columns];
    for col in 0..columns {
        let mut y = vec![0.0; columns];
        for i in 0..columns {
            let rhs = if i == col {1.0} else {0.0};
            y[i] = (rhs - (0..i).map(|k| l[i*columns+k]*y[k]).sum::<f64>())/l[i*columns+i];
        }
        for i in (0..columns).rev() {
            inv[i*columns+col] = (y[i] - (i+1..columns).map(|k| l[k*columns+i]*inv[k*columns+col]).sum::<f64>())/l[i*columns+i];
        }
    }
    let li = cholesky(&inv, columns)?;
    let mut result = vec![0.0; weights.len()];
    for row in 0..rows {
        let mut w: Vec<f64> = weights[row*columns..(row+1)*columns].iter().map(|&v| f64::from(v)).collect();
        for i in 0..columns {
            let q = (w[i]*64.0).round().clamp(-128.0,127.0)/64.0;
            result[row*columns+i] = q as f32;
            let error = (w[i]-q)/li[i*columns+i];
            for j in i+1..columns { w[j] -= error*li[j*columns+i]; }
        }
        // Keep independent rounding when it reconstructs this layer better
        // on the same calibration inputs. This is not a validation guarantee.
        let original=&weights[row*columns..(row+1)*columns];
        let naive:Vec<f32>=original.iter().map(|&v|((f64::from(v)*64.0).round().clamp(-128.0,127.0)/64.0) as f32).collect();
        let loss=|q:&[f32]| -> f64 {
            let d:Vec<f64>=original.iter().zip(q).map(|(&a,&b)|f64::from(a)-f64::from(b)).collect();
            (0..columns).map(|i| d[i]*(0..columns).map(|j|gram[i*columns+j]*d[j]).sum::<f64>()).sum()
        };
        let candidate=&mut result[row*columns..(row+1)*columns];
        let (candidate_loss,naive_loss)=(loss(candidate),loss(&naive));
        if !candidate_loss.is_finite() || candidate_loss>naive_loss {candidate.copy_from_slice(&naive);}
    }
    Ok(result)
}

fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut l=vec![0.0; n*n];
    for i in 0..n {
        for j in 0..=i {
            let x=a[i*n+j]-(0..j).map(|k|l[i*n+k]*l[j*n+k]).sum::<f64>();
            l[i*n+j]=if i==j {
                if !x.is_finite() || x<=0.0 {return Err("GPTQ calibration matrix is not positive definite".into());}
                x.sqrt()
            } else {x/l[j*n+j]};
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rare_residual_removal_uses_inclusive_threshold_and_preserves_other_buckets() {
        let mut w = vec![1., 2., 3., 4., 5., 6.];
        let mut b = vec![7., 8., 9.];
        assert_eq!(zero_rare_l1_residual_slices(3, &mut w, &mut b, &[0, 4, 5], 4).unwrap(), 2);
        assert_eq!(w, [0., 0., 0., 0., 5., 6.]);
        assert_eq!(b, [0., 0., 9.]);
        assert!(zero_rare_l1_residual_slices(3, &mut w, &mut b, &[0], 0).is_err());
    }
    #[test]
    fn bucket_csv_sorts_counts_and_includes_zeros() {
        let path = std::env::temp_dir().join(format!("bulletou-bucket-sorted-{}.csv", std::process::id()));
        let counts = SfnnBucketCounts { arch: "test".into(), positions: 7, counts: vec![0, 3, 1, 3] };
        counts.write_sorted_csv(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "bucket,count\n1,3\n3,3\n2,1\n0,0\n");
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn export_preserves_settings_path_and_epoch_schedules_with_automatic_gptq() {
        let dir = std::env::temp_dir().join(format!("bulletou-gptq-settings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        std::fs::write(&path, r#"{
            "arch": "SFNN_halfkahm2_2304_15_64_k9k9_progress8",
            "teacher": "teacher.bin",
            "sfnn_gptq": {"epoch1": true, "epoch13": false},
            "sfnn_qat_l1": {"epoch1": true, "epoch6": false}
        }"#).unwrap();
        let first = training_args_for_export(&path, 1).unwrap();
        let sixth = training_args_for_export(&path, 6).unwrap();
        assert_eq!(sixth.settings_file.as_deref(), Some(path.as_path()));
        assert!(sixth.sfnn_gptq);
        assert!(!training_args_for_export(&path, 13).unwrap().sfnn_gptq);
        assert!(first.sfnn_qat_l1);
        assert!(!sixth.sfnn_qat_l1);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn correlated_inputs_can_compensate_rounding() {
        let w=[0.49/64.0, 0.49/64.0];
        let q=quantize_rows(&w,1,2,&[1.0,1.0,1.0,1.0],0.01).unwrap();
        let err=(f64::from(w[0]+w[1])-f64::from(q[0]+q[1])).abs();
        assert!(err < f64::from(w[0]+w[1]).abs()*0.1, "{q:?}");
    }
    #[test]
    fn invalid_or_empty_calibration_is_rejected() {
        assert!(quantize_rows(&[1.0],1,1,&[0.0],0.01).is_err());
        assert!(quantize_rows(&[f32::NAN],1,1,&[1.0],0.01).is_err());
        assert!(quantize_rows(&[1.0],1,1,&[1.0],0.0).is_err());
    }
    #[test]
    fn independent_inputs_reduce_to_nearest_with_clipping() {
        let w=[2.5,-3.0,0.123,-0.321];
        let q=quantize_rows(&w,2,2,&[2.0,0.0,0.0,1.0],0.01).unwrap();
        for (a,b) in q.iter().zip(w) { assert_eq!(*a, (f64::from(b)*64.0).round().clamp(-128.0,127.0) as f32/64.0); }
    }
    #[test]
    fn calibration_reconstruction_is_no_worse_than_nearest_rounding() {
        let w=[0.3, -0.7, 0.1, 0.9, 0.333, -0.121];
        let g=[3.0,1.0,0.6, 1.0,2.0,-0.2, 0.6,-0.2,1.0];
        let q=quantize_rows(&w,2,3,&g,0.01).unwrap();
        for row in 0..2 {
            let original=&w[row*3..row*3+3];
            let naive:Vec<f32>=original.iter().map(|v|(f64::from(*v)*64.0).round().clamp(-128.0,127.0) as f32/64.0).collect();
            let loss=|values:&[f32]| {
                let d:Vec<f64>=original.iter().zip(values).map(|(&a,&b)| f64::from(a-b)).collect();
                (0..3).map(|i|(0..3).map(|j|d[i]*g[i*3+j]*d[j]).sum::<f64>()).sum::<f64>()
            };
            assert!(loss(&q[row*3..row*3+3])<=loss(&naive)+1e-12);
        }
    }
    #[test]
    fn automatic_calibration_file_selection_is_stable() {
        let dir=std::env::temp_dir().join(format!("bulletou-gptq-teacher-test-{}",std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        for name in ["z.bin","a.bin","README.md"] {std::fs::write(dir.join(name),[]).unwrap();}
        let mut args=Args::try_parse_from(["bulletou","--teacher",dir.to_str().unwrap()]).unwrap();
        assert_eq!(calibration_teacher(&args).unwrap(),dir.join("a.bin"));
        args.sfnn_gptq_calibration_teacher=Some(dir.join("z.bin"));
        assert_eq!(calibration_teacher(&args).unwrap(),dir.join("z.bin"));
        for name in ["z.bin","a.bin","README.md"] {std::fs::remove_file(dir.join(name)).unwrap();}
        std::fs::remove_dir(dir).unwrap();
    }
    #[test]
    #[ignore = "requires GPU and BULLETOU_GPTQ_TEST_CHECKPOINT / BULLETOU_GPTQ_TEST_TEACHER"]
    fn automatic_export_from_real_checkpoint() {
        let source=PathBuf::from(std::env::var_os("BULLETOU_GPTQ_TEST_CHECKPOINT").unwrap());
        let teacher=PathBuf::from(std::env::var_os("BULLETOU_GPTQ_TEST_TEACHER").unwrap());
        let dir=source.parent().unwrap().join(format!("bulletou-auto-gptq-test-{}",std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        struct Cleanup(PathBuf); impl Drop for Cleanup {fn drop(&mut self){
            for name in ["state.bin","bulletou-settings.json","nn-gptq.bin","progress.bin","gptq-validation.json"] {let _=std::fs::remove_file(self.0.join(name));}
            let _=std::fs::remove_dir(&self.0);
        }}
        let _cleanup=Cleanup(dir.clone());
        std::fs::hard_link(source.join("state.bin"),dir.join("state.bin")).unwrap();
        std::fs::copy(source.join("bulletou-settings.json"),dir.join("bulletou-settings.json")).unwrap();
        let mut raw=vec![std::ffi::OsString::from("bulletou")];
        raw.extend(bulletou_settings_json_args(&dir.join("bulletou-settings.json")).unwrap());
        let mut args=args_at_epoch(&Args::try_parse_from(raw).unwrap(),5).unwrap();
        args.sfnn_gptq=true; args.sfnn_gptq_calibration_teacher=Some(teacher); args.sfnn_gptq_calibration_positions=65536;
        args.test_positions=Some(ValidationPositionCount::Count(2048));
        run_automatic(&args,&dir,5,1296).unwrap();
        assert!(dir.join("nn-gptq.bin").is_file());
        assert!(dir.join("progress.bin").is_file());
        let metrics:serde_json::Value=serde_json::from_slice(&std::fs::read(dir.join("gptq-validation.json")).unwrap()).unwrap();
        assert_eq!(metrics["mode"],"cpu-exact"); assert_eq!(metrics["epoch"],5); assert_eq!(metrics["superbatch"],1296);
        assert!(metrics["gptq_qacc"].as_f64().is_some()); assert!(metrics["gptq_qloss"].as_f64().is_some());
        assert!(run_automatic(&args,&dir,5,1296).is_err(),"must not overwrite output");
    }
}
