/*!
bulletou — BulletOu trainer entry point.

Dispatches to the appropriate training routine via `--eval-type`. The
"family" eval-types train all three KPPT components (KK + KKP + KPP)
sequentially in a single invocation and assemble the result into
`<output>/final/`:

    bulletou --eval-type KPPT            (KPPT family, KPP int16 × 2)
    bulletou --eval-type KPP_KKPT        (KPP_KKPT factorised, KPP int16)

To train a single component standalone (= for development / smoke testing):

    bulletou --eval-type KPPT_KK         KK only
    bulletou --eval-type KPPT_KKP        KKP only
    bulletou --eval-type KPPT_KPP        KPP only, KPPT layout
    bulletou --eval-type KPP_KKPT_KPP    KPP only, KPP_KKPT layout

Teacher data is given via `--teacher`. The argument is either a single
file (`.hcpe` / `.hcpe3` / `.pack` / `.psv`), a directory containing such
files (all matching files are concatenated), or a comma-separated list
of either. Format is inferred from the file extension; all files must
share the same extension.

Usage:

    cargo run --release --features device-cuda --example bulletou -- \
        --eval-type KPPT \
        --teacher /data/shogi/train_set/ \
        --output checkpoints/my-kppt \
        --superbatches 20
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::{ShogiKk, ShogiKkp, ShogiKpp},
    nn::optimiser,
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader},
        yaneuraou_kppt::{
            KppFormat, bundle_component_state, parse_model_weights_bin, save_yaneuraou_eval,
            unbundle_component_state,
        },
    },
};
use clap::{Parser, ValueEnum};

// ----- eval-type ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "SCREAMING_SNAKE_CASE")]
enum EvalType {
    /// KPPT family: train KK, KKP, and KPP sequentially and assemble the
    /// three-file KPPT eval (`KK_synthesized.bin` / `KKP_synthesized.bin` /
    /// `KPP_synthesized.bin`) into `<output>/final/`.
    Kppt,
    /// KPP_KKPT family (factorised KPPT): same as `KPPT` but KPP is written
    /// in the KPP_KKPT layout (no turn channel; half the KPP file size).
    KppKkpt,
    /// KPPT KK component only.
    KpptKk,
    /// KPPT KKP component only.
    KpptKkp,
    /// KPPT KPP component only (with turn channel; ~740 MB).
    KpptKpp,
    /// KPP_KKPT KPP component only (no turn channel; ~388 MB).
    KppKkptKpp,
}

impl EvalType {
    fn default_net_id(self) -> &'static str {
        match self {
            EvalType::Kppt => "shogi_kppt",
            EvalType::KppKkpt => "shogi_kpp_kkpt",
            EvalType::KpptKk => "shogi_kk",
            EvalType::KpptKkp => "shogi_kkp",
            EvalType::KpptKpp => "shogi_kpp",
            EvalType::KppKkptKpp => "shogi_kpp_factorised",
        }
    }

    fn default_output(self) -> &'static str {
        match self {
            EvalType::Kppt => "checkpoints/shogi_kppt",
            EvalType::KppKkpt => "checkpoints/shogi_kpp_kkpt",
            EvalType::KpptKk => "checkpoints/shogi_kk",
            EvalType::KpptKkp => "checkpoints/shogi_kkp",
            EvalType::KpptKpp => "checkpoints/shogi_kpp",
            EvalType::KppKkptKpp => "checkpoints/shogi_kpp_factorised",
        }
    }

    /// Suggested f32 -> i{16,32} quantisation scale for the YaneuraOu writer.
    /// KK / KKP entries are i32 (large dynamic range) so 4000 = eval_scale * 10.
    /// KPP entries are i16 (smaller dynamic range) so the scale is an
    /// order of magnitude smaller.
    fn default_yaneuraou_quant_scale(self) -> f32 {
        match self {
            EvalType::Kppt | EvalType::KppKkpt | EvalType::KpptKk | EvalType::KpptKkp => 4000.0,
            EvalType::KpptKpp | EvalType::KppKkptKpp => 400.0,
        }
    }

    /// On-disk KPP layout to write at checkpoint time. KK / KKP eval types
    /// don't have a KPP file so this is ignored.
    fn kpp_format(self) -> KppFormat {
        match self {
            EvalType::KppKkpt | EvalType::KppKkptKpp => KppFormat::KppKkpt,
            _ => KppFormat::Kppt,
        }
    }
}

// (teacher-path expansion and format inference live in
//  `bullet_lib::teacher_path` so the single-component examples can share them.)

// ----- CLI ---------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
#[command(name = "bulletou")]
#[command(about = "BulletOu unified trainer")]
struct Args {
    /// Evaluation function type to train.
    #[arg(long, value_enum)]
    eval_type: EvalType,

    /// Teacher data: either a single file (`.hcpe` / `.hcpe3` / `.pack` /
    /// `.psv`), a directory containing such files (all matching files are
    /// concatenated), or a comma-separated list of either. Format is
    /// inferred from the extension; all included files must share the same
    /// extension.
    #[arg(long)]
    teacher: String,

    /// Checkpoint output directory. Defaults to a per-eval-type path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Net identifier (prefix of the saved checkpoint subdirectory).
    /// Defaults to a per-eval-type name.
    #[arg(long)]
    net_id: Option<String>,

    /// Mini-batch size (positions per gradient step).
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of mini-batches per superbatch. Default ≈ 100M positions per
    /// superbatch (100_000_000 / batch_size).
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// Cap on the number of superbatches per epoch. If omitted, there is no
    /// cap (= run until the dataloader reaches EOF). Specify this to stop
    /// each epoch early (e.g. to fit a quick smoke test). Mutually exclusive
    /// with `--max-epochs` in practical use.
    #[arg(long)]
    superbatches: Option<usize>,

    /// Number of epochs to train. One epoch = one full pass through the
    /// teacher data (= one dataloader EOF). After each epoch the dataloader
    /// is rebuilt from scratch and the LR scheduler restarts at superbatch
    /// 1, so for example `--lr-step 8` applies independently within each
    /// epoch. Default 1.
    #[arg(long, default_value = "1")]
    max_epochs: usize,

    /// Starting superbatch counter (>1 to resume / extend).
    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    /// Initial Adam learning rate.
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// LR gamma (multiplicative drop applied every `lr_step` superbatches).
    #[arg(long, default_value = "0.1")]
    lr_gamma: f32,

    /// LR step: apply `lr_gamma` every N superbatches.
    #[arg(long, default_value = "8")]
    lr_step: usize,

    /// Start of the WDL linear schedule (0 = pure eval, 1 = pure game result).
    #[arg(long, default_value = "0.0")]
    start_wdl: f32,

    /// End of the WDL linear schedule.
    #[arg(long, default_value = "1.0")]
    end_wdl: f32,

    /// Eval-to-score sigmoid scale.
    #[arg(long, default_value = "400")]
    scale: u32,

    /// f32 -> integer quantisation scale for the YaneuraOu output. If
    /// omitted, an eval-type-specific default is used (4000 for KK/KKP,
    /// 400 for KPP).
    #[arg(long)]
    yaneuraou_quant_scale: Option<f32>,

    /// Save every N superbatches (1 = save every superbatch, 5 = every 5th).
    #[arg(long, default_value = "1")]
    save_rate: usize,

    /// Dataloader worker threads (CPU side).
    #[arg(long, default_value = "4")]
    threads: usize,

    /// GPU-side batch queue depth.
    #[arg(long, default_value = "32")]
    batch_queue_size: usize,

    /// Loader shuffle buffer size in megabytes.
    #[arg(long, default_value = "256")]
    buffer_mb: usize,

    /// Drop positions whose |score| >= this. Useful to exclude ±32000 mate
    /// stamps. Set to 0 to disable.
    #[arg(long, default_value = "32000")]
    score_drop_abs: u16,
}

impl Args {
    fn output_dir(&self) -> PathBuf {
        self.output.clone().unwrap_or_else(|| PathBuf::from(self.eval_type.default_output()))
    }

    fn net_id(&self) -> String {
        self.net_id.clone().unwrap_or_else(|| self.eval_type.default_net_id().to_string())
    }

    fn yaneuraou_scale(&self) -> f32 {
        self.yaneuraou_quant_scale.unwrap_or_else(|| self.eval_type.default_yaneuraou_quant_scale())
    }

    fn kpp_format(&self) -> KppFormat {
        self.eval_type.kpp_format()
    }
}

// ----- dispatch ----------------------------------------------------------

fn main() {
    let args = Args::parse();
    match args.eval_type {
        EvalType::Kppt | EvalType::KppKkpt => run_kppt_all(&args),
        // Single-component eval-types do not currently auto-resume. (For
        // resume, drive the full family with `--eval-type KPPT` / `KPP_KKPT`.)
        EvalType::KpptKk => run_kppt_kk(&args, None),
        EvalType::KpptKkp => run_kppt_kkp(&args, None),
        // KPP trains the same network for both the KPPT and KPP_KKPT layouts;
        // only the writer differs, selected inside `run_training_inline!` via
        // `args.kpp_format()`.
        EvalType::KpptKpp | EvalType::KppKkptKpp => run_kppt_kpp(&args, None),
    }
}

/// Count numbered subdirectories under `output_dir` whose names parse as
/// `usize`. Used so a resumed run extends the numbering rather than
/// overwriting the previous run's checkpoint dirs.
fn count_existing_numbered_dirs(output_dir: &std::path::Path) -> usize {
    let Ok(rd) = std::fs::read_dir(output_dir) else { return 0 };
    rd.flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| name.parse::<usize>().is_ok())
        .count()
}

/// Find the latest numbered subdirectory under `output_dir` (4-or-more-digit
/// name parsable as `usize`) whose `state.bin` exists. Returns `None` if no
/// resumable checkpoint is found.
fn find_latest_state_bin(output_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut latest: Option<(usize, std::path::PathBuf)> = None;
    let rd = std::fs::read_dir(output_dir).ok()?;
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let Ok(n) = name.parse::<usize>() else { continue };
        let state_bin = path.join("state.bin");
        if !state_bin.is_file() {
            continue;
        }
        match &latest {
            None => latest = Some((n, state_bin)),
            Some((m, _)) if n > *m => latest = Some((n, state_bin)),
            _ => {}
        }
    }
    latest.map(|(_, p)| p)
}

// ----- KPPT family: KK + KKP + KPP sequential dispatch -------------------

/// Run the three KPPT components (KK, KKP, KPP) sequentially, then assemble
/// the three resulting `.bin` files into `<output>/final/` so the engine has
/// a single directory to point at.
///
/// `--eval-type KPPT` uses the KPPT KPP layout (int16 × 2, with turn channel).
/// `--eval-type KPP_KKPT` uses the KPP_KKPT KPP layout (int16, no turn channel).
fn run_kppt_all(args: &Args) {
    let output_dir = args.output_dir();

    let kpp_eval_type = match args.eval_type {
        EvalType::Kppt => EvalType::KpptKpp,
        EvalType::KppKkpt => EvalType::KppKkptKpp,
        _ => unreachable!("run_kppt_all called with non-family eval_type"),
    };

    eprintln!("=== bulletou: running {} family (3 components) ===", match args.eval_type {
        EvalType::Kppt => "KPPT",
        EvalType::KppKkpt => "KPP_KKPT",
        _ => unreachable!(),
    });

    // ---- Resume support -------------------------------------------------
    // If `<output>` already contains a numbered dir with a `state.bin`,
    // unbundle each component's records into a per-component
    // `optimiser_state/` triplet under `<output>/.bulletou_resume/<comp>/`,
    // and let each child run_kppt_* call `trainer.load_from_checkpoint(<comp>)`
    // immediately after building its trainer.
    let resume_state_bin = find_latest_state_bin(&output_dir);
    let resume_dirs: Option<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> =
        resume_state_bin.as_ref().map(|state_bin_path| {
            eprintln!("=== resume detected: {} ===", state_bin_path.display());
            let bytes = std::fs::read(state_bin_path).unwrap_or_else(|e| {
                eprintln!("error: failed to read {}: {e}", state_bin_path.display());
                std::process::exit(1);
            });
            let records = parse_model_weights_bin(&bytes).unwrap_or_else(|e| {
                eprintln!("error: failed to parse state.bin: {e}");
                std::process::exit(1);
            });
            let resume_root = output_dir.join(".bulletou_resume");
            // Fresh extraction each run; old contents may correspond to a
            // different save point.
            let _ = std::fs::remove_dir_all(&resume_root);
            let mut paths: Vec<std::path::PathBuf> = Vec::new();
            for comp in ["kk", "kkp", "kpp"] {
                let comp_dir = resume_root.join(comp);
                unbundle_component_state(&records, comp, &comp_dir.join("optimiser_state")).unwrap_or_else(
                    |e| {
                        eprintln!("error: state.bin missing `{comp}/*` records: {e}");
                        std::process::exit(1);
                    },
                );
                paths.push(comp_dir);
            }
            (paths[0].clone(), paths[1].clone(), paths[2].clone())
        });

    for (label, child_eval_type, net_id, resume_dir) in [
        ("KK", EvalType::KpptKk, "kk", resume_dirs.as_ref().map(|d| d.0.clone())),
        ("KKP", EvalType::KpptKkp, "kkp", resume_dirs.as_ref().map(|d| d.1.clone())),
        ("KPP", kpp_eval_type, "kpp", resume_dirs.as_ref().map(|d| d.2.clone())),
    ] {
        eprintln!("\n=== [{label}] training ===");
        let mut child = args.clone();
        child.eval_type = child_eval_type;
        child.net_id = Some(net_id.to_string());
        // Force the child's yaneuraou_quant_scale default to match the child's
        // eval-type when the user didn't override it.
        if args.yaneuraou_quant_scale.is_none() {
            child.yaneuraou_quant_scale = Some(child_eval_type.default_yaneuraou_quant_scale());
        }
        match child_eval_type {
            EvalType::KpptKk => run_kppt_kk(&child, resume_dir.as_deref()),
            EvalType::KpptKkp => run_kppt_kkp(&child, resume_dir.as_deref()),
            EvalType::KpptKpp | EvalType::KppKkptKpp => run_kppt_kpp(&child, resume_dir.as_deref()),
            _ => unreachable!(),
        }
    }

    // Cleanup the scratch resume dir if it was used.
    let _ = std::fs::remove_dir_all(output_dir.join(".bulletou_resume"));

    // Re-organise per-component checkpoint subdirs into a flat, zero-padded
    // series `0001/`, `0002/`, ..., each containing the three `.bin` files
    // at the corresponding save point. The original `kk-*/` / `kkp-*/` /
    // `kpp-*/` subdirs are removed after assembly.
    match assemble_numbered_dirs(&output_dir) {
        Ok((first_idx, last_idx)) => {
            // Append the new run's full loss history to a top-level
            // `<output>/learn.log` so the user has a single growing file
            // spanning all resumes. Per-save `<output>/0NNN/learn.log` files
            // are kept as snapshots.
            if let Err(e) = append_to_top_level_log(&output_dir, first_idx, last_idx) {
                eprintln!(
                    "warning: failed to update {}: {e}",
                    output_dir.join("learn.log").display()
                );
            }
        }
        Err(e) => {
            eprintln!("error: failed to assemble numbered checkpoint dirs: {e}");
            std::process::exit(1);
        }
    }
}

/// Format the current wall-clock time as `YYYY-MM-DDTHH:MM:SSZ` (UTC).
/// Inlined gregorian conversion so we don't have to pull in `chrono`.
fn current_utc_iso8601() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let tod = secs % 86400;
    let hours = tod / 3600;
    let minutes = (tod % 3600) / 60;
    let seconds = tod % 60;
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0usize;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md as i64 {
            m = i;
            break;
        }
        remaining_days -= md as i64;
    }
    let d = remaining_days + 1;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m + 1, d, hours, minutes, seconds)
}

/// Append a "run section" to `<output>/learn.log`. The section header records
/// the timestamp and the range of numbered dirs produced by this run, and the
/// body is the contents of the latest dir's `learn.log` (= the full per-batch
/// loss history of all three components for this run).
fn append_to_top_level_log(
    output_dir: &std::path::Path,
    first_idx: usize,
    last_idx: usize,
) -> std::io::Result<()> {
    use std::io::Write;
    let latest_log = output_dir.join(format!("{last_idx:04}")).join("learn.log");
    let body = std::fs::read_to_string(&latest_log)?;
    let timestamp = current_utc_iso8601();
    let header = format!("# === run @ {timestamp} saved {first_idx:04}/-{last_idx:04}/ ===\n");
    let top = output_dir.join("learn.log");
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&top)?;
    file.write_all(header.as_bytes())?;
    file.write_all(body.as_bytes())?;
    if !body.ends_with('\n') {
        file.write_all(b"\n")?;
    }
    Ok(())
}

/// List checkpoint subdirs for `net_id_prefix` under `output_dir`, sorted by
/// `(epoch, sb)`. Subdir names are `<prefix>-<sb>` (single-epoch) or
/// `<prefix>-e<epoch>-<sb>` (multi-epoch).
fn list_component_checkpoints_sorted(
    output_dir: &std::path::Path,
    net_id_prefix: &str,
) -> Vec<std::path::PathBuf> {
    let mut entries: Vec<(usize, usize, std::path::PathBuf)> = Vec::new();
    let prefix = format!("{net_id_prefix}-");
    let Ok(rd) = std::fs::read_dir(output_dir) else { return Vec::new() };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        let parsed: Option<(usize, usize)> = (|| {
            if let Some(after_e) = rest.strip_prefix('e') {
                let (e_str, sb_str) = after_e.split_once('-')?;
                Some((e_str.parse().ok()?, sb_str.parse().ok()?))
            } else {
                rest.parse::<usize>().ok().map(|sb| (1, sb))
            }
        })();
        let Some((epoch, sb)) = parsed else { continue };
        entries.push((epoch, sb, path));
    }
    entries.sort();
    entries.into_iter().map(|(_, _, p)| p).collect()
}

/// Walk the per-component checkpoint subdirs (`kk-*` / `kkp-*` / `kpp-*`)
/// produced by the three children of `run_kppt_all`, and assemble them into
/// flat `<output>/0001/`, `0002/`, ... directories each containing the
/// three `.bin` files. Removes the per-component subdirs after assembly so
/// the user sees a clean numbered layout.
///
/// Returns `(first_idx, last_idx)` of the numbered dirs written in this run
/// (1-based, inclusive). On resume the range starts above the previously
/// existing count, so the caller can locate the latest dir to inspect.
fn assemble_numbered_dirs(output_dir: &std::path::Path) -> std::io::Result<(usize, usize)> {
    let kk_dirs = list_component_checkpoints_sorted(output_dir, "kk");
    let kkp_dirs = list_component_checkpoints_sorted(output_dir, "kkp");
    let kpp_dirs = list_component_checkpoints_sorted(output_dir, "kpp");

    let n = kk_dirs.len().min(kkp_dirs.len()).min(kpp_dirs.len());
    if n == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "no checkpoint subdirs under {} (kk={}, kkp={}, kpp={})",
                output_dir.display(),
                kk_dirs.len(),
                kkp_dirs.len(),
                kpp_dirs.len()
            ),
        ));
    }
    if kk_dirs.len() != n || kkp_dirs.len() != n || kpp_dirs.len() != n {
        eprintln!(
            "  warning: component save counts differ (kk={}, kkp={}, kpp={}); using the common prefix of {n}",
            kk_dirs.len(),
            kkp_dirs.len(),
            kpp_dirs.len()
        );
    }

    // When resuming, do not overwrite the previous run's numbered dirs --
    // start at `existing_count + 1` so new saves extend the series.
    let existing_count = count_existing_numbered_dirs(output_dir);

    eprintln!(
        "\n=== assembling {n} checkpoint dir(s) under {} (starting at #{}) ===",
        output_dir.display(),
        existing_count + 1
    );
    for i in 0..n {
        let idx = existing_count + i + 1;
        let dst = output_dir.join(format!("{idx:04}"));
        std::fs::create_dir_all(&dst)?;
        // engine-facing quantised .bin files
        std::fs::copy(kk_dirs[i].join("KK_synthesized.bin"), dst.join("KK_synthesized.bin"))?;
        std::fs::copy(kkp_dirs[i].join("KKP_synthesized.bin"), dst.join("KKP_synthesized.bin"))?;
        std::fs::copy(kpp_dirs[i].join("KPP_synthesized.bin"), dst.join("KPP_synthesized.bin"))?;
        // bundle the three components' resume state (Adam weights + momentum + velocity)
        // into a single `state.bin` so the dir holds everything needed to resume.
        let mut state_buf: Vec<u8> = Vec::new();
        bundle_component_state(&mut state_buf, "kk", &kk_dirs[i].join("optimiser_state"))?;
        bundle_component_state(&mut state_buf, "kkp", &kkp_dirs[i].join("optimiser_state"))?;
        bundle_component_state(&mut state_buf, "kpp", &kpp_dirs[i].join("optimiser_state"))?;
        std::fs::write(dst.join("state.bin"), &state_buf)?;
        // Bundle the three components' bullet `log.txt` (CSV
        // `superbatch,batch,loss` lines accumulated up to this save) into a
        // single `learn.log` with per-component section headers.
        let mut log_buf = String::new();
        for (label, dir) in [("kk", &kk_dirs[i]), ("kkp", &kkp_dirs[i]), ("kpp", &kpp_dirs[i])] {
            log_buf.push_str(&format!("# component: {label}\n"));
            match std::fs::read_to_string(dir.join("log.txt")) {
                Ok(s) => log_buf.push_str(&s),
                Err(_) => log_buf.push_str("# (log.txt missing)\n"),
            }
            if !log_buf.ends_with('\n') {
                log_buf.push('\n');
            }
        }
        std::fs::write(dst.join("learn.log"), log_buf)?;
        eprintln!("  -> {}/", dst.display());
    }

    // Remove the now-redundant per-component subdirs.
    for d in kk_dirs.iter().chain(kkp_dirs.iter()).chain(kpp_dirs.iter()) {
        if let Err(e) = std::fs::remove_dir_all(d) {
            eprintln!("  warning: failed to remove {}: {e}", d.display());
        }
    }

    Ok((existing_count + 1, existing_count + n))
}

// `Trainer<G, O, S>` の concrete type は bullet API として直接露出していないので、
// 3 branch を generic helper でまとめる代わりに、共通の schedule / settings /
// loader dispatch をマクロで encapsulate する。
//
// 各 branch は: (a) save_format / weight ID を決め、(b) ValueTrainerBuilder で
// trainer を構築し、(c) `run_training_inline!(args, trainer)` を呼ぶ。
macro_rules! run_training_inline {
    ($args:expr, $trainer:expr) => {{
        let args: &Args = $args;
        let trainer = $trainer;

        let batches_per_superbatch =
            args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));

        let data_files_owned = expand_teacher(&args.teacher).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });
        let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

        let format = infer_data_format(&data_files_ref).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });

        // --superbatches が未指定なら epoch ごとに loader EOF まで回す (= usize::MAX で
        // 上限なし、loader 側で EOF が来たら trainer.run が返る)。
        let end_superbatch = args.superbatches.unwrap_or(usize::MAX);

        let net_id_base = args.net_id();
        let output_dir_buf = args.output_dir();
        let yaneuraou_scale = args.yaneuraou_scale();
        let kpp_format = args.kpp_format();
        let max_epochs = args.max_epochs.max(1);

        let output_dir_str = args.output_dir();
        let output_dir = output_dir_str.to_str().unwrap_or("checkpoints");

        // Tracks whether bullet fired the save callback at least once across
        // all epochs. If 教師 is smaller than a single superbatch (or any
        // other reason no superbatch boundary is crossed), bullet writes no
        // checkpoint at all and we'd end up with an empty output dir. After
        // all epochs finish we check this flag and, if no save happened, do
        // a final fallback save so at least the current trainer state is
        // persisted. This is *not* an EOF-triggered save — it fires exactly
        // once per training run and only as a last resort.
        let saved_any = std::cell::Cell::new(false);
        // Remember the last per-epoch net_id we used so the fallback save can
        // reuse the same naming convention (so assembly pairs the dirs by
        // sort order alongside any future numbered checkpoints).
        let mut last_net_id_for_epoch: String = net_id_base.clone();
        // The error_record returned by the most recent `trainer.run` call.
        // bullet writes `log.txt` itself at each save, but if zero saves
        // happened we need to write it ourselves in the fallback path.
        let mut last_error_record: Vec<(usize, usize, f32)> = Vec::new();

        for epoch in 1..=max_epochs {
            if max_epochs > 1 {
                eprintln!("\n=== epoch {epoch} / {max_epochs} ===");
            }

            // checkpoint dir 名は max_epochs=1 のとき従来通り `<net_id>-<superbatch>`、
            // 複数 epoch のときは `<net_id>-e<epoch>-<superbatch>` で重複を避ける。
            let net_id_for_epoch = if max_epochs > 1 {
                format!("{net_id_base}-e{epoch}")
            } else {
                net_id_base.clone()
            };
            last_net_id_for_epoch = net_id_for_epoch.clone();

            let schedule = TrainingSchedule {
                net_id: net_id_for_epoch.clone(),
                eval_scale: args.scale as f32,
                steps: TrainingSteps {
                    batch_size: args.batch_size,
                    batches_per_superbatch,
                    start_superbatch: args.start_superbatch,
                    end_superbatch,
                },
                wdl_scheduler: wdl::LinearWDL { start: args.start_wdl, end: args.end_wdl },
                lr_scheduler: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
                save_rate: args.save_rate,
            };

            let net_id_for_cb = net_id_for_epoch.clone();
            let output_dir_for_cb = output_dir_buf.clone();
            let saved_any_ref = &saved_any;
            let on_checkpoint_saved = move |superbatch: usize| {
                saved_any_ref.set(true);
                let ckpt_dir = output_dir_for_cb.join(format!("{net_id_for_cb}-{superbatch}"));
                match save_yaneuraou_eval(&ckpt_dir, yaneuraou_scale, kpp_format) {
                    Ok(()) => eprintln!("  also wrote YaneuraOu eval binary in {}", ckpt_dir.display()),
                    Err(e) => {
                        eprintln!("  WARN: failed to write YaneuraOu eval binary in {}: {e}", ckpt_dir.display())
                    }
                }
            };

            let settings = LocalSettings {
                threads: args.threads,
                test_set: None,
                output_directory: output_dir,
                batch_queue_size: args.batch_queue_size,
                on_checkpoint_saved: Some(&on_checkpoint_saved),
            };

            last_error_record = match format {
                DataFormat::Hcpe => {
                    let loader =
                        HcpeDataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Hcpe3 => {
                    let loader =
                        Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Pack => {
                    let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                        .with_single_epoch(true);
                    trainer.run(&schedule, &settings, &loader)
                }
                DataFormat::Psv => {
                    let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
                    trainer.run(&schedule, &settings, &loader)
                }
            };
        }

        // End-of-training fallback save (see the comment on `saved_any`):
        // executes only when bullet never crossed a superbatch boundary.
        if !saved_any.get() {
            let ckpt_dir = output_dir_buf.join(format!("{last_net_id_for_epoch}-1"));
            eprintln!(
                "  WARN: no superbatch completed during training (教師 < 1 superbatch); writing fallback save to {}",
                ckpt_dir.display()
            );
            let ckpt_dir_str = ckpt_dir.to_str().expect("checkpoint path is utf-8");
            trainer.save_to_checkpoint(ckpt_dir_str);
            // bullet's save loop normally writes `log.txt` itself, but for the
            // fallback path no save ever fired, so write the in-memory loss
            // record (same `superbatch,batch,loss` CSV format) ourselves.
            if let Err(e) = write_loss_csv(&ckpt_dir.join("log.txt"), &last_error_record) {
                eprintln!("  WARN: failed to write log.txt in {}: {e}", ckpt_dir.display());
            }
            match save_yaneuraou_eval(&ckpt_dir, yaneuraou_scale, kpp_format) {
                Ok(()) => eprintln!("  also wrote YaneuraOu eval binary in {}", ckpt_dir.display()),
                Err(e) => eprintln!("  WARN: failed to write YaneuraOu eval binary in {}: {e}", ckpt_dir.display()),
            }
        }
    }};
}

/// Write loss records as CSV (`superbatch,batch,loss`), matching the format
/// bullet writes to `log.txt` at each save. Used by the end-of-training
/// fallback save path.
fn write_loss_csv(path: &std::path::Path, records: &[(usize, usize, f32)]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (sb, b, loss) in records {
        writeln!(file, "{sb},{b},{loss}")?;
    }
    Ok(())
}

// ----- KPPT: KK ---------------------------------------------------------

fn run_kppt_kk(args: &Args, resume_dir: Option<&std::path::Path>) {
    let qa: i16 = 256;
    let qb: i16 = 64;
    let qab: i16 = qa.checked_mul(qb).expect("qa*qb fits in i16");

    let save_format: Vec<SavedFormat> = vec![
        SavedFormat::id("kkw").round().quantise::<i16>(qa),
        SavedFormat::id("kkb").round().quantise::<i16>(qa),
        SavedFormat::id("outw").transpose().round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qab),
    ];

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKk)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let kk = builder.new_affine("kk", 6561, 1);
        let out = builder.new_affine("out", 2, 1);
        let stm_eval = kk.forward(stm_inputs);
        let ntm_eval = kk.forward(ntm_inputs);
        let combined = stm_eval.concat(ntm_eval);
        out.forward(combined)
    });

    if let Some(dir) = resume_dir {
        eprintln!("  [KK] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT: KKP --------------------------------------------------------

fn run_kppt_kkp(args: &Args, resume_dir: Option<&std::path::Path>) {
    let qa: i16 = 256;
    let qb: i16 = 64;
    let qab: i16 = qa.checked_mul(qb).expect("qa*qb fits in i16");

    let save_format: Vec<SavedFormat> = vec![
        SavedFormat::id("kkpw").round().quantise::<i16>(qa),
        SavedFormat::id("kkpb").round().quantise::<i16>(qa),
        SavedFormat::id("outw").transpose().round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qab),
    ];

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKkp)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let kkp = builder.new_affine("kkp", 81 * 81 * 1548, 1);
        let out = builder.new_affine("out", 2, 1);
        let stm_eval = kkp.forward(stm_inputs);
        let ntm_eval = kkp.forward(ntm_inputs);
        let combined = stm_eval.concat(ntm_eval);
        out.forward(combined)
    });

    if let Some(dir) = resume_dir {
        eprintln!("  [KKP] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}

// ----- KPPT: KPP --------------------------------------------------------

fn run_kppt_kpp(args: &Args, resume_dir: Option<&std::path::Path>) {
    let qa: i16 = 256;
    let qb: i16 = 64;
    let qab: i16 = qa.checked_mul(qb).expect("qa*qb fits in i16");

    let save_format: Vec<SavedFormat> = vec![
        SavedFormat::id("kppw").round().quantise::<i16>(qa),
        SavedFormat::id("kppb").round().quantise::<i16>(qa),
        SavedFormat::id("outw").transpose().round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qab),
    ];

    let mut builder = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(ShogiKpp)
        .save_format(&save_format)
        .loss_fn(|out, tgt| out.sigmoid().squared_error(tgt));

    if args.score_drop_abs > 0 {
        builder = builder.score_drop_abs(args.score_drop_abs);
    }

    let mut trainer = builder.build(|builder, stm_inputs, ntm_inputs| {
        let kpp = builder.new_affine("kpp", 81 * 1548 * 1548, 1);
        let out = builder.new_affine("out", 2, 1);
        let stm_eval = kpp.forward(stm_inputs);
        let ntm_eval = kpp.forward(ntm_inputs);
        let combined = stm_eval.concat(ntm_eval);
        out.forward(combined)
    });

    if let Some(dir) = resume_dir {
        eprintln!("  [KPP] restoring optimiser state from {}", dir.display());
        trainer.load_from_checkpoint(dir.to_str().expect("resume dir UTF-8"));
    }

    run_training_inline!(args, &mut trainer);
}
