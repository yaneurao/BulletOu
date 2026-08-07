//! Held-out validation against a fixed-record teacher test set.
//!
//! Used by the `bulletou` example to compute "value-sign agreement"
//! accuracy after training: random-pick N positions from a test
//! `.hcpe` / `.psv` / `.bin`, run them through the trained model, then for each one
//! check whether the network's raw output and the actual **game
//! result** (win/loss for the side to move) have the same sign.
//!
//! Game result (= the outcome of the game the position came from) is
//! used rather than the teacher's centipawn score because the trainer's
//! loss target is itself derived from the score: comparing the model's
//! output sign against the score sign would mostly measure how well the
//! model fits the teacher rather than how well it predicts wins. Sign
//! agreement against the actual game result is harder and a more
//! honest measure of value-network quality.
//!
//! The accuracy definition matches dlshogi's `binary_accuracy`
//! (`dlshogi/train.py`) so the two trainers' value accuracy numbers
//! are directly comparable:
//!
//! ```text
//!     pred  = model_output >= 0       (= P(STM win) ≥ 0.5)
//!     truth = game_result  >= 0       (= STM did not lose: Win or Draw)
//!     match = pred == truth
//! ```
//!
//! Drawn games are bucketed with wins (`truth = true`) — the model is
//! "correct on a draw" iff it predicts ≥ 0. This is asymmetric but is
//! dlshogi's convention. Mate stamps (positions whose teacher score
//! abs ≥ `score_drop_abs`) are excluded from BOTH accuracy and loss,
//! consistent with the trainer's own `score_drop_abs` filter.
//!
//! This is a pure-Rust module that does **no GPU work** — it only
//! reads bytes, decodes them into `PackedSfenValue`, and computes the
//! accuracy from a model-output array supplied by the caller. The
//! GPU forward pass is the caller's responsibility (see
//! `ValueTrainer::eval_packed_batch`).

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::shogi::PackedSfenValue;
use crate::teacher_path::{DataFormat, expand_teacher, infer_data_format};
use crate::value::loader::{
    WinRateModelTargetParams,
    hcpe::{HCPE_RECORD_SIZE, decode_hcpe_record},
};

const PSV_RECORD_SIZE: usize = std::mem::size_of::<PackedSfenValue>();

/// Outcome of a sign-agreement validation pass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AccuracyReport {
    /// Number of **decisive** positions (= STM win or loss) actually
    /// used for the accuracy comparison: sampled minus
    /// `score_drop_abs`-filtered minus drawn games. Drawn games are
    /// excluded so the metric is symmetric W vs L (matches dlshogi
    /// author's actual validation set, which contains no draws).
    pub compared: usize,
    /// Of the `compared` decisive positions, how many had
    /// `(model_out >= 0) == (game_result > 0)` (= sign of the model's
    /// output matches the actual winner).
    pub sign_matches: usize,
    /// Number of decisive positions whose model output was non-negative
    /// (`model_output >= 0`). This is diagnostic only; it helps detect
    /// short validation runs where the network is still effectively a
    /// majority-class predictor.
    pub predicted_nonnegative: usize,
    /// Number of decisive positions whose model output was negative
    /// (`model_output < 0`). Diagnostic counterpart to
    /// [`predicted_nonnegative`](Self::predicted_nonnegative).
    pub predicted_negative: usize,
    /// Number of decisive positions whose model output was exactly zero.
    /// These are included in `predicted_nonnegative` because the
    /// BulletOu/YaneuraOu metric uses `model_output >= 0`.
    pub predicted_zero: usize,
    /// Number of sampled positions whose game ended in a draw
    /// (`game_result == 0`). Excluded from BOTH `compared` and
    /// `sign_matches` (= excluded from accuracy entirely). Still
    /// counted in `loss_sampled` so the loss average matches the
    /// trainer's training-loss subset.
    pub drawn_games: usize,
    /// Number of sampled positions whose teacher score's absolute
    /// value was at or above the `score_drop_abs` threshold (mate
    /// stamps); excluded from BOTH accuracy and loss.
    pub filtered_by_score_cap: usize,
    /// Number of positions used in the loss average (= sampled minus
    /// `score_drop_abs`-filtered, including drawn games — matches the
    /// trainer's training loss subset).
    pub loss_sampled: usize,
    /// Mean test-set loss over the `loss_sampled` subset. `None` when
    /// the caller didn't pass game results (= loss not requested) or
    /// when `loss_sampled == 0`.
    pub test_loss: Option<f32>,
}

impl AccuracyReport {
    /// `sign_matches / compared`, or `NaN` if no positions were compared.
    pub fn accuracy(&self) -> f32 {
        if self.compared == 0 { f32::NAN } else { self.sign_matches as f32 / self.compared as f32 }
    }
}

/// Fixed subset information for a validation set.
///
/// Draw/mate filtering depends only on the teacher records and
/// `score_drop_abs`, not on the network output. Build this once when a
/// held-out set is loaded, then reuse it for every validation pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationSampleMask {
    /// Indices used for value-sign accuracy: not score-capped and not drawn.
    pub accuracy_indices: Vec<usize>,
    /// Indices used for value loss: not score-capped. Draws are kept.
    pub loss_indices: Vec<usize>,
    /// Number of sampled draw positions excluded from accuracy.
    pub drawn_games: usize,
    /// Number of sampled positions excluded by `score_drop_abs`.
    pub filtered_by_score_cap: usize,
}

impl ValidationSampleMask {
    pub fn compared(&self) -> usize {
        self.accuracy_indices.len()
    }

    pub fn loss_sampled(&self) -> usize {
        self.loss_indices.len()
    }
}

/// Loss formula used for `test_value_loss`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationLossKind {
    /// Sigmoid probability-space value loss:
    /// `abs(sigmoid(model_output / model_output_scale) - target)^pow_exp`, where
    /// `pow_exp=2` is MSE,
    /// `model_output_scale=1` gives the historical logit-style behaviour and
    /// the score component is `sigmoid(teacher_score / eval_scale)`.
    SigmoidPow { pow_exp: f32 },
    /// Win-rate-model value loss with shogi WRM transforms.
    WinRateModel { pow_exp: f32, nnue2score: f32, in_offset: f32, in_scaling: f32, target: WinRateModelTargetParams },
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn wrm_probability(score: f32, offset: f32, scaling: f32) -> f32 {
    let q = (score - offset) / scaling;
    let qm = (-score - offset) / scaling;
    0.5 * (1.0 + sigmoid(q) - sigmoid(qm))
}

/// Compute sign-agreement accuracy AND the matching test-set loss
/// from parallel arrays.
///
/// `model_outputs[i]` is the raw network output for position `i`. For
/// [`ValidationLossKind::SigmoidPow`], `model_output_scale` controls whether
/// that output is interpreted as a logit (`1.0`) or a centipawn-like value
/// (`eval_scale`-style).
/// `teacher_results[i]` is the actual game outcome from the position's
/// STM perspective: `+1` (STM won), `0` (draw), `-1` (STM lost).
///
/// **Accuracy** is sign agreement on **decisive games only** (draws
/// are excluded entirely):
///
/// ```text
///   pred  = model_output >= 0
///   truth = game_result  >  0       (Win → true; Loss → false; Draw → skipped)
///   match = pred == truth
/// ```
///
/// Drawn games (game_result == 0) are NOT counted in either the
/// numerator or the denominator. dlshogi's published `binary_accuracy`
/// bucketed draws into the win side, but the dlshogi author's actual
/// validation set contains no draws — so excluding them gives a
/// metric that more closely matches what the dlshogi authors actually
/// optimise against. It also removes a structural bias toward "0
/// output" predictions (a draw + `model_output == 0` would have been
/// counted as correct under the old formula).
///
/// **Loss subset**: same as the trainer — `score_drop_abs`-filtered
/// positions are skipped, drawn games are kept. The formula matches
/// `bullet_lib::value::loader::DefaultDataLoader::prepare`:
///
/// ```text
///   blend  = 1 - lambda
///   result_norm = result == +1 ? 1.0 : result == -1 ? 0.0 : 0.5
///   target = blend * result_norm + (1 - blend) * sigmoid(score / scale)
///   loss   = (sigmoid(model_out / model_output_scale) - target)^2
/// ```
///
/// `score_drop_abs == Some(cap)` filters out positions with
/// `|teacher_score| >= cap` (= mate stamps such as ±32000); these are
/// excluded from BOTH accuracy and loss. `score_drop_abs == None`
/// keeps all positions.
///
/// When `teacher_results.is_empty()`, accuracy falls back to comparing
/// model output sign vs `teacher_scores[i]` sign (the legacy "score
/// agreement" metric, kept only so the unit tests can exercise the
/// loop without mocking game results). Production callers always
/// supply `teacher_results`.
pub fn compute_sign_accuracy(
    model_outputs: &[f32],
    teacher_scores: &[i16],
    teacher_results: &[i8],
    score_drop_abs: Option<u16>,
    lambda: f32,
    eval_scale: f32,
) -> AccuracyReport {
    compute_sign_accuracy_with_loss(
        model_outputs,
        teacher_scores,
        teacher_results,
        score_drop_abs,
        lambda,
        eval_scale,
        1.0,
        ValidationLossKind::SigmoidPow { pow_exp: 2.0 },
    )
}

pub fn compute_sign_accuracy_with_loss(
    model_outputs: &[f32],
    teacher_scores: &[i16],
    teacher_results: &[i8],
    score_drop_abs: Option<u16>,
    lambda: f32,
    eval_scale: f32,
    model_output_scale: f32,
    loss_kind: ValidationLossKind,
) -> AccuracyReport {
    assert_eq!(model_outputs.len(), teacher_scores.len(), "model_outputs and teacher_scores length mismatch");
    let have_results = !teacher_results.is_empty();
    if have_results {
        assert_eq!(model_outputs.len(), teacher_results.len(), "model_outputs and teacher_results length mismatch");
    }
    if teacher_scores.len() == model_outputs.len() {
        let mask = build_validation_sample_mask(teacher_scores, teacher_results, score_drop_abs);
        return compute_sign_accuracy_with_loss_masked(
            model_outputs,
            teacher_scores,
            teacher_results,
            &mask,
            lambda,
            eval_scale,
            model_output_scale,
            loss_kind,
        );
    }
    let mut report = AccuracyReport::default();
    let blend = 1.0 - lambda;
    let inv_scale = if eval_scale > 0.0 { 1.0 / eval_scale } else { 0.0025 };
    let model_inv_scale = if model_output_scale > 0.0 { 1.0 / model_output_scale } else { 1.0 };
    let mut loss_sum = 0.0f32;
    for (i, (m, &s)) in model_outputs.iter().zip(teacher_scores.iter()).enumerate() {
        if let Some(cap) = score_drop_abs {
            if s.unsigned_abs() >= cap {
                report.filtered_by_score_cap += 1;
                continue;
            }
        }
        // Accuracy: sign agreement on decisive games only. Draws
        // (game_result == 0, or score == 0 in the legacy fallback)
        // are excluded from BOTH the numerator and the denominator —
        // they would otherwise reward "0 output" predictions
        // structurally and diverge from the dlshogi author's actual
        // validation set, which contains no draws.
        let pred = *m >= 0.0;
        if have_results {
            let r = teacher_results[i];
            if r == 0 {
                report.drawn_games += 1;
                // Not counted in `compared` / `sign_matches`; loss
                // block below still keeps the draw (the trainer's
                // training-loss subset includes it).
            } else {
                report.compared += 1;
                if pred {
                    report.predicted_nonnegative += 1;
                } else {
                    report.predicted_negative += 1;
                }
                if *m == 0.0 {
                    report.predicted_zero += 1;
                }
                let truth = r > 0;
                if pred == truth {
                    report.sign_matches += 1;
                }
            }
        } else if s == 0 {
            report.drawn_games += 1;
        } else {
            report.compared += 1;
            if pred {
                report.predicted_nonnegative += 1;
            } else {
                report.predicted_negative += 1;
            }
            if *m == 0.0 {
                report.predicted_zero += 1;
            }
            if pred == (s > 0) {
                report.sign_matches += 1;
            }
        }
        // Loss: include drawn games (matches the trainer's loss
        // averaging), exclude only mate stamps (already `continue`d
        // above).
        if have_results {
            let result_norm = match teacher_results[i].signum() {
                1 => 1.0,
                -1 => 0.0,
                _ => 0.5,
            };
            let score_norm = match loss_kind {
                ValidationLossKind::SigmoidPow { .. } => sigmoid(inv_scale * f32::from(s)),
                ValidationLossKind::WinRateModel { target, .. } => target.probability(f32::from(s)),
            };
            let target = blend * result_norm + (1.0 - blend) * score_norm;
            let model_p = match loss_kind {
                ValidationLossKind::SigmoidPow { .. } => sigmoid(*m * model_inv_scale),
                ValidationLossKind::WinRateModel { nnue2score, in_offset, in_scaling, .. } => {
                    wrm_probability(*m * nnue2score, in_offset, in_scaling)
                }
            };
            let diff = model_p - target;
            loss_sum += match loss_kind {
                ValidationLossKind::SigmoidPow { pow_exp } => diff.abs().powf(pow_exp),
                ValidationLossKind::WinRateModel { pow_exp, .. } => diff.abs().powf(pow_exp),
            };
            report.loss_sampled += 1;
        }
    }
    if have_results && report.loss_sampled > 0 {
        report.test_loss = Some(loss_sum / report.loss_sampled as f32);
    }
    report
}

pub fn build_validation_sample_mask(
    teacher_scores: &[i16],
    teacher_results: &[i8],
    score_drop_abs: Option<u16>,
) -> ValidationSampleMask {
    let have_results = !teacher_results.is_empty();
    if have_results {
        assert_eq!(teacher_scores.len(), teacher_results.len(), "teacher_scores and teacher_results length mismatch");
    }
    let mut mask = ValidationSampleMask::default();
    for (i, &s) in teacher_scores.iter().enumerate() {
        if let Some(cap) = score_drop_abs {
            if s.unsigned_abs() >= cap {
                mask.filtered_by_score_cap += 1;
                continue;
            }
        }
        mask.loss_indices.push(i);
        let is_draw = if have_results { teacher_results[i] == 0 } else { s == 0 };
        if is_draw {
            mask.drawn_games += 1;
        } else {
            mask.accuracy_indices.push(i);
        }
    }
    mask
}

pub fn compute_sign_accuracy_with_loss_masked(
    model_outputs: &[f32],
    teacher_scores: &[i16],
    teacher_results: &[i8],
    mask: &ValidationSampleMask,
    lambda: f32,
    eval_scale: f32,
    model_output_scale: f32,
    loss_kind: ValidationLossKind,
) -> AccuracyReport {
    assert_eq!(model_outputs.len(), teacher_scores.len(), "model_outputs and teacher_scores length mismatch");
    let have_results = !teacher_results.is_empty();
    if have_results {
        assert_eq!(model_outputs.len(), teacher_results.len(), "model_outputs and teacher_results length mismatch");
    }

    let mut report = AccuracyReport {
        compared: mask.compared(),
        drawn_games: mask.drawn_games,
        filtered_by_score_cap: mask.filtered_by_score_cap,
        loss_sampled: if have_results { mask.loss_sampled() } else { 0 },
        ..AccuracyReport::default()
    };

    for &i in &mask.accuracy_indices {
        let m = model_outputs[i];
        let pred = m >= 0.0;
        if pred {
            report.predicted_nonnegative += 1;
        } else {
            report.predicted_negative += 1;
        }
        if m == 0.0 {
            report.predicted_zero += 1;
        }
        let truth = if have_results { teacher_results[i] > 0 } else { teacher_scores[i] > 0 };
        if pred == truth {
            report.sign_matches += 1;
        }
    }

    if have_results {
        let blend = 1.0 - lambda;
        let inv_scale = if eval_scale > 0.0 { 1.0 / eval_scale } else { 0.0025 };
        let model_inv_scale = if model_output_scale > 0.0 { 1.0 / model_output_scale } else { 1.0 };
        let mut loss_sum = 0.0f32;
        for &i in &mask.loss_indices {
            let s = teacher_scores[i];
            let m = model_outputs[i];
            let result_norm = match teacher_results[i].signum() {
                1 => 1.0,
                -1 => 0.0,
                _ => 0.5,
            };
            let score_norm = match loss_kind {
                ValidationLossKind::SigmoidPow { .. } => sigmoid(inv_scale * f32::from(s)),
                ValidationLossKind::WinRateModel { target, .. } => target.probability(f32::from(s)),
            };
            let target = blend * result_norm + (1.0 - blend) * score_norm;
            let model_p = match loss_kind {
                ValidationLossKind::SigmoidPow { .. } => sigmoid(m * model_inv_scale),
                ValidationLossKind::WinRateModel { nnue2score, in_offset, in_scaling, .. } => {
                    wrm_probability(m * nnue2score, in_offset, in_scaling)
                }
            };
            let diff = model_p - target;
            loss_sum += match loss_kind {
                ValidationLossKind::SigmoidPow { pow_exp } => diff.abs().powf(pow_exp),
                ValidationLossKind::WinRateModel { pow_exp, .. } => diff.abs().powf(pow_exp),
            };
        }
        if report.loss_sampled > 0 {
            report.test_loss = Some(loss_sum / report.loss_sampled as f32);
        }
    }

    report
}

/// Reservoir-sample `n` positions uniformly from a fixed-record teacher
/// spec, decoding each into `PackedSfenValue`.
///
/// `seed = 0` selects a time-based seed (= non-reproducible);
/// any other value uses that seed verbatim (= reproducible).
///
/// The teacher may be any path accepted by [`expand_teacher`], but the
/// format must be fixed-record `.hcpe`, `.psv`, or PSV-compatible `.bin`. Variable-length
/// `.hcpe3` / `.pack` teachers are intentionally rejected here; export
/// them to `.psv` first when a held-out validation sample is needed.
///
/// On disk-too-small (= file has fewer than `n` valid records), all
/// available records are returned and the caller can decide whether to
/// proceed.
pub fn read_random_teacher_positions(teacher: &str, n: usize, seed: u64) -> std::io::Result<Vec<PackedSfenValue>> {
    let paths = expand_teacher(teacher).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let path_refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
    let format =
        infer_data_format(&path_refs).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    match format {
        DataFormat::Hcpe | DataFormat::Psv => read_random_fixed_teacher_positions(&paths, format, n, seed),
        DataFormat::Hcpe3 | DataFormat::Pack => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "--test-teacher format {format:?} is variable-length; export it to .psv with export_teacher_psv first"
            ),
        )),
    }
}

/// Read the first `n` positions from a fixed-record teacher spec,
/// decoding each into `PackedSfenValue`.
///
/// This is intended for cross-tool parity runs where the validation
/// subset must be byte-for-byte identical to another trainer that consumes
/// the held-out file sequentially.
pub fn read_teacher_positions_prefix(teacher: &str, n: usize) -> std::io::Result<Vec<PackedSfenValue>> {
    let paths = expand_teacher(teacher).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let path_refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
    let format =
        infer_data_format(&path_refs).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    match format {
        DataFormat::Hcpe | DataFormat::Psv => read_fixed_teacher_positions_prefix(&paths, format, n),
        DataFormat::Hcpe3 | DataFormat::Pack => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "--test-teacher format {format:?} is variable-length; export it to .psv with export_teacher_psv first"
            ),
        )),
    }
}

/// Read all positions from a fixed-record teacher spec.
pub fn read_all_teacher_positions(teacher: &str) -> std::io::Result<Vec<PackedSfenValue>> {
    let paths = expand_teacher(teacher).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let path_refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
    let format =
        infer_data_format(&path_refs).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    match format {
        DataFormat::Hcpe | DataFormat::Psv => read_all_fixed_teacher_positions(&paths, format),
        DataFormat::Hcpe3 | DataFormat::Pack => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "--test-teacher format {format:?} is variable-length; export it to .psv with export_teacher_psv first"
            ),
        )),
    }
}

/// Backwards-compatible HCPE-only entry point.
pub fn read_random_hcpe_positions(path: &str, n: usize, seed: u64) -> std::io::Result<Vec<PackedSfenValue>> {
    read_random_fixed_teacher_positions(&[path.to_string()], DataFormat::Hcpe, n, seed)
}

#[derive(Debug, Clone)]
struct FixedTeacherFile {
    path: String,
    start_record: usize,
    records: usize,
}

fn read_random_fixed_teacher_positions(
    paths: &[String],
    format: DataFormat,
    n: usize,
    seed: u64,
) -> std::io::Result<Vec<PackedSfenValue>> {
    let (files, total_records, record_size) = fixed_teacher_files(paths, format)?;
    if total_records == 0 {
        return Ok(Vec::new());
    }

    let indices = sample_record_indices(total_records, n, seed);
    read_fixed_teacher_indices(&files, format, record_size, &indices)
}

fn read_fixed_teacher_positions_prefix(
    paths: &[String],
    format: DataFormat,
    n: usize,
) -> std::io::Result<Vec<PackedSfenValue>> {
    let (files, total_records, record_size) = fixed_teacher_files(paths, format)?;
    if total_records == 0 {
        return Ok(Vec::new());
    }

    let take = total_records.min(n);
    let indices = (0..take).collect::<Vec<_>>();
    read_fixed_teacher_indices(&files, format, record_size, &indices)
}

fn read_all_fixed_teacher_positions(paths: &[String], format: DataFormat) -> std::io::Result<Vec<PackedSfenValue>> {
    let (files, total_records, _record_size) = fixed_teacher_files(paths, format)?;
    if total_records == 0 {
        return Ok(Vec::new());
    }

    let mut out: Vec<PackedSfenValue> = Vec::with_capacity(total_records);
    for info in &files {
        let mut file = BufReader::new(File::open(&info.path)?);
        match format {
            DataFormat::Hcpe => {
                for _ in 0..info.records {
                    let mut rec = [0u8; HCPE_RECORD_SIZE];
                    file.read_exact(&mut rec)?;
                    if let Some(psv) = decode_hcpe_record(&rec) {
                        out.push(psv);
                    }
                    // silently skip records whose HCP failed to decode (= corrupted)
                }
            }
            DataFormat::Psv => {
                for _ in 0..info.records {
                    let mut rec = [0u8; PSV_RECORD_SIZE];
                    file.read_exact(&mut rec)?;
                    let mut psv = PackedSfenValue::default();
                    psv.as_bytes_mut().copy_from_slice(&rec);
                    out.push(psv);
                }
            }
            DataFormat::Hcpe3 | DataFormat::Pack => unreachable!("caller validated fixed-record format"),
        }
    }
    Ok(out)
}

fn fixed_teacher_files(paths: &[String], format: DataFormat) -> std::io::Result<(Vec<FixedTeacherFile>, usize, usize)> {
    let record_size = fixed_record_size(format).expect("caller validated fixed-record format");
    let mut files = Vec::with_capacity(paths.len());
    let mut total_records = 0usize;
    for path in paths {
        let file_size = usize::try_from(std::fs::metadata(path)?.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{path}: file is too large to index on this platform"),
            )
        })?;
        if file_size % record_size != 0 {
            let name = fixed_record_size_name(format);
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{path}: size {file_size} is not a multiple of {name} ({record_size}) — \
                     not a valid {format:?} file?"
                ),
            ));
        }
        let records = file_size / record_size;
        files.push(FixedTeacherFile { path: path.clone(), start_record: total_records, records });
        total_records = total_records
            .checked_add(records)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "teacher record count overflow"))?;
    }
    Ok((files, total_records, record_size))
}

fn fixed_record_size(format: DataFormat) -> Option<usize> {
    match format {
        DataFormat::Hcpe => Some(HCPE_RECORD_SIZE),
        DataFormat::Psv => Some(PSV_RECORD_SIZE),
        DataFormat::Hcpe3 | DataFormat::Pack => None,
    }
}

fn fixed_record_size_name(format: DataFormat) -> &'static str {
    match format {
        DataFormat::Hcpe => "HCPE_RECORD_SIZE",
        DataFormat::Psv => "PSV_RECORD_SIZE",
        DataFormat::Hcpe3 | DataFormat::Pack => "record size",
    }
}

fn sample_record_indices(total_records: usize, n: usize, seed: u64) -> Vec<usize> {
    let mut rng = SeededXorShift::from_seed(seed);
    if total_records <= n {
        (0..total_records).collect()
    } else {
        // Floyd's algorithm for sampling without replacement, O(n).
        let mut chosen = std::collections::BTreeSet::new();
        for j in (total_records - n)..total_records {
            let pick = (rng.next_u64() as usize) % (j + 1);
            if !chosen.insert(pick) {
                chosen.insert(j);
            }
        }
        chosen.into_iter().collect()
    }
}

fn read_fixed_teacher_indices(
    files: &[FixedTeacherFile],
    format: DataFormat,
    record_size: usize,
    indices: &[usize],
) -> std::io::Result<Vec<PackedSfenValue>> {
    let mut out: Vec<PackedSfenValue> = Vec::with_capacity(indices.len());
    let mut current_file_index = usize::MAX;
    let mut current_file: Option<File> = None;
    let mut file_index = 0usize;

    for &global_idx in indices {
        while file_index + 1 < files.len()
            && global_idx >= files[file_index].start_record.saturating_add(files[file_index].records)
        {
            file_index += 1;
        }
        let info = &files[file_index];
        if current_file_index != file_index {
            current_file = Some(File::open(&info.path)?);
            current_file_index = file_index;
        }
        let local_idx = global_idx.checked_sub(info.start_record).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("internal validation sampler index underflow for {}", info.path),
            )
        })?;
        let offset = (local_idx as u64)
            .checked_mul(record_size as u64)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "record offset overflow"))?;
        let file = current_file.as_mut().expect("file opened above");
        file.seek(SeekFrom::Start(offset))?;

        match format {
            DataFormat::Hcpe => {
                let mut rec = [0u8; HCPE_RECORD_SIZE];
                file.read_exact(&mut rec)?;
                if let Some(psv) = decode_hcpe_record(&rec) {
                    out.push(psv);
                }
                // silently skip records whose HCP failed to decode (= corrupted)
            }
            DataFormat::Psv => {
                let mut rec = [0u8; PSV_RECORD_SIZE];
                file.read_exact(&mut rec)?;
                let mut psv = PackedSfenValue::default();
                psv.as_bytes_mut().copy_from_slice(&rec);
                out.push(psv);
            }
            DataFormat::Hcpe3 | DataFormat::Pack => unreachable!("caller validated fixed-record format"),
        }
    }
    Ok(out)
}

/// Seeded xorshift RNG. Same algorithm as the loader's `SimpleRand`
/// but takes a user-supplied seed for reproducibility.
struct SeededXorShift(u64);

impl SeededXorShift {
    fn from_seed(seed: u64) -> Self {
        if seed == 0 {
            // Treat 0 as "use a time-based seed" so the user's default
            // CLI behaviour gets a different sample on each run unless
            // they explicitly want determinism.
            let s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEADBEEFCAFEBABEu64)
                | 1; // xorshift requires non-zero state
            Self(s)
        } else {
            Self(seed)
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_all_match() {
        let m = [0.5, -0.3, 1.2, -0.01];
        let t = [200i16, -150, 800, -10];
        let r = compute_sign_accuracy(&m, &t, &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 4);
        assert_eq!(r.sign_matches, 4);
        assert!((r.accuracy() - 1.0).abs() < 1e-6);
        assert_eq!(r.test_loss, None, "no results → no loss");
    }

    #[test]
    fn accuracy_half_mismatch() {
        let m = [0.5, -0.3, 1.2, -0.01];
        // first 2 match, last 2 flipped
        let t = [200i16, -150, -800, 10];
        let r = compute_sign_accuracy(&m, &t, &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 4);
        assert_eq!(r.sign_matches, 2);
        assert!((r.accuracy() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn accuracy_excludes_draws() {
        // Draws (game_result == 0) are excluded from BOTH the numerator
        // and denominator of accuracy; they're still counted in
        // `loss_sampled` so the loss average matches the trainer's
        // training-loss subset.
        let m = [0.5, -0.1, 0.3, -0.2];
        let t = [200i16, 100, 0, -50]; // teacher scores (unused for
        // accuracy when results given)
        let game = [1i8, 0, 0, -1]; // Win, Draw, Draw, Loss
        let r = compute_sign_accuracy(&m, &t, &game, None, 1.0, 400.0);
        // i=0: Win, pred=true, truth=true → match
        // i=1: Draw → skipped from accuracy
        // i=2: Draw → skipped from accuracy
        // i=3: Loss, pred=false, truth=false → match
        assert_eq!(r.compared, 2, "only the 2 decisive positions count");
        assert_eq!(r.sign_matches, 2, "both decisive positions matched");
        assert_eq!(r.drawn_games, 2);
        assert_eq!(r.loss_sampled, 4, "loss subset still includes draws");
        assert!(r.test_loss.is_some());
    }

    #[test]
    fn accuracy_reports_decisive_prediction_sign_distribution() {
        let m = [0.0, 0.2, -0.1, -0.3, 0.4];
        let t = [1i16, 2, 3, 4, 5];
        let game = [1i8, -1, -1, 0, 1];
        let r = compute_sign_accuracy(&m, &t, &game, None, 1.0, 400.0);
        assert_eq!(r.compared, 4, "draw is excluded from decisive sign diagnostics");
        assert_eq!(r.predicted_nonnegative, 3, "zero is counted on the >=0 side");
        assert_eq!(r.predicted_negative, 1);
        assert_eq!(r.predicted_zero, 1);
    }

    #[test]
    fn accuracy_falls_back_to_score_sign_when_no_results() {
        // Legacy path (no teacher_results): the function falls back to
        // comparing model sign vs teacher score sign and skipping
        // score==0 positions. Used only in unit tests.
        let m = [0.5, 0.3, 1.2];
        let t = [200i16, 0, 0];
        let r = compute_sign_accuracy(&m, &t, &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 1, "two zero-score positions skipped");
        assert_eq!(r.sign_matches, 1);
        assert_eq!(r.drawn_games, 2);
        assert_eq!(r.loss_sampled, 0, "no results → no loss");
    }

    #[test]
    fn accuracy_score_drop_filters_mate_stamps() {
        let m = [0.5, 0.3, 1.2, -0.5];
        let t = [200i16, 32000, -32000, -10];
        let r = compute_sign_accuracy(&m, &t, &[], Some(32000), 1.0, 400.0);
        assert_eq!(r.compared, 2);
        assert_eq!(r.sign_matches, 2);
        assert_eq!(r.filtered_by_score_cap, 2);
        assert_eq!(r.drawn_games, 0);
    }

    #[test]
    fn accuracy_zero_when_empty() {
        let r = compute_sign_accuracy(&[], &[], &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 0);
        assert!(r.accuracy().is_nan());
        assert_eq!(r.test_loss, None);
    }

    #[test]
    fn test_loss_pure_eval_target() {
        // lambda=1.0, blend=0 → target = sigmoid(score/scale) only
        let m = [0.0, 0.0]; // sigmoid(0) = 0.5
        let t = [400i16, -400]; // sigmoid(±1) ≈ 0.731 / 0.269
        let r = compute_sign_accuracy(&m, &t, &[1, -1], None, 1.0, 400.0);
        assert_eq!(r.compared, 2);
        let loss = r.test_loss.expect("loss requested");
        // expected: ((0.5 - 0.731)^2 + (0.5 - 0.269)^2) / 2 ≈ 0.0533
        assert!((loss - 0.0533).abs() < 1e-3, "loss={loss}");
    }

    #[test]
    fn test_loss_pure_wdl_target() {
        // lambda=0.0, blend=1 → target = result/2 mapping (Win=1, Loss=0, Draw=0.5)
        let m = [0.0, 0.0, 0.0]; // sigmoid(0) = 0.5
        // results: +1 (win), -1 (loss), 0 (draw)
        let r = compute_sign_accuracy(&m, &[100i16, -100, 100], &[1, -1, 0], None, 0.0, 400.0);
        // draw position has teacher_score=100 (not 0) so accuracy still includes it,
        // but here we just check loss.
        // Targets: 1.0, 0.0, 0.5
        // Losses:  0.25, 0.25, 0.0
        // Mean: 0.1667
        let loss = r.test_loss.expect("loss requested");
        assert!((loss - 0.1666666).abs() < 1e-3, "loss={loss}");
    }

    #[test]
    fn seeded_rng_is_reproducible() {
        let mut a = SeededXorShift::from_seed(12345);
        let mut b = SeededXorShift::from_seed(12345);
        for _ in 0..20 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn read_random_hcpe_rejects_bad_size() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-validate-test-bad-size-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::write(&tmp, vec![0u8; 37]).unwrap(); // not a multiple of 38
        let path = tmp.to_str().unwrap();
        // Result::unwrap_err() needs T: Debug; PackedSfenValue is not, so
        // pattern-match instead.
        match read_random_hcpe_positions(path, 5, 1) {
            Ok(_) => panic!("expected InvalidData error, got Ok"),
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn read_random_teacher_positions_supports_psv() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-validate-test-fixed-psv-{}-{}.psv",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        fn psv_with(score: i16, result: i8) -> PackedSfenValue {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut()[32..34].copy_from_slice(&score.to_le_bytes());
            psv.as_bytes_mut()[38] = result as u8;
            psv
        }

        let records = [psv_with(123, 1), psv_with(-45, -1), psv_with(0, 0)];
        let mut bytes = Vec::new();
        for psv in &records {
            bytes.extend_from_slice(psv.as_bytes());
        }
        std::fs::write(&tmp, bytes).unwrap();

        let got = read_random_teacher_positions(tmp.to_str().unwrap(), 10, 1).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got.iter().map(PackedSfenValue::score).collect::<Vec<_>>(), vec![123, -45, 0]);
        assert_eq!(got.iter().map(PackedSfenValue::game_result).collect::<Vec<_>>(), vec![1, -1, 0]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn read_random_teacher_positions_treats_bin_as_psv() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-validate-test-fixed-bin-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        fn psv_with(score: i16, result: i8) -> PackedSfenValue {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut()[32..34].copy_from_slice(&score.to_le_bytes());
            psv.as_bytes_mut()[38] = result as u8;
            psv
        }

        let records = [psv_with(321, 1), psv_with(-54, -1)];
        let mut bytes = Vec::new();
        for psv in &records {
            bytes.extend_from_slice(psv.as_bytes());
        }
        std::fs::write(&tmp, bytes).unwrap();

        let got = read_random_teacher_positions(tmp.to_str().unwrap(), 10, 1).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got.iter().map(PackedSfenValue::score).collect::<Vec<_>>(), vec![321, -54]);
        assert_eq!(got.iter().map(PackedSfenValue::game_result).collect::<Vec<_>>(), vec![1, -1]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn read_teacher_positions_prefix_reads_first_records() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-validate-test-prefix-psv-{}-{}.psv",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        fn psv_with(score: i16, result: i8) -> PackedSfenValue {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut()[32..34].copy_from_slice(&score.to_le_bytes());
            psv.as_bytes_mut()[38] = result as u8;
            psv
        }

        let records = [psv_with(11, 1), psv_with(22, -1), psv_with(33, 1), psv_with(44, -1)];
        let mut bytes = Vec::new();
        for psv in &records {
            bytes.extend_from_slice(psv.as_bytes());
        }
        std::fs::write(&tmp, bytes).unwrap();

        let got = read_teacher_positions_prefix(tmp.to_str().unwrap(), 3).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got.iter().map(PackedSfenValue::score).collect::<Vec<_>>(), vec![11, 22, 33]);
        assert_eq!(got.iter().map(PackedSfenValue::game_result).collect::<Vec<_>>(), vec![1, -1, 1]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn read_all_teacher_positions_reads_every_fixed_record() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-validate-test-all-psv-{}-{}.psv",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        fn psv_with(score: i16, result: i8) -> PackedSfenValue {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut()[32..34].copy_from_slice(&score.to_le_bytes());
            psv.as_bytes_mut()[38] = result as u8;
            psv
        }

        let records = [psv_with(101, 1), psv_with(202, -1), psv_with(303, 1), psv_with(404, -1)];
        let mut bytes = Vec::new();
        for psv in &records {
            bytes.extend_from_slice(psv.as_bytes());
        }
        std::fs::write(&tmp, bytes).unwrap();

        let got = read_all_teacher_positions(tmp.to_str().unwrap()).unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got.iter().map(PackedSfenValue::score).collect::<Vec<_>>(), vec![101, 202, 303, 404]);
        assert_eq!(got.iter().map(PackedSfenValue::game_result).collect::<Vec<_>>(), vec![1, -1, 1, -1]);

        let _ = std::fs::remove_file(&tmp);
    }
}
