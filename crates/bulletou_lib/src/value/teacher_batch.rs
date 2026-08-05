//! Teacher-to-`FastBatchHost` helpers shared by fixture exporters and future
//! fast backend trainers.

use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    sync::{atomic::Ordering, mpsc},
};

use rayon::prelude::*;

use crate::{
    game::{
        inputs::{
            Factorised, HALFKP_MAX_ACTIVE_FEATURES, KP_MAX_ACTIVE, ShogiHalfKP, ShogiHalfKPPieceFactorizer,
            ShogiHalfKa2, SparseInputType, fill_halfka2_feature_indices_from_board, fill_halfkp_feature_indices,
            fill_ka2_feature_indices_from_board, fill_kp_feature_indices,
        },
        outputs::{ShogiSfnnLayerStackBucket, ShogiSfnnLayerStackBucketKind},
    },
    shogi::{PackedSfenValue, ShogiBoard},
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    value::{
        FastBatchHost, FastBatchLayout, NoOutputBuckets,
        loader::{
            DataLoader, DefaultDataLoader, DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader,
            ShogiPackLoader, WinRateModelTargetParams, load_and_map_shuffled_batches_with_prefetch,
            teacher_shuffle_window_records, win_rate_model_score_from_table, win_rate_model_score_table,
        },
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeacherDataloaderPos {
    pub byte_offset: u64,
    pub plies: usize,
}

#[derive(Debug, Clone)]
pub struct HalfkpTeacherBatchConfig<'a> {
    pub teacher: &'a str,
    pub batch_size: usize,
    pub batch_index: usize,
    /// Exact concatenated dataloader position to resume from. When set,
    /// supported loaders start from this position instead of deriving a skip
    /// from `batch_index`; `batch_index` is still used for source labels.
    pub dataloader_resume_pos: Option<TeacherDataloaderPos>,
    pub buffer_mb: usize,
    /// HCPE decode threads. `0` means loader default/auto.
    pub loader_threads: usize,
    /// CPU worker threads used while materialising the prepared batch.
    pub threads: usize,
    /// Prepared-batch queue depth used to overlap CPU materialisation with GPU consumption.
    pub queue_depth: usize,
    /// Lambda on teacher eval score when target values are prepared.
    pub lambda: f32,
    /// Eval-to-score sigmoid scale used while preparing teacher targets.
    pub scale: f32,
    /// Use win-rate-model target conversion while preparing teacher targets.
    pub win_rate_model: bool,
    /// WRM target-side score-to-probability parameters.
    pub wrm_target: WinRateModelTargetParams,
    /// Add tatara-style HalfKP piece-input virtual rows to the FT input.
    pub ft_factorize: bool,
    /// Drop positions whose absolute teacher score is at least this value.
    pub score_drop_abs: Option<u16>,
    /// Shuffle window size in mini-batches. `0` disables in-trainer shuffling.
    pub teacher_shuffle_buffer_batches: usize,
    /// Base seed for deterministic teacher shuffle windows.
    pub teacher_shuffle_seed: u64,
    /// Print CPU batch materialisation timing for profiling runs.
    pub profile_prepare: bool,
}

#[derive(Debug, Clone)]
pub struct HalfkpTeacherBatch {
    pub batch: FastBatchHost,
    pub source: String,
    pub dataloader_pos: Option<TeacherDataloaderPos>,
}

#[derive(Debug, Clone)]
pub struct KpTeacherBatchConfig<'a> {
    pub teacher: &'a str,
    pub batch_size: usize,
    pub batch_index: usize,
    pub dataloader_resume_pos: Option<TeacherDataloaderPos>,
    pub buffer_mb: usize,
    pub loader_threads: usize,
    pub threads: usize,
    pub queue_depth: usize,
    pub lambda: f32,
    pub scale: f32,
    pub win_rate_model: bool,
    pub wrm_target: WinRateModelTargetParams,
    pub score_drop_abs: Option<u16>,
    pub teacher_shuffle_buffer_batches: usize,
    pub teacher_shuffle_seed: u64,
    /// Print CPU batch materialisation timing for profiling runs.
    pub profile_prepare: bool,
}

#[derive(Debug, Clone)]
pub struct KpTeacherBatch {
    pub batch: FastBatchHost,
    pub source: String,
    pub dataloader_pos: Option<TeacherDataloaderPos>,
}

#[derive(Debug, Clone)]
pub struct KpptTeacherBatchConfig<'a> {
    pub teacher: &'a str,
    pub batch_size: usize,
    pub batch_index: usize,
    pub dataloader_resume_pos: Option<TeacherDataloaderPos>,
    pub buffer_mb: usize,
    pub loader_threads: usize,
    pub threads: usize,
    pub queue_depth: usize,
    pub lambda: f32,
    pub scale: f32,
    pub win_rate_model: bool,
    pub wrm_target: WinRateModelTargetParams,
    pub score_drop_abs: Option<u16>,
    pub teacher_shuffle_buffer_batches: usize,
    pub teacher_shuffle_seed: u64,
    /// Print CPU batch materialisation timing for profiling runs.
    pub profile_prepare: bool,
}

#[derive(Debug, Clone)]
pub struct KpptTeacherBatch {
    pub batch: FastBatchHost,
    pub source: String,
    pub dataloader_pos: Option<TeacherDataloaderPos>,
}

#[derive(Debug, Clone)]
pub struct SfnnTeacherBatchConfig<'a> {
    pub teacher: &'a str,
    pub batch_size: usize,
    pub batch_index: usize,
    pub dataloader_resume_pos: Option<TeacherDataloaderPos>,
    pub layerstack_bucket: ShogiSfnnLayerStackBucketKind,
    pub buffer_mb: usize,
    pub loader_threads: usize,
    pub threads: usize,
    pub queue_depth: usize,
    pub lambda: f32,
    pub scale: f32,
    pub win_rate_model: bool,
    pub wrm_target: WinRateModelTargetParams,
    pub score_drop_abs: Option<u16>,
    pub teacher_shuffle_buffer_batches: usize,
    pub teacher_shuffle_seed: u64,
    /// Print CPU batch materialisation timing for profiling runs.
    pub profile_prepare: bool,
}

#[derive(Debug, Clone)]
pub struct SfnnTeacherBatch {
    pub batch: FastBatchHost,
    pub source: String,
    pub dataloader_pos: Option<TeacherDataloaderPos>,
    pub timing: TeacherBatchTiming,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TeacherBatchTiming {
    /// Time spent in the producer before the next complete batch is yielded by
    /// the data loader. This includes file I/O, decoding, incomplete-batch
    /// assembly, and optional teacher-shuffle window preparation.
    pub producer_load_sec: f64,
    /// Time spent materialising `FastBatchHost` from decoded teacher positions.
    pub producer_prepare_sec: f64,
    /// Time the consumer spent waiting for a prepared batch from the producer
    /// queue. A large value means the GPU-side training loop was starved by the
    /// teacher producer.
    pub consumer_queue_wait_sec: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeacherBatchError {
    message: String,
}

impl TeacherBatchError {
    fn invalid_input(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for TeacherBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for TeacherBatchError {}

#[derive(Debug, Clone, Copy)]
pub struct WrmTargetCalibrationConfig<'a> {
    pub teacher: &'a str,
    pub positions: usize,
    pub buffer_mb: usize,
    pub loader_threads: usize,
    pub score_drop_abs: Option<u16>,
}

#[derive(Debug, Clone, Copy)]
pub struct WrmTargetCalibrationReport {
    pub params: WinRateModelTargetParams,
    pub requested_positions: usize,
    pub observed_positions: usize,
    pub fitted_positions: usize,
    pub decisive_positions: usize,
    pub drawn_positions: usize,
    pub filtered_positions: usize,
    pub bce_loss: f64,
    pub fitted: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ScoreWinrateAnalysisConfig<'a> {
    pub teacher: &'a str,
    /// Number of teacher-prefix positions used to fit the score->win-rate curves.
    pub fit_positions: usize,
    /// Number of positions after the fit prefix used for held-out evaluation.
    pub eval_positions: usize,
    /// Score bucket width used only for the printed/CSV calibration table.
    pub bin_size: u16,
    pub buffer_mb: usize,
    pub loader_threads: usize,
    pub score_drop_abs: Option<u16>,
}

#[derive(Debug, Clone, Copy)]
pub struct ScoreWinrateModelMetrics {
    pub bce: f64,
    pub brier: f64,
}

#[derive(Debug, Clone)]
pub struct ScoreWinrateAnalysisReport {
    pub sigmoid_scale: f32,
    pub sigmoid_fitted: bool,
    pub sigmoid_fit: ScoreWinrateModelMetrics,
    pub sigmoid_eval: ScoreWinrateModelMetrics,
    pub wrm_params: WinRateModelTargetParams,
    pub wrm_fitted: bool,
    pub wrm_fit: ScoreWinrateModelMetrics,
    pub wrm_eval: ScoreWinrateModelMetrics,
    pub requested_fit_positions: usize,
    pub observed_fit_positions: usize,
    pub used_fit_positions: usize,
    pub decisive_fit_positions: usize,
    pub drawn_fit_positions: usize,
    pub filtered_fit_positions: usize,
    pub requested_eval_positions: usize,
    pub observed_eval_positions: usize,
    pub used_eval_positions: usize,
    pub decisive_eval_positions: usize,
    pub drawn_eval_positions: usize,
    pub filtered_eval_positions: usize,
    pub bin_size: u16,
    pub bins: Vec<ScoreWinrateBinReport>,
}

#[derive(Debug, Clone)]
pub struct ScoreWinrateBinReport {
    pub score_min: i32,
    pub score_max: i32,
    pub count: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub empirical: f64,
    pub sigmoid: f64,
    pub wrm: f64,
}

#[derive(Clone, Copy, Default)]
struct WrmTargetOutcomeBin {
    wins: u32,
    losses: u32,
    draws: u32,
}

impl WrmTargetOutcomeBin {
    fn count(self) -> usize {
        self.wins as usize + self.losses as usize + self.draws as usize
    }

    fn add_result(&mut self, result: i8) {
        match result {
            r if r > 0 => self.wins += 1,
            r if r < 0 => self.losses += 1,
            _ => self.draws += 1,
        }
    }

    fn decisive_count(self) -> usize {
        self.wins as usize + self.losses as usize
    }

    fn decisive_positives_negatives(self) -> (f64, f64) {
        (f64::from(self.wins), f64::from(self.losses))
    }
}

#[derive(Default)]
struct ScoreWinrateObservedCounts {
    observed: usize,
    used: usize,
    decisive: usize,
    draws: usize,
    filtered: usize,
}

const SCORE_WINRATE_SIGMOID_SCALE_MIN: i32 = 50;
const SCORE_WINRATE_SIGMOID_SCALE_MAX: i32 = 20_000;
const SCORE_WINRATE_WRM_OFFSET_MAX: i32 = 10_000;
const SCORE_WINRATE_WRM_SCALING_MIN: i32 = 20;
const SCORE_WINRATE_WRM_SCALING_MAX: i32 = 20_000;

pub fn estimate_wrm_target_from_teacher_prefix(
    config: &WrmTargetCalibrationConfig<'_>,
) -> Result<WrmTargetCalibrationReport, TeacherBatchError> {
    if config.positions == 0 {
        return Ok(WrmTargetCalibrationReport {
            params: WinRateModelTargetParams::DEFAULT,
            requested_positions: 0,
            observed_positions: 0,
            fitted_positions: 0,
            decisive_positions: 0,
            drawn_positions: 0,
            filtered_positions: 0,
            bce_loss: 0.0,
            fitted: false,
        });
    }

    let data_files_owned = expand_teacher(config.teacher).map_err(TeacherBatchError::invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(TeacherBatchError::invalid_input)?;
    let mut bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
    let mut observed_positions = 0usize;
    let mut filtered_positions = 0usize;
    let mut decisive_positions = 0usize;
    let mut drawn_positions = 0usize;
    let mut observe = |chunk: &[PackedSfenValue]| {
        let remaining = config.positions.saturating_sub(observed_positions);
        let take = remaining.min(chunk.len());
        for pos in &chunk[..take] {
            observed_positions += 1;
            if let Some(cap) = config.score_drop_abs {
                if pos.score().unsigned_abs() >= cap {
                    filtered_positions += 1;
                    continue;
                }
            }
            let bin = &mut bins[(i32::from(pos.score()) - i32::from(i16::MIN)) as usize];
            match pos.game_result() {
                r if r > 0 => {
                    bin.wins += 1;
                    decisive_positions += 1;
                }
                r if r < 0 => {
                    bin.losses += 1;
                    decisive_positions += 1;
                }
                _ => {
                    bin.draws += 1;
                    drawn_positions += 1;
                }
            }
        }
        observed_positions >= config.positions
    };

    match format {
        DataFormat::Hcpe => {
            let loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(config.positions.min(65_536).max(1))
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, config.positions, &mut observe);
        }
        DataFormat::Hcpe3 => {
            let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.positions.min(65_536).max(1))
                .with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, config.positions, &mut observe);
        }
        DataFormat::Pack => {
            let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.positions.min(65_536).max(1))
                .with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, config.positions, &mut observe);
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, config.positions, &mut observe);
        }
    }

    let (params, bce_loss, fitted) =
        fit_sigmoid_target_params(&bins).unwrap_or((WinRateModelTargetParams::DEFAULT, 0.0, false));
    // Drawn games are observed and reported, but they are intentionally
    // excluded from target-curve fitting. Treating many draws as 0.5 makes
    // the best curve artificially flat and can produce an unusably large
    // sigmoid/WRM scale.
    let fitted_positions = decisive_positions;
    Ok(WrmTargetCalibrationReport {
        params,
        requested_positions: config.positions,
        observed_positions,
        fitted_positions,
        decisive_positions,
        drawn_positions,
        filtered_positions,
        bce_loss,
        fitted,
    })
}

fn fit_sigmoid_target_params(bins: &[WrmTargetOutcomeBin]) -> Option<(WinRateModelTargetParams, f64, bool)> {
    let (scale, bce_loss, fitted) = fit_sigmoid_target_scale(bins)?;
    Some((WinRateModelTargetParams { offset: 0.0, scaling: scale }, bce_loss, fitted))
}

pub fn analyze_score_winrate_from_teacher(
    config: &ScoreWinrateAnalysisConfig<'_>,
) -> Result<ScoreWinrateAnalysisReport, TeacherBatchError> {
    if config.fit_positions == 0 {
        return Err(TeacherBatchError::invalid_input("--fit-positions must be > 0"));
    }
    if config.eval_positions == 0 {
        return Err(TeacherBatchError::invalid_input("--analyze-positions must be > 0"));
    }
    if config.bin_size == 0 {
        return Err(TeacherBatchError::invalid_input("--bin-size must be > 0"));
    }

    let total_limit = config
        .fit_positions
        .checked_add(config.eval_positions)
        .ok_or_else(|| TeacherBatchError::invalid_input("--fit-positions + --analyze-positions overflowed usize"))?;
    let data_files_owned = expand_teacher(config.teacher).map_err(TeacherBatchError::invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(TeacherBatchError::invalid_input)?;

    let mut fit_bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
    let mut eval_bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
    let mut fit_counts = ScoreWinrateObservedCounts::default();
    let mut eval_counts = ScoreWinrateObservedCounts::default();
    let mut observe = |chunk: &[PackedSfenValue]| {
        for pos in chunk {
            if fit_counts.observed < config.fit_positions {
                observe_score_winrate_position(pos, config.score_drop_abs, &mut fit_bins, &mut fit_counts);
            } else if eval_counts.observed < config.eval_positions {
                observe_score_winrate_position(pos, config.score_drop_abs, &mut eval_bins, &mut eval_counts);
            } else {
                return true;
            }
        }
        fit_counts.observed >= config.fit_positions && eval_counts.observed >= config.eval_positions
    };

    match format {
        DataFormat::Hcpe => {
            let loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(total_limit.min(65_536).max(1))
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, total_limit, &mut observe);
        }
        DataFormat::Hcpe3 => {
            let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(total_limit.min(65_536).max(1))
                .with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, total_limit, &mut observe);
        }
        DataFormat::Pack => {
            let loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(total_limit.min(65_536).max(1))
                .with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, total_limit, &mut observe);
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
            scan_packed_teacher_prefix(&loader, total_limit, &mut observe);
        }
    }

    if fit_counts.used == 0 {
        return Err(TeacherBatchError::invalid_input(
            "teacher did not yield usable fit positions; lower --fit-positions or --score-drop-abs",
        ));
    }
    if eval_counts.used == 0 {
        return Err(TeacherBatchError::invalid_input(
            "teacher did not yield usable held-out analysis positions; lower --fit-positions/--analyze-positions or --score-drop-abs",
        ));
    }

    let (wrm_params, _, wrm_fitted) =
        fit_wrm_target_params(&fit_bins).unwrap_or((WinRateModelTargetParams::DEFAULT, 0.0, false));
    let (sigmoid_scale, _, sigmoid_fitted) = fit_sigmoid_target_scale(&fit_bins).unwrap_or((600.0, 0.0, false));

    let sigmoid_fit = score_winrate_metrics(&fit_bins, |score| sigmoid_score_probability(score, sigmoid_scale))
        .ok_or_else(|| TeacherBatchError::invalid_input("fit set has no usable score/result samples"))?;
    let sigmoid_eval = score_winrate_metrics(&eval_bins, |score| sigmoid_score_probability(score, sigmoid_scale))
        .ok_or_else(|| TeacherBatchError::invalid_input("held-out set has no usable score/result samples"))?;
    let wrm_fit = score_winrate_metrics(&fit_bins, |score| wrm_params.probability(score))
        .ok_or_else(|| TeacherBatchError::invalid_input("fit set has no usable score/result samples"))?;
    let wrm_eval = score_winrate_metrics(&eval_bins, |score| wrm_params.probability(score))
        .ok_or_else(|| TeacherBatchError::invalid_input("held-out set has no usable score/result samples"))?;
    let bins = score_winrate_bin_reports(&eval_bins, config.bin_size, sigmoid_scale, wrm_params);

    Ok(ScoreWinrateAnalysisReport {
        sigmoid_scale,
        sigmoid_fitted,
        sigmoid_fit,
        sigmoid_eval,
        wrm_params,
        wrm_fitted,
        wrm_fit,
        wrm_eval,
        requested_fit_positions: config.fit_positions,
        observed_fit_positions: fit_counts.observed,
        used_fit_positions: fit_counts.used,
        decisive_fit_positions: fit_counts.decisive,
        drawn_fit_positions: fit_counts.draws,
        filtered_fit_positions: fit_counts.filtered,
        requested_eval_positions: config.eval_positions,
        observed_eval_positions: eval_counts.observed,
        used_eval_positions: eval_counts.used,
        decisive_eval_positions: eval_counts.decisive,
        drawn_eval_positions: eval_counts.draws,
        filtered_eval_positions: eval_counts.filtered,
        bin_size: config.bin_size,
        bins,
    })
}

fn observe_score_winrate_position(
    pos: &PackedSfenValue,
    score_drop_abs: Option<u16>,
    bins: &mut [WrmTargetOutcomeBin],
    counts: &mut ScoreWinrateObservedCounts,
) {
    counts.observed += 1;
    if let Some(cap) = score_drop_abs {
        if pos.score().unsigned_abs() >= cap {
            counts.filtered += 1;
            return;
        }
    }
    let score = pos.score();
    let result = pos.game_result();
    let bin = &mut bins[(i32::from(score) - i32::from(i16::MIN)) as usize];
    bin.add_result(result);
    counts.used += 1;
    if result == 0 {
        counts.draws += 1;
    } else {
        counts.decisive += 1;
    }
}

fn scan_packed_teacher_prefix<D, F>(loader: &D, limit: usize, f: &mut F)
where
    D: DataLoader<PackedSfenValue>,
    F: FnMut(&[PackedSfenValue]) -> bool,
{
    if limit == 0 {
        return;
    }
    loader.map_chunks(0, |chunk| f(chunk));
}

fn fit_wrm_target_params(bins: &[WrmTargetOutcomeBin]) -> Option<(WinRateModelTargetParams, f64, bool)> {
    let mut samples = Vec::new();
    let mut fitted_positions = 0usize;
    let mut total_pos = 0.0f64;
    let mut total_neg = 0.0f64;
    for (idx, bin) in bins.iter().enumerate() {
        let count = bin.decisive_count();
        if count == 0 {
            continue;
        }
        let score = (idx as i32 + i32::from(i16::MIN)) as f32;
        let (positives, negatives) = bin.decisive_positives_negatives();
        samples.push((score, positives, negatives));
        fitted_positions += count;
        total_pos += positives;
        total_neg += negatives;
    }
    if fitted_positions == 0 || total_pos <= 0.0 || total_neg <= 0.0 {
        return None;
    }

    let mut best_candidates = vec![WinRateModelTargetParams::DEFAULT];
    if let Some((scale, _, _)) = fit_sigmoid_target_scale(bins) {
        best_candidates.push(WinRateModelTargetParams { offset: 0.0, scaling: scale });
    }
    let mut best = best_candidates[0];
    let mut best_loss = wrm_target_bce_loss(&samples, best);
    for params in best_candidates.into_iter().skip(1) {
        let loss = wrm_target_bce_loss(&samples, params);
        if loss < best_loss {
            best = params;
            best_loss = loss;
        }
    }

    for offset in (0..=SCORE_WINRATE_WRM_OFFSET_MAX).step_by(40) {
        for scaling in (SCORE_WINRATE_WRM_SCALING_MIN..=SCORE_WINRATE_WRM_SCALING_MAX).step_by(40) {
            let params = WinRateModelTargetParams { offset: offset as f32, scaling: scaling as f32 };
            let loss = wrm_target_bce_loss(&samples, params);
            if loss < best_loss {
                best = params;
                best_loss = loss;
            }
        }
    }

    for step in [20i32, 10, 5, 2, 1] {
        let span = step * 6;
        let offset_min = ((best.offset as i32) - span).max(0);
        let offset_max = ((best.offset as i32) + span).min(SCORE_WINRATE_WRM_OFFSET_MAX);
        let scaling_min = ((best.scaling as i32) - span).max(SCORE_WINRATE_WRM_SCALING_MIN);
        let scaling_max = ((best.scaling as i32) + span).min(SCORE_WINRATE_WRM_SCALING_MAX);
        let mut offset = offset_min;
        while offset <= offset_max {
            let mut scaling = scaling_min;
            while scaling <= scaling_max {
                let params = WinRateModelTargetParams { offset: offset as f32, scaling: scaling as f32 };
                let loss = wrm_target_bce_loss(&samples, params);
                if loss < best_loss {
                    best = params;
                    best_loss = loss;
                }
                scaling += step;
            }
            offset += step;
        }
    }

    let total = total_pos + total_neg;
    Some((best, best_loss / total, true))
}

fn wrm_target_bce_loss(samples: &[(f32, f64, f64)], params: WinRateModelTargetParams) -> f64 {
    let mut loss = 0.0f64;
    for &(score, positives, negatives) in samples {
        let p = f64::from(params.probability(score)).clamp(1.0e-6, 1.0 - 1.0e-6);
        loss -= positives * p.ln() + negatives * (1.0 - p).ln();
    }
    loss
}

fn fit_sigmoid_target_scale(bins: &[WrmTargetOutcomeBin]) -> Option<(f32, f64, bool)> {
    let mut samples = Vec::new();
    let mut fitted_positions = 0usize;
    let mut total_pos = 0.0f64;
    let mut total_neg = 0.0f64;
    for (idx, bin) in bins.iter().enumerate() {
        let count = bin.decisive_count();
        if count == 0 {
            continue;
        }
        let score = (idx as i32 + i32::from(i16::MIN)) as f32;
        let (positives, negatives) = bin.decisive_positives_negatives();
        samples.push((score, positives, negatives));
        fitted_positions += count;
        total_pos += positives;
        total_neg += negatives;
    }
    if fitted_positions == 0 || total_pos <= 0.0 || total_neg <= 0.0 {
        return None;
    }

    let mut best = 600.0f32;
    let mut best_loss = sigmoid_target_bce_loss(&samples, best);

    for scale in (SCORE_WINRATE_SIGMOID_SCALE_MIN..=SCORE_WINRATE_SIGMOID_SCALE_MAX).step_by(10) {
        let scale = scale as f32;
        let loss = sigmoid_target_bce_loss(&samples, scale);
        if loss < best_loss {
            best = scale;
            best_loss = loss;
        }
    }

    for step in [5i32, 2, 1] {
        let span = step * 10;
        let scale_min = ((best as i32) - span).max(SCORE_WINRATE_SIGMOID_SCALE_MIN);
        let scale_max = ((best as i32) + span).min(SCORE_WINRATE_SIGMOID_SCALE_MAX);
        let mut scale = scale_min;
        while scale <= scale_max {
            let candidate = scale as f32;
            let loss = sigmoid_target_bce_loss(&samples, candidate);
            if loss < best_loss {
                best = candidate;
                best_loss = loss;
            }
            scale += step;
        }
    }

    let total = total_pos + total_neg;
    Some((best, best_loss / total, true))
}

fn sigmoid_target_bce_loss(samples: &[(f32, f64, f64)], scale: f32) -> f64 {
    let mut loss = 0.0f64;
    for &(score, positives, negatives) in samples {
        let p = f64::from(sigmoid_score_probability(score, scale)).clamp(1.0e-6, 1.0 - 1.0e-6);
        loss -= positives * p.ln() + negatives * (1.0 - p).ln();
    }
    loss
}

fn sigmoid_score_probability(score: f32, scale: f32) -> f32 {
    logistic_f64(f64::from(score) / f64::from(scale.max(f32::MIN_POSITIVE))) as f32
}

fn logistic_f64(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn score_winrate_metrics<F>(bins: &[WrmTargetOutcomeBin], mut probability: F) -> Option<ScoreWinrateModelMetrics>
where
    F: FnMut(f32) -> f32,
{
    let mut total = 0usize;
    let mut bce = 0.0f64;
    let mut brier = 0.0f64;
    for (idx, bin) in bins.iter().enumerate() {
        let count = bin.decisive_count();
        if count == 0 {
            continue;
        }
        let score = (idx as i32 + i32::from(i16::MIN)) as f32;
        let p = f64::from(probability(score)).clamp(1.0e-6, 1.0 - 1.0e-6);
        let (positives, negatives) = bin.decisive_positives_negatives();
        bce -= positives * p.ln() + negatives * (1.0 - p).ln();
        brier += f64::from(bin.wins) * (1.0 - p).powi(2);
        brier += f64::from(bin.losses) * p.powi(2);
        total += count;
    }

    (total > 0).then_some(ScoreWinrateModelMetrics { bce: bce / total as f64, brier: brier / total as f64 })
}

#[derive(Default)]
struct ScoreWinrateDisplayBin {
    outcome: WrmTargetOutcomeBin,
    sigmoid_sum: f64,
    wrm_sum: f64,
}

fn score_winrate_bin_reports(
    exact_bins: &[WrmTargetOutcomeBin],
    bin_size: u16,
    sigmoid_scale: f32,
    wrm_params: WinRateModelTargetParams,
) -> Vec<ScoreWinrateBinReport> {
    let mut grouped: BTreeMap<i32, ScoreWinrateDisplayBin> = BTreeMap::new();
    for (idx, bin) in exact_bins.iter().enumerate() {
        let count = bin.count();
        if count == 0 {
            continue;
        }
        let score = (idx as i32 + i32::from(i16::MIN)) as i16;
        let lower = score_bin_lower(score, bin_size);
        let entry = grouped.entry(lower).or_default();
        entry.outcome.wins += bin.wins;
        entry.outcome.losses += bin.losses;
        entry.outcome.draws += bin.draws;
        entry.sigmoid_sum += f64::from(sigmoid_score_probability(score as f32, sigmoid_scale)) * count as f64;
        entry.wrm_sum += f64::from(wrm_params.probability(score as f32)) * count as f64;
    }

    grouped
        .into_iter()
        .filter_map(|(score_min, bin)| {
            let total_count = bin.outcome.count();
            if total_count == 0 {
                return None;
            }
            let decisive_count = bin.outcome.decisive_count();
            let empirical =
                if decisive_count == 0 { f64::NAN } else { f64::from(bin.outcome.wins) / decisive_count as f64 };
            Some(ScoreWinrateBinReport {
                score_min,
                score_max: score_min + i32::from(bin_size) - 1,
                count: total_count,
                wins: bin.outcome.wins as usize,
                losses: bin.outcome.losses as usize,
                draws: bin.outcome.draws as usize,
                empirical,
                sigmoid: bin.sigmoid_sum / total_count as f64,
                wrm: bin.wrm_sum / total_count as f64,
            })
        })
        .collect()
}

fn score_bin_lower(score: i16, bin_size: u16) -> i32 {
    let bin_size = i32::from(bin_size);
    i32::from(score).div_euclid(bin_size) * bin_size
}

pub fn load_halfkp_teacher_fast_batch(
    config: &HalfkpTeacherBatchConfig<'_>,
) -> Result<HalfkpTeacherBatch, TeacherBatchError> {
    let mut loaded = None;
    for_each_halfkp_teacher_fast_batch(config, 1, |batch| {
        loaded = Some(batch);
        Ok::<(), TeacherBatchError>(())
    })?;

    loaded.ok_or_else(|| {
        TeacherBatchError::invalid_input(format!(
            "teacher did not yield complete batch index {} of {} positions; use a smaller --batch-size or batch-index",
            config.batch_index, config.batch_size
        ))
    })
}

pub fn for_each_halfkp_teacher_fast_batch<F, E>(
    config: &HalfkpTeacherBatchConfig<'_>,
    batch_count: usize,
    visitor: F,
) -> Result<usize, TeacherBatchError>
where
    F: FnMut(HalfkpTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    validate_config(config)?;
    if batch_count == 0 {
        return Ok(0);
    }

    let data_files_owned = expand_teacher(config.teacher).map_err(TeacherBatchError::invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(TeacherBatchError::invalid_input)?;

    match format {
        DataFormat::Hcpe => {
            let mut loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(config.batch_size)
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(false);
            if let Some(pos) = config.dataloader_resume_pos {
                if pos.plies != 0 {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "HCPE dataloader resume position must have plies=0, got {}",
                        pos.plies
                    )));
                }
            }
            let total_bytes = total_hcpe_teacher_bytes(&data_files_owned)?;
            let (loader_start_position, base_byte_offset) = if let Some(pos) = config.dataloader_resume_pos {
                if config.teacher_shuffle_buffer_batches > 0 {
                    let record_index =
                        fixed_record_resume_start_position("HCPE", pos, crate::value::loader::hcpe::HCPE_RECORD_SIZE)?;
                    (record_index, pos.byte_offset % total_bytes)
                } else {
                    loader = loader.with_exact_resume_offset(pos.byte_offset);
                    (0, pos.byte_offset % total_bytes)
                }
            } else {
                let consumed_records = checked_batch_start_position("HCPE", config.batch_index, config.batch_size)?;
                let base_byte_offset = (consumed_records as u64)
                    .checked_mul(crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64)
                    .ok_or_else(|| {
                        TeacherBatchError::invalid_input(format!(
                            "HCPE dataloader resume byte offset overflow: consumed_records={consumed_records}"
                        ))
                    })?;
                (config.batch_index, base_byte_offset % total_bytes)
            };
            visit_halfkp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    hcpe_dataloader_pos_after_batch(base_byte_offset, total_bytes, config.batch_size, visited_batches)
                },
                visitor,
            )
        }
        DataFormat::Hcpe3 => {
            let mut loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("HCPE3", config.batch_index, config.batch_size)?
            };
            visit_halfkp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Pack => {
            let mut loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("Pack", config.batch_index, config.batch_size)?
            };
            visit_halfkp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(false);
            let record_size = std::mem::size_of::<PackedSfenValue>();
            let total_records = total_fixed_record_teacher_records("PSV", &data_files_owned, record_size)?;
            let (loader_start_position, base_record_index) = match config.dataloader_resume_pos {
                Some(pos) => {
                    let record_index = fixed_record_resume_start_position("PSV", pos, record_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
                None => {
                    let record_index = checked_batch_start_position("PSV", config.batch_index, config.batch_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
            };
            visit_halfkp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    fixed_record_dataloader_pos_after_batch(
                        base_record_index,
                        total_records,
                        record_size,
                        config.batch_size,
                        visited_batches,
                    )
                },
                visitor,
            )
        }
    }
}

pub fn for_each_kp_teacher_fast_batch<F, E>(
    config: &KpTeacherBatchConfig<'_>,
    batch_count: usize,
    visitor: F,
) -> Result<usize, TeacherBatchError>
where
    F: FnMut(KpTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    validate_kp_config(config)?;
    if batch_count == 0 {
        return Ok(0);
    }

    let data_files_owned = expand_teacher(config.teacher).map_err(TeacherBatchError::invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(TeacherBatchError::invalid_input)?;

    match format {
        DataFormat::Hcpe => {
            let mut loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(config.batch_size)
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(false);
            if let Some(pos) = config.dataloader_resume_pos {
                if pos.plies != 0 {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "HCPE dataloader resume position must have plies=0, got {}",
                        pos.plies
                    )));
                }
            }
            let total_bytes = total_hcpe_teacher_bytes(&data_files_owned)?;
            let (loader_start_position, base_byte_offset) = if let Some(pos) = config.dataloader_resume_pos {
                if config.teacher_shuffle_buffer_batches > 0 {
                    let record_index =
                        fixed_record_resume_start_position("HCPE", pos, crate::value::loader::hcpe::HCPE_RECORD_SIZE)?;
                    (record_index, pos.byte_offset % total_bytes)
                } else {
                    loader = loader.with_exact_resume_offset(pos.byte_offset);
                    (0, pos.byte_offset % total_bytes)
                }
            } else {
                let consumed_records = checked_batch_start_position("HCPE", config.batch_index, config.batch_size)?;
                let base_byte_offset = (consumed_records as u64)
                    .checked_mul(crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64)
                    .ok_or_else(|| {
                        TeacherBatchError::invalid_input(format!(
                            "HCPE dataloader resume byte offset overflow: consumed_records={consumed_records}"
                        ))
                    })?;
                (config.batch_index, base_byte_offset % total_bytes)
            };
            visit_kp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    hcpe_dataloader_pos_after_batch(base_byte_offset, total_bytes, config.batch_size, visited_batches)
                },
                visitor,
            )
        }
        DataFormat::Hcpe3 => {
            let mut loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("HCPE3", config.batch_index, config.batch_size)?
            };
            visit_kp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Pack => {
            let mut loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("Pack", config.batch_index, config.batch_size)?
            };
            visit_kp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(false);
            let record_size = std::mem::size_of::<PackedSfenValue>();
            let total_records = total_fixed_record_teacher_records("PSV", &data_files_owned, record_size)?;
            let (loader_start_position, base_record_index) = match config.dataloader_resume_pos {
                Some(pos) => {
                    let record_index = fixed_record_resume_start_position("PSV", pos, record_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
                None => {
                    let record_index = checked_batch_start_position("PSV", config.batch_index, config.batch_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
            };
            visit_kp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    fixed_record_dataloader_pos_after_batch(
                        base_record_index,
                        total_records,
                        record_size,
                        config.batch_size,
                        visited_batches,
                    )
                },
                visitor,
            )
        }
    }
}

pub fn for_each_kppt_teacher_fast_batch<I, F, E>(
    input_getter: I,
    input_label: &'static str,
    config: &KpptTeacherBatchConfig<'_>,
    batch_count: usize,
    visitor: F,
) -> Result<usize, TeacherBatchError>
where
    I: SparseInputType<RequiredDataType = PackedSfenValue> + Send,
    F: FnMut(KpptTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    validate_kppt_config(config)?;
    if batch_count == 0 {
        return Ok(0);
    }

    let data_files_owned = expand_teacher(config.teacher).map_err(TeacherBatchError::invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(TeacherBatchError::invalid_input)?;

    match format {
        DataFormat::Hcpe => {
            let mut loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(config.batch_size)
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(false);
            if let Some(pos) = config.dataloader_resume_pos {
                if pos.plies != 0 {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "HCPE dataloader resume position must have plies=0, got {}",
                        pos.plies
                    )));
                }
            }
            let total_bytes = total_hcpe_teacher_bytes(&data_files_owned)?;
            let (loader_start_position, base_byte_offset) = if let Some(pos) = config.dataloader_resume_pos {
                if config.teacher_shuffle_buffer_batches > 0 {
                    let record_index =
                        fixed_record_resume_start_position("HCPE", pos, crate::value::loader::hcpe::HCPE_RECORD_SIZE)?;
                    (record_index, pos.byte_offset % total_bytes)
                } else {
                    loader = loader.with_exact_resume_offset(pos.byte_offset);
                    (0, pos.byte_offset % total_bytes)
                }
            } else {
                let consumed_records = checked_batch_start_position("HCPE", config.batch_index, config.batch_size)?;
                let base_byte_offset = (consumed_records as u64)
                    .checked_mul(crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64)
                    .ok_or_else(|| {
                        TeacherBatchError::invalid_input(format!(
                            "HCPE dataloader resume byte offset overflow: consumed_records={consumed_records}"
                        ))
                    })?;
                (config.batch_index, base_byte_offset % total_bytes)
            };
            visit_kppt_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    hcpe_dataloader_pos_after_batch(base_byte_offset, total_bytes, config.batch_size, visited_batches)
                },
                visitor,
            )
        }
        DataFormat::Hcpe3 => {
            let mut loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("HCPE3", config.batch_index, config.batch_size)?
            };
            visit_kppt_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Pack => {
            let mut loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("Pack", config.batch_index, config.batch_size)?
            };
            visit_kppt_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(false);
            let record_size = std::mem::size_of::<PackedSfenValue>();
            let total_records = total_fixed_record_teacher_records("PSV", &data_files_owned, record_size)?;
            let (loader_start_position, base_record_index) = match config.dataloader_resume_pos {
                Some(pos) => {
                    let record_index = fixed_record_resume_start_position("PSV", pos, record_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
                None => {
                    let record_index = checked_batch_start_position("PSV", config.batch_index, config.batch_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
            };
            visit_kppt_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    fixed_record_dataloader_pos_after_batch(
                        base_record_index,
                        total_records,
                        record_size,
                        config.batch_size,
                        visited_batches,
                    )
                },
                visitor,
            )
        }
    }
}

pub fn for_each_sfnn_halfka2_teacher_fast_batch<F, E>(
    config: &SfnnTeacherBatchConfig<'_>,
    batch_count: usize,
    visitor: F,
) -> Result<usize, TeacherBatchError>
where
    F: FnMut(SfnnTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    for_each_sfnn_teacher_fast_batch(ShogiHalfKa2, "halfka2", config, batch_count, visitor)
}

pub fn for_each_sfnn_teacher_fast_batch<I, F, E>(
    input_getter: I,
    input_label: &'static str,
    config: &SfnnTeacherBatchConfig<'_>,
    batch_count: usize,
    visitor: F,
) -> Result<usize, TeacherBatchError>
where
    I: SparseInputType<RequiredDataType = PackedSfenValue> + Send,
    F: FnMut(SfnnTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    validate_sfnn_config(config)?;
    if batch_count == 0 {
        return Ok(0);
    }

    let data_files_owned = expand_teacher(config.teacher).map_err(TeacherBatchError::invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(TeacherBatchError::invalid_input)?;

    match format {
        DataFormat::Hcpe => {
            let mut loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(config.batch_size)
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(false);
            if let Some(pos) = config.dataloader_resume_pos {
                if pos.plies != 0 {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "HCPE dataloader resume position must have plies=0, got {}",
                        pos.plies
                    )));
                }
            }
            let total_bytes = total_hcpe_teacher_bytes(&data_files_owned)?;
            let (loader_start_position, base_byte_offset) = if let Some(pos) = config.dataloader_resume_pos {
                if config.teacher_shuffle_buffer_batches > 0 {
                    let record_index =
                        fixed_record_resume_start_position("HCPE", pos, crate::value::loader::hcpe::HCPE_RECORD_SIZE)?;
                    (record_index, pos.byte_offset % total_bytes)
                } else {
                    loader = loader.with_exact_resume_offset(pos.byte_offset);
                    (0, pos.byte_offset % total_bytes)
                }
            } else {
                let consumed_records = checked_batch_start_position("HCPE", config.batch_index, config.batch_size)?;
                let base_byte_offset = (consumed_records as u64)
                    .checked_mul(crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64)
                    .ok_or_else(|| {
                        TeacherBatchError::invalid_input(format!(
                            "HCPE dataloader resume byte offset overflow: consumed_records={consumed_records}"
                        ))
                    })?;
                (config.batch_index, base_byte_offset % total_bytes)
            };
            visit_sfnn_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    hcpe_dataloader_pos_after_batch(base_byte_offset, total_bytes, config.batch_size, visited_batches)
                },
                visitor,
            )
        }
        DataFormat::Hcpe3 => {
            let mut loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("HCPE3", config.batch_index, config.batch_size)?
            };
            visit_sfnn_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Pack => {
            let mut loader = ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(false);
            let offset_handle = loader.consumed_offset_handle();
            let plies_handle = loader.consumed_plies_handle();
            let loader_start_position = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                checked_batch_start_position("Pack", config.batch_index, config.batch_size)?
            };
            visit_sfnn_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                |_| {
                    Some(TeacherDataloaderPos {
                        byte_offset: offset_handle.load(Ordering::Acquire),
                        plies: plies_handle.load(Ordering::Acquire),
                    })
                },
                visitor,
            )
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(false);
            let record_size = std::mem::size_of::<PackedSfenValue>();
            let total_records = total_fixed_record_teacher_records("PSV", &data_files_owned, record_size)?;
            let (loader_start_position, base_record_index) = match config.dataloader_resume_pos {
                Some(pos) => {
                    let record_index = fixed_record_resume_start_position("PSV", pos, record_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
                None => {
                    let record_index = checked_batch_start_position("PSV", config.batch_index, config.batch_size)?;
                    (record_index, (record_index as u64) % total_records)
                }
            };
            visit_sfnn_batches(
                input_getter,
                input_label,
                loader,
                format,
                config,
                batch_count,
                loader_start_position,
                move |visited_batches| {
                    fixed_record_dataloader_pos_after_batch(
                        base_record_index,
                        total_records,
                        record_size,
                        config.batch_size,
                        visited_batches,
                    )
                },
                visitor,
            )
        }
    }
}

fn checked_batch_start_position(
    label: &'static str,
    batch_index: usize,
    batch_size: usize,
) -> Result<usize, TeacherBatchError> {
    batch_index.checked_mul(batch_size).ok_or_else(|| {
        TeacherBatchError::invalid_input(format!(
            "{label} dataloader resume position overflow: batch_index={batch_index} batch_size={batch_size}"
        ))
    })
}

fn total_fixed_record_teacher_records(
    label: &'static str,
    paths: &[String],
    record_size: usize,
) -> Result<u64, TeacherBatchError> {
    if record_size == 0 {
        return Err(TeacherBatchError::invalid_input(format!("{label} record size must be > 0")));
    }
    let mut total = 0u64;
    for path in paths {
        let len = std::fs::metadata(path)
            .map_err(|err| TeacherBatchError::invalid_input(format!("failed to stat {label} teacher {path}: {err}")))?
            .len();
        if len % record_size as u64 != 0 {
            return Err(TeacherBatchError::invalid_input(format!(
                "{label} teacher file {path} has byte size {len}, not aligned to record size {record_size}"
            )));
        }
        total = total
            .checked_add(len / record_size as u64)
            .ok_or_else(|| TeacherBatchError::invalid_input(format!("{label} teacher record count overflow")))?;
    }
    if total == 0 {
        return Err(TeacherBatchError::invalid_input(format!("{label} teacher contains no records")));
    }
    Ok(total)
}

fn fixed_record_resume_start_position(
    label: &'static str,
    pos: TeacherDataloaderPos,
    record_size: usize,
) -> Result<usize, TeacherBatchError> {
    if pos.plies != 0 {
        return Err(TeacherBatchError::invalid_input(format!(
            "{label} dataloader resume position must have plies=0, got {}",
            pos.plies
        )));
    }
    if record_size == 0 {
        return Err(TeacherBatchError::invalid_input(format!("{label} record size must be > 0")));
    }
    if pos.byte_offset % record_size as u64 != 0 {
        return Err(TeacherBatchError::invalid_input(format!(
            "{label} dataloader resume byte offset {} is not aligned to record size {record_size}",
            pos.byte_offset
        )));
    }
    let record_index = (pos.byte_offset / record_size as u64) as usize;
    Ok(record_index)
}

fn fixed_record_dataloader_pos_after_batch(
    base_record_index: u64,
    total_records: u64,
    record_size: usize,
    batch_size: usize,
    visited_batches: usize,
) -> Option<TeacherDataloaderPos> {
    if total_records == 0 {
        return None;
    }
    let completed_batches = visited_batches.checked_add(1)?;
    let consumed_records = (completed_batches as u64).checked_mul(batch_size as u64)?;
    let record_index = base_record_index.checked_add(consumed_records)? % total_records;
    let byte_offset = record_index.checked_mul(record_size as u64)?;
    Some(TeacherDataloaderPos { byte_offset, plies: 0 })
}

fn total_hcpe_teacher_bytes(paths: &[String]) -> Result<u64, TeacherBatchError> {
    let mut total = 0u64;
    for path in paths {
        let len = std::fs::metadata(path)
            .map_err(|err| TeacherBatchError::invalid_input(format!("failed to stat HCPE teacher {path}: {err}")))?
            .len();
        total = total.checked_add(len).ok_or_else(|| {
            TeacherBatchError::invalid_input(format!("HCPE teacher byte size overflow while adding {path}"))
        })?;
    }
    if total == 0 {
        return Err(TeacherBatchError::invalid_input("HCPE teacher contains no bytes"));
    }
    Ok(total)
}

fn hcpe_dataloader_pos_after_batch(
    base_byte_offset: u64,
    total_bytes: u64,
    batch_size: usize,
    visited_batches: usize,
) -> Option<TeacherDataloaderPos> {
    let completed_batches = visited_batches.checked_add(1)?;
    let consumed_records = completed_batches.checked_mul(batch_size)?;
    let consumed_bytes = (consumed_records as u64).checked_mul(crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64)?;
    let raw_offset = base_byte_offset.checked_add(consumed_bytes)?;
    Some(TeacherDataloaderPos { byte_offset: raw_offset % total_bytes, plies: 0 })
}

fn validate_config(config: &HalfkpTeacherBatchConfig<'_>) -> Result<(), TeacherBatchError> {
    if config.batch_size == 0 {
        return Err(TeacherBatchError::invalid_input("--batch-size must be greater than zero"));
    }
    validate_teacher_shuffle_size(config.batch_size, config.teacher_shuffle_buffer_batches)?;
    if !(0.0..=1.0).contains(&config.lambda) {
        return Err(TeacherBatchError::invalid_input("--lambda must be in [0, 1]"));
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(TeacherBatchError::invalid_input("--scale must be finite and > 0"));
    }
    validate_wrm_target(config.win_rate_model, config.wrm_target)?;
    Ok(())
}

fn validate_kp_config(config: &KpTeacherBatchConfig<'_>) -> Result<(), TeacherBatchError> {
    if config.batch_size == 0 {
        return Err(TeacherBatchError::invalid_input("--batch-size must be greater than zero"));
    }
    validate_teacher_shuffle_size(config.batch_size, config.teacher_shuffle_buffer_batches)?;
    if !(0.0..=1.0).contains(&config.lambda) {
        return Err(TeacherBatchError::invalid_input("--lambda must be in [0, 1]"));
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(TeacherBatchError::invalid_input("--scale must be finite and > 0"));
    }
    validate_wrm_target(config.win_rate_model, config.wrm_target)?;
    Ok(())
}

fn validate_kppt_config(config: &KpptTeacherBatchConfig<'_>) -> Result<(), TeacherBatchError> {
    if config.batch_size == 0 {
        return Err(TeacherBatchError::invalid_input("--batch-size must be greater than zero"));
    }
    validate_teacher_shuffle_size(config.batch_size, config.teacher_shuffle_buffer_batches)?;
    if !(0.0..=1.0).contains(&config.lambda) {
        return Err(TeacherBatchError::invalid_input("--lambda must be in [0, 1]"));
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(TeacherBatchError::invalid_input("--scale must be finite and > 0"));
    }
    validate_wrm_target(config.win_rate_model, config.wrm_target)?;
    Ok(())
}

fn validate_sfnn_config(config: &SfnnTeacherBatchConfig<'_>) -> Result<(), TeacherBatchError> {
    if config.batch_size == 0 {
        return Err(TeacherBatchError::invalid_input("--batch-size must be greater than zero"));
    }
    validate_teacher_shuffle_size(config.batch_size, config.teacher_shuffle_buffer_batches)?;
    if !(0.0..=1.0).contains(&config.lambda) {
        return Err(TeacherBatchError::invalid_input("--lambda must be in [0, 1]"));
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(TeacherBatchError::invalid_input("--scale must be finite and > 0"));
    }
    validate_wrm_target(config.win_rate_model, config.wrm_target)?;
    Ok(())
}

fn validate_wrm_target(enabled: bool, params: WinRateModelTargetParams) -> Result<(), TeacherBatchError> {
    if enabled && WinRateModelTargetParams::new(params.offset, params.scaling).is_none() {
        return Err(TeacherBatchError::invalid_input(format!(
            "WRM target parameters must be finite and scaling > 0 (offset={}, scaling={})",
            params.offset, params.scaling
        )));
    }
    Ok(())
}

fn validate_teacher_shuffle_size(batch_size: usize, buffer_batches: usize) -> Result<(), TeacherBatchError> {
    if buffer_batches == 0 {
        return Ok(());
    }
    teacher_shuffle_window_records(batch_size, buffer_batches).ok_or_else(|| {
        TeacherBatchError::invalid_input(format!(
            "teacher shuffle buffer size overflow: batch_size={batch_size}, buffer_batches={buffer_batches}"
        ))
    })?;
    Ok(())
}

fn visit_halfkp_batches<D, P, F, E>(
    loader: D,
    format: DataFormat,
    config: &HalfkpTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_position: usize,
    dataloader_pos: P,
    visitor: F,
) -> Result<usize, TeacherBatchError>
where
    D: DataLoader<PackedSfenValue>,
    P: FnMut(usize) -> Option<TeacherDataloaderPos> + Send,
    F: FnMut(HalfkpTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    if config.ft_factorize {
        visit_halfkp_batches_with_input(
            Factorised::from_parts(ShogiHalfKP, ShogiHalfKPPieceFactorizer),
            loader,
            format,
            config,
            batch_count,
            loader_start_position,
            dataloader_pos,
            visitor,
        )
    } else {
        visit_halfkp_batches_direct(loader, format, config, batch_count, loader_start_position, dataloader_pos, visitor)
    }
}

fn prepare_halfkp_direct_fast_batch(
    data: &[PackedSfenValue],
    config: &HalfkpTeacherBatchConfig<'_>,
    threads: usize,
    rayon_pool: Option<&rayon::ThreadPool>,
) -> FastBatchHost {
    let batch_size = data.len();
    let max_active = HALFKP_MAX_ACTIVE_FEATURES;
    let sparse_len = batch_size * max_active;
    let mut batch = FastBatchHost {
        layout: FastBatchLayout { batch_size, max_active, output_size: 1, hand_count_dim: 0 },
        stm: vec![-1; sparse_len],
        nstm: vec![-1; sparse_len],
        buckets: vec![0; batch_size],
        targets: vec![0.0; batch_size],
        weights: vec![0.0; batch_size],
        hand_count: None,
    };

    let chunk_size = batch_size.div_ceil(threads.max(1));
    let sparse_chunk_size = max_active * chunk_size;
    let result_blend = 1.0 - config.lambda;
    let score_blend = config.lambda;
    let rscale = 1.0 / config.scale;
    let wrm_score_table = config.win_rate_model.then(|| win_rate_model_score_table(config.wrm_target));
    let fill_chunk = |data_chunk: &[PackedSfenValue],
                      stm_chunk: &mut [i32],
                      nstm_chunk: &mut [i32],
                      targets: &mut [f32],
                      weights: &mut [f32]| {
        for i in 0..data_chunk.len() {
            let pos = &data_chunk[i];
            let sparse_offset = max_active * i;
            let (stm_count, nstm_count) = fill_halfkp_feature_indices(
                pos,
                &mut stm_chunk[sparse_offset..sparse_offset + max_active],
                &mut nstm_chunk[sparse_offset..sparse_offset + max_active],
            );
            assert!(
                stm_count <= max_active && nstm_count <= max_active,
                "More inputs provided than the specified maximum!"
            );

            let mut weight = 1.0;
            if let Some(cap) = config.score_drop_abs {
                if pos.score().unsigned_abs() >= cap {
                    weight = 0.0;
                }
            }
            weights[i] = weight;

            let score = if config.win_rate_model {
                win_rate_model_score_from_table(
                    wrm_score_table.as_deref().expect("WRM score table must be initialised"),
                    pos.score(),
                )
            } else {
                let score = f32::from(pos.score());
                1.0 / (1.0 + (-rscale * score).exp())
            };
            let result = match pos.game_result() {
                r if r > 0 => 1.0,
                r if r < 0 => 0.0,
                _ => 0.5,
            };
            targets[i] = result_blend * result + score_blend * score;
        }
    };

    if let Some(pool) = rayon_pool
        && threads > 1
        && batch_size > 1
    {
        pool.install(|| {
            data.par_chunks(chunk_size)
                .zip(batch.stm.par_chunks_mut(sparse_chunk_size))
                .zip(batch.nstm.par_chunks_mut(sparse_chunk_size))
                .zip(batch.targets.par_chunks_mut(chunk_size))
                .zip(batch.weights.par_chunks_mut(chunk_size))
                .for_each(|((((data_chunk, stm_chunk), nstm_chunk), targets), weights)| {
                    fill_chunk(data_chunk, stm_chunk, nstm_chunk, targets, weights);
                });
        });
    } else {
        data.chunks(chunk_size)
            .zip(batch.stm.chunks_mut(sparse_chunk_size))
            .zip(batch.nstm.chunks_mut(sparse_chunk_size))
            .zip(batch.targets.chunks_mut(chunk_size))
            .zip(batch.weights.chunks_mut(chunk_size))
            .for_each(|((((data_chunk, stm_chunk), nstm_chunk), targets), weights)| {
                fill_chunk(data_chunk, stm_chunk, nstm_chunk, targets, weights);
            });
    }

    batch
}

fn prepare_kp_direct_fast_batch(
    data: &[PackedSfenValue],
    config: &KpTeacherBatchConfig<'_>,
    threads: usize,
    rayon_pool: Option<&rayon::ThreadPool>,
) -> FastBatchHost {
    let batch_size = data.len();
    let max_active = KP_MAX_ACTIVE;
    let sparse_len = batch_size * max_active;
    let mut batch = FastBatchHost {
        layout: FastBatchLayout { batch_size, max_active, output_size: 1, hand_count_dim: 0 },
        stm: vec![-1; sparse_len],
        nstm: vec![-1; sparse_len],
        buckets: vec![0; batch_size],
        targets: vec![0.0; batch_size],
        weights: vec![0.0; batch_size],
        hand_count: None,
    };

    let chunk_size = batch_size.div_ceil(threads.max(1));
    let sparse_chunk_size = max_active * chunk_size;
    let result_blend = 1.0 - config.lambda;
    let score_blend = config.lambda;
    let rscale = 1.0 / config.scale;
    let wrm_score_table = config.win_rate_model.then(|| win_rate_model_score_table(config.wrm_target));
    let fill_chunk = |data_chunk: &[PackedSfenValue],
                      stm_chunk: &mut [i32],
                      nstm_chunk: &mut [i32],
                      targets: &mut [f32],
                      weights: &mut [f32]| {
        for i in 0..data_chunk.len() {
            let pos = &data_chunk[i];
            let sparse_offset = max_active * i;
            let (stm_count, nstm_count) = fill_kp_feature_indices(
                pos,
                &mut stm_chunk[sparse_offset..sparse_offset + max_active],
                &mut nstm_chunk[sparse_offset..sparse_offset + max_active],
            );
            assert!(
                stm_count <= max_active && nstm_count <= max_active,
                "More inputs provided than the specified maximum!"
            );

            let mut weight = 1.0;
            if let Some(cap) = config.score_drop_abs {
                if pos.score().unsigned_abs() >= cap {
                    weight = 0.0;
                }
            }
            weights[i] = weight;

            let score = if config.win_rate_model {
                win_rate_model_score_from_table(
                    wrm_score_table.as_deref().expect("WRM score table must be initialised"),
                    pos.score(),
                )
            } else {
                let score = f32::from(pos.score());
                1.0 / (1.0 + (-rscale * score).exp())
            };
            let result = match pos.game_result() {
                r if r > 0 => 1.0,
                r if r < 0 => 0.0,
                _ => 0.5,
            };
            targets[i] = result_blend * result + score_blend * score;
        }
    };

    if let Some(pool) = rayon_pool
        && threads > 1
        && batch_size > 1
    {
        pool.install(|| {
            data.par_chunks(chunk_size)
                .zip(batch.stm.par_chunks_mut(sparse_chunk_size))
                .zip(batch.nstm.par_chunks_mut(sparse_chunk_size))
                .zip(batch.targets.par_chunks_mut(chunk_size))
                .zip(batch.weights.par_chunks_mut(chunk_size))
                .for_each(|((((data_chunk, stm_chunk), nstm_chunk), targets), weights)| {
                    fill_chunk(data_chunk, stm_chunk, nstm_chunk, targets, weights);
                });
        });
    } else {
        data.chunks(chunk_size)
            .zip(batch.stm.chunks_mut(sparse_chunk_size))
            .zip(batch.nstm.chunks_mut(sparse_chunk_size))
            .zip(batch.targets.chunks_mut(chunk_size))
            .zip(batch.weights.chunks_mut(chunk_size))
            .for_each(|((((data_chunk, stm_chunk), nstm_chunk), targets), weights)| {
                fill_chunk(data_chunk, stm_chunk, nstm_chunk, targets, weights);
            });
    }

    batch
}

fn load_and_map_packed_batches<D, F>(
    loader: &D,
    start_position: usize,
    batch_size: usize,
    shuffle_buffer_batches: usize,
    shuffle_seed: u64,
    mut f: F,
) where
    D: DataLoader<PackedSfenValue>,
    F: FnMut(&[PackedSfenValue]) -> bool,
{
    if shuffle_buffer_batches > 0 {
        load_and_map_shuffled_batches_with_prefetch(
            loader.clone(),
            start_position,
            batch_size,
            shuffle_buffer_batches,
            shuffle_seed,
            f,
        );
        return;
    }

    let mut incomplete_buf = Vec::new();

    loader.map_chunks(start_position, |chunk| {
        let remainder = if !incomplete_buf.is_empty() {
            let remainder = batch_size - incomplete_buf.len();

            if chunk.len() >= remainder {
                incomplete_buf.extend_from_slice(&chunk[..remainder]);
                let should_break = f(&incomplete_buf);
                incomplete_buf.clear();

                if should_break {
                    return true;
                }
            } else {
                incomplete_buf.extend_from_slice(chunk);
            }

            remainder
        } else {
            0
        };

        if chunk.len() >= remainder {
            let chunks = chunk[remainder..chunk.len()].chunks_exact(batch_size);
            incomplete_buf.extend_from_slice(chunks.remainder());

            for batch in chunks {
                let should_break = f(batch);

                if should_break {
                    return true;
                }
            }
        }

        false
    });
}
fn visit_halfkp_batches_direct<D, P, F, E>(
    loader: D,
    format: DataFormat,
    config: &HalfkpTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_position: usize,
    mut dataloader_pos: P,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    D: DataLoader<PackedSfenValue> + Send,
    P: FnMut(usize) -> Option<TeacherDataloaderPos> + Send,
    F: FnMut(HalfkpTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    let threads = config.threads.max(1);
    let rayon_pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("bulletou-halfkp-direct-prepare-{index}"))
                .build()
                .map_err(|err| {
                    TeacherBatchError::invalid_input(format!(
                        "failed to create HalfKP direct teacher prepare thread pool with {threads} threads: {err}"
                    ))
                })?,
        )
    } else {
        None
    };

    if config.queue_depth > 1 && !config.profile_prepare {
        let (sender, receiver) =
            mpsc::sync_channel::<Result<HalfkpTeacherBatch, TeacherBatchError>>(config.queue_depth);
        return std::thread::scope(|scope| {
            let producer = scope.spawn(move || -> Result<usize, TeacherBatchError> {
                let mut produced_batches = 0usize;
                let mut producer_error = None;
                load_and_map_packed_batches(
                    &loader,
                    loader_start_position,
                    config.batch_size,
                    config.teacher_shuffle_buffer_batches,
                    config.teacher_shuffle_seed,
                    |raw_batch| {
                    let batch_index = config.batch_index + produced_batches;
                    let batch = prepare_halfkp_direct_fast_batch(raw_batch, config, threads, rayon_pool.as_ref());
                    if let Err(err) = batch.validate() {
                        producer_error = Some(TeacherBatchError::invalid_input(err.to_string()));
                        return true;
                    }

                    let source = format!("{format:?} teacher batch {batch_index}: {}", config.teacher);
                    let dataloader_pos = dataloader_pos(produced_batches);
                    if sender.send(Ok(HalfkpTeacherBatch { batch, source, dataloader_pos })).is_err() {
                        return true;
                    }

                    produced_batches += 1;
                    produced_batches >= batch_count
                    },
                );

                if let Some(err) = producer_error {
                    let _ = sender.send(Err(err.clone()));
                    return Err(err);
                }
                if produced_batches != batch_count {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                        config.batch_index, config.batch_size
                    )));
                }
                Ok(produced_batches)
            });

            let mut consumed_batches = 0usize;
            let mut visit_error = None;
            while consumed_batches < batch_count {
                match receiver.recv() {
                    Ok(Ok(batch)) => {
                        if let Err(err) = visitor(batch) {
                            visit_error = Some(TeacherBatchError::invalid_input(format!(
                                "teacher batch callback failed at batch {}: {err}",
                                config.batch_index + consumed_batches
                            )));
                            break;
                        }
                        consumed_batches += 1;
                    }
                    Ok(Err(err)) => {
                        visit_error = Some(err);
                        break;
                    }
                    Err(_) => break,
                }
            }
            drop(receiver);

            let producer_result = producer
                .join()
                .map_err(|_| TeacherBatchError::invalid_input("HalfKP direct teacher producer thread panicked"))?;
            if let Some(err) = visit_error {
                return Err(err);
            }
            producer_result?;
            if consumed_batches != batch_count {
                return Err(TeacherBatchError::invalid_input(format!(
                    "teacher did not deliver {batch_count} complete prepared batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                    config.batch_index, config.batch_size
                )));
            }
            Ok(consumed_batches)
        });
    }

    let mut visited_batches = 0usize;
    let mut visit_error = None;
    load_and_map_packed_batches(
        &loader,
        loader_start_position,
        config.batch_size,
        config.teacher_shuffle_buffer_batches,
        config.teacher_shuffle_seed,
        |raw_batch| {
            let batch_index = config.batch_index + visited_batches;
            let prepare_started = config.profile_prepare.then(std::time::Instant::now);
            let batch = prepare_halfkp_direct_fast_batch(raw_batch, config, threads, rayon_pool.as_ref());
            if let Some(started) = prepare_started {
                println!(
                    "  profile_teacher : batch={batch_index:<6} prepare {:>9.3} ms",
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
            if let Err(err) = batch.validate() {
                visit_error = Some(TeacherBatchError::invalid_input(err.to_string()));
                return true;
            }

            let source = format!("{format:?} teacher batch {batch_index}: {}", config.teacher);
            let dataloader_pos = dataloader_pos(visited_batches);
            if let Err(err) = visitor(HalfkpTeacherBatch { batch, source, dataloader_pos }) {
                visit_error = Some(TeacherBatchError::invalid_input(format!(
                    "teacher batch callback failed at batch {batch_index}: {err}"
                )));
                return true;
            }

            visited_batches += 1;
            visited_batches >= batch_count
        },
    );
    if let Some(err) = visit_error {
        return Err(err);
    }
    if visited_batches != batch_count {
        return Err(TeacherBatchError::invalid_input(format!(
            "teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
            config.batch_index, config.batch_size
        )));
    }
    Ok(visited_batches)
}

fn visit_kp_batches<D, P, F, E>(
    loader: D,
    format: DataFormat,
    config: &KpTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_position: usize,
    mut dataloader_pos: P,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    D: DataLoader<PackedSfenValue> + Send,
    P: FnMut(usize) -> Option<TeacherDataloaderPos> + Send,
    F: FnMut(KpTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    let threads = config.threads.max(1);
    let rayon_pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("bulletou-kp-direct-prepare-{index}"))
                .build()
                .map_err(|err| {
                    TeacherBatchError::invalid_input(format!(
                        "failed to create KP direct teacher prepare thread pool with {threads} threads: {err}"
                    ))
                })?,
        )
    } else {
        None
    };

    if config.queue_depth > 1 && !config.profile_prepare {
        let (sender, receiver) = mpsc::sync_channel::<Result<KpTeacherBatch, TeacherBatchError>>(config.queue_depth);
        return std::thread::scope(|scope| {
            let producer = scope.spawn(move || -> Result<usize, TeacherBatchError> {
                let mut produced_batches = 0usize;
                let mut producer_error = None;
                load_and_map_packed_batches(
                    &loader,
                    loader_start_position,
                    config.batch_size,
                    config.teacher_shuffle_buffer_batches,
                    config.teacher_shuffle_seed,
                    |raw_batch| {
                    let batch_index = config.batch_index + produced_batches;
                    let batch = prepare_kp_direct_fast_batch(raw_batch, config, threads, rayon_pool.as_ref());
                    if let Err(err) = batch.validate() {
                        producer_error = Some(TeacherBatchError::invalid_input(err.to_string()));
                        return true;
                    }

                    let source = format!("{format:?} KP teacher batch {batch_index}: {}", config.teacher);
                    let dataloader_pos = dataloader_pos(produced_batches);
                    if sender.send(Ok(KpTeacherBatch { batch, source, dataloader_pos })).is_err() {
                        return true;
                    }

                    produced_batches += 1;
                    produced_batches >= batch_count
                    },
                );

                if let Some(err) = producer_error {
                    let _ = sender.send(Err(err.clone()));
                    return Err(err);
                }
                if produced_batches != batch_count {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "teacher did not yield {batch_count} complete KP batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                        config.batch_index, config.batch_size
                    )));
                }
                Ok(produced_batches)
            });

            let mut consumed_batches = 0usize;
            let mut visit_error = None;
            while consumed_batches < batch_count {
                match receiver.recv() {
                    Ok(Ok(batch)) => {
                        if let Err(err) = visitor(batch) {
                            visit_error = Some(TeacherBatchError::invalid_input(format!(
                                "teacher batch callback failed at KP batch {}: {err}",
                                config.batch_index + consumed_batches
                            )));
                            break;
                        }
                        consumed_batches += 1;
                    }
                    Ok(Err(err)) => {
                        visit_error = Some(err);
                        break;
                    }
                    Err(_) => break,
                }
            }
            drop(receiver);

            let producer_result =
                producer.join().map_err(|_| TeacherBatchError::invalid_input("KP teacher producer thread panicked"))?;
            if let Some(err) = visit_error {
                return Err(err);
            }
            producer_result?;
            if consumed_batches != batch_count {
                return Err(TeacherBatchError::invalid_input(format!(
                    "teacher did not deliver {batch_count} complete prepared KP batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                    config.batch_index, config.batch_size
                )));
            }
            Ok(consumed_batches)
        });
    }

    let mut visited_batches = 0usize;
    let mut visit_error = None;
    load_and_map_packed_batches(
        &loader,
        loader_start_position,
        config.batch_size,
        config.teacher_shuffle_buffer_batches,
        config.teacher_shuffle_seed,
        |raw_batch| {
            let batch_index = config.batch_index + visited_batches;
            let prepare_started = config.profile_prepare.then(std::time::Instant::now);
            let batch = prepare_kp_direct_fast_batch(raw_batch, config, threads, rayon_pool.as_ref());
            if let Some(started) = prepare_started {
                println!(
                    "  profile_teacher : batch={batch_index:<6} prepare {:>9.3} ms",
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
            if let Err(err) = batch.validate() {
                visit_error = Some(TeacherBatchError::invalid_input(err.to_string()));
                return true;
            }

            let source = format!("{format:?} KP teacher batch {batch_index}: {}", config.teacher);
            let dataloader_pos = dataloader_pos(visited_batches);
            if let Err(err) = visitor(KpTeacherBatch { batch, source, dataloader_pos }) {
                visit_error = Some(TeacherBatchError::invalid_input(format!(
                    "teacher batch callback failed at KP batch {batch_index}: {err}"
                )));
                return true;
            }

            visited_batches += 1;
            visited_batches >= batch_count
        },
    );
    if let Some(err) = visit_error {
        return Err(err);
    }
    if visited_batches != batch_count {
        return Err(TeacherBatchError::invalid_input(format!(
            "teacher did not yield {batch_count} complete KP batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
            config.batch_index, config.batch_size
        )));
    }
    Ok(visited_batches)
}

fn visit_halfkp_batches_with_input<I, D, P, F, E>(
    input_getter: I,
    loader: D,
    format: DataFormat,
    config: &HalfkpTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_position: usize,
    mut dataloader_pos: P,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    I: SparseInputType<RequiredDataType = PackedSfenValue> + Send,
    D: DataLoader<PackedSfenValue> + Send,
    P: FnMut(usize) -> Option<TeacherDataloaderPos> + Send,
    F: FnMut(HalfkpTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    let threads = config.threads.max(1);
    let rayon_pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("bulletou-halfkp-prepare-{index}"))
                .build()
                .map_err(|err| {
                    TeacherBatchError::invalid_input(format!(
                        "failed to create HalfKP teacher prepare thread pool with {threads} threads: {err}"
                    ))
                })?,
        )
    } else {
        None
    };
    let dataloader = DefaultDataLoader::new(
        input_getter,
        NoOutputBuckets,
        (|_, blend| blend) as fn(&PackedSfenValue, f32) -> f32,
        None,
        config.win_rate_model,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    )
    .with_win_rate_model_target(config.wrm_target)
    .with_teacher_shuffle(config.teacher_shuffle_buffer_batches, config.teacher_shuffle_seed);

    if config.queue_depth > 1 && !config.profile_prepare {
        let (sender, receiver) =
            mpsc::sync_channel::<Result<HalfkpTeacherBatch, TeacherBatchError>>(config.queue_depth);
        return std::thread::scope(|scope| {
            let producer = scope.spawn(move || -> Result<usize, TeacherBatchError> {
                let mut produced_batches = 0usize;
                let mut producer_error = None;
                dataloader.load_and_map_batches_from_position(loader_start_position, config.batch_size, |batch| {
                    let batch_index = config.batch_index + produced_batches;
                    let prepared = match rayon_pool.as_ref() {
                        Some(pool) => dataloader.prepare_with_pool(batch, pool, threads, 1.0 - config.lambda),
                        None => dataloader.prepare(batch, threads, 1.0 - config.lambda),
                    };
                    let batch = FastBatchHost::from(prepared);
                    if let Err(err) = batch.validate() {
                        producer_error = Some(TeacherBatchError::invalid_input(err.to_string()));
                        return true;
                    }

                    let source = format!("{format:?} teacher batch {batch_index}: {}", config.teacher);
                    let dataloader_pos = dataloader_pos(produced_batches);
                    if sender.send(Ok(HalfkpTeacherBatch { batch, source, dataloader_pos })).is_err() {
                        return true;
                    }

                    produced_batches += 1;
                    produced_batches >= batch_count
                });

                if let Some(err) = producer_error {
                    let _ = sender.send(Err(err.clone()));
                    return Err(err);
                }
                if produced_batches != batch_count {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                        config.batch_index, config.batch_size
                    )));
                }
                Ok(produced_batches)
            });

            let mut consumed_batches = 0usize;
            let mut visit_error = None;
            while consumed_batches < batch_count {
                match receiver.recv() {
                    Ok(Ok(batch)) => {
                        if let Err(err) = visitor(batch) {
                            visit_error = Some(TeacherBatchError::invalid_input(format!(
                                "teacher batch callback failed at batch {}: {err}",
                                config.batch_index + consumed_batches
                            )));
                            break;
                        }
                        consumed_batches += 1;
                    }
                    Ok(Err(err)) => {
                        visit_error = Some(err);
                        break;
                    }
                    Err(_) => break,
                }
            }
            drop(receiver);

            let producer_result = producer
                .join()
                .map_err(|_| TeacherBatchError::invalid_input("HalfKP teacher producer thread panicked"))?;
            if let Some(err) = visit_error {
                return Err(err);
            }
            producer_result?;
            if consumed_batches != batch_count {
                return Err(TeacherBatchError::invalid_input(format!(
                    "teacher did not deliver {batch_count} complete prepared batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                    config.batch_index, config.batch_size
                )));
            }
            Ok(consumed_batches)
        });
    }

    let mut visited_batches = 0usize;
    let mut visit_error = None;
    dataloader.load_and_map_batches_from_position(loader_start_position, config.batch_size, |batch| {
        let batch_index = config.batch_index + visited_batches;
        let prepare_started = config.profile_prepare.then(std::time::Instant::now);
        let prepared = match rayon_pool.as_ref() {
            Some(pool) => dataloader.prepare_with_pool(batch, pool, threads, 1.0 - config.lambda),
            None => dataloader.prepare(batch, threads, 1.0 - config.lambda),
        };
        if let Some(started) = prepare_started {
            println!(
                "  profile_teacher : batch={batch_index:<6} prepare {:>9.3} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let batch = FastBatchHost::from(prepared);
        if let Err(err) = batch.validate() {
            visit_error = Some(TeacherBatchError::invalid_input(err.to_string()));
            return true;
        }

        let source = format!("{format:?} teacher batch {batch_index}: {}", config.teacher);
        let dataloader_pos = dataloader_pos(visited_batches);
        if let Err(err) = visitor(HalfkpTeacherBatch { batch, source, dataloader_pos }) {
            visit_error = Some(TeacherBatchError::invalid_input(format!(
                "teacher batch callback failed at batch {batch_index}: {err}"
            )));
            return true;
        }

        visited_batches += 1;
        visited_batches >= batch_count
    });

    if let Some(err) = visit_error {
        return Err(err);
    }
    if visited_batches != batch_count {
        return Err(TeacherBatchError::invalid_input(format!(
            "teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
            config.batch_index, config.batch_size
        )));
    }

    Ok(visited_batches)
}

fn visit_kppt_batches<I, D, P, F, E>(
    input_getter: I,
    input_label: &'static str,
    loader: D,
    format: DataFormat,
    config: &KpptTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_position: usize,
    mut dataloader_pos: P,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    I: SparseInputType<RequiredDataType = PackedSfenValue> + Send,
    D: DataLoader<PackedSfenValue> + Send,
    P: FnMut(usize) -> Option<TeacherDataloaderPos> + Send,
    F: FnMut(KpptTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    let threads = config.threads.max(1);
    let rayon_pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |index| format!("bulletou-kppt-{input_label}-prepare-{index}"))
                .build()
                .map_err(|err| {
                    TeacherBatchError::invalid_input(format!(
                        "failed to create KPPT {input_label} teacher prepare thread pool with {threads} threads: {err}"
                    ))
                })?,
        )
    } else {
        None
    };
    let dataloader = DefaultDataLoader::new(
        input_getter,
        NoOutputBuckets,
        (|_, blend| blend) as fn(&PackedSfenValue, f32) -> f32,
        None,
        config.win_rate_model,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    )
    .with_win_rate_model_target(config.wrm_target)
    .with_teacher_shuffle(config.teacher_shuffle_buffer_batches, config.teacher_shuffle_seed);

    if config.queue_depth > 1 && !config.profile_prepare {
        let (sender, receiver) = mpsc::sync_channel::<Result<KpptTeacherBatch, TeacherBatchError>>(config.queue_depth);
        return std::thread::scope(|scope| {
            let producer = scope.spawn(move || -> Result<usize, TeacherBatchError> {
                let mut produced_batches = 0usize;
                let mut producer_error = None;
                dataloader.load_and_map_batches_from_position(loader_start_position, config.batch_size, |batch| {
                    let batch_index = config.batch_index + produced_batches;
                    let prepared = match rayon_pool.as_ref() {
                        Some(pool) => dataloader.prepare_with_pool(batch, pool, threads, 1.0 - config.lambda),
                        None => dataloader.prepare(batch, threads, 1.0 - config.lambda),
                    };
                    let batch = FastBatchHost::from(prepared);
                    if let Err(err) = batch.validate() {
                        producer_error = Some(TeacherBatchError::invalid_input(err.to_string()));
                        return true;
                    }

                    let source = format!("{format:?} KPPT/{input_label} teacher batch {batch_index}: {}", config.teacher);
                    let dataloader_pos = dataloader_pos(produced_batches);
                    if sender.send(Ok(KpptTeacherBatch { batch, source, dataloader_pos })).is_err() {
                        return true;
                    }

                    produced_batches += 1;
                    produced_batches >= batch_count
                });

                if let Some(err) = producer_error {
                    let _ = sender.send(Err(err.clone()));
                    return Err(err);
                }
                if produced_batches != batch_count {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "KPPT/{input_label} teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                        config.batch_index, config.batch_size
                    )));
                }
                Ok(produced_batches)
            });

            let mut consumed_batches = 0usize;
            let mut visit_error = None;
            while consumed_batches < batch_count {
                match receiver.recv() {
                    Ok(Ok(batch)) => {
                        if let Err(err) = visitor(batch) {
                            visit_error = Some(TeacherBatchError::invalid_input(format!(
                                "KPPT/{input_label} teacher batch callback failed at batch {}: {err}",
                                config.batch_index + consumed_batches
                            )));
                            break;
                        }
                        consumed_batches += 1;
                    }
                    Ok(Err(err)) => {
                        visit_error = Some(err);
                        break;
                    }
                    Err(_) => break,
                }
            }
            drop(receiver);

            let producer_result = producer.join().map_err(|_| {
                TeacherBatchError::invalid_input(format!("KPPT/{input_label} teacher producer thread panicked"))
            })?;
            if let Some(err) = visit_error {
                return Err(err);
            }
            producer_result?;
            if consumed_batches != batch_count {
                return Err(TeacherBatchError::invalid_input(format!(
                    "KPPT/{input_label} teacher did not deliver {batch_count} complete prepared batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                    config.batch_index, config.batch_size
                )));
            }
            Ok(consumed_batches)
        });
    }

    let mut visited_batches = 0usize;
    let mut visit_error = None;
    dataloader.load_and_map_batches_from_position(loader_start_position, config.batch_size, |batch| {
        let batch_index = config.batch_index + visited_batches;
        let prepare_started = config.profile_prepare.then(std::time::Instant::now);
        let prepared = match rayon_pool.as_ref() {
            Some(pool) => dataloader.prepare_with_pool(batch, pool, threads, 1.0 - config.lambda),
            None => dataloader.prepare(batch, threads, 1.0 - config.lambda),
        };
        if let Some(started) = prepare_started {
            println!(
                "  profile_teacher : input={input_label:<3} batch={batch_index:<6} prepare {:>9.3} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let batch = FastBatchHost::from(prepared);
        if let Err(err) = batch.validate() {
            visit_error = Some(TeacherBatchError::invalid_input(err.to_string()));
            return true;
        }

        let source = format!("{format:?} KPPT/{input_label} teacher batch {batch_index}: {}", config.teacher);
        let dataloader_pos = dataloader_pos(visited_batches);
        if let Err(err) = visitor(KpptTeacherBatch { batch, source, dataloader_pos }) {
            visit_error = Some(TeacherBatchError::invalid_input(format!(
                "KPPT/{input_label} teacher batch callback failed at batch {batch_index}: {err}"
            )));
            return true;
        }

        visited_batches += 1;
        visited_batches >= batch_count
    });

    if let Some(err) = visit_error {
        return Err(err);
    }
    if visited_batches != batch_count {
        return Err(TeacherBatchError::invalid_input(format!(
            "KPPT/{input_label} teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
            config.batch_index, config.batch_size
        )));
    }

    Ok(visited_batches)
}

fn visit_sfnn_batches<I, D, P, F, E>(
    input_getter: I,
    input_label: &'static str,
    loader: D,
    format: DataFormat,
    config: &SfnnTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_position: usize,
    mut dataloader_pos: P,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    I: SparseInputType<RequiredDataType = PackedSfenValue> + Send,
    D: DataLoader<PackedSfenValue> + Send,
    P: FnMut(usize) -> Option<TeacherDataloaderPos> + Send,
    F: FnMut(SfnnTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    let threads = config.threads.max(1);
    let input_size = input_getter.num_inputs();
    let max_active = input_getter.max_active();
    let rayon_pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |index| format!("bulletou-sfnn-{input_label}-prepare-{index}"))
                .build()
                .map_err(|err| {
                    TeacherBatchError::invalid_input(format!(
                        "failed to create SFNN/{input_label} teacher prepare thread pool with {threads} threads: {err}"
                    ))
                })?,
        )
    } else {
        None
    };
    let dataloader = DefaultDataLoader::new(
        input_getter,
        ShogiSfnnLayerStackBucket::new(config.layerstack_bucket),
        (|_, blend| blend) as fn(&PackedSfenValue, f32) -> f32,
        None,
        config.win_rate_model,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    )
    .with_win_rate_model_target(config.wrm_target)
    .with_teacher_shuffle(config.teacher_shuffle_buffer_batches, config.teacher_shuffle_seed);

    if config.queue_depth > 1 && !config.profile_prepare {
        let (sender, receiver) = mpsc::sync_channel::<Result<SfnnTeacherBatch, TeacherBatchError>>(config.queue_depth);
        return std::thread::scope(|scope| {
            let producer = scope.spawn(move || -> Result<usize, TeacherBatchError> {
                let mut produced_batches = 0usize;
                let mut producer_error = None;
                let mut producer_ready = std::time::Instant::now();
                dataloader.load_and_map_batches_from_position(loader_start_position, config.batch_size, |batch| {
                    let producer_load_sec = producer_ready.elapsed().as_secs_f64();
                    let batch_index = config.batch_index + produced_batches;
                    let prepare_started = std::time::Instant::now();
                    let batch = match input_label {
                        "halfka2" => prepare_sfnn_fast_batch_from_board_features(
                            input_label,
                            batch,
                            config,
                            rayon_pool.as_ref(),
                            fill_halfka2_feature_indices_from_board,
                            input_size,
                            max_active,
                        ),
                        "ka2" => prepare_sfnn_fast_batch_from_board_features(
                            input_label,
                            batch,
                            config,
                            rayon_pool.as_ref(),
                            fill_ka2_feature_indices_from_board,
                            input_size,
                            max_active,
                        ),
                        _ => {
                            let prepared = match rayon_pool.as_ref() {
                                Some(pool) => dataloader.prepare_with_pool(batch, pool, threads, 1.0 - config.lambda),
                                None => dataloader.prepare(batch, threads, 1.0 - config.lambda),
                            };
                            FastBatchHost::from(prepared)
                        }
                    };
                    let producer_prepare_sec = prepare_started.elapsed().as_secs_f64();
                    if let Err(err) = batch.validate() {
                        producer_error = Some(TeacherBatchError::invalid_input(err.to_string()));
                        return true;
                    }

                    let source =
                        format!("{format:?} SFNN/{input_label} teacher batch {batch_index}: {}", config.teacher);
                    let dataloader_pos = dataloader_pos(produced_batches);
                    let timing = TeacherBatchTiming {
                        producer_load_sec,
                        producer_prepare_sec,
                        consumer_queue_wait_sec: 0.0,
                    };
                    if sender
                        .send(Ok(SfnnTeacherBatch { batch, source, dataloader_pos, timing }))
                        .is_err()
                    {
                        return true;
                    }

                    produced_batches += 1;
                    producer_ready = std::time::Instant::now();
                    produced_batches >= batch_count
                });

                if let Some(err) = producer_error {
                    let _ = sender.send(Err(err.clone()));
                    return Err(err);
                }
                if produced_batches != batch_count {
                    return Err(TeacherBatchError::invalid_input(format!(
                        "SFNN/{input_label} teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                        config.batch_index, config.batch_size
                    )));
                }
                Ok(produced_batches)
            });

            let mut consumed_batches = 0usize;
            let mut visit_error = None;
            while consumed_batches < batch_count {
                let recv_started = std::time::Instant::now();
                match receiver.recv() {
                    Ok(Ok(mut batch)) => {
                        batch.timing.consumer_queue_wait_sec = recv_started.elapsed().as_secs_f64();
                        if let Err(err) = visitor(batch) {
                            visit_error = Some(TeacherBatchError::invalid_input(format!(
                                "SFNN/{input_label} teacher batch callback failed at batch {}: {err}",
                                config.batch_index + consumed_batches
                            )));
                            break;
                        }
                        consumed_batches += 1;
                    }
                    Ok(Err(err)) => {
                        visit_error = Some(err);
                        break;
                    }
                    Err(_) => break,
                }
            }
            drop(receiver);

            let producer_result = producer.join().map_err(|_| {
                TeacherBatchError::invalid_input(format!("SFNN/{input_label} teacher producer thread panicked"))
            })?;
            if let Some(err) = visit_error {
                return Err(err);
            }
            producer_result?;
            if consumed_batches != batch_count {
                return Err(TeacherBatchError::invalid_input(format!(
                    "SFNN/{input_label} teacher did not deliver {batch_count} complete prepared batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
                    config.batch_index, config.batch_size
                )));
            }
            Ok(consumed_batches)
        });
    }

    let mut visited_batches = 0usize;
    let mut visit_error = None;
    let mut producer_ready = std::time::Instant::now();
    dataloader.load_and_map_batches_from_position(loader_start_position, config.batch_size, |batch| {
        let producer_load_sec = producer_ready.elapsed().as_secs_f64();
        let batch_index = config.batch_index + visited_batches;
        let prepare_started = config.profile_prepare.then(std::time::Instant::now);
        let prepare_timing_started = std::time::Instant::now();
        let batch = match input_label {
            "halfka2" => prepare_sfnn_fast_batch_from_board_features(
                input_label,
                batch,
                config,
                rayon_pool.as_ref(),
                fill_halfka2_feature_indices_from_board,
                input_size,
                max_active,
            ),
            "ka2" => prepare_sfnn_fast_batch_from_board_features(
                input_label,
                batch,
                config,
                rayon_pool.as_ref(),
                fill_ka2_feature_indices_from_board,
                input_size,
                max_active,
            ),
            _ => {
                let prepared = match rayon_pool.as_ref() {
                    Some(pool) => dataloader.prepare_with_pool(batch, pool, threads, 1.0 - config.lambda),
                    None => dataloader.prepare(batch, threads, 1.0 - config.lambda),
                };
                FastBatchHost::from(prepared)
            }
        };
        let producer_prepare_sec = prepare_timing_started.elapsed().as_secs_f64();
        if let Some(started) = prepare_started {
            println!(
                "  profile_teacher : batch={batch_index:<6} prepare {:>9.3} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        if let Err(err) = batch.validate() {
            visit_error = Some(TeacherBatchError::invalid_input(err.to_string()));
            return true;
        }

        let source = format!("{format:?} SFNN/{input_label} teacher batch {batch_index}: {}", config.teacher);
        let dataloader_pos = dataloader_pos(visited_batches);
        let timing = TeacherBatchTiming { producer_load_sec, producer_prepare_sec, consumer_queue_wait_sec: 0.0 };
        if let Err(err) = visitor(SfnnTeacherBatch { batch, source, dataloader_pos, timing }) {
            visit_error = Some(TeacherBatchError::invalid_input(format!(
                "SFNN/{input_label} teacher batch callback failed at batch {batch_index}: {err}"
            )));
            return true;
        }

        visited_batches += 1;
        producer_ready = std::time::Instant::now();
        visited_batches >= batch_count
    });

    if let Some(err) = visit_error {
        return Err(err);
    }
    if visited_batches != batch_count {
        return Err(TeacherBatchError::invalid_input(format!(
            "SFNN/{input_label} teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
            config.batch_index, config.batch_size
        )));
    }

    Ok(visited_batches)
}

fn prepare_sfnn_fast_batch_from_board_features(
    input_label: &'static str,
    data: &[PackedSfenValue],
    config: &SfnnTeacherBatchConfig<'_>,
    pool: Option<&rayon::ThreadPool>,
    fill_features: fn(&ShogiBoard, &mut [i32], &mut [i32]) -> (usize, usize),
    input_size: usize,
    max_active: usize,
) -> FastBatchHost {
    let batch_size = data.len();
    let sparse_len = max_active * batch_size;
    let mut batch = FastBatchHost {
        layout: FastBatchLayout { batch_size, max_active, output_size: 1, hand_count_dim: 0 },
        stm: vec![-1; sparse_len],
        nstm: vec![-1; sparse_len],
        buckets: vec![0; batch_size],
        targets: vec![0.0; batch_size],
        weights: vec![0.0; batch_size],
        hand_count: None,
    };
    let threads = config.threads.max(1);
    let chunk_size = batch_size.div_ceil(threads).max(1);
    let sparse_chunk_size = max_active * chunk_size;
    let target_blend = 1.0 - config.lambda;
    let rscale = 1.0 / config.scale;
    let layerstack_bucket = config.layerstack_bucket;
    let wrm_score_table = config.win_rate_model.then(|| win_rate_model_score_table(config.wrm_target));

    let fill_chunk = |chunk_index: usize,
                      data_chunk: &[PackedSfenValue],
                      stm_chunk: &mut [i32],
                      nstm_chunk: &mut [i32],
                      buckets_chunk: &mut [i32],
                      targets_chunk: &mut [f32],
                      weights_chunk: &mut [f32]| {
        for (i, pos) in data_chunk.iter().enumerate() {
            let board = pos.decode();
            let sparse_offset = max_active * i;
            let (stm_count, nstm_count) = fill_features(
                &board,
                &mut stm_chunk[sparse_offset..sparse_offset + max_active],
                &mut nstm_chunk[sparse_offset..sparse_offset + max_active],
            );
            assert!(
                stm_count <= max_active && nstm_count <= max_active,
                "SFNN/{input_label} active feature count exceeded max_active {max_active}"
            );
            for &idx in &stm_chunk[sparse_offset..sparse_offset + stm_count] {
                assert!(
                    idx >= 0 && (idx as usize) < input_size,
                    "SFNN/{input_label} STM feature index exceeded input size"
                );
            }
            for &idx in &nstm_chunk[sparse_offset..sparse_offset + nstm_count] {
                assert!(
                    idx >= 0 && (idx as usize) < input_size,
                    "SFNN/{input_label} NSTM feature index exceeded input size"
                );
            }

            buckets_chunk[i] = layerstack_bucket.bucket_from_board(&board) as i32;
            let mut weight = 1.0;
            if let Some(cap) = config.score_drop_abs {
                if pos.score().unsigned_abs() >= cap {
                    weight = 0.0;
                }
            }
            weights_chunk[i] = weight;

            let score = if config.win_rate_model {
                win_rate_model_score_from_table(
                    wrm_score_table.as_deref().expect("WRM score table must be initialised"),
                    pos.score(),
                )
            } else {
                let score = f32::from(pos.score());
                1.0 / (1.0 + (-(rscale * score)).exp())
            };
            let result = match pos.game_result() {
                r if r > 0 => 1.0,
                r if r < 0 => 0.0,
                _ => 0.5,
            };
            targets_chunk[i] = target_blend * result + (1.0 - target_blend) * score;
        }
        let _ = chunk_index;
    };

    if let Some(pool) = pool
        && threads > 1
        && batch_size > 1
    {
        pool.install(|| {
            data.par_chunks(chunk_size)
                .enumerate()
                .zip(batch.stm.par_chunks_mut(sparse_chunk_size))
                .zip(batch.nstm.par_chunks_mut(sparse_chunk_size))
                .zip(batch.buckets.par_chunks_mut(chunk_size))
                .zip(batch.targets.par_chunks_mut(chunk_size))
                .zip(batch.weights.par_chunks_mut(chunk_size))
                .for_each(
                    |(((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), targets_chunk), weights_chunk)| {
                        let (chunk_index, data_chunk) = data_chunk;
                        fill_chunk(
                            chunk_index,
                            data_chunk,
                            stm_chunk,
                            nstm_chunk,
                            buckets_chunk,
                            targets_chunk,
                            weights_chunk,
                        );
                    },
                );
        });
    } else {
        for (chunk_index, data_chunk) in data.chunks(chunk_size).enumerate() {
            let start = chunk_index * chunk_size;
            let sparse_start = start * max_active;
            let sparse_end = sparse_start + data_chunk.len() * max_active;
            let end = start + data_chunk.len();
            fill_chunk(
                chunk_index,
                data_chunk,
                &mut batch.stm[sparse_start..sparse_end],
                &mut batch.nstm[sparse_start..sparse_end],
                &mut batch.buckets[start..end],
                &mut batch.targets[start..end],
                &mut batch.weights[start..end],
            );
        }
    }

    batch
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_teacher_path(name: &str, ext: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("bulletou_teacher_batch_test_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("teacher.{ext}"));
        fs::write(&path, b"").unwrap();
        path
    }

    fn write_tiny_pack(path: &std::path::Path) {
        let mut bytes = Vec::new();
        bytes.push(1); // hirate start position
        for (move16, eval) in [
            (59u16 | (60u16 << 7), 10i16),
            (21u16 | (20u16 << 7), -20i16),
            (14u16 | (15u16 << 7), 30i16),
            (66u16 | (65u16 << 7), -40i16),
        ] {
            bytes.extend_from_slice(&move16.to_le_bytes());
            bytes.extend_from_slice(&eval.to_le_bytes());
        }
        bytes.extend_from_slice(&(1u16 | (1u16 << 7)).to_le_bytes()); // black-win end marker
        bytes.push(0); // reason
        fs::write(path, bytes).unwrap();
    }

    fn config() -> HalfkpTeacherBatchConfig<'static> {
        HalfkpTeacherBatchConfig {
            teacher: "missing.hcpe",
            batch_size: 2,
            batch_index: 0,
            dataloader_resume_pos: None,
            buffer_mb: 1,
            loader_threads: 1,
            threads: 1,
            queue_depth: 1,
            lambda: 1.0,
            scale: 400.0,
            win_rate_model: false,
            wrm_target: WinRateModelTargetParams::DEFAULT,
            ft_factorize: false,
            score_drop_abs: Some(32_000),
            teacher_shuffle_buffer_batches: 0,
            teacher_shuffle_seed: 0,
            profile_prepare: false,
        }
    }

    fn sfnn_config() -> SfnnTeacherBatchConfig<'static> {
        SfnnTeacherBatchConfig {
            teacher: "missing.hcpe",
            batch_size: 2,
            batch_index: 0,
            dataloader_resume_pos: None,
            layerstack_bucket: ShogiSfnnLayerStackBucketKind::KingRank9,
            buffer_mb: 1,
            loader_threads: 1,
            threads: 1,
            queue_depth: 2,
            lambda: 1.0,
            scale: 400.0,
            win_rate_model: false,
            wrm_target: WinRateModelTargetParams::DEFAULT,
            score_drop_abs: Some(32_000),
            teacher_shuffle_buffer_batches: 0,
            teacher_shuffle_seed: 0,
            profile_prepare: false,
        }
    }

    fn score_bin_index(score: i16) -> usize {
        (i32::from(score) - i32::from(i16::MIN)) as usize
    }

    #[test]
    fn score_winrate_wrm_fit_beats_plain_sigmoid_on_wrm_shaped_data() {
        let expected = WinRateModelTargetParams { offset: 260.0, scaling: 360.0 };
        let mut bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
        for score in (-1200..=1200).step_by(25) {
            let p = expected.probability(score as f32);
            let count = 2000u32;
            let wins = (p * count as f32).round() as u32;
            let losses = count - wins;
            bins[score_bin_index(score as i16)] = WrmTargetOutcomeBin { wins, losses, draws: 0 };
        }

        let (sigmoid_scale, _, sigmoid_fitted) = fit_sigmoid_target_scale(&bins).unwrap();
        let (wrm_params, _, wrm_fitted) = fit_wrm_target_params(&bins).unwrap();
        let sigmoid_metrics =
            score_winrate_metrics(&bins, |score| sigmoid_score_probability(score, sigmoid_scale)).unwrap();
        let wrm_metrics = score_winrate_metrics(&bins, |score| wrm_params.probability(score)).unwrap();

        assert!(sigmoid_fitted);
        assert!(wrm_fitted);
        assert!((wrm_params.offset - expected.offset).abs() <= 2.0, "wrm_params={wrm_params:?}");
        assert!((wrm_params.scaling - expected.scaling).abs() <= 2.0, "wrm_params={wrm_params:?}");
        assert!(wrm_metrics.bce < sigmoid_metrics.bce, "wrm={wrm_metrics:?}, sigmoid={sigmoid_metrics:?}");
    }

    #[test]
    fn score_winrate_wrm_fit_can_fall_back_to_plain_sigmoid_shape() {
        let expected_scale = 2400.0f32;
        let mut bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
        for score in (-1800..=1800).step_by(25) {
            let p = sigmoid_score_probability(score as f32, expected_scale);
            let count = 2000u32;
            let wins = (p * count as f32).round() as u32;
            let losses = count - wins;
            bins[score_bin_index(score as i16)] = WrmTargetOutcomeBin { wins, losses, draws: 0 };
        }

        let (sigmoid_scale, _, _) = fit_sigmoid_target_scale(&bins).unwrap();
        let (wrm_params, _, _) = fit_wrm_target_params(&bins).unwrap();
        let sigmoid_metrics =
            score_winrate_metrics(&bins, |score| sigmoid_score_probability(score, sigmoid_scale)).unwrap();
        let wrm_metrics = score_winrate_metrics(&bins, |score| wrm_params.probability(score)).unwrap();

        assert!((sigmoid_scale - expected_scale).abs() <= 2.0, "sigmoid_scale={sigmoid_scale}");
        assert!(
            wrm_metrics.bce <= sigmoid_metrics.bce + 1.0e-7,
            "wrm={wrm_metrics:?}, sigmoid={sigmoid_metrics:?}, wrm_params={wrm_params:?}"
        );
    }

    #[test]
    fn default_target_fit_uses_plain_sigmoid_params() {
        let expected_scale = 1800.0f32;
        let mut bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
        for score in (-1200..=1200).step_by(25) {
            let p = sigmoid_score_probability(score as f32, expected_scale);
            let count = 2000u32;
            let wins = (p * count as f32).round() as u32;
            let losses = count - wins;
            bins[score_bin_index(score as i16)] = WrmTargetOutcomeBin { wins, losses, draws: 0 };
        }

        let (params, _, fitted) = fit_sigmoid_target_params(&bins).unwrap();

        assert!(fitted);
        assert_eq!(params.offset, 0.0);
        assert!((params.scaling - expected_scale).abs() <= 2.0, "params={params:?}");
    }

    #[test]
    fn target_fit_ignores_draws() {
        let expected_scale = 1000.0f32;
        let mut decisive_bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
        let mut draw_heavy_bins = vec![WrmTargetOutcomeBin::default(); usize::from(u16::MAX) + 1];
        for score in (-2000..=2000).step_by(25) {
            let p = sigmoid_score_probability(score as f32, expected_scale);
            let count = 2000u32;
            let wins = (p * count as f32).round() as u32;
            let losses = count - wins;
            let idx = score_bin_index(score as i16);
            decisive_bins[idx] = WrmTargetOutcomeBin { wins, losses, draws: 0 };
            // These draws would flatten the fitted curve badly if they were
            // treated as 0.5 labels. They must not affect target fitting.
            draw_heavy_bins[idx] = WrmTargetOutcomeBin { wins, losses, draws: 20_000 };
        }

        let (decisive_scale, decisive_bce, _) = fit_sigmoid_target_scale(&decisive_bins).unwrap();
        let (draw_heavy_scale, draw_heavy_bce, _) = fit_sigmoid_target_scale(&draw_heavy_bins).unwrap();
        let (decisive_wrm, decisive_wrm_bce, _) = fit_wrm_target_params(&decisive_bins).unwrap();
        let (draw_heavy_wrm, draw_heavy_wrm_bce, _) = fit_wrm_target_params(&draw_heavy_bins).unwrap();

        assert!((decisive_scale - expected_scale).abs() <= 2.0, "decisive_scale={decisive_scale}");
        assert_eq!(draw_heavy_scale, decisive_scale);
        assert_eq!(draw_heavy_bce, decisive_bce);
        assert_eq!(draw_heavy_wrm, decisive_wrm);
        assert_eq!(draw_heavy_wrm_bce, decisive_wrm_bce);
    }

    #[test]
    fn score_winrate_bin_lower_uses_mathematical_floor_for_negative_scores() {
        assert_eq!(score_bin_lower(-1, 50), -50);
        assert_eq!(score_bin_lower(-50, 50), -50);
        assert_eq!(score_bin_lower(-51, 50), -100);
        assert_eq!(score_bin_lower(0, 50), 0);
        assert_eq!(score_bin_lower(49, 50), 0);
        assert_eq!(score_bin_lower(50, 50), 50);
    }

    #[test]
    fn config_rejects_zero_batch_size() {
        let mut config = config();
        config.batch_size = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("batch-size"));
    }

    #[test]
    fn config_rejects_invalid_lambda() {
        let mut config = config();
        config.lambda = 1.5;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("lambda"));
    }

    #[test]
    fn zero_batch_count_does_not_touch_missing_teacher() {
        let visited = for_each_halfkp_teacher_fast_batch(&config(), 0, |_| Ok::<(), TeacherBatchError>(())).unwrap();
        assert_eq!(visited, 0);
    }

    #[test]
    fn sfnn_config_rejects_zero_batch_size() {
        let mut config = sfnn_config();
        config.batch_size = 0;
        let err = validate_sfnn_config(&config).unwrap_err();
        assert!(err.to_string().contains("batch-size"));
    }

    #[test]
    fn sfnn_zero_batch_count_does_not_touch_missing_teacher() {
        let visited =
            for_each_sfnn_halfka2_teacher_fast_batch(&sfnn_config(), 0, |_| Ok::<(), TeacherBatchError>(())).unwrap();
        assert_eq!(visited, 0);
    }

    #[test]
    fn hcpe_resume_rejects_nonzero_plies() {
        let path = tmp_teacher_path("hcpe_resume_rejects_nonzero_plies", "hcpe");
        let teacher = path.to_string_lossy().into_owned();
        let mut config = config();
        config.teacher = Box::leak(teacher.into_boxed_str());
        config.dataloader_resume_pos = Some(TeacherDataloaderPos { byte_offset: 0, plies: 1 });

        let err = for_each_halfkp_teacher_fast_batch(&config, 1, |_| Ok::<(), TeacherBatchError>(())).unwrap_err();
        assert!(err.to_string().contains("plies=0"));

        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn hcpe_dataloader_pos_wraps_at_teacher_end() {
        let total_bytes = 10 * crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64;
        let pos =
            hcpe_dataloader_pos_after_batch(8 * crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64, total_bytes, 4, 0)
                .unwrap();
        assert_eq!(
            pos,
            TeacherDataloaderPos { byte_offset: 2 * crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64, plies: 0 }
        );
    }

    #[test]
    fn psv_resume_offset_maps_to_record_position() {
        let record_size = std::mem::size_of::<PackedSfenValue>();
        let pos = TeacherDataloaderPos { byte_offset: (6 * record_size) as u64, plies: 0 };
        assert_eq!(fixed_record_resume_start_position("PSV", pos, record_size).unwrap(), 6);

        let bad_plies = TeacherDataloaderPos { byte_offset: 0, plies: 1 };
        assert!(
            fixed_record_resume_start_position("PSV", bad_plies, record_size)
                .unwrap_err()
                .to_string()
                .contains("plies=0")
        );

        let bad_alignment = TeacherDataloaderPos { byte_offset: 1, plies: 0 };
        assert!(
            fixed_record_resume_start_position("PSV", bad_alignment, record_size)
                .unwrap_err()
                .to_string()
                .contains("aligned")
        );

        let wrapped = fixed_record_dataloader_pos_after_batch(8, 10, record_size, 4, 0).expect("wrapped PSV position");
        assert_eq!(wrapped, TeacherDataloaderPos { byte_offset: (2 * record_size) as u64, plies: 0 });
    }

    #[test]
    fn pack_resume_position_roundtrips() {
        let path = tmp_teacher_path("pack_resume_position_roundtrips", "pack");
        write_tiny_pack(&path);
        let teacher = path.to_string_lossy().into_owned();
        let mut config = config();
        config.teacher = Box::leak(teacher.into_boxed_str());
        config.batch_size = 2;

        let first = load_halfkp_teacher_fast_batch(&config).unwrap();
        assert!(first.source.starts_with("Pack teacher batch 0"));
        assert_eq!(first.dataloader_pos, Some(TeacherDataloaderPos { byte_offset: 0, plies: 2 }));

        config.batch_index = 1;
        config.dataloader_resume_pos = first.dataloader_pos;
        let second = load_halfkp_teacher_fast_batch(&config).unwrap();
        assert!(second.source.starts_with("Pack teacher batch 1"));
        assert_eq!(second.dataloader_pos, Some(TeacherDataloaderPos { byte_offset: 0, plies: 4 }));

        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
