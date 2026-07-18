//! Teacher-to-`FastBatchHost` helpers shared by fixture exporters and future
//! fast backend trainers.

use std::{error::Error, fmt, sync::atomic::Ordering};

use crate::{
    game::{
        inputs::{Factorised, ShogiHalfKP, ShogiHalfKPPieceFactorizer, ShogiHalfKa2, SparseInputType},
        outputs::ShogiLayerStackBucket9,
    },
    shogi::PackedSfenValue,
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    value::{
        FastBatchHost, NoOutputBuckets,
        loader::{
            DataLoader, DefaultDataLoader, DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader,
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
    /// Lambda on teacher eval score when target values are prepared.
    pub lambda: f32,
    /// Eval-to-score sigmoid scale used while preparing teacher targets.
    pub scale: f32,
    /// Use nnue-pytorch WRM target conversion while preparing teacher targets.
    pub nnue_pytorch_wrm_loss: bool,
    /// Add tatara-style HalfKP piece-input virtual rows to the FT input.
    pub ft_factorize: bool,
    /// Drop positions whose absolute teacher score is at least this value.
    pub score_drop_abs: Option<u16>,
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
pub struct SfnnTeacherBatchConfig<'a> {
    pub teacher: &'a str,
    pub batch_size: usize,
    pub batch_index: usize,
    pub dataloader_resume_pos: Option<TeacherDataloaderPos>,
    pub buffer_mb: usize,
    pub loader_threads: usize,
    pub threads: usize,
    pub lambda: f32,
    pub scale: f32,
    pub nnue_pytorch_wrm_loss: bool,
    pub score_drop_abs: Option<u16>,
    /// Print CPU batch materialisation timing for profiling runs.
    pub profile_prepare: bool,
}

#[derive(Debug, Clone)]
pub struct SfnnTeacherBatch {
    pub batch: FastBatchHost,
    pub source: String,
    pub dataloader_pos: Option<TeacherDataloaderPos>,
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
            let (loader_start_batch, base_byte_offset) = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset);
                (0, pos.byte_offset % total_bytes)
            } else {
                let consumed_records = config.batch_index.checked_mul(config.batch_size).ok_or_else(|| {
                    TeacherBatchError::invalid_input(format!(
                        "HCPE dataloader resume position overflow: batch_index={} batch_size={}",
                        config.batch_index, config.batch_size
                    ))
                })?;
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
                loader_start_batch,
                move |visited_batches| {
                    hcpe_dataloader_pos_after_batch(
                        base_byte_offset,
                        total_bytes,
                        config.batch_size,
                        visited_batches,
                    )
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
            let loader_start_batch = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                config.batch_index
            };
            visit_halfkp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_batch,
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
            let loader_start_batch = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                config.batch_index
            };
            visit_halfkp_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_batch,
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
            visit_halfkp_batches(loader, format, config, batch_count, config.batch_index, |_| None, visitor)
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
            let (loader_start_batch, base_byte_offset) = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset);
                (0, pos.byte_offset % total_bytes)
            } else {
                let consumed_records = config.batch_index.checked_mul(config.batch_size).ok_or_else(|| {
                    TeacherBatchError::invalid_input(format!(
                        "HCPE dataloader resume position overflow: batch_index={} batch_size={}",
                        config.batch_index, config.batch_size
                    ))
                })?;
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
                loader,
                format,
                config,
                batch_count,
                loader_start_batch,
                move |visited_batches| {
                    hcpe_dataloader_pos_after_batch(
                        base_byte_offset,
                        total_bytes,
                        config.batch_size,
                        visited_batches,
                    )
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
            let loader_start_batch = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                config.batch_index
            };
            visit_sfnn_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_batch,
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
            let loader_start_batch = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset, pos.plies);
                0
            } else {
                config.batch_index
            };
            visit_sfnn_batches(
                loader,
                format,
                config,
                batch_count,
                loader_start_batch,
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
            visit_sfnn_batches(loader, format, config, batch_count, config.batch_index, |_| None, visitor)
        }
    }
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
    if !(0.0..=1.0).contains(&config.lambda) {
        return Err(TeacherBatchError::invalid_input("--lambda must be in [0, 1]"));
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(TeacherBatchError::invalid_input("--scale must be finite and > 0"));
    }
    Ok(())
}

fn validate_sfnn_config(config: &SfnnTeacherBatchConfig<'_>) -> Result<(), TeacherBatchError> {
    if config.batch_size == 0 {
        return Err(TeacherBatchError::invalid_input("--batch-size must be greater than zero"));
    }
    if !(0.0..=1.0).contains(&config.lambda) {
        return Err(TeacherBatchError::invalid_input("--lambda must be in [0, 1]"));
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(TeacherBatchError::invalid_input("--scale must be finite and > 0"));
    }
    Ok(())
}

fn visit_halfkp_batches<D, P, F, E>(
    loader: D,
    format: DataFormat,
    config: &HalfkpTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_batch: usize,
    dataloader_pos: P,
    visitor: F,
) -> Result<usize, TeacherBatchError>
where
    D: DataLoader<PackedSfenValue>,
    P: FnMut(usize) -> Option<TeacherDataloaderPos>,
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
            loader_start_batch,
            dataloader_pos,
            visitor,
        )
    } else {
        visit_halfkp_batches_with_input(
            ShogiHalfKP,
            loader,
            format,
            config,
            batch_count,
            loader_start_batch,
            dataloader_pos,
            visitor,
        )
    }
}

fn visit_halfkp_batches_with_input<I, D, P, F, E>(
    input_getter: I,
    loader: D,
    format: DataFormat,
    config: &HalfkpTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_batch: usize,
    mut dataloader_pos: P,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    I: SparseInputType<RequiredDataType = PackedSfenValue>,
    D: DataLoader<PackedSfenValue>,
    P: FnMut(usize) -> Option<TeacherDataloaderPos>,
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
        config.nnue_pytorch_wrm_loss,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    );
    let mut visited_batches = 0usize;
    let mut visit_error = None;
    dataloader.load_and_map_batches(loader_start_batch, config.batch_size, |batch| {
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

fn visit_sfnn_batches<D, P, F, E>(
    loader: D,
    format: DataFormat,
    config: &SfnnTeacherBatchConfig<'_>,
    batch_count: usize,
    loader_start_batch: usize,
    mut dataloader_pos: P,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    D: DataLoader<PackedSfenValue>,
    P: FnMut(usize) -> Option<TeacherDataloaderPos>,
    F: FnMut(SfnnTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    let threads = config.threads.max(1);
    let rayon_pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("bulletou-sfnn-prepare-{index}"))
                .build()
                .map_err(|err| {
                    TeacherBatchError::invalid_input(format!(
                        "failed to create SFNN teacher prepare thread pool with {threads} threads: {err}"
                    ))
                })?,
        )
    } else {
        None
    };
    let dataloader = DefaultDataLoader::new(
        ShogiHalfKa2,
        ShogiLayerStackBucket9::KingRank9,
        (|_, blend| blend) as fn(&PackedSfenValue, f32) -> f32,
        None,
        config.nnue_pytorch_wrm_loss,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    );
    let mut visited_batches = 0usize;
    let mut visit_error = None;
    dataloader.load_and_map_batches(loader_start_batch, config.batch_size, |batch| {
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

        let source = format!("{format:?} SFNN teacher batch {batch_index}: {}", config.teacher);
        let dataloader_pos = dataloader_pos(visited_batches);
        if let Err(err) = visitor(SfnnTeacherBatch { batch, source, dataloader_pos }) {
            visit_error = Some(TeacherBatchError::invalid_input(format!(
                "SFNN teacher batch callback failed at batch {batch_index}: {err}"
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
            "SFNN teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
            config.batch_index, config.batch_size
        )));
    }

    Ok(visited_batches)
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
            lambda: 1.0,
            scale: 400.0,
            nnue_pytorch_wrm_loss: false,
            ft_factorize: false,
            score_drop_abs: Some(32_000),
            profile_prepare: false,
        }
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
        let pos = hcpe_dataloader_pos_after_batch(8 * crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64, total_bytes, 4, 0)
            .unwrap();
        assert_eq!(pos, TeacherDataloaderPos { byte_offset: 2 * crate::value::loader::hcpe::HCPE_RECORD_SIZE as u64, plies: 0 });
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
