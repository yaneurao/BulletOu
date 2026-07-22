//! Teacher-to-`FastBatchHost` helpers shared by fixture exporters and future
//! fast backend trainers.

use std::{
    error::Error,
    fmt,
    sync::{atomic::Ordering, mpsc},
};

use rayon::prelude::*;

use crate::{
    game::{
        inputs::{
            Factorised, HALFKP_MAX_ACTIVE_FEATURES, KP_MAX_ACTIVE, ShogiHalfKP, ShogiHalfKPPieceFactorizer,
            ShogiHalfKa2, SparseInputType, fill_halfkp_feature_indices, fill_kp_feature_indices,
        },
        outputs::{ShogiSfnnLayerStackBucket, ShogiSfnnLayerStackBucketKind},
    },
    shogi::PackedSfenValue,
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    value::{
        FastBatchHost, FastBatchLayout, NoOutputBuckets,
        loader::{
            DataLoader, DefaultDataLoader, DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader,
            ShogiPackLoader, win_rate_model_score,
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
    pub nnue_pytorch_wrm_loss: bool,
    pub score_drop_abs: Option<u16>,
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
    pub nnue_pytorch_wrm_loss: bool,
    pub score_drop_abs: Option<u16>,
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
            let (loader_start_position, base_byte_offset) = if let Some(pos) = config.dataloader_resume_pos {
                loader = loader.with_exact_resume_offset(pos.byte_offset);
                (0, pos.byte_offset % total_bytes)
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
                loader = loader.with_exact_resume_offset(pos.byte_offset);
                (0, pos.byte_offset % total_bytes)
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
                loader = loader.with_exact_resume_offset(pos.byte_offset);
                (0, pos.byte_offset % total_bytes)
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
                loader = loader.with_exact_resume_offset(pos.byte_offset);
                (0, pos.byte_offset % total_bytes)
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
    if !(0.0..=1.0).contains(&config.lambda) {
        return Err(TeacherBatchError::invalid_input("--lambda must be in [0, 1]"));
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(TeacherBatchError::invalid_input("--scale must be finite and > 0"));
    }
    Ok(())
}

fn validate_kp_config(config: &KpTeacherBatchConfig<'_>) -> Result<(), TeacherBatchError> {
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

fn validate_kppt_config(config: &KpptTeacherBatchConfig<'_>) -> Result<(), TeacherBatchError> {
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

            let score = if config.nnue_pytorch_wrm_loss {
                win_rate_model_score(pos.score())
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

            let score = if config.nnue_pytorch_wrm_loss {
                win_rate_model_score(pos.score())
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

fn load_and_map_packed_batches<D, F>(loader: &D, start_position: usize, batch_size: usize, mut f: F)
where
    D: DataLoader<PackedSfenValue>,
    F: FnMut(&[PackedSfenValue]) -> bool,
{
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
                load_and_map_packed_batches(&loader, loader_start_position, config.batch_size, |raw_batch| {
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
    load_and_map_packed_batches(&loader, loader_start_position, config.batch_size, |raw_batch| {
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
                load_and_map_packed_batches(&loader, loader_start_position, config.batch_size, |raw_batch| {
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
                });

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
    load_and_map_packed_batches(&loader, loader_start_position, config.batch_size, |raw_batch| {
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
    });
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
        config.nnue_pytorch_wrm_loss,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    );

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
        config.nnue_pytorch_wrm_loss,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    );

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
        config.nnue_pytorch_wrm_loss,
        false,
        config.scale,
        config.score_drop_abs,
        loader,
    );

    if config.queue_depth > 1 && !config.profile_prepare {
        let (sender, receiver) = mpsc::sync_channel::<Result<SfnnTeacherBatch, TeacherBatchError>>(config.queue_depth);
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

                    let source =
                        format!("{format:?} SFNN/{input_label} teacher batch {batch_index}: {}", config.teacher);
                    let dataloader_pos = dataloader_pos(produced_batches);
                    if sender.send(Ok(SfnnTeacherBatch { batch, source, dataloader_pos })).is_err() {
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
                        "SFNN/{input_label} teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
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

        let source = format!("{format:?} SFNN/{input_label} teacher batch {batch_index}: {}", config.teacher);
        let dataloader_pos = dataloader_pos(visited_batches);
        if let Err(err) = visitor(SfnnTeacherBatch { batch, source, dataloader_pos }) {
            visit_error = Some(TeacherBatchError::invalid_input(format!(
                "SFNN/{input_label} teacher batch callback failed at batch {batch_index}: {err}"
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
            "SFNN/{input_label} teacher did not yield {batch_count} complete batches starting at batch index {} of {} positions; use a smaller --batch-size, batch-index, or batch count",
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
            queue_depth: 1,
            lambda: 1.0,
            scale: 400.0,
            nnue_pytorch_wrm_loss: false,
            ft_factorize: false,
            score_drop_abs: Some(32_000),
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
            nnue_pytorch_wrm_loss: false,
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
