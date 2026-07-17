//! Teacher-to-`FastBatchHost` helpers shared by fixture exporters and future
//! fast backend trainers.

use std::{error::Error, fmt};

use crate::{
    game::inputs::ShogiHalfKP,
    shogi::PackedSfenValue,
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    value::{
        FastBatchHost, NoOutputBuckets,
        loader::{
            DataLoader, DefaultDataLoader, DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader,
            ShogiPackLoader,
        },
    },
};

#[derive(Debug, Clone)]
pub struct HalfkpTeacherBatchConfig<'a> {
    pub teacher: &'a str,
    pub batch_size: usize,
    pub batch_index: usize,
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
    /// Drop positions whose absolute teacher score is at least this value.
    pub score_drop_abs: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct HalfkpTeacherBatch {
    pub batch: FastBatchHost,
    pub source: String,
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
            let loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(config.batch_size)
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(true);
            visit_halfkp_batches(loader, format, config, batch_count, visitor)
        }
        DataFormat::Hcpe3 => {
            let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(true);
            visit_halfkp_batches(loader, format, config, batch_count, visitor)
        }
        DataFormat::Pack => {
            let loader =
                ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true).with_single_epoch(true);
            visit_halfkp_batches(loader, format, config, batch_count, visitor)
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
            visit_halfkp_batches(loader, format, config, batch_count, visitor)
        }
    }
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

fn visit_halfkp_batches<D, F, E>(
    loader: D,
    format: DataFormat,
    config: &HalfkpTeacherBatchConfig<'_>,
    batch_count: usize,
    mut visitor: F,
) -> Result<usize, TeacherBatchError>
where
    D: DataLoader<PackedSfenValue>,
    F: FnMut(HalfkpTeacherBatch) -> Result<(), E>,
    E: fmt::Display,
{
    let threads = config.threads.max(1);
    let dataloader = DefaultDataLoader::new(
        ShogiHalfKP,
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
    dataloader.load_and_map_batches(config.batch_index, config.batch_size, |batch| {
        let batch_index = config.batch_index + visited_batches;
        let prepared = dataloader.prepare(batch, threads, 1.0 - config.lambda);
        let batch = FastBatchHost::from(prepared);
        if let Err(err) = batch.validate() {
            visit_error = Some(TeacherBatchError::invalid_input(err.to_string()));
            return true;
        }

        let source = format!("{format:?} teacher batch {batch_index}: {}", config.teacher);
        if let Err(err) = visitor(HalfkpTeacherBatch { batch, source }) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> HalfkpTeacherBatchConfig<'static> {
        HalfkpTeacherBatchConfig {
            teacher: "missing.hcpe",
            batch_size: 2,
            batch_index: 0,
            buffer_mb: 1,
            loader_threads: 1,
            threads: 1,
            lambda: 1.0,
            scale: 400.0,
            nnue_pytorch_wrm_loss: false,
            score_drop_abs: Some(32_000),
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
}
