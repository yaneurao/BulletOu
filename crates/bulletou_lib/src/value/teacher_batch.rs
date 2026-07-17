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
    validate_config(config)?;

    let data_files_owned = expand_teacher(config.teacher).map_err(TeacherBatchError::invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(TeacherBatchError::invalid_input)?;
    let source = format!("{format:?} teacher batch {}: {}", config.batch_index, config.teacher);

    let batch = match format {
        DataFormat::Hcpe => {
            let loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                config.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(config.batch_size)
            .with_loader_threads(config.loader_threads)
            .with_single_epoch(true);
            materialise_halfkp_batch(loader, config)?
        }
        DataFormat::Hcpe3 => {
            let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true)
                .with_buffer_records(config.batch_size)
                .with_single_epoch(true);
            materialise_halfkp_batch(loader, config)?
        }
        DataFormat::Pack => {
            let loader =
                ShogiPackLoader::new_concat_multiple(&data_files_ref, config.buffer_mb, |_| true).with_single_epoch(true);
            materialise_halfkp_batch(loader, config)?
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
            materialise_halfkp_batch(loader, config)?
        }
    };

    Ok(HalfkpTeacherBatch { batch, source })
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

fn materialise_halfkp_batch<D>(
    loader: D,
    config: &HalfkpTeacherBatchConfig<'_>,
) -> Result<FastBatchHost, TeacherBatchError>
where
    D: DataLoader<PackedSfenValue>,
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
    let mut selected_batch = None;
    let mut seen_batches = 0usize;
    dataloader.load_and_map_batches(0, config.batch_size, |batch| {
        if seen_batches != config.batch_index {
            seen_batches += 1;
            return false;
        }

        let prepared = dataloader.prepare(batch, threads, 1.0 - config.lambda);
        selected_batch = Some(FastBatchHost::from(prepared));
        true
    });

    let batch = selected_batch.ok_or_else(|| {
        TeacherBatchError::invalid_input(format!(
            "teacher did not yield complete batch index {} of {} positions; use a smaller --batch-size or batch-index",
            config.batch_index, config.batch_size
        ))
    })?;
    batch.validate().map_err(|err| TeacherBatchError::invalid_input(err.to_string()))?;
    Ok(batch)
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
}
