use std::borrow::Cow;

use bullet_compiler::tensor::TValue;
use bullet_trainer::run::{
    dataloader::{DataLoader, DataLoadingError, PreparedBatchHost},
    schedule::TrainingSteps,
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    trainer::schedule::wdl::WdlScheduler,
    value::loader::{self, PreparedData},
};

pub struct ValueDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    D: loader::DataLoader<I::RequiredDataType>,
{
    pub dataloader: loader::DefaultDataLoader<I, O, D>,
    pub steps: TrainingSteps,
    pub threads: usize,
    pub wdl: W,
}

impl<I, O, D, W> DataLoader for ValueDataLoader<I, O, D, W>
where
    I: SparseInputType,
    I::RequiredDataType: loader::LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
    W: WdlScheduler,
    D: loader::DataLoader<I::RequiredDataType>,
{
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError> {
        let ValueDataLoader { dataloader, steps, threads, wdl } = self;
        let start_batch = steps.batches_per_superbatch * (steps.start_superbatch - 1);

        assert_eq!(batch_size, steps.batch_size);

        let mut batch_no = 0;
        let mut superbatch = steps.start_superbatch;

        dataloader.load_and_map_batches(start_batch, batch_size, |batch| {
            let blend = wdl.blend(batch_no, superbatch, steps.end_superbatch);
            let prepared_data = dataloader.prepare(batch, threads, blend);

            batch_no += 1;

            if batch_no % steps.batches_per_superbatch == 0 {
                batch_no = 0;
                superbatch += 1;
            }

            f(prepared_data.into())
        });

        Ok(())
    }
}

impl<I: SparseInputType, O> From<PreparedData<I, O>> for PreparedBatchHost {
    fn from(prepared_data: PreparedData<I, O>) -> Self {
        let mut inputs = Vec::with_capacity(5 + usize::from(prepared_data.hand_count.is_some()));
        inputs.push((Cow::Borrowed("stm"), TValue::I32(prepared_data.stm)));
        inputs.push((Cow::Borrowed("nstm"), TValue::I32(prepared_data.nstm)));
        inputs.push((Cow::Borrowed("buckets"), TValue::I32(prepared_data.buckets)));
        inputs.push((Cow::Borrowed("targets"), TValue::F32(prepared_data.targets)));
        inputs.push((Cow::Borrowed("entry_weights"), TValue::F32(prepared_data.weights)));

        if let Some(hc) = prepared_data.hand_count {
            inputs.push((Cow::Borrowed("hand_count"), TValue::F32(hc)));
        }

        PreparedBatchHost { batch_size: prepared_data.batch_size, inputs }
    }
}
