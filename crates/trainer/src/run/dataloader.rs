use std::{borrow::Cow, sync::Arc};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::{Buffer, SyncOnDrop, SyncOnValue},
    runtime::{Device, Gpu, Stream},
};

use crate::model::TensorMap;

#[derive(Debug)]
pub enum DataLoadingError {
    TooManyBatchesReceived,
    NoBatchesReceived,
    Message(String),
}

pub trait DataLoader: Send + Sync + 'static {
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(self, batch_size: usize, f: F) -> Result<(), DataLoadingError>;
}

pub struct PreparedBatchHost {
    pub batch_size: usize,
    pub inputs: Vec<(Cow<'static, str>, TValue)>,
}

pub struct PreparedBatchUploadPlan<G: Gpu> {
    entries: Vec<PreparedBatchUploadEntry<G>>,
}

struct PreparedBatchUploadEntry<G: Gpu> {
    name: String,
    input_index: usize,
    tensor: Arc<Buffer<G>>,
}

impl PreparedBatchHost {
    fn input(&self, id: &str) -> Option<&TValue> {
        self.inputs.iter().find_map(|(name, value)| (name.as_ref() == id).then_some(value))
    }

    fn planned_input(&self, name: &str, input_index: usize) -> Option<&TValue> {
        match self.inputs.get(input_index) {
            Some((actual_name, value)) if actual_name.as_ref() == name => Some(value),
            _ => self.input(name),
        }
    }

    pub fn upload_plan<G: Gpu>(&self, tensors: &TensorMap<G>) -> Result<PreparedBatchUploadPlan<G>, G::Error> {
        let mut entries = Vec::with_capacity(tensors.len());

        for (id, tensor) in tensors {
            let (input_index, value) = self
                .inputs
                .iter()
                .enumerate()
                .find_map(|(index, (name, value))| (name.as_ref() == id).then_some((index, value)))
                .ok_or("Missing input!".into())?;

            if tensor.size() != value.size() {
                return Err(format!("Mismatched sizes: {} != {}", tensor.size(), value.size()).into());
            }

            if tensor.dtype() != value.dtype() {
                return Err(format!("Mismatched DType: {:?} != {:?}", tensor.dtype(), value.dtype()).into());
            }

            entries.push(PreparedBatchUploadEntry {
                name: id.clone(),
                input_index,
                tensor: tensor.clone(),
            });
        }

        Ok(PreparedBatchUploadPlan { entries })
    }

    pub fn copy_to_device_async<'a, G: Gpu>(
        &'a self,
        stream: &Arc<Stream<G>>,
        tensors: &TensorMap<G>,
    ) -> Result<SyncOnValue<G, &'a Self>, G::Error> {
        let plan = self.upload_plan(tensors)?;
        self.copy_to_device_with_plan_async(stream, &plan)
    }

    pub fn copy_to_device_with_plan_async<'a, G: Gpu>(
        &'a self,
        stream: &Arc<Stream<G>>,
        plan: &PreparedBatchUploadPlan<G>,
    ) -> Result<SyncOnValue<G, &'a Self>, G::Error> {
        let mut sync = SyncOnDrop::with_capacity(stream.clone(), plan.entries.len());

        for entry in &plan.entries {
            let value = self
                .planned_input(&entry.name, entry.input_index)
                .ok_or("Missing input!".into())?;

            if entry.tensor.size() != value.size() {
                return Err(format!("Mismatched sizes: {} != {}", entry.tensor.size(), value.size()).into());
            }

            if entry.tensor.dtype() != value.dtype() {
                return Err(format!("Mismatched DType: {:?} != {:?}", entry.tensor.dtype(), value.dtype()).into());
            }

            unsafe {
                let guard = entry.tensor.acquire(stream.clone())?;
                stream.memcpy_h2d(value.ptr(), guard.ptr(), guard.bytes())?;
                sync.attach(guard)?;
            }
        }

        Ok(SyncOnValue::new(sync, self))
    }

    pub fn to_device<G: Gpu>(&self, device: &Arc<Device<G>>) -> Result<TensorMap<G>, G::Error> {
        self.inputs
            .iter()
            .map(|(id, value)| Buffer::from_host(device, value).map(|tensor| (id.to_string(), tensor)))
            .collect()
    }
}
