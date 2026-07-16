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

impl PreparedBatchHost {
    fn input(&self, id: &str) -> Option<&TValue> {
        self.inputs.iter().find_map(|(name, value)| (name.as_ref() == id).then_some(value))
    }

    pub fn copy_to_device_async<'a, G: Gpu>(
        &'a self,
        stream: &Arc<Stream<G>>,
        tensors: &TensorMap<G>,
    ) -> Result<SyncOnValue<G, &'a Self>, G::Error> {
        let mut sync = SyncOnDrop::with_capacity(stream.clone(), tensors.len());

        for (id, tensor) in tensors {
            let value = self.input(id).ok_or("Missing input!".into())?;

            if tensor.size() != value.size() {
                return Err(format!("Mismatched sizes: {} != {}", tensor.size(), value.size()).into());
            }

            if tensor.dtype() != value.dtype() {
                return Err(format!("Mismatched DType: {:?} != {:?}", tensor.dtype(), value.dtype()).into());
            }

            unsafe {
                let guard = tensor.clone().acquire(stream.clone())?;
                stream.memcpy_h2d(value.ptr(), guard.ptr(), guard.bytes())?;
                sync.attach(guard)?;
            }
        }

        Ok(SyncOnValue::new(sync, self))
    }

    pub fn to_device<G: Gpu>(self, device: &Arc<Device<G>>) -> Result<TensorMap<G>, G::Error> {
        self.inputs
            .iter()
            .map(|(id, value)| Buffer::from_host(device, value).map(|tensor| (id.to_string(), tensor)))
            .collect()
    }
}
