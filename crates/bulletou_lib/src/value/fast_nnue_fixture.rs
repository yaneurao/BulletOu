//! NNUE forward fixture writer for the isolated cuda-oxide workspace.
//!
//! The `cuda-oxide/` workspace intentionally does not depend on the root
//! BulletOu workspace. This tiny binary format is the bridge: root-side code can
//! export an already-built `FastBatchHost` plus `NnueForwardWeights`, and
//! `bulletou-cuda-train --nnue-forward-fixture` can consume the file.

use std::{
    fmt,
    io::{self, Write},
    path::Path,
};

use crate::value::{FastBatchHost, FastNnueError, NnueForwardWeights};

pub const NNUE_FORWARD_FIXTURE_MAGIC: &[u8; 8] = b"BOUNFWD1";

#[derive(Debug)]
pub enum NnueForwardFixtureError {
    Io(io::Error),
    Nnue(FastNnueError),
    BatchLayout(String),
}

impl fmt::Display for NnueForwardFixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "NNUE forward fixture I/O error: {err}"),
            Self::Nnue(err) => write!(f, "NNUE forward fixture weight error: {err}"),
            Self::BatchLayout(message) => write!(f, "NNUE forward fixture batch layout error: {message}"),
        }
    }
}

impl std::error::Error for NnueForwardFixtureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Nnue(err) => Some(err),
            Self::BatchLayout(_) => None,
        }
    }
}

impl From<io::Error> for NnueForwardFixtureError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<FastNnueError> for NnueForwardFixtureError {
    fn from(err: FastNnueError) -> Self {
        Self::Nnue(err)
    }
}

pub fn write_nnue_forward_fixture_file(
    path: impl AsRef<Path>,
    weights: NnueForwardWeights<'_>,
    batch: &FastBatchHost,
) -> Result<(), NnueForwardFixtureError> {
    let mut writer = io::BufWriter::new(std::fs::File::create(path)?);
    write_nnue_forward_fixture(&mut writer, weights, batch)?;
    writer.flush()?;
    Ok(())
}

pub fn write_nnue_forward_fixture(
    writer: &mut impl Write,
    weights: NnueForwardWeights<'_>,
    batch: &FastBatchHost,
) -> Result<(), NnueForwardFixtureError> {
    weights.validate()?;
    batch.validate().map_err(NnueForwardFixtureError::BatchLayout)?;

    writer.write_all(NNUE_FORWARD_FIXTURE_MAGIC)?;
    for value in [
        weights.shape.input_size,
        weights.shape.l1,
        weights.shape.l2,
        weights.shape.l3,
        batch.layout.batch_size,
        batch.layout.max_active,
    ] {
        write_u64(writer, value as u64)?;
    }

    write_i32_slice(writer, &batch.stm)?;
    write_i32_slice(writer, &batch.nstm)?;
    write_f32_slice(writer, weights.l0w)?;
    write_f32_slice(writer, weights.l0b)?;
    write_f32_slice(writer, weights.l1w)?;
    write_f32_slice(writer, weights.l1b)?;
    write_f32_slice(writer, weights.l2w)?;
    write_f32_slice(writer, weights.l2b)?;
    write_f32_slice(writer, weights.outw)?;
    write_f32_slice(writer, weights.outb)?;

    Ok(())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_i32_slice(writer: &mut impl Write, values: &[i32]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_f32_slice(writer: &mut impl Write, values: &[f32]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{FastBatchLayout, NnueForwardShape};

    #[test]
    fn writes_tiny_fixture_in_cuda_oxide_order() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let weights = tiny_weights(shape);
        let batch = tiny_batch();
        let mut bytes = Vec::new();

        write_nnue_forward_fixture(&mut bytes, weights, &batch).unwrap();

        assert_eq!(&bytes[..8], NNUE_FORWARD_FIXTURE_MAGIC);
        assert_eq!(bytes.len(), 204);
        assert_eq!(u64_at(&bytes, 8), 4);
        assert_eq!(u64_at(&bytes, 16), 2);
        assert_eq!(u64_at(&bytes, 24), 2);
        assert_eq!(u64_at(&bytes, 32), 1);
        assert_eq!(u64_at(&bytes, 40), 2);
        assert_eq!(u64_at(&bytes, 48), 3);
    }

    #[test]
    fn rejects_invalid_batch_layout() {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let weights = tiny_weights(shape);
        let mut batch = tiny_batch();
        batch.stm.pop();
        let mut bytes = Vec::new();

        let err = write_nnue_forward_fixture(&mut bytes, weights, &batch).unwrap_err();

        assert!(matches!(err, NnueForwardFixtureError::BatchLayout(_)));
    }

    fn u64_at(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    fn tiny_batch() -> FastBatchHost {
        FastBatchHost {
            layout: FastBatchLayout { batch_size: 2, max_active: 3, output_size: 1, hand_count_dim: 0 },
            stm: vec![0, 1, -1, 3, -1, -1],
            nstm: vec![2, -1, -1, 1, 2, -1],
            buckets: vec![0, 0],
            targets: vec![0.0, 0.0],
            weights: vec![1.0, 1.0],
            hand_count: None,
        }
    }

    fn tiny_weights(shape: NnueForwardShape) -> NnueForwardWeights<'static> {
        NnueForwardWeights {
            shape,
            l0w: &[
                0.2, 0.3, // feature 0
                0.4, -0.1, // feature 1
                -0.3, 0.5, // feature 2
                0.7, 0.9, // feature 3
            ],
            l0b: &[0.1, 0.2],
            l1w: &[
                0.5, -0.2, // combined 0
                0.1, 0.3, // combined 1
                -0.4, 0.2, // combined 2
                0.6, 0.1, // combined 3
            ],
            l1b: &[0.05, 0.1],
            l2w: &[
                0.7,  // hidden1 0
                -0.2, // hidden1 1
            ],
            l2b: &[0.2],
            outw: &[1.5],
            outb: &[0.05],
        }
    }
}
