//! SFNN forward fixture writer for the isolated cuda-oxide workspace.
//!
//! The `cuda-oxide/` workspace intentionally does not depend on the root
//! BulletOu workspace. This binary format bridges root-side `FastBatchHost`
//! plus `SfnnForwardWeights` into `bulletou-cuda-train --sfnn-forward-fixture`.

use std::{
    fmt,
    io::{self, Write},
    path::Path,
};

use crate::value::{FastBatchHost, FastSfnnError, SfnnForwardWeights};

pub const SFNN_FORWARD_FIXTURE_MAGIC: &[u8; 8] = b"BOUSFWD1";

#[derive(Debug)]
pub enum SfnnForwardFixtureError {
    Io(io::Error),
    Sfnn(FastSfnnError),
    BatchLayout(String),
}

impl fmt::Display for SfnnForwardFixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "SFNN forward fixture I/O error: {err}"),
            Self::Sfnn(err) => write!(f, "SFNN forward fixture weight error: {err}"),
            Self::BatchLayout(message) => write!(f, "SFNN forward fixture batch layout error: {message}"),
        }
    }
}

impl std::error::Error for SfnnForwardFixtureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Sfnn(err) => Some(err),
            Self::BatchLayout(_) => None,
        }
    }
}

impl From<io::Error> for SfnnForwardFixtureError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<FastSfnnError> for SfnnForwardFixtureError {
    fn from(err: FastSfnnError) -> Self {
        Self::Sfnn(err)
    }
}

pub fn write_sfnn_forward_fixture_file(
    path: impl AsRef<Path>,
    weights: SfnnForwardWeights<'_>,
    batch: &FastBatchHost,
) -> Result<(), SfnnForwardFixtureError> {
    let mut writer = io::BufWriter::new(std::fs::File::create(path)?);
    write_sfnn_forward_fixture(&mut writer, weights, batch)?;
    writer.flush()?;
    Ok(())
}

pub fn write_sfnn_forward_fixture(
    writer: &mut impl Write,
    weights: SfnnForwardWeights<'_>,
    batch: &FastBatchHost,
) -> Result<(), SfnnForwardFixtureError> {
    weights.validate()?;
    batch.validate().map_err(SfnnForwardFixtureError::BatchLayout)?;

    writer.write_all(SFNN_FORWARD_FIXTURE_MAGIC)?;
    for value in [
        weights.shape.input_size,
        weights.shape.ft_size,
        weights.shape.l1_hidden,
        weights.shape.l2_size,
        weights.shape.num_stacks,
        batch.layout.batch_size,
        batch.layout.max_active,
    ] {
        write_u64(writer, value as u64)?;
    }

    write_i32_slice(writer, &batch.stm)?;
    write_i32_slice(writer, &batch.nstm)?;
    write_i32_slice(writer, &batch.buckets)?;
    write_f32_slice(writer, weights.l0w)?;
    write_f32_slice(writer, weights.l0b)?;
    write_f32_slice(writer, weights.l1w)?;
    write_f32_slice(writer, weights.l1b)?;
    write_f32_slice(writer, weights.l2w)?;
    write_f32_slice(writer, weights.l2b)?;
    write_f32_slice(writer, weights.l3w)?;
    write_f32_slice(writer, weights.l3b)?;

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
    use crate::value::{FastBatchLayout, SfnnForwardShape};

    #[test]
    fn writes_tiny_fixture_in_cuda_oxide_order() {
        let shape = tiny_shape();
        let weights = tiny_weights(shape);
        let batch = tiny_batch();
        let mut bytes = Vec::new();

        write_sfnn_forward_fixture(&mut bytes, weights, &batch).unwrap();

        assert_eq!(&bytes[..8], SFNN_FORWARD_FIXTURE_MAGIC);
        assert_eq!(bytes.len(), 424);
        assert_eq!(u64_at(&bytes, 8), 4);
        assert_eq!(u64_at(&bytes, 16), 4);
        assert_eq!(u64_at(&bytes, 24), 2);
        assert_eq!(u64_at(&bytes, 32), 2);
        assert_eq!(u64_at(&bytes, 40), 2);
        assert_eq!(u64_at(&bytes, 48), 2);
        assert_eq!(u64_at(&bytes, 56), 3);
    }

    #[test]
    fn rejects_invalid_batch_layout() {
        let shape = tiny_shape();
        let weights = tiny_weights(shape);
        let mut batch = tiny_batch();
        batch.buckets.pop();
        let mut bytes = Vec::new();

        let err = write_sfnn_forward_fixture(&mut bytes, weights, &batch).unwrap_err();

        assert!(matches!(err, SfnnForwardFixtureError::BatchLayout(_)));
    }

    fn u64_at(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    fn tiny_shape() -> SfnnForwardShape {
        SfnnForwardShape { input_size: 4, ft_size: 4, l1_hidden: 2, l2_size: 2, num_stacks: 2 }
    }

    fn tiny_batch() -> FastBatchHost {
        FastBatchHost {
            layout: FastBatchLayout { batch_size: 2, max_active: 3, output_size: 1, hand_count_dim: 0 },
            stm: vec![0, 1, -1, 3, -1, -1],
            nstm: vec![2, -1, -1, 0, 2, -1],
            buckets: vec![0, 1],
            targets: vec![0.0, 0.0],
            weights: vec![1.0, 1.0],
            hand_count: None,
        }
    }

    fn tiny_weights(shape: SfnnForwardShape) -> SfnnForwardWeights<'static> {
        SfnnForwardWeights {
            shape,
            l0w: &[
                0.2, 0.1, -0.1, 0.0, // feature 0
                -0.1, 0.2, 0.1, 0.2, // feature 1
                0.0, -0.2, 0.2, 0.1, // feature 2
                0.3, 0.0, -0.3, 0.2, // feature 3
            ],
            l0b: &[0.1, 0.2, 0.3, 0.4],
            l1w: &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, // combined 0
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // combined 1
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // combined 2
                0.0, 0.0, 1.0, 0.0, 1.0, 0.0, // combined 3
            ],
            l1b: &[0.0; 6],
            l2w: &[
                1.0, 0.0, 0.0, 0.0, // l2 input 0
                0.0, 1.0, 0.0, 0.0, // l2 input 1
                1.0, 0.0, 1.0, 0.0, // l2 input 2
                0.0, 1.0, 0.0, 1.0, // l2 input 3
            ],
            l2b: &[0.0; 4],
            l3w: &[
                2.0, -0.5, // l2 output 0
                -1.0, 0.8, // l2 output 1
            ],
            l3b: &[0.1, -0.02],
        }
    }
}
