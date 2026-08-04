//! Scalar SFNN forward reference for the fixed-layout fast backend.
//!
//! This mirrors the `SFNN_halfka2_1024_7_64_k3k3` forward path built in
//! `examples/bulletou.rs`: sparse FT, CReLU, pairwise multiplication,
//! LayerStack-selected L1/L2/L3, and the optional SFNN PSQT shortcut.

use std::{collections::BTreeMap, fmt};

use bullet_compiler::tensor::TValue;
use bullet_gpu::runtime::Gpu;
use bullet_trainer::model::Model;

use crate::{
    game::inputs::{HALFKA2_DIMENSIONS, PIECE_INPUTS},
    value::{FastBatchHost, fast_batch::active_feature_indices},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardShape {
    pub input_size: usize,
    pub ft_size: usize,
    pub l1_hidden: usize,
    pub l1_skip: bool,
    pub l2_size: usize,
    pub num_stacks: usize,
    pub l1_group_count: usize,
}

pub const SFNN_HALFKA2_1024_7_64_K3K3: SfnnForwardShape = SfnnForwardShape {
    input_size: HALFKA2_DIMENSIONS,
    ft_size: 1024,
    l1_hidden: 7,
    l1_skip: true,
    l2_size: 64,
    num_stacks: 9,
    l1_group_count: 1,
};

pub const SFNN_HALFKA2_FT_FACTORIZED_INPUT_SIZE: usize = HALFKA2_DIMENSIONS + PIECE_INPUTS;

impl SfnnForwardShape {
    pub fn has_l1_skip(self) -> bool {
        self.l1_skip
    }

    pub fn l1_out(self) -> usize {
        self.l1_hidden + usize::from(self.l1_skip)
    }

    pub fn l2_in(self) -> usize {
        self.l1_hidden * 2
    }

    pub fn pairwise_size(self) -> usize {
        self.ft_size / 2
    }

    pub fn l1_group_count(self) -> usize {
        self.l1_group_count
    }

    pub fn has_grouped_l1(self) -> bool {
        self.l1_group_count > 1
    }

    pub fn l1_group_input(self) -> usize {
        self.ft_size / self.l1_group_count
    }

    pub fn l1_group_output(self) -> usize {
        self.l1_out() / self.l1_group_count
    }

    pub fn l1w_len(self) -> usize {
        if self.has_grouped_l1() {
            self.num_stacks
                .saturating_mul(self.l1_group_count)
                .saturating_mul(self.l1_group_output())
                .saturating_mul(self.l1_group_input())
        } else {
            self.ft_size.saturating_mul(self.num_stacks).saturating_mul(self.l1_out())
        }
    }

    pub fn validate(self) -> Result<(), FastSfnnError> {
        if self.input_size == 0 {
            return Err(FastSfnnError::Shape("input_size must be > 0".to_string()));
        }
        if self.ft_size == 0 {
            return Err(FastSfnnError::Shape("ft_size must be > 0".to_string()));
        }
        if self.ft_size % 2 != 0 {
            return Err(FastSfnnError::Shape(format!("ft_size must be even, got {}", self.ft_size)));
        }
        if self.l1_hidden == 0 {
            return Err(FastSfnnError::Shape("l1_hidden must be > 0".to_string()));
        }
        if self.l2_size == 0 {
            return Err(FastSfnnError::Shape("l2_size must be > 0".to_string()));
        }
        if self.num_stacks == 0 {
            return Err(FastSfnnError::Shape("num_stacks must be > 0".to_string()));
        }
        if self.l1_group_count == 0 {
            return Err(FastSfnnError::Shape("l1_group_count must be > 0".to_string()));
        }
        if self.has_grouped_l1() {
            if self.ft_size % self.l1_group_count != 0 {
                return Err(FastSfnnError::Shape(format!(
                    "grouped L1 requires ft_size to be divisible by group count: ft_size={}, group_count={}",
                    self.ft_size, self.l1_group_count
                )));
            }
            if self.l1_out() % self.l1_group_count != 0 {
                return Err(FastSfnnError::Shape(format!(
                    "grouped L1 requires l1_out to be divisible by group count: l1_out={}, group_count={}",
                    self.l1_out(),
                    self.l1_group_count
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfnnForwardWorkspaceLayout {
    pub shape: SfnnForwardShape,
    pub batch_size: usize,
}

impl SfnnForwardWorkspaceLayout {
    pub fn new(shape: SfnnForwardShape, batch_size: usize) -> Self {
        Self { shape, batch_size }
    }

    pub fn l0_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn combined_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.ft_size)
    }

    pub fn l1_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l1_out())
    }

    pub fn l2_input_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_in())
    }

    pub fn l2_len(self) -> usize {
        self.batch_size.saturating_mul(self.shape.l2_size)
    }

    pub fn output_len(self) -> usize {
        self.batch_size
    }

    pub fn total_activation_f32_len(self) -> usize {
        self.l0_len()
            .saturating_mul(2)
            .saturating_add(self.combined_len())
            .saturating_add(self.l1_len())
            .saturating_add(self.l2_input_len())
            .saturating_add(self.l2_len())
            .saturating_add(self.output_len())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SfnnForwardWeights<'a> {
    pub shape: SfnnForwardShape,
    pub l0w: &'a [f32],
    pub l0b: &'a [f32],
    pub l1w: &'a [f32],
    pub l1b: &'a [f32],
    pub l2w: &'a [f32],
    pub l2b: &'a [f32],
    pub l3w: &'a [f32],
    pub l3b: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SfnnForwardOwnedWeights {
    pub shape: SfnnForwardShape,
    pub l0w: Vec<f32>,
    pub l0b: Vec<f32>,
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub l3w: Vec<f32>,
    pub l3b: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SfnnForwardTrace {
    pub layout: SfnnForwardWorkspaceLayout,
    pub stm_l0: Vec<f32>,
    pub nstm_l0: Vec<f32>,
    pub combined: Vec<f32>,
    pub l1: Vec<f32>,
    pub l2_input: Vec<f32>,
    pub l2: Vec<f32>,
    pub outputs: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FastSfnnError {
    Shape(String),
    BatchLayout(String),
    InvalidBucket { sample: usize, bucket: i32, num_stacks: usize },
    MissingWeight { name: &'static str },
    WeightType { name: &'static str, expected: &'static str, actual: &'static str },
    WeightLength { name: &'static str, expected: usize, actual: usize },
}

impl fmt::Display for FastSfnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) => write!(f, "invalid SFNN shape: {message}"),
            Self::BatchLayout(message) => write!(f, "invalid fast batch layout: {message}"),
            Self::InvalidBucket { sample, bucket, num_stacks } => write!(
                f,
                "invalid LayerStack bucket for sample {sample}: got {bucket}, expected 0..{}",
                num_stacks.saturating_sub(1)
            ),
            Self::MissingWeight { name } => write!(f, "missing SFNN weight `{name}`"),
            Self::WeightType { name, expected, actual } => {
                write!(f, "weight type mismatch for {name}: expected {expected}, got {actual}")
            }
            Self::WeightLength { name, expected, actual } => {
                write!(f, "weight length mismatch for {name}: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for FastSfnnError {}

impl SfnnForwardOwnedWeights {
    pub fn from_weight_map(
        shape: SfnnForwardShape,
        weights: &BTreeMap<String, Vec<f32>>,
    ) -> Result<Self, FastSfnnError> {
        Self::from_weight_getter(shape, |name| weights.get(name).cloned().ok_or(FastSfnnError::MissingWeight { name }))
    }

    pub fn from_model<G: Gpu>(shape: SfnnForwardShape, model: &Model<G>) -> Result<Self, FastSfnnError> {
        Self::from_weight_getter(shape, |name| match model.get_weights(name) {
            Some(TValue::F32(values)) => Ok(values),
            Some(TValue::I32(_)) => Err(FastSfnnError::WeightType { name, expected: "f32", actual: "i32" }),
            None => Err(FastSfnnError::MissingWeight { name }),
        })
    }

    pub fn from_weight_getter<F>(shape: SfnnForwardShape, mut get: F) -> Result<Self, FastSfnnError>
    where
        F: FnMut(&'static str) -> Result<Vec<f32>, FastSfnnError>,
    {
        let weights = Self {
            shape,
            l0w: get("l0w")?,
            l0b: get("l0b")?,
            l1w: get("l1w")?,
            l1b: get("l1b")?,
            l2w: get("l2w")?,
            l2b: get("l2b")?,
            l3w: get("l3w")?,
            l3b: get("l3b")?,
        };

        weights.validate()?;
        Ok(weights)
    }

    pub fn as_borrowed(&self) -> SfnnForwardWeights<'_> {
        SfnnForwardWeights {
            shape: self.shape,
            l0w: &self.l0w,
            l0b: &self.l0b,
            l1w: &self.l1w,
            l1b: &self.l1b,
            l2w: &self.l2w,
            l2b: &self.l2b,
            l3w: &self.l3w,
            l3b: &self.l3b,
        }
    }

    pub fn validate(&self) -> Result<(), FastSfnnError> {
        self.as_borrowed().validate()
    }

    pub fn forward_batch(&self, batch: &FastBatchHost) -> Result<Vec<f32>, FastSfnnError> {
        self.as_borrowed().forward_batch(batch)
    }

    pub fn forward_batch_trace(&self, batch: &FastBatchHost) -> Result<SfnnForwardTrace, FastSfnnError> {
        self.as_borrowed().forward_batch_trace(batch)
    }
}

impl<'a> SfnnForwardWeights<'a> {
    pub fn validate(&self) -> Result<(), FastSfnnError> {
        let shape = self.shape;
        shape.validate()?;
        expect_len("l0w", shape.input_size * shape.ft_size, self.l0w.len())?;
        expect_len("l0b", shape.ft_size, self.l0b.len())?;
        expect_len("l1w", shape.l1w_len(), self.l1w.len())?;
        expect_len("l1b", shape.num_stacks * shape.l1_out(), self.l1b.len())?;
        expect_len("l2w", shape.l2_in() * shape.num_stacks * shape.l2_size, self.l2w.len())?;
        expect_len("l2b", shape.num_stacks * shape.l2_size, self.l2b.len())?;
        expect_len("l3w", shape.l2_size * shape.num_stacks, self.l3w.len())?;
        expect_len("l3b", shape.num_stacks, self.l3b.len())?;
        Ok(())
    }

    pub fn forward_batch(&self, batch: &FastBatchHost) -> Result<Vec<f32>, FastSfnnError> {
        Ok(self.forward_batch_trace(batch)?.outputs)
    }

    pub fn forward_batch_trace(&self, batch: &FastBatchHost) -> Result<SfnnForwardTrace, FastSfnnError> {
        self.validate()?;
        batch.validate().map_err(FastSfnnError::BatchLayout)?;

        let shape = self.shape;
        let batch_size = batch.layout.batch_size;
        let layout = SfnnForwardWorkspaceLayout::new(shape, batch_size);
        let mut trace = SfnnForwardTrace {
            layout,
            stm_l0: vec![0.0; layout.l0_len()],
            nstm_l0: vec![0.0; layout.l0_len()],
            combined: vec![0.0; layout.combined_len()],
            l1: vec![0.0; layout.l1_len()],
            l2_input: vec![0.0; layout.l2_input_len()],
            l2: vec![0.0; layout.l2_len()],
            outputs: vec![0.0; layout.output_len()],
        };

        for sample in 0..batch_size {
            let stack = bucket_for_sample(batch, sample, shape.num_stacks)?;

            let l0_start = sample * shape.ft_size;
            let l0_end = l0_start + shape.ft_size;
            let pairwise = shape.pairwise_size();
            let combined_start = sample * shape.ft_size;
            let combined_mid = combined_start + pairwise;
            let combined_end = combined_start + shape.ft_size;
            let l1_start = sample * shape.l1_out();
            let l1_end = l1_start + shape.l1_out();
            let l2_input_start = sample * shape.l2_in();
            let l2_input_end = l2_input_start + shape.l2_in();
            let l2_start = sample * shape.l2_size;
            let l2_end = l2_start + shape.l2_size;

            let stm_l0 = &mut trace.stm_l0[l0_start..l0_end];
            let nstm_l0 = &mut trace.nstm_l0[l0_start..l0_end];
            affine_sparse_padded(
                self.l0w,
                self.l0b,
                shape.ft_size,
                shape.input_size,
                batch.stm_sample(sample).expect("validated batch sample"),
                stm_l0,
            );
            affine_sparse_padded(
                self.l0w,
                self.l0b,
                shape.ft_size,
                shape.input_size,
                batch.nstm_sample(sample).expect("validated batch sample"),
                nstm_l0,
            );

            crelu_in_place(stm_l0);
            crelu_in_place(nstm_l0);
            pairwise_mul_scaled(stm_l0, &mut trace.combined[combined_start..combined_mid]);
            pairwise_mul_scaled(nstm_l0, &mut trace.combined[combined_mid..combined_end]);

            let combined = &trace.combined[combined_start..combined_end];
            let l1 = &mut trace.l1[l1_start..l1_end];
            affine_sfnn_l1(self.l1w, self.l1b, combined, shape, stack, l1);

            let l1_skip = if shape.has_l1_skip() { l1[shape.l1_hidden] } else { 0.0 };
            fill_l2_input(l1, shape.l1_hidden, &mut trace.l2_input[l2_input_start..l2_input_end]);

            let l2_input = &trace.l2_input[l2_input_start..l2_input_end];
            let l2 = &mut trace.l2[l2_start..l2_end];
            affine_stacked(self.l2w, self.l2b, l2_input, shape.l2_size, shape.num_stacks, stack, l2);
            crelu_in_place(l2);

            trace.outputs[sample] = affine_stacked_scalar(self.l3w, self.l3b, l2, shape.num_stacks, stack) + l1_skip;
        }

        Ok(trace)
    }
}

fn affine_sfnn_l1(
    weights: &[f32],
    bias: &[f32],
    input: &[f32],
    shape: SfnnForwardShape,
    stack: usize,
    out: &mut [f32],
) {
    if !shape.has_grouped_l1() {
        affine_stacked(weights, bias, input, shape.l1_out(), shape.num_stacks, stack, out);
        return;
    }

    let rows = shape.l1_out();
    let bias_base = stack * rows;
    out.copy_from_slice(&bias[bias_base..bias_base + rows]);
    let group_count = shape.l1_group_count();
    let group_input = shape.l1_group_input();
    let group_output = shape.l1_group_output();
    let stack_stride = group_count * group_output * group_input;
    let stack_base = stack * stack_stride;
    for group in 0..group_count {
        let input_base = group * group_input;
        let group_weight_base = stack_base + group * group_output * group_input;
        for local_out in 0..group_output {
            let out_col = group * group_output + local_out;
            let weight_base = group_weight_base + local_out * group_input;
            let mut sum = out[out_col];
            for local_in in 0..group_input {
                let x = input[input_base + local_in];
                if x != 0.0 {
                    sum += weights[weight_base + local_in] * x;
                }
            }
            out[out_col] = sum;
        }
    }
}

fn bucket_for_sample(batch: &FastBatchHost, sample: usize, num_stacks: usize) -> Result<usize, FastSfnnError> {
    let bucket = batch.buckets[sample];
    if bucket < 0 || bucket as usize >= num_stacks {
        Err(FastSfnnError::InvalidBucket { sample, bucket, num_stacks })
    } else {
        Ok(bucket as usize)
    }
}

fn expect_len(name: &'static str, expected: usize, actual: usize) -> Result<(), FastSfnnError> {
    if expected == actual {
        Ok(())
    } else {
        Err(FastSfnnError::WeightLength { name, expected, actual })
    }
}

fn affine_sparse_padded(weights: &[f32], bias: &[f32], rows: usize, cols: usize, active: &[i32], out: &mut [f32]) {
    out.copy_from_slice(&bias[..rows]);
    for feature in active_feature_indices(active, cols) {
        let base = feature * rows;
        for row in 0..rows {
            out[row] += weights[base + row];
        }
        if let Some(virtual_feature) = halfka2_ft_factorized_virtual_feature(feature, cols) {
            let base = virtual_feature * rows;
            for row in 0..rows {
                out[row] += weights[base + row];
            }
        }
    }
}

fn halfka2_ft_factorized_virtual_feature(feature: usize, cols: usize) -> Option<usize> {
    if cols == SFNN_HALFKA2_FT_FACTORIZED_INPUT_SIZE && feature < HALFKA2_DIMENSIONS {
        Some(HALFKA2_DIMENSIONS + feature % PIECE_INPUTS)
    } else {
        None
    }
}

fn affine_stacked(
    weights: &[f32],
    bias: &[f32],
    input: &[f32],
    rows: usize,
    _num_stacks: usize,
    stack: usize,
    out: &mut [f32],
) {
    let bias_base = stack * rows;
    out.copy_from_slice(&bias[bias_base..bias_base + rows]);
    let stack_base = stack * rows * input.len();
    for (input_idx, &x) in input.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for row in 0..rows {
            out[row] += weights[stack_base + row * input.len() + input_idx] * x;
        }
    }
}

fn affine_stacked_scalar(weights: &[f32], bias: &[f32], input: &[f32], _num_stacks: usize, stack: usize) -> f32 {
    let mut out = bias[stack];
    let stack_base = stack * input.len();
    for (input_idx, &x) in input.iter().enumerate() {
        if x != 0.0 {
            out += weights[stack_base + input_idx] * x;
        }
    }
    out
}

fn pairwise_mul_scaled(input: &[f32], out: &mut [f32]) {
    debug_assert_eq!(input.len() / 2, out.len());
    const SCALE: f32 = 127.0 / 128.0;
    let half = input.len() / 2;
    for idx in 0..half {
        out[idx] = input[idx] * input[half + idx] * SCALE;
    }
}

fn fill_l2_input(l1: &[f32], l1_hidden: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), l1_hidden * 2);
    const SCALE: f32 = 127.0 / 128.0;
    for row in 0..l1_hidden {
        out[row] = (l1[row].abs() * l1[row].abs() * SCALE).clamp(0.0, 1.0);
        out[l1_hidden + row] = l1[row].clamp(0.0, 1.0);
    }
}

fn crelu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{FastBatchHost, FastBatchLayout};
    use std::collections::BTreeMap;

    #[test]
    fn validates_weight_lengths() {
        let shape = tiny_shape();
        let weights = tiny_weights(shape);

        weights.validate().unwrap();
    }

    #[test]
    fn workspace_layout_counts_forward_activations() {
        let shape = SfnnForwardShape {
            input_size: 4,
            ft_size: 6,
            l1_hidden: 2,
            l1_skip: true,
            l2_size: 3,
            num_stacks: 2,
            l1_group_count: 1,
        };
        let layout = SfnnForwardWorkspaceLayout::new(shape, 5);

        assert_eq!(layout.l0_len(), 30);
        assert_eq!(layout.combined_len(), 30);
        assert_eq!(layout.l1_len(), 15);
        assert_eq!(layout.l2_input_len(), 20);
        assert_eq!(layout.l2_len(), 15);
        assert_eq!(layout.output_len(), 5);
        assert_eq!(layout.total_activation_f32_len(), 145);
    }

    #[test]
    fn scalar_forward_uses_layerstack_bucket_and_psqt_skip() {
        let shape = tiny_shape();
        let weights = tiny_weights(shape);
        let batch = tiny_batch();

        let outputs = weights.forward_batch(&batch).unwrap();

        assert_close_slice("outputs", &outputs, &[0.06307903, 0.04701126]);
    }

    #[test]
    fn scalar_trace_exposes_intermediate_activations() {
        let shape = tiny_shape();
        let weights = tiny_weights(shape);
        let batch = tiny_batch();

        let trace = weights.forward_batch_trace(&batch).unwrap();

        assert_eq!(trace.layout, SfnnForwardWorkspaceLayout::new(shape, 2));
        assert_close_slice("stm_l0", &trace.stm_l0, &[0.2, 0.5, 0.3, 0.6, 0.4, 0.2, 0.0, 0.6]);
        assert_close_slice("nstm_l0", &trace.nstm_l0, &[0.1, 0.0, 0.5, 0.5, 0.3, 0.1, 0.4, 0.5]);
        assert_close_slice(
            "combined",
            &trace.combined,
            &[0.05953125, 0.29765625, 0.049609375, 0.0, 0.0, 0.1190625, 0.1190625, 0.049609375],
        );
        assert_close_slice("l1", &trace.l1, &[0.05953125, 0.29765625, 0.0, 0.049609375, 0.0, 0.1190625]);
        assert_close_slice(
            "l2_input",
            &trace.l2_input,
            &[0.0035162825, 0.08790706, 0.05953125, 0.29765625, 0.0024418628, 0.0, 0.049609375, 0.0],
        );
        assert_close_slice("l2", &trace.l2, &[0.0035162825, 0.08790706, 0.05205124, 0.0]);
        assert_close_slice("outputs", &trace.outputs, &[0.06307903, 0.04701126]);
    }

    #[test]
    fn scalar_forward_rejects_out_of_range_bucket() {
        let shape = tiny_shape();
        let weights = tiny_weights(shape);
        let mut batch = tiny_batch();
        batch.buckets[1] = 2;

        let err = weights.forward_batch(&batch).unwrap_err();

        assert_eq!(err, FastSfnnError::InvalidBucket { sample: 1, bucket: 2, num_stacks: 2 });
    }

    #[test]
    fn owned_weights_delegate_to_borrowed_forward() {
        let shape = tiny_shape();
        let borrowed = tiny_weights(shape);
        let owned = SfnnForwardOwnedWeights {
            shape,
            l0w: borrowed.l0w.to_vec(),
            l0b: borrowed.l0b.to_vec(),
            l1w: borrowed.l1w.to_vec(),
            l1b: borrowed.l1b.to_vec(),
            l2w: borrowed.l2w.to_vec(),
            l2b: borrowed.l2b.to_vec(),
            l3w: borrowed.l3w.to_vec(),
            l3b: borrowed.l3b.to_vec(),
        };
        let batch = tiny_batch();

        let borrowed_outputs = borrowed.forward_batch(&batch).unwrap();
        let owned_outputs = owned.forward_batch(&batch).unwrap();

        assert_eq!(borrowed_outputs, owned_outputs);
    }

    #[test]
    fn owned_weights_can_be_loaded_from_weight_map() {
        let shape = tiny_shape();
        let borrowed = tiny_weights(shape);
        let mut map = BTreeMap::new();
        map.insert("l0w".to_string(), borrowed.l0w.to_vec());
        map.insert("l0b".to_string(), borrowed.l0b.to_vec());
        map.insert("l1w".to_string(), borrowed.l1w.to_vec());
        map.insert("l1b".to_string(), borrowed.l1b.to_vec());
        map.insert("l2w".to_string(), borrowed.l2w.to_vec());
        map.insert("l2b".to_string(), borrowed.l2b.to_vec());
        map.insert("l3w".to_string(), borrowed.l3w.to_vec());
        map.insert("l3b".to_string(), borrowed.l3b.to_vec());

        let owned = SfnnForwardOwnedWeights::from_weight_map(shape, &map).unwrap();

        assert_eq!(owned.as_borrowed().l3w, borrowed.l3w);
    }

    #[test]
    fn owned_weights_report_missing_weight() {
        let shape = tiny_shape();
        let map = BTreeMap::new();

        let err = SfnnForwardOwnedWeights::from_weight_map(shape, &map).unwrap_err();

        assert_eq!(err, FastSfnnError::MissingWeight { name: "l0w" });
    }

    #[test]
    fn shape_validation_requires_even_ft_size() {
        let shape = SfnnForwardShape {
            input_size: 4,
            ft_size: 3,
            l1_hidden: 2,
            l1_skip: true,
            l2_size: 2,
            num_stacks: 2,
            l1_group_count: 1,
        };

        let err = shape.validate().unwrap_err();

        assert!(matches!(err, FastSfnnError::Shape(message) if message.contains("ft_size")));
    }

    fn tiny_shape() -> SfnnForwardShape {
        SfnnForwardShape {
            input_size: 4,
            ft_size: 4,
            l1_hidden: 2,
            l1_skip: true,
            l2_size: 2,
            num_stacks: 2,
            l1_group_count: 1,
        }
    }

    fn tiny_batch() -> FastBatchHost {
        let layout = FastBatchLayout { batch_size: 2, max_active: 3, output_size: 1, hand_count_dim: 0 };
        FastBatchHost {
            layout,
            stm: vec![0, 1, -1, 3, -1, -1],
            nstm: vec![2, -1, -1, 0, 2, -1],
            buckets: vec![0, 1],
            targets: vec![0.0; layout.target_len()],
            weights: vec![1.0; layout.batch_size],
            hand_count: None,
        }
    }

    fn tiny_weights(shape: SfnnForwardShape) -> SfnnForwardWeights<'static> {
        assert_eq!(shape, tiny_shape());
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

    fn assert_close_slice(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() < 1.0e-6, "{name}[{idx}] mismatch: expected {expected}, got {actual}");
        }
    }
}
