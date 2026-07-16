use std::path::PathBuf;

use bulletou_lib::value::{
    FastBatchHost, FastBatchLayout, NNUE_HALFKP_256X2_32_32, NnueForwardOwnedWeights, NnueForwardShape,
    write_nnue_forward_fixture_file,
    yaneuraou_kppt::{extract_component_section, parse_model_weights_bin},
};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "export_nnue_forward_fixture")]
#[command(about = "Export a root BulletOu NNUE forward fixture for the nested cuda-oxide smoke runner")]
struct Args {
    /// Output fixture path. The file is a little-endian `BOUNFWD1` buffer
    /// accepted by `bulletou-cuda-train --nnue-forward-fixture`.
    #[arg(long)]
    out: PathBuf,

    /// Fixture size/shape. `tiny` is cheap; `halfkp` matches
    /// NNUE_HALFKP_256x2_32_32 and writes roughly 123 MiB.
    #[arg(long, value_enum, default_value = "tiny")]
    case: FixtureCase,

    /// Optional BulletOu `optimiser_state/weights.bin` or bundled `state.bin`.
    /// Only valid with `--case halfkp`. Bundled state records are read from
    /// `nnue/weights/*`.
    #[arg(long)]
    weights_bin: Option<PathBuf>,

    /// Override synthetic batch size.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Override padded sparse feature slots per sample.
    #[arg(long)]
    max_active: Option<usize>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureCase {
    Tiny,
    Halfkp,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.weights_bin.is_some() && args.case != FixtureCase::Halfkp {
        return Err("--weights-bin is only supported with --case halfkp".into());
    }

    let fixture = match args.case {
        FixtureCase::Tiny => Fixture::tiny(args.batch_size, args.max_active)?,
        FixtureCase::Halfkp => Fixture::halfkp(args.batch_size, args.max_active, args.weights_bin.as_ref())?,
    };

    let outputs = fixture.weights.forward_batch(&fixture.batch)?;
    write_nnue_forward_fixture_file(&args.out, fixture.weights.as_borrowed(), &fixture.batch)?;

    println!("exported NNUE forward fixture");
    println!("  out        : {}", args.out.display());
    println!("  case       : {}", fixture.label);
    println!("  weights    : {}", fixture.weights_source);
    println!(
        "  shape      : input={} l1={} l2={} l3={}",
        fixture.weights.shape.input_size, fixture.weights.shape.l1, fixture.weights.shape.l2, fixture.weights.shape.l3
    );
    println!(
        "  batch      : {} samples, max_active={}",
        fixture.batch.layout.batch_size, fixture.batch.layout.max_active
    );
    println!("  cpu output : {:?}", outputs);

    Ok(())
}

struct Fixture {
    label: &'static str,
    weights_source: String,
    weights: NnueForwardOwnedWeights,
    batch: FastBatchHost,
}

impl Fixture {
    fn tiny(batch_size: Option<usize>, max_active: Option<usize>) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let batch_size = batch_size.unwrap_or(2);
        let max_active = max_active.unwrap_or(3);
        require_nonzero("batch-size", batch_size)?;
        require_nonzero("max-active", max_active)?;

        Ok(Self {
            label: "tiny",
            weights_source: "deterministic tiny".to_string(),
            weights: tiny_weights(shape),
            batch: if batch_size == 2 && max_active == 3 {
                tiny_batch()
            } else {
                synthetic_batch(batch_size, max_active, shape.input_size)
            },
        })
    }

    fn halfkp(
        batch_size: Option<usize>,
        max_active: Option<usize>,
        weights_bin: Option<&PathBuf>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = NNUE_HALFKP_256X2_32_32;
        let batch_size = batch_size.unwrap_or(2);
        let max_active = max_active.unwrap_or(38);
        require_nonzero("batch-size", batch_size)?;
        require_nonzero("max-active", max_active)?;

        let (weights, weights_source) = match weights_bin {
            Some(path) => (load_halfkp_weights(path)?, path.display().to_string()),
            None => (synthetic_halfkp_weights(shape), "deterministic halfkp".to_string()),
        };

        Ok(Self {
            label: "halfkp-256x2-32-32",
            weights_source,
            weights,
            batch: synthetic_batch(batch_size, max_active, shape.input_size),
        })
    }
}

fn require_nonzero(name: &'static str, value: usize) -> Result<(), Box<dyn std::error::Error>> {
    if value == 0 { Err(format!("--{name} must be greater than zero").into()) } else { Ok(()) }
}

fn load_halfkp_weights(path: &PathBuf) -> Result<NnueForwardOwnedWeights, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let records = parse_model_weights_bin(&bytes)?;
    let weights = if records.contains_key("l0w") {
        records
    } else {
        extract_component_section(&records, "nnue", "weights")
    };

    Ok(NnueForwardOwnedWeights::from_weight_map(NNUE_HALFKP_256X2_32_32, &weights)?)
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

fn synthetic_batch(batch_size: usize, max_active: usize, input_size: usize) -> FastBatchHost {
    let (stm, nstm) = deterministic_sparse_batch(batch_size, max_active, input_size);
    FastBatchHost {
        layout: FastBatchLayout { batch_size, max_active, output_size: 1, hand_count_dim: 0 },
        stm,
        nstm,
        buckets: vec![0; batch_size],
        targets: vec![0.0; batch_size],
        weights: vec![1.0; batch_size],
        hand_count: None,
    }
}

fn tiny_weights(shape: NnueForwardShape) -> NnueForwardOwnedWeights {
    NnueForwardOwnedWeights {
        shape,
        l0w: vec![
            0.2, 0.3, // feature 0
            0.4, -0.1, // feature 1
            -0.3, 0.5, // feature 2
            0.7, 0.9, // feature 3
        ],
        l0b: vec![0.1, 0.2],
        l1w: vec![
            0.5, -0.2, // combined 0
            0.1, 0.3, // combined 1
            -0.4, 0.2, // combined 2
            0.6, 0.1, // combined 3
        ],
        l1b: vec![0.05, 0.1],
        l2w: vec![
            0.7,  // hidden1 0
            -0.2, // hidden1 1
        ],
        l2b: vec![0.2],
        outw: vec![1.5],
        outb: vec![0.05],
    }
}

fn synthetic_halfkp_weights(shape: NnueForwardShape) -> NnueForwardOwnedWeights {
    NnueForwardOwnedWeights {
        shape,
        l0w: deterministic_f32_vec(l0w_len(shape), 0x4B1D_5EED, 0.006, 0.0),
        l0b: deterministic_f32_vec(l0b_len(shape), 0x10B1_A5ED, 0.025, 0.12),
        l1w: deterministic_f32_vec(l1w_len(shape), 0xC1A5_51C1, 0.0015, 0.0),
        l1b: deterministic_f32_vec(l1b_len(shape), 0xB1A5_0010, 0.006, 0.03),
        l2w: deterministic_f32_vec(l2w_len(shape), 0xD2A5_E002, 0.003, 0.0),
        l2b: deterministic_f32_vec(l2b_len(shape), 0xB2A5_0020, 0.004, 0.02),
        outw: deterministic_f32_vec(outw_len(shape), 0x0A17_0003, 0.02, 0.0),
        outb: deterministic_f32_vec(outb_len(), 0x0B17_0004, 0.002, 0.01),
    }
}

fn l0w_len(shape: NnueForwardShape) -> usize {
    shape.input_size * shape.l1
}

fn l0b_len(shape: NnueForwardShape) -> usize {
    shape.l1
}

fn l1w_len(shape: NnueForwardShape) -> usize {
    shape.l1 * 2 * shape.l2
}

fn l1b_len(shape: NnueForwardShape) -> usize {
    shape.l2
}

fn l2w_len(shape: NnueForwardShape) -> usize {
    shape.l2 * shape.l3
}

fn l2b_len(shape: NnueForwardShape) -> usize {
    shape.l3
}

fn outw_len(shape: NnueForwardShape) -> usize {
    shape.l3
}

fn outb_len() -> usize {
    1
}

fn deterministic_sparse_batch(batch_size: usize, max_active: usize, input_size: usize) -> (Vec<i32>, Vec<i32>) {
    let mut stm = Vec::with_capacity(batch_size * max_active);
    let mut nstm = Vec::with_capacity(batch_size * max_active);
    for sample in 0..batch_size {
        let active = max_active.saturating_sub(sample % 5);
        let nstm_active = max_active.saturating_sub((sample + 2) % 7);
        for slot in 0..max_active {
            stm.push(if slot < active {
                deterministic_feature_index(sample, slot, input_size, 0x1357_2468) as i32
            } else {
                -1
            });
            nstm.push(if slot < nstm_active {
                deterministic_feature_index(sample, slot, input_size, 0x2468_1357) as i32
            } else {
                -1
            });
        }
    }
    (stm, nstm)
}

fn deterministic_feature_index(sample: usize, slot: usize, input_size: usize, seed: u64) -> usize {
    let mixed = mix_u64(seed ^ ((sample as u64) << 32) ^ slot as u64);
    (mixed as usize) % input_size
}

fn deterministic_f32_vec(len: usize, seed: u64, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let mixed = mix_u64(seed ^ idx as u64);
            let centered = (mixed % 2001) as i32 - 1000;
            bias + centered as f32 * (scale / 1000.0)
        })
        .collect()
}

fn mix_u64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
