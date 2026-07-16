use std::path::PathBuf;

use bulletou_lib::{
    game::{inputs::ShogiHalfKa2, outputs::ShogiLayerStackBucket9},
    shogi::PackedSfenValue,
    teacher_path::{DataFormat, expand_teacher, infer_data_format},
    value::{
        FastBatchHost, FastBatchLayout, SFNN_HALFKA2_1024_7_64_K3K3, SfnnForwardOwnedWeights, SfnnForwardShape,
        loader::{
            DataLoader, DefaultDataLoader, DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader,
        },
        write_sfnn_forward_fixture_file,
        yaneuraou_kppt::{extract_component_section, parse_model_weights_bin},
    },
};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "export_sfnn_forward_fixture")]
#[command(about = "Export a root BulletOu SFNN forward fixture for the nested cuda-oxide smoke runner")]
struct Args {
    /// Output fixture path. The file is a little-endian `BOUSFWD1` buffer
    /// accepted by `bulletou-cuda-train --sfnn-forward-fixture`.
    #[arg(long)]
    out: PathBuf,

    /// Fixture size/shape. `tiny` is cheap; `halfka2` matches
    /// SFNN_halfka2_1024_7_64_k3k3 and writes roughly 516 MiB.
    #[arg(long, value_enum, default_value = "tiny")]
    case: FixtureCase,

    /// Optional BulletOu `optimiser_state/weights.bin` or bundled `state.bin`.
    /// Only valid with `--case halfka2`. Bundled state records are read from
    /// `nnue/weights/*`.
    #[arg(long)]
    weights_bin: Option<PathBuf>,

    /// Optional teacher data path (file, directory, or comma-separated list).
    /// When present, `--case halfka2` exports the first real loader batch using
    /// the same ShogiHalfKa2 prepare path as the trainer.
    #[arg(long)]
    teacher: Option<String>,

    /// Override synthetic batch size.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Override padded sparse feature slots per sample.
    #[arg(long)]
    max_active: Option<usize>,

    /// CPU worker threads used while materialising a teacher batch.
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Loader read buffer size in MiB for teacher formats that use a read buffer.
    #[arg(long, default_value = "64")]
    buffer_mb: usize,

    /// HCPE decode threads. 0 means loader default/auto.
    #[arg(long, default_value = "0")]
    loader_threads: usize,

    /// Lambda on teacher eval score when target values are prepared.
    #[arg(long, default_value = "1.0")]
    lambda: f32,

    /// Eval-to-score sigmoid scale used while preparing teacher targets.
    #[arg(long, default_value = "400.0")]
    scale: f32,

    /// Use nnue-pytorch WRM target conversion while preparing teacher targets.
    #[arg(long)]
    nnue_pytorch_wrm_loss: bool,

    /// Drop positions whose |score| >= this by setting entry weight to zero.
    /// Set to 0 to disable.
    #[arg(long, default_value = "32000")]
    score_drop_abs: u16,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureCase {
    Tiny,
    Halfka2,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.weights_bin.is_some() && args.case != FixtureCase::Halfka2 {
        return Err(invalid_input("--weights-bin is only supported with --case halfka2"));
    }
    if args.teacher.is_some() && args.case != FixtureCase::Halfka2 {
        return Err(invalid_input("--teacher is only supported with --case halfka2"));
    }
    if args.teacher.is_some() && args.max_active.is_some() {
        return Err(invalid_input(
            "--max-active cannot be used with --teacher; it comes from ShogiHalfKa2::max_active()",
        ));
    }
    if !(0.0..=1.0).contains(&args.lambda) {
        return Err(invalid_input("--lambda must be in [0, 1]"));
    }
    if !(args.scale.is_finite() && args.scale > 0.0) {
        return Err(invalid_input("--scale must be finite and > 0"));
    }

    let fixture = match args.case {
        FixtureCase::Tiny => Fixture::tiny(args.batch_size, args.max_active)?,
        FixtureCase::Halfka2 => Fixture::halfka2(&args)?,
    };

    let outputs = fixture.weights.forward_batch(&fixture.batch)?;
    write_sfnn_forward_fixture_file(&args.out, fixture.weights.as_borrowed(), &fixture.batch)?;

    println!("exported SFNN forward fixture");
    println!("  out        : {}", args.out.display());
    println!("  case       : {}", fixture.label);
    println!("  weights    : {}", fixture.weights_source);
    println!("  batch src  : {}", fixture.batch_source);
    println!(
        "  shape      : input={} ft={} l1_hidden={} l2={} stacks={}",
        fixture.weights.shape.input_size,
        fixture.weights.shape.ft_size,
        fixture.weights.shape.l1_hidden,
        fixture.weights.shape.l2_size,
        fixture.weights.shape.num_stacks
    );
    println!(
        "  batch      : {} samples, max_active={}, buckets={:?}",
        fixture.batch.layout.batch_size, fixture.batch.layout.max_active, fixture.batch.buckets
    );
    println!("  cpu output : {:?}", outputs);

    Ok(())
}

struct Fixture {
    label: &'static str,
    weights_source: String,
    batch_source: String,
    weights: SfnnForwardOwnedWeights,
    batch: FastBatchHost,
}

impl Fixture {
    fn tiny(batch_size: Option<usize>, max_active: Option<usize>) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = SfnnForwardShape { input_size: 4, ft_size: 4, l1_hidden: 2, l2_size: 2, num_stacks: 2 };
        let batch_size = batch_size.unwrap_or(2);
        let max_active = max_active.unwrap_or(3);
        require_nonzero("batch-size", batch_size)?;
        require_nonzero("max-active", max_active)?;

        Ok(Self {
            label: "tiny",
            weights_source: "deterministic tiny".to_string(),
            batch_source: "deterministic tiny".to_string(),
            weights: tiny_weights(shape),
            batch: if batch_size == 2 && max_active == 3 {
                tiny_batch()
            } else {
                synthetic_batch(batch_size, max_active, shape.input_size, shape.num_stacks)
            },
        })
    }

    fn halfka2(args: &Args) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = SFNN_HALFKA2_1024_7_64_K3K3;
        let batch_size = args.batch_size.unwrap_or(2);
        let max_active = args.max_active.unwrap_or(40);
        require_nonzero("batch-size", batch_size)?;
        require_nonzero("max-active", max_active)?;

        let (weights, weights_source) = match args.weights_bin.as_ref() {
            Some(path) => (load_halfka2_weights(path)?, path.display().to_string()),
            None => (synthetic_halfka2_weights(shape), "deterministic halfka2".to_string()),
        };
        let (batch, batch_source) = match args.teacher.as_ref() {
            Some(_) => load_halfka2_teacher_batch(args)?,
            None => (
                synthetic_batch(batch_size, max_active, shape.input_size, shape.num_stacks),
                "deterministic halfka2".to_string(),
            ),
        };

        Ok(Self { label: "halfka2-1024-7-64-k3k3", weights_source, batch_source, weights, batch })
    }
}

fn require_nonzero(name: &'static str, value: usize) -> Result<(), Box<dyn std::error::Error>> {
    if value == 0 { Err(invalid_input(format!("--{name} must be greater than zero"))) } else { Ok(()) }
}

fn invalid_input(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()))
}

fn load_halfka2_weights(path: &PathBuf) -> Result<SfnnForwardOwnedWeights, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let records = parse_model_weights_bin(&bytes)?;
    let weights = if records.contains_key("l0w") {
        records
    } else {
        extract_component_section(&records, "nnue", "weights")
    };

    Ok(SfnnForwardOwnedWeights::from_weight_map(SFNN_HALFKA2_1024_7_64_K3K3, &weights)?)
}

fn load_halfka2_teacher_batch(args: &Args) -> Result<(FastBatchHost, String), Box<dyn std::error::Error>> {
    let teacher = args.teacher.as_ref().expect("checked by caller");
    let data_files_owned = expand_teacher(teacher).map_err(invalid_input)?;
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(String::as_str).collect();
    let format = infer_data_format(&data_files_ref).map_err(invalid_input)?;
    let batch_size = args.batch_size.unwrap_or(2);
    require_nonzero("batch-size", batch_size)?;
    let source = format!("{format:?} teacher: {teacher}");

    let batch = match format {
        DataFormat::Hcpe => {
            let loader = HcpeDataLoader::new_concat_multiple(
                &data_files_ref,
                args.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_buffer_records(batch_size)
            .with_loader_threads(args.loader_threads)
            .with_single_epoch(true);
            materialise_first_halfka2_batch(loader, args)?
        }
        DataFormat::Hcpe3 => {
            let loader = Hcpe3DataLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true)
                .with_buffer_records(batch_size)
                .with_single_epoch(true);
            materialise_first_halfka2_batch(loader, args)?
        }
        DataFormat::Pack => {
            let loader =
                ShogiPackLoader::new_concat_multiple(&data_files_ref, args.buffer_mb, |_| true).with_single_epoch(true);
            materialise_first_halfka2_batch(loader, args)?
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_files_ref).with_single_epoch(true);
            materialise_first_halfka2_batch(loader, args)?
        }
    };

    Ok((batch, source))
}

fn materialise_first_halfka2_batch<D>(loader: D, args: &Args) -> Result<FastBatchHost, Box<dyn std::error::Error>>
where
    D: DataLoader<PackedSfenValue>,
{
    let batch_size = args.batch_size.unwrap_or(2);
    let threads = args.threads.max(1);
    let score_drop_abs = (args.score_drop_abs > 0).then_some(args.score_drop_abs);
    let dataloader = DefaultDataLoader::new(
        ShogiHalfKa2,
        ShogiLayerStackBucket9::KingRank9,
        (|_, blend| blend) as fn(&PackedSfenValue, f32) -> f32,
        None,
        args.nnue_pytorch_wrm_loss,
        false,
        args.scale,
        score_drop_abs,
        loader,
    );
    let mut first_batch = None;
    dataloader.load_and_map_batches(0, batch_size, |batch| {
        let prepared = dataloader.prepare(batch, threads, 1.0 - args.lambda);
        first_batch = Some(FastBatchHost::from(prepared));
        true
    });

    let batch = first_batch.ok_or_else(|| {
        invalid_input(format!(
            "teacher did not yield a complete batch of {batch_size} positions; use a smaller --batch-size"
        ))
    })?;
    batch.validate().map_err(invalid_input)?;
    Ok(batch)
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

fn synthetic_batch(batch_size: usize, max_active: usize, input_size: usize, num_stacks: usize) -> FastBatchHost {
    let (stm, nstm) = deterministic_sparse_batch(batch_size, max_active, input_size);
    FastBatchHost {
        layout: FastBatchLayout { batch_size, max_active, output_size: 1, hand_count_dim: 0 },
        stm,
        nstm,
        buckets: (0..batch_size).map(|idx| (idx % num_stacks) as i32).collect(),
        targets: vec![0.0; batch_size],
        weights: vec![1.0; batch_size],
        hand_count: None,
    }
}

fn tiny_weights(shape: SfnnForwardShape) -> SfnnForwardOwnedWeights {
    SfnnForwardOwnedWeights {
        shape,
        l0w: vec![
            0.2, 0.1, -0.1, 0.0, // feature 0
            -0.1, 0.2, 0.1, 0.2, // feature 1
            0.0, -0.2, 0.2, 0.1, // feature 2
            0.3, 0.0, -0.3, 0.2, // feature 3
        ],
        l0b: vec![0.1, 0.2, 0.3, 0.4],
        l1w: vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, // combined 0
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // combined 1
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // combined 2
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, // combined 3
        ],
        l1b: vec![0.0; 6],
        l2w: vec![
            1.0, 0.0, 0.0, 0.0, // l2 input 0
            0.0, 1.0, 0.0, 0.0, // l2 input 1
            1.0, 0.0, 1.0, 0.0, // l2 input 2
            0.0, 1.0, 0.0, 1.0, // l2 input 3
        ],
        l2b: vec![0.0; 4],
        l3w: vec![
            2.0, -0.5, // l2 output 0
            -1.0, 0.8, // l2 output 1
        ],
        l3b: vec![0.1, -0.02],
    }
}

fn synthetic_halfka2_weights(shape: SfnnForwardShape) -> SfnnForwardOwnedWeights {
    SfnnForwardOwnedWeights {
        shape,
        l0w: deterministic_f32_vec(sfnn_l0w_len(shape), 0x5F23_4CB8, 0.004, 0.0),
        l0b: deterministic_f32_vec(sfnn_l0b_len(shape), 0x10B1_5F23, 0.02, 0.10),
        l1w: deterministic_f32_vec(sfnn_l1w_len(shape), 0xC1A5_5F23, 0.002, 0.0),
        l1b: deterministic_f32_vec(sfnn_l1b_len(shape), 0xB1A5_5F23, 0.004, 0.02),
        l2w: deterministic_f32_vec(sfnn_l2w_len(shape), 0xD2A5_5F23, 0.003, 0.0),
        l2b: deterministic_f32_vec(sfnn_l2b_len(shape), 0xB2A5_5F23, 0.004, 0.02),
        l3w: deterministic_f32_vec(sfnn_l3w_len(shape), 0xD3A5_5F23, 0.02, 0.0),
        l3b: deterministic_f32_vec(sfnn_l3b_len(shape), 0xB3A5_5F23, 0.002, 0.01),
    }
}

fn sfnn_l0w_len(shape: SfnnForwardShape) -> usize {
    shape.input_size * shape.ft_size
}

fn sfnn_l0b_len(shape: SfnnForwardShape) -> usize {
    shape.ft_size
}

fn sfnn_l1w_len(shape: SfnnForwardShape) -> usize {
    shape.ft_size * shape.num_stacks * shape.l1_out()
}

fn sfnn_l1b_len(shape: SfnnForwardShape) -> usize {
    shape.num_stacks * shape.l1_out()
}

fn sfnn_l2w_len(shape: SfnnForwardShape) -> usize {
    shape.l2_in() * shape.num_stacks * shape.l2_size
}

fn sfnn_l2b_len(shape: SfnnForwardShape) -> usize {
    shape.num_stacks * shape.l2_size
}

fn sfnn_l3w_len(shape: SfnnForwardShape) -> usize {
    shape.l2_size * shape.num_stacks
}

fn sfnn_l3b_len(shape: SfnnForwardShape) -> usize {
    shape.num_stacks
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
