//! Export BulletOu teacher data as flat PackedSfenValue (`.psv`).
//!
//! Tatara's `nnue_train` consumes 40-byte PackedSfenValue records, while
//! BulletOu can train directly from HCPE/HCPE3/pack.  This utility runs the
//! same BulletOu teacher decoding path and writes the resulting PSV stream so
//! both trainers can be benchmarked on the same positions.

use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    mem::size_of,
    path::{Path, PathBuf},
};

use bulletou_lib::{
    shogi::PackedSfenValue,
    teacher_path::{expand_teacher, infer_data_format, DataFormat},
    value::loader::{DataLoader, DirectSequentialDataLoader, Hcpe3DataLoader, HcpeDataLoader, ShogiPackLoader},
};
use clap::Parser;

const PSV_RECORD_BYTES: usize = size_of::<PackedSfenValue>();

#[derive(Debug, Parser)]
struct Args {
    /// Teacher file, directory, or comma-separated list (.hcpe/.hcpe3/.pack/.psv).
    #[arg(long)]
    teacher: String,

    /// Output PSV path.
    #[arg(long)]
    out: PathBuf,

    /// Maximum number of positions to export. Omit to export one full pass.
    #[arg(long)]
    positions: Option<usize>,

    /// Number of accepted positions to skip before exporting.
    #[arg(long, default_value_t = 0)]
    start_position: usize,

    /// Decode/read buffer size in MiB for HCPE/HCPE3/pack loaders.
    #[arg(long, default_value_t = 64)]
    buffer_mb: usize,

    /// HCPE decode worker threads. 0 means auto. Ignored for other formats.
    #[arg(long, default_value_t = 0)]
    loader_threads: usize,
}

#[derive(Debug)]
struct ExportStats {
    written_positions: usize,
    known_input_positions: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let data_files = expand_teacher(&args.teacher)?;
    let data_file_refs = data_files.iter().map(String::as_str).collect::<Vec<_>>();
    let format = infer_data_format(&data_file_refs)?;

    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }

    let stats = match format {
        DataFormat::Hcpe => {
            let mut loader = HcpeDataLoader::new_concat_multiple(
                &data_file_refs,
                args.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_single_epoch(true);
            if args.loader_threads != 0 {
                loader = loader.with_loader_threads(args.loader_threads);
            }
            export_loader(loader, &args.out, args.start_position, args.positions)?
        }
        DataFormat::Hcpe3 => {
            let loader = Hcpe3DataLoader::new_concat_multiple(
                &data_file_refs,
                args.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_single_epoch(true);
            export_loader(loader, &args.out, args.start_position, args.positions)?
        }
        DataFormat::Pack => {
            let loader = ShogiPackLoader::new_concat_multiple(
                &data_file_refs,
                args.buffer_mb,
                (|_| true) as fn(&PackedSfenValue) -> bool,
            )
            .with_single_epoch(true);
            export_loader(loader, &args.out, args.start_position, args.positions)?
        }
        DataFormat::Psv => {
            let loader = DirectSequentialDataLoader::new(&data_file_refs).with_single_epoch(true);
            export_loader(loader, &args.out, args.start_position, args.positions)?
        }
    };

    let written_bytes = stats.written_positions.saturating_mul(PSV_RECORD_BYTES);
    let input_positions = stats.known_input_positions.map(|n| n.to_string()).unwrap_or_else(|| "unknown".to_string());

    println!("format             : {format:?}");
    println!("input_files        : {}", data_files.len());
    println!("input_positions    : {input_positions}");
    println!("start_position     : {}", args.start_position);
    println!("written_positions  : {}", stats.written_positions);
    println!("written_bytes      : {written_bytes}");
    println!("out                : {}", args.out.display());

    Ok(())
}

fn export_loader<D>(
    loader: D,
    out_path: &Path,
    start_position: usize,
    positions: Option<usize>,
) -> io::Result<ExportStats>
where
    D: DataLoader<PackedSfenValue>,
{
    let known_input_positions = loader.count_positions();
    let out_file = File::create(out_path)?;
    let mut writer = BufWriter::new(out_file);
    let mut written_positions = 0usize;
    let mut write_error: Option<io::Error> = None;

    if positions == Some(0) {
        writer.flush()?;
        return Ok(ExportStats { written_positions, known_input_positions });
    }

    loader.map_chunks(start_position, |chunk| {
        if write_error.is_some() {
            return true;
        }

        let take =
            positions.map(|limit| limit.saturating_sub(written_positions)).unwrap_or(chunk.len()).min(chunk.len());

        for psv in &chunk[..take] {
            if let Err(err) = writer.write_all(psv.as_bytes()) {
                write_error = Some(err);
                return true;
            }
        }

        written_positions += take;
        positions.is_some_and(|limit| written_positions >= limit)
    });

    if let Some(err) = write_error {
        return Err(err);
    }
    writer.flush()?;

    Ok(ExportStats { written_positions, known_input_positions })
}
