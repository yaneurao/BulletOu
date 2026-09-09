use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use bulletou_lib::{
    game::outputs::{
        SHOGI_SFNN_PROGRESS_WEIGHT_COUNT, ShogiProgressKPAbs, ShogiSfnnProgressQ16Params,
        shogi_sfnn_progress_0_to_255_from_sum_q16,
    },
    shogi::PackedSfenValue,
    value::loader::shogipack::ShogiPackGameReader,
};
use clap::Parser;

const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1.0e-8;
const Q16_SCALE: f64 = 65_536.0;

#[derive(Parser, Debug)]
#[command(name = "bulletou progress-train")]
#[command(about = "Train a shared 0..255 SFNN progress classifier from complete YaneuraOu .pack games")]
#[command(
    after_help = "Target for position i in a game containing L evaluable positions:\n  target = i / (L - 1)\n\nThe first position is 0 and the last position before the terminal marker is 1.\nThe exported progress.bin is shared by progress2/progress4/progress8 and other progressN architectures."
)]
pub struct ProgressTrainArgs {
    /// Complete game-record .pack file. PSV/bin data cannot be used because it does not preserve game boundaries.
    #[arg(long)]
    teacher: PathBuf,

    /// Output bias-free progress.bin: 125,388 f64 little-endian weights (tatara / PR #326 format).
    #[arg(long)]
    output: PathBuf,

    /// Number of full training passes over the .pack file.
    #[arg(long, default_value_t = 5)]
    epochs: usize,

    /// Number of positions whose gradients are averaged for one Adam update.
    #[arg(long, default_value_t = 4096)]
    batch_size: usize,

    /// Fixed Adam learning rate. No hidden learning-rate multiplier is applied.
    #[arg(long, default_value_t = 0.0002)]
    lr: f32,

    /// Hold out every Nth game for validation. Use 0 to disable validation.
    #[arg(long, default_value_t = 20)]
    validation_game_stride: u64,

    /// Stop after this many source games. Zero scans the entire file.
    #[arg(long, default_value_t = 0)]
    max_games: u64,

    /// Print scan/training progress at least this often.
    #[arg(long, default_value_t = 10.0)]
    log_interval_seconds: f64,

    /// Replace an existing output file.
    #[arg(long)]
    overwrite: bool,
}

#[derive(Clone, Copy)]
struct ValidationSample {
    pos: PackedSfenValue,
    target: f32,
    endpoint: Endpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endpoint {
    First,
    Middle,
    Last,
}

#[derive(Default)]
struct Metrics {
    positions: u64,
    mse_sum: f64,
    mae_sum: f64,
    first_sum: f64,
    first_count: u64,
    last_sum: f64,
    last_count: u64,
    bucket4: [u64; 4],
    bucket8: [u64; 8],
}

impl Metrics {
    fn add(&mut self, prediction: f32, target: f32, endpoint: Endpoint) {
        let prediction = prediction.clamp(0.0, 1.0);
        let error = f64::from(prediction - target);
        self.positions += 1;
        self.mse_sum += error * error;
        self.mae_sum += error.abs();
        if endpoint == Endpoint::First {
            self.first_sum += f64::from(prediction);
            self.first_count += 1;
        }
        if endpoint == Endpoint::Last {
            self.last_sum += f64::from(prediction);
            self.last_count += 1;
        }
        self.bucket4[hard_bucket(prediction, 4)] += 1;
        self.bucket8[hard_bucket(prediction, 8)] += 1;
    }

    fn mse(&self) -> f64 {
        divide_or_nan(self.mse_sum, self.positions)
    }

    fn mae(&self) -> f64 {
        divide_or_nan(self.mae_sum, self.positions)
    }

    fn first_mean(&self) -> f64 {
        divide_or_nan(self.first_sum, self.first_count)
    }

    fn last_mean(&self) -> f64 {
        divide_or_nan(self.last_sum, self.last_count)
    }
}

fn divide_or_nan(sum: f64, count: u64) -> f64 {
    if count == 0 { f64::NAN } else { sum / count as f64 }
}

struct AdamState {
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    beta1_power: f32,
    beta2_power: f32,
}

impl AdamState {
    fn new(parameter_count: usize) -> Self {
        Self {
            first_moment: vec![0.0; parameter_count],
            second_moment: vec![0.0; parameter_count],
            beta1_power: 1.0,
            beta2_power: 1.0,
        }
    }

    fn update(&mut self, parameters: &mut [f32], gradient: &mut [f32], samples: usize, lr: f32) {
        let inverse_samples = 1.0 / samples as f32;
        self.beta1_power *= ADAM_BETA1;
        self.beta2_power *= ADAM_BETA2;
        let correction1 = (1.0 - self.beta1_power).max(f32::MIN_POSITIVE);
        let correction2 = (1.0 - self.beta2_power).max(f32::MIN_POSITIVE);

        for (((parameter, first), second), grad) in parameters
            .iter_mut()
            .zip(self.first_moment.iter_mut())
            .zip(self.second_moment.iter_mut())
            .zip(gradient.iter_mut())
        {
            let g = *grad * inverse_samples;
            *first = ADAM_BETA1.mul_add(*first, (1.0 - ADAM_BETA1) * g);
            *second = ADAM_BETA2.mul_add(*second, (1.0 - ADAM_BETA2) * g * g);
            let first_hat = *first / correction1;
            let second_hat = *second / correction2;
            *parameter -= lr * first_hat / (second_hat.sqrt() + ADAM_EPS);
            *parameter = parameter.clamp(-16.0, 16.0);
            *grad = 0.0;
        }
    }
}

struct ProgressPrinter {
    pass_started: Instant,
    last_printed: Instant,
    interval: Duration,
}

impl ProgressPrinter {
    fn new(interval_seconds: f64) -> Self {
        let now = Instant::now();
        Self { pass_started: now, last_printed: now, interval: Duration::from_secs_f64(interval_seconds.max(0.1)) }
    }

    fn maybe_print(&mut self, label: &str, reader: &ShogiPackGameReader, games: u64, positions: u64) {
        if self.last_printed.elapsed() < self.interval {
            return;
        }
        self.print(label, reader, games, positions);
        self.last_printed = Instant::now();
    }

    fn print(&self, label: &str, reader: &ShogiPackGameReader, games: u64, positions: u64) {
        let elapsed = self.pass_started.elapsed().as_secs_f64();
        let consumed = reader.bytes_consumed();
        let file_len = reader.file_len();
        let percent = if file_len == 0 { 100.0 } else { consumed as f64 * 100.0 / file_len as f64 };
        println!(
            "  [{label}] games={} positions={} source={}/{} ({:.2}%) elapsed={:.1}s pos/s={:.0}",
            format_u64(games),
            format_u64(positions),
            format_u64(consumed),
            format_u64(file_len),
            percent,
            elapsed,
            positions as f64 / elapsed.max(f64::MIN_POSITIVE),
        );
    }
}

struct PreparedData {
    total_games: u64,
    usable_games: u64,
    skipped_short_games: u64,
    total_positions: u64,
    training_games: u64,
    training_positions: u64,
    validation_games: u64,
    validation: Vec<ValidationSample>,
}

pub fn run_progress_train(args: &ProgressTrainArgs) -> Result<(), String> {
    validate_args(args)?;

    println!("progress-train:");
    println!("  teacher            = {}", args.teacher.display());
    println!("  output             = {}", args.output.display());
    println!("  target             = position_index / (game_positions - 1)");
    println!("  output scale       = shared scalar 0..255 (progressN-independent)");
    println!("  optimiser          = Adam, lr={:.8}, batch_positions={}", args.lr, format_u64(args.batch_size as u64));
    println!("  epochs             = {}", args.epochs);
    if args.validation_game_stride == 0 {
        println!("  validation         = disabled");
    } else {
        println!("  validation         = every {}th complete game", args.validation_game_stride);
    }
    if args.max_games == 0 {
        println!("  source limit       = all games in the file");
    } else {
        println!("  source limit       = {} games", format_u64(args.max_games));
    }

    let prepared = prepare_data(args)?;
    println!("source scan complete:");
    println!("  games              = {}", format_u64(prepared.total_games));
    println!("  evaluable positions= {}", format_u64(prepared.total_positions));
    println!("  usable games       = {}", format_u64(prepared.usable_games));
    println!("  skipped (<2 pos)   = {}", format_u64(prepared.skipped_short_games));
    println!(
        "  training split     = {} games, {} positions",
        format_u64(prepared.training_games),
        format_u64(prepared.training_positions)
    );
    println!(
        "  validation split   = {} games, {} positions ({:.1} MiB cached)",
        format_u64(prepared.validation_games),
        format_u64(prepared.validation.len() as u64),
        (prepared.validation.len() * std::mem::size_of::<ValidationSample>()) as f64 / (1024.0 * 1024.0)
    );

    if prepared.training_positions == 0 {
        return Err("no training positions remain after validation splitting".to_string());
    }

    let parameter_count = SHOGI_SFNN_PROGRESS_WEIGHT_COUNT;
    let mut parameters = vec![0.0f32; parameter_count];
    let mut adam = AdamState::new(parameter_count);
    let baseline = evaluate_f32(&parameters, &prepared.validation);
    if baseline.positions > 0 {
        print_metrics("baseline validation", &baseline);
    }

    let mut best_parameters = parameters.clone();
    let mut best_selection_loss = f64::INFINITY;
    let mut best_epoch = 0usize;

    for epoch in 1..=args.epochs {
        let train_metrics = train_epoch(args, epoch, &mut parameters, &mut adam)?;
        print_metrics(&format!("epoch {epoch} train"), &train_metrics);

        let validation_metrics = evaluate_f32(&parameters, &prepared.validation);
        if validation_metrics.positions > 0 {
            print_metrics(&format!("epoch {epoch} validation"), &validation_metrics);
        }

        let selection_loss =
            if validation_metrics.positions > 0 { validation_metrics.mse() } else { train_metrics.mse() };
        if selection_loss < best_selection_loss {
            best_selection_loss = selection_loss;
            best_parameters.clone_from(&parameters);
            best_epoch = epoch;
            println!("  [best] epoch={epoch} selection_mse={selection_loss:.9}");
        }
    }

    let q16 = quantize_parameters(&best_parameters)?;
    let quantized_metrics = evaluate_q16(&q16, &prepared.validation);
    if quantized_metrics.positions > 0 {
        print_metrics("quantized validation", &quantized_metrics);
    }
    write_progress_bin(&args.output, &best_parameters, args.overwrite)?;

    println!("progress-train complete:");
    println!("  source games       = {}", format_u64(prepared.total_games));
    println!("  source positions   = {}", format_u64(prepared.total_positions));
    println!("  selected epoch     = {}", best_epoch);
    println!("  selected MSE       = {:.9}", best_selection_loss);
    println!("  output             = {}", args.output.display());
    println!("  output bytes       = {}", format_u64(fs::metadata(&args.output).map_err(|e| e.to_string())?.len()));
    Ok(())
}

fn validate_args(args: &ProgressTrainArgs) -> Result<(), String> {
    if args.teacher.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("pack")
    {
        return Err(format!("--teacher must be one complete-game .pack file; got {}", args.teacher.display()));
    }
    if args.epochs == 0 {
        return Err("--epochs must be at least 1".to_string());
    }
    if args.batch_size == 0 {
        return Err("--batch-size must be at least 1".to_string());
    }
    if !args.lr.is_finite() || args.lr <= 0.0 {
        return Err("--lr must be finite and greater than zero".to_string());
    }
    if !args.log_interval_seconds.is_finite() || args.log_interval_seconds <= 0.0 {
        return Err("--log-interval-seconds must be finite and greater than zero".to_string());
    }
    if args.output.exists() && !args.overwrite {
        return Err(format!("{} already exists; pass --overwrite to replace it", args.output.display()));
    }
    Ok(())
}

fn prepare_data(args: &ProgressTrainArgs) -> Result<PreparedData, String> {
    let mut reader = ShogiPackGameReader::open(&args.teacher)
        .map_err(|e| format!("failed to open {}: {e}", args.teacher.display()))?;
    let mut printer = ProgressPrinter::new(args.log_interval_seconds);
    let mut total_games = 0u64;
    let mut usable_games = 0u64;
    let mut skipped_short_games = 0u64;
    let mut total_positions = 0u64;
    let mut training_games = 0u64;
    let mut training_positions = 0u64;
    let mut validation_games = 0u64;
    let mut validation = Vec::new();

    while args.max_games == 0 || total_games < args.max_games {
        let Some(game) =
            reader.next_game().map_err(|e| format!("failed while scanning {}: {e}", args.teacher.display()))?
        else {
            break;
        };
        total_games += 1;
        total_positions += game.len() as u64;
        if game.len() < 2 {
            skipped_short_games += 1;
            printer.maybe_print("scan", &reader, total_games, total_positions);
            continue;
        }
        usable_games += 1;

        if is_validation_game(total_games, args.validation_game_stride) {
            validation_games += 1;
            let denominator = (game.len() - 1) as f32;
            validation.reserve(game.len());
            for (index, pos) in game.into_iter().enumerate() {
                validation.push(ValidationSample {
                    pos,
                    target: index as f32 / denominator,
                    endpoint: endpoint(index, denominator as usize + 1),
                });
            }
        } else {
            training_games += 1;
            training_positions += game.len() as u64;
        }
        printer.maybe_print("scan", &reader, total_games, total_positions);
    }
    printer.print("scan", &reader, total_games, total_positions);

    Ok(PreparedData {
        total_games,
        usable_games,
        skipped_short_games,
        total_positions,
        training_games,
        training_positions,
        validation_games,
        validation,
    })
}

fn train_epoch(
    args: &ProgressTrainArgs,
    epoch: usize,
    parameters: &mut [f32],
    adam: &mut AdamState,
) -> Result<Metrics, String> {
    let mut reader = ShogiPackGameReader::open(&args.teacher)
        .map_err(|e| format!("failed to open {} for epoch {epoch}: {e}", args.teacher.display()))?;
    let mut printer = ProgressPrinter::new(args.log_interval_seconds);
    let mut metrics = Metrics::default();
    let mut gradient = vec![0.0f32; parameters.len()];
    let mut active = Vec::with_capacity(96);
    let mut games = 0u64;
    let mut batch_positions = 0usize;

    while args.max_games == 0 || games < args.max_games {
        let Some(game) = reader
            .next_game()
            .map_err(|e| format!("failed during epoch {epoch} in {}: {e}", args.teacher.display()))?
        else {
            break;
        };
        games += 1;
        if game.len() < 2 || is_validation_game(games, args.validation_game_stride) {
            printer.maybe_print(&format!("epoch {epoch}"), &reader, games, metrics.positions);
            continue;
        }

        let game_len = game.len();
        let denominator = (game_len - 1) as f32;
        for (index, pos) in game.iter().enumerate() {
            let target = index as f32 / denominator;
            ShogiProgressKPAbs::collect_active_indices(pos, &mut active);
            let prediction = predict_f32(parameters, &active);
            metrics.add(prediction, target, endpoint(index, game_len));

            let error = prediction - target;
            let scale = 2.0 * error * prediction * (1.0 - prediction);
            for &feature in &active {
                gradient[feature] += scale;
            }
            batch_positions += 1;
            if batch_positions == args.batch_size {
                adam.update(parameters, &mut gradient, batch_positions, args.lr);
                batch_positions = 0;
            }
        }
        printer.maybe_print(&format!("epoch {epoch}"), &reader, games, metrics.positions);
    }

    if batch_positions > 0 {
        adam.update(parameters, &mut gradient, batch_positions, args.lr);
    }
    printer.print(&format!("epoch {epoch}"), &reader, games, metrics.positions);
    Ok(metrics)
}

fn evaluate_f32(parameters: &[f32], samples: &[ValidationSample]) -> Metrics {
    let mut metrics = Metrics::default();
    let mut active = Vec::with_capacity(96);
    for sample in samples {
        ShogiProgressKPAbs::collect_active_indices(&sample.pos, &mut active);
        metrics.add(predict_f32(parameters, &active), sample.target, sample.endpoint);
    }
    metrics
}

fn evaluate_q16(parameters: &ShogiSfnnProgressQ16Params, samples: &[ValidationSample]) -> Metrics {
    let mut metrics = Metrics::default();
    let mut active = Vec::with_capacity(96);
    for sample in samples {
        ShogiProgressKPAbs::collect_active_indices(&sample.pos, &mut active);
        let sum = active.iter().fold(0i64, |sum, &feature| sum + i64::from(parameters.weights_q16[feature]));
        let progress = shogi_sfnn_progress_0_to_255_from_sum_q16(sum);
        metrics.add(f32::from(progress) / 255.0, sample.target, sample.endpoint);
    }
    metrics
}

fn predict_f32(parameters: &[f32], active: &[usize]) -> f32 {
    let sum = active.iter().fold(0.0, |sum, &feature| sum + parameters[feature]);
    sigmoid(sum)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn hard_bucket(progress: f32, buckets: usize) -> usize {
    ((progress * buckets as f32).floor() as usize).min(buckets - 1)
}

fn endpoint(index: usize, game_len: usize) -> Endpoint {
    if index == 0 {
        Endpoint::First
    } else if index + 1 == game_len {
        Endpoint::Last
    } else {
        Endpoint::Middle
    }
}

fn is_validation_game(game_number: u64, stride: u64) -> bool {
    stride > 0 && game_number % stride == 0
}

fn quantize_parameters(parameters: &[f32]) -> Result<ShogiSfnnProgressQ16Params, String> {
    let weights_q16 = parameters.iter().map(|&value| f64_to_i32_q16(f64::from(value))).collect();
    ShogiSfnnProgressQ16Params::new(weights_q16)
}

fn f64_to_i32_q16(value: f64) -> i32 {
    (value * Q16_SCALE).round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

fn write_progress_bin(path: &Path, parameters: &[f32], overwrite: bool) -> Result<(), String> {
    if path.exists() && !overwrite {
        return Err(format!("{} already exists; pass --overwrite to replace it", path.display()));
    }
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }

    let temp = temporary_output_path(path);
    let file = File::create(&temp).map_err(|e| format!("failed to create {}: {e}", temp.display()))?;
    let mut writer = BufWriter::new(file);
    for &weight in parameters {
        writer
            .write_all(&f64::from(weight).to_le_bytes())
            .map_err(|e| format!("failed to write {}: {e}", temp.display()))?;
    }
    writer.flush().map_err(|e| format!("failed to flush {}: {e}", temp.display()))?;
    drop(writer);

    if path.exists() {
        fs::remove_file(path).map_err(|e| format!("failed to replace {}: {e}", path.display()))?;
    }
    fs::rename(&temp, path).map_err(|e| format!("failed to rename {} to {}: {e}", temp.display(), path.display()))
}

fn temporary_output_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn print_metrics(label: &str, metrics: &Metrics) {
    println!(
        "  [{label}] positions={} mse={:.9} mae={:.9} first_mean={:.5} last_mean={:.5}",
        format_u64(metrics.positions),
        metrics.mse(),
        metrics.mae(),
        metrics.first_mean(),
        metrics.last_mean(),
    );
    println!("    progress4 buckets = {}", format_histogram(&metrics.bucket4));
    println!("    progress8 buckets = {}", format_histogram(&metrics.bucket8));
}

fn format_histogram(histogram: &[u64]) -> String {
    let total: u64 = histogram.iter().sum();
    histogram
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            let percentage = if total == 0 { 0.0 } else { count as f64 * 100.0 / total as f64 };
            format!("b{index}:{}({percentage:.2}%)", format_u64(count))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_u64(value: u64) -> String {
    let text = value.to_string();
    let mut output = String::with_capacity(text.len() + text.len() / 3);
    for (index, ch) in text.chars().enumerate() {
        if index > 0 && (text.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(ch);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_cover_zero_and_one() {
        let len = 101usize;
        assert_eq!(0.0 / (len - 1) as f32, 0.0);
        assert_eq!((len - 1) as f32 / (len - 1) as f32, 1.0);
        assert_eq!(endpoint(0, len), Endpoint::First);
        assert_eq!(endpoint(len - 1, len), Endpoint::Last);
    }

    #[test]
    fn stable_sigmoid_and_bucket_boundaries() {
        assert!(sigmoid(-100.0) < 1.0e-20);
        assert!((sigmoid(0.0) - 0.5).abs() < f32::EPSILON);
        assert!(sigmoid(100.0) > 0.999_999);
        assert_eq!(hard_bucket(0.0, 4), 0);
        assert_eq!(hard_bucket(0.25, 4), 1);
        assert_eq!(hard_bucket(1.0, 4), 3);
    }

    #[test]
    fn q16_round_trip_is_precise() {
        let value = 1.234_567f64;
        let q16 = f64_to_i32_q16(value);
        assert!((f64::from(q16) / Q16_SCALE - value).abs() <= 0.5 / Q16_SCALE);
    }
}
