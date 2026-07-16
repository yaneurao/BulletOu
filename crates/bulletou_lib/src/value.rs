pub(crate) mod builder;
mod dataloader;
pub mod fast_batch;
pub mod fast_nnue;
pub mod loader;
pub mod nnue_save;
pub mod nnue_save_sfnn1536;
mod save;
pub mod yaneuraou_kppt;

use std::cell::RefCell;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};
pub use dataloader::FastValueDataLoader;
pub use fast_batch::{
    FastBatchHost, FastBatchLayout, FastReferenceError, ForwardComparison, compare_forward_outputs,
};
pub use fast_nnue::{
    FastNnueError, NNUE_HALFKP_256X2_32_32, NnueForwardOwnedWeights, NnueForwardShape, NnueForwardWeights,
    NnueForwardTrace, NnueForwardWorkspaceLayout,
};
use bullet_compiler::tensor::TValue;
use bullet_trainer::{
    Trainer,
    model::save::SavedFormat,
    optimiser::OptimiserState,
    run::{self, dataloader::PreparedBatchHost, logger},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::ExecutionContext,
    trainer::{
        schedule::{TrainingSchedule, lr::LrScheduler, wdl::WdlScheduler},
        settings::LocalSettings,
    },
    value::{
        dataloader::ValueDataLoader,
        loader::{DefaultDataLoader, LoadableDataType},
    },
};

use crate::value::loader::PreparedData;

/// Value network trainer, generally for training NNUE networks.
pub struct ValueTrainer<
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
>(Trainer<ExecutionContext, Opt, ValueTrainerState<Inp, Out>>);

impl<Opt, Inp, Out> std::ops::Deref for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    type Target = Trainer<ExecutionContext, Opt, ValueTrainerState<Inp, Out>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Opt, Inp, Out> std::ops::DerefMut for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;
type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;

#[derive(Clone)]
pub struct ValueTrainerState<Inp: SparseInputType, Out> {
    input_getter: Inp,
    output_getter: Out,
    blend_getter: B<Inp>,
    weight_getter: Option<Wgt<Inp>>,
    saved_format: Vec<SavedFormat>,
    use_win_rate_model: bool,
    wdl: bool,
    /// `Some(cap)` のとき `|score| >= cap` の局面を loss から除外。
    /// builder の `score_drop_abs(cap)` で設定。
    score_drop_abs: Option<u16>,
}

impl<Inp: SparseInputType, Out> ValueTrainerState<Inp, Out>
where
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn prepare(
        &self,
        batch: &[Inp::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> PreparedBatchHost {
        PreparedBatchHost::from(PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            batch,
            threads,
            blend,
            scale,
            self.score_drop_abs,
        ))
    }
}

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    /// Run training to completion (or until the dataloader yields EOF, if it
    /// is configured to do so). Returns the in-memory error-record produced
    /// by the per-32-batch loss callback — same shape as the rows in the
    /// `log.txt` that bullet writes at each save (`(superbatch, batch, loss)`).
    /// Most callers ignore this value; it is exposed so a caller that wants to
    /// persist a partial-superbatch log on its own (e.g. for a fallback save
    /// when no superbatch boundary was crossed) can do so.
    pub fn run(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
    ) -> Vec<(usize, usize, f32)> {
        logger::clear_colours();
        println!("{}", logger::ansi("Training Preamble", "34;1"));

        schedule.display();
        settings.display();

        if settings.test_set.is_some() {
            println!(
                "{}",
                logger::ansi("Warning: Validation data not currently implemented! Please bother me on discord.", "31")
            )
        }

        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            self.state.score_drop_abs,
            dataloader.clone(),
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();

        let steps = schedule.steps;

        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        self.train_custom(
            run::schedule::TrainingSchedule {
                steps,
                log_rate: 128,
                batch_queue_size: settings.batch_queue_size,
                delay_loss_readback: true,
                lr_schedule: Box::new(|a, b| lr_scheduler.lr(a, b)),
            },
            ValueDataLoader { steps, threads: settings.threads, dataloader, wdl: schedule.wdl_scheduler.clone() },
            |_, superbatch, curr_batch, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if curr_batch % 32 == 0
                    || (steps.batches_per_superbatch < 32 && curr_batch == steps.batches_per_superbatch)
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, steps.batches_per_superbatch as f32);

                    error_record.borrow_mut().push((superbatch, curr_batch, normalised_loss));

                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |trainer, superbatch| {
                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));

                    if let Some(ref callback) = settings.on_checkpoint_saved {
                        callback(superbatch);
                    }
                }
            },
        )
        .unwrap();

        error_record.into_inner()
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        self.0.optimiser.model.set_fwd_batch_size(1).unwrap();

        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let host_data = self.state.prepare(&[pos], 1, 1.0, 1.0);

        let model = &self.optimiser.model;
        let device = model.device();
        let stream = device.new_stream().unwrap();

        let inputs = host_data.to_device(&device).unwrap();
        let outputs = model.make_forward_output_tensors(1).unwrap();
        model.forward(&stream, &inputs, &outputs).unwrap().value().unwrap();

        let output = outputs.get("outputs/output").unwrap().clone();
        let TValue::F32(output) = output.to_host().unwrap() else { panic!() };
        output
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        let vals = self.eval_raw_output(fen);

        match vals[..] {
            [mut loss, mut draw, mut win] => {
                let max = win.max(draw).max(loss);
                win = (win - max).exp();
                draw = (draw - max).exp();
                loss = (loss - max).exp();

                (win + draw / 2.0) / (win + draw + loss)
            }
            [score] => score,
            _ => panic!("Invalid output size!"),
        }
    }

    /// Run a forward pass on a slice of pre-decoded positions, splitting into
    /// chunks of `batch_size` so the GPU can process many positions per kernel
    /// launch. Returns the raw scalar output of the network for each input
    /// position (= sigmoid logit for value-network checkpoints), in input
    /// order.
    ///
    /// Use this for offline validation against a held-out test set: skipping
    /// the `FromStr` step in [`eval_raw_output`] avoids per-position FEN
    /// round-tripping and the larger batch fully utilises the GPU.
    pub fn eval_packed_batch(
        &mut self,
        positions: &[Inp::RequiredDataType],
        batch_size: usize,
    ) -> Vec<f32>
    where
        Inp::RequiredDataType: LoadableDataType,
    {
        assert!(batch_size > 0, "batch_size must be > 0");
        let mut out = Vec::with_capacity(positions.len());
        for chunk in positions.chunks(batch_size) {
            let n = chunk.len();
            self.0.optimiser.model.set_fwd_batch_size(n).unwrap();
            // eval_scale / blend factors are loss-side only; both default 1.0
            // is fine here because we never feed the prepared batch into the
            // loss path — we only read the raw forward output.
            let host_data = self.state.prepare(chunk, n, 1.0, 1.0);
            let model = &self.optimiser.model;
            let device = model.device();
            let stream = device.new_stream().unwrap();
            let inputs = host_data.to_device(&device).unwrap();
            let outputs = model.make_forward_output_tensors(n).unwrap();
            model.forward(&stream, &inputs, &outputs).unwrap().value().unwrap();
            let output = outputs.get("outputs/output").unwrap().clone();
            let TValue::F32(values) = output.to_host().unwrap() else { panic!() };
            // For scalar-output value networks the tensor shape is (n, 1) and
            // `to_host()` already returns it row-major flattened, so we can
            // append directly.
            out.extend_from_slice(&values);
        }
        out
    }

    pub fn measure_max_cpu_throughput(
        &self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
    ) {
        let steps = schedule.steps;
        let threads = settings.threads;
        let wdl = schedule.wdl_scheduler.clone();
        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            self.state.score_drop_abs,
            dataloader.clone(),
        );

        let dataloader = ValueDataLoader { steps, threads, dataloader, wdl };

        self.0.measure_max_cpu_throughput(dataloader, steps).unwrap()
    }
}
