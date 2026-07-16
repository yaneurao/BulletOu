pub mod dataloader;
pub mod logger;
pub mod schedule;

use std::{
    sync::mpsc::{self, TryRecvError},
    thread,
    time::Instant,
};

use bullet_compiler::tensor::{DValue, TValue};
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    runtime::Gpu,
};

use crate::{
    DataLoadingError, Trainer, TrainerError,
    optimiser::OptimiserState,
    run::{
        dataloader::{DataLoader, PreparedBatchHost},
        schedule::{TrainingSchedule, TrainingSteps},
    },
};

struct PendingLoss<G: Gpu> {
    copy: SyncOnValue<G, TValue>,
    superbatch: usize,
    curr_batch: usize,
    batch_size: usize,
}

fn finish_loss<G, O, S, B, C>(
    trainer: &mut Trainer<G, O, S>,
    pending: PendingLoss<G>,
    steps: TrainingSteps,
    log_rate: usize,
    timer: &Instant,
    superbatch_timer: &mut Instant,
    running_loss: &mut f32,
    superbatch_positions: &mut usize,
    batch_callback: &mut B,
    superbatch_callback: &mut C,
) -> Result<(), TrainerError<G>>
where
    G: Gpu,
    O: OptimiserState<G>,
    B: FnMut(&mut Trainer<G, O, S>, usize, usize, f32),
    C: FnMut(&mut Trainer<G, O, S>, usize),
{
    let TValue::F32(loss) = pending
        .copy
        .value()
        .map_err(TrainerError::Unexpected)?
    else {
        panic!()
    };
    let [loss] = loss[..] else { panic!() };
    let error = loss / pending.batch_size as f32;

    *running_loss += error;
    *superbatch_positions += pending.batch_size;

    if pending.curr_batch % log_rate == 0 {
        logger::report_superbatch_progress(
            pending.superbatch,
            steps.batches_per_superbatch,
            pending.curr_batch,
            superbatch_timer,
            *superbatch_positions,
        );
    }

    let completed_batch = pending.curr_batch + 1;
    batch_callback(trainer, pending.superbatch, completed_batch, error);

    if completed_batch % steps.batches_per_superbatch == 0 {
        let error = *running_loss / steps.batches_per_superbatch as f32;
        *running_loss = 0.0;

        let total_time = timer.elapsed().as_secs_f32();
        let sb_time = superbatch_timer.elapsed().as_secs_f32();

        logger::report_superbatch_finished(pending.superbatch, error, sb_time, total_time, *superbatch_positions);
        logger::report_time_left(steps, pending.superbatch, total_time);

        superbatch_callback(trainer, pending.superbatch);

        *superbatch_positions = 0;
        *superbatch_timer = Instant::now();
    }

    Ok(())
}

pub fn train_custom<G: Gpu, O: OptimiserState<G>, S>(
    trainer: &mut Trainer<G, O, S>,
    schedule: TrainingSchedule,
    dataloader: impl DataLoader,
    mut batch_callback: impl FnMut(&mut Trainer<G, O, S>, usize, usize, f32),
    mut superbatch_callback: impl FnMut(&mut Trainer<G, O, S>, usize),
) -> Result<(), TrainerError<G>> {
    trainer.optimiser.model.set_bwd_batch_size(schedule.steps.batch_size).map_err(TrainerError::Unexpected)?;

    let model = &trainer.optimiser.model;
    let device = model.device();
    let props = device.props();

    logger::clear_colours();
    println!(
        "{}",
        logger::ansi(format!("Training on {} ({})", props.name(), props.arch().unwrap_or("unknown")), "34;1")
    );

    let timer = Instant::now();
    let batch_queue_size = schedule.batch_queue_size.max(1);
    let lr = schedule.lr_schedule;
    let steps = schedule.steps;

    let (sender, receiver) = mpsc::sync_channel::<PreparedBatchHost>(batch_queue_size);

    let dataloader = thread::spawn(move || {
        let mut batch_no = 0;
        let mut superbatch = steps.start_superbatch;

        dataloader.map_batches(steps.batch_size, |batch| {
            if batch.batch_size != steps.batch_size {
                panic!("Dataloader returned a batch with incorrect batch size!");
            }

            // メインスレッドが既に学習を終えて receiver を drop している場合、
            // sender.send は Err を返す。これはレース状況下の正常な終端なので
            // unwrap で panic させず黙って抜ける。
            if sender.send(batch).is_err() {
                return true;
            }

            batch_no += 1;

            if batch_no % steps.batches_per_superbatch == 0 {
                batch_no = 0;
                superbatch += 1;

                if superbatch > steps.end_superbatch {
                    return true;
                }
            }

            false
        })
    });

    let mut prev_lr = lr(0, 1);
    let mut superbatch = steps.start_superbatch;
    let mut curr_batch = 0;
    let mut superbatch_timer = Instant::now();
    let mut running_loss = 0.0;
    let mut superbatch_positions = 0;

    let first_batch =
        receiver.recv().map_err(|_| TrainerError::DataLoadingError(DataLoadingError::NoBatchesReceived))?;

    let copy_stream = device.new_stream().map_err(TrainerError::Unexpected)?;
    let compute_stream = device.new_stream().map_err(TrainerError::Unexpected)?;
    let loss_stream = device.new_stream().map_err(TrainerError::Unexpected)?;

    let outputs = [
        model.make_backward_output_tensors().map_err(TrainerError::Unexpected)?,
        model.make_backward_output_tensors().map_err(TrainerError::Unexpected)?,
    ];
    let losses = [
        outputs[0]
            .get("outputs/loss")
            .expect("`Trainer` must have a \"loss\" output!")
            .clone(),
        outputs[1]
            .get("outputs/loss")
            .expect("`Trainer` must have a \"loss\" output!")
            .clone(),
    ];
    let gradients = model.make_gradient_tensors().map_err(TrainerError::Unexpected)?;
    let tlr = Buffer::from_host(&device, &TValue::F32(vec![0.0])).map_err(TrainerError::Unexpected)?;
    let tgf = Buffer::from_host(&device, &TValue::F32(vec![0.0])).map_err(TrainerError::Unexpected)?;

    let mut next_batch_size = first_batch.batch_size;
    let mut batch_on_device = first_batch.to_device(&device).map_err(TrainerError::Unexpected)?;

    let mut next_on_device = batch_on_device
        .iter()
        .map(|(id, tensor)| {
            // This buffer is only read after a full H2D copy from the next host batch.
            let buf = unsafe { Buffer::uninit(&device, tensor.dtype(), tensor.size()) };
            (id.clone(), buf.unwrap())
        })
        .collect();

    let mut batch_queued = true;
    let mut output_slot = 0usize;
    let mut pending_loss: Option<PendingLoss<G>> = None;
    let mut lr_value = TValue::F32(vec![0.0]);
    let mut gradient_factor_value = TValue::F32(vec![0.0]);

    while batch_queued {
        if superbatch > steps.end_superbatch {
            // 学習は end_superbatch まで正常完了しており、その先に届く batch は
            // double-buffer 化された dataloader の先読みによる余分。Err にして
            // caller を panic させる必要はないので正常終了として break する。
            break;
        }

        // ignore startup time from loading the first batch of data
        // because it just poisons the reported pos/sec when reading
        // from binpacked data
        if superbatch == steps.start_superbatch && curr_batch == 0 {
            superbatch_timer = Instant::now();
        }

        let lrate = lr(curr_batch, superbatch);
        let this_batch_size = next_batch_size;
        let batch_superbatch = superbatch;
        let batch_curr_batch = curr_batch;
        let batch_ends_superbatch = batch_curr_batch + 1 == steps.batches_per_superbatch;
        let mut dataloader_exhausted = false;
        let mut early_next_batch = None;
        let mut early_next_copy = None;

        lr_value.write(0, DValue::F32(lrate));
        let lrdrop = tlr.copy_from_host_async(&copy_stream, &lr_value).map_err(TrainerError::Unexpected)?;
        gradient_factor_value.write(0, DValue::F32(1.0 / this_batch_size as f32));
        let gfdrop = tgf
            .copy_from_host_async(&copy_stream, &gradient_factor_value)
            .map_err(TrainerError::Unexpected)?;

        if curr_batch == 0 {
            if lrate < prev_lr {
                println!("LR dropped to {}", logger::ansi(lrate, logger::num_cs()));
            } else if lrate > prev_lr {
                println!("LR increased to {}", logger::ansi(lrate, logger::num_cs()));
            }
        }

        prev_lr = lrate;

        match receiver.try_recv() {
            Ok(next_batch_host) => {
                next_batch_size = next_batch_host.batch_size;
                early_next_batch = Some(next_batch_host);
                early_next_copy = Some(
                    early_next_batch
                        .as_ref()
                        .unwrap()
                        .copy_to_device_async(&copy_stream, &next_on_device)
                        .map_err(TrainerError::Unexpected)?,
                );
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                dataloader_exhausted = true;
            }
        }

        let compute_block1 = trainer
            .optimiser
            .model
            .backward(&compute_stream, &batch_on_device, &outputs[output_slot], &gradients)
            .map_err(TrainerError::GradientCalculationError)?;

        let compute_block1 = unsafe { compute_block1.detach_value() };

        let mut scalar_upload = unsafe { lrdrop.detach_value() };
        scalar_upload
            .merge(unsafe { gfdrop.detach_value() })
            .map_err(TrainerError::Unexpected)?;
        scalar_upload.sync().map_err(TrainerError::Unexpected)?;

        let compute_block2 = trainer
            .optimiser
            .update(&compute_stream, tgf.clone(), tlr.clone(), &gradients)
            .map_err(TrainerError::OptimiserUpdateError)?;

        if let Some(next_copy) = early_next_copy {
            compute_block1.sync().map_err(TrainerError::Unexpected)?;
            compute_block2.sync().map_err(TrainerError::Unexpected)?;

            // The copy was enqueued before current batch compute, so this
            // drop usually only waits for the tail of H2D copy, if any.
            drop(next_copy);
            drop(early_next_batch);
            std::mem::swap(&mut batch_on_device, &mut next_on_device);
        } else if dataloader_exhausted {
            batch_queued = false;
            compute_block1.sync().map_err(TrainerError::Unexpected)?;
            compute_block2.sync().map_err(TrainerError::Unexpected)?;
        } else if let Ok(next_batch_host) = receiver.recv() {
            next_batch_size = next_batch_host.batch_size;
            let next_copy = next_batch_host
                .copy_to_device_async(&copy_stream, &next_on_device)
                .map_err(TrainerError::Unexpected)?;

            compute_block1.sync().map_err(TrainerError::Unexpected)?;
            compute_block2.sync().map_err(TrainerError::Unexpected)?;

            // `next_copy` borrows `next_batch_host`, so keeping it alive until
            // here preserves the host buffers while H2D copy overlaps the
            // current batch's compute. Dropping it syncs the copy stream.
            drop(next_copy);
            std::mem::swap(&mut batch_on_device, &mut next_on_device);
        } else {
            batch_queued = false;
            compute_block1.sync().map_err(TrainerError::Unexpected)?;
            compute_block2.sync().map_err(TrainerError::Unexpected)?;
        }

        if let Some(pending) = pending_loss.take() {
            finish_loss(
                trainer,
                pending,
                steps,
                schedule.log_rate,
                &timer,
                &mut superbatch_timer,
                &mut running_loss,
                &mut superbatch_positions,
                &mut batch_callback,
                &mut superbatch_callback,
            )?;
        }

        let current_loss = losses[output_slot]
            .to_host_async(&loss_stream)
            .map_err(TrainerError::Unexpected)?;
        let current_loss = PendingLoss {
            copy: current_loss,
            superbatch: batch_superbatch,
            curr_batch: batch_curr_batch,
            batch_size: this_batch_size,
        };

        curr_batch += 1;
        if curr_batch % steps.batches_per_superbatch == 0 {
            superbatch += 1;
            curr_batch = 0;
        }

        // Delay only ordinary batches. The final batch of a superbatch must be
        // completed immediately because `superbatch_callback` may save the
        // model, and that save must observe exactly the weights after this
        // superbatch's last update, not after the next batch.
        if schedule.delay_loss_readback && batch_queued && !batch_ends_superbatch {
            pending_loss = Some(current_loss);
            output_slot ^= 1;
        } else {
            finish_loss(
                trainer,
                current_loss,
                steps,
                schedule.log_rate,
                &timer,
                &mut superbatch_timer,
                &mut running_loss,
                &mut superbatch_positions,
                &mut batch_callback,
                &mut superbatch_callback,
            )?;
            if schedule.delay_loss_readback {
                output_slot ^= 1;
            }
        }
    }

    let total_time = timer.elapsed().as_secs();
    let (hours, minutes, seconds) = logger::seconds_to_hms(total_time as u32);

    println!(
        "Total Training Time: {}h {}m {}s",
        logger::ansi(hours, logger::num_cs()),
        logger::ansi(minutes, logger::num_cs()),
        logger::ansi(seconds, logger::num_cs()),
    );

    // dataloader が `sender.send` でブロックしたままだと join がデッドロックする。
    // (sync_channel(32) のバッファが満杯で、main loop は break で抜けたが
    //  receiver はまだスコープに残っているので sender 側は永久に待ち続ける。)
    // receiver を明示的に drop して sender.send を Err にし、dataloader 側の
    // `if sender.send(batch).is_err() { return true; }` を発動させてから join。
    drop(receiver);
    dataloader.join().unwrap()?;

    Ok(())
}
