use std::fmt::Debug;

use lr::LrScheduler;
use wdl::WdlScheduler;

pub mod lr;
pub mod wdl;

pub use bullet_trainer::run::{logger::ansi, schedule::TrainingSteps};

#[derive(Clone, Debug)]
pub struct TrainingSchedule<LR: LrScheduler, WDL: WdlScheduler> {
    pub net_id: String,
    pub eval_scale: f32,
    pub steps: TrainingSteps,
    pub wdl_scheduler: WDL,
    pub lr_scheduler: LR,
    pub save_rate: usize,
}

impl<LR: LrScheduler, WDL: WdlScheduler> TrainingSchedule<LR, WDL> {
    pub fn net_id(&self) -> String {
        self.net_id.clone()
    }

    pub fn should_save(&self, superbatch: usize) -> bool {
        superbatch.is_multiple_of(self.save_rate) || superbatch == self.steps.end_superbatch
    }

    pub fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        self.lr_scheduler.lr(batch, superbatch)
    }

    pub fn wdl(&self, batch: usize, superbatch: usize) -> f32 {
        self.wdl_scheduler.blend(batch, superbatch, self.steps.end_superbatch)
    }

    pub fn display(&self) {
        println!("Net Name               : {}", ansi(self.net_id.clone(), "32;1"));
        self.steps.display();
        println!("Eval Scale             : {}", ansi(format!("{:.0}", self.eval_scale), 31));
        println!("Save Rate              : {}", ansi(self.save_rate, 31));
        // YaneuraOu convention: target = λ × teacher_eval + (1−λ) × game_result.
        // The underlying scheduler stores 1 − λ (= WDL weight). Probe it at
        // superbatch 1 and the final superbatch — equal values collapse to a
        // single number, unequal values show the linear taper endpoints.
        let probe_end = self.steps.end_superbatch.min(1_000_000_000).max(1);
        let lambda_start = 1.0 - self.wdl_scheduler.blend(0, 1, probe_end);
        let lambda_end = 1.0 - self.wdl_scheduler.blend(0, probe_end, probe_end);
        let lambda_str = if (lambda_start - lambda_end).abs() < 1e-6 {
            format!("{:.3}", lambda_start)
        } else {
            format!("{:.3} -> {:.3}", lambda_start, lambda_end)
        };
        println!("Lambda                 : {}", ansi(lambda_str, 31));
        println!("LR Scheduler           : {}", self.lr_scheduler.colourful());
    }

    /// For evaluation passes, in order to ensure that we exhaust the test set at the
    /// same time as we exhaust the training set.
    pub fn steps_for_validation(&self, validation_freq: usize) -> TrainingSteps {
        let mut res = self.steps;

        res.batches_per_superbatch = self.steps.batches_per_superbatch / validation_freq;

        res
    }
}
