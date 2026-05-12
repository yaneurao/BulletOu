use super::logger::ansi;

#[derive(Clone, Copy, Debug)]
pub struct TrainingSteps {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
}

impl TrainingSteps {
    pub fn display(&self) {
        println!("Batch Size             : {}", ansi(self.batch_size, 31));
        println!("Batches / Superbatch   : {}", ansi(self.batches_per_superbatch, 31));
        println!("Positions / Superbatch : {}", ansi(self.batches_per_superbatch * self.batch_size, 31));
        println!("Start Superbatch       : {}", ansi(self.start_superbatch, 31));
        // `usize::MAX` is the sentinel for "no cap" (set when `--superbatches`
        // is omitted). Showing the raw integer is just noise.
        let end_label: String = if self.end_superbatch == usize::MAX {
            "u64_max".to_string()
        } else {
            self.end_superbatch.to_string()
        };
        println!("End Superbatch         : {}", ansi(end_label, 31));
    }
}

pub struct TrainingSchedule<'a> {
    pub steps: TrainingSteps,
    pub lr_schedule: Box<dyn Fn(usize, usize) -> f32 + 'a>,
    pub log_rate: usize,
}
