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
        // `usize::MAX` is the sentinel for "no cap" (set when `--superbatches`
        // is omitted). When start == end (the common case for save_rate=1
        // chunks), collapse to a single line; otherwise show the inclusive
        // range. Either way, one line replaces the old "Start / End" pair.
        let sb_label: String = if self.end_superbatch == usize::MAX {
            format!("{}.. (no cap)", self.start_superbatch)
        } else if self.start_superbatch == self.end_superbatch {
            self.start_superbatch.to_string()
        } else {
            format!("{}..={}", self.start_superbatch, self.end_superbatch)
        };
        println!("Superbatch             : {}", ansi(sb_label, 31));
    }
}

pub struct TrainingSchedule<'a> {
    pub steps: TrainingSteps,
    pub lr_schedule: Box<dyn Fn(usize, usize) -> f32 + 'a>,
    pub log_rate: usize,
    pub batch_queue_size: usize,
    pub delay_loss_readback: bool,
}
