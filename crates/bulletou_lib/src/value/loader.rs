mod direct;
pub mod hcpe;
pub mod hcpe3;
mod montybinpack;
mod rng;
pub mod sfbinpack;
pub mod shogipack;
mod text;
pub mod viribinpack;

pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use hcpe::HcpeDataLoader;
pub use hcpe3::Hcpe3DataLoader;
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;
pub use shogipack::ShogiPackLoader;
pub use text::InMemoryTextLoader;
pub use viribinpack::{ViriBinpackLoader, ViriFilter};

use bulletformat::BulletFormat;
use rayon::prelude::*;
use std::{
    sync::{OnceLock, mpsc},
    thread,
};

use crate::game::{inputs::SparseInputType, outputs::OutputBuckets};

use super::Wgt;

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, f: F);
}

pub(crate) type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;

#[derive(Clone)]
pub struct DefaultDataLoader<I: SparseInputType, O, D> {
    input_getter: I,
    output_getter: O,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    use_win_rate_model: bool,
    wdl: bool,
    scale: f32,
    /// `Some(cap)` のとき `|score| >= cap` の局面を loss から除外（weight を 0 にする）。
    /// 特徴量デコードはそのまま走るが GPU 側で勾配寄与ゼロ。
    score_drop_abs: Option<u16>,
    /// `> 0` のとき、loader が返した局面を `batch_size * N` 件の
    /// window に貯め、window 内で deterministic Fisher-Yates shuffle してから
    /// batch に切る。seed + window index で再現可能。
    teacher_shuffle_buffer_batches: usize,
    teacher_shuffle_seed: u64,
    loader: D,
}

impl<I: SparseInputType, O, D> DefaultDataLoader<I, O, D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        scale: f32,
        score_drop_abs: Option<u16>,
        loader: D,
    ) -> Self {
        if use_win_rate_model && !wdl {
            initialise_win_rate_model_score_table();
        }
        Self {
            input_getter,
            output_getter,
            blend_getter,
            weight_getter,
            use_win_rate_model,
            wdl,
            scale,
            score_drop_abs,
            teacher_shuffle_buffer_batches: 0,
            teacher_shuffle_seed: 0,
            loader,
        }
    }

    pub fn with_teacher_shuffle(mut self, buffer_batches: usize, seed: u64) -> Self {
        self.teacher_shuffle_buffer_batches = buffer_batches;
        self.teacher_shuffle_seed = seed;
        self
    }
}

impl<I, O, D> DefaultDataLoader<I, O, D>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    pub fn load_and_map_batches<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        f: F,
    ) {
        self.load_and_map_batches_from_position(start_batch * batch_size, batch_size, f);
    }

    pub fn load_and_map_batches_from_position<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_position: usize,
        batch_size: usize,
        mut f: F,
    ) {
        if self.teacher_shuffle_buffer_batches > 0 {
            self.load_and_map_shuffled_batches_from_position(start_position, batch_size, f);
            return;
        }

        let mut incomplete_buf = Vec::new();

        self.loader.map_chunks(start_position, |chunk| {
            let remainder = if !incomplete_buf.is_empty() {
                let remainder = batch_size - incomplete_buf.len();

                if chunk.len() >= remainder {
                    incomplete_buf.extend_from_slice(&chunk[..remainder]);
                    let should_break = f(&incomplete_buf);
                    incomplete_buf.clear();

                    if should_break {
                        return true;
                    }
                } else {
                    incomplete_buf.extend_from_slice(chunk);
                }

                remainder
            } else {
                0
            };

            if chunk.len() >= remainder {
                let chunks = chunk[remainder..chunk.len()].chunks_exact(batch_size);
                incomplete_buf.extend_from_slice(chunks.remainder());

                for batch in chunks {
                    let should_break = f(batch);

                    if should_break {
                        return true;
                    }
                }
            }

            false
        });
    }

    fn load_and_map_shuffled_batches_from_position<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_position: usize,
        batch_size: usize,
        f: F,
    ) {
        load_and_map_shuffled_batches_with_prefetch(
            self.loader.clone(),
            start_position,
            batch_size,
            self.teacher_shuffle_buffer_batches,
            self.teacher_shuffle_seed,
            f,
        );
    }

    pub fn prepare(&self, data: &[I::RequiredDataType], threads: usize, blend: f32) -> PreparedData<I, O> {
        PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
            self.score_drop_abs,
        )
    }

    pub fn prepare_with_pool(
        &self,
        data: &[I::RequiredDataType],
        pool: &rayon::ThreadPool,
        threads: usize,
        blend: f32,
    ) -> PreparedData<I, O> {
        PreparedData::new_with_pool(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
            self.score_drop_abs,
            Some(pool),
        )
    }
}

/// A batch of data, in the correct format for the GPU.
pub struct PreparedData<I: SparseInputType, O> {
    pub(crate) input_getter: I,
    pub(crate) output_getter: O,
    pub(crate) batch_size: usize,
    pub(crate) stm: Vec<i32>,
    pub(crate) nstm: Vec<i32>,
    pub(crate) buckets: Vec<i32>,
    pub(crate) targets: Vec<f32>,
    pub(crate) weights: Vec<f32>,
    /// HandCount dense auxiliary input。`I::hand_count_dims()` が `Some` のとき
    /// `Some(hand_count_dim * batch_size)` 長の flat Vec を保持する (列方向 = batch index)。
    /// 次元数 (`hand_count_dim`) はここでは持たず、consumer (model 定義側) が
    /// `input_getter.hand_count_dims()` から取得する前提。
    pub(crate) hand_count: Option<Vec<f32>>,
}

impl<I, O> PreparedData<I, O>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
        score_drop_abs: Option<u16>,
    ) -> Self {
        Self::new_with_pool(
            input_getter,
            output_getter,
            blend_getter,
            weight_getter,
            use_win_rate_model,
            wdl,
            data,
            threads,
            blend,
            scale,
            score_drop_abs,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_pool(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
        score_drop_abs: Option<u16>,
        pool: Option<&rayon::ThreadPool>,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active();
        let chunk_size = batch_size.div_ceil(threads);
        let input_size = input_getter.num_inputs();
        let output_size = if wdl { 3 } else { 1 };
        let sparse_size = max_active * batch_size;
        let hand_count_dims = input_getter.hand_count_dims();
        let hand_count_dim = hand_count_dims.unwrap_or(0);

        let hand_count_init = hand_count_dims.map(|dims| vec![0.0; dims * batch_size]);

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: vec![-1; sparse_size],
            nstm: vec![-1; sparse_size],
            buckets: vec![0; batch_size],
            targets: vec![0.0; output_size * batch_size],
            weights: vec![0.0; batch_size],
            hand_count: hand_count_init,
        };

        let sparse_chunk_size = max_active * chunk_size;

        if hand_count_dim == 0
            && let Some(pool) = pool
            && threads > 1
            && batch_size > 1
        {
            pool.install(|| {
                data.par_chunks(chunk_size)
                    .zip(prep.stm.par_chunks_mut(sparse_chunk_size))
                    .zip(prep.nstm.par_chunks_mut(sparse_chunk_size))
                    .zip(prep.buckets.par_chunks_mut(chunk_size))
                    .zip(prep.targets.par_chunks_mut(output_size * chunk_size))
                    .zip(prep.weights.par_chunks_mut(chunk_size))
                    .for_each(
                        |(((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk)| {
                            let inp = &prep.input_getter;
                            let out = &prep.output_getter;

                            for i in 0..data_chunk.len() {
                                let pos = &data_chunk[i];
                                let mut j_stm: usize = 0;
                                let mut j_nstm: usize = 0;
                                let sparse_offset = max_active * i;

                                inp.map_features_split(pos, |our_opt, opp_opt| {
                                    if let Some(our) = our_opt {
                                        assert!(our < input_size, "STM feature index exceeded input size!");
                                        stm_chunk[sparse_offset + j_stm] = our as i32;
                                        j_stm += 1;
                                    }
                                    if let Some(opp) = opp_opt {
                                        assert!(opp < input_size, "NSTM feature index exceeded input size!");
                                        nstm_chunk[sparse_offset + j_nstm] = opp as i32;
                                        j_nstm += 1;
                                    }
                                });

                                assert!(
                                    j_stm <= max_active && j_nstm <= max_active,
                                    "More inputs provided than the specified maximum!"
                                );

                                if O::BUCKETS > 1 {
                                    buckets_chunk[i] = out.bucket(pos) as i32;
                                }
                                let mut weight = weight_getter.map_or(1.0, |w| w(pos));
                                if let Some(cap) = score_drop_abs {
                                    if pos.score().unsigned_abs() >= cap {
                                        weight = 0.0;
                                    }
                                }
                                weights_chunk[i] = weight;

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = if use_win_rate_model {
                                        win_rate_model_score(pos.score())
                                    } else {
                                        let score = f32::from(pos.score());
                                        sigmoid(rscale * score)
                                    };
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    let blend = blend_getter(pos, blend);
                                    assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }
                        },
                    );
            });

            return prep;
        }

        // HandCount 用の並列チャンクを事前に materialise。Option は並列ループ内で扱う。
        let hand_count_chunk_size = hand_count_dim * chunk_size;
        let num_chunks = batch_size.div_ceil(chunk_size);
        let hand_count_slices: Vec<Option<&mut [f32]>> = if let Some(hc) = prep.hand_count.as_mut() {
            hc.chunks_mut(hand_count_chunk_size).map(Some).collect()
        } else {
            (0..num_chunks).map(|_| None).collect()
        };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.chunks_mut(sparse_chunk_size))
                .zip(prep.nstm.chunks_mut(sparse_chunk_size))
                .zip(prep.buckets.chunks_mut(chunk_size))
                .zip(prep.targets.chunks_mut(output_size * chunk_size))
                .zip(prep.weights.chunks_mut(chunk_size))
                .zip(hand_count_slices)
                .for_each(
                    |(
                        (((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk),
                        hand_count_chunk,
                    )| {
                        let inp = &prep.input_getter;
                        let out = &prep.output_getter;
                        s.spawn(move || {
                            let chunk_len = data_chunk.len();
                            let mut hand_count_chunk = hand_count_chunk;

                            for i in 0..chunk_len {
                                let pos = &data_chunk[i];

                                if let Some(hc_slice) = hand_count_chunk.as_deref_mut() {
                                    let offset = hand_count_dim * i;
                                    let end = offset + hand_count_dim;
                                    // 事前に 0 で埋め済み。fill_hand_count は
                                    // 書き込みのみで読まないので再初期化は不要。
                                    inp.fill_hand_count(pos, &mut hc_slice[offset..end]);
                                }
                                // STM と NSTM は独立カウンタで管理: 非対称 feature
                                // (HandThreat defensive 等) で |STM_active| != |NSTM_active|
                                // を許可するため。symmetric な input type は
                                // map_features_split の default impl 経由で
                                // 両側同時に進むので従来挙動と一致する。
                                let mut j_stm: usize = 0;
                                let mut j_nstm: usize = 0;
                                let sparse_offset = max_active * i;

                                inp.map_features_split(pos, |our_opt, opp_opt| {
                                    if let Some(our) = our_opt {
                                        assert!(our < input_size, "STM feature index exceeded input size!");
                                        stm_chunk[sparse_offset + j_stm] = our as i32;
                                        j_stm += 1;
                                    }
                                    if let Some(opp) = opp_opt {
                                        assert!(opp < input_size, "NSTM feature index exceeded input size!");
                                        nstm_chunk[sparse_offset + j_nstm] = opp as i32;
                                        j_nstm += 1;
                                    }
                                });

                                assert!(
                                    j_stm <= max_active && j_nstm <= max_active,
                                    "More inputs provided than the specified maximum!"
                                );

                                if O::BUCKETS > 1 {
                                    buckets_chunk[i] = out.bucket(pos) as i32;
                                }
                                let mut weight = weight_getter.map_or(1.0, |w| w(pos));
                                if let Some(cap) = score_drop_abs {
                                    if pos.score().unsigned_abs() >= cap {
                                        weight = 0.0;
                                    }
                                }
                                weights_chunk[i] = weight;

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = if use_win_rate_model {
                                        win_rate_model_score(pos.score())
                                    } else {
                                        let score = f32::from(pos.score());
                                        sigmoid(rscale * score)
                                    };
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    let blend = blend_getter(pos, blend);
                                    assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }
                        });
                    },
                );
        });

        prep
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

pub(crate) fn teacher_shuffle_window_records(batch_size: usize, buffer_batches: usize) -> Option<usize> {
    if buffer_batches == 0 || batch_size == 0 {
        return None;
    }
    batch_size.checked_mul(buffer_batches)
}

pub(crate) const TEACHER_SHUFFLE_PREFETCH_BUFFERS: usize = 2;

pub(crate) fn load_and_map_shuffled_batches_with_prefetch<T, D, F>(
    loader: D,
    start_position: usize,
    batch_size: usize,
    shuffle_buffer_batches: usize,
    shuffle_seed: u64,
    mut f: F,
) where
    T: Copy + Send,
    D: DataLoader<T>,
    F: FnMut(&[T]) -> bool,
{
    let Some(window_records) = teacher_shuffle_window_records(batch_size, shuffle_buffer_batches) else {
        return;
    };

    thread::scope(|scope| {
        // Capacity 1 is intentional: while the consumer owns one shuffled window,
        // the producer can fill/shuffle exactly one more window in the background.
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Vec<T>>(TEACHER_SHUFFLE_PREFETCH_BUFFERS - 1);
        let (empty_tx, empty_rx) = mpsc::sync_channel::<Vec<T>>(TEACHER_SHUFFLE_PREFETCH_BUFFERS);
        for _ in 0..TEACHER_SHUFFLE_PREFETCH_BUFFERS {
            empty_tx
                .send(Vec::with_capacity(window_records))
                .expect("teacher shuffle empty-buffer channel is live during setup");
        }

        let producer = scope.spawn(move || {
            produce_shuffled_teacher_windows(
                loader,
                start_position,
                batch_size,
                window_records,
                shuffle_seed,
                ready_tx,
                empty_rx,
            );
        });

        let mut stopped = false;
        while let Ok(mut window) = ready_rx.recv() {
            for batch in window.chunks_exact(batch_size) {
                if f(batch) {
                    stopped = true;
                    break;
                }
            }
            if stopped {
                break;
            }
            window.clear();
            if empty_tx.send(window).is_err() {
                break;
            }
        }

        drop(ready_rx);
        drop(empty_tx);
        if let Err(payload) = producer.join() {
            std::panic::resume_unwind(payload);
        }
    });
}

fn produce_shuffled_teacher_windows<T, D>(
    loader: D,
    start_position: usize,
    batch_size: usize,
    window_records: usize,
    shuffle_seed: u64,
    ready_tx: mpsc::SyncSender<Vec<T>>,
    empty_rx: mpsc::Receiver<Vec<T>>,
) where
    T: Copy,
    D: DataLoader<T>,
{
    let mut incomplete_buf = Vec::new();
    let mut shuffle_buf = match empty_rx.recv() {
        Ok(mut buf) => {
            buf.clear();
            buf
        }
        Err(_) => return,
    };
    let mut window_index = start_position / window_records;
    let mut stopped = false;

    loader.map_chunks(start_position, |chunk| {
        let mut emit_complete_batch = |batch: &[T]| -> bool {
            shuffle_buf.extend_from_slice(batch);
            if shuffle_buf.len() >= window_records {
                debug_assert_eq!(shuffle_buf.len(), window_records);
                return send_and_reuse_teacher_shuffle_window(
                    &mut shuffle_buf,
                    &ready_tx,
                    &empty_rx,
                    shuffle_seed,
                    &mut window_index,
                );
            }
            false
        };

        let remainder = if !incomplete_buf.is_empty() {
            let remainder = batch_size - incomplete_buf.len();

            if chunk.len() >= remainder {
                incomplete_buf.extend_from_slice(&chunk[..remainder]);
                if emit_complete_batch(&incomplete_buf) {
                    incomplete_buf.clear();
                    stopped = true;
                    return true;
                }
                incomplete_buf.clear();
            } else {
                incomplete_buf.extend_from_slice(chunk);
            }

            remainder
        } else {
            0
        };

        if chunk.len() >= remainder {
            let chunks = chunk[remainder..chunk.len()].chunks_exact(batch_size);
            incomplete_buf.extend_from_slice(chunks.remainder());

            for batch in chunks {
                if emit_complete_batch(batch) {
                    stopped = true;
                    return true;
                }
            }
        }

        false
    });

    if !stopped && !shuffle_buf.is_empty() {
        send_final_teacher_shuffle_window(shuffle_buf, &ready_tx, shuffle_seed, window_index);
    }
}

fn send_and_reuse_teacher_shuffle_window<T>(
    shuffle_buf: &mut Vec<T>,
    ready_tx: &mpsc::SyncSender<Vec<T>>,
    empty_rx: &mpsc::Receiver<Vec<T>>,
    shuffle_seed: u64,
    window_index: &mut usize,
) -> bool {
    let mut full_window = Vec::new();
    std::mem::swap(shuffle_buf, &mut full_window);
    shuffle_teacher_buffer(&mut full_window, shuffle_seed, *window_index);
    if ready_tx.send(full_window).is_err() {
        return true;
    }
    *window_index = window_index.saturating_add(1);

    match empty_rx.recv() {
        Ok(mut next) => {
            next.clear();
            *shuffle_buf = next;
            false
        }
        Err(_) => true,
    }
}

fn send_final_teacher_shuffle_window<T>(
    mut shuffle_buf: Vec<T>,
    ready_tx: &mpsc::SyncSender<Vec<T>>,
    shuffle_seed: u64,
    window_index: usize,
) {
    shuffle_teacher_buffer(&mut shuffle_buf, shuffle_seed, window_index);
    let _ = ready_tx.send(shuffle_buf);
}

pub(crate) fn shuffle_teacher_buffer<T>(data: &mut [T], seed: u64, window_index: usize) {
    if data.len() <= 1 {
        return;
    }
    let mut state = seed ^ (window_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for i in (1..data.len()).rev() {
        let j = (next_shuffle_u64(&mut state) as usize) % (i + 1);
        data.swap(i, j);
    }
}

fn next_shuffle_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64(*state)
}

fn splitmix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

static WIN_RATE_MODEL_SCORE_TABLE: OnceLock<Box<[f32]>> = OnceLock::new();

pub(crate) fn initialise_win_rate_model_score_table() -> &'static [f32] {
    WIN_RATE_MODEL_SCORE_TABLE.get_or_init(|| {
        let mut values = Vec::with_capacity(usize::from(u16::MAX) + 1);
        for raw_score in i32::from(i16::MIN)..=i32::from(i16::MAX) {
            let score = raw_score as f32;
            let p = (score - 270.0) / 380.0;
            let pm = (-score - 270.0) / 380.0;
            values.push(0.5 * (1.0 + sigmoid(p) - sigmoid(pm)));
        }
        values.into_boxed_slice()
    })
}

pub(crate) fn win_rate_model_score(score: i16) -> f32 {
    let table = initialise_win_rate_model_score_table();
    let index = (i32::from(score) - i32::from(i16::MIN)) as usize;
    table[index]
}

#[cfg(test)]
mod tests {
    use crate::{
        game::{inputs::SparseInputType, outputs::OutputBuckets},
        value::loader::{DataLoader, GameResult, LoadableDataType, PreparedData},
    };

    #[derive(Clone, Copy)]
    struct TinyPos {
        a: usize,
        b: usize,
        score: i16,
        result: GameResult,
    }

    impl LoadableDataType for TinyPos {
        fn score(&self) -> i16 {
            self.score
        }

        fn result(&self) -> GameResult {
            self.result
        }
    }

    #[derive(Clone)]
    struct TinyInput;

    impl SparseInputType for TinyInput {
        type RequiredDataType = TinyPos;

        fn num_inputs(&self) -> usize {
            8
        }

        fn max_active(&self) -> usize {
            2
        }

        fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
            f(pos.a, pos.b);
            f((pos.a + 1) % 8, (pos.b + 1) % 8);
        }

        fn shorthand(&self) -> String {
            "tiny".to_string()
        }

        fn description(&self) -> String {
            "Tiny test input".to_string()
        }
    }

    #[derive(Clone, Copy, Default)]
    struct TinyBuckets;

    impl OutputBuckets<TinyPos> for TinyBuckets {
        const BUCKETS: usize = 3;

        fn bucket(&self, pos: &TinyPos) -> usize {
            pos.a % 3
        }
    }

    #[derive(Clone)]
    struct VecLoader<T> {
        data: Vec<T>,
        chunk_size: usize,
        paths: Vec<String>,
    }

    impl<T> VecLoader<T> {
        fn new(data: Vec<T>, chunk_size: usize) -> Self {
            Self { data, chunk_size, paths: vec!["memory".to_string()] }
        }
    }

    impl<T: Copy + Send + Sync + 'static> DataLoader<T> for VecLoader<T> {
        fn data_file_paths(&self) -> &[String] {
            &self.paths
        }

        fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, mut f: F) {
            for chunk in self.data[start_position..].chunks(self.chunk_size) {
                if f(chunk) {
                    break;
                }
            }
        }
    }

    #[test]
    fn teacher_shuffle_is_reproducible_and_preserves_records() {
        let mut a: Vec<u32> = (0..256).collect();
        let mut b = a.clone();
        let original = a.clone();

        super::shuffle_teacher_buffer(&mut a, 1234, 5);
        super::shuffle_teacher_buffer(&mut b, 1234, 5);

        assert_eq!(a, b);
        assert_ne!(a, original);
        a.sort_unstable();
        assert_eq!(a, original);
    }

    #[test]
    fn teacher_shuffle_window_records_uses_batch_multiple() {
        assert_eq!(super::teacher_shuffle_window_records(65_536, 61), Some(3_997_696));
        assert_eq!(super::teacher_shuffle_window_records(65_536, 0), None);
    }

    #[test]
    fn teacher_shuffle_prefetch_matches_window_shuffle_order() {
        let data: Vec<u32> = (0..24).collect();
        let mut batches = Vec::new();

        super::load_and_map_shuffled_batches_with_prefetch(VecLoader::new(data.clone(), 5), 0, 4, 2, 99, |batch| {
            batches.push(batch.to_vec());
            false
        });

        assert!(batches.iter().all(|batch| batch.len() == 4));
        let flattened: Vec<_> = batches.into_iter().flatten().collect();
        let mut expected = data;
        for (window_index, window) in expected.chunks_mut(8).enumerate() {
            super::shuffle_teacher_buffer(window, 99, window_index);
        }
        assert_eq!(flattened, expected);
    }

    #[test]
    fn teacher_shuffle_prefetch_stops_without_deadlock() {
        let data: Vec<u32> = (0..64).collect();
        let mut seen_batches = 0usize;

        super::load_and_map_shuffled_batches_with_prefetch(VecLoader::new(data, 3), 0, 4, 2, 7, |batch| {
            assert_eq!(batch.len(), 4);
            seen_batches += 1;
            seen_batches == 3
        });

        assert_eq!(seen_batches, 3);
    }

    #[test]
    fn prepare_with_pool_matches_scoped_prepare() {
        let data: Vec<_> = (0..16)
            .map(|i| TinyPos {
                a: i % 7,
                b: (i * 3) % 7,
                score: (i as i16) * 10 - 80,
                result: [GameResult::Loss, GameResult::Draw, GameResult::Win][i % 3],
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();

        let baseline = PreparedData::new(
            TinyInput,
            TinyBuckets,
            (|_, blend| blend) as fn(&TinyPos, f32) -> f32,
            None,
            true,
            false,
            &data,
            4,
            0.0,
            400.0,
            None,
        );
        let pooled = PreparedData::new_with_pool(
            TinyInput,
            TinyBuckets,
            (|_, blend| blend) as fn(&TinyPos, f32) -> f32,
            None,
            true,
            false,
            &data,
            4,
            0.0,
            400.0,
            None,
            Some(&pool),
        );

        assert_eq!(baseline.stm, pooled.stm);
        assert_eq!(baseline.nstm, pooled.nstm);
        assert_eq!(baseline.buckets, pooled.buckets);
        assert_eq!(baseline.targets, pooled.targets);
        assert_eq!(baseline.weights, pooled.weights);
        assert_eq!(baseline.hand_count, pooled.hand_count);
    }
}
