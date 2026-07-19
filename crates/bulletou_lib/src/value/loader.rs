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
use std::sync::OnceLock;

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
            loader,
        }
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
                                    buckets_chunk[i] = i32::from(out.bucket(pos));
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
                                    buckets_chunk[i] = i32::from(out.bucket(pos));
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
        value::loader::{GameResult, LoadableDataType, PreparedData},
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

        fn bucket(&self, pos: &TinyPos) -> u8 {
            (pos.a % 3) as u8
        }
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
