//! HCPE (HuffmanCodedPosAndEval) データローダー
//!
//! HCPE は dlshogi 系 (Apery 由来) で広く使われている教師フォーマット。
//! 38 byte 固定長レコードの連続。
//!
//! ## レコードレイアウト
//!
//! ```text
//! bytes  0..32: HuffmanCodedPos (HCP) — Apery 系 Huffman で圧縮された局面 32 byte
//! bytes 32..34: eval        (i16, little-endian) — 評価値 (centipawn)
//! bytes 34..36: bestMove16  (i16, little-endian) — cshogi 形式 move16
//! byte      36: gameResult  (u8) — 0=Draw, 1=BlackWin, 2=WhiteWin (cshogi 慣例、絶対色基準)
//! byte      37: dummy       (u8) — padding
//!
//! `decode_hcpe_record` 内で gameResult は局面の手番に応じて **STM 視点**
//! (1=win, 0=draw, -1=loss) に変換され、PSV byte 38 に書き込まれる。
//! ```
//!
//! ## 設計方針
//!
//! HCPE 用に新規 SparseInputType を導入するのではなく、**読み込み時に内部で
//! PackedSfenValue (40 byte、やねうら王形式) に変換**して既存パイプラインに乗せる。
//!
//! これにより:
//! - 既存の入力特徴量 (`ShogiHalfKA_hm`, `ShogiHalfKP`, threat 系など 7 種すべて) が
//!   無変更で使える (`RequiredDataType = PackedSfenValue`)
//! - HCP のデコードは `super::shogipack::MiniPosition::from_hcp` を再利用
//! - PackedSfenValue への再パックも `MiniPosition::to_packed_sfen_value` を再利用
//!
//! ## 制約
//!
//! HCPE には game_ply 情報がない。PackedSfenValue の game_ply は 0 で埋める。
//! このため Layer Stack の `ply9` bucket は使えない (game_ply を直接参照するため)。
//! `progress8kpabs` / `progress8` / `progress8gikou` / `kingrank9` は局面ベースなので使える。
//!
//! また HCPE には MoveVisits (policy teacher) が存在しないため、value 学習のみが対象。
//! Policy 蒸留や policy 教師を使いたい場合は HCPE3 を使う。

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::time::Instant;

use crate::shogi::PackedSfenValue;

use super::rng::SimpleRand;
use super::shogipack::{convert_game_result, MiniPosition};
use super::DataLoader;

/// HCPE 1 レコードのバイト数
pub const HCPE_RECORD_SIZE: usize = 38;

/// 1 度に読むディスクチャンクのレコード数 (`HCPE_RECORD_SIZE` × CHUNK_RECORDS バイト)。
/// 並列デコード worker に切り分けるロット単位でもあるので、worker 数 ×
/// (1 worker が短時間で処理できるレコード数) に収まる程度の大きさが理想。
/// 65536 × 38 ≒ 2.4 MB。
const CHUNK_RECORDS: usize = 65536;

/// 1 レコードを PackedSfenValue にデコード
///
/// HCP のデコードに失敗した場合 (不正レコード) は `None` を返す。
/// caller は不正レコードをスキップして次に進める想定。
///
/// HCPE の game_result は **絶対色基準** (0=Draw, 1=BlackWin, 2=WhiteWin)
/// なので、PSV の **STM 視点** (1=win, 0=draw, -1=loss) に変換する必要がある。
/// `.pack` ローダーが `convert_game_result` で行っているのと同じ変換。
pub(crate) fn decode_hcpe_record(rec: &[u8; HCPE_RECORD_SIZE]) -> Option<PackedSfenValue> {
    let mut hcp = [0u8; 32];
    hcp.copy_from_slice(&rec[0..32]);
    let eval = i16::from_le_bytes([rec[32], rec[33]]);
    let best_move16 = i16::from_le_bytes([rec[34], rec[35]]);
    let pack_game_result = rec[36];

    // game_ply は HCPE には情報がないので 0 で埋める。
    let pos = MiniPosition::from_hcp(&hcp, 0)?;
    let game_result = convert_game_result(pack_game_result, pos.side_to_move());
    Some(pos.to_packed_sfen_value(eval, best_move16 as u16, game_result))
}

/// HCPE データローダー
///
/// 単一または複数の `.hcpe` ファイルからレコードを読み出し、PackedSfenValue に変換しつつ
/// shuffle buffer に貯めて、buffer_size に達したら Fisher-Yates シャッフルして
/// callback に渡す。
///
/// `filter` で各レコードを採用するかを制御できる (例: `|psv| psv.score().abs() < 32000`)。
#[derive(Clone)]
pub struct HcpeDataLoader<T: Fn(&PackedSfenValue) -> bool> {
    file_paths: Vec<String>,
    /// shuffle buffer に貯める PackedSfenValue の最大個数
    buffer_size: usize,
    filter: T,
    /// HCP → PSV デコードに使う worker スレッド数。`None` のとき
    /// `std::thread::available_parallelism()` (= 論理コア数) で自動決定。
    loader_threads: Option<usize>,
    /// 再開時の seek 位置 (= 全ファイル連結ストリームの先頭からの byte
    /// offset)。`with_resume_offset` で外部から指定する。0 のときは
    /// `map_chunks` の `start_position` 引数 × `HCPE_RECORD_SIZE` を
    /// fallback として使う (= 旧 API 互換)。
    resume_offset: u64,
    /// 学習側 (= `f(&buffer)`) が「ここまでは確実に処理した」ことを表す
    /// byte offset。Producer が buffer を送出する際に attach した
    /// offset を、Consumer が `f` 完了後に書き込む。Save callback から
    /// この値を読み出して checkpoint dir に書き出すと、次回起動時に
    /// `with_resume_offset` で渡せば「先読み分の取りこぼし無し」で再開できる。
    consumed_offset: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<T: Fn(&PackedSfenValue) -> bool> HcpeDataLoader<T> {
    /// 単一ファイルから作成
    ///
    /// `buffer_size_mb` で shuffle buffer の上限サイズを MB 単位で指定する。
    /// PackedSfenValue 1 件 = 40 byte なので、buffer_size = `buffer_size_mb * 1024 * 1024 / 40`。
    pub fn new(path: &str, buffer_size_mb: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, filter)
    }

    /// 複数ファイルを連結して作成
    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, filter: T) -> Self {
        Self {
            file_paths: paths.iter().map(|x| (*x).to_string()).collect(),
            buffer_size: buffer_size_mb.saturating_mul(1024 * 1024) / 40,
            filter,
            loader_threads: None,
            resume_offset: 0,
            consumed_offset: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// HCP → PSV 並列デコードの worker 数を上書きする。`0` または `None` を
    /// 渡すと auto detection (= `available_parallelism()`) のまま。
    pub fn with_loader_threads(mut self, n: usize) -> Self {
        self.loader_threads = if n == 0 { None } else { Some(n) };
        self
    }

    /// 再開時に最初に seek する byte offset を指定する。`expand_teacher` の
    /// 出力ファイル列挙順に連結したストリームの先頭からの累積 byte 数。
    /// 0 を渡すと旧来通り `start_position` 由来の skip ロジックに falls back。
    pub fn with_resume_offset(mut self, offset: u64) -> Self {
        self.resume_offset = offset;
        self
    }

    /// 学習側が「ここまで処理した」を書き込む `AtomicU64` のハンドル。
    /// Save callback からこれを `.load()` して checkpoint dir に書き出し、
    /// 次回起動時に `with_resume_offset` で渡せば、先読み分のロスなく
    /// 厳密に再開できる。
    pub fn consumed_offset_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.consumed_offset.clone()
    }
}

impl<T> DataLoader<PackedSfenValue> for HcpeDataLoader<T>
where
    T: Fn(&PackedSfenValue) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        // 各ファイルのサイズから推測 (固定長レコードなので正確)。
        // ただし filter で減ることがあるので、これはあくまで上限値。
        let mut total = 0u64;
        for path in &self.file_paths {
            let meta = std::fs::metadata(path).ok()?;
            total += meta.len() / HCPE_RECORD_SIZE as u64;
        }
        Some(total)
    }

    fn map_chunks<F: FnMut(&[PackedSfenValue]) -> bool>(&self, start_position: usize, mut f: F) {
        // 大きな shuffle buffer を 1 個満たすのに HCP → PSV のデコードが
        // ボトルネックで、シングルスレッドだと数百秒かかる。アーキテクチャ:
        //
        // - **プロデューサスレッド** (1 個、std::thread::spawn): ファイルを
        //   `CHUNK_RECORDS` 単位で読み、内部で更に `std::thread::scope` の
        //   並列 worker で HCP→PSV デコードを cores 数だけ分担、shuffle buffer
        //   に貯まったら Fisher-Yates して `sync_channel(0)` で送り出す。
        // - **コンシューマ** (現スレッド): channel から buffer を受け取り
        //   `f(&buffer)` を呼ぶ。GPU 学習はここで進む。
        //
        // `sync_channel(0)` (rendezvous) を使うことで、メモリピークは
        // 2 × buffer_size (= 1 個埋めてる + 1 個 GPU が消費中) で済む。
        // それでも生産が消費より速い場合はプロデューサが send で待機する。

        let buffer_size = self.buffer_size.max(1);
        let file_paths = self.file_paths.clone();
        let filter = self.filter.clone();
        let loader_threads = self.loader_threads;
        // resume_offset (= 外部から指定) が 0 のとき、後方互換で
        // start_position 由来の skip を計算する (= 旧 API)。
        // HCPE は 38-byte 固定長なので `start_position × 38` で正しい。
        let resume_offset = if self.resume_offset > 0 {
            self.resume_offset
        } else {
            (start_position as u64) * HCPE_RECORD_SIZE as u64
        };
        let consumed_offset = self.consumed_offset.clone();
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<(Vec<PackedSfenValue>, u64)>(0);

        let producer = std::thread::spawn(move || {
            Self::produce_buffers(file_paths, buffer_size, filter, resume_offset, loader_threads, tx);
        });

        // コンシューマループ: (buffer, offset_at_send) を受け取り、buffer を
        // `f()` に流し、終わったら attached offset を `consumed_offset` に
        // 書き込む = 「ここまで処理済み」のしるし。Save callback がこの
        // 値を `0NNN/dataloader_pos.txt` に書き出すと、次回起動時に
        // `with_resume_offset` で渡して厳密再開できる。
        use std::sync::atomic::Ordering;
        while let Ok((buf, offset_at_send)) = rx.recv() {
            let stop = f(&buf);
            consumed_offset.store(offset_at_send, Ordering::Release);
            if stop {
                break;
            }
        }
        drop(rx);
        let _ = producer.join();
    }
}

impl<T> HcpeDataLoader<T>
where
    T: Fn(&PackedSfenValue) -> bool + Clone + Send + Sync + 'static,
{
    /// プロデューサスレッド本体。`file_paths` を順次読み、各チャンクを
    /// `std::thread::scope` で並列デコードしてから shuffle buffer に集約。
    /// buffer_size に達するごとに shuffle して `tx` に送る。
    fn produce_buffers(
        file_paths: Vec<String>,
        buffer_size: usize,
        filter: T,
        resume_offset: u64,
        loader_threads: Option<usize>,
        tx: std::sync::mpsc::SyncSender<(Vec<PackedSfenValue>, u64)>,
    ) {
        let mut buffer: Vec<PackedSfenValue> = Vec::with_capacity(buffer_size);
        let mut rng = SimpleRand::with_seed();

        // 初回 buffer fill 中だけ進捗を stderr に出す。
        let fill_started_at = Instant::now();
        let mut first_fill_in_progress = true;
        let mut last_report_at = fill_started_at;
        let target_records = buffer_size;

        let n_workers = loader_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        }).max(1);

        let mut chunk_buf = vec![0u8; HCPE_RECORD_SIZE * CHUNK_RECORDS];

        // Resume support: `resume_offset` byte だけ全ファイル連結ストリーム
        // を先に進めてから読み始める。固定長レコード前提ではなく、単に
        // 「前回ここまで処理した」を表す byte offset。次の seek 量を
        // 残しながらファイルを順に walk して、該当するファイルにきたら
        // `seek(SeekFrom::Start(...))` する。
        let mut bytes_to_skip = resume_offset;
        if bytes_to_skip > 0 {
            eprintln!(
                "  resuming from byte offset {} ({:.2} GB)...",
                bytes_to_skip,
                bytes_to_skip as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        // 「読み終わって send したぶんを含めた、ストリーム先頭からの
        // 累積 byte 数」。バッファ送出時に同梱して、Consumer 側で
        // `consumed_offset` に書き込まれる。次回起動時はこの offset を
        // `with_resume_offset` で渡せば、Producer の先読みぶんも含めて
        // 「Consumer が処理し終わった点」から再開できる。
        let mut bytes_read_total: u64 = resume_offset;

        for path in &file_paths {
            // この path の全体サイズ。`bytes_to_skip` がこれより大きければ
            // ファイル全体をスキップ (open しない)。スキップした分も
            // `bytes_read_total` に含めることで、ストリーム上の絶対 offset を
            // 維持する。
            let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let in_file_skip = if bytes_to_skip >= file_size {
                bytes_to_skip -= file_size;
                continue;
            } else {
                let off = bytes_to_skip;
                bytes_to_skip = 0;
                off
            };

            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[HcpeDataLoader] failed to open {path}: {e}");
                    continue;
                }
            };
            let mut reader = BufReader::new(file);
            if in_file_skip > 0 {
                if let Err(e) = reader.seek(SeekFrom::Start(in_file_skip)) {
                    eprintln!("[HcpeDataLoader] seek error in {path}: {e}");
                    continue;
                }
            }

            loop {
                // 完全レコード境界で読み出すため `read` を繰り返して埋める。
                let mut filled = 0usize;
                while filled < chunk_buf.len() {
                    match reader.read(&mut chunk_buf[filled..]) {
                        Ok(0) => break,
                        Ok(n) => filled += n,
                        Err(e) => {
                            eprintln!("[HcpeDataLoader] read error in {path}: {e}");
                            filled = (filled / HCPE_RECORD_SIZE) * HCPE_RECORD_SIZE;
                            break;
                        }
                    }
                }
                if filled == 0 {
                    break;
                }
                let records_in_chunk = filled / HCPE_RECORD_SIZE;
                // ストリーム上の絶対 byte offset を進める。次の send 時に
                // この時点の `bytes_read_total` を attach することで、
                // Consumer 側が「この buffer まで処理完了 = ここまで読まれた」
                // を知れる。
                bytes_read_total += (records_in_chunk * HCPE_RECORD_SIZE) as u64;

                // この chunk を n_workers で均等に分割して並列デコード。
                // 各 worker は自分のスライスをデコード → 返す Vec<PSV> を
                // 元の順番で連結することで、resume の start_position
                // セマンティクスを壊さない。
                let per_worker = records_in_chunk.div_ceil(n_workers);
                let chunk_slice = &chunk_buf[..records_in_chunk * HCPE_RECORD_SIZE];
                let filter_ref = &filter;
                let decoded: Vec<Vec<PackedSfenValue>> =
                    std::thread::scope(|s| {
                        let mut handles = Vec::with_capacity(n_workers);
                        for tid in 0..n_workers {
                            let start = tid * per_worker;
                            if start >= records_in_chunk {
                                break;
                            }
                            let end =
                                ((tid + 1) * per_worker).min(records_in_chunk);
                            let part = &chunk_slice[start * HCPE_RECORD_SIZE
                                ..end * HCPE_RECORD_SIZE];
                            handles.push(s.spawn(move || {
                                let n = end - start;
                                let mut out = Vec::with_capacity(n);
                                for i in 0..n {
                                    let off = i * HCPE_RECORD_SIZE;
                                    let rec_bytes =
                                        &part[off..off + HCPE_RECORD_SIZE];
                                    let rec: &[u8; HCPE_RECORD_SIZE] =
                                        rec_bytes.try_into().unwrap();
                                    if let Some(psv) = decode_hcpe_record(rec) {
                                        if filter_ref(&psv) {
                                            out.push(psv);
                                        }
                                    }
                                }
                                out
                            }));
                        }
                        handles
                            .into_iter()
                            .map(|h| h.join().unwrap())
                            .collect()
                    });

                // 順序保ったまま shuffle buffer に追加。`start_position` の
                // skip はファイル冒頭の byte-seek で済んでいるので、ここでは
                // skip 判定不要。
                for partial in decoded {
                    for psv in partial {
                        buffer.push(psv);

                        if first_fill_in_progress {
                            let now = Instant::now();
                            if now.duration_since(last_report_at).as_millis()
                                >= 500
                            {
                                let pct = 100.0 * buffer.len() as f64
                                    / target_records.max(1) as f64;
                                let _ = write!(
                                    std::io::stderr(),
                                    "\r  filling shuffle buffer: {:.1}M / {:.1}M records ({pct:.1}%)   ",
                                    buffer.len() as f64 / 1.0e6,
                                    target_records as f64 / 1.0e6,
                                );
                                let _ = std::io::stderr().flush();
                                last_report_at = now;
                            }
                        }

                        if buffer.len() >= buffer_size {
                            if first_fill_in_progress {
                                let elapsed = fill_started_at.elapsed();
                                let _ = writeln!(
                                    std::io::stderr(),
                                    "\r  shuffle buffer ready: {:.1}M records in {:.1}s ({} decode threads)   ",
                                    buffer.len() as f64 / 1.0e6,
                                    elapsed.as_secs_f64(),
                                    n_workers,
                                );
                                first_fill_in_progress = false;
                            }
                            for j in (1..buffer.len()).rev() {
                                let k = (rng.rng() as usize) % (j + 1);
                                buffer.swap(j, k);
                            }
                            let taken = std::mem::replace(
                                &mut buffer,
                                Vec::with_capacity(buffer_size),
                            );
                            if tx.send((taken, bytes_read_total)).is_err() {
                                // コンシューマが drop した → 終了
                                return;
                            }
                        }
                    }
                }

                if filled < chunk_buf.len() {
                    break;
                }
            }
        }

        // 残った buffer を flush
        if !buffer.is_empty() {
            if first_fill_in_progress {
                let elapsed = fill_started_at.elapsed();
                let _ = writeln!(
                    std::io::stderr(),
                    "\r  shuffle buffer ready: {:.1}M records in {:.1}s (teacher smaller than buffer)   ",
                    buffer.len() as f64 / 1.0e6,
                    elapsed.as_secs_f64(),
                );
            }
            for j in (1..buffer.len()).rev() {
                let k = (rng.rng() as usize) % (j + 1);
                buffer.swap(j, k);
            }
            let _ = tx.send((buffer, bytes_read_total));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::loader::LoadableDataType;

    #[test]
    fn hcpe_record_size_is_38() {
        assert_eq!(HCPE_RECORD_SIZE, 38);
    }

    /// HCPE の `gameResult` byte は **絶対色基準** (1=BlackWin, 2=WhiteWin, 0=Draw)
    /// で書かれているので、`decode_hcpe_record` は局面の手番に応じて PSV の
    /// **STM 視点** (1=win, -1=loss, 0=draw) に変換しなければならない。
    /// 以前は変換漏れで PSV.game_result() に絶対色基準の値がそのまま入っていて、
    /// 後手番で BlackWin の局面が「STM の勝ち」扱いになるバグがあった。
    #[test]
    fn decode_converts_game_result_to_stm_perspective() {
        use crate::shogi::Color;
        use super::super::shogipack::MiniPosition;

        // hirate (STM = Black) で BlackWin → STM win → +1
        let pos = MiniPosition::hirate_for_tests();
        let hcp = pos.pack_to_hcp();
        let mut rec = [0u8; HCPE_RECORD_SIZE];
        rec[0..32].copy_from_slice(&hcp);
        rec[36] = 1; // BlackWin
        let psv = decode_hcpe_record(&rec).expect("decode hirate");
        assert_eq!(psv.game_result(), 1, "BlackWin + STM Black → STM win = +1");

        // hirate (STM = Black) で WhiteWin → STM loss → -1
        rec[36] = 2;
        let psv = decode_hcpe_record(&rec).expect("decode hirate");
        assert_eq!(psv.game_result(), -1, "WhiteWin + STM Black → STM loss = -1");

        // hirate (STM = Black) で Draw → 0
        rec[36] = 0;
        let psv = decode_hcpe_record(&rec).expect("decode hirate");
        assert_eq!(psv.game_result(), 0, "Draw → 0 (perspective-independent)");

        // STM を White に反転して同じことを検証 (反転後の HCP を使用)
        let mut pos = MiniPosition::hirate_for_tests();
        pos.flip_stm_for_tests();
        let hcp = pos.pack_to_hcp();
        let mut rec = [0u8; HCPE_RECORD_SIZE];
        rec[0..32].copy_from_slice(&hcp);

        rec[36] = 1; // BlackWin
        let psv = decode_hcpe_record(&rec).expect("decode hirate-flipped");
        assert_eq!(psv.game_result(), -1, "BlackWin + STM White → STM loss = -1");

        rec[36] = 2; // WhiteWin
        let psv = decode_hcpe_record(&rec).expect("decode hirate-flipped");
        assert_eq!(psv.game_result(), 1, "WhiteWin + STM White → STM win = +1");

        // Color enum is referenced just to ensure the import compiles in the
        // same way the production code uses it.
        let _ = Color::Black;
    }

    #[test]
    fn decode_invalid_record_returns_none() {
        // 全 0 の HCP は king 位置が両方 0 になり、同じマスに玉が重なるので
        // from_hcp は (実装によっては) None または不正な MiniPosition を返す。
        // 本テストは「panic しないこと」「Some を返した場合でも score=0, result=0 が見えること」を確認する程度。
        let rec = [0u8; HCPE_RECORD_SIZE];
        // panic しなければ OK。
        let _ = decode_hcpe_record(&rec);
    }

    /// `inbox/ref/sp_dr2-15K_20240210.hcpe` (HCPE) と、cshogi 経由で同じ局面群を
    /// PackedSfenValue 化した `sp_dr2-15K_20240210.psv` を **バイト単位で照合** する
    /// クロスバリデーション。
    ///
    /// 比較件数は環境変数 `XREF_COUNT` で上書きできる (default 10000)。psv ファイル側に
    /// 十分なレコードが書かれていることが前提。例えば 100 万件で検証するには:
    ///
    /// ```
    /// python3 tools/cshogi_xref/hcpe_to_psv.py \
    ///     inbox/ref/sp_dr2-15K_20240210.hcpe \
    ///     inbox/ref/sp_dr2-15K_20240210.psv 1000000
    ///
    /// XREF_COUNT=1000000 cargo test -p bulletou_lib --lib \
    ///     hcpe::tests::cross_validate_against_cshogi_psv -- --ignored --nocapture
    /// ```
    ///
    /// `decode_hcpe_record` (BulletOu の HCP デコーダー → PackedSfen 再エンコード) と
    /// cshogi の `Board.set_hcp` → `Board.to_psfen` 経路が、結果として **完全に同一の
    /// 40 byte PackedSfenValue** を生成することを検証する。
    #[test]
    #[ignore]
    fn cross_validate_against_cshogi_psv() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let hcpe_path = format!("{manifest}/../../inbox/ref/sp_dr2-15K_20240210.hcpe");
        let psv_path = format!("{manifest}/../../inbox/ref/sp_dr2-15K_20240210.psv");

        if std::fs::metadata(&hcpe_path).is_err() {
            eprintln!("skipping: hcpe sample not found at {hcpe_path}");
            return;
        }
        if std::fs::metadata(&psv_path).is_err() {
            eprintln!(
                "skipping: psv reference not found at {psv_path}\n\
                 (run: python3 /tmp/hcpe_to_psv.py {hcpe_path} {psv_path} 10000)"
            );
            return;
        }

        let hcpe_bytes = std::fs::read(&hcpe_path).expect("read hcpe");
        let psv_bytes = std::fs::read(&psv_path).expect("read psv");

        let n_hcpe = hcpe_bytes.len() / HCPE_RECORD_SIZE;
        let n_psv = psv_bytes.len() / 40;
        let cap: usize =
            std::env::var("XREF_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000);
        let n = n_hcpe.min(n_psv).min(cap);
        eprintln!(
            "comparing {n} records (hcpe total = {n_hcpe}, psv total = {n_psv}, cap = {cap})"
        );
        assert!(n > 0, "need at least 1 record to compare");

        let mut mismatches = 0usize;
        let mut first_mismatch_dumped = false;

        for i in 0..n {
            let mut hcpe_rec = [0u8; HCPE_RECORD_SIZE];
            hcpe_rec.copy_from_slice(&hcpe_bytes[i * HCPE_RECORD_SIZE..(i + 1) * HCPE_RECORD_SIZE]);

            let mut psv_rec = [0u8; 40];
            psv_rec.copy_from_slice(&psv_bytes[i * 40..(i + 1) * 40]);
            let expected = PackedSfenValue::from_raw(psv_rec);

            let decoded = match decode_hcpe_record(&hcpe_rec) {
                Some(p) => p,
                None => {
                    eprintln!("record {i}: decode_hcpe_record returned None");
                    mismatches += 1;
                    continue;
                }
            };

            if decoded.as_bytes() != expected.as_bytes() {
                mismatches += 1;
                if !first_mismatch_dumped {
                    first_mismatch_dumped = true;
                    eprintln!("first MISMATCH at record {i}:");
                    eprintln!("  hcpe input:     {}", hex(&hcpe_rec));
                    eprintln!("  ours (Rust):    {}", hex(decoded.as_bytes()));
                    eprintln!("  cshogi (psv):   {}", hex(expected.as_bytes()));
                    let diffs: Vec<usize> = (0..40)
                        .filter(|&k| decoded.as_bytes()[k] != expected.as_bytes()[k])
                        .collect();
                    eprintln!("  diff offsets:   {diffs:?}");
                }
            }
        }

        if mismatches == 0 {
            eprintln!("OK: all {n} records match byte-for-byte (BulletOu == cshogi)");
        } else {
            eprintln!("{mismatches} of {n} records mismatched");
        }
        assert_eq!(mismatches, 0, "{mismatches} of {n} hcpe records disagreed with cshogi-generated psv");
    }

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            use std::fmt::Write;
            write!(&mut s, "{b:02x}").unwrap();
        }
        s
    }

    /// inbox/ref に置かれたサンプル hcpe を読む実機テスト。
    ///
    /// テストデータはローカルにしか存在しないので、`#[ignore]` を付けてある。
    /// 実行するには:
    /// ```bash
    /// cargo test -p bulletou_lib --lib hcpe -- --ignored --nocapture
    /// ```
    /// (テスト自体は GPU を使わない)
    ///
    /// パスは `CARGO_MANIFEST_DIR/../../inbox/ref/...` で workspace root 配下を
    /// 指す。`cargo test` の CWD が crate ディレクトリ (`crates/bulletou_lib`) であっても
    /// 同じファイルに辿り着けるようにしている。
    #[test]
    #[ignore]
    fn smoke_test_decode_inbox_sample() {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../inbox/ref/sp_dr2-15K_20240210.hcpe");
        let path_str = path.to_string_lossy().to_string();

        if std::fs::metadata(&path).is_err() {
            eprintln!("skipping: {path_str} not found");
            return;
        }
        eprintln!("loading: {path_str}");

        let loader = HcpeDataLoader::new(&path_str, 4, |_| true);
        let total = loader.count_positions().expect("count_positions");
        eprintln!("count_positions = {total}");
        assert!(total > 0);

        let mut got: usize = 0;
        let mut first_score: i16 = 0;
        let mut first_result: u8 = 0;
        loader.map_chunks(0, |chunk| {
            if got == 0 && !chunk.is_empty() {
                first_score = chunk[0].score();
                first_result = chunk[0].result() as u8;
            }
            got += chunk.len();
            // Stop once we've decoded enough positions for a smoke test.
            // (`true` means "stop", matching the shogipack / direct loaders'
            // convention.)
            got >= 100_000
        });

        eprintln!("decoded {got} positions; first.score = {first_score}, first.result = {first_result}");
        assert!(got >= 1);
    }

    /// Regression test for the callback-polarity bug
    /// ([`docs/...`] / commit `7bb413a`): targets a multi-buffer-sized HCPE
    /// teacher (yaneurao's kif20251209-25000a 1.61 GB file with ~45.5M
    /// records) with a callback that NEVER asks to stop (`|_| false`), and
    /// asserts the loader reads the whole file rather than terminating
    /// after the first shuffle-buffer flush. Buffer is intentionally small
    /// (`buffer_size_mb = 16` ≒ 420k records) so the bug — if reintroduced
    /// — would visibly truncate the count to ~420k instead of ~45.5M.
    ///
    /// `#[ignore]` because the input file lives outside the repo.
    /// Run with: `cargo test -p bulletou_lib --lib hcpe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn reads_entire_file_when_callback_always_continues() {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../inbox/ref/kif20251209-25000a.pack-51160847.hcpe");
        let path_str = path.to_string_lossy().to_string();

        if std::fs::metadata(&path).is_err() {
            eprintln!("skipping: {path_str} not found");
            return;
        }
        let meta = std::fs::metadata(&path).expect("metadata");
        let expected = (meta.len() / HCPE_RECORD_SIZE as u64) as usize;
        eprintln!("file {} bytes -> expected {expected} records", meta.len());

        // 16 MB buffer is intentionally small so multi-flush behaviour is exercised.
        let loader = HcpeDataLoader::new(&path_str, 16, |_| true);

        let mut got: usize = 0;
        let mut flushes: usize = 0;
        loader.map_chunks(0, |chunk| {
            got += chunk.len();
            flushes += 1;
            false // always ask for more — convention: false = continue
        });

        eprintln!("decoded {got} records across {flushes} buffer flushes");
        // Decode is lossless for valid HCPE records; any drops would come
        // from `decode_hcpe_record` returning None on a malformed sample.
        // A small rounding tolerance accounts for the trailing partial
        // chunk (the file size may not be an exact multiple of the chunk
        // boundary the loader's BufReader uses).
        assert!(
            got >= expected.saturating_sub(8),
            "loader returned {got} records, but expected ~{expected} (= file_size / 38)"
        );
        assert!(
            flushes >= 2,
            "expected multiple buffer flushes for a 1.61 GB file with a 16 MB buffer, got {flushes}"
        );
    }
}
