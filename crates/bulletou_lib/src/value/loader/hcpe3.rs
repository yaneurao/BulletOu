//! HCPE3 (HuffmanCodedPosAndEval3) データローダー
//!
//! HCPE3 は dlshogi 系のゲーム単位可変長フォーマット。
//!
//! ## ファイルレイアウト
//!
//! 1 ゲームのレイアウト:
//!
//! ```text
//! [ HuffmanCodedPosAndEval3 (36 byte) ]
//!     hcp        : u8[32]   - 開始局面 (Apery 系 HCP)
//!     moveNum    : u16 LE   - そのゲームの手数
//!     result     : u8       - 下位 2bit: 勝敗 (0=Draw 1=BlackWin 2=WhiteWin)
//!                              他: 千日手 / 入玉宣言 / 最大手数フラグ
//!     gameInfo   : u8       - 対戦相手情報 (本 loader では未使用)
//!
//! 以下を moveNum 回繰り返し:
//!
//!     [ MoveInfo (6 byte) ]
//!         selectedMove16 : i16 LE   - cshogi 形式 move16
//!         eval           : i16 LE   - その ply の評価値
//!         candidateNum   : u16 LE   - 続く MoveVisits の個数
//!
//!     [ MoveVisits × candidateNum ] (4 byte × candidateNum)
//!         move16   : i16 LE
//!         visitNum : u16 LE
//! ```
//!
//! ## 設計方針
//!
//! [`super::hcpe::HcpeDataLoader`] と同じく、各 ply を **PackedSfenValue (40 byte)** に
//! 変換してから既存の SparseInputType パイプラインに渡す:
//!
//! - 開始局面を `MiniPosition::from_hcp` で復元
//! - `MoveInfo[i]` を読み、現在の局面を `MiniPosition::to_packed_sfen_value` で psv 化
//! - `MoveVisits[i]` (policy teacher) は読み飛ばす (value 学習のみが対象)
//! - `MiniPosition::do_move(selectedMove16)` で次の ply へ
//!
//! ## PackedSfenValue 各フィールドの規約
//!
//! - `sfen`        : `MiniPosition::pack_to_psfen()` の結果 (32 byte)
//! - `score`       : `MoveInfo[i].eval`
//! - `move`        : `MoveInfo[i].selectedMove16` を u16 として
//! - `gamePly`     : ply (0-indexed)。`MiniPosition::from_hcp` で 0、`do_move` で +1 されるので自然に入る
//! - `game_result` : ヘッダ `result & 0x3` を i8 として保存 (0/1/2)
//!
//! ## 制約
//!
//! - **policy 学習はできない**: MoveVisits は読み飛ばしているため。policy を使いたい場合は
//!   別の loader を作るか、`hcpe3_cache_re_eval.py` 経由のフローを使う
//! - 終局フラグ (千日手 / 入玉宣言 / 最大手数) は無視され、ply ごとの psv の `game_result` には
//!   下位 2bit (勝敗) しか出ない

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::shogi::PackedSfenValue;

use super::rng::SimpleRand;
use super::shogipack::MiniPosition;
use super::DataLoader;

const HEADER_SIZE: usize = 36;
const MOVE_INFO_SIZE: usize = 6;
const MOVE_VISITS_SIZE: usize = 4;

/// 1 ゲームを読み、その全 ply を PackedSfenValue に展開して `out` に push する。
///
/// 戻り値:
/// - `Ok(true)`  : ゲーム 1 つを読んだ (次のゲームを試すべき)
/// - `Ok(false)` : EOF (ファイルの末尾)
/// - `Err(_)`    : I/O エラー or 不正データ
///
/// 不正な HCP / 想定外の途中切れに当たった場合は、そのゲーム分の出力は捨てて `Ok(true)` を返し、
/// caller が次のゲームに進む。
///
/// Note: 現行の `map_chunks` は producer thread 内で inline に header→plies を
/// 読む形に書き直されているのでこの関数は production code path からは使われない。
/// クロス検証テスト (cshogi 経由 PSV との byte 単位照合) で参照されているため
/// `#[allow(dead_code)]` で残してある。
#[allow(dead_code)]
fn decode_one_game<R: Read>(reader: &mut R, out: &mut Vec<PackedSfenValue>) -> std::io::Result<bool> {
    // ----- ヘッダ -----
    let mut hdr = [0u8; HEADER_SIZE];
    let mut filled = 0usize;
    while filled < HEADER_SIZE {
        match reader.read(&mut hdr[filled..])? {
            0 => {
                if filled == 0 {
                    return Ok(false); // クリーン EOF
                } else {
                    // 不完全ヘッダ。打ち切り扱い。
                    return Ok(false);
                }
            }
            n => filled += n,
        }
    }

    let mut hcp = [0u8; 32];
    hcp.copy_from_slice(&hdr[0..32]);
    let move_num = u16::from_le_bytes([hdr[32], hdr[33]]) as usize;
    let result_bits = (hdr[34] & 0x3) as i8;
    // hdr[35] = gameInfo は未使用

    // ----- 開始局面復元 -----
    let mut pos = match MiniPosition::from_hcp(&hcp, 0) {
        Some(p) => p,
        None => {
            // 不正な HCP。残りの MoveInfo + MoveVisits を読み飛ばしてゲーム終わりまで進める。
            skip_remaining_game(reader, move_num)?;
            return Ok(true);
        }
    };

    // ----- 各 ply -----
    for i in 0..move_num {
        let mut mi = [0u8; MOVE_INFO_SIZE];
        if !read_exact_or_eof(reader, &mut mi)? {
            return Ok(true); // 途中切れ。次のゲームは試さない (= 実質終了)
        }
        let selected_move16 = u16::from_le_bytes([mi[0], mi[1]]);
        let eval = i16::from_le_bytes([mi[2], mi[3]]);
        let cand_num = u16::from_le_bytes([mi[4], mi[5]]) as usize;

        // 現在の局面 (i 手目を指す前) を PackedSfenValue 化
        let psv = pos.to_packed_sfen_value(eval, selected_move16, result_bits);
        out.push(psv);

        // MoveVisits を読み飛ばし
        if cand_num > 0 {
            skip_bytes(reader, MOVE_VISITS_SIZE * cand_num)?;
        }

        // 次の ply へ進む (最後の ply は進めない)
        if i + 1 < move_num {
            pos.do_move(selected_move16);
        }
    }

    Ok(true)
}

/// `buf` を最後まで埋めるか、EOF なら false を返す。
fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => {
                return if filled == 0 { Ok(false) } else { Ok(false) };
            }
            n => filled += n,
        }
    }
    Ok(true)
}

/// `n` byte 読み飛ばし。
fn skip_bytes<R: Read>(reader: &mut R, n: usize) -> std::io::Result<()> {
    let mut remaining = n;
    let mut buf = [0u8; 4096];
    while remaining > 0 {
        let chunk = remaining.min(buf.len());
        reader.read_exact(&mut buf[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

/// 残りの MoveInfo + MoveVisits を読み飛ばす (不正 HCP のフォールバック用)。
/// (現行 production path では `skip_remaining_game_and_count` を使う。
/// クロス検証テストとの兼ね合いで残している)
#[allow(dead_code)]
fn skip_remaining_game<R: Read>(reader: &mut R, move_num: usize) -> std::io::Result<()> {
    for _ in 0..move_num {
        let mut mi = [0u8; MOVE_INFO_SIZE];
        if !read_exact_or_eof(reader, &mut mi)? {
            return Ok(());
        }
        let cand_num = u16::from_le_bytes([mi[4], mi[5]]) as usize;
        if cand_num > 0 {
            skip_bytes(reader, MOVE_VISITS_SIZE * cand_num)?;
        }
    }
    Ok(())
}

// =============================================================================
// DataLoader 実装
// =============================================================================

/// HCPE3 データローダー
///
/// 各 `.hcpe3` ファイルから 1 ゲームずつ読み、moveNum 個の PackedSfenValue に展開して
/// shuffle buffer に貯める。buffer_size に達したら Fisher-Yates シャッフルして callback に渡す。
///
/// `filter` で各 PackedSfenValue を採用するかを制御できる。
#[derive(Clone)]
pub struct Hcpe3DataLoader<T: Fn(&PackedSfenValue) -> bool> {
    file_paths: Vec<String>,
    buffer_size: usize,
    filter: T,
    /// 連結ファイル列の先頭からの累積 byte 数で、ここに居る game header から
    /// 再開する。0 のとき先頭から。
    resume_offset: u64,
    /// true のときは `(resume_offset, resume_plies) == (0, 0)` でも
    /// `start_position` 由来の legacy skip を行わず、先頭から読む。
    resume_offset_explicit: bool,
    /// `resume_offset` の game の中で「最初に push する ply」のインデックス。
    /// 0..resume_plies の MoveInfo は読まれ・do_move で局面に反映されるが、
    /// PSV としては push されない (= 前回 run で処理済みとして扱う)。
    resume_plies: usize,
    /// 学習側 (= `f(&buffer)`) が「ここまで処理した」位置 (byte offset)。
    /// Producer が attach した値を Consumer が `f` 完了後に書き込む。
    consumed_offset: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// 同上の plies (game header からの ply index)。
    consumed_plies: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl<T: Fn(&PackedSfenValue) -> bool> Hcpe3DataLoader<T> {
    /// 単一ファイルから作成
    pub fn new(path: &str, buffer_size_mb: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, filter)
    }

    /// 複数ファイルを連結して作成
    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, filter: T) -> Self {
        Self {
            file_paths: paths.iter().map(|x| (*x).to_string()).collect(),
            buffer_size: buffer_size_mb.saturating_mul(1024 * 1024) / 40,
            filter,
            resume_offset: 0,
            resume_offset_explicit: false,
            resume_plies: 0,
            consumed_offset: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            consumed_plies: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// shuffle buffer に貯める PackedSfenValue 件数を直接指定する。
    ///
    /// `trainer.run` を save chunk ごとに分割して呼ぶ場合、loader がそれより
    /// 大きな chunk を返すと、呼び出し側は必要 batch 数に達した時点で
    /// chunk の残りを消費できない。chunk 境界と save chunk 境界を揃えるための
    /// escape hatch。
    pub fn with_buffer_records(mut self, records: usize) -> Self {
        self.buffer_size = records.max(1);
        self
    }

    /// 再開位置の (byte offset, plies) を指定する。
    /// `byte_offset` は連結ファイル列の先頭からの累積 byte 数で、その位置に
    /// game header があることが前提。`plies` はその game の何手目から
    /// 学習を再開するか (0-indexed)。
    pub fn with_resume_offset(mut self, byte_offset: u64, plies: usize) -> Self {
        self.resume_offset = byte_offset;
        self.resume_plies = plies;
        self.resume_offset_explicit = false;
        self
    }

    /// 再開位置の (byte offset, plies) を明示指定する。
    /// `(0, 0)` でも `start_position` 由来の skip を行わず、先頭から読む。
    pub fn with_exact_resume_offset(mut self, byte_offset: u64, plies: usize) -> Self {
        self.resume_offset = byte_offset;
        self.resume_plies = plies;
        self.resume_offset_explicit = true;
        self
    }

    /// Consumer が「ここまで処理した」byte offset を書き込むハンドル。
    pub fn consumed_offset_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.consumed_offset.clone()
    }

    /// Consumer が「ここまで処理した」ply index を書き込むハンドル。
    pub fn consumed_plies_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
        self.consumed_plies.clone()
    }
}

impl<T> DataLoader<PackedSfenValue> for Hcpe3DataLoader<T>
where
    T: Fn(&PackedSfenValue) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        // HCPE3 は可変長なので、ファイルサイズから正確な局面数は出せない。
        None
    }

    fn map_chunks<F: FnMut(&[PackedSfenValue]) -> bool>(&self, start_position: usize, mut f: F) {
        // Producer / consumer split (= prefetch + resume pointer 機構)。
        // Producer は HCPE3 ファイルを順次読み、`resume_offset` で示される
        // game header に seek → `resume_plies` 個の MoveInfo を do_move 付き
        // で読み飛ばし (PSV push なし) → そこから expand 開始。
        // 各 buffer 送出時に「次に push される PSV の (game_header_offset,
        // ply_within_game)」を attach する。Consumer は f() 完了後に
        // `consumed_offset` / `consumed_plies` にその値を書き込み、save
        // callback がそれを `dataloader_pos.txt` に保存する。
        let buffer_size = self.buffer_size.max(1);
        let file_paths = self.file_paths.clone();
        let filter = self.filter.clone();
        let resume_offset = self.resume_offset;
        let resume_offset_explicit = self.resume_offset_explicit;
        let resume_plies = self.resume_plies;
        let consumed_offset = self.consumed_offset.clone();
        let consumed_plies = self.consumed_plies.clone();

        let (tx, rx) =
            std::sync::mpsc::sync_channel::<(Vec<PackedSfenValue>, u64, usize)>(0);

        let producer = std::thread::spawn(move || {
            Self::produce_buffers(
                file_paths,
                buffer_size,
                filter,
                resume_offset,
                resume_offset_explicit,
                resume_plies,
                start_position,
                tx,
            );
        });

        use std::sync::atomic::Ordering;
        while let Ok((buf, next_offset, next_plies)) = rx.recv() {
            let stop = f(&buf);
            consumed_offset.store(next_offset, Ordering::Release);
            consumed_plies.store(next_plies, Ordering::Release);
            if stop {
                break;
            }
        }
        drop(rx);
        let _ = producer.join();
    }
}

impl<T> Hcpe3DataLoader<T>
where
    T: Fn(&PackedSfenValue) -> bool + Clone + Send + Sync + 'static,
{
    /// プロデューサスレッド本体。`resume_offset` byte に seek し、その game
    /// の `resume_plies` 手目から PSV を expand → shuffle buffer に集約 →
    /// `(buffer, next_game_offset, next_plies)` で送出。
    fn produce_buffers(
        file_paths: Vec<String>,
        buffer_size: usize,
        filter: T,
        resume_offset: u64,
        resume_offset_explicit: bool,
        resume_plies: usize,
        start_position: usize,
        tx: std::sync::mpsc::SyncSender<(Vec<PackedSfenValue>, u64, usize)>,
    ) {
        let mut buffer: Vec<PackedSfenValue> = Vec::with_capacity(buffer_size);
        let mut rng = SimpleRand::with_seed();
        // `resume_offset == 0` のとき (= fresh start, または legacy 互換) は
        // 旧来の `start_position` 由来 filter 通過 record skip を実施。
        let mut skipped = 0usize;
        let legacy_skip_mode = !resume_offset_explicit && resume_offset == 0 && resume_plies == 0;

        // `resume_offset` を含むファイルを線形に探す。
        let mut cumulative_size: u64 = 0;
        let mut first_file_idx: usize = file_paths.len();
        let mut in_file_seek: u64 = 0;
        if resume_offset > 0 {
            for (idx, path) in file_paths.iter().enumerate() {
                let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                if cumulative_size + file_size > resume_offset {
                    first_file_idx = idx;
                    in_file_seek = resume_offset - cumulative_size;
                    break;
                }
                cumulative_size += file_size;
            }
            if first_file_idx >= file_paths.len() {
                // resume_offset がすべてのファイルサイズの合計を超えている → 何もしない
                return;
            }
        } else {
            first_file_idx = 0;
        }

        // `next_after_push` = 直近 push した PSV の次の resume 位置。
        // 各 push 後に (game_header_offset, ply+1) で更新する。Buffer send
        // 時にこのペアを attach する。初期値は呼び出し時点の resume 位置。
        let mut next_after_push: (u64, usize) = (resume_offset, resume_plies);

        // `current_global_offset` = 連結ファイル列の先頭からの現在の読み込み
        // 累積 byte 数。各 read で進める。
        let mut current_global_offset: u64 = if resume_offset > 0 { resume_offset } else { 0 };
        let mut first_game_for_this_resume = true;

        'files: for (idx, path) in file_paths.iter().enumerate() {
            if idx < first_file_idx {
                continue;
            }
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[Hcpe3DataLoader] failed to open {path}: {e}");
                    continue;
                }
            };
            let mut reader = BufReader::with_capacity(1 << 20, file);
            if idx == first_file_idx && in_file_seek > 0 {
                if let Err(e) = reader.seek(SeekFrom::Start(in_file_seek)) {
                    eprintln!("[Hcpe3DataLoader] seek error in {path}: {e}");
                    continue;
                }
            }

            loop {
                let game_header_offset = current_global_offset;

                // Game header (36 byte) を読む
                let mut hdr = [0u8; HEADER_SIZE];
                match read_exact_or_eof(&mut reader, &mut hdr) {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => {
                        eprintln!("[Hcpe3DataLoader] read error in {path}: {e}");
                        break;
                    }
                }
                current_global_offset += HEADER_SIZE as u64;

                let mut hcp = [0u8; 32];
                hcp.copy_from_slice(&hdr[0..32]);
                let move_num = u16::from_le_bytes([hdr[32], hdr[33]]) as usize;
                let result_bits = (hdr[34] & 0x3) as i8;

                let mut pos = match MiniPosition::from_hcp(&hcp, 0) {
                    Some(p) => p,
                    None => {
                        // 不正な HCP。残り body を skip して次の game へ
                        let body_bytes =
                            match skip_remaining_game_and_count(&mut reader, move_num) {
                                Ok(b) => b,
                                Err(e) => {
                                    eprintln!(
                                        "[Hcpe3DataLoader] skip error in {path}: {e}"
                                    );
                                    break;
                                }
                            };
                        current_global_offset += body_bytes;
                        first_game_for_this_resume = false;
                        continue;
                    }
                };

                // resume_plies は最初の game でのみ有効。それ以外の game は 0。
                let initial_skip = if first_game_for_this_resume {
                    first_game_for_this_resume = false;
                    resume_plies
                } else {
                    0
                };

                for i in 0..move_num {
                    let mut mi = [0u8; MOVE_INFO_SIZE];
                    match read_exact_or_eof(&mut reader, &mut mi) {
                        Ok(true) => {}
                        _ => break 'files, // 途中切れ
                    }
                    current_global_offset += MOVE_INFO_SIZE as u64;
                    let selected_move16 = u16::from_le_bytes([mi[0], mi[1]]);
                    let eval = i16::from_le_bytes([mi[2], mi[3]]);
                    let cand_num = u16::from_le_bytes([mi[4], mi[5]]) as usize;

                    let do_push = i >= initial_skip;
                    if do_push {
                        let psv =
                            pos.to_packed_sfen_value(eval, selected_move16, result_bits);
                        if filter(&psv) {
                            if legacy_skip_mode && skipped < start_position {
                                skipped += 1;
                            } else {
                                buffer.push(psv);
                                // 次に push する PSV の位置 = (この game,
                                // ply i+1)。i+1 == move_num の場合は次 game
                                // の頭にあたるが、Producer 側の resume ロジック
                                // が plies >= move_num を「この game を完全に
                                // 飛ばす」と解釈するので問題ない。
                                next_after_push = (game_header_offset, i + 1);

                                if buffer.len() >= buffer_size {
                                    for j in (1..buffer.len()).rev() {
                                        let k = (rng.rng() as usize) % (j + 1);
                                        buffer.swap(j, k);
                                    }
                                    let taken = std::mem::replace(
                                        &mut buffer,
                                        Vec::with_capacity(buffer_size),
                                    );
                                    if tx
                                        .send((
                                            taken,
                                            next_after_push.0,
                                            next_after_push.1,
                                        ))
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                        }
                    }

                    // MoveVisits を読み飛ばし (cand_num * 4 byte)
                    if cand_num > 0 {
                        let bytes = MOVE_VISITS_SIZE * cand_num;
                        if let Err(e) = skip_bytes(&mut reader, bytes) {
                            eprintln!("[Hcpe3DataLoader] skip MoveVisits error in {path}: {e}");
                            break 'files;
                        }
                        current_global_offset += bytes as u64;
                    }

                    // do_move (最後の ply は進めない)
                    if i + 1 < move_num {
                        pos.do_move(selected_move16);
                    }
                }
            }
        }

        // Trailing buffer
        if !buffer.is_empty() {
            for j in (1..buffer.len()).rev() {
                let k = (rng.rng() as usize) % (j + 1);
                buffer.swap(j, k);
            }
            let _ = tx.send((buffer, next_after_push.0, next_after_push.1));
        }
    }
}

/// `skip_remaining_game` と同じだが、skip した byte 数も返す (resume 用)。
fn skip_remaining_game_and_count<R: Read>(
    reader: &mut R,
    move_num: usize,
) -> std::io::Result<u64> {
    let mut total: u64 = 0;
    for _ in 0..move_num {
        let mut mi = [0u8; MOVE_INFO_SIZE];
        if !read_exact_or_eof(reader, &mut mi)? {
            return Ok(total);
        }
        total += MOVE_INFO_SIZE as u64;
        let cand_num = u16::from_le_bytes([mi[4], mi[5]]) as usize;
        if cand_num > 0 {
            let bytes = MOVE_VISITS_SIZE * cand_num;
            skip_bytes(reader, bytes)?;
            total += bytes as u64;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::loader::LoadableDataType;
    use std::io::BufReader;

    /// `inbox/ref/sp_dr2-15K_20240210.hcpe3` を decode_one_game で読み出した全 ply を、
    /// `tools/cshogi_xref/hcpe3_to_psv.py` で cshogi 経由で生成した
    /// `sp_dr2-15K_20240210.hcpe3.psv` と byte 単位で照合する。
    ///
    /// 事前準備:
    /// ```
    /// python3 tools/cshogi_xref/hcpe3_to_psv.py \
    ///     inbox/ref/sp_dr2-15K_20240210.hcpe3 \
    ///     inbox/ref/sp_dr2-15K_20240210.hcpe3.psv 0 10000
    /// ```
    ///
    /// 比較件数は環境変数 `XREF_COUNT` で上書きできる (default 10000)。100 万件で
    /// 検証するなら:
    /// ```
    /// python3 tools/cshogi_xref/hcpe3_to_psv.py \
    ///     inbox/ref/sp_dr2-15K_20240210.hcpe3 \
    ///     inbox/ref/sp_dr2-15K_20240210.hcpe3.psv 0 1000000
    ///
    /// XREF_COUNT=1000000 cargo test -p bulletou_lib --lib \
    ///     hcpe3::tests::cross_validate_against_cshogi_psv -- --ignored --nocapture
    /// ```
    ///
    /// 注: decode_one_game は ply を順序通り出力する (shuffle なし)。
    /// この単体テストでは Hcpe3DataLoader (shuffle あり) ではなく、decode_one_game 直接を呼ぶ。
    #[test]
    #[ignore]
    fn cross_validate_against_cshogi_psv() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let hcpe3_path = format!("{manifest}/../../inbox/ref/sp_dr2-15K_20240210.hcpe3");
        let psv_path = format!("{manifest}/../../inbox/ref/sp_dr2-15K_20240210.hcpe3.psv");

        if std::fs::metadata(&hcpe3_path).is_err() {
            eprintln!("skipping: hcpe3 sample not found at {hcpe3_path}");
            return;
        }
        if std::fs::metadata(&psv_path).is_err() {
            eprintln!(
                "skipping: psv reference not found at {psv_path}\n\
                 (run: python3 tools/cshogi_xref/hcpe3_to_psv.py {hcpe3_path} {psv_path} 0 10000)"
            );
            return;
        }

        // 比較件数は XREF_COUNT で上書き可能 (default 10000)
        let target_count: usize =
            std::env::var("XREF_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000);

        // Rust 側: decode_one_game を順次呼び、最初の target_count ply を集める。
        let file = std::fs::File::open(&hcpe3_path).expect("open hcpe3");
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let mut ours: Vec<PackedSfenValue> = Vec::with_capacity(target_count);
        loop {
            match decode_one_game(&mut reader, &mut ours) {
                Ok(true) => {
                    if ours.len() >= target_count {
                        ours.truncate(target_count);
                        break;
                    }
                }
                Ok(false) => break,
                Err(e) => panic!("decode error: {e}"),
            }
        }
        eprintln!("Rust side: decoded {} positions (target {})", ours.len(), target_count);

        // cshogi 側: psv ファイルを直接読む
        let psv_bytes = std::fs::read(&psv_path).expect("read psv");
        let n_psv = psv_bytes.len() / 40;
        eprintln!("cshogi side: psv file has {n_psv} records");

        let n = ours.len().min(n_psv);
        assert!(n > 0, "need at least 1 record to compare");

        let mut mismatches = 0usize;
        let mut first_dumped = false;
        for i in 0..n {
            let mut psv_rec = [0u8; 40];
            psv_rec.copy_from_slice(&psv_bytes[i * 40..(i + 1) * 40]);

            if ours[i].as_bytes() != &psv_rec {
                mismatches += 1;
                if !first_dumped {
                    first_dumped = true;
                    eprintln!("first MISMATCH at record {i}:");
                    eprintln!("  ours (Rust):    {}", hex(ours[i].as_bytes()));
                    eprintln!("  cshogi (psv):   {}", hex(&psv_rec));
                    let diffs: Vec<usize> = (0..40)
                        .filter(|&k| ours[i].as_bytes()[k] != psv_rec[k])
                        .collect();
                    eprintln!("  diff offsets:   {diffs:?}");
                }
            }
        }

        if mismatches == 0 {
            eprintln!("OK: all {n} HCPE3 records match byte-for-byte (BulletOu == cshogi)");
        } else {
            eprintln!("{mismatches} of {n} HCPE3 records mismatched");
        }
        assert_eq!(mismatches, 0, "{mismatches} of {n} HCPE3 records disagreed with cshogi-generated psv");
    }

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            use std::fmt::Write;
            write!(&mut s, "{b:02x}").unwrap();
        }
        s
    }

    #[test]
    #[ignore]
    fn smoke_test_loader_pipeline() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let path = format!("{manifest}/../../inbox/ref/sp_dr2-15K_20240210.hcpe3");
        if std::fs::metadata(&path).is_err() {
            eprintln!("skipping: {path} not found");
            return;
        }

        let loader = Hcpe3DataLoader::new(&path, 4, |_| true);
        let mut got = 0usize;
        let mut first_score: i16 = 0;
        let mut first_result: u8 = 0;
        loader.map_chunks(0, |chunk| {
            if got == 0 && !chunk.is_empty() {
                first_score = chunk[0].score();
                first_result = chunk[0].result() as u8;
            }
            got += chunk.len();
            // Stop once we've decoded enough positions for a smoke test.
            // (`true` means "stop", matching the shogipack / direct loaders.)
            got >= 50_000
        });
        eprintln!(
            "Hcpe3DataLoader: produced {got} positions; first.score = {first_score}, first.result = {first_result}"
        );
        assert!(got > 0);
    }
}
