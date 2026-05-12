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
use std::io::{BufReader, Read};

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
        }
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
        let mut buffer: Vec<PackedSfenValue> = Vec::with_capacity(self.buffer_size.max(1));
        let mut rng = SimpleRand::with_seed();
        let mut skipped = 0usize;

        for path in &self.file_paths {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[Hcpe3DataLoader] failed to open {path}: {e}");
                    continue;
                }
            };
            let mut reader = BufReader::with_capacity(1 << 20, file);

            loop {
                let mut game: Vec<PackedSfenValue> = Vec::new();
                match decode_one_game(&mut reader, &mut game) {
                    Ok(true) => {}
                    Ok(false) => break, // EOF
                    Err(e) => {
                        eprintln!("[Hcpe3DataLoader] decode error in {path}: {e}");
                        break;
                    }
                }

                for psv in game.into_iter() {
                    if !(self.filter)(&psv) {
                        continue;
                    }
                    if skipped < start_position {
                        skipped += 1;
                        continue;
                    }
                    buffer.push(psv);

                    if buffer.len() >= self.buffer_size {
                        for j in (1..buffer.len()).rev() {
                            let k = (rng.rng() as usize) % (j + 1);
                            buffer.swap(j, k);
                        }
                        // Convention (same as shogipack / direct loaders): `f`
                        // returns `true` to signal "stop, no more data needed"
                        // and `false` for "continue, send more". The previous
                        // `!f(...)` was inverted, causing a single buffer's
                        // worth of records to be flushed and then the loader
                        // to exit prematurely on every HCPE3 training run.
                        if f(&buffer) {
                            return;
                        }
                        buffer.clear();
                    }
                }
            }
        }

        if !buffer.is_empty() {
            for j in (1..buffer.len()).rev() {
                let k = (rng.rng() as usize) % (j + 1);
                buffer.swap(j, k);
            }
            f(&buffer);
        }
    }
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
    /// XREF_COUNT=1000000 cargo test -p bullet_lib --lib \
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
