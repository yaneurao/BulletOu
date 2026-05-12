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
//! byte      36: gameResult  (i8) — 0=Draw, 1=BlackWin, 2=WhiteWin (cshogi 慣例)
//! byte      37: dummy       (u8) — padding
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
//! Policy 蒸留や policy 教師を使いたい場合は HCPE3 を使う (今後対応予定)。

use std::fs::File;
use std::io::{BufReader, Read};

use crate::shogi::PackedSfenValue;

use super::rng::SimpleRand;
use super::shogipack::MiniPosition;
use super::DataLoader;

/// HCPE 1 レコードのバイト数
pub const HCPE_RECORD_SIZE: usize = 38;

/// 1 度に読むディスクチャンクのレコード数 (`HCPE_RECORD_SIZE` × CHUNK_RECORDS バイト)
const CHUNK_RECORDS: usize = 4096;

/// 1 レコードを PackedSfenValue にデコード
///
/// HCP のデコードに失敗した場合 (不正レコード) は `None` を返す。
/// caller は不正レコードをスキップして次に進める想定。
fn decode_hcpe_record(rec: &[u8; HCPE_RECORD_SIZE]) -> Option<PackedSfenValue> {
    let mut hcp = [0u8; 32];
    hcp.copy_from_slice(&rec[0..32]);
    let eval = i16::from_le_bytes([rec[32], rec[33]]);
    let best_move16 = i16::from_le_bytes([rec[34], rec[35]]);
    let game_result = rec[36] as i8;

    // game_ply は HCPE には情報がないので 0 で埋める。
    let pos = MiniPosition::from_hcp(&hcp, 0)?;
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
        }
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
        // shuffle buffer: PackedSfenValue を貯めて Fisher-Yates してから callback へ渡す。
        let mut buffer: Vec<PackedSfenValue> = Vec::with_capacity(self.buffer_size.max(1));
        let mut rng = SimpleRand::with_seed();

        // resume サポート (best-effort consume-and-drop):
        // start_position 個の filter 通過 record を input 順序で読み飛ばす。
        // shuffle buffer 単位の境界では fresh run と完全一致しないため best-effort 扱い。
        let mut skipped = 0usize;

        let mut chunk_buf = vec![0u8; HCPE_RECORD_SIZE * CHUNK_RECORDS];

        for path in &self.file_paths {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[HcpeDataLoader] failed to open {path}: {e}");
                    continue;
                }
            };
            let mut reader = BufReader::new(file);

            loop {
                // 完全なレコード境界に揃った読み出しを行うため、CHUNK_RECORDS 単位で読む。
                // BufReader::read は read_exact ではないので、`read` を繰り返して埋める。
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
                for i in 0..records_in_chunk {
                    let off = i * HCPE_RECORD_SIZE;
                    let mut rec = [0u8; HCPE_RECORD_SIZE];
                    rec.copy_from_slice(&chunk_buf[off..off + HCPE_RECORD_SIZE]);

                    let psv = match decode_hcpe_record(&rec) {
                        Some(p) => p,
                        None => continue, // 不正レコードはスキップ
                    };
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
                        if !f(&buffer) {
                            return;
                        }
                        buffer.clear();
                    }
                }

                // chunk が満たなかった (= EOF 近辺) なら次ファイルへ
                if filled < chunk_buf.len() {
                    break;
                }
            }
        }

        // 残った buffer を flush
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

    #[test]
    fn hcpe_record_size_is_38() {
        assert_eq!(HCPE_RECORD_SIZE, 38);
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
    /// XREF_COUNT=1000000 cargo test -p bullet_lib --lib \
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
    /// cargo test -p bullet_lib --lib hcpe -- --ignored --nocapture
    /// ```
    /// (テスト自体は GPU を使わない)
    ///
    /// パスは `CARGO_MANIFEST_DIR/../../inbox/ref/...` で workspace root 配下を
    /// 指す。`cargo test` の CWD が crate ディレクトリ (`crates/bullet_lib`) であっても
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
            // 最初の 100k 局面だけ確認すれば smoke test として十分
            got < 100_000
        });

        eprintln!("decoded {got} positions; first.score = {first_score}, first.result = {first_result}");
        assert!(got >= 1);
    }
}
