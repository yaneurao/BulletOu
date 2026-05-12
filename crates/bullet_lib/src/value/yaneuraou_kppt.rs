//! YaneuraOu KPPT-format binary writer.
//!
//! Convert BulletOu's per-superbatch model dump into YaneuraOu's
//! `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` files
//! so the trained weights can be loaded by a YaneuraOu KPPT engine.
//!
//! ## Source file
//!
//! BulletOu writes two binaries per checkpoint:
//! - `raw.bin`         — saved_format order, raw float bytes only (no IDs/lengths)
//! - `optimiser_state/weights.bin` — IDs + lengths + float bytes (full self-describing)
//!
//! We read **`optimiser_state/weights.bin`** because it carries the IDs that we
//! need to find `kkw` / `kkpw`. `raw.bin` does not, and would require us to
//! also know the saved_format ordering and per-weight shape.
//!
//! ## File layout (KPPT32)
//!
//! - `KK_synthesized.bin`  : `int32_t kk[81][81][2]`           (51 KB)
//! - `KKP_synthesized.bin` : `int32_t kkp[81][81][1548][2]`    (77 MB)
//! - `KPP_synthesized.bin` : `int16_t kpp[81][1548][1548][2]`  (740 MB)
//!
//! All values are `[stm_independent, stm_dependent]`. The KK/KKP entries are
//! `int32_t × 2`; the KPP entries are `int16_t × 2` per
//! `evaluate_kppt.h::ValueKpp`. Currently only `[0]` (turn-independent) is
//! filled from the trained weights; `[1]` (turn-dependent) is set to 0 because
//! we do not train the turn term yet.
//!
//! ## KPP symmetry
//!
//! `kpp[k][p1][p2] == kpp[k][p2][p1]` by construction. Training only uses the
//! upper-triangle canonical ordering `(p_lo, p_hi)` with `p_lo < p_hi`. The
//! writer fills both `[p1][p2]` and `[p2][p1]` from the single trained value;
//! the diagonal `p1 == p2` is never updated (and stays at the initialised
//! value of 0).
//!
//! ## KPPT vs KPP_KKPT
//!
//! Two file-format variants are supported via [`KppFormat`]:
//!
//! - [`KppFormat::Kppt`] (default): KPP is `int16_t × 2` per entry (740 MB).
//!   `[0]` = turn-independent, `[1]` = turn-dependent. Used by the standard
//!   YaneuraOu KPPT eval (e.g. elmo(WCSC27)).
//! - [`KppFormat::KppKkpt`]: KPP is `int16_t` per entry (388 MB; *no* turn
//!   channel). The "factorised" variant where the turn term lives only in
//!   KK / KKP, not in KPP. Used by older YaneuraOu KPP_KKPT evals.
//!
//! KK and KKP file layouts are **identical** between the two variants (both
//! are `int32_t × 2`). Only the KPP file differs.
//!
//! ## Coordinate mapping
//!
//! BulletOu's `ShogiKk` / `ShogiKkp` produce a `stm_idx = my_king * 81 + inverse(opp_king)`
//! index. From the **Black STM** point of view that is `bk * 81 + inverse(wk)`.
//! YaneuraOu's `kk[bk][wk]` uses raw (un-inverted) square coordinates. The
//! mapping is therefore:
//!
//!     bullet_idx(bk, wk) = bk * 81 + (80 - wk)
//!
//! for the KK array, and analogously for KKP with the BonaPiece sub-index.
//!
//! ## Quantisation
//!
//! BulletOu's `kkw` / `kkpw` are f32. YaneuraOu's KPPT32 expects i32 values on
//! a centipawn-ish scale. The `eval_scale` argument is multiplied in before
//! rounding to i32. A reasonable starting value for shogi NNUE-style trainers
//! is the same as `TrainingSchedule::eval_scale` (often 400) times some small
//! integer such as 10, but the optimal scale will be tuned empirically once
//! Phase 4 makes round-trip tests possible.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

const SQ_NB: usize = 81;
const FE_END: usize = 1548;
const KK_TOTAL: usize = SQ_NB * SQ_NB; // 6561
const KKP_TOTAL: usize = SQ_NB * SQ_NB * FE_END; // 10,156,428
const KPP_TOTAL: usize = SQ_NB * FE_END * FE_END; // 194,100,624

/// Which on-disk KPP binary layout to write.
///
/// See the module-level doc for the distinction between the two variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KppFormat {
    /// `int16_t kpp[81][1548][1548][2]` — KPPT (with turn channel). 740 MB.
    #[default]
    Kppt,
    /// `int16_t kpp[81][1548][1548]` — KPP_KKPT (no turn channel). 388 MB.
    KppKkpt,
}

/// Parse BulletOu's `optimiser_state/weights.bin` into a map of weight ID -> f32 vector.
///
/// Format (see `crates/trainer/src/model/utils.rs::write_to_byte_buffer`):
///
/// ```text
/// for each weight:
///     <id ASCII bytes>     // ID string
///     \n
///     <usize LE>           // number of f32 values that follow
///     <f32 LE> × N         // raw float weights
/// ```
///
/// Note: this is **not** the format of the bare `raw.bin` file (which has no
/// IDs or lengths; see the module-level doc).
pub fn parse_model_weights_bin(bytes: &[u8]) -> io::Result<BTreeMap<String, Vec<f32>>> {
    let mut map = BTreeMap::new();
    let mut offset = 0usize;

    while offset < bytes.len() {
        // ---- Read the ID up to and including the trailing '\n' ----
        let mut id = String::new();
        loop {
            if offset >= bytes.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "raw.bin: EOF inside id"));
            }
            let ch = bytes[offset];
            offset += 1;
            if ch == b'\n' {
                break;
            }
            id.push(ch as char);
        }

        // ---- Read the usize little-endian length ----
        const USIZE_BYTES: usize = std::mem::size_of::<usize>();
        if offset + USIZE_BYTES > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("raw.bin: EOF inside length (id={id:?}, offset={offset}, total={})", bytes.len()),
            ));
        }
        let len_bytes: [u8; USIZE_BYTES] = bytes[offset..offset + USIZE_BYTES].try_into().unwrap();
        let len = usize::from_le_bytes(len_bytes);
        offset += USIZE_BYTES;

        // ---- Read len f32 values ----
        if offset + len.saturating_mul(4) > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "raw.bin: EOF inside values (id={id:?}, len={len}, need={} bytes, have={} bytes)",
                    len.saturating_mul(4),
                    bytes.len().saturating_sub(offset),
                ),
            ));
        }
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let off = offset + i * 4;
            let v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            values.push(v);
        }
        offset += len * 4;

        map.insert(id, values);
    }

    Ok(map)
}

/// Convert a checkpoint directory's `optimiser_state/weights.bin` into YaneuraOu
/// KPPT binaries placed alongside it. Only the components that are actually
/// present are written: e.g. if the model has only `kkw` / `kkb`, the KKP file
/// is skipped.
///
/// `eval_scale` multiplies the f32 weight before rounding to i{32,16}.
/// Equivalent to [`save_yaneuraou_eval`] with `KppFormat::Kppt`.
pub fn save_yaneuraou_kppt(checkpoint_dir: &Path, eval_scale: f32) -> io::Result<()> {
    save_yaneuraou_eval(checkpoint_dir, eval_scale, KppFormat::Kppt)
}

/// Variant of [`save_yaneuraou_kppt`] that selects the on-disk KPP layout
/// (`KppFormat::Kppt` for standard KPPT, `KppFormat::KppKkpt` for the
/// factorised KPP_KKPT). KK and KKP outputs are identical in either case.
pub fn save_yaneuraou_eval(
    checkpoint_dir: &Path,
    eval_scale: f32,
    kpp_format: KppFormat,
) -> io::Result<()> {
    let weights_path = checkpoint_dir.join("optimiser_state").join("weights.bin");
    let bytes = std::fs::read(&weights_path)?;
    let weights = parse_model_weights_bin(&bytes)?;

    let mut wrote_any = false;

    if let Some(kkw) = weights.get("kkw") {
        let kk_path = checkpoint_dir.join("KK_synthesized.bin");
        write_kk_bin(&kk_path, kkw, eval_scale)?;
        wrote_any = true;
    }

    if let Some(kkpw) = weights.get("kkpw") {
        let kkp_path = checkpoint_dir.join("KKP_synthesized.bin");
        write_kkp_bin(&kkp_path, kkpw, eval_scale)?;
        wrote_any = true;
    }

    if let Some(kppw) = weights.get("kppw") {
        let kpp_path = checkpoint_dir.join("KPP_synthesized.bin");
        match kpp_format {
            KppFormat::Kppt => write_kpp_bin(&kpp_path, kppw, eval_scale)?,
            KppFormat::KppKkpt => write_kpp_bin_factorised(&kpp_path, kppw, eval_scale)?,
        }
        wrote_any = true;
    }

    if !wrote_any {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "optimiser_state/weights.bin contains none of `kkw` / `kkpw` / `kppw`; nothing to write",
        ));
    }

    Ok(())
}

fn quantise_to_i16(f: f32, scale: f32) -> i16 {
    let scaled = f * scale;
    if scaled.is_nan() {
        0
    } else if scaled >= i16::MAX as f32 {
        i16::MAX
    } else if scaled <= i16::MIN as f32 {
        i16::MIN
    } else {
        scaled.round() as i16
    }
}

fn quantise_to_i32(f: f32, scale: f32) -> i32 {
    let scaled = f * scale;
    // saturate to i32 range
    if scaled.is_nan() {
        0
    } else if scaled >= i32::MAX as f32 {
        i32::MAX
    } else if scaled <= i32::MIN as f32 {
        i32::MIN
    } else {
        scaled.round() as i32
    }
}

fn write_kk_bin(path: &Path, kkw: &[f32], scale: f32) -> io::Result<()> {
    if kkw.len() != KK_TOTAL {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("kkw size {} != expected {}", kkw.len(), KK_TOTAL),
        ));
    }
    let mut buf = Vec::with_capacity(KK_TOTAL * 2 * 4);
    for bk in 0..SQ_NB {
        for wk in 0..SQ_NB {
            let wk_inv = 80 - wk;
            let bullet_idx = bk * SQ_NB + wk_inv;
            let v0 = quantise_to_i32(kkw[bullet_idx], scale);
            buf.extend_from_slice(&v0.to_le_bytes()); // [stm_independent]
            buf.extend_from_slice(&0i32.to_le_bytes()); // [stm_dependent], Phase 2 では 0
        }
    }
    std::fs::write(path, &buf)
}

fn write_kkp_bin(path: &Path, kkpw: &[f32], scale: f32) -> io::Result<()> {
    if kkpw.len() != KKP_TOTAL {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("kkpw size {} != expected {}", kkpw.len(), KKP_TOTAL),
        ));
    }
    let mut buf = Vec::with_capacity(KKP_TOTAL * 2 * 4);
    for bk in 0..SQ_NB {
        for wk in 0..SQ_NB {
            for bp in 0..FE_END {
                let wk_inv = 80 - wk;
                let bullet_idx = (bk * SQ_NB + wk_inv) * FE_END + bp;
                let v0 = quantise_to_i32(kkpw[bullet_idx], scale);
                buf.extend_from_slice(&v0.to_le_bytes());
                buf.extend_from_slice(&0i32.to_le_bytes());
            }
        }
    }
    std::fs::write(path, &buf)
}

/// `KPP_synthesized.bin` (KPPT) = `int16_t kpp[81][1548][1548][2]` (約 740 MB)。
///
/// 訓練側 (`ShogiKpp`) は p1 < p2 の canonical な三角行列のみ学習している
/// ので、各 `(k, p1, p2)` で `p1 != p2` のとき `kpp[k][p1][p2]` と
/// `kpp[k][p2][p1]` の **両方に同じ値**を書く。対角 `p1 == p2` は学習されて
/// おらず、対応する `kppw` エントリは初期値 0 のままなので、出力もそのまま 0。
///
/// 740 MB を一度にバッファするのは避けて、`BufWriter` で逐次書き出す。
fn write_kpp_bin(path: &Path, kppw: &[f32], scale: f32) -> io::Result<()> {
    if kppw.len() != KPP_TOTAL {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("kppw size {} != expected {}", kppw.len(), KPP_TOTAL),
        ));
    }
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    for k in 0..SQ_NB {
        for p1 in 0..FE_END {
            for p2 in 0..FE_END {
                // canonical lookup: trained tensor stores only (lo, hi) with lo < hi.
                // diagonal (p1 == p2) is never trained -> kppw[..] = 0 by construction.
                let (lo, hi) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
                let bullet_idx = k * FE_END * FE_END + lo * FE_END + hi;
                let v0 = quantise_to_i16(kppw[bullet_idx], scale);
                writer.write_all(&v0.to_le_bytes())?; // [stm_independent]
                writer.write_all(&0i16.to_le_bytes())?; // [stm_dependent]
            }
        }
    }
    writer.flush()?;
    Ok(())
}

/// `KPP_synthesized.bin` (KPP_KKPT factorised) = `int16_t kpp[81][1548][1548]`
/// (約 388 MB)。
///
/// KPPT との唯一の違いは末尾の `[2]` (手番チャンネル) が無いこと。
/// 学習側 (`ShogiKpp`) は元々 `[0]` (手番無関係項) のみを学習しているので、
/// 出力で `[1]` を省くだけで KPP_KKPT 形式になる。`(p1, p2)` の対称コピーは
/// KPPT と同じ。
fn write_kpp_bin_factorised(path: &Path, kppw: &[f32], scale: f32) -> io::Result<()> {
    if kppw.len() != KPP_TOTAL {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("kppw size {} != expected {}", kppw.len(), KPP_TOTAL),
        ));
    }
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    for k in 0..SQ_NB {
        for p1 in 0..FE_END {
            for p2 in 0..FE_END {
                let (lo, hi) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
                let bullet_idx = k * FE_END * FE_END + lo * FE_END + hi;
                let v0 = quantise_to_i16(kppw[bullet_idx], scale);
                writer.write_all(&v0.to_le_bytes())?; // single i16, no turn channel
            }
        }
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip_minimal() {
        // optimiser_state/weights.bin と同じレイアウトの 1 重み分のバイト列を
        // 手作りして parse できるか確認
        let mut buf = Vec::new();
        buf.extend_from_slice(b"hello\n");
        let len: usize = 2;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&1.5f32.to_le_bytes());
        buf.extend_from_slice(&(-2.25f32).to_le_bytes());

        let map = parse_model_weights_bin(&buf).unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map["hello"], vec![1.5, -2.25]);
    }

    #[test]
    fn quantise_saturates() {
        assert_eq!(quantise_to_i32(f32::NAN, 1.0), 0);
        assert_eq!(quantise_to_i32(1e30, 1.0), i32::MAX);
        assert_eq!(quantise_to_i32(-1e30, 1.0), i32::MIN);
        assert_eq!(quantise_to_i32(1.7, 2.0), 3); // 3.4 -> round = 3
        assert_eq!(quantise_to_i32(-1.7, 2.0), -3); // -3.4 -> round = -3
    }

    #[test]
    fn write_kk_produces_correct_size() {
        let kkw = vec![0.0f32; KK_TOTAL];
        let tmp = std::env::temp_dir().join("bulletou_test_kk.bin");
        write_kk_bin(&tmp, &kkw, 1.0).unwrap();
        let meta = std::fs::metadata(&tmp).unwrap();
        // expected: 81 * 81 * 2 * 4 = 52488 byte
        assert_eq!(meta.len(), (KK_TOTAL * 2 * 4) as u64);
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn quantise_i16_saturates() {
        assert_eq!(quantise_to_i16(f32::NAN, 1.0), 0);
        assert_eq!(quantise_to_i16(1e10, 1.0), i16::MAX);
        assert_eq!(quantise_to_i16(-1e10, 1.0), i16::MIN);
        assert_eq!(quantise_to_i16(1.7, 2.0), 3);
        assert_eq!(quantise_to_i16(-1.7, 2.0), -3);
    }

    #[test]
    fn write_kpp_symmetry_and_size() {
        // 三角行列に 1 件だけ非零値を入れて、symmetric にコピーされること + サイズ
        // を確認する。776 MB の全実体生成は CI コストが高いので、ここでは fe_end を
        // 通常通り扱いつつバッファ全体を 0 初期化し、(k=0, p1=2, p2=5) にだけ値を
        // 置く。
        let mut kppw = vec![0.0f32; KPP_TOTAL];
        kppw[0 * FE_END * FE_END + 2 * FE_END + 5] = 1.0;

        let tmp = std::env::temp_dir().join("bulletou_test_kpp.bin");
        write_kpp_bin(&tmp, &kppw, 100.0).unwrap();

        let meta = std::fs::metadata(&tmp).unwrap();
        // expected: 81 * 1548 * 1548 * 2 * 2 = 776,402,496 byte
        assert_eq!(meta.len(), (KPP_TOTAL * 2 * 2) as u64);

        // ファイルの (k=0, p1=2, p2=5) と (k=0, p1=5, p2=2) の両方が
        // i16(100) = 100 になっていることを確認
        let data = std::fs::read(&tmp).unwrap();
        let read_entry = |k: usize, p1: usize, p2: usize| -> i16 {
            let off = ((k * FE_END * FE_END + p1 * FE_END + p2) * 2) * 2; // *2 (channel) *2 (i16)
            i16::from_le_bytes([data[off], data[off + 1]])
        };
        assert_eq!(read_entry(0, 2, 5), 100, "(0, 2, 5) should be 100");
        assert_eq!(read_entry(0, 5, 2), 100, "(0, 5, 2) should be 100 by symmetry");
        // 対角 (0, 2, 2) は 0
        assert_eq!(read_entry(0, 2, 2), 0);
        // 別の k は 0
        assert_eq!(read_entry(1, 2, 5), 0);

        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn write_kpp_factorised_half_size_and_symmetric() {
        // KPP_KKPT 形式は KPPT の半分のサイズ (手番チャンネルなし)、
        // 対称性は維持されることを確認する。
        let mut kppw = vec![0.0f32; KPP_TOTAL];
        kppw[0 * FE_END * FE_END + 2 * FE_END + 5] = 1.0;

        let tmp = std::env::temp_dir().join("bulletou_test_kpp_factorised.bin");
        write_kpp_bin_factorised(&tmp, &kppw, 100.0).unwrap();

        let meta = std::fs::metadata(&tmp).unwrap();
        // expected: 81 * 1548 * 1548 * 2 = 388,201,248 byte (i16 × 1 channel)
        assert_eq!(meta.len(), (KPP_TOTAL * 2) as u64);

        let data = std::fs::read(&tmp).unwrap();
        let read_entry = |k: usize, p1: usize, p2: usize| -> i16 {
            let off = (k * FE_END * FE_END + p1 * FE_END + p2) * 2; // *2 (i16), no channel
            i16::from_le_bytes([data[off], data[off + 1]])
        };
        assert_eq!(read_entry(0, 2, 5), 100, "(0, 2, 5) should be 100");
        assert_eq!(read_entry(0, 5, 2), 100, "(0, 5, 2) should be 100 by symmetry");
        assert_eq!(read_entry(0, 2, 2), 0); // diagonal
        assert_eq!(read_entry(1, 2, 5), 0); // different king

        let _ = std::fs::remove_file(tmp);
    }
}
