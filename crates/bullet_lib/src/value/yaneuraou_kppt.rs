//! YaneuraOu KPPT-format binary writer.
//!
//! Convert BulletOu's per-superbatch `raw.bin` (the f32 checkpoint dump) into
//! YaneuraOu's `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin`
//! files so the trained weights can be loaded by a YaneuraOu KPPT engine.
//!
//! ## File layout (KPPT32)
//!
//! - `KK_synthesized.bin`  : `int32_t kk[81][81][2]`           (51 KB)
//! - `KKP_synthesized.bin` : `int32_t kkp[81][81][1548][2]`    (76 MB)
//! - `KPP_synthesized.bin` : `int16_t kpp[81][1548][1548][2]`  (740 MB, Phase 3)
//!
//! The trailing `[2]` is `[stm_independent, stm_dependent]`. Phase 2 only writes
//! `[0]` (filled with the learned weight) and zeros out `[1]`. Proper handling
//! of the turn term is Phase 3 work.
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

/// Parse a BulletOu `raw.bin` checkpoint file into a map of weight ID -> f32 vector.
///
/// `raw.bin` format (see `crates/trainer/src/model/utils.rs::write_to_byte_buffer`):
///
/// ```text
/// for each weight:
///     <id ASCII bytes>     // ID string
///     \n
///     <usize LE>           // number of f32 values that follow
///     <f32 LE> × N         // raw float weights
/// ```
pub fn parse_raw_bin(bytes: &[u8]) -> io::Result<BTreeMap<String, Vec<f32>>> {
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

/// Convert a checkpoint directory's `raw.bin` into YaneuraOu KPPT binaries
/// alongside it. Only the components that are actually present in `raw.bin`
/// are written: e.g. if the model has only `kkw` / `kkb` the KKP file is skipped.
///
/// `eval_scale` multiplies the f32 weight before rounding to i32.
pub fn save_yaneuraou_kppt(checkpoint_dir: &Path, eval_scale: f32) -> io::Result<()> {
    let raw_path = checkpoint_dir.join("raw.bin");
    let bytes = std::fs::read(&raw_path)?;
    let weights = parse_raw_bin(&bytes)?;

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

    if !wrote_any {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw.bin contains neither `kkw` nor `kkpw`; nothing to write",
        ));
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip_minimal() {
        // 手作りで raw.bin と同じレイアウトの 1 重み分のバイト列を作って parse できるか確認
        let mut buf = Vec::new();
        buf.extend_from_slice(b"hello\n");
        let len: usize = 2;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&1.5f32.to_le_bytes());
        buf.extend_from_slice(&(-2.25f32).to_le_bytes());

        let map = parse_raw_bin(&buf).unwrap();
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
}
