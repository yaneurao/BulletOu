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
//! ```text
//! bullet_idx(bk, wk) = bk * 81 + (80 - wk)
//! ```
//!
//! for the KK array, and analogously for KKP with the BonaPiece sub-index.
//!
//! ## Quantisation
//!
//! BulletOu's `kkw` / `kkpw` / `kppw` are f32. YaneuraOu's KPPT32 expects i32
//! values on a centipawn-ish scale (KK / KKP) or i16 (KPP). The `eval_scale`
//! argument is multiplied in before rounding. A reasonable starting value for
//! shogi NNUE-style trainers is the same as `TrainingSchedule::eval_scale`
//! (often 400) times some small integer such as 10 for the i32 components,
//! and an order of magnitude smaller for the i16 KPP component.

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

/// Serialise an iterator of `(id, weights)` pairs into the same
/// "id\n<usize LE><f32 LE × N>" record stream that
/// [`parse_model_weights_bin`] reads. Returns the in-memory byte buffer.
pub fn write_model_weights_bin<'a>(records: impl IntoIterator<Item = (&'a str, &'a [f32])>) -> Vec<u8> {
    let mut buf = Vec::new();
    for (id, values) in records {
        buf.extend_from_slice(id.as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(&values.len().to_le_bytes());
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

pub const STATE_BACKEND_RECORD_PREFIX: &str = "meta/state_backend/";
pub const STATE_BACKEND_MARKER_VALUE: f32 = 1.0;
pub const STATE_BACKEND_BULLET: &str = "bullet";
pub const STATE_BACKEND_LEGACY_RUST_CUDA: &str = "cuda-oxide";

pub fn state_backend_record_id(backend: &str) -> String {
    format!("{STATE_BACKEND_RECORD_PREFIX}{backend}")
}

pub fn write_state_backend_marker(backend: &str) -> Vec<u8> {
    let id = state_backend_record_id(backend);
    let value = [STATE_BACKEND_MARKER_VALUE];
    write_model_weights_bin([(id.as_str(), value.as_slice())])
}

pub fn detect_state_backend(records: &BTreeMap<String, Vec<f32>>) -> Option<String> {
    records.iter().find_map(|(id, values)| {
        if values.as_slice() != [STATE_BACKEND_MARKER_VALUE] {
            return None;
        }
        id.strip_prefix(STATE_BACKEND_RECORD_PREFIX).filter(|backend| !backend.is_empty()).map(ToOwned::to_owned)
    })
}

/// Bundle one component's `optimiser_state/` files into the running combined-state buffer
/// `out`, with every record's ID prefixed by
/// `<component>/<section>/` so the three components do not clash on shared
/// IDs (e.g. all components have `outw`).
///
/// `component` is `"kk"` / `"kkp"` / `"kpp"` / `"nnue"`. Required sections
/// are `"weights"`, `"momentum"`, and `"velocity"`. Ranger additionally writes
/// `"slow"` and a text `"step_ranger"` file; these are bundled when present.
pub fn bundle_component_state(out: &mut Vec<u8>, component: &str, optimiser_state_dir: &Path) -> io::Result<()> {
    for section in ["weights", "momentum", "velocity"] {
        let path = optimiser_state_dir.join(format!("{section}.bin"));
        let bytes = std::fs::read(&path)?;
        let parsed = parse_model_weights_bin(&bytes)?;
        let mut records: Vec<(String, &[f32])> =
            parsed.iter().map(|(k, v)| (format!("{component}/{section}/{k}"), v.as_slice())).collect();
        records.sort_by(|a, b| a.0.cmp(&b.0));
        let chunk = write_model_weights_bin(records.iter().map(|(k, v)| (k.as_str(), *v)));
        out.extend_from_slice(&chunk);
    }
    for section in ["slow"] {
        let path = optimiser_state_dir.join(format!("{section}.bin"));
        if !path.is_file() {
            continue;
        }
        let bytes = std::fs::read(&path)?;
        let parsed = parse_model_weights_bin(&bytes)?;
        let mut records: Vec<(String, &[f32])> =
            parsed.iter().map(|(k, v)| (format!("{component}/{section}/{k}"), v.as_slice())).collect();
        records.sort_by(|a, b| a.0.cmp(&b.0));
        let chunk = write_model_weights_bin(records.iter().map(|(k, v)| (k.as_str(), *v)));
        out.extend_from_slice(&chunk);
    }
    let step_path = optimiser_state_dir.join("step_ranger.txt");
    if step_path.is_file() {
        let text = std::fs::read_to_string(&step_path)?;
        let mut records: Vec<(String, Vec<f32>)> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((id, step)) = line.split_once(',') else { continue };
            let Ok(step) = step.trim().parse::<u64>() else { continue };
            records.push((format!("{component}/step_ranger/{}", id.trim()), vec![step as f32]));
        }
        records.sort_by(|a, b| a.0.cmp(&b.0));
        let chunk = write_model_weights_bin(records.iter().map(|(k, v)| (k.as_str(), v.as_slice())));
        out.extend_from_slice(&chunk);
    }
    Ok(())
}

/// Extract one component's records from a `state.bin` buffer parsed by
/// [`parse_model_weights_bin`]. Returns a map from un-prefixed weight ID
/// (e.g. `kkw`) to its f32 buffer, for the requested `section` (one of
/// `"weights"`, `"momentum"`, `"velocity"`) of the requested `component`
/// (`"kk"` / `"kkp"` / `"kpp"`).
pub fn extract_component_section(
    state_records: &BTreeMap<String, Vec<f32>>,
    component: &str,
    section: &str,
) -> BTreeMap<String, Vec<f32>> {
    let prefix = format!("{component}/{section}/");
    state_records
        .iter()
        .filter_map(|(k, v)| k.strip_prefix(&prefix).map(|tail| (tail.to_string(), v.clone())))
        .collect()
}

/// Write a single component's optimiser state files to `optimiser_state_dir` from the bundled
/// `state.bin` buffer, so a freshly-built bullet trainer can pick it up
/// via `Optimiser::load_from_checkpoint(<dir>)`. Returns an error if the
/// component's records are missing from `state_records`.
pub fn unbundle_component_state(
    state_records: &BTreeMap<String, Vec<f32>>,
    component: &str,
    optimiser_state_dir: &Path,
) -> io::Result<()> {
    std::fs::create_dir_all(optimiser_state_dir)?;
    for section in ["weights", "momentum", "velocity"] {
        let extracted = extract_component_section(state_records, component, section);
        if extracted.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("state.bin: no `{component}/{section}/*` records"),
            ));
        }
        let records: Vec<(&str, &[f32])> = extracted.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let chunk = write_model_weights_bin(records.into_iter());
        std::fs::write(optimiser_state_dir.join(format!("{section}.bin")), chunk)?;
    }
    let slow = extract_component_section(state_records, component, "slow");
    if slow.is_empty() {
        // Older state.bin files, and AdamW/RAdam checkpoints, have no Ranger
        // slow weights. Initialising slow weights from current weights lets a
        // user intentionally switch to Ranger with `--resume` without crashing.
        let weights = extract_component_section(state_records, component, "weights");
        let records: Vec<(&str, &[f32])> = weights.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let chunk = write_model_weights_bin(records.into_iter());
        std::fs::write(optimiser_state_dir.join("slow.bin"), chunk)?;
    } else {
        let records: Vec<(&str, &[f32])> = slow.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let chunk = write_model_weights_bin(records.into_iter());
        std::fs::write(optimiser_state_dir.join("slow.bin"), chunk)?;
    }
    let steps = extract_component_section(state_records, component, "step_ranger");
    if !steps.is_empty() {
        let mut lines = String::new();
        for (id, v) in steps {
            let step = v.first().copied().unwrap_or(0.0).max(0.0).round() as u64;
            lines.push_str(&format!("{id},{step}\n"));
        }
        std::fs::write(optimiser_state_dir.join("step_ranger.txt"), lines)?;
    }
    Ok(())
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
pub fn save_yaneuraou_eval(checkpoint_dir: &Path, eval_scale: f32, kpp_format: KppFormat) -> io::Result<()> {
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
            buf.extend_from_slice(&0i32.to_le_bytes()); // [stm_dependent] は学習対象外 (0)
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
    fn state_backend_marker_round_trip_and_component_extract_ignores_meta() {
        let mut buf = write_state_backend_marker(STATE_BACKEND_LEGACY_RUST_CUDA);
        let records = [("nnue/weights/l0w", [1.0f32, 2.0].as_slice()), ("nnue/weights/l0b", [3.0f32].as_slice())];
        buf.extend_from_slice(&write_model_weights_bin(records));

        let map = parse_model_weights_bin(&buf).unwrap();
        assert_eq!(detect_state_backend(&map).as_deref(), Some(STATE_BACKEND_LEGACY_RUST_CUDA));

        let nnue_weights = extract_component_section(&map, "nnue", "weights");
        assert_eq!(nnue_weights.len(), 2);
        assert_eq!(nnue_weights["l0w"], vec![1.0, 2.0]);
        assert_eq!(nnue_weights["l0b"], vec![3.0]);
        assert!(!nnue_weights.contains_key(&state_backend_record_id(STATE_BACKEND_LEGACY_RUST_CUDA)));
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
    fn bundle_and_unbundle_state_round_trip() {
        // 3 component 分の optimiser_state を作って bundle → unbundle で
        // 戻ることを確認。
        use std::fs;

        let tmp = std::env::temp_dir().join("bulletou_state_roundtrip");
        let _ = fs::remove_dir_all(&tmp);

        // Synthesise per-component optimiser_state dirs with three sections each.
        for comp in ["kk", "kkp", "kpp"] {
            let dir = tmp.join(format!("{comp}_orig")).join("optimiser_state");
            fs::create_dir_all(&dir).unwrap();
            for section in ["weights", "momentum", "velocity"] {
                let id_a = format!("{comp}_idA");
                let id_b = format!("{comp}_idB");
                let bytes = write_model_weights_bin(
                    [(id_a.as_str(), [1.0f32, 2.0, 3.0].as_slice()), (id_b.as_str(), [4.0f32, 5.0].as_slice())]
                        .into_iter()
                        .map(|(k, v)| (k, v)),
                );
                fs::write(dir.join(format!("{section}.bin")), bytes).unwrap();
            }
            if comp == "kkp" {
                let slow_bytes = write_model_weights_bin(
                    [("kkp_idA", [7.0f32, 8.0, 9.0].as_slice()), ("kkp_idB", [10.0f32, 11.0].as_slice())].into_iter(),
                );
                fs::write(dir.join("slow.bin"), slow_bytes).unwrap();
                fs::write(dir.join("step_ranger.txt"), "kkp_idA,12\nkkp_idB,18\n").unwrap();
            }
        }

        // Bundle
        let mut bundled: Vec<u8> = Vec::new();
        for comp in ["kk", "kkp", "kpp"] {
            bundle_component_state(&mut bundled, comp, &tmp.join(format!("{comp}_orig/optimiser_state"))).unwrap();
        }

        // Re-parse and unbundle each component back to its own dir
        let parsed = parse_model_weights_bin(&bundled).unwrap();
        for comp in ["kk", "kkp", "kpp"] {
            let dst = tmp.join(format!("{comp}_restored")).join("optimiser_state");
            unbundle_component_state(&parsed, comp, &dst).unwrap();
            for section in ["weights", "momentum", "velocity"] {
                let orig = fs::read(tmp.join(format!("{comp}_orig/optimiser_state/{section}.bin"))).unwrap();
                let restored = fs::read(dst.join(format!("{section}.bin"))).unwrap();
                assert_eq!(orig, restored, "{comp}/{section}: round-trip mismatch");
            }
            if comp == "kkp" {
                let orig = fs::read(tmp.join("kkp_orig/optimiser_state/slow.bin")).unwrap();
                let restored = fs::read(dst.join("slow.bin")).unwrap();
                assert_eq!(orig, restored, "kkp/slow: round-trip mismatch");
                let step = fs::read_to_string(dst.join("step_ranger.txt")).unwrap();
                assert_eq!(step, "kkp_idA,12\nkkp_idB,18\n");
            } else {
                assert!(dst.join("slow.bin").is_file(), "{comp}/slow fallback should be written");
                assert!(!dst.join("step_ranger.txt").exists(), "{comp}/step_ranger should remain absent");
            }
        }

        let _ = fs::remove_dir_all(&tmp);
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
