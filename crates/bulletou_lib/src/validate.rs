//! Held-out validation against an hcpe test set.
//!
//! Used by the `bulletou` example to compute "value-sign agreement"
//! accuracy after training: random-pick N positions from a test
//! `.hcpe`, run them through the trained model, then for each one
//! check whether the network's raw output and the teacher's
//! centipawn score have the same sign (= both predict the side-to-
//! move winning, or both predict losing).
//!
//! This is a pure-Rust module that does **no GPU work** — it only
//! reads bytes, decodes them into `PackedSfenValue`, and computes the
//! accuracy from a model-output array supplied by the caller. The
//! GPU forward pass is the caller's responsibility (see
//! `ValueTrainer::eval_packed_batch`).

use std::fs::File;
use std::io::Read;

use crate::shogi::PackedSfenValue;
use crate::value::loader::hcpe::{HCPE_RECORD_SIZE, decode_hcpe_record};

/// Outcome of a sign-agreement validation pass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AccuracyReport {
    /// Number of positions actually compared (= sampled minus
    /// `score_drop_abs`-filtered minus draws-in-teacher).
    pub compared: usize,
    /// Of the compared positions, how many had matching sign.
    pub sign_matches: usize,
    /// Number of sampled positions whose teacher score was 0
    /// (draw/stalemate sentinel) and were skipped from sign checks.
    pub draws_in_teacher: usize,
    /// Number of sampled positions whose teacher score's absolute
    /// value was at or above the `score_drop_abs` threshold (mate
    /// stamps) and were skipped.
    pub filtered_by_score_cap: usize,
}

impl AccuracyReport {
    /// `sign_matches / compared`, or `NaN` if no positions were compared.
    pub fn accuracy(&self) -> f32 {
        if self.compared == 0 { f32::NAN } else { self.sign_matches as f32 / self.compared as f32 }
    }
}

/// Compute sign-agreement accuracy from parallel arrays.
///
/// `model_outputs[i]` is the raw network output for position `i`; its
/// **sign** says which side the model prefers (positive = side-to-move
/// winning, after the canonical `out.sigmoid().squared_error(tgt)`
/// loss). `teacher_scores[i]` is the centipawn score from the test set,
/// likewise sign-encodes the teacher's preference.
///
/// `score_drop_abs == Some(cap)` filters out positions with
/// `|teacher_score| >= cap` (= mate stamps such as ±32000).
/// `score_drop_abs == None` keeps all positions.
///
/// Positions whose teacher score is exactly `0` are also skipped (the
/// "no opinion" case is undefined for sign agreement).
pub fn compute_sign_accuracy(
    model_outputs: &[f32],
    teacher_scores: &[i16],
    score_drop_abs: Option<u16>,
) -> AccuracyReport {
    assert_eq!(
        model_outputs.len(),
        teacher_scores.len(),
        "model_outputs and teacher_scores length mismatch"
    );
    let mut report = AccuracyReport::default();
    for (m, &s) in model_outputs.iter().zip(teacher_scores.iter()) {
        if let Some(cap) = score_drop_abs {
            if s.unsigned_abs() >= cap {
                report.filtered_by_score_cap += 1;
                continue;
            }
        }
        if s == 0 {
            report.draws_in_teacher += 1;
            continue;
        }
        report.compared += 1;
        // Both signs positive or both negative → match.
        let model_sign_positive = *m > 0.0;
        let teacher_sign_positive = s > 0;
        if model_sign_positive == teacher_sign_positive {
            report.sign_matches += 1;
        }
    }
    report
}

/// Reservoir-sample `n` positions uniformly from the hcpe file at
/// `path`, decoding each into `PackedSfenValue`. Returns `Err` on I/O
/// failure or non-hcpe sized files.
///
/// `seed = 0` selects a time-based seed (= non-reproducible);
/// any other value uses that seed verbatim (= reproducible).
///
/// The file must be a flat `.hcpe` (38-byte fixed records). hcpe3 /
/// pack / psv are not supported here yet — for those, transcode to
/// hcpe first.
///
/// On disk-too-small (= file has fewer than `n` valid records), all
/// available records are returned and the caller can decide whether to
/// proceed.
pub fn read_random_hcpe_positions(
    path: &str,
    n: usize,
    seed: u64,
) -> std::io::Result<Vec<PackedSfenValue>> {
    let meta = std::fs::metadata(path)?;
    let file_size = meta.len() as usize;
    if file_size % HCPE_RECORD_SIZE != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{path}: size {file_size} is not a multiple of HCPE_RECORD_SIZE ({HCPE_RECORD_SIZE}) — \
                 not a valid hcpe file?"
            ),
        ));
    }
    let total_records = file_size / HCPE_RECORD_SIZE;
    if total_records == 0 {
        return Ok(Vec::new());
    }

    // Pick `n` distinct record indices (or all of them if total <= n).
    let mut rng = SeededXorShift::from_seed(seed);
    let indices: Vec<usize> = if total_records <= n {
        (0..total_records).collect()
    } else {
        // Floyd's algorithm for sampling without replacement, O(n).
        let mut chosen = std::collections::BTreeSet::new();
        for j in (total_records - n)..total_records {
            let pick = (rng.next_u64() as usize) % (j + 1);
            if !chosen.insert(pick) {
                chosen.insert(j);
            }
        }
        chosen.into_iter().collect()
    };

    // Read selected records. Sorting indices lets us do a single
    // sequential pass through the file (one seek + one read per record
    // group is good enough at this scale; reading a 35MB file fully into
    // memory is also fine but we avoid the alloc).
    let mut file = File::open(path)?;
    let mut out: Vec<PackedSfenValue> = Vec::with_capacity(indices.len());
    let mut rec = [0u8; HCPE_RECORD_SIZE];
    for idx in indices {
        let off = idx * HCPE_RECORD_SIZE;
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(off as u64))?;
        file.read_exact(&mut rec)?;
        if let Some(psv) = decode_hcpe_record(&rec) {
            out.push(psv);
        }
        // silently skip records whose HCP failed to decode (= corrupted)
    }
    Ok(out)
}

/// Seeded xorshift RNG. Same algorithm as the loader's `SimpleRand`
/// but takes a user-supplied seed for reproducibility.
struct SeededXorShift(u64);

impl SeededXorShift {
    fn from_seed(seed: u64) -> Self {
        if seed == 0 {
            // Treat 0 as "use a time-based seed" so the user's default
            // CLI behaviour gets a different sample on each run unless
            // they explicitly want determinism.
            let s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEADBEEFCAFEBABEu64)
                | 1; // xorshift requires non-zero state
            Self(s)
        } else {
            Self(seed)
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_all_match() {
        let m = [0.5, -0.3, 1.2, -0.01];
        let t = [200i16, -150, 800, -10];
        let r = compute_sign_accuracy(&m, &t, None);
        assert_eq!(r.compared, 4);
        assert_eq!(r.sign_matches, 4);
        assert!((r.accuracy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accuracy_half_mismatch() {
        let m = [0.5, -0.3, 1.2, -0.01];
        // first 2 match, last 2 flipped
        let t = [200i16, -150, -800, 10];
        let r = compute_sign_accuracy(&m, &t, None);
        assert_eq!(r.compared, 4);
        assert_eq!(r.sign_matches, 2);
        assert!((r.accuracy() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn accuracy_skips_teacher_draws() {
        let m = [0.5, 0.3, 1.2];
        let t = [200i16, 0, 0];
        let r = compute_sign_accuracy(&m, &t, None);
        assert_eq!(r.compared, 1, "two zero-score positions skipped");
        assert_eq!(r.sign_matches, 1);
        assert_eq!(r.draws_in_teacher, 2);
    }

    #[test]
    fn accuracy_score_drop_filters_mate_stamps() {
        let m = [0.5, 0.3, 1.2, -0.5];
        let t = [200i16, 32000, -32000, -10];
        let r = compute_sign_accuracy(&m, &t, Some(32000));
        assert_eq!(r.compared, 2);
        assert_eq!(r.sign_matches, 2);
        assert_eq!(r.filtered_by_score_cap, 2);
        assert_eq!(r.draws_in_teacher, 0);
    }

    #[test]
    fn accuracy_zero_when_empty() {
        let r = compute_sign_accuracy(&[], &[], None);
        assert_eq!(r.compared, 0);
        assert!(r.accuracy().is_nan());
    }

    #[test]
    fn seeded_rng_is_reproducible() {
        let mut a = SeededXorShift::from_seed(12345);
        let mut b = SeededXorShift::from_seed(12345);
        for _ in 0..20 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn read_random_hcpe_rejects_bad_size() {
        let tmp = std::env::temp_dir().join(format!(
            "bulletou-validate-test-bad-size-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&tmp, vec![0u8; 37]).unwrap(); // not a multiple of 38
        let path = tmp.to_str().unwrap();
        // Result::unwrap_err() needs T: Debug; PackedSfenValue is not, so
        // pattern-match instead.
        match read_random_hcpe_positions(path, 5, 1) {
            Ok(_) => panic!("expected InvalidData error, got Ok"),
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
