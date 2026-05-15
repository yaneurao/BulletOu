//! Held-out validation against an hcpe test set.
//!
//! Used by the `bulletou` example to compute "value-sign agreement"
//! accuracy after training: random-pick N positions from a test
//! `.hcpe`, run them through the trained model, then for each one
//! check whether the network's raw output and the actual **game
//! result** (win/loss for the side to move) have the same sign.
//!
//! Game result (= the outcome of the game the position came from) is
//! used rather than the teacher's centipawn score because the trainer's
//! loss target is itself derived from the score: comparing the model's
//! output sign against the score sign would mostly measure how well the
//! model fits the teacher rather than how well it predicts wins. Sign
//! agreement against the actual game result is harder and a more
//! honest measure of value-network quality.
//!
//! Drawn games (`game_result == 0`) are excluded from the accuracy
//! count because "win" / "loss" doesn't apply. Mate stamps (positions
//! whose teacher score abs ≥ `score_drop_abs`) are also excluded so
//! the accuracy and loss subsets stay consistent with the trainer's
//! `score_drop_abs` filter.
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
    /// Number of positions actually used for the accuracy comparison
    /// (= sampled minus `score_drop_abs`-filtered minus drawn games).
    pub compared: usize,
    /// Of the compared positions, how many had matching sign between
    /// the model's output and the game result.
    pub sign_matches: usize,
    /// Number of sampled positions whose game ended in a draw
    /// (`game_result == 0`); skipped from accuracy because there is no
    /// "winning side" to agree with. Still included in the loss subset.
    pub drawn_games: usize,
    /// Number of sampled positions whose teacher score's absolute
    /// value was at or above the `score_drop_abs` threshold (mate
    /// stamps); excluded from BOTH accuracy and loss.
    pub filtered_by_score_cap: usize,
    /// Number of positions used in the loss average (= sampled minus
    /// `score_drop_abs`-filtered, **including** drawn games so the
    /// metric is directly comparable to the trainer's training loss).
    pub loss_sampled: usize,
    /// Mean test-set loss over the `loss_sampled` subset. `None` when
    /// the caller didn't pass game results (= loss not requested) or
    /// when `loss_sampled == 0`.
    pub test_loss: Option<f32>,
}

impl AccuracyReport {
    /// `sign_matches / compared`, or `NaN` if no positions were compared.
    pub fn accuracy(&self) -> f32 {
        if self.compared == 0 { f32::NAN } else { self.sign_matches as f32 / self.compared as f32 }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute sign-agreement accuracy AND the matching test-set loss
/// from parallel arrays.
///
/// `model_outputs[i]` is the raw network output for position `i`; its
/// **sign** says which side the model prefers (positive = side-to-move
/// winning, after the canonical `out.sigmoid().squared_error(tgt)`
/// loss). `teacher_results[i]` is the actual game outcome from the
/// position's STM perspective: `+1` (STM won), `0` (draw), `-1` (STM
/// lost).
///
/// **Accuracy subset**: positions with `score_drop_abs`-filtered or
/// drawn (`teacher_results[i] == 0`) games are skipped. The remaining
/// positions are compared sign-vs-sign: model output > 0 should agree
/// with `teacher_results[i] > 0`. This measures how well the model
/// predicts the actual winner of the game, not how well it mimics the
/// teacher's score (which would be a much easier target).
///
/// **Loss subset**: same as the trainer — `score_drop_abs`-filtered
/// positions are skipped, drawn games are **kept**. The formula
/// matches `bullet_lib::value::loader::DefaultDataLoader::prepare`:
///
/// ```text
///   blend  = 1 - lambda
///   result_norm = result == +1 ? 1.0 : result == -1 ? 0.0 : 0.5
///   target = blend * result_norm + (1 - blend) * sigmoid(score / scale)
///   loss   = (sigmoid(model_out) - target)^2
/// ```
///
/// `score_drop_abs == Some(cap)` filters out positions with
/// `|teacher_score| >= cap` (= mate stamps such as ±32000); these are
/// excluded from BOTH accuracy and loss. `score_drop_abs == None`
/// keeps all positions.
///
/// When `teacher_results.is_empty()`, accuracy falls back to comparing
/// model output sign vs `teacher_scores[i]` sign (the legacy "score
/// agreement" metric, kept only so the unit tests can exercise the
/// loop without mocking game results). Production callers always
/// supply `teacher_results`.
pub fn compute_sign_accuracy(
    model_outputs: &[f32],
    teacher_scores: &[i16],
    teacher_results: &[i8],
    score_drop_abs: Option<u16>,
    lambda: f32,
    eval_scale: f32,
) -> AccuracyReport {
    assert_eq!(
        model_outputs.len(),
        teacher_scores.len(),
        "model_outputs and teacher_scores length mismatch"
    );
    let have_results = !teacher_results.is_empty();
    if have_results {
        assert_eq!(
            model_outputs.len(),
            teacher_results.len(),
            "model_outputs and teacher_results length mismatch"
        );
    }
    let mut report = AccuracyReport::default();
    let blend = 1.0 - lambda;
    let inv_scale = if eval_scale > 0.0 { 1.0 / eval_scale } else { 0.0025 };
    let mut loss_sum = 0.0f32;
    for (i, (m, &s)) in model_outputs.iter().zip(teacher_scores.iter()).enumerate() {
        if let Some(cap) = score_drop_abs {
            if s.unsigned_abs() >= cap {
                report.filtered_by_score_cap += 1;
                continue;
            }
        }
        // Accuracy: sign(model) vs sign(game_result), skipping draws.
        // When teacher_results is empty (test-only path), fall back to
        // sign(score) so the function still reports a number.
        let model_sign_positive = *m > 0.0;
        if have_results {
            let r = teacher_results[i];
            if r == 0 {
                report.drawn_games += 1;
            } else {
                report.compared += 1;
                let result_sign_positive = r > 0;
                if model_sign_positive == result_sign_positive {
                    report.sign_matches += 1;
                }
            }
        } else if s == 0 {
            report.drawn_games += 1;
        } else {
            report.compared += 1;
            if model_sign_positive == (s > 0) {
                report.sign_matches += 1;
            }
        }
        // Loss: include drawn games (matches the trainer's loss
        // averaging), exclude only mate stamps (already `continue`d
        // above).
        if have_results {
            let result_norm = match teacher_results[i].signum() {
                1 => 1.0,
                -1 => 0.0,
                _ => 0.5,
            };
            let score_norm = sigmoid(inv_scale * f32::from(s));
            let target = blend * result_norm + (1.0 - blend) * score_norm;
            let model_p = sigmoid(*m);
            let diff = model_p - target;
            loss_sum += diff * diff;
            report.loss_sampled += 1;
        }
    }
    if have_results && report.loss_sampled > 0 {
        report.test_loss = Some(loss_sum / report.loss_sampled as f32);
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
        let r = compute_sign_accuracy(&m, &t, &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 4);
        assert_eq!(r.sign_matches, 4);
        assert!((r.accuracy() - 1.0).abs() < 1e-6);
        assert_eq!(r.test_loss, None, "no results → no loss");
    }

    #[test]
    fn accuracy_half_mismatch() {
        let m = [0.5, -0.3, 1.2, -0.01];
        // first 2 match, last 2 flipped
        let t = [200i16, -150, -800, 10];
        let r = compute_sign_accuracy(&m, &t, &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 4);
        assert_eq!(r.sign_matches, 2);
        assert!((r.accuracy() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn accuracy_skips_drawn_games() {
        // game_result-driven path: positions whose game ended in a
        // draw are skipped from accuracy regardless of their score.
        let m = [0.5, 0.3, 1.2];
        let t = [200i16, -150, 800];
        let game = [1i8, 0, 0]; // first decisive (STM win), other two drawn
        let r = compute_sign_accuracy(&m, &t, &game, None, 1.0, 400.0);
        assert_eq!(r.compared, 1, "two drawn games skipped from accuracy");
        assert_eq!(r.sign_matches, 1, "model output > 0 agrees with STM win");
        assert_eq!(r.drawn_games, 2);
        // Loss is still computed across all 3 (draws contribute too).
        assert_eq!(r.loss_sampled, 3);
        assert!(r.test_loss.is_some(), "loss should be computed when results provided");
    }

    #[test]
    fn accuracy_falls_back_to_score_sign_when_no_results() {
        // Legacy path (no teacher_results): the function falls back to
        // comparing model sign vs teacher score sign and skipping
        // score==0 positions. Used only in unit tests.
        let m = [0.5, 0.3, 1.2];
        let t = [200i16, 0, 0];
        let r = compute_sign_accuracy(&m, &t, &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 1, "two zero-score positions skipped");
        assert_eq!(r.sign_matches, 1);
        assert_eq!(r.drawn_games, 2);
        assert_eq!(r.loss_sampled, 0, "no results → no loss");
    }

    #[test]
    fn accuracy_score_drop_filters_mate_stamps() {
        let m = [0.5, 0.3, 1.2, -0.5];
        let t = [200i16, 32000, -32000, -10];
        let r = compute_sign_accuracy(&m, &t, &[], Some(32000), 1.0, 400.0);
        assert_eq!(r.compared, 2);
        assert_eq!(r.sign_matches, 2);
        assert_eq!(r.filtered_by_score_cap, 2);
        assert_eq!(r.drawn_games, 0);
    }

    #[test]
    fn accuracy_zero_when_empty() {
        let r = compute_sign_accuracy(&[], &[], &[], None, 1.0, 400.0);
        assert_eq!(r.compared, 0);
        assert!(r.accuracy().is_nan());
        assert_eq!(r.test_loss, None);
    }

    #[test]
    fn test_loss_pure_eval_target() {
        // lambda=1.0, blend=0 → target = sigmoid(score/scale) only
        let m = [0.0, 0.0]; // sigmoid(0) = 0.5
        let t = [400i16, -400]; // sigmoid(±1) ≈ 0.731 / 0.269
        let r = compute_sign_accuracy(&m, &t, &[1, -1], None, 1.0, 400.0);
        assert_eq!(r.compared, 2);
        let loss = r.test_loss.expect("loss requested");
        // expected: ((0.5 - 0.731)^2 + (0.5 - 0.269)^2) / 2 ≈ 0.0533
        assert!((loss - 0.0533).abs() < 1e-3, "loss={loss}");
    }

    #[test]
    fn test_loss_pure_wdl_target() {
        // lambda=0.0, blend=1 → target = result/2 mapping (Win=1, Loss=0, Draw=0.5)
        let m = [0.0, 0.0, 0.0]; // sigmoid(0) = 0.5
        // results: +1 (win), -1 (loss), 0 (draw)
        let r = compute_sign_accuracy(&m, &[100i16, -100, 100], &[1, -1, 0], None, 0.0, 400.0);
        // draw position has teacher_score=100 (not 0) so accuracy still includes it,
        // but here we just check loss.
        // Targets: 1.0, 0.0, 0.5
        // Losses:  0.25, 0.25, 0.0
        // Mean: 0.1667
        let loss = r.test_loss.expect("loss requested");
        assert!((loss - 0.1666666).abs() < 1e-3, "loss={loss}");
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
