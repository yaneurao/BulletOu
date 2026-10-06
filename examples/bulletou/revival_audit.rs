//! Append-only, epoch-scoped audit shared by FT/L1/L2 revival.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const HEADER: &str = "epoch,run,layer,bucket,unit,pair,positions,upper_hits,zero_hits,contribution,relative_contribution,selected,contribution_threshold";

pub struct RevivalAudit {
    pub path: PathBuf,
    pub epoch: usize,
    pub run: usize,
    summaries: RefCell<BTreeMap<String, LayerSummary>>,
}

struct LayerSummary {
    eligible: usize,
    reset_candidates: usize,
    mean: Option<f64>,
    threshold: f64,
    revived: Option<usize>,
}

fn legacy_summary_header() -> String {
    let mut header = "epoch,run".to_string();
    for layer in ["ft", "l1", "l2"] {
        for field in ["eligible", "revived", "revive_rate", "mean_relative_contribution", "contribution_threshold"] {
            header.push_str(&format!(",{layer}_{field}"));
        }
    }
    header
}

fn previous_summary_header() -> String {
    format!("{},ft_reset_candidates,l1_reset_candidates,l2_reset_candidates", legacy_summary_header())
}

fn summary_header() -> String {
    let mut header = "epoch,run".to_string();
    for layer in ["ft", "l1", "l2"] {
        for field in ["eligible", "reset_candidates", "reset_candidate_rate", "mean_relative_contribution", "contribution_threshold"] {
            header.push_str(&format!(",{layer}_{field}"));
        }
    }
    header.push_str(",ft_revived,l1_revived,l2_revived");
    header
}

// Rearrange previous schemas without guessing unknown historical candidate counts.
// Replace only after the complete upgraded file has been flushed successfully.
fn upgrade_summary(path: &Path) -> Result<(), String> {
    let existing = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.to_string()),
    };
    if existing.is_empty() { return Ok(()); }
    let mut lines = existing.lines();
    let header = lines.next().unwrap();
    if header == summary_header() { return Ok(()); }
    let previous = header == previous_summary_header();
    if !previous && header != legacy_summary_header() {
        return Err("unexpected summary header; existing file was not modified".into());
    }
    let mut upgraded = format!("{}\n", summary_header());
    for line in lines {
        if line.is_empty() { continue; }
        let old: Vec<_> = line.split(',').collect();
        if old.len() != if previous { 20 } else { 17 } {
            return Err("invalid legacy summary record; existing file was not modified".into());
        }
        let mut fields: Vec<String> = old[..17].iter().map(|v| v.to_string()).collect();
        for (layer, base) in [2, 7, 12].into_iter().enumerate() {
            // Actual reset counts were inside each layer's block in both old schemas.
            fields.push(old[base + 1].to_string());
            let candidates = if previous { old[17 + layer] } else { "" };
            fields[base + 1] = candidates.to_string();
            fields[base + 2] = if candidates.is_empty() {
                String::new()
            } else {
                let eligible: usize = old[base].parse().map_err(|_| "invalid legacy eligible count")?;
                let count: usize = candidates.parse().map_err(|_| "invalid legacy candidate count")?;
                if count > eligible { return Err("legacy candidate count exceeds eligible count".into()); }
                if eligible == 0 { String::new() } else { format!("{:.10}", count as f64 / eligible as f64) }
            };
        }
        upgraded.push_str(&fields.join(","));
        upgraded.push('\n');
    }
    let temporary = path.with_extension(format!("csv.upgrade-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos()));
    let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary).map_err(|e| e.to_string())?;
    let result = (|| -> std::io::Result<()> {
        file.write_all(upgraded.as_bytes())?;
        file.sync_all()
    })();
    drop(file);
    if let Err(e) = result {
        let _ = fs::remove_file(&temporary);
        return Err(e.to_string());
    }
    if let Err(e) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(e.to_string());
    }
    Ok(())
}

// Shared by startup and row appends so both use the same schema migration.
fn open_summary_file(path: &Path) -> Result<File, String> {
    let header = summary_header();
    upgrade_summary(path)?;
    fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    let mut file =
        OpenOptions::new().create(true).append(true).read(true).open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() == 0 {
        writeln!(file, "{header}").map_err(|e| e.to_string())?;
    } else {
        let mut existing = String::new();
        BufReader::new(&file).read_line(&mut existing).map_err(|e| e.to_string())?;
        if existing.trim_end() != header {
            return Err("unexpected summary header; existing file was not modified".into());
        }
        use std::io::{Read, Seek, SeekFrom};
        file.seek(SeekFrom::End(-1)).map_err(|e| e.to_string())?;
        let mut last = [0];
        file.read_exact(&mut last).map_err(|e| e.to_string())?;
        if last[0] != b'\n' {
            writeln!(file).map_err(|e| e.to_string())?;
        }
    }
    Ok(file)
}

pub fn ensure_summary_header(output_dir: &Path) -> Result<(), String> {
    let path = output_dir.join("revive-summary.csv");
    let mut file = open_summary_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    file.flush().map_err(|e| format!("{}: {e}", path.display()))
}

pub struct Row {
    pub bucket: usize,
    pub unit: Option<usize>,
    pub pair: Option<usize>,
    pub positions: usize,
    pub upper_hits: Option<usize>,
    pub zero_hits: Option<usize>,
    pub contribution: f64,
    pub relative_contribution: f64,
    pub selected: bool,
    pub threshold: f64,
}

impl RevivalAudit {
    pub fn new(output: &Path, epoch: usize) -> Result<Self, String> {
        let path = output.join("revive.csv");
        let mut max_run = 0usize;
        match File::open(&path) {
            Ok(file) => {
                let mut lines = BufReader::new(file).lines();
                if let Some(header) = lines.next() {
                    if header.map_err(|e| format!("{}: {e}", path.display()))? != HEADER {
                        return Err(format!(
                            "{}: unexpected revival CSV header; existing file was not modified",
                            path.display()
                        ));
                    }
                }
                for (i, line) in lines.enumerate() {
                    let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
                    if line.is_empty() {
                        continue;
                    }
                    let fields: Vec<_> = line.split(',').collect();
                    let parse = || -> Option<(usize, usize)> {
                        if fields.len() != 13 {
                            return None;
                        }
                        Some((fields[0].parse().ok()?, fields[1].parse().ok()?))
                    };
                    let (saved_epoch, run) = parse().ok_or_else(|| {
                        format!(
                            "{}: invalid revival record at line {}; existing file was not modified",
                            path.display(),
                            i + 2
                        )
                    })?;
                    if saved_epoch == epoch {
                        max_run = max_run.max(run);
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
        let run = max_run.checked_add(1).ok_or("revival run number overflow")?;
        Ok(Self { path, epoch, run, summaries: RefCell::new(BTreeMap::new()) })
    }

    // Each layer persists its selection before mutating weights, as before.
    // One RevivalAudit is shared by all measured layers in an epoch invocation.
    pub fn append(&self, layer: &str, rows: impl IntoIterator<Item = Row>) -> Result<(), String> {
        assert!(matches!(layer, "FT" | "L1" | "L2"));
        let mut report = String::new();
        let rows: Vec<_> = rows.into_iter().collect();
        let threshold = rows.first().map(|r| r.threshold).unwrap_or(0.0);
        let mut values = Vec::new();
        let mut selected_pairs = std::collections::BTreeSet::new();
        let mut reset_candidates = 0;
        if layer == "FT" {
            // The shared pair is eligible only when EVERY bucket is sampled.
            // Its decision score is the maximum relative contribution over buckets.
            if !rows.is_empty() && rows.iter().all(|r| r.positions >= 1024) {
                let mut pairs = BTreeMap::<usize, f64>::new();
                for row in &rows {
                    let value = pairs.entry(row.pair.expect("FT pair")).or_default();
                    *value = value.max(row.relative_contribution);
                    if row.selected { selected_pairs.insert(row.pair.unwrap()); }
                }
                values.extend(pairs.into_values());
                reset_candidates = selected_pairs.len();
            }
        } else {
            values.extend(rows.iter().filter(|r| r.positions >= 1024).map(|r| r.relative_contribution));
            reset_candidates = rows.iter().filter(|r| r.positions >= 1024 && r.selected).count();
        }
        let summary = LayerSummary {
            eligible: values.len(),
            reset_candidates,
            mean: (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64),
            threshold,
            revived: None,
        };
        let optional = |v: Option<usize>| v.map(|v| v.to_string()).unwrap_or_default();
        for row in rows {
            report.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{:.10},{:.10},{},{}\n",
                self.epoch,
                self.run,
                layer,
                row.bucket,
                optional(row.unit),
                optional(row.pair),
                row.positions,
                optional(row.upper_hits),
                optional(row.zero_hits),
                row.contribution,
                row.relative_contribution,
                row.selected,
                row.threshold
            ));
        }
        let write = || -> std::io::Result<()> {
            fs::create_dir_all(self.path.parent().unwrap())?;
            let mut file = OpenOptions::new().create(true).append(true).read(true).open(&self.path)?;
            let len = file.metadata()?.len();
            if len == 0 {
                writeln!(file, "{HEADER}")?;
            } else {
                // A valid last record may have been written without a newline.
                use std::io::{Read, Seek, SeekFrom};
                file.seek(SeekFrom::End(-1))?;
                let mut last = [0];
                file.read_exact(&mut last)?;
                if last[0] != b'\n' {
                    writeln!(file)?;
                }
            }
            file.write_all(report.as_bytes())?;
            file.flush()
        };
        write().map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.summaries.borrow_mut().insert(layer.to_string(), summary);
        Ok(())
    }

    /// Called after measurement only, or after reset AND mean compensation succeed.
    pub fn complete_layer(&self, layer: &str, revived: usize) -> Result<(), String> {
        let mut summaries = self.summaries.borrow_mut();
        let summary = summaries.get_mut(layer).ok_or("missing revival calibration summary")?;
        if revived > summary.eligible {
            return Err("revived count exceeds eligible count".into());
        }
        summary.revived = Some(revived);
        Ok(())
    }

    pub fn append_summary(&self) -> Result<PathBuf, String> {
        let path = self.path.with_file_name("revive-summary.csv");
        let summaries = self.summaries.borrow();
        let mut record = format!("{},{}", self.epoch, self.run);
        for layer in ["FT", "L1", "L2"] {
            if let Some(s) = summaries.get(layer) {
                s.revived.ok_or("revival summary requested before successful completion")?;
                let rate =
                    if s.eligible == 0 { String::new() } else { format!("{:.10}", s.reset_candidates as f64 / s.eligible as f64) };
                let mean = s.mean.map(|v| format!("{v:.10}")).unwrap_or_default();
                record.push_str(&format!(",{},{},{rate},{mean},{:.10}", s.eligible, s.reset_candidates, s.threshold));
            } else {
                record.push_str(",,,,,");
            }
        }
        for layer in ["FT", "L1", "L2"] {
            record.push(',');
            if let Some(s) = summaries.get(layer) {
                record.push_str(&s.revived.ok_or("revival summary requested before successful completion")?.to_string());
            }
        }
        let write = || -> Result<(), String> {
            let mut file = open_summary_file(&path)?;
            writeln!(file, "{record}").map_err(|e| e.to_string())?;
            file.flush().map_err(|e| e.to_string())
        };
        write().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                "bulletou-revival-audit-{}-{}",
                std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
            )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn startup_summary_header_is_created_once_and_preserves_records() {
        let t = Temp::new();
        ensure_summary_header(&t.0).unwrap();
        let path = t.0.join("revive-summary.csv");
        let header_only = format!("{}\n", summary_header());
        assert_eq!(fs::read_to_string(&path).unwrap(), header_only);
        assert!(!t.0.join("revive.csv").exists());
        ensure_summary_header(&t.0).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), header_only);

        let a = RevivalAudit::new(&t.0, 1).unwrap();
        a.append("L2", [row(false)]).unwrap();
        a.complete_layer("L2", 0).unwrap();
        a.append_summary().unwrap();
        let with_record = fs::read_to_string(&path).unwrap();
        assert_eq!(with_record.lines().count(), 2);
        ensure_summary_header(&t.0).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), with_record);
    }

    #[test]
    fn startup_summary_initializes_empty_file_and_upgrades_legacy_header() {
        let t = Temp::new();
        fs::create_dir_all(&t.0).unwrap();
        let path = t.0.join("revive-summary.csv");
        fs::write(&path, "").unwrap();
        ensure_summary_header(&t.0).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), format!("{}\n", summary_header()));
        fs::write(&path, format!("{}\n", legacy_summary_header())).unwrap();
        ensure_summary_header(&t.0).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), format!("{}\n", summary_header()));
        fs::write(&path, "unexpected,header\n").unwrap();
        assert!(ensure_summary_header(&t.0).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "unexpected,header\n");
    }

    fn row(ft: bool) -> Row {
        Row {
            bucket: 2,
            unit: (!ft).then_some(3),
            pair: ft.then_some(4),
            positions: 1234,
            upper_hits: (!ft).then_some(0),
            zero_hits: (!ft).then_some(12),
            contribution: 0.12,
            relative_contribution: 0.003,
            selected: false,
            threshold: 0.01,
        }
    }
    #[test]
    fn unified_audit_summary_counts_completed_units_and_excludes_small_buckets() {
        let t = Temp::new();
        let a = RevivalAudit::new(&t.0, 3).unwrap();
        let mut x = row(false);
        x.relative_contribution = 0.2;
        x.selected = true;
        let mut y = row(false);
        y.unit = Some(4);
        y.relative_contribution = 0.6;
        let mut excluded = row(false);
        excluded.positions = 1023;
        a.append("L1", [x, y, excluded]).unwrap();
        assert!(a.append_summary().is_err());
        assert!(!t.0.join("revive-summary.csv").exists());
        a.complete_layer("L1", 1).unwrap();
        let path = a.append_summary().unwrap();
        let original = fs::read_to_string(&path).unwrap();
        let fields: Vec<_> = original.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(fields.len(), 20);
        assert_eq!(&fields[2..7], &["", "", "", "", ""]);
        assert_eq!(&fields[7..12], &["2", "1", "0.5000000000", "0.4000000000", "0.0100000000"]);
        assert_eq!(&fields[12..17], &["", "", "", "", ""]);
        let retry = RevivalAudit::new(&t.0, 3).unwrap();
        retry.append("L2", [row(false)]).unwrap();
        retry.complete_layer("L2", 0).unwrap();
        fs::write(&path, original.trim_end()).unwrap();
        retry.append_summary().unwrap();
        let result = fs::read_to_string(&path).unwrap();
        assert!(result.starts_with(&original));
        assert!(result.lines().nth(2).unwrap().starts_with("3,2,"));
        assert_eq!(result.lines().count(), 3);
    }

    #[test]
    fn unified_audit_summary_ft_uses_unique_pairs_and_max_bucket_contribution() {
        let t = Temp::new();
        for epoch in [1, 2] {
            let a = RevivalAudit::new(&t.0, epoch).unwrap();
            let mut rows = Vec::new();
            for bucket in 0..2 {
                for pair in 0..2 {
                    let mut r = row(true);
                    r.bucket = bucket;
                    r.pair = Some(pair);
                    r.selected = pair == 0;
                    r.relative_contribution = if pair == 0 { 0.1 + bucket as f64 * 0.2 } else { 0.7 };
                    if epoch == 2 && bucket == 1 {
                        r.positions = 100;
                    }
                    rows.push(r);
                }
            }
            a.append("FT", rows).unwrap();
            a.complete_layer("FT", if epoch == 1 { 1 } else { 0 }).unwrap();
            let path = a.append_summary().unwrap();
            let text = fs::read_to_string(path).unwrap();
            let fields: Vec<_> = text.lines().last().unwrap().split(',').collect();
            if epoch == 1 {
                assert_eq!(&fields[2..7], &["2", "1", "0.5000000000", "0.5000000000", "0.0100000000"]);
                assert_eq!(fields[17], "1");
            } else {
                assert_eq!(&fields[2..7], &["0", "0", "", "", "0.0100000000"]);
                assert_eq!(fields[17], "0");
            }
        }
    }

    #[test]
    fn measurement_only_counts_candidates_without_reporting_resets() {
        let t = Temp::new();
        let a = RevivalAudit::new(&t.0, 1).unwrap();
        let mut selected = row(false);
        selected.selected = true;
        let mut small = row(false);
        small.positions = 1023;
        small.selected = true;
        a.append("L2", [selected, small, row(false)]).unwrap();
        a.complete_layer("L2", 0).unwrap();
        let mut pairs = Vec::new();
        for bucket in 0..2 {
            let mut r = row(true);
            r.bucket = bucket;
            r.selected = true;
            pairs.push(r);
        }
        a.append("FT", pairs).unwrap();
        a.complete_layer("FT", 0).unwrap();
        let text = fs::read_to_string(a.append_summary().unwrap()).unwrap();
        let fields: Vec<_> = text.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(fields[3], "1");
        assert_eq!(fields[13], "1");
        assert_eq!(fields[14], "0.5000000000");
        assert_eq!(&fields[17..], &["0", "", "0"]);
    }

    #[test]
    fn legacy_summary_is_extended_with_unknown_historical_candidates() {
        let t = Temp::new();
        fs::create_dir_all(&t.0).unwrap();
        let path = t.0.join("revive-summary.csv");
        let old_record = "1,1,2,1,0.5000000000,0.4000000000,0.0100000000,,,,,,,,,,";
        assert_eq!(old_record.split(',').count(), 17);
        fs::write(&path, format!("{}\n{old_record}", legacy_summary_header())).unwrap();
        let a = RevivalAudit::new(&t.0, 2).unwrap();
        a.append("L1", [row(false)]).unwrap();
        a.complete_layer("L1", 0).unwrap();
        a.append_summary().unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().next().unwrap(), summary_header());
        let historical: Vec<_> = text.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(&historical[2..7], &["2", "", "", "0.4000000000", "0.0100000000"]);
        assert_eq!(&historical[17..], &["1", "", ""]);
        assert_eq!(text.lines().count(), 3);
        assert_eq!(fs::read_dir(&t.0).unwrap().count(), 2);
        let malformed = format!("{}\n1,1", legacy_summary_header());
        fs::write(&path, &malformed).unwrap();
        assert!(a.append_summary().is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), malformed);
    }

    #[test]
    fn previous_summary_moves_actual_resets_to_the_end_and_keeps_candidates() {
        let t = Temp::new();
        fs::create_dir_all(&t.0).unwrap();
        let path = t.0.join("revive-summary.csv");
        let old = "1,1,4,0,0.0000000000,0.2,0.01,2,1,0.5000000000,0.4,0.01,0,0,,,0.01,2,1,0";
        fs::write(&path, format!("{}\n{old}", previous_summary_header())).unwrap();
        upgrade_summary(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().next().unwrap(), summary_header());
        let fields: Vec<_> = text.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(&fields[2..7], &["4", "2", "0.5000000000", "0.2", "0.01"]);
        assert_eq!(&fields[7..12], &["2", "1", "0.5000000000", "0.4", "0.01"]);
        assert_eq!(&fields[12..17], &["0", "0", "", "", "0.01"]);
        assert_eq!(&fields[17..], &["0", "1", "0"]);
        upgrade_summary(&path).unwrap();
        assert_eq!(fs::read_to_string(path).unwrap(), text);
    }

    #[test]
    fn measurement_columns_are_identical_with_or_without_resets() {
        let t = Temp::new();
        for (epoch, revived) in [(1, 0), (2, 1)] {
            let a = RevivalAudit::new(&t.0, epoch).unwrap();
            let mut candidate = row(false);
            candidate.selected = true;
            a.append("L2", [candidate, row(false)]).unwrap();
            a.complete_layer("L2", revived).unwrap();
            a.append_summary().unwrap();
        }
        let text = fs::read_to_string(t.0.join("revive-summary.csv")).unwrap();
        let rows: Vec<Vec<_>> = text.lines().skip(1).map(|line| line.split(',').collect()).collect();
        assert_eq!(&rows[0][2..17], &rows[1][2..17]);
        assert_eq!(rows[0][19], "0");
        assert_eq!(rows[1][19], "1");
    }

    #[test]
    fn unified_audit_summary_preserves_incompatible_file() {
        let t = Temp::new();
        let a = RevivalAudit::new(&t.0, 1).unwrap();
        a.append("L1", [row(false)]).unwrap();
        a.complete_layer("L1", 0).unwrap();
        let path = t.0.join("revive-summary.csv");
        fs::write(&path, "manual,header\nkeep,this\n").unwrap();
        assert!(a.append_summary().is_err());
        assert_eq!(fs::read_to_string(path).unwrap(), "manual,header\nkeep,this\n");
    }

    #[test]
    fn unified_audit_appends_all_layers_and_numbers_per_epoch() {
        let t = Temp::new();
        let a = RevivalAudit::new(&t.0, 3).unwrap();
        a.append("FT", [row(true)]).unwrap();
        a.append("L1", [row(false)]).unwrap();
        a.append("L2", [row(false)]).unwrap();
        let original = fs::read_to_string(&a.path).unwrap();
        assert_eq!(original.lines().count(), 4);
        assert!(original.contains("3,1,FT,2,,4,1234,,,"));
        assert!(original.contains("3,1,L1,2,3,,1234,0,12,"));
        let next = RevivalAudit::new(&t.0, 4).unwrap();
        assert_eq!(next.run, 1);
        next.append("L2", [row(false)]).unwrap();
        let retry = RevivalAudit::new(&t.0, 3).unwrap();
        assert_eq!(retry.run, 2);
        retry.append("FT", [row(true)]).unwrap();
        let result = fs::read_to_string(&a.path).unwrap();
        assert!(result.starts_with(&original));
        assert!(result.contains("3,2,FT,"));
        assert_eq!(result.matches(HEADER).count(), 1);
        assert_eq!(fs::read_dir(&t.0).unwrap().count(), 1);
    }
    #[test]
    fn unified_audit_preserves_legacy_and_missing_final_newline() {
        let t = Temp::new();
        fs::create_dir_all(&t.0).unwrap();
        fs::write(t.0.join("ft-revive-1.csv"), "legacy").unwrap();
        let a = RevivalAudit::new(&t.0, 0).unwrap();
        a.append("FT", [row(true)]).unwrap();
        let text = fs::read_to_string(&a.path).unwrap();
        fs::write(&a.path, text.trim_end()).unwrap();
        let a = RevivalAudit::new(&t.0, 0).unwrap();
        assert_eq!(a.run, 2);
        a.append("L1", [row(false)]).unwrap();
        assert_eq!(fs::read_to_string(&a.path).unwrap().lines().count(), 3);
        assert_eq!(fs::read_to_string(t.0.join("ft-revive-1.csv")).unwrap(), "legacy");
    }
    #[test]
    fn unified_audit_rejects_incompatible_or_incomplete_csv_without_overwrite() {
        let t = Temp::new();
        fs::create_dir_all(&t.0).unwrap();
        for text in ["other,header\n".to_string(), format!("{HEADER}\n3,1,FT,")] {
            fs::write(t.0.join("revive.csv"), &text).unwrap();
            assert!(RevivalAudit::new(&t.0, 3).is_err());
            assert_eq!(fs::read_to_string(t.0.join("revive.csv")).unwrap(), text);
        }
    }
}
