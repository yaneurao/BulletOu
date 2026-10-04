//! Append-only, epoch-scoped audit shared by FT/L1/L2 revival.
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const HEADER: &str = "epoch,run,layer,bucket,unit,pair,positions,upper_hits,zero_hits,contribution,relative_contribution,selected,contribution_threshold";

pub struct RevivalAudit {
    pub path: PathBuf,
    pub epoch: usize,
    pub run: usize,
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
        Ok(Self { path, epoch, run })
    }

    // Each layer persists its selection before mutating weights, as before.
    // One RevivalAudit is shared by all enabled layers in an epoch invocation.
    pub fn append(&self, layer: &str, rows: impl IntoIterator<Item = Row>) -> Result<(), String> {
        assert!(matches!(layer, "FT" | "L1" | "L2"));
        let mut report = String::new();
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
        write().map_err(|e| format!("{}: {e}", self.path.display()))
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
