//! Teacher-path resolution shared by the BulletOu trainer binaries.
//!
//! A `--teacher` argument may be:
//! - a single file with one of the supported extensions
//!   (`.hcpe` / `.hcpe3` / `.pack` / `.psv`),
//! - a directory containing such files (all matching files are concatenated,
//!   sorted by filename), or
//! - a comma-separated list of either.
//!
//! [`expand_teacher`] turns the user-supplied string into a concrete
//! `Vec<String>` of file paths. [`infer_data_format`] then classifies the
//! resulting list into a [`DataFormat`] enum that the caller dispatches on.

use std::path::Path;

/// Supported teacher-file extensions (lowercase, no leading dot).
pub const TEACHER_EXTS: &[&str] = &["hcpe", "hcpe3", "pack", "psv"];

/// Teacher file format inferred from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// dlshogi-style `.hcpe` (38-byte fixed-length).
    Hcpe,
    /// dlshogi-style `.hcpe3` (per-game variable-length).
    Hcpe3,
    /// YaneuraOu-ScriptCollection `gensfen` `.pack` (per-game variable-length).
    Pack,
    /// Flat `PackedSfenValue` dump (`.psv`, 40-byte fixed-length).
    Psv,
}

/// Resolve a `--teacher` argument into a concrete list of file paths.
///
/// See the module-level docs for the resolution rules. Returns an `Err` with
/// a user-friendly message when a path does not exist or a directory has no
/// matching files.
pub fn expand_teacher(teacher: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for part in teacher.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let path = Path::new(part);
        if path.is_dir() {
            let mut found: Vec<String> = Vec::new();
            let entries =
                std::fs::read_dir(path).map_err(|e| format!("failed to read directory {part}: {e}"))?;
            for entry in entries {
                let entry = entry.map_err(|e| format!("failed to enumerate directory {part}: {e}"))?;
                let p = entry.path();
                if !p.is_file() {
                    continue;
                }
                let ext = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
                if let Some(e) = ext.as_deref() {
                    if TEACHER_EXTS.contains(&e) {
                        found.push(p.to_string_lossy().into_owned());
                    }
                }
            }
            if found.is_empty() {
                return Err(format!(
                    "no teacher files found in directory {part}\n  expected files with extension: .hcpe / .hcpe3 / .pack / .psv"
                ));
            }
            found.sort();
            out.extend(found);
        } else if path.is_file() {
            out.push(part.to_string());
        } else {
            return Err(format!("teacher path does not exist: {part}"));
        }
    }
    if out.is_empty() {
        return Err("no teacher paths provided".to_string());
    }
    Ok(out)
}

/// Infer the [`DataFormat`] common to all the given paths.
///
/// Returns an `Err` if a path has an unrecognised extension, or if multiple
/// paths have different extensions (mixed-format teacher sets are rejected
/// because the trainer dispatches one loader for the whole batch).
pub fn infer_data_format(paths: &[&str]) -> Result<DataFormat, String> {
    let mut found: Option<DataFormat> = None;
    for p in paths {
        let ext = Path::new(p).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
        let fmt = match ext.as_deref() {
            Some("hcpe") => DataFormat::Hcpe,
            Some("hcpe3") => DataFormat::Hcpe3,
            Some("pack") => DataFormat::Pack,
            Some("psv") => DataFormat::Psv,
            _ => {
                return Err(format!(
                    "cannot infer data format from path: {p}\n  expected file extension: .hcpe / .hcpe3 / .pack / .psv"
                ));
            }
        };
        match found {
            None => found = Some(fmt),
            Some(prev) if prev == fmt => {}
            Some(prev) => {
                return Err(format!("mixed data formats: {p} is {fmt:?} but previous file(s) were {prev:?}"));
            }
        }
    }
    found.ok_or_else(|| "no data files provided".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bulletou_teacher_test_{name}"));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn expand_single_file() {
        let d = tmp_dir("single_file");
        let f = d.join("a.hcpe");
        fs::write(&f, b"x").unwrap();
        let got = expand_teacher(f.to_str().unwrap()).unwrap();
        assert_eq!(got, vec![f.to_string_lossy().into_owned()]);
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn expand_directory_enumerates_matching_files_sorted() {
        let d = tmp_dir("dir_enum");
        for n in ["c.hcpe", "a.hcpe", "b.hcpe", "ignored.txt"] {
            fs::write(d.join(n), b"x").unwrap();
        }
        let got = expand_teacher(d.to_str().unwrap()).unwrap();
        assert_eq!(got.len(), 3);
        assert!(got[0].ends_with("a.hcpe"));
        assert!(got[1].ends_with("b.hcpe"));
        assert!(got[2].ends_with("c.hcpe"));
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn expand_empty_directory_errors() {
        let d = tmp_dir("empty_dir");
        // Put a non-matching file so the dir is not empty but no teacher files.
        fs::write(d.join("readme.txt"), b"").unwrap();
        let err = expand_teacher(d.to_str().unwrap()).unwrap_err();
        assert!(err.contains("no teacher files"));
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn expand_nonexistent_errors() {
        let d = tmp_dir("does_not_exist");
        fs::remove_dir_all(&d).unwrap(); // sweep
        let err = expand_teacher(d.to_str().unwrap()).unwrap_err();
        assert!(err.contains("does not exist"));
    }

    #[test]
    fn infer_format_uniform_ok() {
        assert_eq!(infer_data_format(&["a.hcpe", "b.hcpe"]).unwrap(), DataFormat::Hcpe);
        assert_eq!(infer_data_format(&["a.pack"]).unwrap(), DataFormat::Pack);
        assert_eq!(infer_data_format(&["a.psv"]).unwrap(), DataFormat::Psv);
        assert_eq!(infer_data_format(&["a.HCPE3"]).unwrap(), DataFormat::Hcpe3);
    }

    #[test]
    fn infer_format_mixed_errors() {
        let err = infer_data_format(&["a.hcpe", "b.pack"]).unwrap_err();
        assert!(err.contains("mixed"));
    }

    #[test]
    fn infer_format_unknown_ext_errors() {
        let err = infer_data_format(&["a.txt"]).unwrap_err();
        assert!(err.contains("cannot infer"));
    }
}
