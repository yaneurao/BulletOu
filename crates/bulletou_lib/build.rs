use std::{env, fs, path::PathBuf, process::Command, time::SystemTime};

fn git_output(args: &[&str]) -> Option<String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let output = Command::new("git").arg("-C").arg(manifest_dir).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_string())
}

fn emit_git_rerun_hints() {
    let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") else {
        return;
    };
    let root = PathBuf::from(manifest_dir).join("../..");
    let git_dir = root.join(".git");
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
    println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
    if let Ok(head) = fs::read_to_string(git_dir.join("HEAD")) {
        if let Some(reference) = head.trim().strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed={}", git_dir.join(reference).display());
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=BULLETOU_GIT_COMMIT");
    println!("cargo:rerun-if-env-changed=BULLETOU_GIT_DIRTY");
    emit_git_rerun_hints();

    let commit = env::var("BULLETOU_GIT_COMMIT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| git_output(&["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());
    let commit_short = if commit == "unknown" { "unknown".to_string() } else { commit.chars().take(12).collect() };
    let dirty = env::var("BULLETOU_GIT_DIRTY")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| git_output(&["status", "--porcelain"]).map(|status| (!status.is_empty()).to_string()))
        .unwrap_or_else(|| "unknown".to_string());
    let built_at_unix = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());

    println!("cargo:rustc-env=BULLETOU_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=BULLETOU_GIT_COMMIT_SHORT={commit_short}");
    println!("cargo:rustc-env=BULLETOU_GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=BULLETOU_BUILT_AT_UNIX={built_at_unix}");
    println!("cargo:rustc-env=BULLETOU_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=BULLETOU_BUILD_TARGET={target}");
}
