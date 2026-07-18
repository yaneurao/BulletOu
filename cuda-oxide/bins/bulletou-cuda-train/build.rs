use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rustc-link-lib=dylib=cublas");
    if let Some(lib_dir) = find_cublas_lib_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    } else {
        println!("cargo:warning=CUDA feature enabled but libcublas was not found in known CUDA library directories");
    }
}

fn find_cublas_lib_dir() -> Option<PathBuf> {
    let mut roots = Vec::new();
    for key in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(key).map(PathBuf::from) {
            roots.push(root);
        }
    }
    roots.extend([
        PathBuf::from("/usr/local/cuda"),
        PathBuf::from("/usr/local/cuda-13.2"),
        PathBuf::from("/usr/local/cuda-12.9"),
        PathBuf::from("/opt/cuda"),
    ]);

    for root in roots {
        for rel in ["lib64", "lib/x64", "lib"] {
            let dir = root.join(rel);
            if has_cublas(&dir) {
                return Some(dir);
            }
        }
    }

    for dir in [PathBuf::from("/usr/lib/x86_64-linux-gnu"), PathBuf::from("/usr/lib/wsl/lib")] {
        if has_cublas(&dir) {
            return Some(dir);
        }
    }

    None
}

fn has_cublas(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    ["libcublas.so", "libcublas.so.12", "cublas.lib"].iter().any(|name| dir.join(name).exists())
}
