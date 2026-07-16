fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(2);
    }
}

#[cfg(feature = "cuda")]
fn run() -> bulletou_cuda_oxide_runtime::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut ptx = None;
    let mut kernel = String::from("noop");
    let mut device = 0usize;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ptx" => ptx = args.next().map(std::path::PathBuf::from),
            "--kernel" => kernel = args.next().unwrap_or_else(|| usage_exit("--kernel requires a value")),
            "--device" => {
                let value = args.next().unwrap_or_else(|| usage_exit("--device requires a value"));
                device = value
                    .parse()
                    .unwrap_or_else(|_| usage_exit("--device must be a non-negative integer"));
            }
            "--help" | "-h" => usage_success(),
            _ => usage_exit(&format!("unknown argument: {arg}")),
        }
    }

    let ptx = ptx.unwrap_or_else(|| usage_exit("--ptx is required"));
    let ctx = bulletou_cuda_oxide_runtime::CudaContext::new(device)?;
    let module = bulletou_cuda_oxide_runtime::load_ptx_module(&ctx, &ptx)?;
    let _kernel = bulletou_cuda_oxide_runtime::resolve_kernel(&module, &kernel)?;
    let roundtrip_ok = bulletou_cuda_oxide_runtime::host_device_roundtrip(&ctx, 1024)?;

    println!("bulletou-cuda-train PTX smoke");
    println!("  ptx       : {}", ptx.display());
    println!("  kernel    : {kernel}");
    println!("  roundtrip : {roundtrip_ok}");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run() -> bulletou_cuda_oxide_runtime::Result<()> {
    let _ = bulletou_cuda_oxide_runtime::backend_status();
    eprintln!(
        "bulletou-cuda-train was built without CUDA support.\n\
         Rebuild with:\n  cargo run -p bulletou-cuda-train --features cuda -- --ptx <PATH> [--kernel <NAME>]"
    );
    Err(bulletou_cuda_oxide_runtime::Error::CudaFeatureDisabled)
}

#[cfg(feature = "cuda")]
fn usage_success() -> ! {
    println!("{}", usage());
    std::process::exit(0);
}

#[cfg(feature = "cuda")]
fn usage_exit(message: &str) -> ! {
    eprintln!("error: {message}\n\n{}", usage());
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn usage() -> &'static str {
    "Usage: bulletou-cuda-train --ptx <PATH> [--kernel <NAME>] [--device <ID>]\n\
     \n\
     CO-004 smoke command: load a generated PTX module, resolve a kernel symbol,\n\
     and verify a host-device-host buffer round trip. Kernel launch is added in\n\
     the next CO-004 step once a repository-local smoke PTX is generated."
}
