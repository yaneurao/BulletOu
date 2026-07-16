fn main() {
    let status = bulletou_cuda_oxide_runtime::backend_status();
    eprintln!(
        "bulletou-cuda-train is not implemented yet ({status:?}). \
         Next step is CO-004: generated PTX smoke loading."
    );
    std::process::exit(2);
}
