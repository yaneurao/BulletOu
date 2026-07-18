use bulletou_cuda_cpp::{RAdamUpdateParams, RangerStateMut, RangerUpdateParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = 0;
    let name = bulletou_cuda_cpp::device_name(device)?;
    println!("bulletou-cuda-cpp smoke");
    println!("  device : {device}");
    println!("  name   : {name}");

    let axpy = bulletou_cuda_cpp::axpy_host(device, 2.0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])?;
    println!("  axpy   : {axpy:?}");
    assert_eq!(axpy, vec![12.0, 24.0, 36.0, 48.0]);

    let mut gradients = vec![0.25, -0.5, 1.0, -1.5];
    let mut weights = vec![0.1, -0.2, 0.3, -0.4];
    let mut momentum = vec![0.0; gradients.len()];
    let mut velocity = vec![0.0; gradients.len()];
    let mut slow_params = weights.clone();
    let params = RangerUpdateParams {
        radam: RAdamUpdateParams {
            step: 1,
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            min_weight: -1.98,
            max_weight: 1.98,
            ..RAdamUpdateParams::default()
        },
        lookahead_alpha: 0.5,
        lookahead_period: 6,
    };
    bulletou_cuda_cpp::ranger_update_host(
        device,
        params,
        RangerStateMut {
            gradients: &mut gradients,
            weights: &mut weights,
            momentum: &mut momentum,
            velocity: &mut velocity,
            slow_params: &mut slow_params,
        },
    )?;
    println!("  ranger : weights={weights:?} gradients={gradients:?}");
    assert!(gradients.iter().all(|&g| g == 0.0));

    println!("  result : ok");
    Ok(())
}
