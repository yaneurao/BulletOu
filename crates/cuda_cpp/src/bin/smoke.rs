use bulletou_cuda_cpp::{
    Context, F32Buffer, RAdamUpdateParams, RangerDeviceStateMut, RangerStateMut, RangerUpdateParams,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = 0;
    let name = bulletou_cuda_cpp::device_name(device)?;
    println!("bulletou-cuda-cpp smoke");
    println!("  device : {device}");
    println!("  name   : {name}");

    let axpy = bulletou_cuda_cpp::axpy_host(device, 2.0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])?;
    println!("  axpy   : {axpy:?}");
    assert_eq!(axpy, vec![12.0, 24.0, 36.0, 48.0]);

    let ctx = Context::new(device)?;
    let x_dev = F32Buffer::from_host(&ctx, &[1.0, 2.0, 3.0, 4.0])?;
    let y_dev = F32Buffer::from_host(&ctx, &[10.0, 20.0, 30.0, 40.0])?;
    let out_dev = F32Buffer::new(&ctx, 4)?;
    bulletou_cuda_cpp::axpy_device(&ctx, 4, 2.0, &x_dev, &y_dev, &out_dev)?;
    let axpy_device = out_dev.download(&ctx)?;
    println!("  axpy_d : {axpy_device:?}");
    assert_eq!(axpy_device, vec![12.0, 24.0, 36.0, 48.0]);

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

    let gradients_dev = F32Buffer::from_host(&ctx, &[0.25, -0.5, 1.0, -1.5])?;
    let weights_dev = F32Buffer::from_host(&ctx, &[0.1, -0.2, 0.3, -0.4])?;
    let momentum_dev = F32Buffer::from_host(&ctx, &[0.0; 4])?;
    let velocity_dev = F32Buffer::from_host(&ctx, &[0.0; 4])?;
    let slow_dev = F32Buffer::from_host(&ctx, &[0.1, -0.2, 0.3, -0.4])?;
    bulletou_cuda_cpp::ranger_update_device(
        &ctx,
        params,
        RangerDeviceStateMut {
            gradients: &gradients_dev,
            weights: &weights_dev,
            momentum: &momentum_dev,
            velocity: &velocity_dev,
            slow_params: &slow_dev,
        },
    )?;
    ctx.synchronize()?;
    let gradients_device = gradients_dev.download(&ctx)?;
    let weights_device = weights_dev.download(&ctx)?;
    println!("  ranger_d: weights={weights_device:?} gradients={gradients_device:?}");
    assert!(gradients_device.iter().all(|&g| g == 0.0));
    assert_eq!(weights_device, weights);

    println!("  result : ok");
    Ok(())
}
