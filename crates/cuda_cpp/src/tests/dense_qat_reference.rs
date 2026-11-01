//! Independent scalar oracle for the integer dense-QAT forward path.
//! Kept in tests until the CUDA training path is wired to it.

fn weight(value: f32) -> i32 {
    (f64::from(value) * 64.0).round().clamp(-128.0, 127.0) as i32
}

fn bias(value: f32) -> i32 {
    (f64::from(value) * 8128.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn input(value: f32) -> i32 {
    (f64::from(value) * 127.0).clamp(0.0, 127.0).floor() as i32
}

fn dense(input: &[i32], weights: &[f32], bias_value: f32) -> i32 {
    assert_eq!(input.len(), weights.len());
    let sum = input.iter().zip(weights).fold(i64::from(bias(bias_value)), |sum, (&x, &w)| {
        sum + i64::from(x) * i64::from(weight(w))
    });
    sum as i32
}

fn linear(raw: i32) -> i32 {
    (raw >> 6).clamp(0, 127)
}

fn square(raw: i32) -> i32 {
    ((i64::from(raw) * i64::from(raw)) >> 19).clamp(0, 127) as i32
}

/// The STE uses the pre-rounding activation, not the quantized value.
fn active_linear(raw: i32) -> bool { raw > 0 && raw < 8128 }
fn active_square(raw: i32) -> bool {
    let x = f64::from(raw) / 8128.0;
    x * x * (127.0 / 128.0) < 1.0
}

#[test]
fn tatara_lattice_rounding_and_signed_boundaries() {
    assert_eq!((weight(-0.5 / 64.0), weight(0.5 / 64.0)), (-1, 1));
    assert_eq!((weight(-100.0), weight(100.0)), (-128, 127));
    assert_eq!((input(-0.1), input(0.5), input(2.0)), (0, 63, 127));
    for raw in [i32::MIN, -8192, -725, -724, -64, -1, 0, 1, 63, 64, 724, 725, 8127, 8128, i32::MAX] {
        assert_eq!(linear(raw), (i64::from(raw) >> 6).clamp(0, 127) as i32);
        assert_eq!(square(raw), ((i64::from(raw).pow(2)) >> 19).clamp(0, 127) as i32);
    }
}

#[test]
fn integer_forward_keeps_skip_and_continuous_ste_boundaries() {
    let x = [input(0.5), input(1.0)];
    let hidden = dense(&x, &[0.5, -0.25], 0.0);
    let skip = dense(&x, &[0.25, 0.0], -0.5);
    assert_eq!(hidden, 63 * 32 - 127 * 16);
    assert_eq!(skip, 63 * 16 - 4064);
    let l2 = dense(&[square(hidden), linear(hidden)], &[1.0, -0.5], 0.0);
    let l3 = dense(&[linear(l2)], &[0.5], 0.0);
    let output = l3.wrapping_add(skip);
    assert_eq!(output, -3056);
    // Both values round to zero but must retain their positive-side STE.
    assert_eq!(linear(1), 0);
    assert_eq!(square(1), 0);
    assert!(active_linear(1) && active_square(1));
    assert!(!active_linear(-1));
    assert!(active_square(-1));
    assert!(!active_linear(8128));
}
