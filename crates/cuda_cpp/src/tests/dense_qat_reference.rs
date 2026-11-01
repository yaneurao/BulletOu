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

#[test]
#[ignore = "requires CUDA; validates the standalone dense integer QAT kernel"]
fn gpu_integer_dense_matches_scalar_reference() {
    use crate::{check, ffi, Context, F32Buffer, I32Buffer};
    unsafe extern "C" {
        fn bulletou_dense_qat_integer_reference(
            ctx: *mut ffi::BulletOuCudaCppContext, input: *mut ffi::BulletOuCudaCppI32Buffer,
            weights: *mut ffi::BulletOuCudaCppF32Buffer, bias: *mut ffi::BulletOuCudaCppF32Buffer,
            buckets: *mut ffi::BulletOuCudaCppI32Buffer, output: *mut ffi::BulletOuCudaCppI32Buffer,
            batch: usize, inputs: usize, outputs: usize, stacks: usize,
        ) -> i32;
    }
    let (batch, inputs, outputs, stacks) = (4, 5, 3, 2);
    let x = [0,1,63,126,127, 127,0,5,64,1, 1,2,3,4,5, 126,127,0,1,99];
    let buckets = [0,1,1,0];
    let mut w: Vec<f32> = (0..stacks*outputs*inputs)
        .map(|i| (i as f32 - 12.5) / 64.0).collect();
    w[0]=-100.0; w[1]=100.0; w[2]=-0.5/64.0; w[3]=0.5/64.0;
    let b = [-0.5/8128.0,0.5/8128.0,1.0,-1.0,100.0,-100.0];
    let ctx=Context::new(0).unwrap();
    let xi=I32Buffer::from_host(&ctx,&x).unwrap();
    let wi=F32Buffer::from_host(&ctx,&w).unwrap();
    let bi=F32Buffer::from_host(&ctx,&b).unwrap();
    let ki=I32Buffer::from_host(&ctx,&buckets).unwrap();
    let out=I32Buffer::new(&ctx,batch*outputs).unwrap();
    // SAFETY: all buffers have the exact lengths required by the CUDA API.
    check(unsafe { bulletou_dense_qat_integer_reference(ctx.as_ptr(),xi.as_ptr(),wi.as_ptr(),bi.as_ptr(),
        ki.as_ptr(),out.as_ptr(),batch,inputs,outputs,stacks) }).unwrap();
    let got=out.download(&ctx).unwrap();
    for sample in 0..batch {
        for unit in 0..outputs {
            let row=buckets[sample] as usize*outputs+unit;
            assert_eq!(got[sample*outputs+unit],dense(&x[sample*inputs..(sample+1)*inputs],
                &w[row*inputs..(row+1)*inputs],b[row]),"sample={sample} unit={unit}");
        }
    }
}

#[test]
#[ignore = "requires CUDA; validates integer activations and their STE"]
fn gpu_integer_activation_uses_unrounded_ste() {
    use crate::{check, ffi, Context, F32Buffer, I32Buffer};
    unsafe extern "C" {
        fn bulletou_dense_qat_activation_reference(
            ctx: *mut ffi::BulletOuCudaCppContext, raw: *mut ffi::BulletOuCudaCppI32Buffer,
            square_gradient: *mut ffi::BulletOuCudaCppF32Buffer, linear_gradient: *mut ffi::BulletOuCudaCppF32Buffer,
            square_output: *mut ffi::BulletOuCudaCppI32Buffer, linear_output: *mut ffi::BulletOuCudaCppI32Buffer,
            gradient: *mut ffi::BulletOuCudaCppF32Buffer, n: usize,
        ) -> i32;
    }
    let raw = [i32::MIN,-8192,-725,-724,-64,-1,0,1,63,64,724,725,8127,8128,8192,i32::MAX];
    let ctx=Context::new(0).unwrap();
    let r=I32Buffer::from_host(&ctx,&raw).unwrap();
    let gs=F32Buffer::from_host(&ctx,&vec![0.25;raw.len()]).unwrap();
    let gl=F32Buffer::from_host(&ctx,&vec![0.75;raw.len()]).unwrap();
    let so=I32Buffer::new(&ctx,raw.len()).unwrap(); let lo=I32Buffer::new(&ctx,raw.len()).unwrap();
    let g=F32Buffer::new(&ctx,raw.len()).unwrap();
    // SAFETY: each buffer has one element for every entry in `raw`.
    check(unsafe { bulletou_dense_qat_activation_reference(ctx.as_ptr(),r.as_ptr(),gs.as_ptr(),gl.as_ptr(),
        so.as_ptr(),lo.as_ptr(),g.as_ptr(),raw.len()) }).unwrap();
    let sv=so.download(&ctx).unwrap(); let lv=lo.download(&ctx).unwrap(); let gv=g.download(&ctx).unwrap();
    for (i,&v) in raw.iter().enumerate() {
        assert_eq!(sv[i],square(v)); assert_eq!(lv[i],linear(v));
        let x=f64::from(v)/8128.0;
        let expected=(if active_square(v) { 0.25*2.0*x*(127.0/128.0) } else { 0.0 })
            +(if active_linear(v) {0.75} else {0.0});
        assert!((f64::from(gv[i])-expected).abs()<1e-6,"raw={v} actual={} expected={expected}",gv[i]);
    }
}
