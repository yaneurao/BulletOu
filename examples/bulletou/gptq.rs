//! Layer-wise GPTQ primitives. Calibration and export integration are separate.

/// Quantize rows of a matrix using a shared input Gram matrix and the fixed
/// NNUE i8 scale. `gram` is row-major X^T X, not its inverse.
pub(crate) fn quantize_rows(weights: &[f32], rows: usize, columns: usize, gram: &[f64], damping: f64) -> Result<Vec<f32>, String> {
    if rows == 0 || columns == 0 || weights.len() != rows.checked_mul(columns).ok_or("GPTQ dimensions overflow")?
        || gram.len() != columns.checked_mul(columns).ok_or("GPTQ dimensions overflow")?
        || !damping.is_finite() || damping <= 0.0 || weights.iter().any(|x| !x.is_finite())
        || gram.iter().any(|x| !x.is_finite()) {
        return Err("invalid GPTQ weights, Gram matrix, or damping".into());
    }
    let mut h = gram.to_vec();
    let mean_diag = (0..columns).map(|i| h[i*columns+i]).sum::<f64>() / columns as f64;
    if !mean_diag.is_finite() || mean_diag <= 0.0 { return Err("GPTQ calibration has no input energy".into()); }
    for i in 0..columns { h[i*columns+i] += damping * mean_diag; }
    let l = cholesky(&h, columns)?;
    let mut inv = vec![0.0; columns*columns];
    for col in 0..columns {
        let mut y = vec![0.0; columns];
        for i in 0..columns {
            let rhs = if i == col {1.0} else {0.0};
            y[i] = (rhs - (0..i).map(|k| l[i*columns+k]*y[k]).sum::<f64>())/l[i*columns+i];
        }
        for i in (0..columns).rev() {
            inv[i*columns+col] = (y[i] - (i+1..columns).map(|k| l[k*columns+i]*inv[k*columns+col]).sum::<f64>())/l[i*columns+i];
        }
    }
    let li = cholesky(&inv, columns)?;
    let mut result = vec![0.0; weights.len()];
    for row in 0..rows {
        let mut w: Vec<f64> = weights[row*columns..(row+1)*columns].iter().map(|&v| f64::from(v)).collect();
        for i in 0..columns {
            let q = (w[i]*64.0).round().clamp(-128.0,127.0)/64.0;
            result[row*columns+i] = q as f32;
            let error = (w[i]-q)/li[i*columns+i];
            for j in i+1..columns { w[j] -= error*li[j*columns+i]; }
        }
    }
    Ok(result)
}

fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut l=vec![0.0; n*n];
    for i in 0..n {
        for j in 0..=i {
            let x=a[i*n+j]-(0..j).map(|k|l[i*n+k]*l[j*n+k]).sum::<f64>();
            l[i*n+j]=if i==j {
                if !x.is_finite() || x<=0.0 {return Err("GPTQ calibration matrix is not positive definite".into());}
                x.sqrt()
            } else {x/l[j*n+j]};
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correlated_inputs_can_compensate_rounding() {
        let w=[0.49/64.0, 0.49/64.0];
        let q=quantize_rows(&w,1,2,&[1.0,1.0,1.0,1.0],0.01).unwrap();
        let err=(f64::from(w[0]+w[1])-f64::from(q[0]+q[1])).abs();
        assert!(err < f64::from(w[0]+w[1]).abs()*0.1, "{q:?}");
    }
    #[test]
    fn invalid_or_empty_calibration_is_rejected() {
        assert!(quantize_rows(&[1.0],1,1,&[0.0],0.01).is_err());
        assert!(quantize_rows(&[f32::NAN],1,1,&[1.0],0.01).is_err());
        assert!(quantize_rows(&[1.0],1,1,&[1.0],0.0).is_err());
    }
    #[test]
    fn independent_inputs_reduce_to_nearest_with_clipping() {
        let w=[2.5,-3.0,0.123,-0.321];
        let q=quantize_rows(&w,2,2,&[2.0,0.0,0.0,1.0],0.01).unwrap();
        for (a,b) in q.iter().zip(w) { assert_eq!(*a, (f64::from(b)*64.0).round().clamp(-128.0,127.0) as f32/64.0); }
    }
}
