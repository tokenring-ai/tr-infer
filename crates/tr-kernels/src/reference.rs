//! Scalar/f64 reference implementations used by tests.
use crate::quant::QAct;

/// y[m][r] = sum_k w[r][k] * x[m][k] with f32 weights and f32 activations (f64 accumulation).
pub fn gemm_f32(w: &[f32], rows: usize, k: usize, x: &[f32], m: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * rows];
    for mi in 0..m {
        for r in 0..rows {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += w[r * k + kk] as f64 * x[mi * k + kk] as f64;
            }
            y[mi * rows + r] = acc as f32;
        }
    }
    y
}

/// Same but with the *quantised* activations dequantised (what the int8 kernel actually computes).
pub fn gemm_qact(w: &[f32], rows: usize, k: usize, xq: &QAct) -> Vec<f32> {
    let nb = xq.nb();
    let mut x = vec![0f32; xq.m * k];
    for mi in 0..xq.m {
        for kk in 0..k {
            x[mi * k + kk] = xq.q[mi * k + kk] as f32 * xq.scale[mi * nb + kk / xq.kb];
        }
    }
    gemm_f32(w, rows, k, &x, xq.m)
}
