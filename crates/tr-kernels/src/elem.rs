//! Elementwise and normalisation kernels (f32, AVX-512).
use std::arch::x86_64::*;

/// exp(x) for 16 lanes, clamp to avoid inf/denormal blowups. Max rel err ~1e-6.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp512(x: __m512) -> __m512 {
    let x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.3)), _mm512_set1_ps(88.3));
    let log2e = _mm512_set1_ps(1.442695041);
    let fx = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), 0x08); // nearest, no exc
    // r = x - fx*ln2 (two-part ln2)
    let r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693145752), x);
    let r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(1.42860677e-6), r);
    // poly (degree 6)
    let mut p = _mm512_set1_ps(1.9875691500e-4);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1));
    let r2 = _mm512_mul_ps(r, r);
    p = _mm512_fmadd_ps(p, r2, r);
    p = _mm512_add_ps(p, _mm512_set1_ps(1.0));
    _mm512_scalef_ps(p, fx)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn sigmoid512(x: __m512) -> __m512 {
    let e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), x));
    _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_add_ps(_mm512_set1_ps(1.0), e))
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn silu512(x: __m512) -> __m512 {
    _mm512_mul_ps(x, sigmoid512(x))
}

/// softplus(x) = ln(1 + e^x), with the usual threshold (x > 20 -> x).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln_1p_fix()
    }
}
trait Ln1pFix {
    fn ln_1p_fix(self) -> f32;
}
impl Ln1pFix for f32 {
    fn ln_1p_fix(self) -> f32 {
        self.ln()
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// y = silu(x) elementwise (n % 16 == 0 fast path, scalar tail).
pub fn silu_vec(x: &[f32], y: &mut [f32]) {
    unsafe { map16(x, y, |v| silu512(v), silu) }
}
pub fn sigmoid_vec(x: &[f32], y: &mut [f32]) {
    unsafe { map16(x, y, |v| sigmoid512(v), sigmoid) }
}
/// y = silu(y) in place.
pub fn silu_inplace(y: &mut [f32]) {
    unsafe {
        let n = y.len();
        let mut i = 0;
        while i + 16 <= n {
            let p = y.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, silu512(_mm512_loadu_ps(p)));
            i += 16;
        }
        while i < n {
            y[i] = silu(y[i]);
            i += 1;
        }
    }
}
/// y = sigmoid(y) in place.
pub fn sigmoid_inplace(y: &mut [f32]) {
    unsafe {
        let n = y.len();
        let mut i = 0;
        while i + 16 <= n {
            let p = y.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, sigmoid512(_mm512_loadu_ps(p)));
            i += 16;
        }
        while i < n {
            y[i] = sigmoid(y[i]);
            i += 1;
        }
    }
}
/// y += a
pub fn add_inplace(y: &mut [f32], a: &[f32]) {
    unsafe {
        let n = y.len().min(a.len());
        let mut i = 0;
        while i + 16 <= n {
            let p = y.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, _mm512_add_ps(_mm512_loadu_ps(p), _mm512_loadu_ps(a.as_ptr().add(i))));
            i += 16;
        }
        while i < n {
            y[i] += a[i];
            i += 1;
        }
    }
}
/// y *= a
pub fn mul_inplace(y: &mut [f32], a: &[f32]) {
    unsafe {
        let n = y.len().min(a.len());
        let mut i = 0;
        while i + 16 <= n {
            let p = y.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, _mm512_mul_ps(_mm512_loadu_ps(p), _mm512_loadu_ps(a.as_ptr().add(i))));
            i += 16;
        }
        while i < n {
            y[i] *= a[i];
            i += 1;
        }
    }
}
/// y += a * b
pub fn fma_inplace(y: &mut [f32], a: &[f32], b: &[f32]) {
    unsafe {
        let n = y.len().min(a.len()).min(b.len());
        let mut i = 0;
        while i + 16 <= n {
            let p = y.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)), _mm512_loadu_ps(p)));
            i += 16;
        }
        while i < n {
            y[i] += a[i] * b[i];
            i += 1;
        }
    }
}
/// y *= s
pub fn scale_inplace(y: &mut [f32], s: f32) {
    unsafe {
        let n = y.len();
        let vs = _mm512_set1_ps(s);
        let mut i = 0;
        while i + 16 <= n {
            let p = y.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, _mm512_mul_ps(_mm512_loadu_ps(p), vs));
            i += 16;
        }
        while i < n {
            y[i] *= s;
            i += 1;
        }
    }
}
/// y = a * b
pub fn mul_vec(a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe { zip16(a, b, y, |p, q| _mm512_mul_ps(p, q), |p, q| p * q) }
}
/// y = silu(g) * u  (SwiGLU)
pub fn swiglu_vec(g: &[f32], u: &[f32], y: &mut [f32]) {
    unsafe { zip16(g, u, y, |p, q| _mm512_mul_ps(silu512(p), q), |p, q| silu(p) * q) }
}
/// y += a * s
pub fn axpy(y: &mut [f32], a: &[f32], s: f32) {
    let n = y.len();
    unsafe {
        let vs = _mm512_set1_ps(s);
        let mut i = 0;
        while i + 16 <= n {
            let p = y.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), vs, _mm512_loadu_ps(p)));
            i += 16;
        }
        while i < n {
            y[i] += a[i] * s;
            i += 1;
        }
    }
}

#[inline]
unsafe fn map16(x: &[f32], y: &mut [f32], f: impl Fn(__m512) -> __m512, g: impl Fn(f32) -> f32) {
    let n = x.len();
    assert_eq!(n, y.len());
    let mut i = 0;
    while i + 16 <= n {
        _mm512_storeu_ps(y.as_mut_ptr().add(i), f(_mm512_loadu_ps(x.as_ptr().add(i))));
        i += 16;
    }
    while i < n {
        y[i] = g(x[i]);
        i += 1;
    }
}
#[inline]
unsafe fn zip16(a: &[f32], b: &[f32], y: &mut [f32], f: impl Fn(__m512, __m512) -> __m512, g: impl Fn(f32, f32) -> f32) {
    let n = a.len();
    assert!(n == b.len() && n == y.len());
    let mut i = 0;
    while i + 16 <= n {
        _mm512_storeu_ps(y.as_mut_ptr().add(i), f(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i))));
        i += 16;
    }
    while i < n {
        y[i] = g(a[i], b[i]);
        i += 1;
    }
}

/// Sum of squares (f32 accumulate in 16 lanes, then reduce).
pub fn sumsq(x: &[f32]) -> f32 {
    unsafe {
        let p = x.as_ptr();
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut i = 0;
        while i + 64 <= x.len() {
            let (v0, v1, v2, v3) = (_mm512_loadu_ps(p.add(i)), _mm512_loadu_ps(p.add(i + 16)), _mm512_loadu_ps(p.add(i + 32)), _mm512_loadu_ps(p.add(i + 48)));
            acc0 = _mm512_fmadd_ps(v0, v0, acc0);
            acc1 = _mm512_fmadd_ps(v1, v1, acc1);
            acc2 = _mm512_fmadd_ps(v2, v2, acc2);
            acc3 = _mm512_fmadd_ps(v3, v3, acc3);
            i += 64;
        }
        while i + 16 <= x.len() {
            let v = _mm512_loadu_ps(p.add(i));
            acc0 = _mm512_fmadd_ps(v, v, acc0);
            i += 16;
        }
        let mut s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
        while i < x.len() {
            s += x[i] * x[i];
            i += 1;
        }
        s
    }
}

/// RMS norm over `x` (one vector of n): y = x * rsqrt(mean(x^2) + eps) * w  (w may be None).
pub fn rmsnorm(x: &[f32], w: Option<&[f32]>, eps: f32, y: &mut [f32]) {
    let n = x.len();
    let r = 1.0 / (sumsq(x) / n as f32 + eps).sqrt();
    match w {
        Some(w) => unsafe { zip16(x, w, y, |p, q| _mm512_mul_ps(_mm512_mul_ps(p, _mm512_set1_ps(r)), q), |p, q| p * r * q) },
        None => unsafe { map16(x, y, |p| _mm512_mul_ps(p, _mm512_set1_ps(r)), |p| p * r) },
    }
}

/// L2 norm as ggml_l2_norm: y = x * rsqrt(sum(x^2) + eps)... ggml uses x / max(sqrt(sum), eps).
/// llama.cpp's ggml_l2_norm computes scale = 1/max(sqrt(sum x^2), eps).
pub fn l2norm(x: &[f32], eps: f32, y: &mut [f32]) {
    let s = sumsq(x).sqrt().max(eps);
    let r = 1.0 / s;
    unsafe { map16(x, y, |p| _mm512_mul_ps(p, _mm512_set1_ps(r)), |p| p * r) }
}

/// In-place softmax over x; returns nothing. Uses f32 with max subtraction.
pub fn softmax_inplace(x: &mut [f32]) {
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - mx).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_sigmoid_silu_accuracy() {
        let xs: Vec<f32> = (-2000..2000).map(|i| i as f32 * 0.04).collect();
        let mut e = vec![0f32; xs.len()];
        unsafe { map16(&xs, &mut e, |v| exp512(v), f32::exp) };
        for (x, y) in xs.iter().zip(&e) {
            let want = x.exp();
            assert!((y - want).abs() <= 2e-6 * want + 1e-30, "exp({x}) = {y} want {want}");
        }
        let mut s = vec![0f32; xs.len()];
        silu_vec(&xs, &mut s);
        for (x, y) in xs.iter().zip(&s) {
            assert!((y - silu(*x)).abs() <= 1e-5 * (1.0 + x.abs()));
        }
    }
    #[test]
    fn norms() {
        let x: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..100).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut y = vec![0f32; 100];
        rmsnorm(&x, Some(&w), 1e-6, &mut y);
        let ms = x.iter().map(|v| v * v).sum::<f32>() / 100.0;
        for i in 0..100 {
            let want = x[i] / (ms + 1e-6).sqrt() * w[i];
            assert!((y[i] - want).abs() < 1e-5);
        }
        l2norm(&x, 1e-6, &mut y);
        let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        for i in 0..100 {
            assert!((y[i] - x[i] / n).abs() < 1e-6);
        }
    }
}

/// x[i] = exp(x[i] - m) in place; entries equal to -inf become exactly 0. Returns the sum.
pub fn exp_sub_inplace(x: &mut [f32], m: f32) -> f32 {
    unsafe {
        let mv = _mm512_set1_ps(m);
        let ninf = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc = _mm512_setzero_ps();
        let n = x.len();
        let mut i = 0;
        while i + 16 <= n {
            let p = x.as_mut_ptr().add(i);
            let v = _mm512_loadu_ps(p);
            let keep = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(v, ninf);
            let e = _mm512_maskz_mov_ps(keep, exp512(_mm512_sub_ps(v, mv)));
            _mm512_storeu_ps(p, e);
            acc = _mm512_add_ps(acc, e);
            i += 16;
        }
        let mut s = _mm512_reduce_add_ps(acc);
        while i < n {
            let e = if x[i] == f32::NEG_INFINITY { 0.0 } else { (x[i] - m).exp() };
            x[i] = e;
            s += e;
            i += 1;
        }
        s
    }
}

/// Maximum of `x` (NaN-free input); `-inf` for an empty slice.
pub fn vmax(x: &[f32]) -> f32 {
    unsafe {
        let mut acc = _mm512_set1_ps(f32::NEG_INFINITY);
        let n = x.len();
        let mut i = 0;
        while i + 16 <= n {
            acc = _mm512_max_ps(acc, _mm512_loadu_ps(x.as_ptr().add(i)));
            i += 16;
        }
        let mut m = _mm512_reduce_max_ps(acc);
        while i < n {
            m = m.max(x[i]);
            i += 1;
        }
        m
    }
}

/// exp(x) for x <= 0 with ~3e-6 relative error (2^f by a degree-4 polynomial, 9 instructions vs
/// exp512's 15): for outputs that are rounded to bf16 anyway (attention probabilities).
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp512_fast(x: __m512) -> __m512 {
    let t = _mm512_mul_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.3)), _mm512_set1_ps(1.442695041));
    let fx = _mm512_roundscale_ps(t, 0x09); // floor, no exc
    let f = _mm512_sub_ps(t, fx);
    let mut p = _mm512_set1_ps(0.01342552);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.05224435));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.24127934));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.69304495));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0));
    _mm512_scalef_ps(p, fx)
}

/// dst[i] = bf16(exp(x[i] - m)) for i < x.len() (x[i] <= m), dst[x.len()..] = 0; returns the f32
/// sum of the exponentials (before rounding). The attention probabilities as the AMX A operand;
/// uses `exp512_fast` (3e-6 relative, below the bf16 rounding of the outputs).
pub fn exp_sub_bf16(x: &[f32], m: f32, dst: &mut [u16]) -> f32 {
    debug_assert!(dst.len() >= x.len());
    unsafe { exp_sub_bf16_impl(x, m, dst) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
unsafe fn exp_sub_bf16_impl(x: &[f32], m: f32, dst: &mut [u16]) -> f32 {
    let mv = _mm512_set1_ps(m);
    let mut acc = _mm512_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 32 <= n {
        let lo = exp512_fast(_mm512_sub_ps(_mm512_loadu_ps(x.as_ptr().add(i)), mv));
        let hi = exp512_fast(_mm512_sub_ps(_mm512_loadu_ps(x.as_ptr().add(i + 16)), mv));
        acc = _mm512_add_ps(acc, _mm512_add_ps(lo, hi));
        let v: __m512i = std::mem::transmute(_mm512_cvtne2ps_pbh(hi, lo));
        _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, v);
        i += 32;
    }
    let mut s = _mm512_reduce_add_ps(acc);
    while i < n {
        let e = (x[i] - m).exp();
        s += e;
        let b = e.to_bits();
        dst[i] = ((b + (((b >> 16) & 1) + 0x7FFF)) >> 16) as u16;
        i += 1;
    }
    dst[n..].fill(0);
    s
}

#[cfg(test)]
mod exp_tests {
    use super::*;
    #[test]
    fn exp512_accuracy() {
        let xs: Vec<f32> = (0..4096).map(|i| -90.0 + i as f32 * (90.0 / 4096.0)).collect();
        for c in xs.chunks_exact(16) {
            let mut out = [0f32; 16];
            unsafe { _mm512_storeu_ps(out.as_mut_ptr(), exp512_fast(_mm512_loadu_ps(c.as_ptr()))) };
            for (x, y) in c.iter().zip(out) {
                let want = x.max(-87.0).exp();
                assert!((y - want).abs() <= want * 1e-5 + 1e-38, "exp_fast({x}) = {y} want {want}"); // ~5e-6 from the single-multiply reduction near -87
            }
        }
        let xs: Vec<f32> = (0..4096).map(|i| -90.0 + i as f32 * (170.0 / 4096.0)).collect();
        for c in xs.chunks_exact(16) {
            let mut out = [0f32; 16];
            unsafe { _mm512_storeu_ps(out.as_mut_ptr(), exp512(_mm512_loadu_ps(c.as_ptr()))) };
            for (x, y) in c.iter().zip(out) {
                let want = x.max(-87.0).min(88.0).exp();
                assert!((y - want).abs() <= want * 4e-7 + 1e-38, "exp({x}) = {y} want {want}");
            }
        }
        let mut v = vec![f32::NEG_INFINITY, 0.0, 1.0, -1.0, 2.5, f32::NEG_INFINITY, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 3.0];
        let mut w = vec![0u16; 40];
        let sb = exp_sub_bf16(&v[1..], 2.5, &mut w);
        for (i, &x) in v[1..].iter().enumerate() {
            let want = if x == f32::NEG_INFINITY { 0.0 } else { (x - 2.5).exp() };
            let got = f32::from_bits((w[i] as u32) << 16);
            assert!((got - want).abs() <= want * 4e-3 + 1e-30, "exp_sub_bf16 {i}: {got} vs {want}");
        }
        assert!(w[16..].iter().all(|&b| b == 0) && (sb - v[1..].iter().map(|x| (x - 2.5).exp()).sum::<f32>()).abs() < 1e-3);
        assert_eq!(vmax(&v), 3.0);
        assert_eq!(vmax(&v[..3]), 1.0);
        let s = exp_sub_inplace(&mut v, 2.5);
        assert_eq!(v[0], 0.0);
        assert!((v[4] - 1.0).abs() < 1e-6 && (v[16] - 0.5f32.exp()).abs() < 1e-6);
        assert!((s - v.iter().sum::<f32>()).abs() < 1e-4);
    }
}

/// Copy `src` into `dst` with non-temporal (streaming) stores: for buffers that other tiles will
/// read next and whose lines currently sit in their caches (avoids the read-for-ownership).
/// `dst` must be 64-byte aligned and the length a multiple of 16. Call `store_fence` once after
/// the last stream_copy and before the barrier that publishes the data.
pub fn stream_copy(dst: &mut [f32], src: &[f32]) {
    debug_assert!(dst.len() == src.len() && dst.len() % 16 == 0 && (dst.as_ptr() as usize) % 64 == 0);
    unsafe {
        let mut i = 0;
        while i < dst.len() {
            _mm512_stream_ps(dst.as_mut_ptr().add(i), _mm512_loadu_ps(src.as_ptr().add(i)));
            i += 16;
        }
    }
}
/// Make all earlier streaming stores globally visible.
#[inline]
pub fn store_fence() {
    unsafe { _mm_sfence() }
}
