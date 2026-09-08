//! Activation quantisation: f32 rows -> int8 per (row, k-block) with f32 scale and int32 block sum.
use std::arch::x86_64::*;

/// Quantised activation rows, laid out row-major: `q[m*k + i]`, `scale[m*nb + b]`, `sum[m*nb + b]`.
#[derive(Clone, Debug)]
pub struct QAct {
    pub m: usize,
    pub k: usize,
    pub kb: usize,
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
    pub sum: Vec<i32>,
    pub pair: bool,
    /// Rows currently holding data (<= m capacity); as_ref() reports this.
    pub m_used: usize,
}

/// Borrowed view of quantised activations (same layout as `QAct`).
#[derive(Clone, Copy)]
pub struct QActRef<'a> {
    pub m: usize,
    pub k: usize,
    pub kb: usize,
    pub q: &'a [i8],
    pub scale: &'a [f32],
    pub sum: &'a [i32],
    /// Two-level quantisation: rows come in (coarse, residual) pairs that the kernel sums, so the
    /// logical row count is m/2. Doubles dot-product work, leaves weight traffic unchanged.
    pub pair: bool,
}
impl<'a> QActRef<'a> {
    pub fn nb(&self) -> usize {
        self.k / self.kb
    }
    /// Quantise `x` (m rows of k f32) into the borrowed buffers.
    pub fn quantize_into(m: usize, k: usize, kb: usize, x: &[f32], q: &'a mut [i8], scale: &'a mut [f32], sum: &'a mut [i32]) -> QActRef<'a> {
        let nb = k / kb;
        for r in 0..m {
            for b in 0..nb {
                let src = &x[r * k + b * kb..r * k + (b + 1) * kb];
                let dst = &mut q[r * k + b * kb..r * k + (b + 1) * kb];
                let (s, su) = unsafe { quant_block(src, dst) };
                scale[r * nb + b] = s;
                sum[r * nb + b] = su;
            }
        }
        QActRef { m, k, kb, q: &*q, scale: &*scale, sum: &*sum, pair: false }
    }
    /// Like `quantize_into` but two-level (coarse + residual) for one logical row.
    pub fn quantize_pair_into(k: usize, kb: usize, x: &[f32], q: &'a mut [i8], scale: &'a mut [f32], sum: &'a mut [i32]) -> QActRef<'a> {
        quantize_pair_blocks(x, k, kb, 0, k / kb, q, scale, sum);
        QActRef { m: 2, k, kb, q: &*q, scale: &*scale, sum: &*sum, pair: true }
    }
}

impl QAct {
    pub fn as_ref(&self) -> QActRef<'_> {
        QActRef { m: self.m_used, k: self.k, kb: self.kb, q: &self.q, scale: &self.scale, sum: &self.sum, pair: self.pair }
    }
    /// Two-level quantisation of one logical row (see `QActRef::pair`). Requires m == 2 capacity.
    pub fn quantize_pair(&mut self, x: &[f32]) {
        assert!(self.m == 2 && x.len() == self.k);
        let nb = self.nb();
        quantize_pair_blocks(x, self.k, self.kb, 0, nb, &mut self.q, &mut self.scale, &mut self.sum);
        self.pair = true;
        self.m_used = 2;
    }
    pub fn zeros(m: usize, k: usize, kb: usize) -> QAct {
        assert!(k % kb == 0 && (kb == 16 || kb == 32));
        QAct { m, k, kb, q: vec![0; m * k], scale: vec![0.0; m * (k / kb)], sum: vec![0; m * (k / kb)], pair: false, m_used: m }
    }
    pub fn nb(&self) -> usize {
        self.k / self.kb
    }
    /// Quantise `x` (m rows of k f32) into self (single level; x may have fewer rows than m).
    pub fn quantize(&mut self, x: &[f32]) {
        assert!(x.len() <= self.m * self.k && x.len() % self.k == 0);
        self.pair = false;
        let m = x.len() / self.k;
        self.m_used = m;
        let nb = self.nb();
        for r in 0..m {
            for b in 0..nb {
                let src = &x[r * self.k + b * self.kb..r * self.k + (b + 1) * self.kb];
                let dst = &mut self.q[r * self.k + b * self.kb..r * self.k + (b + 1) * self.kb];
                let (s, sum) = unsafe { quant_block(src, dst) };
                self.scale[r * nb + b] = s;
                self.sum[r * nb + b] = sum;
            }
        }
    }
}

/// Two-level quantisation of blocks [b0, b1) of one logical row into (coarse row 0 | residual row 1)
/// laid out as `q[0..k]`, `q[k..2k]`, `scale[0..nb]`, `scale[nb..2nb]` (same for `sum`). No allocation.
pub fn quantize_pair_blocks(x: &[f32], k: usize, kb: usize, b0: usize, b1: usize, q: &mut [i8], scale: &mut [f32], sum: &mut [i32]) {
    let nb = k / kb;
    let mut resid = [0f32; 32];
    for b in b0..b1 {
        let src = &x[b * kb..(b + 1) * kb];
        let (s, su) = unsafe { quant_block_resid(src, &mut q[b * kb..(b + 1) * kb], &mut resid[..kb]) };
        scale[b] = s;
        sum[b] = su;
        let (s2, su2) = unsafe { quant_block(&resid[..kb], &mut q[k + b * kb..k + (b + 1) * kb]) };
        scale[nb + b] = s2;
        sum[nb + b] = su2;
    }
}

/// quant_block that also writes the residual `src - q*scale`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn quant_block_resid(src: &[f32], dst: &mut [i8], resid: &mut [f32]) -> (f32, i32) {
    let n = src.len();
    let mut amax = _mm512_setzero_ps();
    let mut i = 0;
    while i < n {
        amax = _mm512_max_ps(amax, _mm512_abs_ps(_mm512_loadu_ps(src.as_ptr().add(i))));
        i += 16;
    }
    let amax = _mm512_reduce_max_ps(amax);
    let scale = amax / 127.0;
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    let vinv = _mm512_set1_ps(inv);
    let vs = _mm512_set1_ps(scale);
    let mut sum = 0i32;
    let mut i = 0;
    while i < n {
        let v = _mm512_loadu_ps(src.as_ptr().add(i));
        let qi = _mm512_cvtps_epi32(_mm512_mul_ps(v, vinv));
        sum += _mm512_reduce_add_epi32(qi);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, _mm512_cvtsepi32_epi8(qi));
        let r = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(qi), vs, v);
        _mm512_storeu_ps(resid.as_mut_ptr().add(i), r);
        i += 16;
    }
    (scale, sum)
}

/// Quantise one block (16 or 32 f32) to int8 with absmax/127 scaling. Returns (scale, sum of q).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn quant_block(src: &[f32], dst: &mut [i8]) -> (f32, i32) {
    let n = src.len();
    debug_assert!(n == 16 || n == 32);
    let mut amax = _mm512_setzero_ps();
    let mut i = 0;
    while i < n {
        let v = _mm512_loadu_ps(src.as_ptr().add(i));
        amax = _mm512_max_ps(amax, _mm512_abs_ps(v));
        i += 16;
    }
    let amax = _mm512_reduce_max_ps(amax);
    let scale = amax / 127.0;
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    let vinv = _mm512_set1_ps(inv);
    let mut sum = 0i32;
    let mut i = 0;
    while i < n {
        let v = _mm512_loadu_ps(src.as_ptr().add(i));
        let q = _mm512_cvtps_epi32(_mm512_mul_ps(v, vinv)); // round-to-nearest-even
        sum += _mm512_reduce_add_epi32(q);
        let b = _mm512_cvtsepi32_epi8(q);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, b);
        i += 16;
    }
    (scale, sum)
}
