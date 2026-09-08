//! TQ (per-32/16 block, 4/5/8-bit unsigned, f16 scale+min) x int8 activations on AVX-512 VNNI.
//!
//! One call computes `y[m][r] += w[r][k] . x[m][k]` for a contiguous range of 16-row strips of a
//! packed matrix, for M <= MAX_M activation rows. Weight bytes are streamed exactly once per call.
use crate::quant::QActRef;
use std::arch::x86_64::*;
use tr_format::codec::{Codec, HDR_BYTES, STRIP};

pub const MAX_M: usize = 8;

/// Compute strips [s0, s1) of a packed matrix (`w` points at strip s0) against `xq` (m rows),
/// writing f32 results into `y[m * ldy + r]` for rows r in [16*s0, 16*s1). `accumulate` adds to y.
///
/// # Safety
/// `w` must hold (s1 - s0) strips of `k / kb` blocks each; `y` must be large enough.
pub unsafe fn gemv_tq(w: *const u8, k: usize, c: Codec, s0: usize, s1: usize, xq: QActRef<'_>, y: *mut f32, ldy: usize, accumulate: bool) {
    gemv_tq_rows(w, k, c, s0, s1, xq, y, ldy, std::ptr::null(), accumulate)
}

/// As `gemv_tq`, with logical row `i` of `xq` written to y row `rows[i]` (`rows` null: identity).
///
/// # Safety
/// As `gemv_tq`; `rows` (if not null) must hold one index per logical row, each a valid y row.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_tq_rows(w: *const u8, k: usize, c: Codec, s0: usize, s1: usize, xq: QActRef<'_>, y: *mut f32, ldy: usize, rows: *const u32, accumulate: bool) {
    debug_assert_eq!(xq.k, k);
    debug_assert_eq!(xq.kb, c.kb);
    match (c.bits, c.kb) {
        (4, 32) => gemv_impl::<4, 32>(w, k, s0, s1, xq, y, ldy, rows, accumulate),
        (5, 32) => gemv_impl::<5, 32>(w, k, s0, s1, xq, y, ldy, rows, accumulate),
        (8, 32) => gemv_impl::<8, 32>(w, k, s0, s1, xq, y, ldy, rows, accumulate),
        (4, 16) => gemv_impl::<4, 16>(w, k, s0, s1, xq, y, ldy, rows, accumulate),
        (5, 16) => gemv_impl::<5, 16>(w, k, s0, s1, xq, y, ldy, rows, accumulate),
        (8, 16) => gemv_impl::<8, 16>(w, k, s0, s1, xq, y, ldy, rows, accumulate),
        _ => panic!("unsupported codec {c:?}"),
    }
}

/// GEMM over any number of activation rows: chunks the rows of `xq` into groups the kernel
/// accepts (MAX_M physical rows) and streams the weight strips once per group.
///
/// # Safety
/// Same as `gemv_tq`; `y` must hold `xq.logical_rows()` rows of stride `ldy`.
pub unsafe fn gemm_tq(w: *const u8, k: usize, c: Codec, s0: usize, s1: usize, xq: QActRef<'_>, y: *mut f32, ldy: usize, accumulate: bool) {
    gemm_tq_rows(w, k, c, s0, s1, xq, y, ldy, None, accumulate)
}

/// As `gemm_tq`, with logical row `i` written to (or accumulated into) y row `rows[i]` when a
/// row map is given — the MoE down projection accumulates each expert's rows straight into the
/// per-token sums.
///
/// # Safety
/// As `gemm_tq`; `rows` must hold one valid y row index per logical row of `xq`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_tq_rows(w: *const u8, k: usize, c: Codec, s0: usize, s1: usize, xq: QActRef<'_>, y: *mut f32, ldy: usize, rows: Option<&[u32]>, accumulate: bool) {
    let per = if xq.pair { 2 } else { 1 };
    let nb = k / xq.kb;
    let logical = xq.m / per;
    let group = MAX_M / per;
    let mut r0 = 0;
    while r0 < logical {
        let r1 = (r0 + group).min(logical);
        let sub = QActRef {
            m: (r1 - r0) * per,
            k,
            kb: xq.kb,
            q: &xq.q[r0 * per * k..r1 * per * k],
            scale: &xq.scale[r0 * per * nb..r1 * per * nb],
            sum: &xq.sum[r0 * per * nb..r1 * per * nb],
            pair: xq.pair,
        };
        match rows {
            Some(rows) => gemv_tq_rows(w, k, c, s0, s1, sub, y, ldy, rows[r0..r1].as_ptr(), accumulate),
            None => gemv_tq(w, k, c, s0, s1, sub, y.add(r0 * ldy), ldy, accumulate),
        }
        r0 = r1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,f16c")]
unsafe fn gemv_impl<const BITS: u8, const KB: usize>(w: *const u8, k: usize, s0: usize, s1: usize, xq: QActRef<'_>, y: *mut f32, ldy: usize, rows: *const u32, accumulate: bool) {
    let m = xq.m;
    let yrow = |i: usize| if rows.is_null() { i } else { *rows.add(i) as usize };
    debug_assert!(m >= 1 && m <= MAX_M);
    let c = Codec::new(BITS, KB);
    let bb = c.block_bytes();
    let nb = k / KB;
    let lo_mask = _mm512_set1_epi8(0x0F);
    let hi_bit = 0x10i8;
    let mut wp = w;
    for s in s0..s1 {
        let mut accf = [_mm512_setzero_ps(); MAX_M];
        for b in 0..nb {
            let blk = wp;
            wp = wp.add(bb);
            // stream ahead: keep ~1.5 KB of upcoming weight bytes in flight
            _mm_prefetch(wp.add(3 * bb) as *const i8, _MM_HINT_T0);
            _mm_prefetch(wp.add(3 * bb + 64) as *const i8, _MM_HINT_T0);
            if bb > 128 {
                _mm_prefetch(wp.add(3 * bb + 128) as *const i8, _MM_HINT_T0);
                _mm_prefetch(wp.add(3 * bb + 192) as *const i8, _MM_HINT_T0);
            }
            // header
            let d = _mm512_cvtph_ps(_mm256_loadu_si256(blk as *const __m256i));
            let mn = _mm512_cvtph_ps(_mm256_loadu_si256(blk.add(32) as *const __m256i));
            let qb = blk.add(HDR_BYTES);
            let mut acc32 = [_mm512_setzero_si512(); MAX_M];
            if BITS == 8 {
                let mut j = 0;
                while j < KB / 4 {
                    let wq = _mm512_loadu_si512(qb.add(j * 64) as *const __m512i);
                    for mi in 0..m {
                        let xa = *(xq.q.as_ptr().add(mi * k + b * KB + 4 * j) as *const i32);
                        acc32[mi] = _mm512_dpbusd_epi32(acc32[mi], wq, _mm512_set1_epi32(xa));
                    }
                    j += 1;
                }
            } else {
                let plane = qb.add(KB / 8 * 64);
                let mut j = 0;
                while j < KB / 8 {
                    let raw = _mm512_loadu_si512(qb.add(j * 64) as *const __m512i);
                    let mut lo = _mm512_and_si512(raw, lo_mask);
                    let mut hi = _mm512_and_si512(_mm512_srli_epi16(raw, 4), lo_mask);
                    if BITS == 5 {
                        let mlo = *(plane.add(j * 16) as *const u64);
                        let mhi = *(plane.add(j * 16 + 8) as *const u64);
                        lo = _mm512_or_si512(lo, _mm512_maskz_set1_epi8(mlo, hi_bit));
                        hi = _mm512_or_si512(hi, _mm512_maskz_set1_epi8(mhi, hi_bit));
                    }
                    for mi in 0..m {
                        let xp = xq.q.as_ptr().add(mi * k + b * KB);
                        let xa = *(xp.add(4 * j) as *const i32);
                        let xb = *(xp.add(KB / 2 + 4 * j) as *const i32);
                        acc32[mi] = _mm512_dpbusd_epi32(acc32[mi], lo, _mm512_set1_epi32(xa));
                        acc32[mi] = _mm512_dpbusd_epi32(acc32[mi], hi, _mm512_set1_epi32(xb));
                    }
                    j += 1;
                }
            }
            // scale: acc += (d * sx) * acc32 + m * (sx * xsum)
            for mi in 0..m {
                let sx = *xq.scale.as_ptr().add(mi * nb + b);
                let u = sx * (*xq.sum.as_ptr().add(mi * nb + b)) as f32;
                let t = _mm512_mul_ps(d, _mm512_set1_ps(sx));
                accf[mi] = _mm512_fmadd_ps(mn, _mm512_set1_ps(u), accf[mi]);
                accf[mi] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc32[mi]), t, accf[mi]);
            }
        }
        if xq.pair {
            // two-level activations: rows (2i, 2i+1) are (coarse, residual) of logical row i
            for mi in 0..m / 2 {
                let yp = y.add(yrow(mi) * ldy + s * STRIP);
                let v = _mm512_add_ps(accf[2 * mi], accf[2 * mi + 1]);
                if accumulate {
                    _mm512_storeu_ps(yp, _mm512_add_ps(_mm512_loadu_ps(yp), v));
                } else {
                    _mm512_storeu_ps(yp, v);
                }
            }
        } else {
            for mi in 0..m {
                let yp = y.add(yrow(mi) * ldy + s * STRIP);
                if accumulate {
                    _mm512_storeu_ps(yp, _mm512_add_ps(_mm512_loadu_ps(yp), accf[mi]));
                } else {
                    _mm512_storeu_ps(yp, accf[mi]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QAct;
    use crate::reference::{gemm_f32, gemm_qact};
    use tr_format::codec::{dequant_ref, pack_ref};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unif(&mut self) -> f32 {
            (self.next() % 100000) as f32 / 100000.0
        }
    }

    fn synth(rows: usize, k: usize, c: Codec, rng: &mut Lcg) -> Vec<u8> {
        let nb = k / c.kb;
        let q: Vec<u32> = (0..rows * k).map(|_| rng.next() % (1 << c.bits)).collect();
        let d: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(rng.unif() * 0.02 + 0.001).to_bits()).collect();
        let m: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(-rng.unif() * 0.2).to_bits()).collect();
        pack_ref(rows, k, c, &q, &d, &m)
    }

    #[test]
    fn gemv_matches_reference_all_codecs() {
        let mut rng = Lcg(7);
        for &(bits, kb, rows, k, m) in &[(4u8, 32usize, 80usize, 2560usize, 1usize), (5, 32, 48, 256, 3), (8, 32, 32, 320, 8), (5, 16, 160, 80, 1), (8, 16, 32, 80, 4), (4, 16, 16, 32, 2)] {
            let c = Codec::new(bits, kb);
            let w = synth(rows, k, c, &mut rng);
            let wf = dequant_ref(&w, rows, k, c);
            let x: Vec<f32> = (0..m * k).map(|_| rng.unif() * 2.0 - 1.0).collect();
            let mut xq = QAct::zeros(m, k, kb);
            xq.quantize(&x);
            let mut y = vec![0f32; m * rows];
            unsafe { gemv_tq(w.as_ptr(), k, c, 0, rows / 16, xq.as_ref(), y.as_mut_ptr(), rows, false) };
            let want_q = gemm_qact(&wf, rows, k, &xq);
            let want_f = gemm_f32(&wf, rows, k, &x, m);
            let scale = want_f.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-6);
            for i in 0..m * rows {
                let e1 = (y[i] - want_q[i]).abs();
                assert!(e1 <= 1e-4 * scale + 1e-5, "bits {bits} kb {kb} i {i}: {} vs exact-int {} (scale {scale})", y[i], want_q[i]);
                let e2 = (y[i] - want_f[i]).abs();
                assert!(e2 <= 2e-2 * scale, "bits {bits} kb {kb} i {i}: {} vs f32 {} (quant err too large)", y[i], want_f[i]);
            }
            // accumulate path
            unsafe { gemv_tq(w.as_ptr(), k, c, 0, rows / 16, xq.as_ref(), y.as_mut_ptr(), rows, true) };
            for i in 0..m * rows {
                assert!((y[i] - 2.0 * want_q[i]).abs() <= 2e-4 * scale + 1e-5);
            }
        }
    }

    #[test]
    fn pair_mode_is_much_more_accurate() {
        let mut rng = Lcg(11);
        let c = Codec::new(8, 32);
        let (rows, k) = (256, 768);
        let w = synth(rows, k, c, &mut rng);
        let wf = dequant_ref(&w, rows, k, c);
        // spiky activations: a few large entries per block, many tiny ones
        let x: Vec<f32> = (0..k).map(|i| if i % 32 == 5 { 0.5 } else { (rng.unif() - 0.5) * 0.01 }).collect();
        let want = gemm_f32(&wf, rows, k, &x, 1);
        let mut xq1 = QAct::zeros(1, k, 32);
        xq1.quantize(&x);
        let mut xq2 = QAct::zeros(2, k, 32);
        xq2.quantize_pair(&x);
        let mut y1 = vec![0f32; rows];
        let mut y2 = vec![0f32; rows];
        unsafe {
            gemv_tq(w.as_ptr(), k, c, 0, rows / 16, xq1.as_ref(), y1.as_mut_ptr(), rows, false);
            gemv_tq(w.as_ptr(), k, c, 0, rows / 16, xq2.as_ref(), y2.as_mut_ptr(), rows, false);
        }
        let e1: f64 = (0..rows).map(|i| ((y1[i] - want[i]) as f64).abs()).sum::<f64>();
        let e2: f64 = (0..rows).map(|i| ((y2[i] - want[i]) as f64).abs()).sum::<f64>();
        let norm: f64 = want.iter().map(|v| (*v as f64).abs()).sum();
        eprintln!("single-level rel err {:.2e}, pair rel err {:.2e}", e1 / norm, e2 / norm);
        assert!(e2 < e1 / 20.0, "pair mode should be >20x more accurate: {e1} vs {e2}");
    }

    #[test]
    fn gemm_many_rows_matches_reference() {
        let mut rng = Lcg(21);
        let c = Codec::new(4, 32);
        let (rows, k, m) = (48, 256, 21);
        let w = synth(rows, k, c, &mut rng);
        let wf = dequant_ref(&w, rows, k, c);
        let x: Vec<f32> = (0..m * k).map(|_| rng.unif() - 0.5).collect();
        let mut xq = QAct::zeros(m, k, 32);
        xq.quantize(&x);
        let want = gemm_qact(&wf, rows, k, &xq);
        let mut y = vec![0f32; m * rows];
        unsafe { gemm_tq(w.as_ptr(), k, c, 0, rows / 16, xq.as_ref(), y.as_mut_ptr(), rows, false) };
        let scale = want.iter().fold(0f32, |a, &b| a.max(b.abs()));
        for i in 0..m * rows {
            assert!((y[i] - want[i]).abs() <= 1e-4 * scale + 1e-5, "i {i}: {} vs {}", y[i], want[i]);
        }
    }

    #[test]
    fn strip_subrange() {
        let mut rng = Lcg(9);
        let c = Codec::new(4, 32);
        let (rows, k) = (64, 128);
        let w = synth(rows, k, c, &mut rng);
        let wf = dequant_ref(&w, rows, k, c);
        let x: Vec<f32> = (0..k).map(|_| rng.unif()).collect();
        let mut xq = QAct::zeros(1, k, 32);
        xq.quantize(&x);
        let want = gemm_qact(&wf, rows, k, &xq);
        let mut y = vec![0f32; rows];
        let strip_bytes = c.strip_bytes(k);
        unsafe { gemv_tq(w.as_ptr().add(2 * strip_bytes), k, c, 2, 4, xq.as_ref(), y.as_mut_ptr(), rows, false) };
        for r in 0..rows {
            let expect = if (32..64).contains(&r) { want[r] } else { 0.0 };
            assert!((y[r] - expect).abs() <= 1e-4 * want.iter().fold(0f32, |a, &b| a.max(b.abs())) + 1e-6, "r {r}: {} vs {expect}", y[r]);
        }
    }
}
