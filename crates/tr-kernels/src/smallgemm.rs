//! Small f32 GEMM with register blocking: y[i][j] = sum_k x[i][k] * w[j][k]  (both operands
//! row-major over k, "NT"). Used where the weights are f32 and small (the MoE router:
//! m x 512 x 320 per tile), where a per-output `dot` is FMA-latency bound.
use std::arch::x86_64::*;

/// x: `mr` rows of `k` (stride ldx), w: `n` rows of `k` (stride ldw), y: `mr` rows of `n` (stride ldy).
/// Requires k % 16 == 0 and n % 4 == 0.
pub fn gemm_f32_nt(x: &[f32], ldx: usize, mr: usize, w: &[f32], ldw: usize, n: usize, k: usize, y: &mut [f32], ldy: usize) {
    assert!(k % 16 == 0 && n % 4 == 0);
    assert!(x.len() >= (mr.max(1) - 1) * ldx + k || mr == 0);
    assert!(w.len() >= (n - 1) * ldw + k);
    assert!(y.len() >= (mr.max(1) - 1) * ldy + n || mr == 0);
    let mut i = 0;
    unsafe {
        while i + 4 <= mr {
            block::<4>(x.as_ptr().add(i * ldx), ldx, w, ldw, n, k, y.as_mut_ptr().add(i * ldy), ldy);
            i += 4;
        }
        match mr - i {
            3 => block::<3>(x.as_ptr().add(i * ldx), ldx, w, ldw, n, k, y.as_mut_ptr().add(i * ldy), ldy),
            2 => block::<2>(x.as_ptr().add(i * ldx), ldx, w, ldw, n, k, y.as_mut_ptr().add(i * ldy), ldy),
            1 => block::<1>(x.as_ptr().add(i * ldx), ldx, w, ldw, n, k, y.as_mut_ptr().add(i * ldy), ldy),
            _ => {}
        }
    }
}

/// MR rows of x against 4 rows of w at a time: 4*MR accumulators, 4 + MR loads per 4*MR FMAs.
#[target_feature(enable = "avx512f")]
unsafe fn block<const MR: usize>(x: *const f32, ldx: usize, w: &[f32], ldw: usize, n: usize, k: usize, y: *mut f32, ldy: usize) {
    let mut j = 0;
    while j < n {
        let mut acc = [[_mm512_setzero_ps(); 4]; MR];
        let wp = w.as_ptr().add(j * ldw);
        let mut kk = 0;
        while kk < k {
            let w0 = _mm512_loadu_ps(wp.add(kk));
            let w1 = _mm512_loadu_ps(wp.add(ldw + kk));
            let w2 = _mm512_loadu_ps(wp.add(2 * ldw + kk));
            let w3 = _mm512_loadu_ps(wp.add(3 * ldw + kk));
            for r in 0..MR {
                let xv = _mm512_loadu_ps(x.add(r * ldx + kk));
                acc[r][0] = _mm512_fmadd_ps(xv, w0, acc[r][0]);
                acc[r][1] = _mm512_fmadd_ps(xv, w1, acc[r][1]);
                acc[r][2] = _mm512_fmadd_ps(xv, w2, acc[r][2]);
                acc[r][3] = _mm512_fmadd_ps(xv, w3, acc[r][3]);
            }
            kk += 16;
        }
        for r in 0..MR {
            let yp = y.add(r * ldy + j);
            *yp = _mm512_reduce_add_ps(acc[r][0]);
            *yp.add(1) = _mm512_reduce_add_ps(acc[r][1]);
            *yp.add(2) = _mm512_reduce_add_ps(acc[r][2]);
            *yp.add(3) = _mm512_reduce_add_ps(acc[r][3]);
        }
        j += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_reference_all_row_tails() {
        let (k, n) = (320, 12);
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for mr in [0usize, 1, 2, 3, 4, 5, 7, 9] {
            let (ldx, ldw, ldy) = (k + 16, k + 32, n + 4);
            let x: Vec<f32> = (0..mr.max(1) * ldx).map(|_| rnd()).collect();
            let w: Vec<f32> = (0..n * ldw).map(|_| rnd()).collect();
            let mut y = vec![0f32; mr.max(1) * ldy];
            gemm_f32_nt(&x, ldx, mr, &w, ldw, n, k, &mut y, ldy);
            for i in 0..mr {
                for j in 0..n {
                    let r: f64 = (0..k).map(|t| x[i * ldx + t] as f64 * w[j * ldw + t] as f64).sum();
                    assert!((y[i * ldy + j] as f64 - r).abs() < 1e-4, "mr {mr} i {i} j {j}: {} vs {r}", y[i * ldy + j]);
                }
            }
        }
    }
}
