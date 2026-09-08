//! AMX-INT8 for the small-m TQ GEMMs (MoE routed experts, 1..16 rows per expert). Measured slower
//! than the VNNI `gemm_tq` at every m (kbench `moe`: 12.3 vs 10.7 µs per 80×2560 expert matrix at
//! m = 5, 24.0 vs 23.6 at m = 16): the per-32-k-block scale epilogue the TQ format forces costs the
//! same in both, and the tile traffic per block outweighs the multiply savings. Kept as a tested,
//! bit-exact reference for a block-64 format; not used by the engine.
//!
//! The packed
//! nibbles of one KB-deep block are expanded to the u8 B-tile rows `[KB/4][16 n][4]` (the packed
//! chunk layout already is that, low/high nibbles for the two halves of the block), one `tdpbsud`
//! per (strip, block) produces the exact int32 block dot for up to 16 activation rows, and the
//! per-block f32 epilogue is the same as `gemv_impl`'s: y += (d*sx) * dot + mn * (sx*sum).
//! Results are bit-identical to `gemm_tq`. Four blocks are in flight per step (accumulators
//! tmm0/1/6/7, A tmm2/3, B tmm4/5) so the tile latencies overlap.
use crate::amx::{Line, TileCfg};
use crate::quant::QActRef;
use std::arch::asm;
use std::arch::x86_64::*;
use tr_format::codec::{Codec, HDR_BYTES, STRIP};

pub const MAX_ROWS: usize = 16;

/// Compute strips [s0, s1) (`w` points at strip s0) against the int8 rows of `xq` (any m; not
/// `pair`), writing/accumulating f32 into y row `rows[i]` (or i) at columns 16*s..
///
/// # Safety
/// As `gemv::gemm_tq_rows`. Requires AMX (`amx::init()` true on this thread's process).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_tq_i8(w: *const u8, k: usize, c: Codec, s0: usize, s1: usize, xq: QActRef<'_>, y: *mut f32, ldy: usize, rows: Option<&[u32]>, accumulate: bool) {
    assert!(!xq.pair && xq.k == k && xq.kb == c.kb && k % c.kb == 0);
    let nb = k / c.kb;
    let mut r0 = 0;
    while r0 < xq.m {
        let g = (xq.m - r0).min(MAX_ROWS);
        let sub = QActRef { m: g, k, kb: c.kb, q: &xq.q[r0 * k..(r0 + g) * k], scale: &xq.scale[r0 * nb..(r0 + g) * nb], sum: &xq.sum[r0 * nb..(r0 + g) * nb], pair: false };
        let rp = rows.map(|r| r[r0..r0 + g].as_ptr()).unwrap_or(std::ptr::null());
        let yb = if rows.is_some() { y } else { y.add(r0 * ldy) };
        match (c.bits, c.kb) {
            (4, 32) => group::<4, 32>(w, k, s0, s1, sub, yb, ldy, rp, accumulate),
            (5, 32) => group::<5, 32>(w, k, s0, s1, sub, yb, ldy, rp, accumulate),
            (8, 32) => group::<8, 32>(w, k, s0, s1, sub, yb, ldy, rp, accumulate),
            (4, 16) => group::<4, 16>(w, k, s0, s1, sub, yb, ldy, rp, accumulate),
            (5, 16) => group::<5, 16>(w, k, s0, s1, sub, yb, ldy, rp, accumulate),
            (8, 16) => group::<8, 16>(w, k, s0, s1, sub, yb, ldy, rp, accumulate),
            _ => panic!("unsupported codec {c:?}"),
        }
        r0 += g;
    }
}

unsafe fn config(rows: usize, kb: usize) {
    let mut cfg = TileCfg { palette: 1, start_row: 0, _res: [0; 14], colsb: [0; 16], rows: [0; 16] };
    for t in 0..2 {
        cfg.colsb[t] = 64; // C: rows x 16 i32 (tmm0/1 and tmm6/7)
        cfg.rows[t] = rows as u8;
        cfg.colsb[6 + t] = 64;
        cfg.rows[6 + t] = rows as u8;
        cfg.colsb[2 + t] = kb as u16; // A: rows x KB i8
        cfg.rows[2 + t] = rows as u8;
        cfg.colsb[4 + t] = 64; // B: KB/4 x [16 n][4] u8
        cfg.rows[4 + t] = (kb / 4) as u8;
    }
    asm!("ldtilecfg [{0}]", in(reg) &cfg as *const TileCfg, options(nostack));
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn group<const BITS: u8, const KB: usize>(w: *const u8, k: usize, s0: usize, s1: usize, xq: QActRef<'_>, y: *mut f32, ldy: usize, rows: *const u32, accumulate: bool) {
    let m = xq.m;
    debug_assert!(m >= 1 && m <= MAX_ROWS);
    let c = Codec::new(BITS, KB);
    let bb = c.block_bytes();
    let nb = k / KB;
    config(m, KB);
    let lo_mask = _mm512_set1_epi8(0x0F);
    let hi_bit = 0x10i8;
    let sa = k; // A row stride (bytes)
    let mut wp = w;
    // B-row buffers and int32 stores for 4 blocks in flight (64-byte aligned)
    let mut bbuf = [Line([0; 32]); 4 * 8];
    let mut ctmp = [Line([0; 32]); 4 * MAX_ROWS];
    let build_b = |blk: *const u8, brow: *mut u8| {
        let qb = blk.add(HDR_BYTES);
        if BITS == 8 {
            for j in 0..KB / 4 {
                _mm512_store_si512(brow.add(j * 64) as *mut __m512i, _mm512_loadu_si512(qb.add(j * 64) as *const __m512i));
            }
        } else {
            let plane = qb.add(KB / 8 * 64);
            for j in 0..KB / 8 {
                let raw = _mm512_loadu_si512(qb.add(j * 64) as *const __m512i);
                let mut lo = _mm512_and_si512(raw, lo_mask);
                let mut hi = _mm512_and_si512(_mm512_srli_epi16(raw, 4), lo_mask);
                if BITS == 5 {
                    let mlo = *(plane.add(j * 16) as *const u64);
                    let mhi = *(plane.add(j * 16 + 8) as *const u64);
                    lo = _mm512_or_si512(lo, _mm512_maskz_set1_epi8(mlo, hi_bit));
                    hi = _mm512_or_si512(hi, _mm512_maskz_set1_epi8(mhi, hi_bit));
                }
                // low nibbles are k = 4j.., high nibbles k = KB/2 + 4j.. (as in gemv_impl)
                _mm512_store_si512(brow.add(j * 64) as *mut __m512i, lo);
                _mm512_store_si512(brow.add((KB / 8 + j) * 64) as *mut __m512i, hi);
            }
        }
    };
    for s in s0..s1 {
        let mut acc = [_mm512_setzero_ps(); MAX_ROWS];
        let mut b0 = 0;
        while b0 < nb {
            let cnt = (nb - b0).min(4);
            // prefetch the blocks after this chunk while it computes
            for i in 0..cnt {
                let p = wp.add((cnt + i) * bb);
                let mut off = 0;
                while off < bb {
                    _mm_prefetch(p.add(off) as *const i8, _MM_HINT_T0);
                    off += 64;
                }
            }
            for i in 0..cnt {
                build_b(wp.add(i * bb), bbuf.as_mut_ptr().add(i * 8) as *mut u8);
            }
            let a = xq.q.as_ptr().add(b0 * KB) as *const u8;
            let bp = bbuf.as_ptr() as *const u8;
            let cp = ctmp.as_mut_ptr() as *mut u8;
            // 4 independent accumulators (tmm0, tmm1, tmm6, tmm7); A/B registers alternate
            asm!(
                "tilezero tmm0",
                "tileloadd tmm4, [{b} + {s64}]",
                "tileloadd tmm2, [{a} + {sa}]",
                "tdpbsud tmm0, tmm2, tmm4",
                b = in(reg) bp, a = in(reg) a, sa = in(reg) sa, s64 = in(reg) 64usize, options(nostack, readonly)
            );
            if cnt > 1 {
                asm!(
                    "tilezero tmm1",
                    "tileloadd tmm5, [{b} + {s64}]",
                    "tileloadd tmm3, [{a} + {sa}]",
                    "tdpbsud tmm1, tmm3, tmm5",
                    b = in(reg) bp.add(512), a = in(reg) a.add(KB), sa = in(reg) sa, s64 = in(reg) 64usize, options(nostack, readonly)
                );
            }
            if cnt > 2 {
                asm!(
                    "tilezero tmm6",
                    "tileloadd tmm4, [{b} + {s64}]",
                    "tileloadd tmm2, [{a} + {sa}]",
                    "tdpbsud tmm6, tmm2, tmm4",
                    b = in(reg) bp.add(1024), a = in(reg) a.add(2 * KB), sa = in(reg) sa, s64 = in(reg) 64usize, options(nostack, readonly)
                );
            }
            if cnt > 3 {
                asm!(
                    "tilezero tmm7",
                    "tileloadd tmm5, [{b} + {s64}]",
                    "tileloadd tmm3, [{a} + {sa}]",
                    "tdpbsud tmm7, tmm3, tmm5",
                    b = in(reg) bp.add(1536), a = in(reg) a.add(3 * KB), sa = in(reg) sa, s64 = in(reg) 64usize, options(nostack, readonly)
                );
            }
            asm!("tilestored [{c} + {s64}], tmm0", c = in(reg) cp, s64 = in(reg) 64usize, options(nostack));
            if cnt > 1 {
                asm!("tilestored [{c} + {s64}], tmm1", c = in(reg) cp.add(1024), s64 = in(reg) 64usize, options(nostack));
            }
            if cnt > 2 {
                asm!("tilestored [{c} + {s64}], tmm6", c = in(reg) cp.add(2048), s64 = in(reg) 64usize, options(nostack));
            }
            if cnt > 3 {
                asm!("tilestored [{c} + {s64}], tmm7", c = in(reg) cp.add(3072), s64 = in(reg) 64usize, options(nostack));
            }
            for i in 0..cnt {
                let b = b0 + i;
                let blk = wp.add(i * bb);
                let ct = cp.add(i * 1024);
                let d = _mm512_cvtph_ps(_mm256_loadu_si256(blk as *const __m256i));
                let mn = _mm512_cvtph_ps(_mm256_loadu_si256(blk.add(32) as *const __m256i));
                for mi in 0..m {
                    let sx = *xq.scale.as_ptr().add(mi * nb + b);
                    let u = sx * (*xq.sum.as_ptr().add(mi * nb + b)) as f32;
                    let t = _mm512_mul_ps(d, _mm512_set1_ps(sx));
                    let dot = _mm512_load_si512(ct.add(mi * 64) as *const __m512i);
                    acc[mi] = _mm512_fmadd_ps(mn, _mm512_set1_ps(u), acc[mi]);
                    acc[mi] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), t, acc[mi]);
                }
            }
            wp = wp.add(cnt * bb);
            b0 += cnt;
        }
        for mi in 0..m {
            let row = if rows.is_null() { mi } else { *rows.add(mi) as usize };
            let yp = y.add(row * ldy + s * STRIP);
            if accumulate {
                _mm512_storeu_ps(yp, _mm512_add_ps(_mm512_loadu_ps(yp), acc[mi]));
            } else {
                _mm512_storeu_ps(yp, acc[mi]);
            }
        }
    }
    asm!("tilerelease", options(nostack));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv::gemm_tq_rows;
    use crate::quant::QAct;
    use tr_format::codec::pack_ref;

    #[test]
    fn matches_vnni_bit_exact() {
        if !crate::amx::init() {
            eprintln!("AMX unavailable, skipping");
            return;
        }
        for &(bits, kb) in &[(4u8, 32usize), (5, 32), (8, 32), (4, 16), (5, 16), (8, 16)] {
            let (rows, k) = (48usize, 160usize);
            let c = Codec::new(bits, kb);
            let nb = k / kb;
            let mut seed = 7u64 + bits as u64 * 31 + kb as u64;
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 33) as u32
            };
            let q: Vec<u32> = (0..rows * k).map(|_| rnd() % (1 << bits)).collect();
            let d: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32((rnd() % 100) as f32 * 0.001 + 0.001).to_bits()).collect();
            let mn: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32((rnd() % 100) as f32 * -0.002).to_bits()).collect();
            let packed = pack_ref(rows, k, c, &q, &d, &mn);
            for &m in &[1usize, 5, 16, 21] {
                let x: Vec<f32> = (0..m * k).map(|_| (rnd() % 2000) as f32 * 0.001 - 1.0).collect();
                let mut xq = QAct::zeros(m, k, kb);
                xq.quantize(&x);
                let ldy = rows + 8;
                // plain rows, then a row map with accumulate (rows reversed into a 2x taller y)
                let mut y_ref = vec![0f32; m * ldy];
                let mut y_amx = vec![0f32; m * ldy];
                unsafe {
                    gemm_tq_rows(packed.as_ptr(), k, c, 0, rows / 16, xq.as_ref(), y_ref.as_mut_ptr(), ldy, None, false);
                    gemm_tq_i8(packed.as_ptr(), k, c, 0, rows / 16, xq.as_ref(), y_amx.as_mut_ptr(), ldy, None, false);
                }
                assert_eq!(y_ref, y_amx, "bits {bits} kb {kb} m {m}");
                let map: Vec<u32> = (0..m as u32).map(|i| 2 * (m as u32 - 1 - i)).collect();
                let mut y_ref = vec![0.5f32; 2 * m * ldy];
                let mut y_amx = vec![0.5f32; 2 * m * ldy];
                unsafe {
                    gemm_tq_rows(packed.as_ptr().add(c.strip_bytes(k)), k, c, 1, rows / 16, xq.as_ref(), y_ref.as_mut_ptr(), ldy, Some(&map), true);
                    gemm_tq_i8(packed.as_ptr().add(c.strip_bytes(k)), k, c, 1, rows / 16, xq.as_ref(), y_amx.as_mut_ptr(), ldy, Some(&map), true);
                }
                assert_eq!(y_ref, y_amx, "row map: bits {bits} kb {kb} m {m}");
                assert!(y_amx.iter().any(|&v| v != 0.5));
            }
        }
    }
}
