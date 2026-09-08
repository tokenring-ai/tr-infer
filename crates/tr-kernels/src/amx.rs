//! AMX-BF16 GEMM for prefill. Packed TQ strips are unpacked to bf16 (w = d*q + mn, rounded once)
//! in tile order, activations are converted to bf16 rows, products accumulate in f32 tiles.
//!
//! Unpacked strip layout: `[k/2 pair-rows][16 n][2]` u16, i.e. pair-row p holds (w[n][2p], w[n][2p+1])
//! for n in 0..16 — exactly one 64-byte B-tile row; tile block t is rows 16t..16t+16 (1 KiB, stride 64).
//! Activation rows are plain `[m][k]` bf16 (an A tile is 16 rows × 64 bytes at stride 2k).
//! Tiles: 0-3 accumulators (2 A × 2 B), 4-5 A, 6-7 B; every tile is 16 rows × 64 bytes except the
//! m-tail, which reconfigures the A/C rows.
use std::arch::asm;
use std::arch::x86_64::*;
use std::sync::OnceLock;
use tr_format::codec::{Codec, HDR_BYTES};

static AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Request XTILEDATA permission from the kernel (once per process, before worker threads start).
/// Returns false when AMX is unavailable or disabled with `TR_AMX=0`.
pub fn init() -> bool {
    *AVAILABLE.get_or_init(|| {
        if std::env::var("TR_AMX").map(|v| v == "0").unwrap_or(false) {
            return false;
        }
        let ret: i64;
        // arch_prctl(ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18)
        unsafe {
            asm!("syscall", inlateout("rax") 158i64 => ret, in("rdi") 0x1023i64, in("rsi") 18i64, lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        ret == 0
    })
}
pub fn available() -> bool {
    AVAILABLE.get().copied().unwrap_or(false)
}

/// 64-byte aligned storage unit for tile buffers.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Line(pub [u16; 32]);

/// Unpacked size of `n_strips` strips of a K-deep matrix, in `Line`s.
pub const fn strip_lines(k: usize) -> usize {
    k / 2
}

#[repr(C, align(64))]
pub(crate) struct TileCfg {
    pub(crate) palette: u8,
    pub(crate) start_row: u8,
    pub(crate) _res: [u8; 14],
    pub(crate) colsb: [u16; 16],
    pub(crate) rows: [u8; 16],
}

unsafe fn tile_config(rows_a0: usize, rows_a1: usize) {
    let mut cfg = TileCfg { palette: 1, start_row: 0, _res: [0; 14], colsb: [0; 16], rows: [0; 16] };
    let r0 = rows_a0.max(1) as u8;
    let r1 = rows_a1.max(1) as u8;
    for i in 0..8 {
        cfg.colsb[i] = 64;
    }
    cfg.rows[0] = r0;
    cfg.rows[1] = r0;
    cfg.rows[2] = r1;
    cfg.rows[3] = r1;
    cfg.rows[4] = r0;
    cfg.rows[5] = r1;
    cfg.rows[6] = 16;
    cfg.rows[7] = 16;
    asm!("ldtilecfg [{0}]", in(reg) &cfg as *const TileCfg, options(nostack));
}
unsafe fn tile_release() {
    asm!("tilerelease", options(nostack));
}

/// Convert rows [r0, r1) of an [rows][k] f32 matrix to bf16 (round to nearest even). k % 32 == 0.
pub fn rows_to_bf16(x: &[f32], k: usize, r0: usize, r1: usize, dst: &mut [u16]) {
    assert!(k % 32 == 0 && x.len() >= r1 * k && dst.len() >= r1 * k, "rows_to_bf16: k {k} rows {r0}..{r1}");
    unsafe { rows_to_bf16_impl(x, k, r0, r1, dst) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
unsafe fn rows_to_bf16_impl(x: &[f32], k: usize, r0: usize, r1: usize, dst: &mut [u16]) {
    let mut i = r0 * k;
    let end = r1 * k;
    while i < end {
        let lo = _mm512_loadu_ps(x.as_ptr().add(i));
        let hi = _mm512_loadu_ps(x.as_ptr().add(i + 16));
        let v: __m512i = std::mem::transmute(_mm512_cvtne2ps_pbh(hi, lo));
        _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, v);
        i += 32;
    }
}

/// Unpack `n_strips` consecutive packed strips (starting at `w`) into `dst` (n_strips * k/2 lines).
///
/// # Safety
/// `w` must hold `n_strips` strips of `k / codec.kb` blocks; `dst` must hold `n_strips * k / 2` lines.
pub unsafe fn unpack_strips(w: *const u8, k: usize, c: Codec, n_strips: usize, dst: *mut Line) {
    match (c.bits, c.kb) {
        (4, 32) => unpack_impl::<4, 32>(w, k, n_strips, dst as *mut u16),
        (5, 32) => unpack_impl::<5, 32>(w, k, n_strips, dst as *mut u16),
        (8, 32) => unpack_impl::<8, 32>(w, k, n_strips, dst as *mut u16),
        (4, 16) => unpack_impl::<4, 16>(w, k, n_strips, dst as *mut u16),
        (5, 16) => unpack_impl::<5, 16>(w, k, n_strips, dst as *mut u16),
        (8, 16) => unpack_impl::<8, 16>(w, k, n_strips, dst as *mut u16),
        _ => panic!("unsupported codec {c:?}"),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512bf16,f16c")]
unsafe fn unpack_impl<const BITS: u8, const KB: usize>(w: *const u8, k: usize, n_strips: usize, dst: *mut u16) {
    let c = Codec::new(BITS, KB);
    let bb = c.block_bytes();
    let nb = k / KB;
    let lo_mask = _mm512_set1_epi8(0x0F);
    let hi_bit = 0x10i8;
    let idx_lo = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let idx_hi = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    let mut wp = w;
    for s in 0..n_strips {
        let sd = dst.add(s * k * 16);
        for b in 0..nb {
            let blk = wp;
            wp = wp.add(bb);
            let d = _mm512_cvtph_ps(_mm256_loadu_si256(blk as *const __m256i));
            let mn = _mm512_cvtph_ps(_mm256_loadu_si256(blk.add(32) as *const __m256i));
            let (d_lo, d_hi) = (_mm512_permutexvar_ps(idx_lo, d), _mm512_permutexvar_ps(idx_hi, d));
            let (m_lo, m_hi) = (_mm512_permutexvar_ps(idx_lo, mn), _mm512_permutexvar_ps(idx_hi, mn));
            // v: 16 n × 4 consecutive k bytes (k0..k0+3) -> pair-rows k0/2 and k0/2 + 1
            let emit = |v: __m512i, k0: usize| {
                for half in 0..2 {
                    let w2 = _mm512_cvtepi32_epi16(if half == 0 { v } else { _mm512_srli_epi32(v, 16) });
                    let x = _mm512_cvtepu8_epi16(w2); // n0k0 n0k1 n1k0 n1k1 ...
                    let f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(x)));
                    let f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(x, 1)));
                    let f_lo = _mm512_fmadd_ps(f_lo, d_lo, m_lo);
                    let f_hi = _mm512_fmadd_ps(f_hi, d_hi, m_hi);
                    let out: __m512i = std::mem::transmute(_mm512_cvtne2ps_pbh(f_hi, f_lo));
                    _mm512_storeu_si512(sd.add((k0 / 2 + half) * 32) as *mut __m512i, out);
                }
            };
            let qb = blk.add(HDR_BYTES);
            if BITS == 8 {
                for j in 0..KB / 4 {
                    emit(_mm512_loadu_si512(qb.add(j * 64) as *const __m512i), b * KB + 4 * j);
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
                    emit(lo, b * KB + 4 * j);
                    emit(hi, b * KB + KB / 2 + 4 * j);
                }
            }
        }
    }
}

/// y[mi*ldy + col0 + 16*s + n] = sum_k a[mi][k] * w_s[n][k] for the `n_strips` unpacked strips in
/// `b`, activation rows `a` (m rows of k bf16, 64-byte aligned rows), f32 output. k % 32 == 0.
///
/// # Safety
/// `b` holds `n_strips * k/2` lines; `a` holds `m * k` u16; `y` rows of `ldy` floats for m rows.
pub unsafe fn gemm_bf16(b: *const Line, k: usize, n_strips: usize, a: *const u16, m: usize, y: *mut f32, ldy: usize, col0: usize) {
    gemm_bf16_stride(b, k, n_strips, a, k, m, y, ldy, col0)
}

/// As `gemm_bf16` with an explicit activation row stride `lda` (u16 elements, multiple of 32).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_bf16_stride(b: *const Line, k: usize, n_strips: usize, a: *const u16, lda: usize, m: usize, y: *mut f32, ldy: usize, col0: usize) {
    let mut cfg = TileRows::default();
    gemm_bf16_kept(b, k, n_strips, a, lda, m, y, ldy, col0, false, &mut cfg);
    tile_release();
}

/// The tile configuration currently loaded (A/C rows of the two row blocks); `default` = none.
/// Threaded through `gemm_bf16_kept` so a sequence of small GEMMs on one core reloads the
/// configuration only when the row split changes. Finish the sequence with `release_tiles`.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct TileRows(usize, usize);

/// Release the tile state after a sequence of `gemm_bf16_kept` calls.
pub fn release_tiles() {
    unsafe { tile_release() }
}

/// `gemm_bf16_stride` without the trailing tile release, tracking the loaded configuration in
/// `cfg`; with `acc` the products are added to the existing `y` values instead of overwriting.
///
/// # Safety
/// As `gemm_bf16`; with `acc` the written `y` region must be initialised.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_bf16_kept(b: *const Line, k: usize, n_strips: usize, a: *const u16, lda: usize, m: usize, y: *mut f32, ldy: usize, col0: usize, acc: bool, cfg: &mut TileRows) {
    assert!(k % 32 == 0 && lda % 32 == 0 && (a as usize) % 64 == 0, "gemm_bf16: k {k} lda {lda}");
    let kt = k / 32;
    let sa = lda * 2; // A row stride (bytes)
    let sy = ldy * 4;
    let mut sp = 0;
    while sp < n_strips {
        let nbt = (n_strips - sp).min(2);
        let b0 = b.add(sp * k / 2) as *const u8;
        let b1 = b.add((sp + 1) * k / 2) as *const u8;
        let mut r0 = 0;
        while r0 < m {
            let rows = (m - r0).min(32);
            let (ra0, ra1) = (rows.min(16), rows.saturating_sub(16));
            if *cfg != TileRows(ra0, ra1) {
                tile_config(ra0, ra1);
                *cfg = TileRows(ra0, ra1);
            }
            let a0 = (a as *const u8).add(r0 * sa);
            let a1 = a0.add(16 * sa);
            let y0 = (y as *mut u8).add(r0 * sy + (col0 + sp * 16) * 4);
            let y1 = y0.add(16 * sy);
            if acc {
                asm!("tileloadd tmm0, [{y} + {s}]", y = in(reg) y0, s = in(reg) sy, options(nostack, readonly));
                if nbt == 2 {
                    asm!("tileloadd tmm1, [{y} + {s}]", y = in(reg) y0.add(64), s = in(reg) sy, options(nostack, readonly));
                }
                if ra1 > 0 {
                    asm!("tileloadd tmm2, [{y} + {s}]", y = in(reg) y1, s = in(reg) sy, options(nostack, readonly));
                    if nbt == 2 {
                        asm!("tileloadd tmm3, [{y} + {s}]", y = in(reg) y1.add(64), s = in(reg) sy, options(nostack, readonly));
                    }
                }
            } else {
                asm!("tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", options(nostack, nomem));
            }
            match (ra1 > 0, nbt == 2) {
                (true, true) => {
                    for t in 0..kt {
                        asm!(
                            "tileloadd tmm4, [{a0} + {sa}]",
                            "tileloadd tmm6, [{b0} + {sb}]",
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            "tileloadd tmm7, [{b1} + {sb}]",
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            "tileloadd tmm5, [{a1} + {sa}]",
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            a0 = in(reg) a0.add(t * 64), a1 = in(reg) a1.add(t * 64), sa = in(reg) sa,
                            b0 = in(reg) b0.add(t * 1024), b1 = in(reg) b1.add(t * 1024), sb = in(reg) 64usize,
                            options(nostack, readonly)
                        );
                    }
                }
                (true, false) => {
                    for t in 0..kt {
                        asm!(
                            "tileloadd tmm4, [{a0} + {sa}]",
                            "tileloadd tmm6, [{b0} + {sb}]",
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            "tileloadd tmm5, [{a1} + {sa}]",
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            a0 = in(reg) a0.add(t * 64), a1 = in(reg) a1.add(t * 64), sa = in(reg) sa,
                            b0 = in(reg) b0.add(t * 1024), sb = in(reg) 64usize,
                            options(nostack, readonly)
                        );
                    }
                }
                (false, true) => {
                    for t in 0..kt {
                        asm!(
                            "tileloadd tmm4, [{a0} + {sa}]",
                            "tileloadd tmm6, [{b0} + {sb}]",
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            "tileloadd tmm7, [{b1} + {sb}]",
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            a0 = in(reg) a0.add(t * 64), sa = in(reg) sa,
                            b0 = in(reg) b0.add(t * 1024), b1 = in(reg) b1.add(t * 1024), sb = in(reg) 64usize,
                            options(nostack, readonly)
                        );
                    }
                }
                (false, false) => {
                    for t in 0..kt {
                        asm!(
                            "tileloadd tmm4, [{a0} + {sa}]",
                            "tileloadd tmm6, [{b0} + {sb}]",
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            a0 = in(reg) a0.add(t * 64), sa = in(reg) sa,
                            b0 = in(reg) b0.add(t * 1024), sb = in(reg) 64usize,
                            options(nostack, readonly)
                        );
                    }
                }
            }
            asm!("tilestored [{y} + {s}], tmm0", y = in(reg) y0, s = in(reg) sy, options(nostack));
            if nbt == 2 {
                asm!("tilestored [{y} + {s}], tmm1", y = in(reg) y0.add(64), s = in(reg) sy, options(nostack));
            }
            if ra1 > 0 {
                asm!("tilestored [{y} + {s}], tmm2", y = in(reg) y1, s = in(reg) sy, options(nostack));
                if nbt == 2 {
                    asm!("tilestored [{y} + {s}], tmm3", y = in(reg) y1.add(64), s = in(reg) sy, options(nostack));
                }
            }
            r0 += rows;
        }
        sp += nbt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tr_format::codec::pack_ref;

    fn bf16_round(x: f32) -> f32 {
        let b = x.to_bits();
        let r = ((b >> 16) & 1) + 0x7FFF;
        f32::from_bits(((b + r) >> 16) << 16)
    }

    #[test]
    fn amx_gemm_matches_reference() {
        if !init() {
            eprintln!("AMX unavailable, skipping");
            return;
        }
        for &(bits, kb) in &[(4u8, 32usize), (5, 32), (8, 32), (4, 16), (5, 16), (8, 16)] {
            let (rows, k, m) = (48usize, 128usize, 37usize);
            let c = Codec::new(bits, kb);
            let nb = k / kb;
            let mut seed = 1u64 + bits as u64;
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 33) as u32
            };
            let q: Vec<u32> = (0..rows * k).map(|_| rnd() % (1 << bits)).collect();
            let d: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(0.01 + (rnd() % 100) as f32 * 1e-4).to_bits()).collect();
            let mn: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(-0.1 + (rnd() % 100) as f32 * 1e-3).to_bits()).collect();
            let packed = pack_ref(rows, k, c, &q, &d, &mn);
            let x: Vec<f32> = (0..m * k).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
            let mut xh = vec![0u16; (m + 1) * k];
            let mut lines = vec![Line([0; 32]); rows / 16 * strip_lines(k)];
            let ldy = rows + 8;
            let mut y = vec![0f32; m * ldy];
            unsafe {
                unpack_strips(packed.as_ptr(), k, c, rows / 16, lines.as_mut_ptr());
                // aligned activation rows
                let off = (64 - (xh.as_ptr() as usize % 64)) % 64 / 2;
                rows_to_bf16(&x, k, 0, m, &mut xh[off..]);
                gemm_bf16(lines.as_ptr(), k, rows / 16, xh.as_ptr().add(off), m, y.as_mut_ptr(), ldy, 0);
            }
            // stage 1: unpacked weights
            for r in 0..rows {
                for kk in 0..k {
                    let w = half::f16::from_bits(d[r * nb + kk / kb]).to_f32() * q[r * k + kk] as f32 + half::f16::from_bits(mn[r * nb + kk / kb]).to_f32();
                    let got = f32::from_bits((lines[(r / 16) * strip_lines(k) + kk / 2].0[(r % 16) * 2 + kk % 2] as u32) << 16);
                    assert_eq!(got, bf16_round(w), "unpack bits {bits} kb {kb} r {r} k {kk}");
                }
            }
            // accumulate variant: a second pass with acc = true doubles every entry
            let mut y2 = y.clone();
            let mut cfg = TileRows::default();
            unsafe {
                let off = (64 - (xh.as_ptr() as usize % 64)) % 64 / 2;
                gemm_bf16_kept(lines.as_ptr(), k, rows / 16, xh.as_ptr().add(off), k, m, y2.as_mut_ptr(), ldy, 0, true, &mut cfg);
                release_tiles();
            }
            for (a, b) in y.iter().zip(&y2) {
                assert!((2.0 * a - b).abs() <= 1e-5 * (a.abs() + 1.0), "acc: {a} -> {b}");
            }
            for mi in 0..m {
                for r in 0..rows {
                    let mut s = 0f64;
                    for kk in 0..k {
                        let b = kk / kb;
                        let w = half::f16::from_bits(d[r * nb + b]).to_f32() * q[r * k + kk] as f32 + half::f16::from_bits(mn[r * nb + b]).to_f32();
                        s += bf16_round(w) as f64 * bf16_round(x[mi * k + kk]) as f64;
                    }
                    let got = y[mi * ldy + r] as f64;
                    assert!((got - s).abs() <= 1e-4 * (s.abs() + 1.0), "bits {bits} kb {kb} m {mi} r {r}: {got} vs {s}");
                }
                for r in rows..ldy {
                    assert_eq!(y[mi * ldy + r], 0.0);
                }
            }
        }
    }
}
