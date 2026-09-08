//! Causal softmax attention for a set of query heads over a KV cache (one kv head).
//! Queries and arithmetic are f32; the cache element type is generic (`KvElem`: f32, f16, bf16).
//! Decode fast path: one query per head against `n_kv` keys.
use std::arch::x86_64::*;

/// Element type of the K/V cache: 16 values load to one f32 vector; rows are stored from f32.
pub trait KvElem: Copy + Send + Sync + 'static {
    /// # Safety: `p` must point at 16 readable elements.
    unsafe fn load16(p: *const Self) -> __m512;
    /// # Safety: `p` must point at 16 writable elements.
    unsafe fn store16(p: *mut Self, v: __m512);
    fn to_f32(self) -> f32;
}
impl KvElem for f32 {
    #[inline(always)]
    unsafe fn load16(p: *const f32) -> __m512 {
        _mm512_loadu_ps(p)
    }
    #[inline(always)]
    unsafe fn store16(p: *mut f32, v: __m512) {
        _mm512_storeu_ps(p, v)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}
/// IEEE half (what llama.cpp keeps in its KV cache by default); F16C conversions, round-to-nearest.
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct F16(pub u16);
impl KvElem for F16 {
    #[inline(always)]
    unsafe fn load16(p: *const F16) -> __m512 {
        _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i))
    }
    #[inline(always)]
    unsafe fn store16(p: *mut F16, v: __m512) {
        _mm256_storeu_si256(p as *mut __m256i, _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v))
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        tr_format::codec::f16_to_f32(self.0)
    }
}
/// bfloat16 (truncated f32 exponent range, 8-bit mantissa); round-to-nearest-even on store.
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct Bf16(pub u16);
impl KvElem for Bf16 {
    #[inline(always)]
    unsafe fn load16(p: *const Bf16) -> __m512 {
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i))))
    }
    #[inline(always)]
    unsafe fn store16(p: *mut Bf16, v: __m512) {
        // RNE: bits + 0x7FFF + lsb(bits >> 16), then take the high half
        let b = _mm512_castps_si512(v);
        let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(b), _mm512_set1_epi32(1));
        let r = _mm512_srli_epi32::<16>(_mm512_add_epi32(_mm512_add_epi32(b, _mm512_set1_epi32(0x7FFF)), lsb));
        _mm256_storeu_si256(p as *mut __m256i, _mm512_cvtepi32_epi16(r))
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// Store an f32 row into the cache (len multiple of 16).
#[inline]
pub fn store_row<T: KvElem>(dst: &mut [T], src: &[f32]) {
    debug_assert!(dst.len() == src.len() && src.len() % 16 == 0);
    let mut i = 0;
    while i < src.len() {
        unsafe { T::store16(dst.as_mut_ptr().add(i), _mm512_loadu_ps(src.as_ptr().add(i))) };
        i += 16;
    }
}

/// q . row for one cache row (len multiple of 16).
#[inline]
pub fn dot_kv<T: KvElem>(q: &[f32], row: &[T]) -> f32 {
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= q.len() {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q.as_ptr().add(i)), T::load16(row.as_ptr().add(i)), acc);
            i += 16;
        }
        _mm512_reduce_add_ps(acc)
    }
}

/// o += s * row (len multiple of 16).
#[inline]
pub fn axpy_kv<T: KvElem>(o: &mut [f32], row: &[T], s: f32) {
    unsafe {
        let sv = _mm512_set1_ps(s);
        let mut i = 0;
        while i + 16 <= o.len() {
            let p = o.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, _mm512_fmadd_ps(sv, T::load16(row.as_ptr().add(i)), _mm512_loadu_ps(p)));
            i += 16;
        }
    }
}

/// out[h*dv..] = softmax(q_h . K^T * scale) V  for each of `n_q_heads` query heads sharing one kv head.
/// k, v: [n_kv][d] row-major (kv head already selected). q: [n_q_heads][d].
pub fn attend_one_kv_head<T: KvElem>(q: &[f32], k: &[T], v: &[T], n_kv: usize, d: usize, scale: f32, out: &mut [f32], scratch: &mut Vec<f32>) {
    attend_ranges(q, k, v, &[(0, n_kv as u32)], d, scale, out, scratch)
}

/// Same, restricted to the cache rows in `ranges` ((start, len) in tokens; QSA selection).
#[allow(clippy::too_many_arguments)]
pub fn attend_ranges<T: KvElem>(q: &[f32], k: &[T], v: &[T], ranges: &[(u32, u32)], d: usize, scale: f32, out: &mut [f32], scratch: &mut Vec<f32>) {
    let mut rows = Vec::new();
    attend_ranges_pf(q, k, v, ranges, d, scale, out, scratch, &mut rows)
}

/// `attend_ranges` with a caller-provided row scratch (prefetching kernel).
#[allow(clippy::too_many_arguments)]
pub fn attend_ranges_pf<T: KvElem>(q: &[f32], k: &[T], v: &[T], ranges: &[(u32, u32)], d: usize, scale: f32, out: &mut [f32], scratch: &mut Vec<f32>, rows: &mut Vec<u32>) {
    let n_heads = q.len() / d;
    let total: usize = ranges.iter().map(|r| r.1 as usize).sum();
    for h in 0..n_heads {
        let o = &mut out[h * d..(h + 1) * d];
        let (_, sum) = attend_partial(&q[h * d..(h + 1) * d], k, v, ranges, 0, total, d, scale, o, scratch, rows);
        let inv = 1.0 / sum;
        for x in o.iter_mut() {
            *x *= inv;
        }
    }
}

/// Partial attention of one query head over tokens [t0, t1) of the concatenated `ranges`
/// (multi-core split of one head). Returns (max score, sum of exp) and leaves the
/// unnormalised weighted value sum in `out` (all zero / -inf when the window is empty).
/// Merge partials with `merge_partials`. Rows are software-prefetched `PF` tokens ahead
/// because QSA selections are scattered 4-token blocks that defeat the hardware prefetcher.
#[allow(clippy::too_many_arguments)]
pub fn attend_partial<T: KvElem>(q: &[f32], k: &[T], v: &[T], ranges: &[(u32, u32)], t0: usize, t1: usize, d: usize, scale: f32, out: &mut [f32], scratch: &mut Vec<f32>, rows: &mut Vec<u32>) -> (f32, f32) {
    out[..d].fill(0.0);
    if t1 <= t0 {
        return (f32::NEG_INFINITY, 0.0);
    }
    // rows of the window in token order
    rows.clear();
    let mut off = 0usize;
    for &(s0, len) in ranges {
        let (a, b) = (off.max(t0), (off + len as usize).min(t1));
        if a < b {
            rows.extend((a..b).map(|j| s0 + (j - off) as u32));
        }
        off += len as usize;
        if off >= t1 {
            break;
        }
    }
    let n = rows.len();
    scratch.clear();
    scratch.resize(n, 0.0);
    const PF: usize = 6;
    let row_bytes = d * std::mem::size_of::<T>();
    let mut mx = f32::NEG_INFINITY;
    for i in 0..n {
        if i + PF < n {
            prefetch_row(k, rows[i + PF] as usize * d, row_bytes);
        }
        let row = rows[i] as usize;
        let s = dot_kv(q, &k[row * d..(row + 1) * d]) * scale;
        scratch[i] = s;
        mx = mx.max(s);
    }
    let sum = crate::elem::exp_sub_inplace(scratch, mx);
    for i in 0..n {
        if i + PF < n {
            prefetch_row(v, rows[i + PF] as usize * d, row_bytes);
        }
        let row = rows[i] as usize;
        axpy_kv(&mut out[..d], &v[row * d..(row + 1) * d], scratch[i]);
    }
    (mx, sum)
}

#[inline(always)]
fn prefetch_row<T: KvElem>(buf: &[T], off: usize, bytes: usize) {
    let p = unsafe { (buf.as_ptr() as *const i8).add(off * std::mem::size_of::<T>()) };
    let mut b = 0;
    while b < bytes {
        unsafe { _mm_prefetch(p.add(b), _MM_HINT_T0) };
        b += 64;
    }
}

/// Queries per block in `attend_block`.
pub const QB: usize = 8;

/// Query-blocked attention (prefill): `QB` queries `q[QB][d]` against the cache rows listed in
/// `rows` (ascending), each with a membership mask `memb[i]` (bit qi set = row visible to query
/// qi). Every K/V row is read once for all 8 queries. `out[QB][d]`; `scores` is scratch.
/// A query with no visible row (mask never set) gets a zero output.
#[allow(clippy::too_many_arguments)]
pub fn attend_block<T: KvElem>(q: &[f32], k: &[T], v: &[T], rows: &[u32], memb: &[u8], d: usize, scale: f32, out: &mut [f32], scores: &mut Vec<f32>) {
    debug_assert!(q.len() >= QB * d && out.len() >= QB * d && d % 16 == 0);
    let n = rows.len();
    scores.clear();
    scores.resize(n * QB, 0.0);
    const PF: usize = 6;
    let row_bytes = d * std::mem::size_of::<T>();
    let nchunk = d / 16;
    let mut mx = [f32::NEG_INFINITY; QB];
    unsafe {
        // K pass: 8 dots per row
        for i in 0..n {
            if i + PF < n {
                prefetch_row(k, rows[i + PF] as usize * d, row_bytes);
            }
            let krow = k.as_ptr().add(rows[i] as usize * d);
            let mut acc = [_mm512_setzero_ps(); QB];
            for c in 0..nchunk {
                let kv = T::load16(krow.add(c * 16));
                for (qi, a) in acc.iter_mut().enumerate() {
                    *a = _mm512_fmadd_ps(_mm512_loadu_ps(q.as_ptr().add(qi * d + c * 16)), kv, *a);
                }
            }
            let mb = memb[i];
            for (qi, a) in acc.iter().enumerate() {
                let s = if mb & (1 << qi) != 0 { _mm512_reduce_add_ps(*a) * scale } else { f32::NEG_INFINITY };
                scores[i * QB + qi] = s;
                if s > mx[qi] {
                    mx[qi] = s;
                }
            }
        }
        // softmax per query (in place, 2 rows = 16 lanes per vector); empty queries stay zero
        let mut inv = [0f32; QB];
        let mut sum = [0f32; QB];
        {
            let mut mxv = [0f32; 16];
            mxv[..QB].copy_from_slice(&mx);
            mxv[QB..].copy_from_slice(&mx);
            for (qi, m) in mxv.iter_mut().enumerate() {
                if *m == f32::NEG_INFINITY {
                    *m = 0.0; // empty query: exp(-inf - 0) handled by the -inf mask
                }
                let _ = qi;
            }
            let mv = _mm512_loadu_ps(mxv.as_ptr());
            let ninf = _mm512_set1_ps(f32::NEG_INFINITY);
            let mut acc = _mm512_setzero_ps();
            let total = n * QB;
            let mut i = 0;
            while i + 16 <= total {
                let p = scores.as_mut_ptr().add(i);
                let s = _mm512_loadu_ps(p);
                let keep = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(s, ninf);
                let e = _mm512_maskz_mov_ps(keep, crate::elem::exp512(_mm512_sub_ps(s, mv)));
                _mm512_storeu_ps(p, e);
                acc = _mm512_add_ps(acc, e);
                i += 16;
            }
            let mut sv = [0f32; 16];
            _mm512_storeu_ps(sv.as_mut_ptr(), acc);
            for qi in 0..QB {
                sum[qi] = sv[qi] + sv[qi + QB];
            }
            while i < total {
                let qi = i % QB;
                let s = scores[i];
                let e = if s == f32::NEG_INFINITY { 0.0 } else { (s - mx[qi]).exp() };
                scores[i] = e;
                sum[qi] += e;
                i += 1;
            }
        }
        for qi in 0..QB {
            inv[qi] = if sum[qi] > 0.0 { 1.0 / sum[qi] } else { 0.0 };
        }
        // V pass: out[qi] += e[i][qi] * v_row (unnormalised, as the per-query kernel; the 1/sum
        // scale is applied once at the end so results match `attend_ranges` bit for bit)
        out[..QB * d].fill(0.0);
        for i in 0..n {
            if i + PF < n {
                prefetch_row(v, rows[i + PF] as usize * d, row_bytes);
            }
            let vrow = v.as_ptr().add(rows[i] as usize * d);
            let mut p = [_mm512_setzero_ps(); QB];
            for qi in 0..QB {
                p[qi] = _mm512_set1_ps(scores[i * QB + qi]);
            }
            for c in 0..nchunk {
                let vv = T::load16(vrow.add(c * 16));
                for qi in 0..QB {
                    let o = out.as_mut_ptr().add(qi * d + c * 16);
                    _mm512_storeu_ps(o, _mm512_fmadd_ps(p[qi], vv, _mm512_loadu_ps(o)));
                }
            }
        }
        for qi in 0..QB {
            for x in out[qi * d..(qi + 1) * d].iter_mut() {
                *x *= inv[qi];
            }
        }
    }
}

/// Combine `n` partials (max_i, sum_i, out_i) into the normalised attention output `o`.
pub fn merge_partials(parts: &[(f32, f32)], outs: &[&[f32]], d: usize, o: &mut [f32]) {
    let mx = parts.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
    let mut l = 0f32;
    o[..d].fill(0.0);
    for (p, po) in parts.iter().zip(outs) {
        if p.1 > 0.0 {
            let w = (p.0 - mx).exp();
            l += p.1 * w;
            crate::elem::axpy(&mut o[..d], &po[..d], w);
        }
    }
    let inv = 1.0 / l;
    for x in o[..d].iter_mut() {
        *x *= inv;
    }
}

/// a . b with four independent accumulator chains (a single chain is FMA-latency bound at
/// 16 MAC / 4 cycles; long dots such as the 10 240-wide hyper-connection injections run 4x faster).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut i = 0;
        while i + 64 <= n {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i + 16)), _mm512_loadu_ps(pb.add(i + 16)), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i + 32)), _mm512_loadu_ps(pb.add(i + 32)), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i + 48)), _mm512_loadu_ps(pb.add(i + 48)), acc3);
            i += 64;
        }
        while i + 16 <= n {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), acc0);
            i += 16;
        }
        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        let mut s = _mm512_reduce_add_ps(acc);
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_caches_track_f32() {
        let (nkv, d) = (40usize, 32usize);
        let q: Vec<f32> = (0..d).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.05).collect();
        let k: Vec<f32> = (0..nkv * d).map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.05).collect();
        let v: Vec<f32> = (0..nkv * d).map(|i| ((i * 5 % 23) as f32 - 11.0) * 0.05).collect();
        let mut kh = vec![F16(0); nkv * d];
        let mut vh = vec![F16(0); nkv * d];
        let mut kb = vec![Bf16(0); nkv * d];
        let mut vb = vec![Bf16(0); nkv * d];
        for r in 0..nkv {
            store_row(&mut kh[r * d..(r + 1) * d], &k[r * d..(r + 1) * d]);
            store_row(&mut vh[r * d..(r + 1) * d], &v[r * d..(r + 1) * d]);
            store_row(&mut kb[r * d..(r + 1) * d], &k[r * d..(r + 1) * d]);
            store_row(&mut vb[r * d..(r + 1) * d], &v[r * d..(r + 1) * d]);
        }
        assert!((kh[5].to_f32() - k[5]).abs() < 1e-3 && (kb[5].to_f32() - k[5]).abs() < 4e-3);
        // bf16 RNE: 1.0 + 2^-9 (halfway between two bf16 values) rounds to even (1.0)
        let mut one = [Bf16(0); 16];
        store_row(&mut one, &[1.0 + 2f32.powi(-9); 16]);
        assert_eq!(one[0].0, 0x3F80);
        let (mut want, mut sc) = (vec![0f32; d], Vec::new());
        attend_one_kv_head(&q, &k, &v, nkv, d, 0.25, &mut want, &mut sc);
        for (name, got) in [("f16", { let mut o = vec![0f32; d]; attend_one_kv_head(&q, &kh, &vh, nkv, d, 0.25, &mut o, &mut sc); o }), ("bf16", { let mut o = vec![0f32; d]; attend_one_kv_head(&q, &kb, &vb, nkv, d, 0.25, &mut o, &mut sc); o })] {
            let tol = if name == "f16" { 2e-3 } else { 1e-2 };
            for i in 0..d {
                assert!((got[i] - want[i]).abs() < tol, "{name} {i}: {} vs {}", got[i], want[i]);
            }
        }
    }
    #[test]
    fn block_matches_per_query() {
        let (nkv, d) = (53usize, 32usize);
        let q: Vec<f32> = (0..QB * d).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.05).collect();
        let k: Vec<f32> = (0..nkv * d).map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.05).collect();
        let v: Vec<f32> = (0..nkv * d).map(|i| ((i * 5 % 23) as f32 - 11.0) * 0.05).collect();
        // query qi sees rows [0, 20 + 3*qi) plus block [40, 44) for even qi; query 7 sees nothing
        let mut memb = vec![0u8; nkv];
        let mut sets: Vec<Vec<(u32, u32)>> = Vec::new();
        for qi in 0..QB {
            let mut rg = Vec::new();
            if qi < 7 {
                rg.push((0u32, (20 + 3 * qi) as u32));
                if qi % 2 == 0 {
                    rg.push((40, 4));
                }
            }
            for &(s0, len) in &rg {
                for j in s0..s0 + len {
                    memb[j as usize] |= 1 << qi;
                }
            }
            sets.push(rg);
        }
        let rows: Vec<u32> = (0..nkv as u32).filter(|&j| memb[j as usize] != 0).collect();
        let memb_rows: Vec<u8> = rows.iter().map(|&j| memb[j as usize]).collect();
        let mut out = vec![0f32; QB * d];
        let mut sc = Vec::new();
        attend_block(&q, &k, &v, &rows, &memb_rows, d, 0.25, &mut out, &mut sc);
        for qi in 0..QB {
            let mut want = vec![0f32; d];
            if !sets[qi].is_empty() {
                attend_ranges(&q[qi * d..(qi + 1) * d], &k, &v, &sets[qi], d, 0.25, &mut want, &mut sc);
            }
            for i in 0..d {
                assert!((out[qi * d + i] - want[i]).abs() < 1e-5, "q{qi} {i}: {} vs {}", out[qi * d + i], want[i]);
            }
        }
    }
    #[test]
    fn split_matches_whole() {
        let (nkv, d) = (37usize, 32usize);
        let q: Vec<f32> = (0..d).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.05).collect();
        let k: Vec<f32> = (0..nkv * d).map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.05).collect();
        let v: Vec<f32> = (0..nkv * d).map(|i| ((i * 5 % 23) as f32 - 11.0) * 0.05).collect();
        let ranges = [(0u32, 5u32), (9, 12), (30, 7)];
        let mut want = vec![0f32; d];
        let mut sc = Vec::new();
        attend_ranges(&q, &k, &v, &ranges, d, 0.25, &mut want, &mut sc);
        let total = 24;
        let ns = 5;
        let mut parts = Vec::new();
        let mut outs = Vec::new();
        for c in 0..ns {
            let (t0, t1) = (c * total / ns, (c + 1) * total / ns);
            let mut o = vec![0f32; d];
            let mut rows = Vec::new();
            parts.push(attend_partial(&q, &k, &v, &ranges, t0, t1, d, 0.25, &mut o, &mut sc, &mut rows));
            outs.push(o);
        }
        let refs: Vec<&[f32]> = outs.iter().map(|o| o.as_slice()).collect();
        let mut got = vec![0f32; d];
        merge_partials(&parts, &refs, d, &mut got);
        for i in 0..d {
            assert!((got[i] - want[i]).abs() < 1e-5, "{i}: {} vs {}", got[i], want[i]);
        }
    }
    #[test]
    fn matches_scalar() {
        let (nq, nkv, d) = (3usize, 7usize, 32usize);
        let q: Vec<f32> = (0..nq * d).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.05).collect();
        let k: Vec<f32> = (0..nkv * d).map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.05).collect();
        let v: Vec<f32> = (0..nkv * d).map(|i| ((i * 5 % 23) as f32 - 11.0) * 0.05).collect();
        let mut out = vec![0f32; nq * d];
        let mut sc = Vec::new();
        attend_one_kv_head(&q, &k, &v, nkv, d, 0.25, &mut out, &mut sc);
        for h in 0..nq {
            let s: Vec<f64> = (0..nkv).map(|j| (0..d).map(|i| q[h * d + i] as f64 * k[j * d + i] as f64).sum::<f64>() * 0.25).collect();
            let mx = s.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            for i in 0..d {
                let want: f64 = (0..nkv).map(|j| e[j] / z * v[j * d + i] as f64).sum();
                assert!((out[h * d + i] as f64 - want).abs() < 1e-5);
            }
        }
    }
}
