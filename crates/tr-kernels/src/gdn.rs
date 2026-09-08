//! Gated DeltaNet recurrence (Qwen3-Next / Flash-Next), f32 state per value head [Dk][Dv].
//! Per step and head h (key head h / rep):
//!   S *= exp(g_h);  kv = k^T S;  delta = (v - kv) * beta_h;  S += k (x) delta;  y = (q*scale)^T S
use std::arch::x86_64::*;

/// One decode step for one value head. `state` is [dk][dv] row-major. q,k: [dk]; v,y: [dv].
/// q is unscaled; `scale` (1/sqrt(dk)) is applied here. Columns [c0, c1) of dv are processed
/// (c1 - c0 must be a multiple of 16), so a head can be split across cores.
pub fn gdn_step_cols(state: &mut [f32], dk: usize, dv: usize, q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, scale: f32, y: &mut [f32], c0: usize, c1: usize) {
    unsafe { gdn_step_cols_impl(state, dk, dv, q, k, v, g, beta, scale, y, c0, c1) }
}

#[target_feature(enable = "avx512f")]
unsafe fn gdn_step_cols_impl(state: &mut [f32], dk: usize, dv: usize, q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, scale: f32, y: &mut [f32], c0: usize, c1: usize) {
    debug_assert!((c1 - c0) % 16 == 0);
    let decay = _mm512_set1_ps(g.exp());
    let vbeta = _mm512_set1_ps(beta);
    let sp = state.as_mut_ptr();
    let mut c = c0;
    while c < c1 {
        // pass 1: decay + kv
        let mut kv = _mm512_setzero_ps();
        for i in 0..dk {
            let p = sp.add(i * dv + c);
            let s = _mm512_mul_ps(_mm512_loadu_ps(p), decay);
            _mm512_storeu_ps(p, s);
            kv = _mm512_fmadd_ps(s, _mm512_set1_ps(*k.get_unchecked(i)), kv);
        }
        let delta = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v.as_ptr().add(c)), kv), vbeta);
        // pass 2: rank-1 update + output
        let mut yo = _mm512_setzero_ps();
        for i in 0..dk {
            let p = sp.add(i * dv + c);
            let s = _mm512_fmadd_ps(_mm512_set1_ps(*k.get_unchecked(i)), delta, _mm512_loadu_ps(p));
            _mm512_storeu_ps(p, s);
            yo = _mm512_fmadd_ps(s, _mm512_set1_ps(*q.get_unchecked(i) * scale), yo);
        }
        _mm512_storeu_ps(y.as_mut_ptr().add(c), yo);
        c += 16;
    }
}

/// One step for one 16-column chunk of a head whose state is stored chunk-major: `state` is
/// [dk][16] contiguous (8 KiB, L1-resident). q,k: [dk] (q unscaled); v,y: the chunk's 16 lanes.
/// Same arithmetic as `gdn_step_cols` (decay, kv, delta, rank-1 update, output) but 8 independent
/// accumulator chains, so it runs at the store/FMA throughput bound instead of FMA latency.
pub fn gdn_step16(state: &mut [f32], dk: usize, q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, scale: f32, y: &mut [f32]) {
    debug_assert!(state.len() >= dk * 16 && dk % 8 == 0 && v.len() >= 16 && y.len() >= 16);
    unsafe { gdn_step16_impl(state, dk, q, k, v, g, beta, scale, y) }
}

#[target_feature(enable = "avx512f")]
unsafe fn gdn_step16_impl(state: &mut [f32], dk: usize, q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, scale: f32, y: &mut [f32]) {
    let decay = _mm512_set1_ps(g.exp());
    let sp = state.as_mut_ptr();
    let kp = k.as_ptr();
    let qp = q.as_ptr();
    // pass 1: decay + kv = k^T S
    let mut acc = [_mm512_setzero_ps(); 8];
    let mut i = 0;
    while i < dk {
        for u in 0..8 {
            let p = sp.add((i + u) * 16);
            let s = _mm512_mul_ps(_mm512_loadu_ps(p), decay);
            _mm512_storeu_ps(p, s);
            acc[u] = _mm512_fmadd_ps(s, _mm512_set1_ps(*kp.add(i + u)), acc[u]);
        }
        i += 8;
    }
    let kv = _mm512_add_ps(
        _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3])),
        _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7])),
    );
    let delta = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v.as_ptr()), kv), _mm512_set1_ps(beta));
    // pass 2: S += k (x) delta; y = (q*scale)^T S
    let vscale = _mm512_set1_ps(scale);
    let mut acc = [_mm512_setzero_ps(); 8];
    let mut i = 0;
    while i < dk {
        for u in 0..8 {
            let p = sp.add((i + u) * 16);
            let s = _mm512_fmadd_ps(_mm512_set1_ps(*kp.add(i + u)), delta, _mm512_loadu_ps(p));
            _mm512_storeu_ps(p, s);
            acc[u] = _mm512_fmadd_ps(s, _mm512_set1_ps(*qp.add(i + u)), acc[u]);
        }
        i += 8;
    }
    let yo = _mm512_add_ps(
        _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3])),
        _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7])),
    );
    _mm512_storeu_ps(y.as_mut_ptr(), _mm512_mul_ps(yo, vscale));
}

/// Sequential run over `m` tokens for one 16-column chunk with chunk-major state ([dk][16]).
/// Rows of q/k (dk each, q unscaled, k as used in the update) live at `q[t*qs..]`, `k[t*ks..]`;
/// v/y lanes at `v[t*vs..+16]`, `y[t*ys..+16]`; `g[t]` is the log-decay, `beta[t]` the update gate.
/// The rank-1 update of token t and the decay/kv pass of token t+1 are fused into one sweep
/// over the state (one load + one store per row per token). Results equal `gdn_step16` up to
/// f32 summation order.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_seq(state: &mut [f32], dk: usize, m: usize, q: &[f32], qs: usize, k: &[f32], ks: usize, v: &[f32], vs: usize, g: &[f32], beta: &[f32], scale: f32, y: &mut [f32], ys: usize) {
    if m == 0 {
        return;
    }
    debug_assert!(state.len() >= dk * 16 && dk % 8 == 0);
    debug_assert!(q.len() >= (m - 1) * qs + dk && k.len() >= (m - 1) * ks + dk);
    debug_assert!(v.len() >= (m - 1) * vs + 16 && y.len() >= (m - 1) * ys + 16);
    debug_assert!(g.len() >= m && beta.len() >= m);
    unsafe { gdn_chunk_seq_impl(state, dk, m, q, qs, k, ks, v, vs, g, beta, scale, y, ys) }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gdn_chunk_seq_impl(state: &mut [f32], dk: usize, m: usize, q: &[f32], qs: usize, k: &[f32], ks: usize, v: &[f32], vs: usize, g: &[f32], beta: &[f32], scale: f32, y: &mut [f32], ys: usize) {
    #[inline(always)]
    unsafe fn hsum8(a: &[__m512; 8]) -> __m512 {
        _mm512_add_ps(
            _mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])),
            _mm512_add_ps(_mm512_add_ps(a[4], a[5]), _mm512_add_ps(a[6], a[7])),
        )
    }
    let sp = state.as_mut_ptr();
    let vscale = _mm512_set1_ps(scale);
    // prologue: decay + kv for token 0
    let mut kv = {
        let decay = _mm512_set1_ps(g[0].exp());
        let kp = k.as_ptr();
        let mut acc = [_mm512_setzero_ps(); 8];
        let mut i = 0;
        while i < dk {
            for u in 0..8 {
                let p = sp.add((i + u) * 16);
                let s = _mm512_mul_ps(_mm512_loadu_ps(p), decay);
                _mm512_storeu_ps(p, s);
                acc[u] = _mm512_fmadd_ps(s, _mm512_set1_ps(*kp.add(i + u)), acc[u]);
            }
            i += 8;
        }
        hsum8(&acc)
    };
    for t in 0..m {
        let kp = k.as_ptr().add(t * ks);
        let qp = q.as_ptr().add(t * qs);
        let delta = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v.as_ptr().add(t * vs)), kv), _mm512_set1_ps(beta[t]));
        let mut yacc = [_mm512_setzero_ps(); 8];
        if t + 1 < m {
            let kn = k.as_ptr().add((t + 1) * ks);
            let decay = _mm512_set1_ps(g[t + 1].exp());
            let mut kacc = [_mm512_setzero_ps(); 8];
            let mut i = 0;
            while i < dk {
                for u in 0..8 {
                    let p = sp.add((i + u) * 16);
                    let s = _mm512_fmadd_ps(_mm512_set1_ps(*kp.add(i + u)), delta, _mm512_loadu_ps(p));
                    yacc[u] = _mm512_fmadd_ps(s, _mm512_set1_ps(*qp.add(i + u)), yacc[u]);
                    let s = _mm512_mul_ps(s, decay);
                    _mm512_storeu_ps(p, s);
                    kacc[u] = _mm512_fmadd_ps(s, _mm512_set1_ps(*kn.add(i + u)), kacc[u]);
                }
                i += 8;
            }
            kv = hsum8(&kacc);
        } else {
            let mut i = 0;
            while i < dk {
                for u in 0..8 {
                    let p = sp.add((i + u) * 16);
                    let s = _mm512_fmadd_ps(_mm512_set1_ps(*kp.add(i + u)), delta, _mm512_loadu_ps(p));
                    _mm512_storeu_ps(p, s);
                    yacc[u] = _mm512_fmadd_ps(s, _mm512_set1_ps(*qp.add(i + u)), yacc[u]);
                }
                i += 8;
            }
        }
        _mm512_storeu_ps(y.as_mut_ptr().add(t * ys), _mm512_mul_ps(hsum8(&yacc), vscale));
    }
}

/// Scalar reference of the same step (whole head).
pub fn gdn_step_ref(state: &mut [f64], dk: usize, dv: usize, q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, scale: f32, y: &mut [f64]) {
    let decay = (g as f64).exp();
    let mut kv = vec![0f64; dv];
    for i in 0..dk {
        for j in 0..dv {
            state[i * dv + j] *= decay;
            kv[j] += state[i * dv + j] * k[i] as f64;
        }
    }
    let delta: Vec<f64> = (0..dv).map(|j| (v[j] as f64 - kv[j]) * beta as f64).collect();
    for j in 0..dv {
        y[j] = 0.0;
    }
    for i in 0..dk {
        for j in 0..dv {
            state[i * dv + j] += k[i] as f64 * delta[j];
            y[j] += state[i * dv + j] * (q[i] as f64 * scale as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_major_kernels_match_reference() {
        let (dk, dv, m) = (128usize, 128usize, 20usize);
        let mut seed = 5u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 % 20000) as f32 / 10000.0 - 1.0
        };
        let q: Vec<f32> = (0..m * dk).map(|_| rnd() * 0.1).collect();
        let k: Vec<f32> = (0..m * dk).map(|_| rnd() * 0.1).collect();
        let v: Vec<f32> = (0..m * dv).map(|_| rnd()).collect();
        let g: Vec<f32> = (0..m).map(|_| -0.5 * rnd().abs()).collect();
        let beta: Vec<f32> = (0..m).map(|_| 0.5 + 0.4 * rnd()).collect();
        let scale = 128f32.powf(-0.5);
        // reference over time
        let mut st_ref = vec![0f64; dk * dv];
        let mut y_ref = vec![0f64; m * dv];
        for t in 0..m {
            gdn_step_ref(&mut st_ref, dk, dv, &q[t * dk..(t + 1) * dk], &k[t * dk..(t + 1) * dk], &v[t * dv..(t + 1) * dv], g[t], beta[t], scale, &mut y_ref[t * dv..(t + 1) * dv]);
        }
        // step16 per token, chunk-major state
        let mut st_a = vec![0f32; dk * dv];
        let mut y_a = vec![0f32; m * dv];
        for t in 0..m {
            for ch in 0..dv / 16 {
                gdn_step16(&mut st_a[ch * dk * 16..(ch + 1) * dk * 16], dk, &q[t * dk..(t + 1) * dk], &k[t * dk..(t + 1) * dk], &v[t * dv + ch * 16..t * dv + ch * 16 + 16], g[t], beta[t], scale, &mut y_a[t * dv + ch * 16..t * dv + ch * 16 + 16]);
            }
        }
        // fused sequential run
        let mut st_b = vec![0f32; dk * dv];
        let mut y_b = vec![0f32; m * dv];
        for ch in 0..dv / 16 {
            gdn_chunk_seq(&mut st_b[ch * dk * 16..(ch + 1) * dk * 16], dk, m, &q, dk, &k, dk, &v[ch * 16..], dv, &g, &beta, scale, &mut y_b[ch * 16..], dv);
        }
        let mx = y_ref.iter().fold(0f64, |a, b| a.max(b.abs())).max(1e-3);
        for i in 0..m * dv {
            assert!((y_a[i] as f64 - y_ref[i]).abs() <= 1e-4 * mx, "step16 {i}: {} vs {}", y_a[i], y_ref[i]);
            assert!((y_b[i] as f64 - y_ref[i]).abs() <= 1e-4 * mx, "seq {i}: {} vs {}", y_b[i], y_ref[i]);
        }
        // states agree with the reference in chunk-major order
        for ch in 0..dv / 16 {
            for i in 0..dk {
                for l in 0..16 {
                    let r = st_ref[i * dv + ch * 16 + l];
                    let a = st_a[ch * dk * 16 + i * 16 + l] as f64;
                    let b = st_b[ch * dk * 16 + i * 16 + l] as f64;
                    assert!((a - r).abs() <= 1e-4 * (r.abs() + 1.0) && (b - r).abs() <= 1e-4 * (r.abs() + 1.0));
                }
            }
        }
    }
    #[test]
    fn step_matches_reference_over_time() {
        let (dk, dv) = (128usize, 128usize);
        let mut st = vec![0f32; dk * dv];
        let mut st_ref = vec![0f64; dk * dv];
        let mut seed = 3u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 % 20000) as f32 / 10000.0 - 1.0
        };
        for t in 0..20 {
            let q: Vec<f32> = (0..dk).map(|_| rnd()).collect();
            let mut k: Vec<f32> = (0..dk).map(|_| rnd()).collect();
            let n = k.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut k {
                *x /= n;
            }
            let v: Vec<f32> = (0..dv).map(|_| rnd()).collect();
            let g = -0.5 * rnd().abs();
            let beta = 0.5 + 0.4 * rnd();
            let mut y = vec![0f32; dv];
            let mut y_ref = vec![0f64; dv];
            gdn_step_cols(&mut st, dk, dv, &q, &k, &v, g, beta, 128f32.powf(-0.5), &mut y, 0, 64);
            gdn_step_cols(&mut st, dk, dv, &q, &k, &v, g, beta, 128f32.powf(-0.5), &mut y, 64, 128);
            gdn_step_ref(&mut st_ref, dk, dv, &q, &k, &v, g, beta, 128f32.powf(-0.5), &mut y_ref);
            let scale = y_ref.iter().fold(0f64, |a, b| a.max(b.abs())).max(1e-3);
            for j in 0..dv {
                assert!((y[j] as f64 - y_ref[j]).abs() <= 1e-4 * scale, "t {t} j {j}: {} vs {}", y[j], y_ref[j]);
            }
        }
    }
}
