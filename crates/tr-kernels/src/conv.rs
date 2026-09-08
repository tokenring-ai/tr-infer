//! Depthwise causal conv1d (kernel K) over channels with a [K-1][C] history state.
//! weight: [C][K] (numpy order of the GGUF tensor: channels outer, taps inner).
//! out[t][c] = sum_j w[c][j] * xpad[t + j][c], xpad = concat(state, x).

pub fn causal_conv1d(x: &[f32], t: usize, c: usize, w: &[f32], k: usize, state: &mut [f32], out: &mut [f32]) {
    assert_eq!(state.len(), (k - 1) * c);
    assert_eq!(w.len(), c * k);
    // Build xpad rows lazily: row r (0..t+k-1) = state[r] if r < k-1 else x[r-(k-1)]
    let row = |r: usize| -> &[f32] {
        if r < k - 1 {
            &state[r * c..(r + 1) * c]
        } else {
            &x[(r - (k - 1)) * c..(r - (k - 1) + 1) * c]
        }
    };
    for ti in 0..t {
        let o = &mut out[ti * c..(ti + 1) * c];
        for ch in 0..c {
            let mut acc = 0f32;
            for j in 0..k {
                acc += w[ch * k + j] * row(ti + j)[ch];
            }
            o[ch] = acc;
        }
    }
    // new state = last k-1 rows of xpad
    let total = t + k - 1;
    let mut new_state = vec![0f32; (k - 1) * c];
    for r in 0..k - 1 {
        new_state[r * c..(r + 1) * c].copy_from_slice(row(total - (k - 1) + r));
    }
    state.copy_from_slice(&new_state);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conv_chain_equals_single_shot() {
        let (c, k) = (5usize, 4usize);
        let w: Vec<f32> = (0..c * k).map(|i| (i as f32 * 0.3).sin()).collect();
        let x: Vec<f32> = (0..10 * c).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut st = vec![0f32; (k - 1) * c];
        let mut out = vec![0f32; 10 * c];
        causal_conv1d(&x, 10, c, &w, k, &mut st, &mut out);
        let mut st2 = vec![0f32; (k - 1) * c];
        let mut out2 = vec![0f32; 10 * c];
        causal_conv1d(&x[..3 * c], 3, c, &w, k, &mut st2, &mut out2[..3 * c]);
        causal_conv1d(&x[3 * c..], 7, c, &w, k, &mut st2, &mut out2[3 * c..]);
        for i in 0..10 * c {
            assert!((out[i] - out2[i]).abs() < 1e-6);
        }
        assert_eq!(st, st2);
    }
}

/// Causal depthwise conv + SiLU over `m` tokens for the 16 channels [c0, c0+16), sequential in
/// time with the history kept in registers. `x`/`y` rows are `xs`/`ys` floats apart; `w` is the
/// [C][K] tap tensor; `state` is the [K-1][cs] history (oldest row first) and is updated in place.
/// Same arithmetic as the scalar per-channel loop used before (fma order: taps 0..K-1 in sequence).
#[allow(clippy::too_many_arguments)]
pub fn conv_silu_seq16(x: &[f32], xs: usize, m: usize, w: &[f32], k: usize, state: &mut [f32], cs: usize, c0: usize, y: &mut [f32], ys: usize) {
    assert!(k >= 2 && k <= 8 && c0 + 16 <= cs);
    unsafe { conv_silu_seq16_impl(x, xs, m, w, k, state, cs, c0, y, ys) }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_silu_seq16_impl(x: &[f32], xs: usize, m: usize, w: &[f32], k: usize, state: &mut [f32], cs: usize, c0: usize, y: &mut [f32], ys: usize) {
    use std::arch::x86_64::*;
    // transpose the taps of these 16 channels: wt[j] = w[c0+l][j] for lane l
    let mut wt = [_mm512_setzero_ps(); 8];
    let mut tmp = [0f32; 16];
    for j in 0..k {
        for l in 0..16 {
            tmp[l] = *w.get_unchecked((c0 + l) * k + j);
        }
        wt[j] = _mm512_loadu_ps(tmp.as_ptr());
    }
    let mut hist = [_mm512_setzero_ps(); 8];
    for j in 0..k - 1 {
        hist[j] = _mm512_loadu_ps(state.as_ptr().add(j * cs + c0));
    }
    for t in 0..m {
        let xv = _mm512_loadu_ps(x.as_ptr().add(t * xs + c0));
        let mut acc = _mm512_mul_ps(wt[k - 1], xv);
        for j in 0..k - 1 {
            acc = _mm512_fmadd_ps(wt[j], hist[j], acc);
        }
        for j in 0..k - 2 {
            hist[j] = hist[j + 1];
        }
        hist[k - 2] = xv;
        _mm512_storeu_ps(y.as_mut_ptr().add(t * ys + c0), crate::elem::silu512(acc));
    }
    for j in 0..k - 1 {
        _mm512_storeu_ps(state.as_mut_ptr().add(j * cs + c0), hist[j]);
    }
}

#[cfg(test)]
mod tests16 {
    use super::*;
    #[test]
    fn seq16_matches_scalar() {
        let (c, k, m) = (32usize, 4usize, 9usize);
        let mut seed = 11u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 % 20000) as f32 / 10000.0 - 1.0
        };
        let x: Vec<f32> = (0..m * c).map(|_| rnd()).collect();
        let w: Vec<f32> = (0..c * k).map(|_| rnd()).collect();
        let st0: Vec<f32> = (0..(k - 1) * c).map(|_| rnd()).collect();
        let mut st_a = st0.clone();
        let mut out_a = vec![0f32; m * c];
        causal_conv1d(&x, m, c, &w, k, &mut st_a, &mut out_a);
        let mut st_b = st0.clone();
        let mut out_b = vec![0f32; m * c];
        for g in 0..c / 16 {
            conv_silu_seq16(&x, c, m, &w, k, &mut st_b, c, g * 16, &mut out_b, c);
        }
        for i in 0..m * c {
            let a = out_a[i] / (1.0 + (-out_a[i]).exp());
            assert!((a - out_b[i]).abs() < 1e-5, "{i}: {a} vs {}", out_b[i]);
        }
        for i in 0..(k - 1) * c {
            assert_eq!(st_a[i], st_b[i]);
        }
    }
}
