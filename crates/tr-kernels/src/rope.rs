//! NEOX-style rotary embedding on the first `n_rot` dims of each head (ggml_rope_multi with all
//! four mrope position components equal reduces to this for text).
//! theta_i = base^(-2i/n_rot) for pair i in 0..n_rot/2; pairs are (i, i + n_rot/2).

/// Precomputed cos/sin table for a position range; row p holds n_rot/2 (cos, sin) pairs.
pub struct RopeTable {
    pub n_rot: usize,
    pub inv_freq: Vec<f32>,
    /// Interleaved M-RoPE sections (pairs per t / h / w component): pair j takes component
    /// j % 3 while j < 3 * sections[j % 3], else the t component. Text positions (t = h = w)
    /// make this the plain rotation above.
    pub sections: [usize; 3],
}

impl RopeTable {
    pub fn new(n_rot: usize, base: f32) -> RopeTable {
        Self::new_mrope(n_rot, base, [n_rot / 2, 0, 0])
    }
    pub fn new_mrope(n_rot: usize, base: f32, sections: [usize; 3]) -> RopeTable {
        let half = n_rot / 2;
        let inv_freq = (0..half).map(|i| (base as f64).powf(-2.0 * i as f64 / n_rot as f64) as f32).collect();
        RopeTable { n_rot, inv_freq, sections }
    }
    /// Rotate with a (t, h, w) position (image tokens); equals `apply(x, p)` when all three are p.
    pub fn apply3(&self, x: &mut [f32], pos: [u32; 3]) {
        if pos[0] == pos[1] && pos[1] == pos[2] {
            return self.apply(x, pos[0]);
        }
        let half = self.n_rot / 2;
        for i in 0..half {
            let c = i % 3;
            let p = if i < 3 * self.sections[c] { pos[c] } else { pos[0] };
            let theta = p as f64 * self.inv_freq[i] as f64;
            let (s, c) = (theta.sin() as f32, theta.cos() as f32);
            let x0 = x[i];
            let x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    }
    /// Rotate one head vector in place at position `pos`. `x.len() >= n_rot`.
    pub fn apply(&self, x: &mut [f32], pos: u32) {
        let half = self.n_rot / 2;
        for i in 0..half {
            let theta = pos as f64 * self.inv_freq[i] as f64;
            let (s, c) = (theta.sin() as f32, theta.cos() as f32);
            let x0 = x[i];
            let x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    }
    /// Rotate `n_heads` consecutive heads of `head_dim` each in `x` (len n_heads*head_dim).
    pub fn apply_heads(&self, x: &mut [f32], head_dim: usize, pos: u32) {
        for h in x.chunks_mut(head_dim) {
            self.apply(h, pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_preserves_norm_and_matches_formula() {
        let t = RopeTable::new(64, 1e7);
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).cos()).collect();
        let orig = x.clone();
        let n0: f32 = orig[..64].iter().map(|v| v * v).sum();
        t.apply(&mut x, 17);
        let n1: f32 = x[..64].iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-3);
        assert_eq!(&x[64..], &orig[64..]);
        // pair 0 rotates by pos*1.0 rad
        let (c, s) = ((17f32).cos(), (17f32).sin());
        assert!((x[0] - (orig[0] * c - orig[32] * s)).abs() < 1e-5);
    }
    #[test]
    fn mrope_interleaved_sections() {
        let t = RopeTable::new_mrope(64, 1e7, [11, 11, 10]);
        let orig: Vec<f32> = (0..256).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut a = orig.clone();
        let mut b = orig.clone();
        t.apply3(&mut a, [5, 5, 5]);
        t.apply(&mut b, 5);
        assert_eq!(a, b);
        // pair 1 rotates by h, pair 2 by w, pair 0 by t
        let mut c = orig.clone();
        t.apply3(&mut c, [3, 40, 700]);
        for (i, p) in [(0usize, 3u32), (1, 40), (2, 700), (3, 3), (31, 40)] {
            let th = p as f64 * t.inv_freq[i] as f64;
            let (s, co) = (th.sin() as f32, th.cos() as f32);
            assert!((c[i] - (orig[i] * co - orig[i + 32] * s)).abs() < 1e-4, "pair {i}");
        }
    }
}
