use rand::{Rng, SeedableRng};

/// A sampling distribution over the top-k / top-p truncated vocabulary (probabilities sum to 1).
#[derive(Clone, Debug, Default)]
pub struct Dist {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}
impl Dist {
    pub fn prob(&self, id: u32) -> f32 {
        self.ids.iter().position(|&x| x == id).map(|i| self.probs[i]).unwrap_or(0.0)
    }
    /// Inverse-CDF draw for the uniform `r` in [0, 1).
    pub fn pick(&self, r: f32) -> u32 {
        let mut acc = 0.0;
        for (i, p) in self.probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return self.ids[i];
            }
        }
        self.ids[self.ids.len() - 1]
    }
}

/// Candidates further than this (times the temperature) below the maximum logit are dropped.
pub const DIST_MARGIN: f32 = 24.0;

/// Maximum of a slice (vectorised: 16 independent accumulators).
pub fn max_f32(x: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 16];
    let mut it = x.chunks_exact(16);
    for ch in &mut it {
        for (a, &v) in acc.iter_mut().zip(ch) {
            *a = if v > *a { v } else { *a };
        }
    }
    let mut m = acc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for &v in it.remainder() {
        m = m.max(v);
    }
    m
}

pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: rand::rngs::StdRng,
}

impl Sampler {
    pub fn new(temp: f32, top_k: usize, top_p: f32, seed: u64) -> Sampler {
        Sampler { temp, top_k, top_p, rng: rand::rngs::StdRng::seed_from_u64(seed) }
    }
    /// Largest logit; lowest id among ties.
    pub fn greedy(logits: &[f32]) -> u32 {
        let mx = max_f32(logits);
        logits.iter().position(|&l| l == mx).unwrap_or(0) as u32
    }
    /// The distribution the current settings sample from: temperature 0 is a point mass on the
    /// argmax; otherwise softmax(logits/temp) over the top-k ids, truncated to top-p and
    /// renormalised. Tokens more than `DIST_MARGIN` (in units of the temperature) below the
    /// maximum are dropped before the top-k selection: their relative probability is below
    /// e^-24 ~ 4e-11, far under the resolution of an f32 uniform, and it makes the selection
    /// run over hundreds of candidates instead of the whole vocabulary (0.8 ms -> ~0.1 ms).
    pub fn dist(&self, logits: &[f32]) -> Dist {
        if self.temp <= 0.0 {
            return Dist { ids: vec![Self::greedy(logits)], probs: vec![1.0] };
        }
        let thr = max_f32(logits) - DIST_MARGIN * self.temp;
        let mut cands: Vec<(u32, f32)> = logits.iter().enumerate().filter(|(_, &l)| l >= thr).map(|(i, &l)| (i as u32, l)).collect();
        Self::dist_from(&mut cands, self.temp, self.top_k, self.top_p)
    }
    /// `dist` over an explicit candidate set `(id, logit)` (any order; consumed). Ties are broken
    /// towards the lower id, so the result is a pure function of the set.
    pub fn dist_from(cands: &mut Vec<(u32, f32)>, temp: f32, top_k: usize, top_p: f32) -> Dist {
        assert!(!cands.is_empty());
        let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0));
        if temp <= 0.0 {
            let best = cands.iter().copied().min_by(cmp).unwrap();
            return Dist { ids: vec![best.0], probs: vec![1.0] };
        }
        let k = if top_k == 0 { cands.len() } else { top_k.min(cands.len()) };
        if k < cands.len() {
            cands.select_nth_unstable_by(k - 1, cmp);
            cands.truncate(k);
        }
        cands.sort_by(cmp);
        let mx = cands[0].1;
        let mut probs: Vec<f32> = cands.iter().map(|c| ((c.1 - mx) / temp).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
        let mut ids: Vec<u32> = cands.iter().map(|c| c.0).collect();
        if top_p < 1.0 {
            let mut acc = 0.0;
            let mut cut = probs.len();
            for (i, p) in probs.iter().enumerate() {
                acc += p;
                if acc >= top_p {
                    cut = i + 1;
                    break;
                }
            }
            probs.truncate(cut);
            ids.truncate(cut);
            let s: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= s;
            }
        }
        Dist { ids, probs }
    }
    pub fn uniform(&mut self) -> f32 {
        self.rng.random()
    }
    /// Draw from a distribution (inverse CDF).
    pub fn sample_dist(&mut self, d: &Dist) -> u32 {
        let r: f32 = self.rng.random();
        d.pick(r)
    }
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temp <= 0.0 {
            return Self::greedy(logits);
        }
        let d = self.dist(logits);
        self.sample_dist(&d)
    }
    /// Speculative sampling's rejection fallback: draw from norm(max(p - q, 0)).
    pub fn residual_sample(&mut self, p: &Dist, q: &Dist) -> u32 {
        let mut ids = Vec::with_capacity(p.ids.len());
        let mut probs = Vec::with_capacity(p.ids.len());
        for (i, &id) in p.ids.iter().enumerate() {
            let r = p.probs[i] - q.prob(id);
            if r > 0.0 {
                ids.push(id);
                probs.push(r);
            }
        }
        if ids.is_empty() {
            // p <= q everywhere can only be p == q up to rounding: fall back to p itself
            return self.sample_dist(p);
        }
        let s: f32 = probs.iter().sum();
        for x in &mut probs {
            *x /= s;
        }
        self.sample_dist(&Dist { ids, probs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dist_matches_sample_semantics() {
        let logits = vec![1.0, 3.0, 2.0, -1.0, 0.5];
        let s = Sampler::new(0.0, 20, 0.95, 1);
        assert_eq!(s.dist(&logits).ids, vec![1]);
        let s = Sampler::new(1.0, 3, 1.0, 1);
        let d = s.dist(&logits);
        assert_eq!(d.ids, vec![1, 2, 0]);
        assert!((d.probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(d.prob(1) > d.prob(2) && d.prob(2) > d.prob(0) && d.prob(3) == 0.0);
        let s = Sampler::new(1.0, 0, 0.5, 1);
        let d = s.dist(&logits);
        assert!(d.ids.len() < 5 && d.ids[0] == 1);
    }
    #[test]
    fn dist_from_matches_dist_and_breaks_ties_low() {
        let mut s = Sampler::new(0.8, 40, 0.9, 5);
        let logits: Vec<f32> = (0..5000).map(|_| s.uniform() * 40.0 - 20.0).collect();
        let d = s.dist(&logits);
        let mut cands: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        let d2 = Sampler::dist_from(&mut cands, 0.8, 40, 0.9);
        assert_eq!(d.ids, d2.ids);
        assert_eq!(d.probs, d2.probs);
        assert_eq!(Sampler::greedy(&[1.0, 3.0, 3.0]), 1);
        assert_eq!(Sampler::dist_from(&mut vec![(7, 3.0), (2, 3.0), (9, 1.0)], 0.0, 0, 1.0).ids, vec![2]);
        assert_eq!(max_f32(&logits), logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    }
    #[test]
    fn residual_excludes_draft_mass() {
        let p = Dist { ids: vec![1, 2, 3], probs: vec![0.5, 0.3, 0.2] };
        let q = Dist { ids: vec![1, 2, 4], probs: vec![0.7, 0.1, 0.2] };
        let mut s = Sampler::new(1.0, 0, 1.0, 7);
        for _ in 0..50 {
            let t = s.residual_sample(&p, &q);
            assert!(t == 2 || t == 3, "residual mass is on ids 2 and 3 only");
        }
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    /// `cargo test --release -p tr-model --lib timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn time_dist_full_vocab() {
        let n = 248_320;
        let mut s = Sampler::new(0.7, 20, 0.95, 3);
        // peaked like real logits: a few hundred within 20 of the maximum
        let logits: Vec<f32> = (0..n).map(|_| { let u = s.uniform(); if u < 0.002 { 10.0 + s.uniform() * 12.0 } else { s.uniform() * 10.0 - 15.0 } }).collect();
        for (temp, top_k) in [(0.0, 20), (0.7, 20), (0.7, 0), (1.0, 100)] {
            let s2 = Sampler::new(temp, top_k, 0.95, 1);
            let t = std::time::Instant::now();
            let mut tot = 0usize;
            for _ in 0..20 {
                tot += s2.dist(&logits).ids.len();
            }
            println!("temp {temp} top_k {top_k}: {:.3} ms per dist ({tot})", t.elapsed().as_secs_f64() * 1e3 / 20.0);
        }
    }
}
