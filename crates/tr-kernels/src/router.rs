//! MoE routing: softmax over expert logits, top-k, renormalise (llama.cpp build_moe_ffn, norm_w),
//! and the optional cumulative-mass policy (`route_mass`): experts in router order until their
//! softmax mass reaches a target, with a floor and a cap on the count.

/// Largest expert count any routing policy may select; the engine's workspaces are sized for it.
pub const MOE_K_MAX: usize = 32;

/// What the target mass is measured against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MassBasis {
    /// The softmax over all experts (absolute router probability).
    All,
    /// The softmax mass of the `max` best experts (the renormalised weights of a top-`max` routing).
    #[default]
    Top,
}

/// Cumulative router mass policy: take experts in descending softmax order until the selected
/// probability mass reaches `mass` (of all experts or of the `max` best, see `MassBasis`), but at
/// least `min` and at most `max`. `mass >= 1` degenerates to a fixed top-`max`; the weights of the
/// chosen experts are renormalised to sum to one exactly as the model's top-k routing does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoePolicy {
    pub min: usize,
    pub max: usize,
    pub mass: f32,
    pub basis: MassBasis,
}

impl MoePolicy {
    /// Check against the model's expert count.
    pub fn validate(&self, n_expert: usize) -> Result<(), String> {
        if self.min < 1 || self.min > self.max {
            return Err(format!("moe policy: need 1 <= min ({}) <= max ({})", self.min, self.max));
        }
        if self.max > MOE_K_MAX || self.max > n_expert {
            return Err(format!("moe policy: max ({}) above the limit ({})", self.max, MOE_K_MAX.min(n_expert)));
        }
        if !(self.mass > 0.0) || self.mass.is_nan() {
            return Err(format!("moe policy: target mass ({}) must be positive", self.mass));
        }
        Ok(())
    }
}

/// Command-line shape of the policy: `mass` enables it; `min`/`max` default to 1 and the model's
/// `expert_used_count`; `basis` defaults to the top-`max` candidates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoeArgs {
    pub mass: Option<f32>,
    pub min: Option<usize>,
    pub max: Option<usize>,
    pub basis: MassBasis,
}

impl MoeArgs {
    pub fn resolve(&self, n_expert_used: usize, n_expert: usize) -> Result<Option<MoePolicy>, String> {
        let Some(mass) = self.mass else {
            if self.min.is_some() || self.max.is_some() {
                return Err("--moe-min / --moe-max need --moe-mass".into());
            }
            return Ok(None);
        };
        let p = MoePolicy { min: self.min.unwrap_or(1), max: self.max.unwrap_or(n_expert_used), mass, basis: self.basis };
        p.validate(n_expert)?;
        Ok(Some(p))
    }
}

/// Insertion-sorted top-k on the logits (descending; ties: lowest index first). A later index only
/// displaces strictly smaller values. Returns the global maximum.
#[inline]
fn topk_sorted(logits: &[f32], k: usize, top_v: &mut [f32; MOE_K_MAX], top_i: &mut [u32; MOE_K_MAX]) {
    let n = logits.len();
    assert!(k >= 1 && k <= MOE_K_MAX && k <= n);
    for (i, &v) in logits.iter().enumerate() {
        if !(v > top_v[k - 1]) {
            continue;
        }
        let mut p = k - 1;
        while p > 0 && top_v[p - 1] < v {
            top_v[p] = top_v[p - 1];
            top_i[p] = top_i[p - 1];
            p -= 1;
        }
        top_v[p] = v;
        top_i[p] = i as u32;
    }
}

/// Softmax weights of the first `k` sorted logits, renormalised over those `k` (the full-softmax
/// denominator cancels, so only the winners are exponentiated).
#[inline]
fn renorm(top_v: &[f32], k: usize, out_w: &mut [f32]) {
    let mx = top_v[0];
    let mut sum = 0f32;
    for j in 0..k {
        out_w[j] = (top_v[j] - mx).exp();
        sum += out_w[j];
    }
    for w in out_w[..k].iter_mut() {
        *w /= sum;
    }
}

pub fn route_topk(logits: &[f32], k: usize, out_ids: &mut [u32], out_w: &mut [f32]) {
    // softmax is monotone, so select on the logits
    let mut top_v = [f32::NEG_INFINITY; MOE_K_MAX];
    let mut top_i = [u32::MAX; MOE_K_MAX];
    topk_sorted(logits, k, &mut top_v, &mut top_i);
    out_ids[..k].copy_from_slice(&top_i[..k]);
    renorm(&top_v, k, out_w);
}

/// Cumulative-mass routing (see [`MoePolicy`]). Writes the chosen experts (descending score) and
/// their renormalised weights into the first `count` slots of `out_ids` / `out_w` and returns `count`.
/// Deterministic across callers (no state); `MassBasis::All` costs one exp per expert.
pub fn route_mass(logits: &[f32], p: &MoePolicy, out_ids: &mut [u32], out_w: &mut [f32]) -> usize {
    let mut top_v = [f32::NEG_INFINITY; MOE_K_MAX];
    let mut top_i = [u32::MAX; MOE_K_MAX];
    topk_sorted(logits, p.max, &mut top_v, &mut top_i);
    let mx = top_v[0];
    let mut denom = 0f32;
    match p.basis {
        MassBasis::All => {
            for &v in logits {
                denom += (v - mx).exp();
            }
        }
        MassBasis::Top => {
            for &v in &top_v[..p.max] {
                denom += (v - mx).exp();
            }
        }
    }
    let mut count = 0usize;
    let mut mass = 0f32;
    while count < p.max {
        mass += (top_v[count] - mx).exp() / denom;
        count += 1;
        if count >= p.min && mass >= p.mass {
            break;
        }
    }
    out_ids[..count].copy_from_slice(&top_i[..count]);
    renorm(&top_v, count, out_w);
    count
}

/// Cumulative softmax mass of the top-`MOE_K_MAX` experts, `cum[j]` = mass of the best j+1
/// (profiling aid for choosing a target mass).
pub fn mass_profile(logits: &[f32], cum: &mut [f32; MOE_K_MAX]) {
    let mut top_v = [f32::NEG_INFINITY; MOE_K_MAX];
    let mut top_i = [u32::MAX; MOE_K_MAX];
    let k = MOE_K_MAX.min(logits.len());
    topk_sorted(logits, k, &mut top_v, &mut top_i);
    let mx = top_v[0];
    let denom: f32 = logits.iter().map(|&v| (v - mx).exp()).sum();
    let mut mass = 0f32;
    for j in 0..MOE_K_MAX {
        if j < k {
            mass += (top_v[j] - mx).exp() / denom;
        }
        cum[j] = mass;
    }
}

#[cfg(test)]
mod mass_tests {
    use super::*;

    fn probs(logits: &[f32]) -> Vec<f32> {
        let mut p = logits.to_vec();
        crate::elem::softmax_inplace(&mut p);
        p
    }

    #[test]
    fn stops_at_the_target_mass_within_bounds() {
        // peaked: 0.61 / 0.19 / 0.09 / 0.05 / 0.025 / 0.015 / ... as in the design note
        let mut logits = vec![-12.0f32; 512];
        for (i, p) in [0.61f32, 0.19, 0.09, 0.05, 0.025, 0.015].iter().enumerate() {
            logits[i * 7] = p.ln();
        }
        let pr = probs(&logits);
        let mut ids = vec![0u32; MOE_K_MAX];
        let mut w = vec![0f32; MOE_K_MAX];
        let pol = MoePolicy { min: 1, max: 10, mass: 0.95, basis: MassBasis::All };
        let n = route_mass(&logits, &pol, &mut ids, &mut w);
        // the 506 background experts hold ~0.3 % of the mass: four experts reach 0.95
        let cum: Vec<f32> = (1..=6).map(|k| pr[..].iter().enumerate().filter(|(i, _)| i % 7 == 0 && i / 7 < k).map(|(_, &p)| p).sum()).collect();
        let expect = cum.iter().position(|&m| m >= 0.95).unwrap() + 1;
        assert_eq!(n, expect);
        assert_eq!(&ids[..n], &(0..n as u32).map(|i| i * 7).collect::<Vec<_>>()[..]);
        assert!((w[..n].iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // same weights as top-n renormalisation
        let (mut tid, mut tw) = (vec![0u32; n], vec![0f32; n]);
        route_topk(&logits, n, &mut tid, &mut tw);
        assert_eq!(&ids[..n], &tid[..]);
        for j in 0..n {
            assert!((w[j] - tw[j]).abs() < 1e-6);
        }
        // floor and cap
        let all = |min, max, mass| MoePolicy { min, max, mass, basis: MassBasis::All };
        assert_eq!(route_mass(&logits, &all(8, 10, 0.5), &mut ids, &mut w), 8);
        assert_eq!(route_mass(&logits, &all(1, 3, 0.999), &mut ids, &mut w), 3);
        assert_eq!(route_mass(&logits, &all(1, 32, 1.0), &mut ids, &mut w), 32);
        // flat: every expert ~ 1/512, so 0.95 needs the cap
        let flat = vec![0.5f32; 512];
        assert_eq!(route_mass(&flat, &all(1, 10, 0.95), &mut ids, &mut w), 10);
        assert_eq!(&ids[..10], &(0..10u32).collect::<Vec<_>>()[..]); // ties: lowest index first
        // relative to the top-max candidates: the flat router needs 10 of 10 for 0.95 too, but the
        // peaked one stops where the renormalised top-10 weights reach the target
        let top = |min, max, mass| MoePolicy { min, max, mass, basis: MassBasis::Top };
        assert_eq!(route_mass(&flat, &top(1, 10, 0.95), &mut ids, &mut w), 10);
        assert_eq!(route_mass(&flat, &top(1, 10, 0.05), &mut ids, &mut w), 1);
        let (mut tid, mut tw) = (vec![0u32; 10], vec![0f32; 10]);
        route_topk(&logits, 10, &mut tid, &mut tw);
        let mut acc = 0f32;
        let expect = tw.iter().position(|&x| { acc += x; acc >= 0.9 }).unwrap() + 1;
        assert_eq!(route_mass(&logits, &top(1, 10, 0.9), &mut ids, &mut w), expect);
        assert_eq!(&ids[..expect], &tid[..expect]);
        // min = max = k reproduces route_topk exactly (either basis)
        let logits: Vec<f32> = (0..512).map(|i| ((i * 37) % 101) as f32 * 0.1).collect();
        let (mut tid, mut tw) = (vec![0u32; 10], vec![0f32; 10]);
        route_topk(&logits, 10, &mut tid, &mut tw);
        for pol in [all(10, 10, 0.5), top(10, 10, 0.5)] {
            assert_eq!(route_mass(&logits, &pol, &mut ids, &mut w), 10);
            assert_eq!(&ids[..10], &tid[..]);
            assert_eq!(&w[..10], &tw[..]);
        }
        assert!(all(0, 10, 0.9).validate(512).is_err());
        assert!(all(1, 33, 0.9).validate(512).is_err());
        assert!(all(1, 20, 0.9).validate(16).is_err());
        assert!(all(1, 10, 0.0).validate(512).is_err());
        assert!(all(1, 10, 0.9).validate(512).is_ok());
        let mut cum = [0f32; MOE_K_MAX];
        mass_profile(&logits, &mut cum);
        let pr = probs(&logits);
        let mut sorted = pr.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut acc = 0f32;
        for j in 0..MOE_K_MAX {
            acc += sorted[j];
            assert!((cum[j] - acc).abs() < 1e-5, "k={}: {} vs {acc}", j + 1, cum[j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn picks_top_and_renormalises() {
        let logits: Vec<f32> = (0..512).map(|i| ((i * 37) % 101) as f32 * 0.1).collect();
        let mut ids = vec![0u32; 10];
        let mut w = vec![0f32; 10];
        route_topk(&logits, 10, &mut ids, &mut w);
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for j in 0..10 {
            assert_eq!(logits[ids[j] as usize], sorted[j]);
        }
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(w[0] >= w[9]);
        // same weights as softmax-then-renormalise (the full denominator cancels), to rounding
        let mut probs = logits.clone();
        crate::elem::softmax_inplace(&mut probs);
        let s: f32 = ids.iter().map(|&i| probs[i as usize]).sum();
        for j in 0..10 {
            let old = probs[ids[j] as usize] / s;
            assert!((w[j] - old).abs() < 1e-6, "{j}: {} vs {old}", w[j]);
        }
    }
}

#[cfg(test)]
mod tie_tests {
    use super::*;
    fn reference(logits: &[f32], k: usize) -> Vec<u32> {
        let mut taken = vec![false; logits.len()];
        let mut ids = Vec::new();
        for _ in 0..k {
            let mut best = u32::MAX;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if !taken[i] && v > bv {
                    bv = v;
                    best = i as u32;
                }
            }
            taken[best as usize] = true;
            ids.push(best);
        }
        ids
    }
    #[test]
    fn matches_sequential_selection_with_ties() {
        let mut seed = 3u32;
        for _ in 0..200 {
            let logits: Vec<f32> = (0..512)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 20) % 40) as f32 * 0.25 // many exact ties
                })
                .collect();
            let mut ids = vec![0u32; 10];
            let mut w = vec![0f32; 10];
            route_topk(&logits, 10, &mut ids, &mut w);
            assert_eq!(ids, reference(&logits, 10));
        }
    }
}
