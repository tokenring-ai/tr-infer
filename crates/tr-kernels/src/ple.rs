//! PLE n-gram hash (llama.cpp qwen4exp.cpp llm_graph_input_ple::set_input).
//! For token i with predecessors (oldest last): ctx[0] = tok_i, ctx[s] = tok_{i-s} (EOS-cut).
//! For n in 2..=ngram: mixed = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ... ^ ctx[n-1]*m[n-1] (u64 wrapping);
//! heads h = (n-2)*per_gram + g get row = mixed % vocab[h] + offset[h].

pub struct PleHash {
    pub ngram: usize,
    pub per_gram: usize,
    pub eos: u32,
    pub mult: Vec<u64>,
    pub offsets: Vec<u64>,
    pub vocab: Vec<u64>,
}

impl PleHash {
    pub fn n_heads(&self) -> usize {
        (self.ngram - 1) * self.per_gram
    }
    /// `prev` holds up to ngram-1 predecessor tokens, most recent LAST (prev[len-1] = t_{i-1});
    /// missing predecessors (before sequence start) are absent. Writes n_heads row indices.
    pub fn rows(&self, tok: u32, prev: &[u32], out: &mut [u32]) {
        let n_prev = self.ngram - 1;
        let mut ctx = vec![0u64; self.ngram];
        ctx[0] = tok as u64;
        let mut cut = false;
        for s in 1..self.ngram {
            // predecessor s positions back
            let t: i64 = if cut || s > prev.len() { -1 } else { prev[prev.len() - s] as i64 };
            cut = cut || t < 0 || t as u32 == self.eos;
            ctx[s] = if cut { self.eos as u64 } else { t as u64 };
        }
        let _ = n_prev;
        for n in 2..=self.ngram {
            let mut mixed = ctx[0].wrapping_mul(self.mult[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(self.mult[j]);
            }
            let base = (n - 2) * self.per_gram;
            for g in 0..self.per_gram {
                let h = base + g;
                out[h] = (mixed % self.vocab[h] + self.offsets[h]) as u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eos_cuts_window_and_hash_is_deterministic() {
        let h = PleHash { ngram: 3, per_gram: 2, eos: 9, mult: vec![23703573157769, 20109073645365, 8052911324071], offsets: vec![0, 100, 200, 300], vocab: vec![97, 89, 83, 79] };
        let mut a = [0u32; 4];
        let mut b = [0u32; 4];
        h.rows(5, &[3, 4], &mut a);
        h.rows(5, &[3, 4], &mut b);
        assert_eq!(a, b);
        // with EOS at i-1 the whole history is EOS, same as no history
        h.rows(5, &[3, 9], &mut a);
        h.rows(5, &[], &mut b);
        assert_eq!(a, b);
        for (i, r) in a.iter().enumerate() {
            assert!(*r as u64 >= h.offsets[i] && (*r as u64) < h.offsets[i] + h.vocab[i]);
        }
    }
}
