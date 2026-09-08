//! Speculative decoding round (draft with the MTP head, verify with the main model, accept with
//! standard speculative sampling — lossless: the emitted tokens follow the main model's
//! distribution exactly; at temperature 0 it reduces to greedy with exact-match acceptance).
//!
//! Sequence state at round start: positions < P committed, `t0` sampled for position P (not
//! stepped), the main model's residual for position P-1 available to the draft head (the carry:
//! written by the last decode step / prefill chunk, or row j of the last verify). One round, two
//! pool runs:
//!   1. draft: the draft head's row for (carry, t0) gives d1, chained rows give d2..dk, each
//!      token sampled inside the pool (`Model::mtp_draft`);
//!   2. verify [t0, d1..dk] in one batched step (per-row logits + state checkpoints); the same
//!      run keeps the main residual rows and refreshes the draft head's K/V for positions
//!      P+1..P+k from them (`worker_batch`, Verify);
//!   3. accept the longest prefix (with the rejection-sampling correction), giving j accepted
//!      drafts and the next pending token t0';
//!   4. commit checkpoint j; residual row j is the next round's carry.
use crate::exec::Model;
use crate::exec_mtp::DraftPlan;
use crate::sampler::Sampler;

#[derive(Clone, Copy, Debug, Default)]
pub struct SpecStats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
}

pub struct Round {
    /// Accepted draft tokens (positions P+1..=P+j), to emit after `t0`.
    pub accepted: Vec<u32>,
    /// The next pending token (position P+j+1), sampled from the main model.
    pub next: u32,
    /// Main-model logits `next` was sampled from (the state's last logits).
    pub next_logits: Vec<f32>,
}

pub struct Speculator {
    pub k: usize,
    pub stats: SpecStats,
}

impl Speculator {
    pub fn new(k: usize) -> Speculator {
        Speculator { k, stats: SpecStats::default() }
    }

    /// One round for pending token `t0` at position `model.n_past`. `k` drafts at most (also
    /// capped by the round's `budget`); drafting stops early at a token in `stop_at` (stop ids,
    /// `</think>`) so the sampling phase never changes inside a draft. The sampler's current
    /// settings are the target settings for every position of the round.
    pub fn round(&mut self, model: &mut Model, t0: u32, budget: usize, sampler: &mut Sampler, stop_at: &[u32]) -> Round {
        let p0 = model.n_past;
        let k = self.k.min(budget).min(model.n_ckpt().saturating_sub(1)).min(model.batch_max().saturating_sub(1)).max(1);
        // 1. draft: one pool run for the whole chain (the row for t0 first, sampled in the pool)
        let plan = DraftPlan { temp: sampler.temp, top_k: sampler.top_k, top_p: sampler.top_p, uniforms: (0..k).map(|_| sampler.uniform()).collect(), stop_at: stop_at.to_vec() };
        let dq = model.mtp_draft(t0, p0, &plan);
        let kd = dq.len();
        // 2. verify (the run also refreshes the draft head's K/V for the draft rows)
        let mut toks = Vec::with_capacity(kd + 1);
        toks.push(t0);
        toks.extend(dq.iter().map(|x| x.0));
        let rows = model.verify(&toks);
        // 3. accept
        let mut j = 0usize;
        let mut next = None;
        let debug = std::env::var("TR_LOGIT_DEBUG").is_ok();
        for (i, (d, q)) in dq.iter().enumerate() {
            let p = sampler.dist(&rows[i]);
            if debug {
                // the same line `generate` prints for the pending token, for every accepted draft row
                let mut top: Vec<(u32, f32)> = rows[i].iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
                top.select_nth_unstable_by(2, |a, b| b.1.partial_cmp(&a.1).unwrap());
                top.truncate(3);
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let (pd, qd) = (p.prob(*d), q.prob(*d));
                let accept = pd > 0.0 && pd >= qd; // temperature 0: exact match
                if accept {
                    eprintln!("top3: {} (draft)", top.iter().map(|(i, l)| format!("{i}:{l:.3}")).collect::<Vec<_>>().join(" "));
                }
            }
            let (pd, qd) = (p.prob(*d), q.prob(*d));
            let accept = pd > 0.0 && (pd >= qd || sampler.uniform() < pd / qd);
            if accept {
                j += 1;
            } else {
                next = Some((sampler.residual_sample(&p, q), i));
                break;
            }
        }
        let (next, row) = match next {
            Some((t, i)) => (t, i),
            None => (sampler.sample(&rows[kd]), kd),
        };
        let next_logits = rows[row].clone();
        // 4. commit: checkpoint j becomes the live state; the next chain starts from residual row j
        model.commit(j);
        self.stats.rounds += 1;
        self.stats.drafted += kd;
        self.stats.accepted += j;
        Round { accepted: toks[1..=j].to_vec(), next, next_logits }
    }
}
