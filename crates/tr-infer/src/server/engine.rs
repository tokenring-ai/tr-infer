//! The engine thread: owns the model, runs one job at a time, streams text events back.
//! Reasoning is split from content by token id (`</think>`), so the HTTP side only forwards.
//! The state left behind by a job is reused when the next prompt extends the tokens already
//! stepped (multi-turn chats that send `reasoning_content` back hit this).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
#[path = "continuous.rs"]
mod continuous;
use std::sync::Arc;
use std::time::Instant;
use tr_model::exec::Model;
use tr_model::exec_batch::ImageSeg;
use tr_model::image::{mrope_positions, ImagePlace, Patches};
use tr_model::sampler::Sampler;
use tr_model::spec::Speculator;
use tr_model::tokenizer::Tok;
use crate::server::prefix_cache::{PrefixCache, Restored, Saved};
use tr_cache::{Role, Span};

pub const THINK_CLOSE: u32 = 248069; // `</think>` (added token, decodes as text)

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
}

#[derive(Clone)]
pub struct GenParams {
    pub max_tokens: usize,
    /// Sampling for the answer (and for everything when `think` is None).
    pub sampling: Sampling,
    /// Sampling used while inside the `<think>` block; switched at `</think>`.
    pub think: Option<Sampling>,
    pub seed: u64,
    pub stop: Vec<String>,
    /// The prompt ends inside an open `<think>` block: output is reasoning until `</think>`.
    pub thinking_open: bool,
}

/// An image of the prompt: its rows in `ids` (already expanded pad tokens) and the encoder input.
pub struct JobImage {
    pub place: ImagePlace,
    pub patches: Arc<Patches>,
    /// Content hash of the preprocessed image (embedding cache and prefix reuse key).
    pub hash: u64,
}

pub struct Job {
    pub n: usize,
    pub queued: Arc<std::sync::atomic::AtomicUsize>,
    /// Request id, for the engine's own log lines.
    pub id: String,
    pub ids: Vec<u32>,
    /// Role of every prompt token (prefix cache tags and snapshot points).
    pub spans: Vec<Span>,
    pub params: GenParams,
    pub tx: EventTx,
    pub cancel: Arc<AtomicBool>,
    pub images: Vec<JobImage>,
}

/// Encoded images kept for prefix reuse and repeated prompts (multi-turn chats resend them).
const EMBD_CACHE_ENTRIES: usize = 16;

#[derive(Clone)]
pub struct EventTx {
    pub inner: SyncSender<(usize, Event)>,
    pub index: usize,
    pub cancel: Arc<AtomicBool>,
}
impl EventTx {
    fn send(&self, event: Event) -> Result<(), ()> {
        self.inner.try_send((self.index, event)).map_err(|_| {
            self.cancel.store(true, Ordering::Relaxed);
        })
    }
}

pub enum Event {
    Error(String),
    /// Prefill finished: prompt tokens, of which `reused` were already in the state (`restored`
    /// of them from the prefix cache, in `restore_s`); images encoded (cache misses) and the
    /// time they took.
    Prefilled { prompt_tokens: usize, reused: usize, restored: usize, restore_s: f64, prompt_s: f64, images: usize, image_s: f64 },
    Reasoning(String),
    Content(String),
    Done(Finish),
}

pub struct Finish {
    pub reason: &'static str, // "stop" | "length" | "cancelled"
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
    pub gen_s: f64,
    /// Speculative decoding: draft tokens proposed and accepted (0/0 without MTP).
    pub drafted: usize,
    pub accepted: usize,
    /// Mean experts per (token, MoE layer) over the request with the mass routing policy (0 = fixed top-k).
    pub experts: f64,
}

/// Incremental detokeniser for one output section: decodes the tokens since the last
/// UTF-8-complete point, holds back trailing U+FFFD (a token ending mid-sequence).
struct Section {
    ids: Vec<u32>,
    start: usize,
    window_emitted: usize,
    text: String,
    sent: usize,
}

impl Section {
    fn new() -> Section {
        Section { ids: Vec::new(), start: 0, window_emitted: 0, text: String::new(), sent: 0 }
    }
    fn push(&mut self, tok: &Tok, id: u32) {
        self.ids.push(id);
        let s = tok.decode(&self.ids[self.start..]).unwrap_or_default();
        let stable = s.trim_end_matches('\u{FFFD}');
        if stable.len() > self.window_emitted {
            self.text.push_str(&stable[self.window_emitted..]);
            self.window_emitted = stable.len();
        }
        if stable.len() == s.len() {
            self.start = self.ids.len();
            self.window_emitted = 0;
        }
    }
    /// Append whatever is still held back (called when the section ends).
    fn finish(&mut self, tok: &Tok) {
        if self.start < self.ids.len() {
            let s = tok.decode(&self.ids[self.start..]).unwrap_or_default();
            if s.len() > self.window_emitted {
                self.text.push_str(&s[self.window_emitted..]);
            }
            self.start = self.ids.len();
            self.window_emitted = 0;
        }
    }
    /// Text not yet sent, keeping `hold` bytes back (must land on a char boundary).
    fn take(&mut self, hold: usize) -> Option<String> {
        let mut end = self.text.len().saturating_sub(hold);
        while end > self.sent && !self.text.is_char_boundary(end) {
            end -= 1;
        }
        if end > self.sent {
            let s = self.text[self.sent..end].to_string();
            self.sent = end;
            Some(s)
        } else {
            None
        }
    }
    fn trailing_ws(&self) -> usize {
        self.text.len() - self.text.trim_end().len()
    }
}

/// Longest suffix of `text` that is a proper prefix of one of the stop strings.
fn stop_holdback(text: &str, stops: &[String]) -> usize {
    let mut best = 0;
    for st in stops {
        for n in (1..st.len()).rev() {
            if text.len() >= n && text.is_char_boundary(text.len() - n) && st.is_char_boundary(n) && text.ends_with(&st[..n]) {
                best = best.max(n);
                break;
            }
        }
    }
    best
}

#[derive(Default)]
pub struct SchedulerStats {
    pub idle_pages: std::sync::atomic::AtomicUsize,
    pub active: std::sync::atomic::AtomicUsize,
    pub free_pages: std::sync::atomic::AtomicUsize,
    pub iterations: std::sync::atomic::AtomicU64,
    pub decode_rows: std::sync::atomic::AtomicU64,
    pub prefill_rows: std::sync::atomic::AtomicU64,
    pub max_batch_sequences: std::sync::atomic::AtomicUsize,
}

pub struct Engine {
    pub stats: Arc<SchedulerStats>,
    pub model: Model,
    pub tok: Arc<Tok>,
    stepped: Vec<u32>,
    /// Images among the stepped rows: (first row, hash, rows); prefix reuse requires them to match.
    stepped_images: Vec<(usize, u64, usize)>,
    embd_cache: HashMap<u64, Arc<Vec<f32>>>,
    embd_order: VecDeque<u64>,
    last_logits: Option<Vec<f32>>,
    stop_ids: Vec<u32>,
    /// Speculative decoding with the MTP draft head (overlay pack loaded and spec_k > 0).
    spec: Option<Speculator>,
    /// Persistent prefix cache (`--cache-dir`).
    prefix: Option<PrefixCache>,
}

/// Per-job output state: routes tokens to the reasoning or content section, switches the
/// sampler at `</think>`, applies stop strings. Shared by the plain and the speculative loop.
struct Emitter {
    tx: EventTx,
    tok: Arc<Tok>,
    params: GenParams,
    reasoning: Section,
    content: Section,
    in_reasoning: bool,
    skip_ws: bool, // drop the whitespace the model puts right after </think>
    n_gen: usize,
    n_reason: usize,
}

impl Emitter {
    /// Emit one generated token; true when a stop string ended the output.
    fn push(&mut self, next: u32, sampler: &mut Sampler) -> bool {
        self.n_gen += 1;
        let mut stopped = false;
        if self.in_reasoning {
            self.n_reason += 1;
            if next == THINK_CLOSE {
                self.in_reasoning = false;
                self.skip_ws = true;
                // the answer is sampled with its own settings
                sampler.temp = self.params.sampling.temp;
                sampler.top_k = self.params.sampling.top_k;
                sampler.top_p = self.params.sampling.top_p;
                self.reasoning.finish(&self.tok);
                // trailing whitespace before </think> is template formatting, not content
                if let Some(s) = self.reasoning.take(self.reasoning.trailing_ws()) {
                    let _ = self.tx.send(Event::Reasoning(s));
                }
            } else {
                self.reasoning.push(&self.tok, next);
                if let Some(s) = self.reasoning.take(self.reasoning.trailing_ws()) {
                    let _ = self.tx.send(Event::Reasoning(s));
                }
            }
        } else {
            self.content.push(&self.tok, next);
            if self.skip_ws {
                let trimmed = self.content.text.trim_start().len();
                let cut = self.content.text.len() - trimmed;
                if cut > 0 {
                    self.content.text.drain(..cut);
                }
                if !self.content.text.is_empty() {
                    self.skip_ws = false;
                }
            }
            // stop strings: search the unsent tail (plus enough context for a straddling match)
            if !self.params.stop.is_empty() {
                let maxlen = self.params.stop.iter().map(|s| s.len()).max().unwrap_or(0);
                let from = self.content.sent.saturating_sub(maxlen);
                let from = (from..=self.content.sent).find(|&b| self.content.text.is_char_boundary(b)).unwrap_or(self.content.sent);
                let mut hit = None;
                for st in &self.params.stop {
                    if let Some(p) = self.content.text[from..].find(st.as_str()) {
                        let p = from + p;
                        hit = Some(hit.map_or(p, |h: usize| h.min(p)));
                    }
                }
                if let Some(p) = hit {
                    self.content.text.truncate(p);
                    stopped = true;
                }
            }
            let hold = if stopped { 0 } else { stop_holdback(&self.content.text, &self.params.stop) };
            if let Some(s) = self.content.take(hold) {
                let _ = self.tx.send(Event::Content(s));
            }
        }
        stopped
    }
    fn flush(&mut self) {
        if self.in_reasoning {
            self.reasoning.finish(&self.tok);
            if let Some(s) = self.reasoning.take(0) {
                let _ = self.tx.send(Event::Reasoning(s));
            }
        } else {
            self.content.finish(&self.tok);
            if let Some(s) = self.content.take(0) {
                let _ = self.tx.send(Event::Content(s));
            }
        }
    }
}

impl Engine {
    pub fn new(model: Model, tok: Arc<Tok>, spec_k: usize, prefix: Option<PrefixCache>) -> Engine {
        let mut stop_ids = vec![model.cfg.eos_id];
        if model.cfg.bos_id != model.cfg.eos_id {
            stop_ids.push(model.cfg.bos_id); // <|endoftext|>
        }
        let spec = if model.has_mtp() && spec_k > 0 { Some(Speculator::new(spec_k)) } else { None };
        Engine { stats: Arc::new(SchedulerStats::default()), model, tok, stepped: Vec::new(), stepped_images: Vec::new(), embd_cache: HashMap::new(), embd_order: VecDeque::new(), last_logits: None, stop_ids, spec, prefix }
    }

    pub fn run(&mut self, rx: Receiver<Job>) {
        for job in rx {
            job.queued.fetch_sub(1, Ordering::Relaxed);
            self.run_job(job);
        }
    }

    /// Embeddings of an image: cached by content hash, else encoded now.
    fn image_embd(&mut self, im: &JobImage) -> (Arc<Vec<f32>>, bool) {
        if let Some(e) = self.embd_cache.get(&im.hash) {
            return (e.clone(), false);
        }
        let e = Arc::new(self.model.encode_image(&im.patches));
        self.embd_cache.insert(im.hash, e.clone());
        self.embd_order.push_back(im.hash);
        while self.embd_order.len() > EMBD_CACHE_ENTRIES {
            if let Some(old) = self.embd_order.pop_front() {
                self.embd_cache.remove(&old);
            }
        }
        (e, true)
    }

    fn run_job(&mut self, job: Job) {
        let moe0 = self.model.moe_stats.snapshot();
        let moe_mean = |m: &tr_model::exec::Model| if m.moe_policy().is_some() { m.moe_stats.mean_since(moe0) } else { 0.0 };
        let Job { id, ids, spans, params, tx, cancel, images, .. } = job;
        let tok = self.tok.clone();
        let t0 = Instant::now();
        // images first (they go through the pool too); cache hits are free
        let mut embds: Vec<Arc<Vec<f32>>> = Vec::with_capacity(images.len());
        let mut n_encoded = 0;
        for im in &images {
            let (e, fresh) = self.image_embd(im);
            n_encoded += usize::from(fresh);
            embds.push(e);
        }
        let image_s = t0.elapsed().as_secs_f64();
        let job_images: Vec<(usize, u64, usize)> = images.iter().map(|i| (i.place.row, i.hash, i.place.nx * i.place.ny)).collect();
        // prefix reuse: the state already covers `stepped`; only the tail must be prefilled.
        // Image rows carry the pad id, so the images inside the reused prefix must match too.
        let same_images = |l: usize| -> bool {
            let a: Vec<_> = self.stepped_images.iter().filter(|x| x.0 < l).collect();
            let b: Vec<_> = job_images.iter().filter(|x| x.0 < l).collect();
            a == b
        };
        let mut keys = self.prefix.as_ref().map(|pc| pc.keys(&ids, &job_images));
        let mut restored: Option<Restored> = None;
        let reuse = if !self.stepped.is_empty() && ids.len() >= self.stepped.len() && ids[..self.stepped.len()] == self.stepped[..] && same_images(self.stepped.len()) && (ids.len() > self.stepped.len() || self.last_logits.is_some()) {
            self.stepped.len()
        } else {
            self.model.reset();
            self.stepped.clear();
            self.stepped_images.clear();
            self.last_logits = None;
            // the prefix cache: the deepest stored restore point of this prompt
            match (self.prefix.as_ref(), keys.as_ref()) {
                (Some(pc), Some(k)) => match pc.restore(&mut self.model, &ids, k) {
                    Some(mut r) => {
                        self.stepped.extend_from_slice(&ids[..r.pos]);
                        self.last_logits = r.last_logits.take();
                        let pos = r.pos;
                        restored = Some(r);
                        pos
                    }
                    None => 0,
                },
                _ => 0,
            }
        };
        let restored_n = restored.as_ref().map(|r| r.pos).unwrap_or(0);
        let restore_s = restored.as_ref().map(|r| r.seconds).unwrap_or(0.0);
        if let Some(r) = &restored {
            eprintln!("{id}: prefix cache restored {} of {} prompt tokens ({} rows chunks, {:.1} MiB, {:.1} ms)", r.pos, ids.len(), r.rows, r.bytes as f64 / (1u64 << 20) as f64, r.seconds * 1e3);
        }
        // snapshot points inside the new part of the prompt (prefill batches end there)
        let snap_pts: Vec<(usize, Role)> = self.prefix.as_ref().map(|pc| pc.snapshot_points(&spans, reuse, ids.len())).unwrap_or_default();
        let places: Vec<ImagePlace> = images.iter().map(|i| i.place).collect();
        let (pos3, after) = mrope_positions(ids.len(), &places);
        let hh = self.model.cfg.hidden;
        let bm = self.model.batch_max();
        let mut logits = self.last_logits.take().unwrap_or_default();
        let mut a = reuse;
        while a < ids.len() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let b = (a + bm).min(ids.len());
            let b = snap_pts.iter().find(|&&(e, _)| e > a).map_or(b, |&(e, _)| b.min(e));
            // image rows inside [a, b)
            let mut segs: Vec<ImageSeg> = Vec::new();
            for (im, e) in images.iter().zip(&embds) {
                let (s0, s1) = (im.place.row.max(a), (im.place.row + im.place.nx * im.place.ny).min(b));
                if s0 < s1 {
                    segs.push(ImageSeg { row: s0 - a, n: s1 - s0, embd: &e[(s0 - im.place.row) * hh..(s1 - im.place.row) * hh] });
                }
            }
            // text-only prompts keep the plain path (a 1-row chunk is a decode step)
            logits = if images.is_empty() { self.model.step_batch(&ids[a..b]) } else { self.model.step_batch_with(&ids[a..b], Some(&pos3[a..b]), Some(after[b - 1]), &segs) };
            self.stepped.extend_from_slice(&ids[a..b]);
            a = b;
            if let Some(&(_, role)) = snap_pts.iter().find(|&&(e, _)| e == b) {
                if let (Some(pc), Some(k)) = (self.prefix.as_ref(), keys.as_ref()) {
                    if pc.snapshot_now(&mut self.model, k[b], b, role, &logits) == Some(false) {
                        eprintln!("{id}: prefix cache: no staging buffer free, snapshot at {b} skipped");
                    }
                }
            }
        }
        self.stepped_images = job_images.into_iter().filter(|x| x.0 < self.stepped.len()).collect();
        let prompt_s = t0.elapsed().as_secs_f64();
        if cancel.load(Ordering::Relaxed) || logits.is_empty() {
            self.last_logits = if logits.is_empty() { None } else { Some(logits) };
            let _ = tx.send(Event::Done(Finish { reason: "cancelled", completion_tokens: 0, reasoning_tokens: 0, gen_s: 0.0, drafted: 0, accepted: 0, experts: moe_mean(&self.model) }));
            return;
        }
        let _ = tx.send(Event::Prefilled { prompt_tokens: ids.len(), reused: reuse, restored: restored_n, restore_s, prompt_s, images: n_encoded, image_s });

        let t1 = Instant::now();
        let first = if params.thinking_open { params.think.unwrap_or(params.sampling) } else { params.sampling };
        let mut sampler = Sampler::new(first.temp, first.top_k, first.top_p, params.seed);
        let mut em = Emitter { tx: tx.clone(), tok: tok.clone(), params: params.clone(), reasoning: Section::new(), content: Section::new(), in_reasoning: params.thinking_open, skip_ws: false, n_gen: 0, n_reason: 0 };
        let mut reason = "length";
        let stats0 = self.spec.as_ref().map(|s| s.stats).unwrap_or_default();
        // the pending token: sampled from the state's last logits, not yet stepped
        let mut next = sampler.sample(&logits);
        loop {
            if cancel.load(Ordering::Relaxed) {
                reason = "cancelled";
                break;
            }
            if self.stop_ids.contains(&next) {
                reason = "stop";
                break;
            }
            if em.push(next, &mut sampler) {
                reason = "stop";
                break;
            }
            if em.n_gen >= params.max_tokens {
                break; // "length": the last token is not stepped (the next request will, if it continues)
            }
            let budget = params.max_tokens - em.n_gen;
            match self.spec.as_mut() {
                Some(sp) => {
                    // draft/verify round: commits `next` and the accepted drafts, yields the new pending token
                    let mut stop_at = self.stop_ids.clone();
                    stop_at.push(THINK_CLOSE);
                    let r = sp.round(&mut self.model, next, budget, &mut sampler, &stop_at);
                    self.stepped.push(next);
                    let mut stopped = false;
                    for &d in &r.accepted {
                        self.stepped.push(d);
                        if self.stop_ids.contains(&d) {
                            stopped = true;
                            reason = "stop";
                            break;
                        }
                        if em.push(d, &mut sampler) {
                            stopped = true;
                            reason = "stop";
                            break;
                        }
                    }
                    debug_assert_eq!(self.stepped.len(), self.model.n_past);
                    logits = r.next_logits;
                    next = r.next;
                    if stopped {
                        break;
                    }
                }
                None => {
                    logits = self.model.step(next, None);
                    self.stepped.push(next);
                    next = sampler.sample(&logits);
                }
            }
        }
        em.flush();
        let (n_gen, n_reason) = (em.n_gen, em.n_reason);
        let stats = self.spec.as_ref().map(|s| s.stats).unwrap_or_default();
        self.last_logits = Some(logits);
        let _ = tx.send(Event::Done(Finish { reason, completion_tokens: n_gen, reasoning_tokens: n_reason, gen_s: t1.elapsed().as_secs_f64(), drafted: stats.drafted - stats0.drafted, accepted: stats.accepted - stats0.accepted, experts: moe_mean(&self.model) }));
        // the prefix cache: store what this request added (the client already has its answer)
        if let (Some(pc), Some(k)) = (self.prefix.as_ref(), keys.as_mut()) {
            let n = self.stepped.len();
            debug_assert_eq!(n, self.model.n_past);
            pc.extend_keys(k, &self.stepped[ids.len()..]);
            let mut all: Vec<Span> = spans.iter().filter(|s| s.start < n).map(|s| Span::new(s.start, s.end.min(n), s.role)).collect();
            let p = ids.len();
            let r_end = if params.thinking_open { (p + n_reason).min(n) } else { p };
            if r_end > p {
                all.push(Span::new(p, r_end, Role::Reasoning));
            }
            if n > r_end {
                all.push(Span::new(r_end, n, Role::Assistant));
            }
            let logits = self.last_logits.clone().unwrap_or_default();
            let s: Saved = pc.save(&mut self.model, &self.stepped, k, &all, &logits);
            if s.rows + s.snapshots > 0 {
                eprintln!("{id}: prefix cache saved {} rows chunks + {} snapshots ({:.1} MiB staged in {:.1} ms)", s.rows, s.snapshots, s.bytes as f64 / (1u64 << 20) as f64, s.seconds * 1e3);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn holdback() {
        let stops = vec!["</s>".to_string(), "END".to_string()];
        assert_eq!(stop_holdback("hello <", &stops), 1);
        assert_eq!(stop_holdback("hello </s", &stops), 3);
        assert_eq!(stop_holdback("hello EN", &stops), 2);
        assert_eq!(stop_holdback("hello", &stops), 0);
        assert_eq!(stop_holdback("hello", &[]), 0);
    }
}

/// Conservative shared-prompt page reservation, including a private append tail per choice.
pub fn continuous_pages(prompt: usize, output: usize, choices: usize) -> Option<usize> {
    let page = tr_model::sequence::PAGE_TOKENS;
    let tail = (prompt % page).checked_add(output)?.checked_add(page - 1)? / page;
    (prompt / page).checked_add(tail.checked_mul(choices)?)
}
