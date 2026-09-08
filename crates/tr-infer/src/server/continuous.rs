//! One scheduler owns all model mutations. Network and disk I/O never run in the pool.
use super::*;
use crate::server::prefix_cache::RestoreObject;
use std::sync::mpsc::TryRecvError;
use tr_model::sequence::{BatchInput, SequenceId};

struct Idle {
    seq: SequenceId,
    ids: Vec<u32>,
    logits: Vec<f32>,
}

struct Choice {
    seq: SequenceId,
    emitter: Emitter,
    sampler: Sampler,
    pending: Option<u32>,
    stepped: Vec<u32>,
    logits: Vec<f32>,
    start: Instant,
    finished: bool,
    expert_rows: u64,
    experts: u64,
}
struct Group {
    expert_rows: u64,
    experts: u64,
    job: Job,
    root: Option<SequenceId>,
    choices: Vec<Choice>,
    credits: usize,
    start: Instant,
    restore: Option<Receiver<anyhow::Result<RestoreObject>>>,
    imported: usize,
    restored: usize,
    reused: usize,
    restore_s: f64,
    logits: Vec<f32>,
}

impl Engine {
    pub fn run_continuous(&mut self, rx: Receiver<Job>, prefill_chunk: usize) {
        let mut waiting = VecDeque::<Job>::new();
        let mut groups = Vec::<Group>::new();
        let mut closed = false;
        let mut cursor = 0usize;
        let mut idle: Option<Idle> = None;
        loop {
            self.stats
                .active
                .store(self.model.sequence_capacity() - self.model.free_sequence_count() - usize::from(idle.is_some()), Ordering::Relaxed);
            self.stats
                .idle_pages
                .store(idle.as_ref().map_or(0, |i| i.ids.len().div_ceil(tr_model::sequence::PAGE_TOKENS)), Ordering::Relaxed);
            self.stats.free_pages.store(self.model.free_page_count(), Ordering::Relaxed);
            while !closed {
                match rx.try_recv() {
                    Ok(job) => waiting.push_back(job),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
            // Drop cancellations before admission so disconnected queued requests hold no credits.
            let mut i = 0;
            while i < waiting.len() {
                if waiting[i].cancel.load(Ordering::Relaxed) {
                    let job = waiting.remove(i).unwrap();
                    job.queued.fetch_sub(1, Ordering::Relaxed);
                    send_cancelled(&job);
                } else {
                    i += 1;
                }
            }
            for g in &mut groups {
                if g.job.cancel.load(Ordering::Relaxed) {
                    g.restore = None;
                    if let Some(root) = g.root.take() {
                        let _ = self.model.release_sequence(root);
                        send_cancelled(&g.job);
                    }
                    for c in &mut g.choices {
                        if !c.finished {
                            finish_choice(&mut self.model, self.prefix.as_ref(), &g.job, c, "cancelled", &mut idle);
                        }
                    }
                }
            }
            groups.retain(|g| g.root.is_some() || g.choices.iter().any(|c| !c.finished));
            // Completed choices release their reservations even while sibling choices run.
            // Before forking, reserve the whole requested group; afterward reserve only live choices.
            for g in &mut groups {
                if g.root.is_none() {
                    let live = g.choices.iter().filter(|c| !c.finished).count();
                    g.credits = continuous_pages(g.job.ids.len(), g.job.params.max_tokens, live).unwrap();
                }
            }
            let mut used_pages: usize = groups.iter().map(|g| g.credits).sum();
            let mut used_sequences: usize =
                groups.iter().map(|g| if g.root.is_some() { g.job.n } else { g.choices.iter().filter(|c| !c.finished).count() }).sum();
            // FIFO admission: a waiting large request cannot starve behind smaller arrivals.
            while let Some(job) = waiting.front() {
                let credits = continuous_pages(job.ids.len(), job.params.max_tokens, job.n).unwrap_or(usize::MAX);
                if credits > self.model.page_capacity() || job.n > self.model.sequence_capacity() {
                    let job = waiting.pop_front().unwrap();
                    job.queued.fetch_sub(1, Ordering::Relaxed);
                    let _ = job.tx.send(Event::Error("request exceeds configured sequence or KV capacity".into()));
                    continue;
                }
                if credits > self.model.page_capacity() - used_pages || job.n > self.model.sequence_capacity() - used_sequences {
                    break;
                }
                let job = waiting.pop_front().unwrap();
                job.queued.fetch_sub(1, Ordering::Relaxed);
                let hot = idle.take().filter(|hot| {
                    if job.ids.starts_with(&hot.ids) && !hot.logits.is_empty() {
                        true
                    } else {
                        let _ = self.model.release_sequence(hot.seq);
                        false
                    }
                });
                let (root, reused, logits) = match hot {
                    Some(hot) => (hot.seq, hot.ids.len(), hot.logits),
                    None => match self.model.allocate_sequence() {
                        Ok(s) => (s, 0, Vec::new()),
                        Err(e) => {
                            let _ = job.tx.send(Event::Error(e.to_string()));
                            continue;
                        }
                    },
                };
                let restore = if reused > 0 {
                    None
                } else {
                    self.prefix.as_ref().map(|p| p.restore_async(job.ids.clone(), p.keys(&job.ids, &[]), self.model.snapshot_bytes(0)))
                };
                used_pages += credits;
                used_sequences += job.n;
                groups.push(Group {
                    expert_rows: 0,
                    experts: 0,
                    job,
                    root: Some(root),
                    choices: Vec::new(),
                    credits,
                    start: Instant::now(),
                    restore,
                    imported: 0,
                    restored: 0,
                    reused,
                    restore_s: 0.0,
                    logits,
                });
            }
            // Consume at most one staged object per request per iteration. Disk completion is
            // polled; neither a blocked read nor a full staging pool stalls decoding.
            for g in &mut groups {
                let Some(reader) = &g.restore else { continue };
                let event = reader.try_recv();
                match event {
                    Ok(Ok(object)) => {
                        let root = g.root.unwrap();
                        let result = (|| -> anyhow::Result<()> {
                            anyhow::ensure!(object.end <= g.job.ids.len(), "cached object exceeds prompt");
                            if object.kind == tr_cache::Kind::Rows {
                                anyhow::ensure!(object.start == g.imported, "non-contiguous restore rows");
                                anyhow::ensure!(
                                    object.bytes().len() == self.model.rows_bytes(object.start, object.end),
                                    "invalid cached row size"
                                );
                                self.model.prepare_sequence(root, object.end)?;
                                self.model.with_sequence(root, |m| m.import_rows(object.start, object.end, object.bytes()))?;
                                g.imported = object.end;
                            } else {
                                let host = self.model.with_sequence(root, |m| m.import_snapshot(object.bytes()))??;
                                anyhow::ensure!(
                                    host.n_past == g.imported && host.n_past == object.end,
                                    "snapshot position differs from restored rows"
                                );
                                anyhow::ensure!(host.extra.len() == self.model.cfg.n_vocab, "cached snapshot has no usable logits");
                                g.logits = host.extra;
                                g.restored = host.n_past;
                                g.reused = host.n_past;
                                g.restore_s = g.start.elapsed().as_secs_f64();
                                g.restore = None;
                            }
                            Ok(())
                        })();
                        if let Err(e) = result {
                            restore_failed(&mut self.model, g, &e.to_string());
                        }
                    }
                    Ok(Err(e)) => restore_failed(&mut self.model, g, &e.to_string()),
                    Err(TryRecvError::Disconnected) => {
                        if g.imported > 0 {
                            restore_failed(&mut self.model, g, "incomplete cache restore");
                        } else {
                            g.restore = None;
                        }
                    }
                    Err(TryRecvError::Empty) => {}
                }
            }
            for g in &mut groups {
                if g.restore.is_none() && g.root.is_some_and(|s| self.model.sequence_position(s).unwrap_or(0) == g.job.ids.len()) {
                    if let Err(e) = start_choices(&mut self.model, &self.tok, &self.stop_ids, self.prefix.as_ref(), g, &mut idle) {
                        fail_group(&mut self.model, g, &e.to_string());
                    }
                }
            }
            // Owned scheduling descriptors make the borrowing boundary explicit: model execution
            // sees immutable tokens while completion processing mutates the request state.
            let mut scheduled: Vec<(usize, Option<usize>, SequenceId, Vec<u32>, bool)> = Vec::new();
            for (gi, g) in groups.iter().enumerate() {
                for (ci, c) in g.choices.iter().enumerate() {
                    if !c.finished {
                        if let Some(t) = c.pending {
                            scheduled.push((gi, Some(ci), c.seq, vec![t], true));
                        }
                    }
                }
            }
            let mut budget = self.model.batch_max() - scheduled.len();
            if !scheduled.is_empty() {
                budget = budget.min(prefill_chunk);
            }
            if !groups.is_empty() {
                for off in 0..groups.len() {
                    let gi = (cursor + off) % groups.len();
                    let g = &groups[gi];
                    if g.restore.is_some() {
                        continue;
                    }
                    let Some(root) = g.root else { continue };
                    let pos = self.model.sequence_position(root).unwrap();
                    let mut end = (pos + budget).min(g.job.ids.len());
                    let mut snapshot_at = None;
                    if let Some(pc) = &self.prefix {
                        if let Some((point, _)) = pc.snapshot_points(&g.job.spans, pos, g.job.ids.len()).first() {
                            snapshot_at = Some(*point);
                            end = end.min(*point);
                        }
                    }
                    if end > pos {
                        scheduled.push((gi, None, root, g.job.ids[pos..end].to_vec(), end == g.job.ids.len() || snapshot_at == Some(end)));
                        budget -= end - pos;
                    }
                    if budget == 0 {
                        break;
                    }
                }
                cursor = (cursor + 1) % groups.len();
            }
            if !scheduled.is_empty() {
                self.stats.iterations.fetch_add(1, Ordering::Relaxed);
                self.stats.max_batch_sequences.fetch_max(scheduled.len(), Ordering::Relaxed);
                self.stats
                    .decode_rows
                    .fetch_add(scheduled.iter().filter(|s| s.1.is_some()).map(|s| s.3.len() as u64).sum(), Ordering::Relaxed);
                self.stats
                    .prefill_rows
                    .fetch_add(scheduled.iter().filter(|s| s.1.is_none()).map(|s| s.3.len() as u64).sum(), Ordering::Relaxed);
                let inputs: Vec<_> =
                    scheduled.iter().map(|(_, _, s, t, logits)| BatchInput { sequence: *s, tokens: t, logits: *logits }).collect();
                match self.model.forward_batch(&inputs) {
                    Ok(outputs) => {
                        for ((gi, ci, _, tokens, _), output) in scheduled.into_iter().zip(outputs) {
                            let g = &mut groups[gi];
                            if let Some(ci) = ci {
                                let c = &mut g.choices[ci];
                                c.stepped.extend_from_slice(&tokens);
                                c.logits = output.logits.unwrap();
                                c.expert_rows += output.expert_rows;
                                c.experts += output.experts;
                                advance_choice(&mut self.model, &self.stop_ids, self.prefix.as_ref(), &g.job, c, &mut idle);
                            } else {
                                g.expert_rows += output.expert_rows;
                                g.experts += output.experts;
                                if let Some(logits) = output.logits {
                                    g.logits = logits;
                                }
                                let root = g.root.unwrap();
                                let pos = self.model.sequence_position(root).unwrap();
                                if let Some(pc) = &self.prefix {
                                    let pts = pc.snapshot_points(&g.job.spans, pos.saturating_sub(tokens.len()), g.job.ids.len());
                                    if pts.iter().any(|&(p, _)| p == pos) {
                                        let ids = &g.job.ids[..pos];
                                        let keys = pc.keys(ids, &[]);
                                        let spans = clipped_spans(&g.job.spans, pos);
                                        let _ = self.model.with_sequence(root, |m| pc.save_ready(m, ids, &keys, &spans, &g.logits));
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        for g in &mut groups {
                            fail_group(&mut self.model, g, &e.to_string());
                        }
                    }
                }
            } else if groups.is_empty() && waiting.is_empty() {
                if closed {
                    if let Some(hot) = idle.take() {
                        let _ = self.model.release_sequence(hot.seq);
                    }
                    break;
                }
                match rx.recv() {
                    Ok(job) => waiting.push_back(job),
                    Err(_) => closed = true,
                }
            } else {
                // Wait briefly for arrivals/cache progress without spinning an executor core.
                match rx.recv_timeout(std::time::Duration::from_millis(1)) {
                    Ok(job) => waiting.push_back(job),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        closed = true;
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => {}
                }
            }
        }
    }
}
fn clipped_spans(spans: &[Span], n: usize) -> Vec<Span> {
    spans.iter().filter(|s| s.start < n).map(|s| Span::new(s.start, s.end.min(n), s.role)).collect()
}
fn restore_failed(model: &mut Model, g: &mut Group, reason: &str) {
    eprintln!("{}: prefix restore skipped: {reason}", g.job.id);
    if let Some(root) = g.root {
        let _ = model.with_sequence(root, |m| m.reset());
    }
    g.restore = None;
    g.imported = 0;
    g.restored = 0;
    g.reused = 0;
    g.logits.clear();
}
fn start_choices(
    model: &mut Model,
    tok: &Arc<Tok>,
    stops: &[u32],
    prefix: Option<&PrefixCache>,
    g: &mut Group,
    idle: &mut Option<Idle>,
) -> anyhow::Result<()> {
    anyhow::ensure!(!g.logits.is_empty(), "prefill completed without logits");
    let root = g.root.unwrap();
    let mut ids = vec![root];
    for _ in 1..g.job.n {
        match model.fork_sequence(root) {
            Ok(s) => ids.push(s),
            Err(e) => {
                for s in ids.into_iter().skip(1) {
                    let _ = model.release_sequence(s);
                }
                return Err(e);
            }
        }
    }
    let _ = g.job.tx.send(Event::Prefilled {
        prompt_tokens: g.job.ids.len(),
        reused: g.reused,
        restored: g.restored,
        restore_s: g.restore_s,
        prompt_s: g.start.elapsed().as_secs_f64(),
        images: 0,
        image_s: 0.0,
    });
    g.root = None;
    for (index, seq) in ids.into_iter().enumerate() {
        let mut params = g.job.params.clone();
        params.seed = params.seed.wrapping_add(index as u64);
        let first = if params.thinking_open { params.think.unwrap_or(params.sampling) } else { params.sampling };
        let mut tx = g.job.tx.clone();
        tx.index = index;
        let mut c = Choice {
            seq,
            sampler: Sampler::new(first.temp, first.top_k, first.top_p, params.seed),
            emitter: Emitter {
                tx,
                tok: tok.clone(),
                params: params.clone(),
                reasoning: Section::new(),
                content: Section::new(),
                in_reasoning: params.thinking_open,
                skip_ws: false,
                n_gen: 0,
                n_reason: 0,
            },
            pending: None,
            stepped: g.job.ids.clone(),
            logits: g.logits.clone(),
            start: Instant::now(),
            finished: false,
            expert_rows: g.expert_rows,
            experts: g.experts,
        };
        advance_choice(model, stops, prefix, &g.job, &mut c, idle);
        g.choices.push(c);
    }
    g.logits.clear();
    Ok(())
}
fn advance_choice(model: &mut Model, stops: &[u32], prefix: Option<&PrefixCache>, job: &Job, c: &mut Choice, idle: &mut Option<Idle>) {
    if job.cancel.load(Ordering::Relaxed) {
        finish_choice(model, prefix, job, c, "cancelled", idle);
        return;
    }
    let token = c.sampler.sample(&c.logits);
    if stops.contains(&token) {
        finish_choice(model, prefix, job, c, "stop", idle);
        return;
    }
    let stop = c.emitter.push(token, &mut c.sampler);
    if stop {
        finish_choice(model, prefix, job, c, "stop", idle);
    } else if c.emitter.n_gen >= job.params.max_tokens {
        finish_choice(model, prefix, job, c, "length", idle);
    } else {
        c.pending = Some(token);
    }
}
fn finish_choice(
    model: &mut Model,
    prefix: Option<&PrefixCache>,
    job: &Job,
    c: &mut Choice,
    reason: &'static str,
    idle: &mut Option<Idle>,
) {
    c.emitter.flush();
    let _ = c.emitter.tx.send(Event::Done(Finish {
        reason,
        completion_tokens: c.emitter.n_gen,
        reasoning_tokens: c.emitter.n_reason,
        gen_s: c.start.elapsed().as_secs_f64(),
        drafted: 0,
        accepted: 0,
        experts: if model.moe_policy().is_some() && c.expert_rows > 0 { c.experts as f64 / c.expert_rows as f64 } else { 0.0 },
    }));
    if reason != "cancelled" {
        if let Some(pc) = prefix {
            let n = c.stepped.len();
            let mut spans = clipped_spans(&job.spans, n);
            let p = job.ids.len();
            let r = if job.params.thinking_open { (p + c.emitter.n_reason).min(n) } else { p };
            if r > p {
                spans.push(Span::new(p, r, Role::Reasoning));
            }
            if n > r {
                spans.push(Span::new(r, n, Role::Assistant));
            }
            let keys = pc.keys(&c.stepped, &[]);
            let _ = model.with_sequence(c.seq, |m| pc.save_ready(m, &c.stepped, &keys, &spans, &c.logits));
        }
    }
    if reason != "cancelled" {
        if let Some(old) = idle.replace(Idle { seq: c.seq, ids: c.stepped.clone(), logits: c.logits.clone() }) {
            let _ = model.release_sequence(old.seq);
        }
    } else {
        let _ = model.release_sequence(c.seq);
    }
    c.finished = true;
    c.pending = None;
}
fn fail_group(model: &mut Model, g: &mut Group, message: &str) {
    let _ = g.job.tx.send(Event::Error(message.to_string()));
    g.restore = None;
    if let Some(s) = g.root.take() {
        let _ = model.release_sequence(s);
    }
    for c in &mut g.choices {
        if !c.finished {
            let _ = model.release_sequence(c.seq);
            c.finished = true;
        }
    }
}
fn send_cancelled(job: &Job) {
    for i in 0..job.n {
        let mut tx = job.tx.clone();
        tx.index = i;
        let _ = tx.send(Event::Done(Finish {
            reason: "cancelled",
            completion_tokens: 0,
            reasoning_tokens: 0,
            gen_s: 0.0,
            drafted: 0,
            accepted: 0,
            experts: 0.0,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservation_accounts_for_shared_prefix_and_private_tails() {
        assert_eq!(continuous_pages(128, 64, 3), Some(5));
        assert_eq!(continuous_pages(129, 64, 3), Some(8));
        assert_eq!(continuous_pages(63, 1, 3), Some(3));
        assert_eq!(continuous_pages(64, 1, 3), Some(4));
        assert_eq!(continuous_pages(1, usize::MAX, 1), None);
        assert_eq!(continuous_pages(1, 64, usize::MAX), None);
    }
    #[test]
    fn stalled_output_cancels_without_blocking_other_choices() {
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let tx = EventTx { inner: tx, index: 2, cancel: cancel.clone() };
        assert!(tx.send(Event::Content("one".into())).is_ok());
        assert!(tx.send(Event::Content("two".into())).is_err());
        assert!(cancel.load(Ordering::Relaxed));
    }
}
