//! Resident sequences and a NUMA-local, reference-counted KV page pool.
//! Only the engine thread changes page tables, and only between pool runs.
use crate::exec::{Model, SyncCell};
use crate::snapshot::HostState;
use crate::state::TileState;
use anyhow::{ensure, Result};
use std::cell::UnsafeCell;
use tr_sys::numa::Arena;

pub const PAGE_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SequenceId {
    slot: usize,
    generation: u64,
}

pub struct BatchInput<'a> {
    pub sequence: SequenceId,
    pub tokens: &'a [u32],
    /// Return logits for the final token of this segment.
    pub logits: bool,
}

pub struct BatchOutput {
    pub sequence: SequenceId,
    pub logits: Option<Vec<f32>>,
    pub expert_rows: u64,
    pub experts: u64,
}

pub(crate) struct Segment {
    pub experts: std::sync::atomic::AtomicU64,
    pub slot: usize,
    pub start: usize,
    pub len: usize,
    pub pos: usize,
    pub prev: Vec<u32>,
}

pub(crate) struct Sequence {
    generation: u64,
    live: bool,
    pub tiles: Vec<SyncCell<TileState>>,
    pub host: HostState,
    pages: Vec<usize>,
}

/// Physical page accounting, independent of allocation and model geometry.
#[derive(Debug)]
struct Pages {
    refs: Vec<usize>,
    free: Vec<usize>,
}
impl Pages {
    fn new(n: usize) -> Self {
        Self { refs: vec![0; n], free: (0..n).rev().collect() }
    }
    fn take(&mut self) -> Result<usize> {
        let p = self.free.pop().ok_or_else(|| anyhow::anyhow!("KV page pool exhausted"))?;
        self.refs[p] = 1;
        Ok(p)
    }
    fn retain(&mut self, p: usize) {
        assert!(self.refs[p] > 0);
        self.refs[p] += 1;
    }
    fn release(&mut self, p: usize) {
        assert!(self.refs[p] > 0);
        self.refs[p] -= 1;
        if self.refs[p] == 0 {
            self.free.push(p);
        }
    }
}

pub(crate) struct Sequences {
    pub slots: Vec<Sequence>,
    pages: Pages,
    // Keep all arena-backed state and shared KV slices alive until the model is dropped.
    _arenas: Vec<Arena>,
    ctx: usize,
}

impl Model {
    /// Enable text-only concurrent execution. `kv_mib` is the KV pool budget per tile;
    /// recurrent/QSA state and the output mailbox are reported separately.
    pub fn enable_sequences(&mut self, max_sequences: usize, kv_mib: usize) -> Result<()> {
        ensure!(self.sequences.is_none(), "sequences already enabled");
        ensure!(
            self.mtp.is_none() && self.vision.is_none() && self.spec_k == 0,
            "continuous batching does not support MTP or vision overlays"
        );
        ensure!(max_sequences > 0 && max_sequences < self.batch_max(), "max sequences must be positive and smaller than batch size");
        let cfg = &self.cfg;
        let ctx = unsafe { (*self.state[0].0.get()).ctx_max };
        let attn_layers: Vec<_> = (0..cfg.n_layer).filter(|&il| !cfg.is_recurrent(il)).collect();
        ensure!(!attn_layers.is_empty(), "paged execution requires attention layers");
        for &il in &attn_layers {
            let r = cfg.qsa_ratio(il);
            ensure!(r == 0 || PAGE_TOKENS % r == 0, "QSA pooling ratio must divide {PAGE_TOKENS}");
        }
        let pooled_page_bytes: usize = attn_layers
            .iter()
            .map(|&il| {
                let r = cfg.qsa_ratio(il);
                if r == 0 {
                    0
                } else {
                    PAGE_TOKENS / r * cfg.idx_dim * 4
                }
            })
            .sum();
        let page_bytes = 2 * PAGE_TOKENS * cfg.head_dim * self.kv.elem_bytes() * attn_layers.len() + pooled_page_bytes;
        let budget = kv_mib.checked_mul(1 << 20).ok_or_else(|| anyhow::anyhow!("KV budget overflow"))?;
        let np = budget / page_bytes;
        ensure!(np > 0 && np <= u32::MAX as usize / PAGE_TOKENS, "invalid KV page budget");
        let qsa_bytes: usize = attn_layers
            .iter()
            .map(|&il| {
                let r = cfg.qsa_ratio(il);
                if r == 0 {
                    0
                } else {
                    r * cfg.idx_dim * 4
                }
            })
            .sum();
        let per_sequence = TileState::ckpt_bytes(cfg) + qsa_bytes + (1 << 20);
        let layout = &self.mbox.layout;
        // Selected output rows are compacted; at most one per sequence.
        let new_layout = crate::exchange::MailboxLayout::new(
            cfg.hidden,
            cfg.hc,
            cfg.hc_lr,
            cfg.n_expert,
            cfg.n_tiles,
            self.vocab_per_tile,
            layout.idx.len,
            self.batch_max(),
            cfg.n_tiles / self.tiles_per_socket,
            max_sequences,
        );
        let extra = max_sequences
            .checked_mul(per_sequence)
            .and_then(|n| n.checked_add(new_layout.total * 4 + (2 << 20)))
            .ok_or_else(|| anyhow::anyhow!("sequence memory size overflow"))?;
        let total = np
            .checked_mul(page_bytes)
            .and_then(|n| n.checked_add(extra))
            .ok_or_else(|| anyhow::anyhow!("sequence memory size overflow"))?;
        eprintln!("continuous state per tile: {} MiB KV ({} pages), {} MiB recurrent/QSA/mailboxes, {} sequences; existing model arenas are additional",
            np * page_bytes >> 20, np, extra >> 20, max_sequences);
        // The pool is physically committed at startup; reject an impossible per-node budget
        // before touching pages instead of allowing a single NUMA tile to start swapping.
        for t in 0..cfg.n_tiles {
            let info = std::fs::read_to_string(format!("/sys/devices/system/node/node{t}/meminfo"))?;
            let kb = |key: &str| -> usize {
                info.lines()
                    .find_map(|line| line.split_once(key).and_then(|(_, value)| value.split_whitespace().next()?.parse().ok()))
                    .unwrap_or(0)
            };
            let reclaimable = kb("MemFree:").saturating_add(kb("FilePages:")).saturating_sub(kb("Unevictable:")).saturating_mul(1024);
            ensure!(total <= reclaimable.saturating_sub(256 << 20),
                "node {t}: continuous state needs {} MiB with 256 MiB headroom, only {} MiB free/reclaimable; reduce KV budget/max sequences or use --ple-mmap",
                total >> 20, reclaimable >> 20);
            eprintln!("node {t}: existing model arenas {} MiB + continuous state {} MiB = {} MiB (file mappings and host/cache staging additional)",
                self.tiles[t].arena.capacity() >> 20, total >> 20, (self.tiles[t].arena.capacity() + total) >> 20);
        }
        let mut arenas: Vec<_> = (0..cfg.n_tiles).map(|t| Arena::new(t, total)).collect::<Result<_>>()?;
        let mut tile_states: Vec<Vec<SyncCell<TileState>>> = (0..max_sequences).map(|_| Vec::new()).collect();
        for a in &mut arenas {
            let mut backing = Vec::new();
            for &il in &attn_layers {
                let n = np * PAGE_TOKENS * cfg.head_dim * self.kv.elem_bytes();
                let r = cfg.qsa_ratio(il);
                let pooled_len = if r == 0 { 0 } else { np * PAGE_TOKENS / r * cfg.idx_dim };
                backing.push((
                    il,
                    crate::state::PagedStorage {
                        k: a.alloc_slice::<u8>(n)?.as_mut_ptr(),
                        v: a.alloc_slice::<u8>(n)?.as_mut_ptr(),
                        bytes: n,
                        pooled: a.alloc_slice::<f32>(pooled_len)?.as_mut_ptr(),
                        pooled_len,
                    },
                ));
            }
            for slot in &mut tile_states {
                let mut st = TileState::new(cfg, a, 0, self.kv, 0, false)?;
                st.ctx_max = ctx;
                for &(il, backing) in &backing {
                    let ast = st.attn[il].as_mut().unwrap();
                    ast.paged = Some(backing);
                    ast.pages = Some(Vec::new());
                }
                slot.push(SyncCell(UnsafeCell::new(st)));
            }
        }
        let mut refs: Vec<_> = arenas.iter_mut().collect();
        self.mbox = crate::exchange::Mailboxes::new(&mut refs, new_layout)?;
        self.sequences = Some(Sequences {
            slots: tile_states
                .into_iter()
                .map(|tiles| Sequence { generation: 0, live: false, tiles, host: HostState::default(), pages: Vec::new() })
                .collect(),
            pages: Pages::new(np),
            _arenas: arenas,
            ctx,
        });
        Ok(())
    }

    pub fn sequence_capacity(&self) -> usize {
        self.sequences.as_ref().map_or(0, |s| s.slots.len())
    }
    pub fn page_capacity(&self) -> usize {
        self.sequences.as_ref().map_or(0, |s| s.pages.refs.len())
    }
    pub fn free_sequence_count(&self) -> usize {
        self.sequences.as_ref().map_or(0, |s| s.slots.iter().filter(|s| !s.live).count())
    }
    pub fn free_page_count(&self) -> usize {
        self.sequences.as_ref().map_or(0, |s| s.pages.free.len())
    }

    fn validate_sequence(&self, id: SequenceId) -> Result<()> {
        let s = self.sequences.as_ref().and_then(|s| s.slots.get(id.slot));
        ensure!(s.is_some_and(|s| s.live && s.generation == id.generation), "stale or invalid sequence handle");
        Ok(())
    }
    pub fn sequence_position(&self, id: SequenceId) -> Result<usize> {
        self.validate_sequence(id)?;
        Ok(self.sequences.as_ref().unwrap().slots[id.slot].host.n_past)
    }
    pub fn allocate_sequence(&mut self) -> Result<SequenceId> {
        let ss = self.sequences.as_mut().ok_or_else(|| anyhow::anyhow!("continuous execution is disabled"))?;
        let (i, s) =
            ss.slots.iter_mut().enumerate().find(|(_, s)| !s.live).ok_or_else(|| anyhow::anyhow!("sequence capacity exhausted"))?;
        s.generation = s.generation.checked_add(1).ok_or_else(|| anyhow::anyhow!("sequence generation overflow"))?;
        s.live = true;
        s.host = HostState::default();
        for st in &s.tiles {
            let st = unsafe { &mut *st.0.get() };
            st.reset();
            for a in st.attn.iter_mut().flatten() {
                a.ik_raw.fill(0.0);
                a.ik_pooled.fill(0.0);
                a.pages.as_mut().unwrap().clear();
            }
        }
        Ok(SequenceId { slot: i, generation: s.generation })
    }
    pub fn release_sequence(&mut self, id: SequenceId) -> Result<()> {
        self.validate_sequence(id)?;
        let ss = self.sequences.as_mut().unwrap();
        let s = &mut ss.slots[id.slot];
        for p in s.pages.drain(..) {
            ss.pages.release(p);
        }
        s.live = false;
        Ok(())
    }
    pub fn fork_sequence(&mut self, id: SequenceId) -> Result<SequenceId> {
        self.validate_sequence(id)?;
        let child = self.allocate_sequence()?;
        let ss = self.sequences.as_mut().unwrap();
        let pages = ss.slots[id.slot].pages[..ss.slots[id.slot].host.n_past.div_ceil(PAGE_TOKENS)].to_vec();
        for &p in &pages {
            ss.pages.retain(p);
        }
        ss.slots[child.slot].pages = pages.clone();
        ss.slots[child.slot].host = ss.slots[id.slot].host.clone();
        for st in &ss.slots[child.slot].tiles {
            for a in unsafe { &mut *st.0.get() }.attn.iter_mut().flatten() {
                a.pages = Some(pages.clone());
            }
        }
        let mut pool = self.pool.take().expect("pool");
        let me = &*self;
        pool.run(|ctx| {
            let ss = me.sequences.as_ref().unwrap();
            let src = unsafe { &*ss.slots[id.slot].tiles[ctx.node].0.get() };
            let dst = unsafe { &mut *ss.slots[child.slot].tiles[ctx.node].0.get() };
            for il in ctx.range_local(me.cfg.n_layer) {
                if let (Some(s), Some(d)) = (&src.gdn[il], &mut dst.gdn[il]) {
                    d.conv.copy_from_slice(s.conv);
                    d.ssm.copy_from_slice(s.ssm);
                }
                if let (Some(s), Some(d)) = (&src.attn[il], &mut dst.attn[il]) {
                    d.ik_raw.copy_from_slice(s.ik_raw);
                }
            }
            let r = ctx.range_local(src.ple_hist.len());
            dst.ple_hist[r.clone()].copy_from_slice(&src.ple_hist[r]);
        });
        self.pool = Some(pool);
        Ok(child)
    }

    /// Ensure all rows through `end` exist and the append tail is private. Allocation is
    /// checked before mutation, including the additional page required by copy-on-write.
    pub fn prepare_sequence(&mut self, id: SequenceId, end: usize) -> Result<()> {
        self.validate_sequence(id)?;
        let ss = self.sequences.as_mut().unwrap();
        ensure!(end <= ss.ctx, "context overflow");
        let s = &mut ss.slots[id.slot];
        let n = end.div_ceil(PAGE_TOKENS);
        let tail = s.host.n_past / PAGE_TOKENS;
        let cow = end > s.host.n_past && tail < s.pages.len() && ss.pages.refs[s.pages[tail]] > 1;
        let needed = n.saturating_sub(s.pages.len()) + usize::from(cow);
        ensure!(needed <= ss.pages.free.len(), "KV page pool exhausted");
        if cow {
            let old = s.pages[tail];
            let new = ss.pages.take()?;
            let bytes = PAGE_TOKENS * self.cfg.head_dim * self.kv.elem_bytes();
            let mut pool = self.pool.take().expect("pool");
            let tiles = &s.tiles;
            let cfg = &self.cfg;
            pool.run(|ctx| {
                let st = unsafe { &*tiles[ctx.node].0.get() };
                for il in ctx.range_local(cfg.n_layer) {
                    if let Some(a) = &st.attn[il] {
                        let p = a.paged.unwrap();
                        unsafe {
                            std::ptr::copy_nonoverlapping(p.k.add(old * bytes), p.k.add(new * bytes), bytes);
                            std::ptr::copy_nonoverlapping(p.v.add(old * bytes), p.v.add(new * bytes), bytes);
                            let r = cfg.qsa_ratio(il);
                            if r > 0 {
                                let n = PAGE_TOKENS / r * cfg.idx_dim;
                                std::ptr::copy_nonoverlapping(p.pooled.add(old * n), p.pooled.add(new * n), n);
                            }
                        }
                    }
                }
            });
            self.pool = Some(pool);
            s.pages[tail] = new;
            ss.pages.release(old);
        }
        while s.pages.len() < n {
            s.pages.push(ss.pages.take()?);
        }
        for st in &s.tiles {
            for a in unsafe { &mut *st.0.get() }.attn.iter_mut().flatten() {
                a.pages = Some(s.pages.clone());
            }
        }
        Ok(())
    }

    /// Run legacy snapshot/import APIs against one sequence, restoring the serial state
    /// afterward. The closure must not allocate/release sequences or execute another batch.
    pub fn with_sequence<R>(&mut self, id: SequenceId, f: impl FnOnce(&mut Model) -> R) -> Result<R> {
        self.validate_sequence(id)?;
        self.swap_sequence(id);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(self)));
        self.swap_sequence(id);
        match result {
            Ok(r) => Ok(r),
            Err(p) => std::panic::resume_unwind(p),
        }
    }
    fn swap_sequence(&mut self, id: SequenceId) {
        let s = &mut self.sequences.as_mut().unwrap().slots[id.slot];
        std::mem::swap(&mut self.state, &mut s.tiles);
        std::mem::swap(&mut self.n_past, &mut s.host.n_past);
        std::mem::swap(&mut self.prev_tokens, &mut s.host.prev_tokens);
        std::mem::swap(&mut self.rope_delta, &mut s.host.rope_delta);
    }

    pub fn forward_batch(&mut self, inputs: &[BatchInput<'_>]) -> Result<Vec<BatchOutput>> {
        ensure!(!inputs.is_empty(), "empty batch");
        let mut total = 0usize;
        for (i, b) in inputs.iter().enumerate() {
            self.validate_sequence(b.sequence)?;
            ensure!(!b.tokens.is_empty() && b.tokens.iter().all(|&t| (t as usize) < self.cfg.n_vocab), "empty segment or invalid token");
            ensure!(!inputs[..i].iter().any(|a| a.sequence == b.sequence), "duplicate sequence in batch");
            total = total.checked_add(b.tokens.len()).ok_or_else(|| anyhow::anyhow!("batch size overflow"))?;
            ensure!(b.tokens.len() <= self.sequences.as_ref().unwrap().ctx - self.sequence_position(b.sequence)?, "context overflow");
        }
        ensure!(total <= self.batch_max(), "batch exceeds workspace capacity");
        // Preflight the aggregate allocation, so a failed batch cannot partly allocate.
        let ss = self.sequences.as_ref().unwrap();
        let mut references = std::collections::HashMap::new();
        let mut need = 0usize;
        for b in inputs {
            let s = &ss.slots[b.sequence.slot];
            let end = s.host.n_past + b.tokens.len();
            let tail = s.host.n_past / PAGE_TOKENS;
            need += end.div_ceil(PAGE_TOKENS).saturating_sub(s.pages.len());
            if let Some(&p) = s.pages.get(tail) {
                let remaining = references.entry(p).or_insert(ss.pages.refs[p]);
                if *remaining > 1 {
                    need += 1;
                    *remaining -= 1;
                }
            }
        }
        ensure!(need <= ss.pages.free.len(), "KV page pool exhausted");
        for b in inputs {
            self.prepare_sequence(b.sequence, self.sequence_position(b.sequence)? + b.tokens.len())?;
        }
        let mut tokens = Vec::with_capacity(total);
        self.segments.clear();
        self.output_rows.clear();
        for b in inputs {
            let s = &self.sequences.as_ref().unwrap().slots[b.sequence.slot];
            self.segments.push(Segment {
                experts: std::sync::atomic::AtomicU64::new(0),
                slot: b.sequence.slot,
                start: tokens.len(),
                len: b.tokens.len(),
                pos: s.host.n_past,
                prev: s.host.prev_tokens.clone(),
            });
            tokens.extend_from_slice(b.tokens);
            if b.logits {
                self.output_rows.push(tokens.len() - 1);
            }
        }
        let mut pool = self.pool.take().expect("pool");
        let me = &*self;
        pool.run(|ctx| me.worker_batch(ctx, &tokens, 0, &[], crate::exec_batch::BatchMode::Continuous, None, &[]));
        self.pool = Some(pool);
        let mut out = Vec::new();
        let mut row = 0;
        for b in inputs {
            let logits = if b.logits {
                let mut v = Vec::with_capacity(self.vocab_per_tile * self.cfg.n_tiles);
                for t in 0..self.cfg.n_tiles {
                    v.extend_from_slice(self.mbox.slot_ro(t, crate::exchange::MailboxLayout::row(self.mbox.layout.logits, row)));
                }
                v.truncate(self.cfg.n_vocab);
                row += 1;
                Some(v)
            } else {
                None
            };
            let s = &mut self.sequences.as_mut().unwrap().slots[b.sequence.slot];
            s.host.n_past += b.tokens.len();
            s.host.prev_tokens.extend_from_slice(b.tokens);
            let keep = self.cfg.ple_ngram.saturating_sub(1);
            let drop = s.host.prev_tokens.len().saturating_sub(keep);
            s.host.prev_tokens.drain(..drop);
            let experts = self.segments[out.len()].experts.load(std::sync::atomic::Ordering::Relaxed);
            out.push(BatchOutput {
                sequence: b.sequence,
                logits,
                expert_rows: (b.tokens.len() * self.manifest.layer_ids.len()) as u64,
                experts,
            });
        }
        self.segments.clear();
        self.output_rows.clear();
        Ok(out)
    }

    pub(crate) fn segment_ranges(&self, m: usize) -> impl Iterator<Item = (usize, usize)> + Clone + '_ {
        std::iter::once((0, m)).take(usize::from(self.segments.is_empty())).chain(self.segments.iter().map(|s| (s.start, s.len)))
    }

    pub(crate) fn row_segment(&self, row: usize) -> Option<&Segment> {
        self.segments.iter().find(|s| row >= s.start && row < s.start + s.len)
    }
    pub(crate) fn row_position(&self, row: usize, pos0: usize) -> usize {
        self.row_segment(row).map_or(pos0 + row, |s| s.pos + row - s.start)
    }
    /// Same SPMD ownership rules as `state`: each worker writes only its assigned rows,
    /// heads or channel chunks, separated by the existing barriers.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn row_state(&self, tile: usize, row: usize) -> &mut TileState {
        match self.row_segment(row) {
            Some(s) => &mut *self.sequences.as_ref().unwrap().slots[s.slot].tiles[tile].0.get(),
            None => &mut *self.state[tile].0.get(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_pages_reclaimed_only_after_last_owner() {
        let mut p = Pages::new(2);
        let a = p.take().unwrap();
        p.retain(a);
        let b = p.take().unwrap();
        assert!(p.take().is_err());
        p.release(a);
        assert!(p.free.is_empty());
        p.release(a);
        assert_eq!(p.take().unwrap(), a);
        p.release(a);
        p.release(b);
        assert_eq!(p.free.len(), 2);
    }
}
