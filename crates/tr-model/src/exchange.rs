//! Cross-tile mailboxes: every tile owns a region in its own memory; producers write locally,
//! consumers pull remote slices after a global barrier. Pull work is split across the tile's
//! cores into a tile-shared buffer, followed by a node barrier, so remote traffic is done once.
use tr_sys::numa::Arena;
use tr_sys::pool::WorkerCtx;

/// Candidates each tile contributes to an in-pool draft sample (see `exec_mtp::draft_pick`).
pub const CAND_PER_TILE: usize = 64;

#[derive(Clone, Copy)]
pub struct Slot {
    pub off: usize, // f32 offset inside a tile's mailbox
    pub len: usize, // f32 per tile
}

/// Fixed mailbox layout (f32 units) shared by all tiles.
pub struct MailboxLayout {
    pub total: usize,
    pub emb: Slot,      // token embedding row (published by the owning tile), hidden
    pub lo: Slot,       // hc low-rank partial sums (K-split), hc_lr per tile
    pub mixed: Slot,    // hidden / n_tiles
    pub rlog: Slot,     // router partial logits, n_expert
    pub part: Slot,     // partial sums, hidden
    pub ple: Slot,      // ple stats (2*hc) + value slice (hidden/n_tiles)
    pub idx: Slot,      // QSA indexer projections, (idx_heads*idx_dim + idx_dim) / n_tiles per tile
    pub logits: Slot,   // lm_head slice, vocab per tile
    pub token: Slot,    // sampled token (as f32 bits), 16
    pub cand: Slot,     // draft chain: this tile's top CAND_PER_TILE (id bits, logit) pairs of its logits slice
    pub rsum: Slot,     // hierarchical all-reduce: this tile's n_sockets slices summed over its own socket
    pub rfull: Slot,    // ... and summed over all sockets (see exec_batch::reduce_scatter)
    pub m_max: usize,
    pub part_w: usize,  // batch layout of `part`: n_tiles column blocks of [m_max][part_w] (see `part_block`)
    pub logits_rows: usize,
}

impl MailboxLayout {
    /// `m_max` rows for the per-token slots (decode uses row 0). Slot `len` is per row.
    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, hc: usize, hc_lr: usize, n_expert: usize, n_tiles: usize, vocab_per_tile: usize, idx_per_tile: usize, m_max: usize, n_sockets: usize, logit_rows: usize) -> MailboxLayout {
        let mut off = 0usize;
        let mut slot = |len: usize, rows: usize| {
            let s = Slot { off, len };
            off += (len * rows + 15) / 16 * 16;
            s
        };
        let emb = slot(hidden, m_max);
        let lo = slot(hc_lr, m_max); // per-tile partial sums of the low-rank down projection
        let mixed = slot(hidden / n_tiles, m_max);
        let rlog = slot(n_expert, m_max);
        let part = slot(hidden, m_max);
        let ple = slot(2 * hc + hidden / n_tiles + 16, m_max);
        let idx = slot(idx_per_tile, m_max);
        let logits = slot(vocab_per_tile, logit_rows.max(1)); // speculative verify returns one row per draft
        let token = slot(16, 1);
        let cand = slot(2 * CAND_PER_TILE, 1);
        let rsum = slot(hidden.max(n_expert) / n_tiles * n_sockets, m_max);
        let rfull = slot(hidden.max(n_expert) / n_tiles * n_sockets, m_max);
        MailboxLayout { total: off, emb, lo, mixed, rlog, part, ple, idx, logits, token, cand, rsum, rfull, m_max, part_w: hidden / n_tiles, logits_rows: logit_rows.max(1) }
    }
    /// Batch-mode view of `part`: column block `b` (columns b*part_w..) of all rows as one
    /// [m_max][part_w] region, so a tile's reduce-scatter reads one contiguous stream per source
    /// tile instead of a short chunk per row. Decode uses row 0 of the row-major view instead;
    /// the two never overlap in time.
    pub fn part_block(&self, b: usize) -> Slot {
        Slot { off: self.part.off + b * self.m_max * self.part_w, len: self.m_max * self.part_w }
    }
    /// The same slot shifted to row `m` (per-token slots).
    pub fn row(s: Slot, m: usize) -> Slot {
        Slot { off: s.off + m * s.len, len: s.len }
    }
}

pub struct Mailboxes {
    pub ptrs: Vec<*mut f32>,
    pub layout: MailboxLayout,
}
unsafe impl Send for Mailboxes {}
unsafe impl Sync for Mailboxes {}

impl Mailboxes {
    pub fn new(arenas: &mut [&mut Arena], layout: MailboxLayout) -> anyhow::Result<Mailboxes> {
        let mut ptrs = Vec::new();
        for a in arenas.iter_mut() {
            let s = a.alloc_slice::<f32>(layout.total)?;
            ptrs.push(s.as_mut_ptr());
        }
        Ok(Mailboxes { ptrs, layout })
    }
    #[inline]
    pub fn slot(&self, tile: usize, s: Slot) -> &'static mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptrs[tile].add(s.off), s.len) }
    }
    /// `rows` consecutive rows of a per-token slot as one mutable slice.
    #[inline]
    pub fn slot_rows(&self, tile: usize, s: Slot, rows: usize) -> &'static mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptrs[tile].add(s.off), s.len * rows) }
    }
    #[inline]
    pub fn slot_rows_ro(&self, tile: usize, s: Slot, rows: usize) -> &'static [f32] {
        unsafe { std::slice::from_raw_parts(self.ptrs[tile].add(s.off), s.len * rows) }
    }
    #[inline]
    pub fn slot_ro(&self, tile: usize, s: Slot) -> &'static [f32] {
        unsafe { std::slice::from_raw_parts(self.ptrs[tile].add(s.off), s.len) }
    }

    /// All-gather: after a global barrier, copy every tile's `slot` (len per tile) into `dst`
    /// (n_tiles*len), splitting the copy across this tile's cores; ends with a node barrier.
    pub fn gather(&self, ctx: &WorkerCtx, s: Slot, len: usize, dst: &mut [f32]) {
        let n = ctx.n_nodes;
        let total = n * len;
        let r = ctx.range_local(total);
        for i in r {
            let t = i / len;
            let j = i % len;
            dst[i] = self.slot_ro(t, s)[j];
        }
        ctx.node_barrier();
    }

    /// All-reduce (sum): dst[j] = sum_t slot_t[j], split across the tile's cores; node barrier.
    pub fn reduce(&self, ctx: &WorkerCtx, s: Slot, len: usize, dst: &mut [f32]) {
        let n = ctx.n_nodes;
        let r = ctx.range_local(len);
        if !r.is_empty() {
            let d = &mut dst[r.clone()];
            d.copy_from_slice(&self.slot_ro(0, s)[r.clone()]);
            for t in 1..n {
                let src = &self.slot_ro(t, s)[r.clone()];
                for (a, b) in d.iter_mut().zip(src) {
                    *a += *b;
                }
            }
        }
        ctx.node_barrier();
    }
}
