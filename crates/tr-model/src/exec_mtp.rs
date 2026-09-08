//! MTP draft head (speculative decoding). One full-attention decoder layer with its own output
//! mixer, fed with the main model's residual of the previous position and the embedding of the
//! current token:
//!
//!   e    = fc_embedding( rms(embed(tok)) * enorm )                              [hidden]
//!   x[s] = fc_hidden( (rms over all hc*hidden of hc_hidden) * hnorm )[s] + e    s in 0..hc
//!
//! The row for position p consumes `hc_hidden(p-1)` and `tok(p)` and predicts `tok(p+1)`; its
//! K/V lands at position p of the draft layer's own cache. Rows run through the batched layer
//! path (`layer_batch`) with the draft layer's `LayerW`; the logits of the last row come from
//! the decode head with the draft mixer (`head_step_with`). Sources of `hc_hidden`:
//! `Carry` = the residual the main model produced for the last committed position (kept per
//! tile in `TileWs::mtp_carry`), `Row(i)` = row i of the batch residual (verify rows / prefill
//! rows of the same pool run), `Chain` = the residual this layer produced last (draft chaining).
use crate::exchange::{MailboxLayout, CAND_PER_TILE as CAND};
use crate::exec::{CoreScratch, Model};
use crate::exec_batch::{img_row, BatchWs, ImageSeg};
use crate::sampler::{Dist, Sampler};
use tr_kernels::elem::{self, rmsnorm};
use tr_sys::pool::WorkerCtx;

/// Sampling settings of an in-pool draft chain: the target's temperature / top-k / top-p, one
/// pre-drawn uniform per draft (the chain length), and the ids drafting must stop after.
pub struct DraftPlan {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub uniforms: Vec<f32>,
    pub stop_at: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpIn {
    Carry,
    Row(usize),
    Chain,
}

impl Model {
    /// Draft-head rows for positions `pos0..pos0+toks.len()` (worker side). `save_carry`: copy
    /// that row of `bw.res` into the carry before anything overwrites it. With `logits`, the last
    /// row's logits land in LOGITS row 0 of every tile (global barrier at the end).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mtp_rows(&self, ctx: &WorkerCtx, bw: &mut BatchWs, st: &mut crate::state::TileState, cs: &mut CoreScratch, toks: &[u32], ins: &[MtpIn], pos0: usize, save_carry: Option<usize>, logits: bool, imgs: &[ImageSeg], main_row0: Option<usize>) {
        let cfg = &self.cfg;
        let mw = &self.mtp.as_ref().expect("MTP overlay not loaded")[ctx.node];
        let m = toks.len();
        assert!(m >= 1 && m <= bw.m_max && ins.len() == m);
        let t = ctx.node;
        let (c, nc) = (ctx.local, ctx.cores_per_node);
        let (hh, hc, tiles) = (cfg.hidden, cfg.hc, cfg.n_tiles);
        let hcd = hc * hh;
        let hl = hh / tiles;
        let ws = unsafe { &mut *self.ws[t].0.get() };
        let mb = &self.mbox;
        let lay = &mb.layout;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }

        // 0. carry: the main model's residual for the last committed position
        if let (Some(r), 0) = (save_carry, c) {
            ws.mtp_carry.copy_from_slice(&bw.res[r * hcd..(r + 1) * hcd]);
        }
        // rope positions: the main batch's rows (prefill hook: draft row r is main row main_row0 + r),
        // else text positions from the cell index
        let r0 = main_row0.unwrap_or(0);
        let p3: Vec<[u32; 3]> = (0..m).map(|r| match main_row0 { Some(o) => [bw.pos3[3 * (o + r)], bw.pos3[3 * (o + r) + 1], bw.pos3[3 * (o + r) + 2]], None => self.rpos(pos0 + r) }).collect();
        ctx.node_barrier();
        for r in ctx.range_local(m) {
            bw.pos3[3 * r..3 * r + 3].copy_from_slice(&p3[r]);
        }
        // 1. token embeddings: owner tiles publish the rows (image rows come from the encoder output)
        for (mi, &tok) in toks.iter().enumerate() {
            let owner = tok as usize / self.vocab_per_tile;
            if t == owner && mi % nc == c && img_row(imgs, r0 + mi, hh).is_none() {
                let slot = mb.slot(t, MailboxLayout::row(lay.emb, mi));
                crate::weights::dequant_tq_row(&self.heads[t].embd, tok as usize % self.vocab_per_tile, &mut slot[..hh]);
            }
        }
        ctx.barrier();
        // 2. per row: normalised hidden source (one rms over all streams) and normalised embedding
        for r in ctx.range_local(m) {
            let src: &[f32] = match ins[r] {
                MtpIn::Carry => &ws.mtp_carry[..],
                MtpIn::Chain => &ws.mtp_res[..],
                MtpIn::Row(i) => &bw.res[i * hcd..(i + 1) * hcd],
            };
            rmsnorm(src, Some(mw.hnorm), cfg.rms_eps, &mut bw.xn[r * hcd..(r + 1) * hcd]);
            let owner = toks[r] as usize / self.vocab_per_tile;
            let e = match img_row(imgs, r0 + r, hh) {
                Some(e) => e,
                None => &mb.slot_ro(owner, MailboxLayout::row(lay.emb, r))[..hh],
            };
            rmsnorm(e, Some(mw.enorm), cfg.rms_eps, &mut bw.emb[r * hh..(r + 1) * hh]);
        }
        ctx.node_barrier();
        // 3. quantise: the hc streams of a row are hc consecutive K=hidden rows of the same buffer
        for r in ctx.range_local(m) {
            self.prep_rows(bw.xn, hcd, r, r + 1, bw.mtp_q, bw.mtp_s, bw.mtp_sum, bw.mtp_h, hc * m);
            self.prep_rows(bw.emb, hh, r, r + 1, bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m);
        }
        ctx.node_barrier();
        lap!("mtp.prep");
        // 4. this tile's rows of fc_hidden (hc*m activation rows) and fc_embedding (m rows) into the
        //    PART mailbox rows: row r = [stream 0 | .. | stream hc-1 | e], hl each
        {
            let act_h = self.act(bw.mtp_q, bw.mtp_s, bw.mtp_sum, bw.mtp_h, hc * m, hh);
            let act_e = self.act(bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m, hh);
            let (nh, ne) = (mw.fc_hid.n_strips(), mw.fc_emb.n_strips());
            let r = ctx.range_local(nh + ne);
            let (buf, stage) = unsafe { (std::slice::from_raw_parts_mut(bw.amx_buf.as_mut_ptr().add(c * (crate::exec_batch::AMX_BUF_BYTES / 64)), crate::exec_batch::AMX_BUF_BYTES / 64), std::slice::from_raw_parts_mut(bw.stage.as_mut_ptr().add(c * bw.stage_per_core), bw.stage_per_core)) };
            let part = lay.part;
            if r.start < nh {
                let (lo, hi) = (r.start, r.end.min(nh));
                unsafe {
                    self.dense_gemm_stream(buf, stage, &mw.fc_hid, lo, hi, &act_h, hc * m, &mut |mi, col0, vals| {
                        let (row, s) = (mi / hc, mi % hc);
                        let dst = mb.slot(t, MailboxLayout::row(part, row));
                        dst[s * hl + col0..s * hl + col0 + vals.len()].copy_from_slice(vals);
                    }, None);
                }
            }
            if r.end > nh {
                let (lo, hi) = (r.start.max(nh) - nh, r.end - nh);
                unsafe {
                    self.dense_gemm_stream(buf, stage, &mw.fc_emb, lo, hi, &act_e, m, &mut |mi, col0, vals| {
                        let dst = mb.slot(t, MailboxLayout::row(part, mi));
                        dst[hc * hl + col0..hc * hl + col0 + vals.len()].copy_from_slice(vals);
                    }, None);
                }
            }
        }
        elem::store_fence();
        lap!("mtp.fc");
        ctx.barrier();
        // 5. assemble the draft layer's input residual: x[r][s][u*hl..] = part_u[r][s] + part_u[r][e]
        for task in ctx.range_local(m * hc) {
            let (r, s) = (task / hc, task % hc);
            for u in 0..tiles {
                let row = mb.slot_ro(u, MailboxLayout::row(lay.part, r));
                let dst = &mut bw.res[r * hcd + s * hh + u * hl..r * hcd + s * hh + (u + 1) * hl];
                for i in 0..hl {
                    dst[i] = row[s * hl + i] + row[hc * hl + i];
                }
            }
        }
        ctx.node_barrier();
        lap!("mtp.fuse");
        if ctx.global == 0 && pos0 == 0 {
            if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                let _ = std::fs::write(format!("{path}.mtp_in.bin"), bw.res[..m * hcd].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
        }
        // 6. the draft layer
        self.layer_batch(ctx, bw, st, cs, &mw.layer, cfg.mtp_layer(), m, pos0, t, toks, &[], false);
        lap!("mtp.layer");
        // 7. chain source and, if wanted, the logits of the last row
        if c == 0 {
            ws.mtp_res.copy_from_slice(&bw.res[(m - 1) * hcd..m * hcd]);
        }
        if ctx.global == 0 && pos0 == 0 {
            if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                let _ = std::fs::write(format!("{path}.mtp_res_out.bin"), bw.res[..m * hcd].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
        }
        if logits {
            cs.res.copy_from_slice(&bw.res[(m - 1) * hcd..m * hcd]);
            self.head_step_with(ctx, cs, t, &mw.hc);
            lap!("mtp.head");
        } else {
            ctx.node_barrier();
        }
    }

    /// Restore the carry from the kept verify residual row when `commit` left it there (core 0;
    /// `mtp_rows` reads the carry only after its first global barrier).
    fn restore_carry(&self, ctx: &WorkerCtx, bw: &BatchWs, carry_row: Option<usize>) {
        if let (Some(r), 0) = (carry_row, ctx.local) {
            let hcd = self.cfg.hc * self.cfg.hidden;
            let ws = unsafe { &mut *self.ws[ctx.node].0.get() };
            ws.mtp_carry.copy_from_slice(&bw.res_keep[r * hcd..(r + 1) * hcd]);
        }
    }

    /// Run draft-head rows for positions `pos0..` with the given hidden sources; returns the last
    /// row's logits. Does not touch the main model's sequence state.
    pub fn mtp_extend(&mut self, toks: &[u32], pos0: usize, ins: &[MtpIn], save_carry: Option<usize>) -> Vec<f32> {
        assert!(self.mtp.is_some(), "MTP overlay not loaded");
        assert!(!toks.is_empty() && toks.len() <= self.batch_max());
        let carry_row = self.mtp_carry_row.take();
        {
            let mut pool = self.pool.take().expect("pool");
            let me: &Model = &*self;
            pool.run(move |ctx| {
                let t = ctx.node;
                let bw: &mut BatchWs = unsafe { &mut *me.batch_ws[t].0.get() };
                let st = unsafe { &mut *me.state[t].0.get() };
                let cs: &mut CoreScratch = me.core_scratch(ctx);
                me.restore_carry(ctx, bw, carry_row);
                me.mtp_rows(ctx, bw, st, cs, toks, ins, pos0, save_carry, true, &[], None);
            });
            self.pool = Some(pool);
        }
        let mut logits = vec![0f32; self.vocab_per_tile * self.cfg.n_tiles];
        for t in 0..self.cfg.n_tiles {
            let src = self.mbox.slot_ro(t, self.mbox.layout.logits);
            logits[t * self.vocab_per_tile..(t + 1) * self.vocab_per_tile].copy_from_slice(&src[..self.vocab_per_tile]);
        }
        logits.truncate(self.cfg.n_vocab);
        logits
    }

    /// A whole draft chain in one pool run: the row for `t0` at `pos0` (input: the carry), then
    /// up to `plan.uniforms.len() - 1` chained rows, each token sampled inside the pool from the
    /// previous row's logits (`draft_pick`). Returns the drafts with the distributions they were
    /// drawn from; stops early after a token in `plan.stop_at`.
    pub fn mtp_draft(&mut self, t0: u32, pos0: usize, plan: &DraftPlan) -> Vec<(u32, Dist)> {
        assert!(self.mtp.is_some(), "MTP overlay not loaded");
        let k = plan.uniforms.len();
        assert!(k >= 1);
        let carry_row = self.mtp_carry_row.take();
        self.draft_out.lock().unwrap().clear();
        {
            let mut pool = self.pool.take().expect("pool");
            let me: &Model = &*self;
            pool.run(move |ctx| {
                let t = ctx.node;
                let bw: &mut BatchWs = unsafe { &mut *me.batch_ws[t].0.get() };
                let st = unsafe { &mut *me.state[t].0.get() };
                let cs: &mut CoreScratch = me.core_scratch(ctx);
                me.restore_carry(ctx, bw, carry_row);
                let mut tok = t0;
                let prof = ctx.global == 0 && me.profile.enabled;
                for i in 0..k {
                    let ins = if i == 0 { MtpIn::Carry } else { MtpIn::Chain };
                    me.mtp_rows(ctx, bw, st, cs, &[tok], &[ins], pos0 + i, None, true, &[], None);
                    let mut tm = std::time::Instant::now();
                    tok = me.draft_pick(ctx, plan, i);
                    if prof {
                        me.profile.lap(&mut tm, "mtp.pick");
                    }
                    if plan.stop_at.contains(&tok) {
                        break;
                    }
                }
            });
            self.pool = Some(pool);
        }
        std::mem::take(&mut *self.draft_out.lock().unwrap())
    }

    /// Sample a draft token from the logits every tile holds in its LOGITS slot (row 0), without
    /// leaving the pool: each core keeps the top candidates of its slice, core 0 merges them into
    /// the tile's CAND slot, and after a global barrier every tile's core 0 builds the same
    /// distribution from all tiles' candidates (`Sampler::dist_from`, deterministic) and draws with
    /// the pre-drawn uniform `plan.uniforms[i]`. The union of per-tile top-N contains the global
    /// top-N, so for top_k <= N the distribution equals the host sampler's; otherwise it is the
    /// draft distribution truncated to n_tiles*N candidates (still a valid draft: the accept step
    /// uses the distribution actually drawn from). Worker 0 records (token, dist) in `draft_out`.
    fn draft_pick(&self, ctx: &WorkerCtx, plan: &DraftPlan, i: usize) -> u32 {
        let t = ctx.node;
        let (c, nc) = (ctx.local, ctx.cores_per_node);
        let lay = &self.mbox.layout;
        let vpt = self.vocab_per_tile;
        let ws = unsafe { &mut *self.ws[t].0.get() };
        let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0));
        // 1. per-core top-N of this core's slice of the tile's logits
        let lg = &self.mbox.slot_ro(t, lay.logits)[..vpt];
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(CAND + 1);
        for j in ctx.range_local(vpt) {
            let id = t * vpt + j;
            if id >= self.cfg.n_vocab {
                break;
            }
            let cand = (id as u32, lg[j]);
            if top.len() < CAND || cmp(&cand, top.last().unwrap()).is_lt() {
                let at = top.partition_point(|x| cmp(x, &cand).is_lt());
                top.insert(at, cand);
                top.truncate(CAND);
            }
        }
        let mine = &mut ws.cand_core[c * 2 * CAND..(c + 1) * 2 * CAND];
        for (j, slot) in mine.chunks_exact_mut(2).enumerate() {
            let (id, l) = top.get(j).copied().unwrap_or((u32::MAX, f32::NEG_INFINITY));
            slot[0] = f32::from_bits(id);
            slot[1] = l;
        }
        ctx.node_barrier();
        // 2. core 0: merge the tile's cores into the CAND mailbox slot
        if c == 0 {
            let mut all: Vec<(u32, f32)> = ws.cand_core[..nc * 2 * CAND].chunks_exact(2).map(|p| (p[0].to_bits(), p[1])).filter(|p| p.1.is_finite()).collect();
            all.sort_by(cmp);
            all.truncate(CAND);
            let out = self.mbox.slot(t, lay.cand);
            for (j, slot) in out.chunks_exact_mut(2).enumerate() {
                let (id, l) = all.get(j).copied().unwrap_or((u32::MAX, f32::NEG_INFINITY));
                slot[0] = f32::from_bits(id);
                slot[1] = l;
            }
        }
        elem::store_fence();
        ctx.barrier();
        // 3. every tile's core 0: the same distribution from all tiles' candidates, the same draw
        if c == 0 {
            let mut cands: Vec<(u32, f32)> = Vec::with_capacity(ctx.n_nodes * CAND);
            for u in 0..ctx.n_nodes {
                cands.extend(self.mbox.slot_ro(u, lay.cand).chunks_exact(2).map(|p| (p[0].to_bits(), p[1])).filter(|p| p.1.is_finite()));
            }
            let d = Sampler::dist_from(&mut cands, plan.temp, plan.top_k, plan.top_p);
            let tok = d.pick(plan.uniforms[i]);
            ws.draft_tok[0] = tok;
            if ctx.global == 0 {
                self.draft_out.lock().unwrap().push((tok, d));
            }
        }
        ctx.node_barrier();
        ws.draft_tok[0]
    }

    /// One chained draft step: the draft layer's own last residual and `tok` at `pos`.
    pub fn mtp_step(&mut self, tok: u32, pos: usize) -> Vec<f32> {
        self.mtp_extend(&[tok], pos, &[MtpIn::Chain], None)
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }
}
