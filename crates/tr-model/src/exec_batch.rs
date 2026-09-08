//! Batched prompt processing (M tokens per step) with the same sharding, state and mailboxes as
//! the decode path. Buffers are tile-shared and row-major [M][...]; work is split across a tile's
//! cores by rows/strips/tasks; activations use single-level int8 (as llama.cpp does for prefill).
use crate::exec::{CoreScratch, GdnW, HcW, Model};
use crate::exchange::{MailboxLayout, Slot};
use crate::exec::kv_dispatch;
use tr_kernels::attn::{attend_block, dot, QB};
use tr_kernels::qsa;
use tr_kernels::elem::{self, rmsnorm, sigmoid, silu, softplus};
use tr_kernels::gemv::gemm_tq;
use tr_kernels::amx::{self, Line};
use crate::weights::TqMat;
use tr_kernels::quant::{quant_block, QActRef};
use tr_sys::numa::Arena;
use tr_sys::pool::WorkerCtx;

/// Per-core unpacked bf16 weight buffer for the AMX GEMMs (1 MiB, in the tile arena).
pub const AMX_BUF_BYTES: usize = 1 << 20;
/// Columns of GEMM output staged per core before streaming into a mailbox slot (20 strips).
pub const STAGE_COLS: usize = 320;

/// Rows of a batch whose input embedding comes from an image encoder instead of `token_embd`:
/// rows `row..row+n` take `embd[(mi-row)*hidden..]`. The token ids at those rows are the image
/// pad id (the PLE hashes them, as llama.cpp does).
#[derive(Clone, Copy)]
pub struct ImageSeg<'a> {
    pub row: usize,
    pub n: usize,
    pub embd: &'a [f32],
}

pub(crate) fn img_row<'a>(imgs: &[ImageSeg<'a>], mi: usize, hh: usize) -> Option<&'a [f32]> {
    imgs.iter().find(|s| mi >= s.row && mi < s.row + s.n).map(|s| &s.embd[(mi - s.row) * hh..(mi - s.row + 1) * hh])
}

/// What a batched step is for: prefill advances the recurrent state in place and returns the last
/// row's logits; verify (speculative decoding) checkpoints the state per row and returns every
/// row's logits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BatchMode {
    Prefill,
    Verify,
    Continuous,
}

pub struct BatchWs {
    pub m_max: usize,
    pub res: &'static mut [f32],       // [M][hc*H]
    pub xn: &'static mut [f32],        // [M][hc*H]
    pub inject: &'static mut [f32],    // [M][8]
    pub xq_q: &'static mut [i8],       // [M][kslice]
    pub xq_s: &'static mut [f32],      // [M][kslice/32]
    pub xq_sum: &'static mut [i32],
    pub lo: &'static mut [f32],        // [M][lr]
    pub lo_q: &'static mut [i8],       // [M][lr]
    pub lo_s: &'static mut [f32],
    pub lo_sum: &'static mut [i32],
    pub gate_local: &'static mut [f32], // [M][hc*H/T]
    pub mixed: &'static mut [f32],     // [M][H]
    pub mixed_q: &'static mut [i8],    // [M][H]
    pub mixed_s: &'static mut [f32],
    pub mixed_sum: &'static mut [i32],
    pub rlog: &'static mut [f32],      // [M][n_expert]
    pub qkv: &'static mut [f32],       // [M][c_local]
    pub qkvc: &'static mut [f32],
    pub z: &'static mut [f32],         // [M][v_local]
    pub beta: &'static mut [f32],      // [M][8] raw projections
    pub alpha: &'static mut [f32],
    pub gate: &'static mut [f32],      // [8][M] log-decay per (v head, token)
    pub bet: &'static mut [f32],       // [8][M] sigmoid(beta)
    pub gq: &'static mut [f32],        // [M][kh*dk] l2-normalised q after conv
    pub gk: &'static mut [f32],        // [M][kh*dk] l2-normalised k after conv
    pub y: &'static mut [f32],         // [M][v_local]
    pub vv: &'static mut [f32],        // [M][v_local]  (final output / attention output)
    pub vv_q: &'static mut [i8],
    pub vv_s: &'static mut [f32],
    pub vv_sum: &'static mut [i32],
    pub qg: &'static mut [f32],        // [M][qh*2*hd]
    pub kn: &'static mut [f32],        // [M][hd]
    pub vn: &'static mut [f32],
    pub ids: &'static mut [u32],       // [M][k_alloc] (the first n_sel[mi] of each row are valid)
    pub w: &'static mut [f32],         // [M][k_alloc]
    pub n_sel: &'static mut [u32],     // [M] experts selected per token
    pub ent_tok: &'static mut [u32],   // [M*k_alloc] entries sorted by expert
    pub ent_w: &'static mut [f32],
    pub ent_expert: &'static mut [u32],
    pub grp_start: &'static mut [u32], // [n_expert+2] group boundaries over the sorted entries
    pub grp_expert: &'static mut [u32],
    pub grp_order: &'static mut [u32], // groups sorted by size, largest first (LPT for round-robin)
    pub n_groups: &'static mut [u32],  // [1]
    pub task_ctr: &'static std::sync::atomic::AtomicU32, // tile-shared dynamic task counter (attention blocks)
    pub hg: &'static mut [f32],        // [M*(k_alloc+1)][ffl]  (rows: entries, then M shared rows)
    pub hu: &'static mut [f32],
    pub act_q: &'static mut [i8],      // [M*(k_alloc+1)][ffl]
    pub act_s: &'static mut [f32],
    pub act_sum: &'static mut [i32],
    pub moe_acc: &'static mut [f32],   // [M][H] per-token MoE output (shared expert + routed experts accumulated)
    pub emb: &'static mut [f32],       // [M][H] (PLE rows)
    pub ple_key: &'static mut [f32],   // [M][hc*H/T]
    pub ple_val: &'static mut [f32],   // [M][H/T]
    pub ple_norm: &'static mut [f32],  // [M][hc*H] normalized (conv input)
    pub ple_gated: &'static mut [f32], // [M][hc*H]
    pub idx_qraw: &'static mut [f32],  // [M][idx_heads*idx_dim]
    pub idx_kraw: &'static mut [f32],  // [M][idx_dim]
    pub idx_q: &'static mut [f32],     // [M][idx_heads*idx_dim]
    pub idx_scores: &'static mut [f32], // [M][nb_max]
    pub nb_max: usize,
    // bf16 activation rows for the AMX GEMMs (64-byte aligned rows: k % 32 == 0)
    pub mixed_h: &'static mut [u16],   // [M][H]
    pub xq_h: &'static mut [u16],      // [M][kslice]
    pub lo_h: &'static mut [u16],      // [M][lr]
    pub vv_h: &'static mut [u16],      // [M][vl]
    pub act_h: &'static mut [u16],     // [M][ffl] shared-expert activations (routing weight folded in)
    pub mtp_q: &'static mut [i8],      // [M][hc*H] = [hc*M][H] quantised MTP input streams
    pub mtp_s: &'static mut [f32],
    pub mtp_sum: &'static mut [i32],
    pub mtp_h: &'static mut [u16],
    pub res_keep: &'static mut [f32],  // [M][hc*H] main-model residual rows of the last verify (the next draft chain starts from the accepted one)
    pub pos3: &'static mut [u32],      // [M][3] M-RoPE (t, h, w) of every row of the current batch
    pub amx_buf: &'static mut [Line],  // [cores_per_node][AMX_BUF_BYTES/64] unpacked strips, per core
    pub stage_per_core: usize,
    pub stage: &'static mut [f32],     // [cores_per_node][M*STAGE_COLS] private GEMM output staging before streaming into a mailbox slot
}

impl BatchWs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(a: &mut Arena, m: usize, hh: usize, hc: usize, t: usize, lr: usize, n_expert: usize, k_alloc: usize, c_local: usize, v_local: usize, qh: usize, hd: usize, ffl: usize, idx_q_len: usize, idx_dim: usize, nb_max: usize, cpn: usize) -> anyhow::Result<BatchWs> {
        let kslice = hc * hh / t;
        let vl = v_local.max(qh * hd);
        let slots = m * (k_alloc + 1);
        // The mixed-row/router staging also holds ceil(M/cores)*(H/T + experts).
        // With few cores this can exceed the dense GEMM strip staging size.
        let stage_per_core = (m * STAGE_COLS).max(m.div_ceil(cpn) * (hh / t + n_expert));
        Ok(BatchWs {
            m_max: m,
            res: a.alloc_slice(m * hc * hh)?,
            xn: a.alloc_slice(m * hc * hh)?,
            inject: a.alloc_slice(m * 8)?,
            xq_q: a.alloc_slice(m * kslice)?,
            xq_s: a.alloc_slice(m * kslice / 32)?,
            xq_sum: a.alloc_slice(m * kslice / 32)?,
            lo: a.alloc_slice(m * lr)?,
            lo_q: a.alloc_slice(m * lr)?,
            lo_s: a.alloc_slice(m * lr / 32)?,
            lo_sum: a.alloc_slice(m * lr / 32)?,
            gate_local: a.alloc_slice(m * hc * hh / t)?,
            mixed: a.alloc_slice(m * hh)?,
            mixed_q: a.alloc_slice(m * hh)?,
            mixed_s: a.alloc_slice(m * hh / 32)?,
            mixed_sum: a.alloc_slice(m * hh / 32)?,
            rlog: a.alloc_slice(m * n_expert)?,
            qkv: a.alloc_slice(m * c_local)?,
            qkvc: a.alloc_slice(m * c_local)?,
            z: a.alloc_slice(m * vl)?,
            beta: a.alloc_slice(m * 8)?,
            alpha: a.alloc_slice(m * 8)?,
            gate: a.alloc_slice(m * 8)?,
            bet: a.alloc_slice(m * 8)?,
            gq: a.alloc_slice(m * (c_local - v_local) / 2)?,
            gk: a.alloc_slice(m * (c_local - v_local) / 2)?,
            y: a.alloc_slice(m * vl)?,
            vv: a.alloc_slice(m * vl)?,
            vv_q: a.alloc_slice(m * vl)?,
            vv_s: a.alloc_slice(m * vl / 32)?,
            vv_sum: a.alloc_slice(m * vl / 32)?,
            qg: a.alloc_slice(m * qh * 2 * hd)?,
            kn: a.alloc_slice(m * hd)?,
            vn: a.alloc_slice(m * hd)?,
            ids: a.alloc_slice(m * k_alloc)?,
            w: a.alloc_slice(m * k_alloc)?,
            n_sel: a.alloc_slice(m)?,
            ent_tok: a.alloc_slice(m * k_alloc)?,
            ent_w: a.alloc_slice(m * k_alloc)?,
            ent_expert: a.alloc_slice(m * k_alloc)?,
            grp_start: a.alloc_slice(n_expert + 2)?,
            grp_expert: a.alloc_slice(n_expert + 1)?,
            grp_order: a.alloc_slice(n_expert + 1)?,
            n_groups: a.alloc_slice(16)?,
            task_ctr: unsafe { &*(a.alloc_slice::<u32>(16)?.as_ptr() as *const std::sync::atomic::AtomicU32) },
            hg: a.alloc_slice(slots * ffl)?,
            hu: a.alloc_slice(slots * ffl)?,
            act_q: a.alloc_slice(slots * ffl)?,
            act_s: a.alloc_slice(slots * ffl / 16)?,
            act_sum: a.alloc_slice(slots * ffl / 16)?,
            moe_acc: a.alloc_slice(m * hh)?,
            emb: a.alloc_slice(m * hh)?,
            ple_key: a.alloc_slice(m * hc * hh / t)?,
            ple_val: a.alloc_slice(m * hh / t)?,
            ple_norm: a.alloc_slice(m * hc * hh)?,
            ple_gated: a.alloc_slice(m * hc * hh)?,
            idx_qraw: a.alloc_slice(m * idx_q_len)?,
            idx_kraw: a.alloc_slice(m * idx_dim)?,
            idx_q: a.alloc_slice(m * idx_q_len)?,
            idx_scores: a.alloc_slice_uninit(m * nb_max)?,
            nb_max,
            mixed_h: a.alloc_slice(m * hh)?,
            xq_h: a.alloc_slice(m * kslice)?,
            lo_h: a.alloc_slice(m * lr)?,
            vv_h: a.alloc_slice(m * vl)?,
            act_h: a.alloc_slice(m * ffl)?,
            mtp_q: a.alloc_slice(m * hc * hh)?,
            mtp_s: a.alloc_slice(m * hc * hh / 32)?,
            mtp_sum: a.alloc_slice(m * hc * hh / 32)?,
            mtp_h: a.alloc_slice(m * hc * hh)?,
            res_keep: a.alloc_slice(m * hc * hh)?,
            pos3: a.alloc_slice(3 * m)?,
            amx_buf: a.alloc_slice(cpn * AMX_BUF_BYTES / 64)?,
            stage_per_core,
            stage: a.alloc_slice(cpn * stage_per_core)?,
        })
    }
}

/// Subset packs (tests): map routed expert ids onto the `packed` experts, keeping them distinct
/// within a token (the batch path assumes at most one entry per (token, expert)).
pub(crate) fn fold_ids(ids: &mut [u32], packed: u32) {
    assert!(ids.len() <= packed as usize, "more routed experts than packed experts");
    for i in 0..ids.len() {
        let mut id = ids[i] % packed;
        while ids[..i].contains(&id) {
            id = (id + 1) % packed;
        }
        ids[i] = id;
    }
}

/// Quantise rows [r0, r1) of an [rows][k] f32 matrix into int8 (single level).
pub(crate) fn quant_rows(x: &[f32], k: usize, kb: usize, r0: usize, r1: usize, q: &mut [i8], s: &mut [f32], su: &mut [i32]) {
    let nb = k / kb;
    for r in r0..r1 {
        for b in 0..nb {
            let (sc, sm) = unsafe { quant_block(&x[r * k + b * kb..r * k + (b + 1) * kb], &mut q[r * k + b * kb..r * k + (b + 1) * kb]) };
            s[r * nb + b] = sc;
            su[r * nb + b] = sm;
        }
    }
}

pub(crate) fn qref<'a>(q: &'a [i8], s: &'a [f32], su: &'a [i32], m: usize, k: usize, kb: usize) -> QActRef<'a> {
    let nb = k / kb;
    QActRef { m, k, kb, q: &q[..m * k], scale: &s[..m * nb], sum: &su[..m * nb], pair: false }
}

/// Activation operand of a dense prefill GEMM: int8 blocks (VNNI) or bf16 rows (AMX).
pub enum Act<'a> {
    Q8(QActRef<'a>),
    H(&'a [u16], usize, usize), // rows, m, k
}

/// Split a range over concatenated strip lists into per-matrix sub-ranges: (matrix, lo, hi).
fn sub_ranges(r: std::ops::Range<usize>, sizes: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut base = 0;
    for (i, &n) in sizes.iter().enumerate() {
        let (lo, hi) = (r.start.max(base), r.end.min(base + n));
        if lo < hi {
            out.push((i, lo - base, hi - base));
        }
        base += n;
    }
    out
}


/// This core's slice of the tile's AMX unpack buffer (disjoint per core; raw pointer so other
/// `bw` fields can be borrowed alongside).
macro_rules! stage_buf {
    ($bw:expr, $ctx:expr) => {
        std::slice::from_raw_parts_mut($bw.stage.as_mut_ptr().add($ctx.local * $bw.stage_per_core), $bw.stage_per_core)
    };
}

/// Copy `src` into a mailbox row that other tiles will read: streaming stores when aligned, so
/// this core does not request ownership of lines still shared in remote caches from the last read.
#[inline]
fn put_row(dst: &mut [f32], src: &[f32]) {
    if dst.len() % 16 == 0 && (dst.as_ptr() as usize) % 64 == 0 {
        elem::stream_copy(dst, src);
    } else {
        dst.copy_from_slice(src);
    }
}

macro_rules! amx_buf {
    ($bw:expr, $ctx:expr) => {
        std::slice::from_raw_parts_mut($bw.amx_buf.as_mut_ptr().add($ctx.local * (AMX_BUF_BYTES / 64)), AMX_BUF_BYTES / 64)
    };
}

impl Model {
    /// AMX path for a GEMM with this K (tile depth is 32 bf16; K = 80 shared-expert down stays
    /// VNNI) and `m` rows: below `amx_min_m` rows the bf16 unpack of every strip costs more than
    /// the VNNI kernel's 8-rows-per-pass, so small batches (speculative verify) stay on VNNI.
    pub(crate) fn use_amx(&self, k: usize, m: usize) -> bool {
        self.amx && k % 32 == 0 && m >= self.amx_min_m
    }
    /// Prepare activation rows [r0, r1) of `x` ([rows][k] f32) for the dense GEMMs: bf16 (AMX) or
    /// int8 per-32 blocks (VNNI). `m` is the batch the GEMM will run with (AMX decision).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prep_rows(&self, x: &[f32], k: usize, r0: usize, r1: usize, q: &mut [i8], s: &mut [f32], su: &mut [i32], h: &mut [u16], m: usize) {
        if self.use_amx(k, m) {
            amx::rows_to_bf16(x, k, r0, r1, h);
        } else {
            quant_rows(x, k, 32, r0, r1, q, s, su);
        }
    }
    pub(crate) fn act<'a>(&self, q: &'a [i8], s: &'a [f32], su: &'a [i32], h: &'a [u16], m: usize, k: usize) -> Act<'a> {
        if self.use_amx(k, m) {
            Act::H(&h[..m * k], m, k)
        } else {
            Act::Q8(qref(q, s, su, m, k, 32))
        }
    }
    /// Strips [s_lo, s_hi) of `mat` against `act`, f32 results into `y[mi*ldy + 16*s + n]`.
    ///
    /// # Safety
    /// `y` must hold `m` rows of `ldy` floats covering columns 16*s_hi.
    unsafe fn dense_gemm(&self, buf: &mut [Line], mat: &TqMat, s_lo: usize, s_hi: usize, act: &Act, y: *mut f32, ldy: usize) {
        if s_lo >= s_hi {
            return;
        }
        match act {
            Act::Q8(xq) => gemm_tq(mat.ptr.add(s_lo * mat.strip_bytes()), mat.k, mat.codec, s_lo, s_hi, *xq, y, ldy, false),
            Act::H(x, m, k) => {
                debug_assert_eq!(*k, mat.k);
                let per = amx::strip_lines(mat.k);
                let max_s = (buf.len() / per).max(1);
                let mut s = s_lo;
                while s < s_hi {
                    let n = (s_hi - s).min(max_s);
                    amx::unpack_strips(mat.ptr.add(s * mat.strip_bytes()), mat.k, mat.codec, n, buf.as_mut_ptr());
                    amx::gemm_bf16(buf.as_ptr(), mat.k, n, x.as_ptr(), *m, y, ldy, s * 16);
                    s += n;
                }
            }
        }
    }
}

impl Model {
    /// `dense_gemm` whose output goes to a mailbox slot read by other tiles: strips are computed in
    /// chunks of up to `STAGE_COLS` columns into the core's private staging buffer, then each row
    /// piece is handed to `put(mi, col0, vals)` which streams it into place (no RFO of lines the
    /// other tiles still hold from their last read). `m` rows; ends with a store fence.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn dense_gemm_stream(&self, buf: &mut [Line], stage: &mut [f32], mat: &TqMat, s_lo: usize, s_hi: usize, act: &Act, m: usize, put: &mut dyn FnMut(usize, usize, &[f32]), prof: Option<&mut std::time::Instant>) {
        let mut s = s_lo;
        let mut tm = prof.map(|t| (t, std::time::Instant::now()));
        while s < s_hi {
            let n = (s_hi - s).min(STAGE_COLS / 16);
            let cols = n * 16;
            // the kernels write column s*16 relative to `y`; shift so it lands at column 0 of the stage
            self.dense_gemm(buf, mat, s, s + n, act, stage.as_mut_ptr().wrapping_sub(s * 16), cols);
            if let Some((_, t)) = tm.as_mut() {
                self.profile.lap(t, "b.dgs.gemm");
            }
            for mi in 0..m {
                put(mi, s * 16, &stage[mi * cols..(mi + 1) * cols]);
            }
            if let Some((_, t)) = tm.as_mut() {
                self.profile.lap(t, "b.dgs.put");
            }
            s += n;
        }
        elem::store_fence();
        if let Some((outer, _)) = tm {
            *outer = std::time::Instant::now();
        }
    }

    /// `dense_gemm_stream` into the column-blocked batch `part` slot of tile `t` (blocks of
    /// `part_w` columns, row stride `part_w` inside a block; `STAGE_COLS` == `part_w` here).
    #[allow(clippy::too_many_arguments)]
    unsafe fn dense_gemm_part(&self, ctx: &WorkerCtx, buf: &mut [Line], stage: &mut [f32], mat: &TqMat, s_lo: usize, s_hi: usize, act: &Act, m: usize) {
        let lay = &self.mbox.layout;
        let w = lay.part_w;
        let spb = w / 16; // strips per block
        let mut s = s_lo;
        while s < s_hi {
            let b = s / spb;
            let e = s_hi.min((b + 1) * spb);
            let blk = self.mbox.slot(ctx.node, lay.part_block(b));
            self.dense_gemm_stream(buf, stage, mat, s, e, act, m, &mut |mi, col0, vals| {
                let c = col0 - b * w;
                put_row(&mut blk[mi * w + c..mi * w + c + vals.len()], vals);
            }, None);
            s = e;
        }
    }

    /// Process `toks` (positions n_past..) as one batch; returns the logits of the last token.
    pub fn step_batch(&mut self, toks: &[u32]) -> Vec<f32> {
        self.step_batch_with(toks, None, None, &[])
    }

    /// `step_batch` with explicit M-RoPE positions per row (`pos3`, text rows otherwise), the
    /// rope position the row after this batch gets (`rope_after`, needed when the batch contains
    /// image rows) and image embedding rows.
    pub fn step_batch_with(&mut self, toks: &[u32], pos3: Option<&[[u32; 3]]>, rope_after: Option<usize>, imgs: &[ImageSeg]) -> Vec<f32> {
        let m = toks.len();
        assert!(m >= 1 && m <= self.batch_max(), "batch {m} > batch_max");
        assert!(pos3.map(|p| p.len() == m).unwrap_or(true));
        if m == 1 && pos3.is_none() && imgs.is_empty() {
            return self.step(toks[0], None);
        }
        let pos0 = self.n_past;
        let prev: Vec<u32> = self.prev_tokens.clone();
        self.mtp_carry_row = None; // the prefill hook saves the carry itself
        {
            let mut pool = self.pool.take().expect("pool");
            let me: &Model = &*self;
            let toks_ref = toks;
            let prev_ref = &prev;
            pool.run(move |ctx| me.worker_batch(ctx, toks_ref, pos0, prev_ref, BatchMode::Prefill, pos3, imgs));
            self.pool = Some(pool);
        }
        if let Some(ra) = rope_after {
            self.rope_delta = ra as i64 - (pos0 + m) as i64;
        }
        let mut logits = vec![0f32; self.vocab_per_tile * self.cfg.n_tiles];
        for t in 0..self.cfg.n_tiles {
            let src = self.mbox.slot_ro(t, self.mbox.layout.logits);
            logits[t * self.vocab_per_tile..(t + 1) * self.vocab_per_tile].copy_from_slice(&src[..self.vocab_per_tile]);
        }
        logits.truncate(self.cfg.n_vocab);
        self.n_past += m;
        self.prev_tokens.extend_from_slice(toks);
        let keep = self.cfg.ple_ngram.saturating_sub(1);
        if self.prev_tokens.len() > keep {
            let drop = self.prev_tokens.len() - keep;
            self.prev_tokens.drain(..drop);
        }
        logits
    }

    pub fn batch_max(&self) -> usize {
        self.batch_ws.first().map(|b| unsafe { (*b.0.get()).m_max }).unwrap_or(1)
    }

    /// Speculative verify: run `toks` (positions n_past..) as one batch and return the logits of
    /// every row, without advancing the sequence. The KV/indexer caches are written in place
    /// (positions above the accept point are simply overwritten later); the in-place recurrent
    /// state is left untouched and per-row checkpoints are taken instead. Must be followed by
    /// exactly one `commit`.
    pub fn verify(&mut self, toks: &[u32]) -> Vec<Vec<f32>> {
        let m = toks.len();
        assert!(m >= 1 && m <= self.batch_max() && m <= self.n_ckpt(), "verify batch {m} (batch_max {}, checkpoints {})", self.batch_max(), self.n_ckpt());
        assert!(self.verify_toks.is_empty(), "verify without commit");
        let pos0 = self.n_past;
        let prev: Vec<u32> = self.prev_tokens.clone();
        {
            let mut pool = self.pool.take().expect("pool");
            let me: &Model = &*self;
            let toks_ref = toks;
            let prev_ref = &prev;
            pool.run(move |ctx| me.worker_batch(ctx, toks_ref, pos0, prev_ref, BatchMode::Verify, None, &[]));
            self.pool = Some(pool);
        }
        let vpt = self.vocab_per_tile;
        let mut rows = Vec::with_capacity(m);
        for mi in 0..m {
            let mut logits = vec![0f32; vpt * self.cfg.n_tiles];
            for t in 0..self.cfg.n_tiles {
                let src = self.mbox.slot_rows_ro(t, self.mbox.layout.logits, m);
                logits[t * vpt..(t + 1) * vpt].copy_from_slice(&src[mi * vpt..(mi + 1) * vpt]);
            }
            logits.truncate(self.cfg.n_vocab);
            rows.push(logits);
        }
        self.verify_pos0 = pos0;
        self.verify_toks = toks.to_vec();
        rows
    }

    /// Accept rows 0..=j of the last `verify`: checkpoint j becomes the live state and the
    /// sequence advances by j+1 tokens.
    pub fn commit(&mut self, j: usize) {
        let m = self.verify_toks.len();
        assert!(j < m, "commit {j} of a verify batch of {m}");
        for s in &self.state {
            unsafe { (*s.0.get()).commit(j) };
        }
        self.n_past = self.verify_pos0 + j + 1;
        let toks = std::mem::take(&mut self.verify_toks);
        self.prev_tokens.extend_from_slice(&toks[..=j]);
        let keep = self.cfg.ple_ngram.saturating_sub(1);
        if self.prev_tokens.len() > keep {
            let drop = self.prev_tokens.len() - keep;
            self.prev_tokens.drain(..drop);
        }
        self.last_commit = j;
        if self.mtp.is_some() {
            self.mtp_carry_row = Some(j);
        }
    }

    /// Checkpoint slots per tile (verify batch limit).
    pub fn n_ckpt(&self) -> usize {
        self.state.first().map(|s| unsafe { (*s.0.get()).ckpt.len() }).unwrap_or(0)
    }

    pub(crate) fn worker_batch(&self, ctx: &WorkerCtx, toks: &[u32], pos0: usize, prev: &[u32], mode: BatchMode, pos3: Option<&[[u32; 3]]>, imgs: &[ImageSeg]) {
        let cfg = &self.cfg;
        let m = toks.len();
        let t = ctx.node;
        let c = ctx.local;
        let nc = ctx.cores_per_node;
        let (hh, hc, _tiles) = (cfg.hidden, cfg.hc, cfg.n_tiles);
        let bw: &mut BatchWs = unsafe { &mut *self.batch_ws[t].0.get() };
        let st = unsafe { &mut *self.state[t].0.get() };
        let cs: &mut CoreScratch = self.core_scratch(ctx);
        let mb = &self.mbox;
        let lay = &mb.layout;

        // ---- rope positions of the rows (text: cell + rope_delta; images: explicit triples)
        for mi in ctx.range_local(m) {
            let p = match pos3 {
                Some(p) => p[mi],
                None => { let p = self.row_position(mi, pos0) as u32; if mode == BatchMode::Continuous { [p; 3] } else { self.rpos(pos0 + mi) } },
            };
            bw.pos3[3 * mi..3 * mi + 3].copy_from_slice(&p);
        }
        // ---- embeddings: owner tiles publish rows, everyone copies into the tile-shared residual
        // (image rows come from the encoder's output in host memory instead)
        for (mi, &tok) in toks.iter().enumerate() {
            let owner = tok as usize / self.vocab_per_tile;
            if t == owner && mi % nc == c && img_row(imgs, mi, hh).is_none() {
                let slot = mb.slot(t, MailboxLayout::row(lay.emb, mi));
                crate::weights::dequant_tq_row(&self.heads[t].embd, tok as usize % self.vocab_per_tile, &mut slot[..hh]);
            }
        }
        ctx.barrier();
        for task in (c..m * hc).step_by(nc) {
            let (mi, s) = (task / hc, task % hc);
            let owner = toks[mi] as usize / self.vocab_per_tile;
            let e = match img_row(imgs, mi, hh) {
                Some(e) => e,
                None => &mb.slot_ro(owner, MailboxLayout::row(lay.emb, mi))[..hh],
            };
            bw.res[mi * hc * hh + s * hh..mi * hc * hh + (s + 1) * hh].copy_from_slice(e);
        }
        ctx.node_barrier();

        for (li, &il) in self.manifest.layer_ids.iter().enumerate() {
            self.layer_batch(ctx, bw, st, cs, &self.layers[li][t], il, m, pos0, t, toks, prev, mode == BatchMode::Verify);
        }
        if mode == BatchMode::Prefill && self.mtp.is_some() {
            // ---- MTP draft head over this chunk: the row for position q takes hc_hidden(q-1) and
            // token q (position 0 has no row); the last row's residual is the carry for the next chunk
            let off = usize::from(pos0 == 0);
            if m > off {
                let ins: Vec<crate::exec_mtp::MtpIn> = (off..m).map(|q| if q == 0 { crate::exec_mtp::MtpIn::Carry } else { crate::exec_mtp::MtpIn::Row(q - 1) }).collect();
                // the main head first (it needs the last row's residual before the draft head overwrites it)
                cs.res.copy_from_slice(&bw.res[(m - 1) * hc * hh..m * hc * hh]);
                self.head_step(ctx, cs, t);
                self.mtp_rows(ctx, bw, st, cs, &toks[off..], &ins, pos0 + off, Some(m - 1), false, imgs, Some(off));
                return;
            }
        }
        match mode {
            BatchMode::Continuous => {
                let hcd = hc * hh;
                // Preserve all selected residuals before compacting them in place.
                for (out, &src) in self.output_rows.iter().enumerate() {
                    for j in ctx.range_local(hcd) { bw.res_keep[out * hcd + j] = bw.res[src * hcd + j]; }
                }
                ctx.node_barrier();
                for j in ctx.range_local(self.output_rows.len() * hcd) { bw.res[j] = bw.res_keep[j]; }
                ctx.node_barrier();
                if !self.output_rows.is_empty() { self.head_batch(ctx, bw, self.output_rows.len(), t); }
            }
            BatchMode::Prefill => {
                // ---- head for the last token: reuse the decode head on a private copy of its residual
                cs.res.copy_from_slice(&bw.res[(m - 1) * hc * hh..m * hc * hh]);
                self.head_step(ctx, cs, t);
            }
            BatchMode::Verify => {
                self.head_batch(ctx, bw, m, t);
                if self.mtp.is_some() {
                    // ---- fold the draft head's refresh into this run: keep the main residual rows (the
                    // next draft chain starts from the accepted one) and run the draft rows whose inputs
                    // are known now: position pos0+q takes hc_hidden(pos0+q-1) = row q-1 and token q.
                    // Rows above the accept point are overwritten by the next chain.
                    let hcd = hc * hh;
                    for mi in ctx.range_local(m) {
                        bw.res_keep[mi * hcd..(mi + 1) * hcd].copy_from_slice(&bw.res[mi * hcd..(mi + 1) * hcd]);
                    }
                    ctx.node_barrier();
                    if m > 1 {
                        let ins: Vec<crate::exec_mtp::MtpIn> = (0..m - 1).map(crate::exec_mtp::MtpIn::Row).collect();
                        self.mtp_rows(ctx, bw, st, cs, &toks[1..], &ins, pos0 + 1, None, false, &[], None);
                    }
                }
            }
        }
    }


    /// One decoder layer over `m` rows of `bw.res` (hyper-connection mix, token mixer, mix, MoE,
    /// combines), positions `pos0..`; `ckpt` = speculative verify (recurrent state checkpointed per row).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn layer_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, st: &mut crate::state::TileState, cs: &mut CoreScratch, lw: &crate::exec::LayerW, il: usize, m: usize, pos0: usize, t: usize, toks: &[u32], prev: &[u32], ckpt: bool) {
        let cfg = &self.cfg;
        let (hh, hc, tiles) = (cfg.hidden, cfg.hc, cfg.n_tiles);
        let hl = hh / tiles;
        let lay = &self.mbox.layout;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }
        let nan_check = std::env::var("TR_NAN_CHECK").is_ok();
        macro_rules! nan { ($name:expr, $v:expr) => { if nan_check && ctx.global == 0 { let v: &[f32] = $v; if let Some(i) = v.iter().position(|x| !x.is_finite()) { eprintln!("NaN: {} pos0 {pos0} m {m} idx {i} val {}", $name, v[i]); } } } }
        if let Some(ple) = &lw.ple {
            self.ple_batch(ctx, bw, st, cs, ple, toks, prev, m, ckpt);
        }
        lap!("b.ple");
        self.hc_a_batch(ctx, bw, &lw.hc_attn, m, t);
        lap!("b.hc_a");
        ctx.barrier(); // A
        lap!("b.barA");
        self.hc_b_batch(ctx, bw, &lw.hc_attn, m, t, false, None);
        lap!("b.hc_b");
        ctx.barrier(); // B
        lap!("b.barB");
        self.gather_rows(ctx, lay.mixed, hl, m, bw.mixed);
        lap!("b.xchg.gather");
        nan!(format!("layer {il} mixed(attn)"), &bw.mixed[..m * hh]);
        let tw = std::time::Instant::now();
        if let Some(g) = &lw.gdn {
            self.gdn_batch(ctx, bw, st, g, il, m, t, ckpt);
            lap!("b.gdn");
            if self.profile.enabled {
                self.profile.pw_add(ctx, crate::exec::PW_GDN, tw);
            }
        } else {
            self.attn_batch(ctx, bw, st, cs, lw.attn.as_ref().unwrap(), il, m, pos0, t);
            lap!("b.attn");
            if self.profile.enabled {
                self.profile.pw_add(ctx, crate::exec::PW_ATTN, tw);
            }
        }
        ctx.barrier(); // C
        lap!("b.barC");
        self.reduce_combine(ctx, bw, m, false);
        lap!("b.xchg.reduce");
        nan!(format!("layer {il} res(after gdn/attn)"), &bw.res[..m * hc * hh]);
        self.hc_a_batch(ctx, bw, &lw.hc_ffn, m, t);
        lap!("b.hc_a");
        ctx.barrier(); // D
        lap!("b.barD");
        self.hc_b_batch(ctx, bw, &lw.hc_ffn, m, t, true, Some(lw.moe.router));
        lap!("b.hc_b");
        ctx.barrier(); // E
        lap!("b.barE");
        self.gather_rows(ctx, lay.mixed, hl, m, bw.mixed);
        lap!("b.xchg.gather");
        {
            let rlog: &'static mut [f32] = unsafe { std::slice::from_raw_parts_mut(bw.rlog.as_mut_ptr(), bw.rlog.len()) };
            self.allreduce_rows(ctx, bw, lay.rlog, cfg.n_expert, m, rlog);
        }
        lap!("b.xchg.rlog");
        nan!(format!("layer {il} mixed(ffn)"), &bw.mixed[..m * hh]);
        nan!(format!("layer {il} rlog"), &bw.rlog[..m * cfg.n_expert]);
        let tw = std::time::Instant::now();
        self.moe_batch(ctx, bw, cs, &lw.moe, m, t);
        lap!("b.moe");
        if self.profile.enabled {
            self.profile.pw_add(ctx, crate::exec::PW_MOE, tw);
        }
        ctx.barrier(); // G
        lap!("b.barG");
        self.reduce_combine(ctx, bw, m, true);
        lap!("b.xchg.reduce.moe");
        nan!(format!("layer {il} res(after moe)"), &bw.res[..m * hc * hh]);
    
    }

    /// Final hyper-connection mix + lm_head for every row of the batch; logits land in rows 0..m
    /// of the LOGITS mailbox of every tile. Ends with a global barrier.
    pub(crate) fn head_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, m: usize, t: usize) {
        let cfg = &self.cfg;
        let (hh, tiles) = (cfg.hidden, cfg.n_tiles);
        let hl = hh / tiles;
        let hw = &self.heads[t];
        let lay = &self.mbox.layout;
        assert!(m <= lay.logits_rows, "verify batch {m} > logits rows {}", lay.logits_rows);
        self.hc_a_batch(ctx, bw, &hw.hc, m, t);
        ctx.barrier();
        self.hc_b_batch(ctx, bw, &hw.hc, m, t, false, None);
        ctx.barrier();
        self.gather_rows(ctx, lay.mixed, hl, m, bw.mixed);
        for mi in ctx.range_local(m) {
            self.prep_rows(bw.mixed, hh, mi, mi + 1, bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m);
        }
        ctx.node_barrier();
        let act = self.act(bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m, hh);
        let r = ctx.range_local(hw.output.n_strips());
        let out = self.mbox.slot_rows(t, lay.logits, m);
        unsafe {
            self.dense_gemm(amx_buf!(bw, ctx), &hw.output, r.start, r.end, &act, out.as_mut_ptr(), self.vocab_per_tile);
        }
        ctx.barrier();
    }

    /// All-gather of a per-token slot: dst[mi][u*len..] = tile u's row mi. Tasks are (tile, row)
    /// so each core streams contiguous rows from one remote mailbox; ends with a node barrier.
    fn gather_rows(&self, ctx: &WorkerCtx, s: Slot, len: usize, m: usize, dst: &mut [f32]) {
        let n = ctx.n_nodes;
        for task in ctx.range_local(n * m) {
            let (u, mi) = (task / m, task % m);
            let src = self.mbox.slot_ro(u, MailboxLayout::row(s, mi));
            dst[mi * n * len + u * len..mi * n * len + (u + 1) * len].copy_from_slice(&src[..len]);
        }
        ctx.node_barrier();
    }

    /// Reduce-scatter of an all-reduce over per-token rows, hierarchical over sockets. The slow
    /// path on this box is the cross-socket link (a core streams ~4x slower from the other
    /// socket), so with n tiles in S sockets of `tps` tiles, tile t = (socket sig, index lam) owns
    /// the S column slices {lam + j*tps} of width w and:
    ///   1. sums them over its own socket's tiles into `rsum` (same-socket reads only),
    ///   2. adds the other sockets' partials of the same slices (tiles with the same lam) into `rfull`.
    /// Cross-socket bytes per tile: (S-1)*S*w*m*4 instead of (n/2)*w*m*4 for a flat reduce-scatter,
    /// and the all-gather afterwards reads same-socket tiles only. `src(u, slice, mi)` yields tile
    /// u's row mi of column slice `slice`. Ends with a global barrier.
    #[allow(clippy::too_many_arguments)]
    fn reduce_scatter(&self, ctx: &WorkerCtx, bw: &mut BatchWs, src: impl Fn(usize, usize, usize) -> &'static [f32], w: usize, m: usize, prof: bool, tm: &mut std::time::Instant, moe: bool) {
        let n = ctx.n_nodes;
        let tps = self.tiles_per_socket;
        let ns = n / tps;
        let t = ctx.node;
        let (sig, lam) = (t / tps, t % tps);
        let lay = &self.mbox.layout;
        assert!(w % 16 == 0 && ns * w <= lay.rsum.len, "reduce_scatter: {ns} slices of {w} do not fit rsum");
        let stage = unsafe { stage_buf!(bw, ctx) };
        let rows = ctx.range_local(m);
        // 1. own-socket sums of the S slices, starting from a different source per core
        for mi in rows.clone() {
            for j in 0..ns {
                let sl = lam + j * tps;
                let acc = &mut stage[j * w..(j + 1) * w];
                for k in 0..tps {
                    let u = sig * tps + (k + if self.segments.is_empty() { ctx.local } else { 0 }) % tps;
                    if k == 0 {
                        acc.copy_from_slice(&src(u, sl, mi)[..w]);
                    } else {
                        elem::add_inplace(acc, &src(u, sl, mi)[..w]);
                    }
                }
            }
            put_row(&mut self.mbox.slot(t, MailboxLayout::row(lay.rsum, mi))[..ns * w], &stage[..ns * w]);
        }
        elem::store_fence();
        if prof {
            self.profile.lap(tm, if moe { "b.xchg.scatter.moe" } else { "b.xchg.scatter.mix" });
        }
        ctx.barrier();
        if prof {
            self.profile.lap(tm, "b.xchg.sbar");
        }
        if ns == 1 {
            // single socket: rsum already holds the full sums; expose them as rfull
            for mi in rows.clone() {
                let own = self.mbox.slot_ro(t, MailboxLayout::row(lay.rsum, mi));
                put_row(&mut self.mbox.slot(t, MailboxLayout::row(lay.rfull, mi))[..w], &own[..w]);
            }
            elem::store_fence();
            ctx.barrier();
            return;
        }
        // 2. add the other sockets' partials of the same slices (their tile with the same lam)
        for mi in rows.clone() {
            let acc = &mut stage[..ns * w];
            acc.copy_from_slice(&self.mbox.slot_ro(t, MailboxLayout::row(lay.rsum, mi))[..ns * w]);
            for d in 1..ns {
                let u = ((sig + d) % ns) * tps + lam;
                elem::add_inplace(acc, &self.mbox.slot_ro(u, MailboxLayout::row(lay.rsum, mi))[..ns * w]);
            }
            put_row(&mut self.mbox.slot(t, MailboxLayout::row(lay.rfull, mi))[..ns * w], acc);
        }
        elem::store_fence();
        if prof {
            self.profile.lap(tm, "b.xchg.xsock");
        }
        ctx.barrier();
        if prof {
            self.profile.lap(tm, "b.xchg.xbar");
        }
    }

    /// Slices held by same-socket tile u after `reduce_scatter`: (slice index, offset in its rfull row).
    fn socket_slices(&self, n: usize, u: usize) -> impl Iterator<Item = (usize, usize)> {
        let tps = self.tiles_per_socket;
        let ns = n / tps;
        let lam = u % tps;
        (0..ns).map(move |j| (lam + j * tps, j))
    }

    /// All-reduce over tiles of a per-token slot into `dst` [m][len] (reduce-scatter + all-gather).
    pub(crate) fn allreduce_rows(&self, ctx: &WorkerCtx, bw: &mut BatchWs, s: Slot, len: usize, m: usize, dst: &mut [f32]) {
        let mut tm = std::time::Instant::now();
        let n = ctx.n_nodes;
        let w = len / n;
        let mb = &self.mbox;
        self.reduce_scatter(ctx, bw, move |u, sl, mi| &mb.slot_ro(u, MailboxLayout::row(s, mi))[sl * w..(sl + 1) * w], w, m, false, &mut tm, false);
        let tps = self.tiles_per_socket;
        let sig = ctx.node / tps;
        for task in ctx.range_local(tps * m) {
            let (k, mi) = (task / m, task % m);
            let u = sig * tps + k;
            let row = mb.slot_ro(u, MailboxLayout::row(mb.layout.rfull, mi));
            for (sl, j) in self.socket_slices(n, u) {
                dst[mi * len + sl * w..mi * len + (sl + 1) * w].copy_from_slice(&row[j * w..(j + 1) * w]);
            }
        }
        ctx.node_barrier();
    }

    /// res[m][s] += sum_tiles(part[m]) * 2*sigmoid(inject[m][s]/hc): hierarchical reduce-scatter,
    /// then the all-gather from same-socket tiles with the combine fused in (the reduced row is
    /// never materialised).
    fn reduce_combine(&self, ctx: &WorkerCtx, bw: &mut BatchWs, m: usize, moe: bool) {
        let (hh, hc) = (self.cfg.hidden, self.cfg.hc);
        let n = ctx.n_nodes;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        let mb = &self.mbox;
        let pw = mb.layout.part_w;
        self.reduce_scatter(ctx, bw, move |u, sl, mi| &mb.slot_ro(u, mb.layout.part_block(sl))[mi * pw..(mi + 1) * pw], pw, m, prof, &mut tm, moe);
        let tps = self.tiles_per_socket;
        let sig = ctx.node / tps;
        for mi in ctx.range_local(m) {
            let mut g = [0f32; 8];
            for s in 0..hc {
                g[s] = 2.0 * sigmoid(bw.inject[mi * 8 + s] / hc as f32);
            }
            for k in 0..tps {
                let u = sig * tps + (k + if self.segments.is_empty() { ctx.local } else { 0 }) % tps;
                let row = mb.slot_ro(u, MailboxLayout::row(mb.layout.rfull, mi));
                for (sl, j) in self.socket_slices(n, u) {
                    let red = &row[j * pw..(j + 1) * pw];
                    for s in 0..hc {
                        let base = mi * hc * hh + s * hh + sl * pw;
                        elem::axpy(&mut bw.res[base..base + pw], red, g[s]);
                    }
                }
            }
        }
        ctx.node_barrier();
    }

    fn hc_a_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, w: &HcW, m: usize, t: usize) {
        let cfg = &self.cfg;
        let (hh, hc) = (cfg.hidden, cfg.hc);
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }
        // per (token, stream) rmsnorm * gamma
        for task in ctx.range_local(m * hc) {
            let (mi, s) = (task / hc, task % hc);
            let base = mi * hc * hh + s * hh;
            rmsnorm(&bw.res[base..base + hh], Some(&w.norm[s * hh..(s + 1) * hh]), cfg.rms_eps, &mut bw.xn[base..base + hh]);
        }
        lap!("b.hc_a.norm");
        ctx.node_barrier();
        lap!("b.hc_a.wait1");
        // per token: inject + quantise this tile's K-slice
        let kslice = hc * hh / cfg.n_tiles;
        let ks0 = t * kslice;
        let rows = ctx.range_local(m);
        let small = m < ctx.cores_per_node;
        if let (Some(inj), true) = (w.inject, small) {
            // few rows: every core takes a K range of all rows (partials in `stage`, reduced after the barrier)
            let kr = ctx.range_local(hc * hh / 32);
            let (k0, k1) = (kr.start * 32, kr.end * 32);
            let stage = unsafe { stage_buf!(bw, ctx) };
            for mi in 0..m {
                let x = &bw.xn[mi * hc * hh..(mi + 1) * hc * hh];
                for s in 0..hc {
                    stage[mi * 8 + s] = if k0 < k1 { dot(&inj[s * hc * hh + k0..s * hc * hh + k1], &x[k0..k1]) } else { 0.0 };
                }
            }
        } else if let (Some(inj), false) = (w.inject, rows.is_empty()) {
            // inject[rows][hc] = xn[rows][hc*hh] . inj[hc][hc*hh]^T (4x4 register blocks: xn read once per 4 rows)
            let x = &bw.xn[rows.start * hc * hh..rows.end * hc * hh];
            let y = &mut bw.inject[rows.start * 8..rows.end * 8];
            tr_kernels::smallgemm::gemm_f32_nt(x, hc * hh, rows.len(), inj, hc * hh, hc, hc * hh, y, 8);
        }
        for mi in rows {
            let xn = &bw.xn[mi * hc * hh..(mi + 1) * hc * hh];
            if self.use_amx(kslice, m) {
                amx::rows_to_bf16(&xn[ks0..ks0 + kslice], kslice, 0, 1, &mut bw.xq_h[mi * kslice..(mi + 1) * kslice]);
            } else {
                let nb = kslice / 32;
                for b in 0..nb {
                    let (sc, su) = unsafe { quant_block(&xn[ks0 + b * 32..ks0 + (b + 1) * 32], &mut bw.xq_q[mi * kslice + b * 32..mi * kslice + (b + 1) * 32]) };
                    bw.xq_s[mi * nb + b] = sc;
                    bw.xq_sum[mi * nb + b] = su;
                }
            }
        }
        lap!("b.hc_a.inj");
        ctx.node_barrier();
        lap!("b.hc_a.wait2");
        if w.inject.is_some() && small {
            // reduce the per-core inject partials (every core, private result rows are tile-shared)
            for mi in ctx.range_local(m) {
                for s in 0..hc {
                    let mut acc = 0f32;
                    for cc in 0..ctx.cores_per_node {
                        acc += bw.stage[cc * bw.stage_per_core + mi * 8 + s];
                    }
                    bw.inject[mi * 8 + s] = acc;
                }
            }
            ctx.node_barrier();
        }
        let sr = ctx.range_local(w.down.n_strips());
        let lo = self.mbox.layout.lo;
        let mb = &self.mbox;
        let act = self.act(bw.xq_q, bw.xq_s, bw.xq_sum, bw.xq_h, m, kslice);
        unsafe {
            self.dense_gemm_stream(amx_buf!(bw, ctx), stage_buf!(bw, ctx), &w.down, sr.start, sr.end, &act, m, &mut |mi, c, vals| {
                put_row(&mut mb.slot(t, MailboxLayout::row(lo, mi))[c..c + vals.len()], vals);
            }, if prof { Some(&mut tm) } else { None });
        }
        lap!("b.hc_a.down");
    }

    fn hc_b_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, w: &HcW, m: usize, t: usize, _ffn: bool, router: Option<&[f32]>) {
        let cfg = &self.cfg;
        let (hh, hc, tiles) = (cfg.hidden, cfg.hc, cfg.n_tiles);
        let lr = cfg.hc_lr;
        let lay = &self.mbox.layout;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }
        // reduce lo over tiles (per token), silu(lo/hc), quantise
        // A fixed reduction order for concurrent rows keeps identical fork inputs from
        // acquiring different quantization rounding solely through worker assignment.
        let u0 = t * ctx.cores_per_node + if self.segments.is_empty() { ctx.local } else { 0 };
        for mi in ctx.range_local(m) {
            let dst = &mut bw.lo[mi * lr..(mi + 1) * lr];
            dst.copy_from_slice(&self.mbox.slot_ro(u0 % tiles, MailboxLayout::row(lay.lo, mi))[..lr]);
            for i in 1..tiles {
                elem::add_inplace(dst, &self.mbox.slot_ro((u0 + i) % tiles, MailboxLayout::row(lay.lo, mi))[..lr]);
            }
            elem::scale_inplace(dst, 1.0 / hc as f32);
            elem::silu_inplace(dst);
        }
        lap!("b.hc_b.lo");
        ctx.node_barrier();
        for mi in ctx.range_local(m) {
            self.prep_rows(bw.lo, lr, mi, mi + 1, bw.lo_q, bw.lo_s, bw.lo_sum, bw.lo_h, m);
        }
        ctx.node_barrier();
        lap!("b.hc_b.wait1");
        let act = self.act(bw.lo_q, bw.lo_s, bw.lo_sum, bw.lo_h, m, lr);
        let gl = hc * hh / tiles;
        let r = ctx.range_local(w.up.n_strips());
        unsafe {
            self.dense_gemm(amx_buf!(bw, ctx), &w.up, r.start, r.end, &act, bw.gate_local.as_mut_ptr(), gl);
        }
        lap!("b.hc_b.up");
        for mi in 0..m {
            elem::sigmoid_inplace(&mut bw.gate_local[mi * gl + r.start * 16..mi * gl + (r.end * 16).min(w.up.rows)]);
        }
        lap!("b.hc_b.sig");
        ctx.node_barrier();
        lap!("b.hc_b.wait2");
        let hl = hh / tiles;
        let inv = 1.0 / hc as f32;
        let rows = ctx.range_local(m);
        let stage = unsafe { stage_buf!(bw, ctx) };
        // mixed rows for this core's tokens: computed in the private stage (rows.len() x hl), then
        // streamed into the `mixed` slot every tile gathers from
        let mixed = &mut stage[..rows.len() * hl];
        for mi in rows.clone() {
            let out = &mut mixed[(mi - rows.start) * hl..(mi - rows.start + 1) * hl];
            let xn = &bw.xn[mi * hc * hh + t * hl..];
            let gate = &bw.gate_local[mi * gl..(mi + 1) * gl];
            elem::mul_vec(&xn[..hl], &gate[..hl], out);
            for s in 1..hc {
                elem::fma_inplace(out, &xn[s * hh..s * hh + hl], &gate[s * hl..(s + 1) * hl]);
            }
            elem::scale_inplace(out, inv);
            put_row(&mut self.mbox.slot(t, MailboxLayout::row(lay.mixed, mi))[..hl], out);
        }
        lap!("b.hc_b.mixonly");
        if let (Some(router), true) = (router, m < ctx.cores_per_node) {
            // few rows: every core computes its expert range for all rows straight from the private
            // mixed rows of... the other cores' rows are not visible yet; use the tile-shared xn/gate
            // recomputation instead: mixed row mi = sum_s xn[mi][s][t*hl..] * gate[mi][s] / hc
            let ne = cfg.n_expert;
            let er = ctx.range_local(ne);
            let mut row = vec![0f32; hl];
            for mi in 0..m {
                let xn = &bw.xn[mi * hc * hh + t * hl..];
                let gate = &bw.gate_local[mi * gl..(mi + 1) * gl];
                elem::mul_vec(&xn[..hl], &gate[..hl], &mut row);
                for s in 1..hc {
                    elem::fma_inplace(&mut row, &xn[s * hh..s * hh + hl], &gate[s * hl..(s + 1) * hl]);
                }
                elem::scale_inplace(&mut row, inv);
                let out = self.mbox.slot(t, MailboxLayout::row(lay.rlog, mi));
                for e in er.clone() {
                    out[e] = dot(&router[e * hl..(e + 1) * hl], &row);
                }
            }
        } else if let (Some(router), false) = (router, rows.is_empty()) {
            // router logits for this core's rows: [rows][n_expert] = mixed[rows][hl] . router[n_expert][hl]^T,
            // in the stage behind the mixed rows, then streamed into `rlog`
            let (x, y) = stage.split_at_mut(rows.len() * hl);
            let ne = cfg.n_expert;
            let y = &mut y[..rows.len() * ne];
            tr_kernels::smallgemm::gemm_f32_nt(x, hl, rows.len(), router, hl, ne, hl, y, ne);
            for mi in rows.clone() {
                put_row(&mut self.mbox.slot(t, MailboxLayout::row(lay.rlog, mi))[..ne], &y[(mi - rows.start) * ne..(mi - rows.start + 1) * ne]);
            }
        }
        elem::store_fence();
        lap!("b.hc_b.mix");
    }

    /// `ckpt`: speculative verify — the live state is only read; the state after row i lands in
    /// `st.ckpt[i]` (the recurrence runs row by row with the decode kernel), so `commit(j)` can
    /// adopt exactly the accepted prefix.
    #[allow(clippy::too_many_arguments)]
    fn gdn_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, st: &mut crate::state::TileState, g: &GdnW, il: usize, m: usize, t: usize, ckpt: bool) {
        let cfg = &self.cfg;
        let tiles = cfg.n_tiles;
        let dk = cfg.d_state;
        let kh = cfg.n_k_heads / tiles;
        let vh = cfg.n_v_heads / tiles;
        let c_local = (2 * kh + vh) * dk;
        let v_off = 2 * kh * dk;
        let vl = vh * dk;
        let hh = cfg.hidden;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }
        let pw_on = self.profile.enabled;
        let mut pwt = std::time::Instant::now();
        #[allow(unused_assignments)]
        macro_rules! pw { ($ph:expr) => { if pw_on { self.profile.pw_add(ctx, $ph, pwt); pwt = std::time::Instant::now(); } } }
        for mi in ctx.range_local(m) {
            self.prep_rows(bw.mixed, hh, mi, mi + 1, bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m);
        }
        ctx.node_barrier();
        let act = self.act(bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m, hh);
        let (ns_qkv, ns_z) = (g.qkv.n_strips(), g.z.n_strips());
        for (i, lo, hi) in sub_ranges(ctx.range_local(ns_qkv + ns_z), &[ns_qkv, ns_z]) {
            unsafe {
                if i == 0 {
                    self.dense_gemm(amx_buf!(bw, ctx), &g.qkv, lo, hi, &act, bw.qkv.as_mut_ptr(), c_local);
                } else {
                    self.dense_gemm(amx_buf!(bw, ctx), &g.z, lo, hi, &act, bw.z.as_mut_ptr(), vl);
                }
            }
        }
        for task in ctx.range_local(m * vh) {
            let (mi, h) = (task / vh, task % vh);
            let x = &bw.mixed[mi * hh..(mi + 1) * hh];
            bw.beta[mi * 8 + h] = dot(&g.beta[h * hh..(h + 1) * hh], x);
            bw.alpha[mi * 8 + h] = dot(&g.alpha[h * hh..(h + 1) * hh], x);
        }
        lap!("b.gdn.proj");
        pw!(crate::exec::PW_GDN_PROJ);
        ctx.node_barrier();
        lap!("b.gdn.wait1");
        pw!(crate::exec::PW_GDN_W1);
        // conv over the batch: 16-channel groups split across cores, sequential in time
        let kc = cfg.d_conv;
        let (gdn_live, ckpts) = (&mut st.gdn, &mut st.ckpt);
        let gs = gdn_live[il].as_mut().unwrap();
        if ckpt {
            assert!(m <= ckpts.len(), "verify batch {m} > checkpoint slots {}", ckpts.len());
            // the kernel runs on slot m-1's history (a copy of the live one, which stays untouched);
            // slot i < m-1 gets the last kc-1 rows of [live history ; raw rows 0..=i]
            let rows_ = ctx.range_local(c_local / 16);
            let (r0, r1) = (rows_.start * 16, rows_.end * 16);
            for i in 0..m {
                let dst = ckpts[i].gdn[il].as_mut().unwrap();
                for r in 0..kc - 1 {
                    let src_row = i + 1 + r; // row of [live(kc-1 rows); x(m rows)]
                    let src: &[f32] = if src_row < kc - 1 { &gs.conv[src_row * c_local..(src_row + 1) * c_local] } else { &bw.qkv[(src_row - (kc - 1)) * c_local..(src_row - (kc - 1) + 1) * c_local] };
                    dst.conv[r * c_local + r0..r * c_local + r1].copy_from_slice(&src[r0..r1]);
                }
            }
            ctx.node_barrier();
            let work = ckpts[m - 1].gdn[il].as_mut().unwrap();
            for grp in rows_ {
                // the kernel's initial history must be the live one: restore this group's slice first
                for r in 0..kc - 1 {
                    work.conv[r * c_local + grp * 16..r * c_local + grp * 16 + 16].copy_from_slice(&gs.conv[r * c_local + grp * 16..r * c_local + grp * 16 + 16]);
                }
                tr_kernels::conv::conv_silu_seq16(bw.qkv, c_local, m, g.conv, kc, work.conv, c_local, grp * 16, bw.qkvc, c_local);
            }
        } else {
            let segments = self.segment_ranges(m);
            for (start, len) in segments {
                let state = unsafe { self.row_state(t, start) }.gdn[il].as_mut().unwrap();
                for grp in ctx.range_local(c_local / 16) {
                    tr_kernels::conv::conv_silu_seq16(&bw.qkv[start * c_local..], c_local, len, g.conv, kc, state.conv, c_local, grp * 16, &mut bw.qkvc[start * c_local..], c_local);
                }
            }
        }
        lap!("b.gdn.conv");
        pw!(crate::exec::PW_GDN_CONV);
        ctx.node_barrier();
        lap!("b.gdn.wait2");
        // per token: l2-normalised q/k per key head, gates per value head (once, not per task)
        let kl = kh * dk;
        for mi in ctx.range_local(m) {
            let row = &bw.qkvc[mi * c_local..(mi + 1) * c_local];
            for khl in 0..kh {
                elem::l2norm(&row[khl * dk..(khl + 1) * dk], cfg.rms_eps, &mut bw.gq[mi * kl + khl * dk..mi * kl + (khl + 1) * dk]);
                elem::l2norm(&row[kh * dk + khl * dk..kh * dk + (khl + 1) * dk], cfg.rms_eps, &mut bw.gk[mi * kl + khl * dk..mi * kl + (khl + 1) * dk]);
            }
            for h in 0..vh {
                let hg = kh * t + (h % kh) + cfg.n_k_heads * (h / kh);
                bw.gate[h * m + mi] = g.a[hg] * softplus(bw.alpha[mi * 8 + h] + g.dt[hg]);
                bw.bet[h * m + mi] = sigmoid(bw.beta[mi * 8 + h]);
            }
        }
        lap!("b.gdn.prep");
        ctx.node_barrier();
        lap!("b.gdn.wait2b");
        // delta rule: tasks (head, 16-col chunk) on chunk-major state, fused sweep over the tokens
        let chunks = dk / 16;
        let scale = (dk as f32).powf(-0.5);
        for task in ctx.range_local(vh * chunks) {
            let h = task / chunks;
            let ch = task % chunks;
            let khl = h % kh;
            let (c0, c1) = ((h * chunks + ch) * dk * 16, (h * chunks + ch + 1) * dk * 16);
            if ckpt {
                // row by row with the decode kernel: slot i = f(slot i-1, row i), the live state only read
                for i in 0..m {
                    let (src, dst) = if i == 0 {
                        (&gs.ssm[c0..c1], &mut ckpts[0].gdn[il].as_mut().unwrap().ssm[c0..c1])
                    } else {
                        let (a, b) = ckpts.split_at_mut(i);
                        (&a[i - 1].gdn[il].as_ref().unwrap().ssm[c0..c1], &mut b[0].gdn[il].as_mut().unwrap().ssm[c0..c1])
                    };
                    dst.copy_from_slice(src);
                    tr_kernels::gdn::gdn_step16(dst, dk, &bw.gq[i * kl + khl * dk..i * kl + (khl + 1) * dk], &bw.gk[i * kl + khl * dk..i * kl + (khl + 1) * dk], &bw.qkvc[i * c_local + v_off + h * dk + ch * 16..i * c_local + v_off + h * dk + ch * 16 + 16], bw.gate[h * m + i], bw.bet[h * m + i], scale, &mut bw.y[i * vl + h * dk + ch * 16..i * vl + h * dk + ch * 16 + 16]);
                }
            } else {
                let segments = self.segment_ranges(m);
                for (start, len) in segments {
                    let gs = unsafe { self.row_state(t, start) }.gdn[il].as_mut().unwrap();
                    tr_kernels::gdn::gdn_chunk_seq(
                        &mut gs.ssm[c0..c1], dk, len,
                        &bw.gq[start * kl + khl * dk..], kl,
                        &bw.gk[start * kl + khl * dk..], kl,
                        &bw.qkvc[start * c_local + v_off + h * dk + ch * 16..], c_local,
                        &bw.gate[h * m + start..h * m + start + len], &bw.bet[h * m + start..h * m + start + len], scale,
                        &mut bw.y[start * vl + h * dk + ch * 16..], vl,
                    );
                }
            }
        }
        lap!("b.gdn.delta");
        pw!(crate::exec::PW_GDN_DELTA);
        ctx.node_barrier();
        lap!("b.gdn.wait3");
        // gated rmsnorm per (token, head); quantise; out-proj partial
        let mut nrm = vec![0f32; dk];
        for task in ctx.range_local(m * vh) {
            let (mi, h) = (task / vh, task % vh);
            let o = mi * vl + h * dk;
            rmsnorm(&bw.y[o..o + dk], Some(g.norm), cfg.rms_eps, &mut nrm);
            elem::sigmoid_vec(&bw.z[o..o + dk], &mut bw.vv[o..o + dk]);
            elem::mul_inplace(&mut bw.vv[o..o + dk], &nrm);
        }
        ctx.node_barrier();
        for mi in ctx.range_local(m) {
            self.prep_rows(bw.vv, vl, mi, mi + 1, bw.vv_q, bw.vv_s, bw.vv_sum, bw.vv_h, m);
        }
        lap!("b.gdn.norm");
        ctx.node_barrier();
        lap!("b.gdn.wait4");
        let act = self.act(bw.vv_q, bw.vv_s, bw.vv_sum, bw.vv_h, m, vl);
        let r = ctx.range_local(g.out.n_strips());
        unsafe {
            self.dense_gemm_part(ctx, amx_buf!(bw, ctx), stage_buf!(bw, ctx), &g.out, r.start, r.end, &act, m);
        }
        lap!("b.gdn.out");
        pw!(crate::exec::PW_GDN_OUT);
        let _ = pwt;
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, st: &mut crate::state::TileState, cs: &mut CoreScratch, a: &crate::exec::AttnW, il: usize, m: usize, pos0: usize, t: usize) {
        let cfg = &self.cfg;
        let hd = cfg.head_dim;
        let qh = cfg.n_head / cfg.n_tiles;
        let tiles = cfg.n_tiles;
        let hh = cfg.hidden;
        let ql = qh * 2 * hd;
        let ol = qh * hd;
        let lay = &self.mbox.layout;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }
        if ctx.local == 0 {
            bw.task_ctr.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        for mi in ctx.range_local(m) {
            self.prep_rows(bw.mixed, hh, mi, mi + 1, bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m);
        }
        ctx.node_barrier();
        let act = self.act(bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m, hh);
        let (nq, nk, nv) = (a.q.n_strips(), a.k.n_strips(), a.v.n_strips());
        let (niq, nik) = a.idx.map(|i| (i.q.n_strips(), i.k.n_strips())).unwrap_or((0, 0));
        let mb = &self.mbox;
        let idx = lay.idx;
        for (i, lo, hi) in sub_ranges(ctx.range_local(nq + nk + nv + niq + nik), &[nq, nk, nv, niq, nik]) {
            unsafe {
                match i {
                    0 => self.dense_gemm(amx_buf!(bw, ctx), &a.q, lo, hi, &act, bw.qg.as_mut_ptr(), ql),
                    1 => self.dense_gemm(amx_buf!(bw, ctx), &a.k, lo, hi, &act, bw.kn.as_mut_ptr(), hd),
                    2 => self.dense_gemm(amx_buf!(bw, ctx), &a.v, lo, hi, &act, bw.vn.as_mut_ptr(), hd),
                    3 => self.dense_gemm_stream(amx_buf!(bw, ctx), stage_buf!(bw, ctx), &a.idx.as_ref().unwrap().q, lo, hi, &act, m, &mut |mi, c, vals| {
                        put_row(&mut mb.slot(t, MailboxLayout::row(idx, mi))[c..c + vals.len()], vals);
                    }, None),
                    _ => self.dense_gemm_stream(amx_buf!(bw, ctx), stage_buf!(bw, ctx), &a.idx.as_ref().unwrap().k, lo, hi, &act, m, &mut |mi, c, vals| {
                        let c = niq * 16 + c;
                        put_row(&mut mb.slot(t, MailboxLayout::row(idx, mi))[c..c + vals.len()], vals);
                    }, None),
                }
            }
        }
        ctx.node_barrier();
        lap!("b.attn.proj");
        let ast_for = |mi: usize| -> &mut crate::state::AttnState {
            let st = unsafe { self.row_state(t, mi) };
            if il == cfg.mtp_layer() { st.mtp_attn.as_mut().unwrap() } else { st.attn[il].as_mut().unwrap() }
        };
        let _ = st; // state selection is per row (the legacy path resolves to self.state)
        let last_pos = (0..m).map(|mi| self.row_position(mi, pos0)).max().unwrap();
        let pos3: Vec<[u32; 3]> = (0..m).map(|mi| [bw.pos3[3 * mi], bw.pos3[3 * mi + 1], bw.pos3[3 * mi + 2]]).collect();
        for mi in ctx.range_local(m) {
            let pos = self.row_position(mi, pos0);
            let ast = ast_for(mi);
            let kdst = &mut cs.head_tmp[..hd];
            rmsnorm(&bw.kn[mi * hd..(mi + 1) * hd], Some(a.k_norm), cfg.rms_eps, kdst);
            self.rope.apply3(kdst, pos3[mi]);
            ast.store(pos, kdst, &bw.vn[mi * hd..(mi + 1) * hd]);
        }
        // ---- QSA indexer (see attn_block): gather, per-token q, sequential block keys, scores
        let ratio = cfg.qsa_ratio(il);
        let (nh, d) = (cfg.idx_heads, cfg.idx_dim);
        let any_sparse = a.idx.is_some() && ratio > 0 && !self.qsa_off && qsa::is_sparse(last_pos, ratio, cfg.idx_top_k);
        if let (Some(iw), true) = (&a.idx, ratio > 0) {
            let (per_q, per_k) = (nh * d / tiles, d / tiles);
            let per = per_q + per_k;
            lap!("b.attn.kv");
            ctx.barrier(); // I
            lap!("b.attn.barI");
            for i in ctx.range_local(m * tiles * per) {
                let mi = i / (tiles * per);
                let rem = i % (tiles * per);
                let (u, j) = (rem / per, rem % per);
                let v = self.mbox.slot_ro(u, MailboxLayout::row(lay.idx, mi))[j];
                if j < per_q {
                    bw.idx_qraw[mi * nh * d + u * per_q + j] = v;
                } else {
                    bw.idx_kraw[mi * d + u * per_k + j - per_q] = v;
                }
            }
            ctx.node_barrier();
            lap!("b.attn.gather");
            for task in ctx.range_local(m * nh) {
                let (mi, h) = (task / nh, task % nh);
                let base = mi * nh * d + h * d;
                let q = &mut bw.idx_q[base..base + d];
                rmsnorm(&bw.idx_qraw[base..base + d], Some(iw.q_norm), cfg.rms_eps, q);
                self.rope.apply3(q, pos3[mi]);
            }
            if ctx.local == 0 {
                for mi in 0..m {
                    let pos = self.row_position(mi, pos0);
                    let ast = ast_for(mi);
                    ast.ik_raw[(pos % ratio) * d..(pos % ratio + 1) * d].copy_from_slice(&bw.idx_kraw[mi * d..(mi + 1) * d]);
                    if (pos + 1) % ratio == 0 {
                        let b = pos / ratio;
                        let tmp = &mut cs.head_tmp[..d];
                        tmp.fill(0.0);
                        for i in 0..ratio {
                            elem::axpy(tmp, &ast.ik_raw[i * d..(i + 1) * d], 1.0 / ratio as f32);
                        }
                        let pk = ast.pooled_row_mut(b, ratio, d);
                        rmsnorm(tmp, Some(iw.k_norm), cfg.rms_eps, pk);
                        self.rope.apply(pk, (b * ratio) as u32);
                    }
                }
            }
            ctx.node_barrier();
            lap!("b.attn.idx");
            if any_sparse {
                const CH: usize = 64;
                let nb_last = qsa::n_complete_blocks(last_pos, ratio);
                let nch = nb_last.div_ceil(CH);
                for task in ctx.range_local(m * nch) {
                    let (mi, c) = (task / nch, task % nch);
                    let pos = self.row_position(mi, pos0);
                    let ast = ast_for(mi);
                    if !qsa::is_sparse(pos, ratio, cfg.idx_top_k) {
                        continue;
                    }
                    let nb = qsa::n_complete_blocks(pos, ratio);
                    let (b0, b1) = (c * CH, ((c + 1) * CH).min(nb));
                    if b0 < b1 {
                        ast.score_pooled(&bw.idx_q[mi * nh * d..(mi + 1) * nh * d], ratio, d, b0, b1, &mut bw.idx_scores[mi * bw.nb_max..(mi + 1) * bw.nb_max]);
                    }
                }
                ctx.node_barrier();
            }
        } else {
            ctx.node_barrier();
        }
        lap!("b.attn.score");
        let scale = (hd as f32).powf(-0.5);
        // (head, block of QB queries) tasks: union of the queries' visible rows, K/V streamed once
        // Causal cost grows with the query block, so pair block i with block nqb-1-i and hand out the
        // pairs contiguously: every core gets about the same number of visible rows.
        // Tasks are claimed dynamically from a tile-shared counter (64 tasks over 13 cores with
        // unequal cost otherwise leave ~20 % of the phase as barrier wait); the counter was reset by
        // core 0 at the top of this function, before the projection's node barriers.
        let segments = self.segment_ranges(m);
        let blocks: Vec<(usize, usize)> = segments.flat_map(|(s, n)| (s..s + n).step_by(QB).map(move |b| (b, (b + QB).min(s + n)))).collect();
        let nqb = blocks.len();
        let ntask = qh * nqb;
        loop {
            let task = bw.task_ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as usize;
            if task >= ntask {
                break;
            }
            let (h, i) = (task / nqb, task % nqb);
            let qb = if i % 2 == 0 { i / 2 } else { nqb - 1 - i / 2 };
            let (m0, m1) = blocks[qb];
            let ast = ast_for(m0);
            let n_rows = self.row_position(m1 - 1, pos0) + 1;
            cs.qb_q.fill(0.0);
            cs.qb_memb.clear();
            cs.qb_memb.resize(n_rows, 0);
            for qi in 0..(m1 - m0) {
                let mi = m0 + qi;
                let pos = self.row_position(mi, pos0);
                let q = &mut cs.qb_q[qi * hd..(qi + 1) * hd];
                rmsnorm(&bw.qg[mi * ql + h * 2 * hd..mi * ql + h * 2 * hd + hd], Some(a.q_norm), cfg.rms_eps, q);
                self.rope.apply3(q, pos3[mi]);
                let bit = 1u8 << qi;
                if any_sparse && qsa::is_sparse(pos, ratio, cfg.idx_top_k) {
                    qsa::select_ranges(&bw.idx_scores[mi * bw.nb_max..(mi + 1) * bw.nb_max], pos, ratio, cfg.idx_top_k, &mut cs.sel_idx, &mut cs.sel_ranges);
                    for &(s0, len) in cs.sel_ranges.iter() {
                        for mb in &mut cs.qb_memb[s0 as usize..(s0 + len) as usize] {
                            *mb |= bit;
                        }
                    }
                } else {
                    for mb in &mut cs.qb_memb[..pos + 1] {
                        *mb |= bit;
                    }
                }
            }
            cs.qb_rows.clear();
            cs.qb_mrows.clear();
            for (j, &mb) in cs.qb_memb.iter().enumerate() {
                if mb != 0 {
                    cs.qb_rows.push(j as u32);
                    cs.qb_mrows.push(mb);
                }
            }
            for row in &mut cs.qb_rows { *row = ast.physical_row(*row as usize) as u32; }
            kv_dispatch!(ast.kv, |T| {
                let (k, v) = ast.kv_read::<T>();
                attend_block(&cs.qb_q, k, v, &cs.qb_rows, &cs.qb_mrows, hd, scale, &mut cs.qb_out, &mut cs.qb_scores);
            });
            for qi in 0..(m1 - m0) {
                let mi = m0 + qi;
                let gate = &bw.qg[mi * ql + h * 2 * hd + hd..mi * ql + (h + 1) * 2 * hd];
                let o = &cs.qb_out[qi * hd..(qi + 1) * hd];
                for i in 0..hd {
                    bw.vv[mi * ol + h * hd + i] = o[i] * sigmoid(gate[i]);
                }
            }
        }
        lap!("b.attn.attend");
        ctx.node_barrier();
        lap!("b.attn.wait");
        for mi in ctx.range_local(m) {
            self.prep_rows(bw.vv, ol, mi, mi + 1, bw.vv_q, bw.vv_s, bw.vv_sum, bw.vv_h, m);
        }
        ctx.node_barrier();
        let act = self.act(bw.vv_q, bw.vv_s, bw.vv_sum, bw.vv_h, m, ol);
        let r = ctx.range_local(a.o.n_strips());
        unsafe {
            self.dense_gemm_part(ctx, amx_buf!(bw, ctx), stage_buf!(bw, ctx), &a.o, r.start, r.end, &act, m);
        }
    }

    fn moe_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, cs: &mut CoreScratch, mo: &crate::exec::MoeW, m: usize, t: usize) {
        let cfg = &self.cfg;
        let ka = tr_kernels::router::MOE_K_MAX; // row stride of ids / w
        let ffl = cfg.n_ff / cfg.n_tiles;
        let hh = cfg.hidden;
        let nc = ctx.cores_per_node;
        let c = ctx.local;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }
        // routing per token (a variable count per token under the mass policy)
        for mi in ctx.range_local(m) {
            let n = self.route(&bw.rlog[mi * cfg.n_expert..(mi + 1) * cfg.n_expert], &mut bw.ids[mi * ka..(mi + 1) * ka], &mut bw.w[mi * ka..(mi + 1) * ka]);
            bw.n_sel[mi] = n as u32;
            if cfg.n_expert_packed < cfg.n_expert {
                fold_ids(&mut bw.ids[mi * ka..mi * ka + n], cfg.n_expert_packed as u32);
            }
            quant_rows(bw.mixed, hh, 32, mi, mi + 1, bw.mixed_q, bw.mixed_s, bw.mixed_sum);
            if self.use_amx(hh, m) {
                amx::rows_to_bf16(bw.mixed, hh, mi, mi + 1, bw.mixed_h);
            }
        }
        ctx.node_barrier();
        lap!("b.moe.route");
        // group entries by expert (core 0), shared expert rows appended as entries n_ent + mi
        if c == 0 {
            let mut order: Vec<usize> = (0..m).flat_map(|mi| (0..bw.n_sel[mi] as usize).map(move |s| mi * ka + s)).collect();
            let n_ent = order.len();
            order.sort_by_key(|&i| bw.ids[i]);
            let mut ng = 0usize;
            let mut last = u32::MAX;
            for (j, &i) in order.iter().enumerate() {
                let e = bw.ids[i];
                bw.ent_tok[j] = (i / ka) as u32;
                bw.ent_w[j] = bw.w[i];
                bw.ent_expert[j] = e;
                if e != last {
                    bw.grp_start[ng] = j as u32;
                    bw.grp_expert[ng] = e;
                    ng += 1;
                    last = e;
                }
            }
            bw.grp_start[ng] = n_ent as u32;
            bw.n_groups[0] = ng as u32;
            bw.n_groups[1] = n_ent as u32;
            if ctx.global == 0 {
                self.moe_stats.add(m as u64, n_ent as u64);
                for mi in 0..m {
                    if let Some(seg) = self.row_segment(mi) {
                        seg.experts.fetch_add(bw.n_sel[mi] as u64, std::sync::atomic::Ordering::Relaxed);
                    }
                    self.moe_stats.profile_row(&bw.rlog[mi * cfg.n_expert..(mi + 1) * cfg.n_expert]);
                }
            }
            let mut ord: Vec<u32> = (0..ng as u32).collect();
            ord.sort_by_key(|&g| std::cmp::Reverse(bw.grp_start[g as usize + 1] - bw.grp_start[g as usize]));
            bw.grp_order[..ng].copy_from_slice(&ord);
        }
        ctx.node_barrier();
        lap!("b.moe.group");
        let ng = bw.n_groups[0] as usize;
        let n_ent = bw.n_groups[1] as usize;
        let ns = mo.gate.n_strips();
        // tasks A: (group, gate|up) over all strips with the group's activations gathered into a
        // contiguous per-core buffer (one GEMM per expert: weights unpacked once per 8 tokens),
        // then shared expert (gate|up) over all m rows
        let ntask = ng * 2 + 2;
        let act_all = self.act(bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m, hh);
        let nb = hh / 32;
        let hg_ptr = bw.hg.as_mut_ptr();
        let hu_ptr = bw.hu.as_mut_ptr();
        if cs.grp_q.len() < m * hh {
            cs.grp_q.resize(m * hh, 0);
            cs.grp_s.resize(m * nb, 0.0);
            cs.grp_sum.resize(m * nb, 0);
        }
        let mut task = c;
        let mut gathered = usize::MAX;
        while task < ntask {
            unsafe {
                if task < ng * 2 {
                    let (grp, which) = (bw.grp_order[task / 2] as usize, task % 2);
                    let (r0, r1) = (bw.grp_start[grp] as usize, bw.grp_start[grp + 1] as usize);
                    let e = bw.grp_expert[grp] as usize;
                    if gathered != grp {
                        for r in r0..r1 {
                            let tok = bw.ent_tok[r] as usize;
                            let i = r - r0;
                            cs.grp_q[i * hh..(i + 1) * hh].copy_from_slice(&bw.mixed_q[tok * hh..(tok + 1) * hh]);
                            cs.grp_s[i * nb..(i + 1) * nb].copy_from_slice(&bw.mixed_s[tok * nb..(tok + 1) * nb]);
                            cs.grp_sum[i * nb..(i + 1) * nb].copy_from_slice(&bw.mixed_sum[tok * nb..(tok + 1) * nb]);
                        }
                        gathered = grp;
                    }
                    let g = r1 - r0;
                    let xq = QActRef { m: g, k: hh, kb: 32, q: &cs.grp_q[..g * hh], scale: &cs.grp_s[..g * nb], sum: &cs.grp_sum[..g * nb], pair: false };
                    let (mat, dst) = if which == 0 { (&mo.gate, hg_ptr) } else { (&mo.up, hu_ptr) };
                    gemm_tq(mat.item(e), mat.k, mat.codec, 0, ns, xq, dst.add(r0 * ffl), ffl, false);
                } else {
                    let (mat, dst) = if task == ng * 2 { (&mo.sh_gate, hg_ptr) } else { (&mo.sh_up, hu_ptr) };
                    self.dense_gemm(amx_buf!(bw, ctx), mat, 0, ns, &act_all, dst.add(n_ent * ffl), ffl);
                }
            }
            task += nc;
        }
        lap!("b.moe.gateup");
        ctx.node_barrier();
        lap!("b.moe.gateup.w");
        // tasks B: activation + quantisation per entry row, routing weight folded into the scale
        let kb = mo.down.codec.kb;
        let nbd = ffl / kb;
        let total_rows = n_ent + m;
        let mut act = vec![0f32; ffl];
        for r in ctx.range_local(total_rows) {
            elem::swiglu_vec(&bw.hg[r * ffl..(r + 1) * ffl], &bw.hu[r * ffl..(r + 1) * ffl], &mut act);
            let wgt = if r < n_ent { bw.ent_w[r] } else { sigmoid(dot(mo.shexp_gate, &bw.mixed[(r - n_ent) * hh..(r - n_ent + 1) * hh])) };
            if self.use_amx(ffl, m) && r >= n_ent {
                // shared-expert rows go through the AMX down GEMM: fold the gate into the values
                for v in act.iter_mut() {
                    *v *= wgt;
                }
                amx::rows_to_bf16(&act, ffl, 0, 1, &mut bw.act_h[(r - n_ent) * ffl..(r - n_ent + 1) * ffl]);
                continue;
            }
            let q = &mut bw.act_q[r * ffl..(r + 1) * ffl];
            let sc = &mut bw.act_s[r * nbd..(r + 1) * nbd];
            let su = &mut bw.act_sum[r * nbd..(r + 1) * nbd];
            QActRef::quantize_into(1, ffl, kb, &act, q, sc, su);
            for s in sc.iter_mut() {
                *s *= wgt;
            }
        }
        ctx.node_barrier();
        lap!("b.moe.act");
        // tasks C: down strips (this core's range): the shared expert for all m rows first, then every
        // group's rows accumulated straight into their tokens' rows (row map), all in this core's
        // columns — no per-entry scratch, no combine pass; then streamed into the PART column blocks
        let r = ctx.range_local(mo.down.n_strips());
        {
            let act = if self.use_amx(ffl, m) {
                Act::H(&bw.act_h[..m * ffl], m, ffl)
            } else {
                Act::Q8(QActRef { m, k: ffl, kb, q: &bw.act_q[n_ent * ffl..(n_ent + m) * ffl], scale: &bw.act_s[n_ent * nbd..(n_ent + m) * nbd], sum: &bw.act_sum[n_ent * nbd..(n_ent + m) * nbd], pair: false })
            };
            unsafe {
                self.dense_gemm(amx_buf!(bw, ctx), &mo.sh_down, r.start, r.end, &act, bw.moe_acc.as_mut_ptr(), hh);
            }
        }
        let (c0, c1) = (r.start * 16, r.end * 16);
        for grp in 0..ng {
            let (r0, r1) = (bw.grp_start[grp] as usize, bw.grp_start[grp + 1] as usize);
            let e = bw.grp_expert[grp] as usize;
            let xq = QActRef { m: r1 - r0, k: ffl, kb, q: &bw.act_q[r0 * ffl..r1 * ffl], scale: &bw.act_s[r0 * nbd..r1 * nbd], sum: &bw.act_sum[r0 * nbd..r1 * nbd], pair: false };
            unsafe {
                tr_kernels::gemv::gemm_tq_rows(mo.down.item(e).add(r.start * mo.down.strip_bytes()), mo.down.k, mo.down.codec, r.start, r.end, xq, bw.moe_acc.as_mut_ptr(), hh, Some(&bw.ent_tok[r0..r1]), true);
            }
        }
        lap!("b.moe.down");
        if c1 > c0 {
            // the PART rows were last read by every tile's reduce: stream them out (no RFO),
            // one piece per column block of the blocked layout
            let w = self.mbox.layout.part_w;
            for mi in 0..m {
                let acc = &bw.moe_acc[mi * hh..(mi + 1) * hh];
                let mut c = c0;
                while c < c1 {
                    let (b, e) = (c / w, c1.min((c / w + 1) * w));
                    let out = self.mbox.slot(t, self.mbox.layout.part_block(b));
                    put_row(&mut out[mi * w + c - b * w..mi * w + e - b * w], &acc[c..e]);
                    c = e;
                }
            }
            elem::store_fence();
        }
        lap!("b.moe.combine");
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_batch(&self, ctx: &WorkerCtx, bw: &mut BatchWs, st: &mut crate::state::TileState, cs: &mut CoreScratch, p: &crate::exec::PleW, toks: &[u32], prev: &[u32], m: usize, ckpt: bool) {
        let cfg = &self.cfg;
        let (hh, hc, tiles, t) = (cfg.hidden, cfg.hc, cfg.n_tiles, ctx.node);
        let hl = hh / tiles;
        let nh = cfg.ple_n_heads();
        let hash = tr_kernels::ple::PleHash { ngram: cfg.ple_ngram, per_gram: cfg.ple_per_gram, eos: cfg.ple_eos, mult: cfg.ple_mult.clone(), offsets: cfg.ple_offsets.clone(), vocab: cfg.ple_vocab.clone() };
        let mut rows = vec![0u32; nh];
        // gather rows: task (token, head)
        for task in ctx.range_local(m * nh) {
            let (mi, h) = (task / nh, task % nh);
            let seg = self.row_segment(mi);
            let mut ctxv: Vec<u32> = seg.map_or(prev, |s| s.prev.as_slice()).to_vec();
            ctxv.extend_from_slice(&toks[seg.map_or(0, |s| s.start)..mi]);
            let keep = cfg.ple_ngram - 1;
            let pv = if ctxv.len() > keep { &ctxv[ctxv.len() - keep..] } else { &ctxv[..] };
            hash.rows(toks[mi], pv, &mut rows);
            let row = rows[h];
            let owner = row as usize / self.ple_rows_per_tile;
            let raw = self.tiles[owner].raw("per_layer_token_embd.weight").expect("PLE table missing");
            let local = row as usize - raw.start_row;
            let src = unsafe { std::slice::from_raw_parts(raw.ptr.add(local * raw.row_bytes), raw.row_bytes) };
            tr_kernels::iq4nl::dequant_row(src, cfg.ple_dim, &mut bw.emb[mi * hh + h * cfg.ple_dim..mi * hh + (h + 1) * cfg.ple_dim]);
        }
        ctx.node_barrier();
        for mi in ctx.range_local(m) {
            self.prep_rows(bw.emb, hh, mi, mi + 1, bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m);
        }
        ctx.node_barrier();
        let act = self.act(bw.mixed_q, bw.mixed_s, bw.mixed_sum, bw.mixed_h, m, hh);
        let (nk, nv) = (p.key.n_strips(), p.value.n_strips());
        let kl = hc * hh / tiles;
        for (i, lo, hi) in sub_ranges(ctx.range_local(nk + nv), &[nk, nv]) {
            unsafe {
                if i == 0 {
                    self.dense_gemm(amx_buf!(bw, ctx), &p.key, lo, hi, &act, bw.ple_key.as_mut_ptr(), kl);
                } else {
                    self.dense_gemm(amx_buf!(bw, ctx), &p.value, lo, hi, &act, bw.ple_val.as_mut_ptr(), hl);
                }
            }
        }
        ctx.node_barrier();
        // per token: query norm (into ple_norm as scratch), partial stats + value slice -> mailbox
        for mi in ctx.range_local(m) {
            let res = &bw.res[mi * hc * hh..(mi + 1) * hc * hh];
            let qn = &mut bw.ple_norm[mi * hc * hh..(mi + 1) * hc * hh];
            for s in 0..hc {
                rmsnorm(&res[s * hh..(s + 1) * hh], Some(&p.norm_query[s * hh..(s + 1) * hh]), cfg.rms_eps, &mut qn[s * hh..(s + 1) * hh]);
            }
            let slot = self.mbox.slot(t, MailboxLayout::row(self.mbox.layout.ple, mi));
            for s in 0..hc {
                let key = &bw.ple_key[mi * kl + s * hl..mi * kl + (s + 1) * hl];
                let mut ss = 0f32;
                let mut dp = 0f32;
                for cidx in 0..hl {
                    let kv = key[cidx];
                    ss += kv * kv;
                    dp += kv * p.norm_key[s * hh + t * hl + cidx] * qn[s * hh + t * hl + cidx];
                }
                slot[2 * s] = ss;
                slot[2 * s + 1] = dp;
            }
            slot[2 * hc..2 * hc + hl].copy_from_slice(&bw.ple_val[mi * hl..(mi + 1) * hl]);
        }
        ctx.barrier(); // P
        // per token: gate, gather value, gated + normalized
        for mi in ctx.range_local(m) {
            let mut gate = [0f32; 8];
            for s in 0..hc {
                let mut ss = 0f32;
                let mut dp = 0f32;
                for u in 0..tiles {
                    let sl = self.mbox.slot_ro(u, MailboxLayout::row(self.mbox.layout.ple, mi));
                    ss += sl[2 * s];
                    dp += sl[2 * s + 1];
                }
                let rms = (ss / hh as f32 + cfg.rms_eps).sqrt();
                let sdot = dp / rms / (hh as f32).sqrt();
                gate[s] = sigmoid(sdot.signum() * sdot.abs().max(1e-6).sqrt());
            }
            let val = &mut cs.emb; // private scratch [hh]
            for u in 0..tiles {
                let sl = self.mbox.slot_ro(u, MailboxLayout::row(self.mbox.layout.ple, mi));
                val[u * hl..(u + 1) * hl].copy_from_slice(&sl[2 * hc..2 * hc + hl]);
            }
            let gated = &mut bw.ple_gated[mi * hc * hh..(mi + 1) * hc * hh];
            let norm = &mut bw.ple_norm[mi * hc * hh..(mi + 1) * hc * hh];
            for s in 0..hc {
                for i in 0..hh {
                    gated[s * hh + i] = val[i] * gate[s];
                }
                rmsnorm(&gated[s * hh..(s + 1) * hh], Some(&p.norm_conv[s * hh..(s + 1) * hh]), cfg.rms_eps, &mut norm[s * hh..(s + 1) * hh]);
            }
        }
        ctx.node_barrier();
        // dilated conv per token reading from the batch or the history; residual update
        let kern = cfg.ple_conv_kernel;
        let dil = cfg.ple_ngram;
        let hist = (kern - 1) * dil;
        let hcd = hc * hh;
        for task in ctx.range_local(m * hc) {
            let (mi, s) = (task / hc, task % hc);
            let start = self.row_segment(mi).map_or(0, |s| s.start);
            let local = mi - start;
            let history = unsafe { self.row_state(t, mi) };
            for i in 0..hh {
                let cidx = s * hh + i;
                let mut acc = p.conv[cidx * kern + (kern - 1)] * bw.ple_norm[mi * hcd + cidx];
                for k in 0..kern - 1 {
                    let back = (kern - 1 - k) * dil;
                    let src = if back <= local { bw.ple_norm[(mi - back) * hcd + cidx] } else { history.ple_hist[(hist - (back - local)) * hcd + cidx] };
                    acc += p.conv[cidx * kern + k] * src;
                }
                bw.res[mi * hcd + cidx] += bw.ple_gated[mi * hcd + cidx] + silu(acc);
            }
        }
        ctx.node_barrier();
        // new history after `n` rows = last `hist` rows of [old history ; normalized rows 0..n]
        let hist_after = |n: usize, r: usize| -> &[f32] {
            let src_idx = n as isize + r as isize - hist as isize;
            if src_idx >= 0 {
                &bw.ple_norm[src_idx as usize * hcd..(src_idx as usize + 1) * hcd]
            } else {
                let hr = (hist as isize + src_idx) as usize;
                &st.ple_hist[hr * hcd..(hr + 1) * hcd]
            }
        };
        if ckpt {
            // per-row checkpoints; the live history stays as it was (commit swaps the accepted slot in)
            for task in ctx.range_local(m * hist) {
                let (i, r) = (task / hist, task % hist);
                let src = hist_after(i + 1, r);
                let dst: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(st.ckpt[i].ple_hist.as_mut_ptr().add(r * hcd), hcd) };
                dst.copy_from_slice(src);
            }
        } else if ctx.local == 0 {
            let segments = self.segment_ranges(m);
            for (start, len) in segments {
                let st = unsafe { self.row_state(t, start) };
                let mut newh = vec![0f32; hist * hcd];
                for r in 0..hist {
                    let offset = len as isize + r as isize - hist as isize;
                    let src = if offset >= 0 { &bw.ple_norm[(start + offset as usize) * hcd..(start + offset as usize + 1) * hcd] }
                        else { let hr = (hist as isize + offset) as usize; &st.ple_hist[hr * hcd..(hr + 1) * hcd] };
                    newh[r * hcd..(r + 1) * hcd].copy_from_slice(src);
                }
                st.ple_hist.copy_from_slice(&newh);
            }
        }
        ctx.node_barrier();
    }
}

