//! Export / import of the sequence state for the persistent prefix cache.
//!
//! Two object shapes, both plain byte images of tile memory so a restore is a memcpy and the
//! continued generation is bit-identical to the un-cached run:
//!
//! * **rows** of positions `[a, b)`: for every attention layer (the packed attention layers in
//!   `manifest.layer_ids` order, then the MTP draft layer when loaded), for every K/V head
//!   (`n_head_kv`, each held by `n_tiles / n_head_kv` tiles; exported from the first): the K rows
//!   then the V rows (`(b - a) * head_dim * kv.elem_bytes()` bytes each), then the QSA indexer's
//!   pooled block keys of the blocks that *complete* inside `[a, b)` (`blocks_done(b) -
//!   blocks_done(a)` rows of `idx_dim` f32, identical on every tile). K and the block keys are
//!   stored roped at their absolute position, so rows are only valid at the position they were
//!   captured at.
//! * **snapshot** at position `P`: per tile, for every recurrent layer, the GDN conv history and
//!   the `[heads_local][dk][dv]` state, then (with MTP) the draft head's carry and chained
//!   residual; then once (tile 0's copy): the PLE conv history and every attention layer's raw
//!   indexer keys of the block in progress; then the host scalars (`n_past`, `prev_tokens`,
//!   `rope_delta`) and an optional f32 vector the caller stores with it (the server's logits of
//!   the last position, so a prompt that ends exactly at `P` can resume without a step).
//!
//! `Model::cache_identity` names everything that changes these bytes; stores are keyed by it.
use crate::exec::Model;
use crate::state::TileState;
use anyhow::{bail, Result};

/// Host-side sequence scalars stored in a snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HostState {
    pub n_past: usize,
    pub prev_tokens: Vec<u32>,
    pub rope_delta: i64,
    /// The caller's vector (the server stores the last logits here); empty when none was given.
    pub extra: Vec<f32>,
}

/// One attention layer's place in a rows object.
#[derive(Clone, Debug)]
struct RowsLayer {
    /// Layer index into `TileState::attn`, or `cfg.mtp_layer()` for the draft head.
    il: usize,
    ratio: usize,
}

struct SendPtr(*mut u8);
unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

const SNAP_MAGIC: u32 = 0x5053_4e41; // "ANSP"

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}
fn blocks_done(p: usize, r: usize) -> usize {
    if r == 0 { 0 } else { p / r }
}

impl Model {
    /// Everything that changes the bytes of a rows chunk or a snapshot (the pack, the overlays,
    /// the K/V element type, the tile split, the layers packed, the layout version).
    pub fn cache_identity(&self) -> Vec<String> {
        let layers: Vec<String> = self.manifest.layer_ids.iter().map(|l| l.to_string()).collect();
        vec![
            self.manifest.hash.clone(),
            self.kv.name().to_string(),
            self.cfg.n_tiles.to_string(),
            self.mtp_hash.clone().unwrap_or_else(|| "-".into()),
            self.vision_hash.clone().unwrap_or_else(|| "-".into()),
            layers.join(","),
            "state-v1".into(),
        ]
    }

    fn rows_layers(&self) -> Vec<RowsLayer> {
        let mut v: Vec<RowsLayer> = self.manifest.layer_ids.iter().filter(|&&il| !self.cfg.is_recurrent(il)).map(|&il| RowsLayer { il, ratio: self.cfg.qsa_ratio(il) }).collect();
        if self.mtp.is_some() {
            let il = self.cfg.mtp_layer();
            v.push(RowsLayer { il, ratio: self.cfg.qsa_ratio(il) });
        }
        v
    }
    fn tile_head(&self, t: usize) -> usize {
        t * self.cfg.n_head_kv / self.cfg.n_tiles
    }
    fn head_owner(&self, h: usize) -> usize {
        h * self.cfg.n_tiles / self.cfg.n_head_kv
    }
    fn hd_bytes(&self) -> usize {
        self.cfg.head_dim * self.kv.elem_bytes()
    }

    /// Bytes of the rows object for positions `[a, b)`.
    pub fn rows_bytes(&self, a: usize, b: usize) -> usize {
        assert!(a <= b);
        let per_head = 2 * (b - a) * self.hd_bytes();
        self.rows_layers().iter().map(|l| self.cfg.n_head_kv * per_head + (blocks_done(b, l.ratio) - blocks_done(a, l.ratio)) * self.cfg.idx_dim * 4).sum()
    }

    /// Copy the K/V and indexer rows of positions `[a, b)` into `dst` (`rows_bytes(a, b)` long).
    pub fn export_rows(&mut self, a: usize, b: usize, dst: &mut [u8]) {
        assert_eq!(dst.len(), self.rows_bytes(a, b), "rows buffer size");
        assert!(b <= self.n_past, "rows {a}..{b} beyond n_past {}", self.n_past);
        self.rows_copy(a, b, SendPtr(dst.as_mut_ptr()), true);
    }

    /// Write the rows of positions `[a, b)` from `src` (an `export_rows` image) into every tile.
    pub fn import_rows(&mut self, a: usize, b: usize, src: &[u8]) {
        assert_eq!(src.len(), self.rows_bytes(a, b), "rows buffer size");
        assert!(b <= self.state.first().map(|s| unsafe { (*s.0.get()).ctx_max }).unwrap_or(0), "rows {a}..{b} beyond the context window");
        self.rows_copy(a, b, SendPtr(src.as_ptr() as *mut u8), false);
    }

    fn rows_copy(&mut self, a: usize, b: usize, buf: SendPtr, export: bool) {
        if a == b {
            return;
        }
        let layers = self.rows_layers();
        let hd = self.hd_bytes();
        let (nkv, tiles) = (self.cfg.n_head_kv, self.cfg.n_tiles);
        let d = self.cfg.idx_dim;
        let n = b - a;
        // layer offsets inside the object
        let mut offs = Vec::with_capacity(layers.len());
        let mut off = 0usize;
        for l in &layers {
            offs.push(off);
            off += nkv * 2 * n * hd + (blocks_done(b, l.ratio) - blocks_done(a, l.ratio)) * d * 4;
        }
        let mut pool = self.pool.take().expect("pool");
        let me: &Model = &*self;
        let buf = &buf;
        pool.run(move |ctx| {
            let t = ctx.node;
            let st: &mut TileState = unsafe { &mut *me.state[t].0.get() };
            let head = me.tile_head(t);
            let owner = me.head_owner(head) == t;
            for (li, l) in layers.iter().enumerate() {
                let ast = if l.il == me.cfg.mtp_layer() { st.mtp_attn.as_mut().unwrap() } else { st.attn[l.il].as_mut().unwrap() };
                let base = offs[li] + head * 2 * n * hd;
                if export == owner || !export {
                    // K then V rows, split among the tile's cores
                    for row in ctx.range_local(n) {
                        let (kb, vb) = (base + row * hd, base + n * hd + row * hd);
                        let (kt, vt) = ast.row_bytes_mut(a + row, hd);
                        unsafe {
                            let kf = std::slice::from_raw_parts_mut(buf.0.add(kb), hd);
                            let vf = std::slice::from_raw_parts_mut(buf.0.add(vb), hd);
                            if export { kf.copy_from_slice(kt); vf.copy_from_slice(vt); }
                            else { kt.copy_from_slice(kf); vt.copy_from_slice(vf); }
                        }
                    }
                }
                // pooled block keys: one copy (tile 0 exports; every tile imports)
                if l.ratio > 0 && (!export || t == 0) {
                    let (b0, b1) = (blocks_done(a, l.ratio), blocks_done(b, l.ratio));
                    let pbase = offs[li] + nkv * 2 * n * hd;
                    for row in ctx.range_local(b1 - b0) {
                        let pk = f32_bytes_mut(ast.pooled_row_mut(b0 + row, l.ratio, d));
                        unsafe {
                            let f = std::slice::from_raw_parts_mut(buf.0.add(pbase + row * d * 4), d * 4);
                            if export { f.copy_from_slice(pk); } else { pk.copy_from_slice(f); }
                        }
                    }
                }
            }
            let _ = tiles;
        });
        self.pool = Some(pool);
    }

    /// Bytes of a snapshot storing `n_extra` caller f32s.
    pub fn snapshot_bytes(&self, n_extra: usize) -> usize {
        self.snapshot_layout().total + 4 * (4 + self.prev_tokens_max() + n_extra)
    }
    fn prev_tokens_max(&self) -> usize {
        self.cfg.ple_ngram.saturating_sub(1).max(1)
    }

    fn snapshot_layout(&self) -> SnapLayout {
        let cfg = &self.cfg;
        let t = cfg.n_tiles;
        let c_local = (2 * cfg.n_k_heads / t + cfg.n_v_heads / t) * cfg.d_state;
        let conv = (cfg.d_conv - 1) * c_local * 4;
        let ssm = cfg.n_v_heads / t * cfg.d_state * cfg.d_state * 4;
        let n_rec = self.manifest.layer_ids.iter().filter(|&&il| cfg.is_recurrent(il)).count();
        let hcd = cfg.hc * cfg.hidden * 4;
        let per_tile = n_rec * (conv + ssm) + if self.mtp.is_some() { 2 * hcd } else { 0 };
        let hist = if cfg.ple_ngram > 0 { (cfg.ple_conv_kernel - 1) * cfg.ple_ngram } else { 0 };
        let ple = hist * hcd;
        let raw: usize = self.rows_layers().iter().map(|l| l.ratio * cfg.idx_dim * 4).sum();
        SnapLayout { conv, ssm, per_tile, ple, total: t * per_tile + ple + raw }
    }

    /// Copy the recurrent state at `n_past` into `dst` (`snapshot_bytes(extra.len())` long).
    pub fn export_snapshot(&mut self, dst: &mut [u8], extra: &[f32]) {
        assert_eq!(dst.len(), self.snapshot_bytes(extra.len()), "snapshot buffer size");
        assert!(self.verify_toks.is_empty(), "snapshot during a verify");
        let lay = self.snapshot_layout();
        self.snapshot_copy(&lay, SendPtr(dst.as_mut_ptr()), true);
        // host trailer
        let mut o = lay.total;
        let put_u32 = |dst: &mut [u8], o: &mut usize, v: u32| {
            dst[*o..*o + 4].copy_from_slice(&v.to_le_bytes());
            *o += 4;
        };
        put_u32(dst, &mut o, SNAP_MAGIC);
        put_u32(dst, &mut o, self.n_past as u32);
        put_u32(dst, &mut o, self.rope_delta as i32 as u32);
        put_u32(dst, &mut o, self.prev_tokens.len() as u32);
        let pm = self.prev_tokens_max();
        assert!(self.prev_tokens.len() <= pm);
        for i in 0..pm {
            put_u32(dst, &mut o, self.prev_tokens.get(i).copied().unwrap_or(0));
        }
        dst[o..o + 4 * extra.len()].copy_from_slice(f32_bytes(extra));
    }

    /// Make `src` (an `export_snapshot` image) the live recurrent state and sequence position.
    /// The K/V rows below the snapshot's position must have been imported (or still be there).
    pub fn import_snapshot(&mut self, src: &[u8]) -> Result<HostState> {
        let lay = self.snapshot_layout();
        let fixed = lay.total + 4 * (4 + self.prev_tokens_max());
        if src.len() < fixed || (src.len() - fixed) % 4 != 0 {
            bail!("snapshot is {} bytes, expected at least {fixed} (+ 4n)", src.len());
        }
        let mut o = lay.total;
        let get_u32 = |o: &mut usize| -> u32 {
            let v = u32::from_le_bytes(src[*o..*o + 4].try_into().unwrap());
            *o += 4;
            v
        };
        if get_u32(&mut o) != SNAP_MAGIC {
            bail!("snapshot trailer magic mismatch");
        }
        let n_past = get_u32(&mut o) as usize;
        let rope_delta = get_u32(&mut o) as i32 as i64;
        let n_prev = get_u32(&mut o) as usize;
        let pm = self.prev_tokens_max();
        if n_prev > pm {
            bail!("snapshot lists {n_prev} previous tokens, at most {pm} expected");
        }
        let mut prev = Vec::with_capacity(n_prev);
        for i in 0..pm {
            let v = get_u32(&mut o);
            if i < n_prev {
                prev.push(v);
            }
        }
        let ctx_max = self.state.first().map(|s| unsafe { (*s.0.get()).ctx_max }).unwrap_or(0);
        if n_past > ctx_max {
            bail!("snapshot position {n_past} beyond the context window {ctx_max}");
        }
        let n_extra = (src.len() - o) / 4;
        let mut extra = vec![0f32; n_extra];
        f32_bytes_mut(&mut extra).copy_from_slice(&src[o..]);
        self.snapshot_copy(&lay, SendPtr(src.as_ptr() as *mut u8), false);
        self.n_past = n_past;
        self.prev_tokens = prev.clone();
        self.rope_delta = rope_delta;
        self.verify_toks.clear();
        self.mtp_carry_row = None;
        self.last_commit = 0;
        Ok(HostState { n_past, prev_tokens: prev, rope_delta, extra })
    }

    fn snapshot_copy(&mut self, lay: &SnapLayout, buf: SendPtr, export: bool) {
        let cfg = &self.cfg;
        let hcd = cfg.hc * cfg.hidden;
        let rec: Vec<usize> = self.manifest.layer_ids.iter().copied().filter(|&il| cfg.is_recurrent(il)).collect();
        let attn = self.rows_layers();
        let carry_row = self.mtp_carry_row;
        let mtp = self.mtp.is_some();
        let tiles = cfg.n_tiles;
        let mut pool = self.pool.take().expect("pool");
        let me: &Model = &*self;
        let buf = &buf;
        let rec = &rec;
        let attn = &attn;
        pool.run(move |ctx| {
            let t = ctx.node;
            let st: &mut TileState = unsafe { &mut *me.state[t].0.get() };
            let ws = unsafe { &mut *me.ws[t].0.get() };
            let tb = t * lay.per_tile;
            // items: (layer conv), (layer ssm), carry, res
            let n_items = 2 * rec.len() + if mtp { 2 } else { 0 };
            for i in ctx.range_local(n_items) {
                let (off, tile_mem): (usize, &mut [u8]) = if i < 2 * rec.len() {
                    let g = st.gdn[rec[i / 2]].as_mut().unwrap();
                    let off = tb + (i / 2) * (lay.conv + lay.ssm);
                    if i % 2 == 0 { (off, f32_bytes_mut(g.conv)) } else { (off + lay.conv, f32_bytes_mut(g.ssm)) }
                } else {
                    let off = tb + rec.len() * (lay.conv + lay.ssm) + (i - 2 * rec.len()) * hcd * 4;
                    if i == 2 * rec.len() {
                        // the carry: in `res_keep` after a commit, else in the workspace
                        if let (true, Some(r)) = (export, carry_row) {
                            let bw = unsafe { &mut *me.batch_ws[t].0.get() };
                            (off, f32_bytes_mut(&mut bw.res_keep[r * hcd..(r + 1) * hcd]))
                        } else {
                            (off, f32_bytes_mut(ws.mtp_carry))
                        }
                    } else {
                        (off, f32_bytes_mut(ws.mtp_res))
                    }
                };
                let f = unsafe { std::slice::from_raw_parts_mut(buf.0.add(off), tile_mem.len()) };
                if export {
                    f.copy_from_slice(tile_mem);
                } else {
                    tile_mem.copy_from_slice(f);
                }
            }
            // the replicated part: exported from tile 0, imported by every tile
            if !export || t == 0 {
                let shared = tiles * lay.per_tile;
                let n_items = 1 + attn.len();
                for i in ctx.range_local(n_items) {
                    let (off, tile_mem): (usize, &mut [u8]) = if i == 0 {
                        (shared, f32_bytes_mut(st.ple_hist))
                    } else {
                        let l = &attn[i - 1];
                        let ast = if l.il == cfg.mtp_layer() { st.mtp_attn.as_mut().unwrap() } else { st.attn[l.il].as_mut().unwrap() };
                        let off = shared + lay.ple + attn[..i - 1].iter().map(|x| x.ratio * cfg.idx_dim * 4).sum::<usize>();
                        (off, f32_bytes_mut(ast.ik_raw))
                    };
                    let f = unsafe { std::slice::from_raw_parts_mut(buf.0.add(off), tile_mem.len()) };
                    if export {
                        f.copy_from_slice(tile_mem);
                    } else {
                        tile_mem.copy_from_slice(f);
                    }
                }
            }
        });
        self.pool = Some(pool);
    }
}

struct SnapLayout {
    conv: usize,
    ssm: usize,
    per_tile: usize,
    ple: usize,
    total: usize,
}
