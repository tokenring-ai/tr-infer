//! Per-sequence recurrent state, one struct per tile (in that tile's memory).
use crate::config::ModelConfig;
use anyhow::Result;
use tr_sys::numa::Arena;

pub struct GdnState {
    pub conv: &'static mut [f32], // [(d_conv-1)][c_local]
    pub ssm: &'static mut [f32],  // [heads_local][dk][dv]
}
/// K/V cache element type (`--kv`). f16 is what llama.cpp uses; 16-bit halves the per-token cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvType {
    F32,
    F16,
    Bf16,
}
impl KvType {
    pub fn elem_bytes(self) -> usize {
        if self == KvType::F32 { 4 } else { 2 }
    }
    pub fn parse(s: &str) -> Option<KvType> {
        match s {
            "f32" => Some(KvType::F32),
            "f16" => Some(KvType::F16),
            "bf16" => Some(KvType::Bf16),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KvType::F32 => "f32",
            KvType::F16 => "f16",
            KvType::Bf16 => "bf16",
        }
    }
}

/// Shared backing is held as raw pointers, never as aliased mutable slices. Page tables
/// determine disjoint writes; the owning sequence arenas outlive all pool invocations.
#[derive(Clone, Copy)]
pub(crate) struct PagedStorage {
    pub k: *mut u8,
    pub v: *mut u8,
    pub bytes: usize,
    pub pooled: *mut f32,
    pub pooled_len: usize,
}
unsafe impl Send for PagedStorage {}
unsafe impl Sync for PagedStorage {}

pub struct AttnState {
    pub(crate) paged: Option<PagedStorage>,
    /// Logical 64-token page to physical page; None is the legacy contiguous layout.
    pub pages: Option<Vec<usize>>,
    pub kv: KvType,
    pub k: &'static mut [u8], // [ctx_max][head_dim] elements of `kv` (pre-faulted at load: a first-touch 2 MiB fault mid-prefill stalls every tile at the next global barrier)
    pub v: &'static mut [u8],
    pub ik_pooled: &'static mut [f32], // [ctx_max/r][idx_dim] indexer keys per block (normed + roped)
    pub ik_raw: &'static mut [f32],    // [r][idx_dim] raw indexer keys of the block in progress
}
/// A snapshot of the destructive (in-place) recurrent state after one row of a speculative
/// verify batch: the GDN states of every recurrent layer and the PLE conv history. `commit`
/// swaps the accepted slot with the live buffers, so slots and live state are interchangeable.
pub struct Ckpt {
    pub gdn: Vec<Option<GdnState>>, // per layer (None for attention layers)
    pub ple_hist: &'static mut [f32],
}

pub struct TileState {
    pub gdn: Vec<Option<GdnState>>,   // per layer
    pub attn: Vec<Option<AttnState>>, // per layer
    pub ple_hist: &'static mut [f32], // [hist][hc*hidden]  (replicated on every tile)
    pub ctx_max: usize,
    /// Verify-row checkpoints (`spec_k + 1` slots; empty without speculative decoding).
    pub ckpt: Vec<Ckpt>,
    /// K/V + indexer cache of the MTP draft layer (positions as the main model's).
    pub mtp_attn: Option<AttnState>,
}

impl TileState {
    pub fn new(cfg: &ModelConfig, arena: &mut Arena, ctx_max: usize, kv: KvType, n_ckpt: usize, mtp: bool) -> Result<TileState> {
        let t = cfg.n_tiles;
        let c_local = (2 * cfg.n_k_heads / t + cfg.n_v_heads / t) * cfg.d_state;
        let heads_local = cfg.n_v_heads / t;
        let gdn_state = |arena: &mut Arena| -> Result<GdnState> {
            Ok(GdnState { conv: arena.alloc_slice((cfg.d_conv - 1) * c_local)?, ssm: arena.alloc_slice(heads_local * cfg.d_state * cfg.d_state)? /* chunk-major: [head][dv/16][dk][16] */ })
        };
        let attn_state = |arena: &mut Arena, il: usize| -> Result<AttnState> {
            let r = cfg.qsa_ratio(il);
            let nb = if r > 0 { ctx_max / r + 1 } else { 0 };
            Ok(AttnState {
                pages: None,
                paged: None,
                kv,
                k: arena.alloc_slice(ctx_max * cfg.head_dim * kv.elem_bytes())?,
                v: arena.alloc_slice(ctx_max * cfg.head_dim * kv.elem_bytes())?,
                ik_pooled: arena.alloc_slice(nb * cfg.idx_dim)?,
                ik_raw: arena.alloc_slice(r * cfg.idx_dim)?,
            })
        };
        let mut gdn = Vec::new();
        let mut attn = Vec::new();
        for il in 0..cfg.n_layer {
            if cfg.is_recurrent(il) {
                gdn.push(Some(gdn_state(arena)?));
                attn.push(None);
            } else {
                gdn.push(None);
                attn.push(Some(attn_state(arena, il)?));
            }
        }
        let hist = if cfg.ple_ngram > 0 { (cfg.ple_conv_kernel - 1) * cfg.ple_ngram } else { 0 };
        let ple_hist = arena.alloc_slice(hist * cfg.hc * cfg.hidden)?;
        let mut ckpt = Vec::new();
        for _ in 0..n_ckpt {
            let mut g = Vec::new();
            for il in 0..cfg.n_layer {
                g.push(if cfg.is_recurrent(il) { Some(gdn_state(arena)?) } else { None });
            }
            ckpt.push(Ckpt { gdn: g, ple_hist: arena.alloc_slice(hist * cfg.hc * cfg.hidden)? });
        }
        let mtp_attn = if mtp { Some(attn_state(arena, cfg.mtp_layer())?) } else { None };
        Ok(TileState { gdn, attn, ple_hist, ctx_max, ckpt, mtp_attn })
    }
    /// Bytes of recurrent state per checkpoint slot (for arena sizing).
    pub fn ckpt_bytes(cfg: &ModelConfig) -> usize {
        let t = cfg.n_tiles;
        let c_local = (2 * cfg.n_k_heads / t + cfg.n_v_heads / t) * cfg.d_state;
        let per_layer = ((cfg.d_conv - 1) * c_local + cfg.n_v_heads / t * cfg.d_state * cfg.d_state) * 4;
        let hist = if cfg.ple_ngram > 0 { (cfg.ple_conv_kernel - 1) * cfg.ple_ngram } else { 0 };
        (0..cfg.n_layer).filter(|&il| cfg.is_recurrent(il)).count() * per_layer + hist * cfg.hc * cfg.hidden * 4
    }
    /// Make checkpoint `j` the live state (pointer swap; the slot then holds the stale buffers).
    pub fn commit(&mut self, j: usize) {
        let c = &mut self.ckpt[j];
        for (live, snap) in self.gdn.iter_mut().zip(c.gdn.iter_mut()) {
            if let (Some(l), Some(s)) = (live.as_mut(), snap.as_mut()) {
                std::mem::swap(&mut l.ssm, &mut s.ssm);
                std::mem::swap(&mut l.conv, &mut s.conv);
            }
        }
        std::mem::swap(&mut self.ple_hist, &mut c.ple_hist);
    }
    pub fn reset(&mut self) {
        // K/V rows are written before they are read; nothing to clear
        for g in self.gdn.iter_mut().flatten() {
            g.conv.fill(0.0);
            g.ssm.fill(0.0);
        }
        self.ple_hist.fill(0.0);
    }
}

impl AttnState {
    #[inline]
    pub fn physical_row(&self, row: usize) -> usize {
        self.pages.as_ref().map_or(row, |p| p[row / crate::sequence::PAGE_TOKENS] * crate::sequence::PAGE_TOKENS + row % crate::sequence::PAGE_TOKENS)
    }

    /// Read-only physical backing, after the batch's KV-write barrier.
    pub(crate) fn kv_read<T: tr_kernels::attn::KvElem>(&self) -> (&[T], &[T]) {
        let (k, v, bytes) = self.paged.map_or((self.k.as_ptr(), self.v.as_ptr(), self.k.len()), |p| (p.k as *const u8, p.v as *const u8, p.bytes));
        let n = bytes / std::mem::size_of::<T>();
        unsafe { (std::slice::from_raw_parts(k.cast(), n), std::slice::from_raw_parts(v.cast(), n)) }
    }
    /// A single logical row. Caller owns its physical page for mutation.
    pub(crate) fn row_bytes_mut(&mut self, row: usize, bytes: usize) -> (&mut [u8], &mut [u8]) {
        let physical = self.physical_row(row);
        let (k, v, len) = self.paged.map_or((self.k.as_mut_ptr(), self.v.as_mut_ptr(), self.k.len()), |p| (p.k, p.v, p.bytes));
        assert!((physical + 1) * bytes <= len);
        unsafe { (std::slice::from_raw_parts_mut(k.add(physical * bytes), bytes), std::slice::from_raw_parts_mut(v.add(physical * bytes), bytes)) }
    }
    pub(crate) fn store(&mut self, row: usize, k: &[f32], v: &[f32]) {
        use crate::exec::kv_dispatch;
        let bytes = k.len() * self.kv.elem_bytes();
        kv_dispatch!(self.kv, |T| {
            let (kd, vd) = self.row_bytes_mut(row, bytes);
            unsafe {
                tr_kernels::attn::store_row(std::slice::from_raw_parts_mut(kd.as_mut_ptr().cast::<T>(), k.len()), k);
                tr_kernels::attn::store_row(std::slice::from_raw_parts_mut(vd.as_mut_ptr().cast::<T>(), v.len()), v);
            }
        });
    }
    fn pooled_physical(&self, block: usize, ratio: usize) -> usize {
        self.physical_row(block * ratio) / ratio
    }
    pub(crate) fn pooled_row(&self, block: usize, ratio: usize, d: usize) -> &[f32] {
        let row = self.pooled_physical(block, ratio);
        let (ptr, len) = self.paged.map_or((self.ik_pooled.as_ptr(), self.ik_pooled.len()), |p| (p.pooled as *const f32, p.pooled_len));
        assert!((row + 1) * d <= len);
        unsafe { std::slice::from_raw_parts(ptr.add(row * d), d) }
    }
    pub(crate) fn pooled_row_mut(&mut self, block: usize, ratio: usize, d: usize) -> &mut [f32] {
        let row = self.pooled_physical(block, ratio);
        let (ptr, len) = self.paged.map_or((self.ik_pooled.as_mut_ptr(), self.ik_pooled.len()), |p| (p.pooled, p.pooled_len));
        assert!((row + 1) * d <= len);
        unsafe { std::slice::from_raw_parts_mut(ptr.add(row * d), d) }
    }
    pub(crate) fn score_pooled(&self, q: &[f32], ratio: usize, d: usize, b0: usize, b1: usize, scores: &mut [f32]) {
        for b in b0..b1 {
            let row = self.pooled_row(b, ratio, d);
            tr_kernels::qsa::score_blocks(q, row, d, 0, 1, &mut scores[b..b + 1]);
        }
    }

    /// Typed views of the caches (T must match `self.kv`).
    #[inline]
    pub fn kv_as<T: tr_kernels::attn::KvElem>(&mut self) -> (&mut [T], &mut [T]) {
        let n = self.k.len() / std::mem::size_of::<T>();
        unsafe { (std::slice::from_raw_parts_mut(self.k.as_mut_ptr() as *mut T, n), std::slice::from_raw_parts_mut(self.v.as_mut_ptr() as *mut T, n)) }
    }
}
