//! Vision encoder (Qwen3-VL style SigLIP ViT + 2x2 merger) run tensor-parallel over the tiles.
//!
//! Tile t owns heads 2t, 2t+1 of every layer: its slice of the fused qkv projection (rows
//! [q | k | v], 3*144), full attention for those heads over all N patches, and the matching K
//! columns of attn_out (padded to 160) whose partial sums are all-reduced. The MLP is split on its
//! hidden dimension (544 of 4352 per tile, up rows / down columns, partials all-reduced), the
//! merger on its hidden (576 of 4608). The residual stream x [N][1152] is replicated on every tile:
//! LayerNorms, bias adds and the patch embedding are computed redundantly per tile (they are
//! trivial next to the GEMMs), which keeps the exchanges at two all-reduces per layer.
//!
//! Weights are bf16 in the AMX strip layout (`Bf16Mat`, no unpacking; 8-bit weights cost ~8 %
//! output error on photos, bf16 0.4 %) and every dense GEMM is AMX-BF16 with f32 accumulation, so
//! the encoder needs AMX. Attention is AMX-BF16 too, flash-style per (head, 32-query block) task:
//! Q rows against K packed as B strips (16 keys × lanes), scores for one key block at a time
//! (`kb` keys, f32), an online softmax whose probabilities go straight to bf16 rows, and P·V
//! accumulated in the AMX tiles against V packed per key block (strips of 16 head dims × the
//! block's keys). The logits need more than bf16 precision (the ViT's attention logits reach 70
//! with row ranges over 100; bf16 q/k cost 0.3 per logit and up to 100 % on single output tokens),
//! so q and k are split into bf16 hi + lo parts and one pass over lanes [q_hi | q_lo | q_hi] ·
//! [k_hi | k_hi | k_lo] accumulates the three cross terms (~16 mantissa bits); P and V stay bf16. Post-attention work is chunked by `m_max` rows so
//! the existing `part`/`rsum`/`rfull` mailbox rows and the socket-aware `allreduce_rows` are
//! reused unchanged.
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{bail, Context, Result};
use tr_format::manifest::Manifest;
use tr_kernels::amx::{self, Line, TileRows};
use tr_kernels::elem;
use tr_sys::numa::Arena;
use tr_sys::pool::WorkerCtx;

use crate::exchange::MailboxLayout;
use crate::exec::{Model, SyncCell};
use crate::exec_batch::STAGE_COLS;
use crate::image::{Fit, PrepParams};
use crate::weights::{Bf16Mat, TileWeights};

/// Query rows per attention task (two AMX row blocks).
const QBV: usize = 32;
/// Key padding: the P·V product's K dimension (bf16 pairs, 32 per tile row).
const NPAD: usize = 32;
/// V output lane padding (head_dim 72 -> 80: five B strips).
const HPAD: usize = 16;
/// Q/K lane padding: the Q·K^T product's K dimension (3 × head_dim 72 = 216 -> 224).
const HKPAD: usize = 32;
/// Default keys per attention block (`TR_VKB`): scores 32 × kb f32 + probabilities bf16 per core
/// (192 KiB at 1024; 256/512/1024 measured within 3 %, larger blocks slightly ahead).
const KBLK: usize = 1024;

#[derive(Clone, Debug)]
pub struct VisionCfg {
    pub hidden: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub patch: usize,
    pub merge: usize,
    pub proj: usize,
    pub eps: f32,
    pub n_pos_side: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Per-tile slices (from the pack's config).
    pub dl: usize,   // q (k, v) width per tile = heads_per_tile * head_dim
    pub opad: usize, // attn_out K slice per tile
    pub fp: usize,   // MLP hidden per tile
    pub mp: usize,   // merger hidden per tile
    pub heads_per_tile: usize,
    pub hdp: usize, // head_dim padded to HPAD (V / output lanes)
    pub hdk: usize, // 3 * head_dim padded to HKPAD (Q / K lanes: hi, lo, hi | hi, hi, lo)
    pub kb: usize,  // keys per attention block
    /// Image limits in merged tokens.
    pub min_tokens: usize,
    pub max_tokens: usize,
}

impl VisionCfg {
    pub fn from_manifest(m: &Manifest, n_tiles: usize, min_tokens: usize, max_tokens: usize) -> Result<VisionCfg> {
        let u = |k: &str| m.config.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).with_context(|| format!("vision pack config lacks {k}"));
        let hidden = u("clip.vision.embedding_length")?;
        let n_head = u("clip.vision.attention.head_count")?;
        let arr = |k: &str| -> [f32; 3] {
            let v: Vec<f32> = m.config.get(k).and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()).unwrap_or_default();
            if v.len() == 3 { [v[0], v[1], v[2]] } else { [0.5; 3] }
        };
        if n_head % n_tiles != 0 {
            bail!("vision: {n_head} heads over {n_tiles} tiles");
        }
        let head_dim = hidden / n_head;
        let hpt = n_head / n_tiles;
        let c = VisionCfg {
            hidden,
            n_layer: u("clip.vision.block_count")?,
            n_head,
            head_dim,
            n_ff: u("clip.vision.feed_forward_length")?,
            patch: u("clip.vision.patch_size")?,
            merge: u("clip.vision.spatial_merge_size")?,
            proj: u("clip.vision.projection_dim")?,
            eps: m.config.get("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
            n_pos_side: (u("vision.n_pos")? as f64).sqrt() as usize,
            mean: arr("clip.vision.image_mean"),
            std: arr("clip.vision.image_std"),
            dl: hpt * head_dim,
            opad: u("vision.attn_out_k_pad")?,
            fp: u("vision.ff_per_tile")?,
            mp: u("vision.merge_per_tile")?,
            heads_per_tile: hpt,
            hdp: head_dim.div_ceil(HPAD) * HPAD,
            hdk: (3 * head_dim).div_ceil(HKPAD) * HKPAD,
            kb: std::env::var("TR_VKB").ok().and_then(|v| v.parse().ok()).filter(|&k: &usize| k % NPAD == 0 && k > 0).unwrap_or(KBLK),
            min_tokens,
            max_tokens,
        };
        if c.n_pos_side * c.n_pos_side != u("vision.n_pos")? {
            bail!("vision: position table is not square");
        }
        if c.hidden % 32 != 0 || (c.patch * c.patch * 3) % 32 != 0 || c.opad % 32 != 0 || c.fp % 32 != 0 || c.mp % 32 != 0 {
            bail!("vision: GEMM K sizes must be multiples of 32");
        }
        Ok(c)
    }
    pub fn prep_params(&self) -> PrepParams {
        let fit = if std::env::var("TR_IMAGE_PAD_CEIL").map(|v| v == "1").unwrap_or(false) { Fit::PadCeil } else { Fit::Stretch };
        PrepParams { patch: self.patch, merge: self.merge, min_tokens: self.min_tokens, max_tokens: self.max_tokens, mean: self.mean, std: self.std, fit }
    }
    /// Patches the workspace is sized for.
    pub fn max_patches(&self) -> usize {
        self.max_tokens * self.merge * self.merge
    }
}

pub struct VLayerW {
    pub qkv: Bf16Mat,
    pub qkv_b: &'static [f32],
    pub out: Bf16Mat,
    pub out_b: &'static [f32],
    pub ln1_w: &'static [f32],
    pub ln1_b: &'static [f32],
    pub ln2_w: &'static [f32],
    pub ln2_b: &'static [f32],
    pub up: Bf16Mat,
    pub up_b: &'static [f32],
    pub down: Bf16Mat,
    pub down_b: &'static [f32],
}

pub struct VisionW {
    pub patch: Bf16Mat,
    pub patch_b: &'static [f32],
    pub pos: &'static [f32],
    pub layers: Vec<VLayerW>,
    pub post_w: &'static [f32],
    pub post_b: &'static [f32],
    pub mm0: Bf16Mat,
    pub mm0_b: &'static [f32],
    pub mm2: Bf16Mat,
    pub mm2_b: &'static [f32],
}

impl VisionW {
    pub fn from_tile(tw: &TileWeights, c: &VisionCfg) -> VisionW {
        let layers = (0..c.n_layer)
            .map(|il| {
                let p = |n: &str| format!("v.blk.{il}.{n}");
                VLayerW {
                    qkv: *tw.bf16(&p("attn_qkv.weight")),
                    qkv_b: tw.f32(&p("attn_qkv.bias")),
                    out: *tw.bf16(&p("attn_out.weight")),
                    out_b: tw.f32(&p("attn_out.bias")),
                    ln1_w: tw.f32(&p("ln1.weight")),
                    ln1_b: tw.f32(&p("ln1.bias")),
                    ln2_w: tw.f32(&p("ln2.weight")),
                    ln2_b: tw.f32(&p("ln2.bias")),
                    up: *tw.bf16(&p("ffn_up.weight")),
                    up_b: tw.f32(&p("ffn_up.bias")),
                    down: *tw.bf16(&p("ffn_down.weight")),
                    down_b: tw.f32(&p("ffn_down.bias")),
                }
            })
            .collect();
        VisionW {
            patch: *tw.bf16("v.patch_embd.weight"),
            patch_b: tw.f32("v.patch_embd.bias"),
            pos: tw.f32("v.position_embd.weight"),
            layers,
            post_w: tw.f32("v.post_ln.weight"),
            post_b: tw.f32("v.post_ln.bias"),
            mm0: *tw.bf16("mm.0.weight"),
            mm0_b: tw.f32("mm.0.bias"),
            mm2: *tw.bf16("mm.2.weight"),
            mm2_b: tw.f32("mm.2.bias"),
        }
    }
}

/// Activation rows as bf16 (the AMX A operand), 64-byte aligned rows.
pub struct ActBuf {
    pub k: usize,
    pub h: &'static mut [u16],
}
impl ActBuf {
    fn new(a: &mut Arena, rows: usize, k: usize) -> Result<ActBuf> {
        Ok(ActBuf { k, h: a.alloc_slice(rows * k)? })
    }
    /// Convert rows [r0, r1) of `x` ([rows][k] f32).
    fn prep(&mut self, x: &[f32], r0: usize, r1: usize) {
        amx::rows_to_bf16(x, self.k, r0, r1, self.h);
    }
    /// Convert one f32 row into row `r`.
    fn prep_row(&mut self, x: &[f32], r: usize) {
        let k = self.k;
        amx::rows_to_bf16(x, k, 0, 1, &mut self.h[r * k..(r + 1) * k]);
    }
    /// Rows [r0, r1) with the buffer re-viewed as rows of `k` (the merger reads 4 patch rows as one).
    fn rows(&self, r0: usize, r1: usize, k: usize) -> &[u16] {
        &self.h[r0 * k..r1 * k]
    }
}

/// Per-tile workspace, sized for `n_max` patches.
pub struct VisionWs {
    pub n_max: usize,
    pub x: &'static mut [f32],      // [N][H] residual (replicated)
    pub xn: ActBuf,                 // [N][H] normed rows (also the merger input viewed as [N/4][4H])
    pub pin: ActBuf,                // [N][patch_len]
    pub qkv: &'static mut [f32],    // [N][3*dl]
    pub qb: &'static mut [u16],     // [hpt][Npad][hdk] bf16 query rows (pre-scaled)
    pub kb: &'static mut [Line],    // [hpt][Npad/16 strips][hdk/2] K as B strips (16 keys x hdk)
    pub vb: &'static mut [Line],    // [hpt][key block][hdp/16 strips][kb/2] V as B strips (16 dims x block keys)
    pub att: &'static mut [f32],    // [N][opad]
    pub att_a: ActBuf,              // [m_max][opad]
    pub red: &'static mut [f32],    // [m_max][proj.max(H)] all-reduce result
    pub xn2: ActBuf,                // [m_max][H]
    pub up: &'static mut [f32],     // [m_max][fp]
    pub act: ActBuf,                // [m_max][fp]
    pub m0: &'static mut [f32],     // [N/4][mp]
    pub m0_a: ActBuf,               // [N/4][mp]
    pub sblk: &'static mut [f32],   // [cores][QBV][kb] scores of one key block
    pub pblk: &'static mut [u16],   // [cores][QBV][kb] probabilities (bf16)
    pub stage: &'static mut [f32],  // [cores][m_max*STAGE_COLS]
    pub ctr: Vec<AtomicUsize>,
    pub tmp: &'static mut [f32], // [cores][4H] per-core row scratch
}

impl VisionWs {
    pub fn bytes(c: &VisionCfg, m_max: usize, cores: usize) -> usize {
        let n = c.max_patches();
        let np = n.div_ceil(NPAD) * NPAD;
        let h = c.hidden;
        let plen = c.patch * c.patch * 3;
        let nkb = np.div_ceil(c.kb);
        n * h * 4 + n * h * 2 + n * plen * 2 + n * 3 * c.dl * 4 + 2 * c.heads_per_tile * np * c.hdk * 2 + c.heads_per_tile * nkb * c.kb * c.hdp * 2 + n * c.opad * 4 + m_max * c.opad * 2 + m_max * c.proj.max(h) * 4 + m_max * h * 2 + m_max * c.fp * 4 + m_max * c.fp * 2 + n / 4 * c.mp * 4 + n / 4 * c.mp * 2 + cores * QBV * c.kb * 6 + cores * m_max * STAGE_COLS * 4 + cores * 4 * h * 4 + (8 << 20)
    }
    pub fn new(a: &mut Arena, c: &VisionCfg, m_max: usize, cores: usize) -> Result<VisionWs> {
        let n = c.max_patches();
        let np = n.div_ceil(NPAD) * NPAD;
        let h = c.hidden;
        let plen = c.patch * c.patch * 3;
        Ok(VisionWs {
            n_max: n,
            x: a.alloc_slice(n * h)?,
            xn: ActBuf::new(a, n, h)?,
            pin: ActBuf::new(a, n, plen)?,
            qkv: a.alloc_slice(n * 3 * c.dl)?,
            qb: a.alloc_slice(c.heads_per_tile * np * c.hdk)?,
            kb: a.alloc_slice(c.heads_per_tile * np * c.hdk / 32)?,
            vb: a.alloc_slice(c.heads_per_tile * np.div_ceil(c.kb) * c.kb * c.hdp / 32)?,
            att: a.alloc_slice(n * c.opad)?,
            att_a: ActBuf::new(a, m_max, c.opad)?,
            red: a.alloc_slice(m_max * c.proj.max(h))?,
            xn2: ActBuf::new(a, m_max, h)?,
            up: a.alloc_slice(m_max * c.fp)?,
            act: ActBuf::new(a, m_max, c.fp)?,
            m0: a.alloc_slice(n / 4 * c.mp)?,
            m0_a: ActBuf::new(a, n / 4, c.mp)?,
            sblk: a.alloc_slice(cores * QBV * c.kb)?,
            pblk: a.alloc_slice(cores * QBV * c.kb)?,
            stage: a.alloc_slice(cores * m_max * STAGE_COLS)?,
            ctr: (0..8 * (c.n_layer + 2)).map(|_| AtomicUsize::new(0)).collect(),
            tmp: a.alloc_slice(cores * 4 * h)?,
        })
    }
}

/// The loaded encoder: config, per-tile weights and workspaces, and the host-side output buffer.
pub struct Vision {
    pub cfg: VisionCfg,
    pub w: Vec<VisionW>,
    pub(crate) ws: Vec<SyncCell<VisionWs>>,
    pub(crate) out: SyncCell<Vec<f32>>,
}

/// One image's encoder input (host memory, read by every tile).
struct VisionIn<'a> {
    n: usize,
    patches: &'a [f32],
    pos: &'a [f32],
    ypos: &'a [u32],
    xpos: &'a [u32],
}

#[inline]
fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32, y: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        y[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/// Layout of the packed attention operands for one image (per tile).
#[derive(Clone, Copy)]
struct AttnGeom {
    np: usize,  // keys padded to NPAD
    hdk: usize, // Q/K lanes
    hdp: usize, // V/output lanes
    kbs: usize, // keys per block
}
impl AttnGeom {
    fn nkb(&self) -> usize {
        self.np.div_ceil(self.kbs)
    }
    /// First K strip line of the strip holding key `r`; pair p is line `+ p`, u32 slot r % 16.
    fn kline(&self, hh: usize, r: usize) -> usize {
        (hh * (self.np / 16) + r / 16) * (self.hdk / 2)
    }
    /// u16 index of V[key r][dim d] in the block-major strips (block stride kbs*hdp; the last
    /// block's strips are shorter: the block's key count / 2 lines each).
    fn vidx(&self, hh: usize, r: usize, d: usize) -> usize {
        let (b, rb) = (r / self.kbs, r % self.kbs);
        let kblk = self.kbs.min(self.np - b * self.kbs);
        (hh * self.nkb() + b) * self.kbs * self.hdp + (d / 16) * kblk * 16 + (rb / 2) * 32 + (d % 16) * 2 + (rb % 2)
    }
}

/// Softmax attention of head `hh` for query rows [r0, r0+mr) over the `n` keys: flash-style over
/// key blocks with an online softmax, scores/probabilities in the per-core `sblk`/`pblk`
/// ([QBV][kbs]), output `o` [mr][hdp] f32 (normalised). Keeps the AMX tile config in `cfg`.
///
/// # Safety
/// `qb`/`kb`/`vb` are the packed operands of `g` (queries pre-scaled), padded keys zero.
#[allow(clippy::too_many_arguments)]
unsafe fn attn_task(g: &AttnGeom, n: usize, hh: usize, r0: usize, mr: usize, qb: *const u16, kb: *const Line, vb: *const Line, sblk: &mut [f32], pblk: &mut [u16], o: &mut [f32], cfg: &mut TileRows) {
    let (np, hdk, hdp, kbs) = (g.np, g.hdk, g.hdp, g.kbs);
    let q = qb.add((hh * np + r0) * hdk);
    o[..mr * hdp].fill(0.0);
    let mut mx = [f32::NEG_INFINITY; QBV];
    let mut l = [0f32; QBV];
    for b in 0..g.nkb() {
        let k0 = b * kbs;
        if k0 >= n {
            break;
        }
        let kblk = kbs.min(np - k0);
        let kv = kblk.min(n - k0);
        // scores S = Q K^T for this key block
        amx::gemm_bf16_kept(kb.add(g.kline(hh, k0)), hdk, kblk / 16, q, hdk, mr, sblk.as_mut_ptr(), kbs, 0, false, cfg);
        // online softmax: rescale the running output/sum, probabilities to bf16
        for qi in 0..mr {
            let s = &sblk[qi * kbs..qi * kbs + kv];
            let mnew = mx[qi].max(elem::vmax(s));
            let sum = elem::exp_sub_bf16(s, mnew, &mut pblk[qi * kbs..qi * kbs + kblk]);
            let alpha = (mx[qi] - mnew).exp();
            l[qi] = l[qi] * alpha + sum;
            mx[qi] = mnew;
            if alpha != 1.0 && b > 0 {
                elem::scale_inplace(&mut o[qi * hdp..(qi + 1) * hdp], alpha);
            }
        }
        // O += P V
        amx::gemm_bf16_kept(vb.add((hh * g.nkb() + b) * kbs * hdp / 32), kblk, hdp / 16, pblk.as_ptr(), kbs, mr, o.as_mut_ptr(), hdp, 0, true, cfg);
    }
    for qi in 0..mr {
        elem::scale_inplace(&mut o[qi * hdp..(qi + 1) * hdp], 1.0 / l[qi]);
    }
}

#[inline]
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + (((b >> 16) & 1) + 0x7FFF)) >> 16) as u16
}
/// x ≈ hi + lo with both parts bf16-representable (returned as f32).
#[inline]
fn bf16_split(x: f32) -> (f32, f32) {
    let hi = f32::from_bits((bf16_bits(x) as u32) << 16);
    let lo = f32::from_bits((bf16_bits(x - hi) as u32) << 16);
    (hi, lo)
}

#[inline]
fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// bf16 strips [s0, s1) of `mat` against `m` activation rows `x` (k = mat.k, 64-byte aligned):
/// y[mi*ldy + col0 + 16*(s-s0) + n].
///
/// # Safety
/// `y` must hold `m` rows of `ldy` floats covering the written columns.
unsafe fn bf16_gemm(mat: &Bf16Mat, s0: usize, s1: usize, x: &[u16], m: usize, y: *mut f32, ldy: usize, col0: usize) {
    if s0 >= s1 {
        return;
    }
    debug_assert!(x.len() >= m * mat.k);
    amx::gemm_bf16(mat.ptr.add(s0 * mat.strip_len()) as *const Line, mat.k, s1 - s0, x.as_ptr(), m, y.add(col0), ldy, 0);
}

impl Model {
    /// Load the vision overlay's weights and workspaces (called from `load_with`).
    pub(crate) fn vision_init(tiles: &mut [TileWeights], cfg: VisionCfg, m_max: usize, cores: usize) -> Result<Vision> {
        let mut w = Vec::new();
        let mut ws = Vec::new();
        for tw in tiles.iter_mut() {
            w.push(VisionW::from_tile(tw, &cfg));
            ws.push(SyncCell(UnsafeCell::new(VisionWs::new(&mut tw.arena, &cfg, m_max, cores)?)));
        }
        Ok(Vision { cfg, w, ws, out: SyncCell(UnsafeCell::new(Vec::new())) })
    }

    /// Encode a preprocessed image into `[n_tokens][hidden]` embeddings (one pool run).
    pub fn encode_image(&mut self, pt: &crate::image::Patches) -> Vec<f32> {
        let v = self.vision.as_ref().expect("no vision encoder loaded");
        let c = &v.cfg;
        let n = pt.n_patches();
        assert!(n > 0 && n <= c.max_patches(), "image of {n} patches exceeds the workspace ({})", c.max_patches());
        assert!(n % (c.merge * c.merge) == 0 && pt.patch_len() == c.patch * c.patch * 3);
        let pos = crate::image::pos_rows(v.w[0].pos, c.n_pos_side, c.hidden, pt.gw, pt.gh, c.merge);
        let n_tok = n / (c.merge * c.merge);
        unsafe { (*v.out.0.get()).resize(n_tok * c.proj, 0.0) };
        let inp = VisionIn { n, patches: &pt.data, pos: &pos, ypos: &pt.ypos, xpos: &pt.xpos };
        let t0 = std::time::Instant::now();
        {
            let mut pool = self.pool.take().expect("pool");
            let me: &Model = &*self;
            let ir = &inp;
            pool.run(move |ctx| me.vision_worker(ctx, ir));
            self.pool = Some(pool);
        }
        if self.profile.enabled {
            eprintln!("vision: {n} patches -> {n_tok} tokens in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let v = self.vision.as_ref().unwrap();
        unsafe { (*v.out.0.get()).clone() }
    }

    /// GEMM over all `n` rows of `act` (re-viewed with k = mat.k) with tasks (strip pair, row
    /// chunk) claimed dynamically from counter `ctr`; `y[r*ldy + col]`. Ends with a node barrier.
    #[allow(clippy::too_many_arguments)]
    unsafe fn vgemm_tasks(&self, ctx: &WorkerCtx, ws: &VisionWs, ctr: usize, mat: &Bf16Mat, act: &ActBuf, n: usize, y: *mut f32, ldy: usize) {
        const SG: usize = 2; // strips per task
        const RC: usize = 512; // rows per task
        let k = mat.k;
        let ns = mat.n_strips().div_ceil(SG);
        let nr = n.div_ceil(RC);
        loop {
            let task = ws.ctr[ctr].fetch_add(1, Ordering::Relaxed);
            if task >= ns * nr {
                break;
            }
            let (si, ri) = (task % ns, task / ns);
            let (s0, s1) = (si * SG, ((si + 1) * SG).min(mat.n_strips()));
            let (r0, r1) = (ri * RC, ((ri + 1) * RC).min(n));
            bf16_gemm(mat, s0, s1, act.rows(r0, r1, k), r1 - r0, y.add(r0 * ldy), ldy, s0 * 16);
        }
        ctx.node_barrier();
    }

    /// Partial-sum GEMM of rows [r_off, r_off+m) of `act` streamed into the `part` mailbox rows
    /// (strips split statically over the tile's cores), then all-reduced into `ws.red[..m*rows]`.
    unsafe fn vgemm_reduce(&self, ctx: &WorkerCtx, ws: &VisionWs, bw: &mut crate::exec_batch::BatchWs, mat: &Bf16Mat, act: &ActBuf, r_off: usize, m: usize) {
        let t = ctx.node;
        let lay = &self.mbox.layout;
        let x = act.rows(r_off, r_off + m, mat.k);
        let r = ctx.range_local(mat.n_strips());
        let stage = std::slice::from_raw_parts_mut((ws.stage.as_ptr() as *mut f32).add(ctx.local * bw.m_max * STAGE_COLS), bw.m_max * STAGE_COLS);
        let mb = &self.mbox;
        let mut s = r.start;
        while s < r.end {
            let ns = (r.end - s).min(STAGE_COLS / 16);
            let cols = ns * 16;
            bf16_gemm(mat, s, s + ns, x, m, stage.as_mut_ptr(), cols, 0);
            for mi in 0..m {
                let dst = &mut mb.slot(t, MailboxLayout::row(lay.part, mi))[s * 16..s * 16 + cols];
                let src = &stage[mi * cols..(mi + 1) * cols];
                if (dst.as_ptr() as usize) % 64 == 0 {
                    elem::stream_copy(dst, src);
                } else {
                    dst.copy_from_slice(src);
                }
            }
            s += ns;
        }
        elem::store_fence();
        ctx.barrier();
        let red: &'static mut [f32] = std::slice::from_raw_parts_mut(ws.red.as_ptr() as *mut f32, ws.red.len());
        self.allreduce_rows(ctx, bw, lay.part, mat.rows, m, red);
    }

    fn vision_worker(&self, ctx: &WorkerCtx, inp: &VisionIn) {
        let v = self.vision.as_ref().unwrap();
        let c = &v.cfg;
        let t = ctx.node;
        let w = &v.w[t];
        let ws: &mut VisionWs = unsafe { &mut *v.ws[t].0.get() };
        let bw = unsafe { &mut *self.batch_ws[t].0.get() };
        let n = inp.n;
        let np = n.div_ceil(NPAD) * NPAD;
        let (h, hd, hdp, hdk, hpt, dl, opad, fp, mp, kbs) = (c.hidden, c.head_dim, c.hdp, c.hdk, c.heads_per_tile, c.dl, c.opad, c.fp, c.mp, c.kb);
        let g = AttnGeom { np, hdk, hdp, kbs };
        let kline = |hh: usize, r: usize| g.kline(hh, r);
        let vidx = |hh: usize, r: usize, d: usize| g.vidx(hh, r, d);
        let kb32 = ws.kb.as_mut_ptr() as *mut u32;
        let vb16 = ws.vb.as_mut_ptr() as *mut u16;
        let m_max = bw.m_max;
        let tmp: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(ws.tmp.as_mut_ptr().add(ctx.local * 4 * h), 4 * h) };
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); } } }
        let mut ctr = 0usize;
        macro_rules! next_ctr { () => {{ ctr += 1; ctr - 1 }} }
        if ctx.local == 0 {
            for a in &ws.ctr {
                a.store(0, Ordering::Relaxed);
            }
        }
        // ---- patch embedding input, key/value padding
        {
            let rows = ctx.range_local(n);
            ws.pin.prep(inp.patches, rows.start, rows.end);
        }
        // padded keys n..np: zero K pairs and V (never written by the layers)
        for i in ctx.range_local((np - n) * hpt) {
            let (hh, r) = (i / (np - n), n + i % (np - n));
            for p in 0..hdk / 2 {
                unsafe { *kb32.add((kline(hh, r) + p) * 16 + r % 16) = 0 };
            }
            for d in 0..hdp {
                unsafe { *vb16.add(vidx(hh, r, d)) = 0 };
            }
        }
        ctx.node_barrier();
        let xp = ws.x.as_mut_ptr();
        unsafe { self.vgemm_tasks(ctx, ws, next_ctr!(), &w.patch, &ws.pin, n, xp, h) };
        for r in ctx.range_local(n) {
            let x = &mut ws.x[r * h..(r + 1) * h];
            let p = &inp.pos[r * h..(r + 1) * h];
            for i in 0..h {
                x[i] += w.patch_b[i] + p[i];
            }
        }
        ctx.node_barrier();
        lap!("v.embed");

        let scale = (hd as f32).powf(-0.5);
        let half = hd / 2; // rope pairs (j, j+half): j < half/2 rotate by y, the rest by x
        let quarter = half / 2;
        let inv_freq: Vec<f32> = (0..quarter).map(|j| (10000f64).powf(-2.0 * j as f64 / half as f64) as f32).collect();
        for il in 0..c.n_layer {
            let lw = &w.layers[il];
            // ---- LN1 -> xn
            for r in ctx.range_local(n) {
                layernorm(&ws.x[r * h..(r + 1) * h], lw.ln1_w, lw.ln1_b, c.eps, &mut tmp[..h]);
                ws.xn.prep_row(&tmp[..h], r);
            }
            ctx.node_barrier();
            lap!("v.ln1");
            // ---- qkv projection (this tile's heads), bias, rope, head layouts
            let qp = ws.qkv.as_mut_ptr();
            unsafe { self.vgemm_tasks(ctx, ws, next_ctr!(), &lw.qkv, &ws.xn, n, qp, 3 * dl) };
            lap!("v.qkv");
            for r in ctx.range_local(n) {
                let row = &mut ws.qkv[r * 3 * dl..(r + 1) * 3 * dl];
                for i in 0..3 * dl {
                    row[i] += lw.qkv_b[i];
                }
                let (py, px) = (inp.ypos[r] as f32, inp.xpos[r] as f32);
                for hh in 0..hpt {
                    for which in 0..2 {
                        let v = &mut row[which * dl + hh * hd..which * dl + (hh + 1) * hd];
                        for j in 0..half {
                            let (p, f) = if j < quarter { (py, inv_freq[j]) } else { (px, inv_freq[j - quarter]) };
                            let th = p * f;
                            let (s, co) = th.sin_cos();
                            let (x0, x1) = (v[j], v[j + half]);
                            v[j] = x0 * co - x1 * s;
                            v[j + half] = x0 * s + x1 * co;
                        }
                        if which == 0 {
                            // Q: scaled [hi | lo | hi] bf16 row (staged as exactly representable f32), zero pad
                            let q = &mut tmp[..hdk];
                            for d in 0..hd {
                                let (hi, lo) = bf16_split(v[d] * scale);
                                q[d] = hi;
                                q[hd + d] = lo;
                                q[2 * hd + d] = hi;
                            }
                            q[3 * hd..].fill(0.0);
                            amx::rows_to_bf16(q, hdk, 0, 1, &mut ws.qb[(hh * np + r) * hdk..(hh * np + r + 1) * hdk]);
                        } else {
                            // K: [hi | hi | lo] bf16 pairs into the strip lines (pad pairs zero)
                            let l = kline(hh, r) * 16 + r % 16;
                            let lane = |e: usize| -> u16 {
                                if e < 2 * hd {
                                    bf16_bits(v[e % hd])
                                } else if e < 3 * hd {
                                    bf16_bits(v[e - 2 * hd] - f32::from_bits((bf16_bits(v[e - 2 * hd]) as u32) << 16))
                                } else {
                                    0
                                }
                            };
                            for p in 0..hdk / 2 {
                                unsafe { *kb32.add(l + p * 16) = lane(2 * p) as u32 | (lane(2 * p + 1) as u32) << 16 };
                            }
                        }
                    }
                    let vv = &row[2 * dl + hh * hd..2 * dl + (hh + 1) * hd];
                    for d in 0..hdp {
                        unsafe { *vb16.add(vidx(hh, r, d)) = if d < hd { bf16_bits(vv[d]) } else { 0 } };
                    }
                }
            }
            ctx.node_barrier();
            lap!("v.rope");
            // ---- attention: (head, query block) tasks
            let nqb = n.div_ceil(QBV);
            let cidx = next_ctr!();
            let sblk: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(ws.sblk.as_mut_ptr().add(ctx.local * QBV * kbs), QBV * kbs) };
            let pblk: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(ws.pblk.as_mut_ptr().add(ctx.local * QBV * kbs), QBV * kbs) };
            let mut cfg = TileRows::default();
            loop {
                let task = ws.ctr[cidx].fetch_add(1, Ordering::Relaxed);
                if task >= hpt * nqb {
                    break;
                }
                let (hh, qb) = (task / nqb, task % nqb);
                let (r0, r1) = (qb * QBV, ((qb + 1) * QBV).min(n));
                let mr = r1 - r0;
                let o = &mut tmp[..mr * hdp];
                unsafe { attn_task(&g, n, hh, r0, mr, ws.qb.as_ptr(), ws.kb.as_ptr(), ws.vb.as_ptr(), sblk, pblk, o, &mut cfg) };
                for qi in 0..mr {
                    let row = &mut ws.att[(r0 + qi) * opad..(r0 + qi + 1) * opad];
                    row[hh * hd..(hh + 1) * hd].copy_from_slice(&o[qi * hdp..qi * hdp + hd]);
                    if hh == hpt - 1 {
                        row[hpt * hd..].fill(0.0);
                    }
                }
            }
            if cfg != TileRows::default() {
                amx::release_tiles();
            }
            ctx.node_barrier();
            lap!("v.attn");
            // ---- attn_out (all-reduce), residual, LN2, MLP (all-reduce), residual: chunks of m_max rows
            let mut r0 = 0;
            while r0 < n {
                let m = (n - r0).min(m_max);
                let rows = ctx.range_local(m);
                ws.att_a.prep(&ws.att[r0 * opad..(r0 + m) * opad], rows.start, rows.end);
                ctx.node_barrier();
                unsafe { self.vgemm_reduce(ctx, ws, bw, &lw.out, &ws.att_a, 0, m) };
                lap!("v.oproj");
                for mi in rows.clone() {
                    let x = &mut ws.x[(r0 + mi) * h..(r0 + mi + 1) * h];
                    let red = &ws.red[mi * h..(mi + 1) * h];
                    for i in 0..h {
                        x[i] += red[i] + lw.out_b[i];
                    }
                    layernorm(x, lw.ln2_w, lw.ln2_b, c.eps, &mut tmp[..h]);
                    ws.xn2.prep_row(&tmp[..h], mi);
                }
                ctx.node_barrier();
                // up: strips over cores, all m rows
                {
                    let r = ctx.range_local(lw.up.n_strips());
                    let up = ws.up.as_ptr() as *mut f32;
                    unsafe { bf16_gemm(&lw.up, r.start, r.end, ws.xn2.rows(0, m, h), m, up, fp, r.start * 16) };
                }
                ctx.node_barrier();
                for mi in rows.clone() {
                    let u = &mut ws.up[mi * fp..(mi + 1) * fp];
                    for i in 0..fp {
                        u[i] = gelu(u[i] + lw.up_b[i]);
                    }
                }
                ws.act.prep(&ws.up[..m * fp], rows.start, rows.end);
                ctx.node_barrier();
                lap!("v.up");
                unsafe { self.vgemm_reduce(ctx, ws, bw, &lw.down, &ws.act, 0, m) };
                lap!("v.down");
                for mi in rows.clone() {
                    let x = &mut ws.x[(r0 + mi) * h..(r0 + mi + 1) * h];
                    let red = &ws.red[mi * h..(mi + 1) * h];
                    for i in 0..h {
                        x[i] += red[i] + lw.down_b[i];
                    }
                }
                ctx.node_barrier();
                r0 += m;
            }
        }
        // ---- post LN over all rows -> xn (viewed as [n/4][4h] by the merger)
        for r in ctx.range_local(n) {
            layernorm(&ws.x[r * h..(r + 1) * h], w.post_w, w.post_b, c.eps, &mut tmp[..h]);
            ws.xn.prep_row(&tmp[..h], r);
        }
        ctx.node_barrier();
        let n4 = n / (c.merge * c.merge);
        let mh = h * c.merge * c.merge;
        // merger fc1: the xn rows re-viewed with k = 4h (vgemm_tasks views rows by the matrix's k)
        debug_assert_eq!(w.mm0.k, mh);
        let m0p = ws.m0.as_mut_ptr();
        unsafe { self.vgemm_tasks(ctx, ws, next_ctr!(), &w.mm0, &ws.xn, n4, m0p, mp) };
        for r in ctx.range_local(n4) {
            let u = &mut ws.m0[r * mp..(r + 1) * mp];
            for i in 0..mp {
                u[i] = gelu(u[i] + w.mm0_b[i]);
            }
        }
        {
            let rows = ctx.range_local(n4);
            ws.m0_a.prep(&ws.m0[..n4 * mp], rows.start, rows.end);
        }
        ctx.node_barrier();
        lap!("v.mm0");
        let mut r0 = 0;
        while r0 < n4 {
            let m = (n4 - r0).min(m_max);
            unsafe { self.vgemm_reduce(ctx, ws, bw, &w.mm2, &ws.m0_a, r0, m) };
            if t == 0 {
                let out: &mut Vec<f32> = unsafe { &mut *v.out.0.get() };
                for mi in ctx.range_local(m) {
                    let o = &mut out[(r0 + mi) * c.proj..(r0 + mi + 1) * c.proj];
                    let red = &ws.red[mi * c.proj..(mi + 1) * c.proj];
                    for i in 0..c.proj {
                        o[i] = red[i] + w.mm2_b[i];
                    }
                }
            }
            ctx.node_barrier();
            r0 += m;
        }
        lap!("v.mm2");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_round(x: f32) -> f32 {
        f32::from_bits((bf16_bits(x) as u32) << 16)
    }

    /// The flash kernel against a scalar softmax attention on the bf16-rounded operands, for
    /// several key counts (partial last block, single block, 32-key blocks) and query tails.
    #[test]
    fn amx_attention_matches_reference() {
        if !amx::init() {
            eprintln!("AMX unavailable, skipping");
            return;
        }
        let (hd, hdk, hdp) = (72usize, 224usize, 80usize);
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 % 20000) as f32 / 10000.0 - 1.0
        };
        for &(n, kbs, hpt) in &[(196usize, 256usize, 1usize), (37, 64, 2), (400, 256, 2), (400, 32, 1), (800, 256, 2)] {
            let np = n.div_ceil(NPAD) * NPAD;
            let g = AttnGeom { np, hdk, hdp, kbs };
            let scale = 25.0f32; // logits ~ N(0, 70): what the ViT reaches; bf16 alone would fail here
            let q: Vec<f32> = (0..hpt * n * hd).map(|_| rnd() * scale).collect();
            let k: Vec<f32> = (0..hpt * n * hd).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..hpt * n * hd).map(|_| rnd()).collect();
            // pack (64-byte aligned buffers)
            let mut qb = vec![Line([0; 32]); hpt * np * hdk / 32];
            let mut kb = vec![Line([0; 32]); hpt * np * hdk / 32];
            let mut vb = vec![Line([0; 32]); hpt * g.nkb() * kbs * hdp / 32];
            let qb16 = qb.as_mut_ptr() as *mut u16;
            let kb32 = kb.as_mut_ptr() as *mut u32;
            let vb16 = vb.as_mut_ptr() as *mut u16;
            for hh in 0..hpt {
                for r in 0..n {
                    for d in 0..hd {
                        let (hi, lo) = bf16_split(q[(hh * n + r) * hd + d]);
                        unsafe {
                            *qb16.add((hh * np + r) * hdk + d) = bf16_bits(hi);
                            *qb16.add((hh * np + r) * hdk + hd + d) = bf16_bits(lo);
                            *qb16.add((hh * np + r) * hdk + 2 * hd + d) = bf16_bits(hi);
                        }
                    }
                    let l = g.kline(hh, r) * 16 + r % 16;
                    let lane = |e: usize| -> u16 {
                        let (hi, lo) = bf16_split(k[(hh * n + r) * hd + e % hd]);
                        if e < 2 * hd { bf16_bits(hi) } else if e < 3 * hd { bf16_bits(lo) } else { 0 }
                    };
                    for p in 0..hdk / 2 {
                        unsafe { *kb32.add(l + p * 16) = lane(2 * p) as u32 | (lane(2 * p + 1) as u32) << 16 };
                    }
                    for d in 0..hd {
                        unsafe { *vb16.add(g.vidx(hh, r, d)) = bf16_bits(v[(hh * n + r) * hd + d]) };
                    }
                }
            }
            let mut sblk = vec![Line([0; 32]); QBV * kbs / 16]; // f32 scores
            let mut pblk = vec![Line([0; 32]); QBV * kbs / 32];
            let sblk: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(sblk.as_mut_ptr() as *mut f32, QBV * kbs) };
            let pblk: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(pblk.as_mut_ptr() as *mut u16, QBV * kbs) };
            let mut o = vec![0f32; QBV * hdp];
            let mut cfg = TileRows::default();
            let mut worst = 0f32;
            for hh in 0..hpt {
                let mut r0 = 0;
                while r0 < n {
                    let mr = (n - r0).min(QBV);
                    unsafe { attn_task(&g, n, hh, r0, mr, qb.as_ptr() as *const u16, kb.as_ptr(), vb.as_ptr(), sblk, pblk, &mut o, &mut cfg) };
                    for qi in 0..mr {
                        let qr = &q[(hh * n + r0 + qi) * hd..(hh * n + r0 + qi + 1) * hd];
                        let mut sc = vec![0f64; n];
                        for j in 0..n {
                            let kr = &k[(hh * n + j) * hd..(hh * n + j + 1) * hd];
                            sc[j] = (0..hd).map(|d| qr[d] as f64 * kr[d] as f64).sum();
                        }
                        let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                        let sum: f64 = e.iter().sum();
                        for d in 0..hd {
                            let want: f64 = (0..n).map(|j| e[j] * bf16_round(v[(hh * n + j) * hd + d]) as f64).sum::<f64>() / sum;
                            let got = o[qi * hdp + d] as f64;
                            if (got - want).abs() as f32 > worst {
                                worst = (got - want).abs() as f32;
                                if worst > 4e-3 {
                                    eprintln!("n {n} kbs {kbs} head {hh} row {} dim {d}: {got} vs {want}", r0 + qi);
                                }
                            }
                        }
                    }
                    r0 += mr;
                }
            }
            amx::release_tiles();
            eprintln!("n {n} kbs {kbs} hpt {hpt}: max abs err {worst:.2e}");
            assert!(worst <= 4e-3, "n {n} kbs {kbs}: max abs err {worst}");
        }
    }
}
