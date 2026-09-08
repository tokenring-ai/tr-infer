//! The qwen4exp decode program (one token per call), executed SPMD on all tiles/cores.
//! See docs/NOTEBOOK.md and the plan: 6 global exchanges per layer (A..E, G; attention layers add
//! I for the QSA indexer all-gather), tile-local work
//! split by 16-row strips, replicated small vectors recomputed per core.
use crate::config::ModelConfig;
use crate::exchange::{MailboxLayout, Mailboxes};
use crate::state::{KvType, TileState};
use crate::weights::{dequant_tq_row, load_weights, LoadOptions, LoadReport, TileWeights, TqMat};
use anyhow::{Context, Result};
use std::cell::UnsafeCell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tr_kernels::router::{MoePolicy, MOE_K_MAX};
use tr_format::Manifest;
use tr_kernels::attn::{attend_partial, dot, merge_partials, store_row, KvElem};
use tr_kernels::qsa;
use tr_kernels::elem::{self, rmsnorm, sigmoid, silu, softplus};
use tr_kernels::gemv::gemv_tq;
use tr_kernels::quant::{QAct, QActRef};
use tr_kernels::rope::RopeTable;
use tr_sys::pool::{Pool, WorkerCtx};

/// Run `$body` with `T` bound to the K/V cache element type.
macro_rules! kv_dispatch {
    ($kv:expr, |$T:ident| $body:expr) => {
        match $kv {
            crate::state::KvType::F32 => {
                type $T = f32;
                $body
            }
            crate::state::KvType::F16 => {
                type $T = tr_kernels::attn::F16;
                $body
            }
            crate::state::KvType::Bf16 => {
                type $T = tr_kernels::attn::Bf16;
                $body
            }
        }
    };
}
pub(crate) use kv_dispatch;

// ---------------------------------------------------------------------------------------------
// weight tables

#[derive(Clone, Copy)]
pub struct HcW {
    pub norm: &'static [f32],
    pub inject: Option<&'static [f32]>,
    pub down: TqMat,
    pub up: TqMat,
}
#[derive(Clone, Copy)]
pub struct GdnW {
    pub qkv: TqMat,
    pub z: TqMat,
    pub out: TqMat,
    pub beta: &'static [f32],
    pub alpha: &'static [f32],
    pub conv: &'static [f32],
    pub dt: &'static [f32],
    pub a: &'static [f32],
    pub norm: &'static [f32],
}
#[derive(Clone, Copy)]
pub struct AttnW {
    pub q: TqMat,
    pub k: TqMat,
    pub v: TqMat,
    pub o: TqMat,
    pub q_norm: &'static [f32],
    pub k_norm: &'static [f32],
    pub idx: Option<IdxW>,
}
/// QSA indexer projections: this tile's row slices of q_proj (idx_heads*idx_dim rows) and k_proj
/// (idx_dim rows); the outputs are all-gathered so every tile scores with the full indexer.
#[derive(Clone, Copy)]
pub struct IdxW {
    pub q: TqMat,
    pub k: TqMat,
    pub q_norm: &'static [f32],
    pub k_norm: &'static [f32],
}
#[derive(Clone, Copy)]
pub struct MoeW {
    pub router: &'static [f32],
    pub shexp_gate: &'static [f32],
    pub gate: TqMat,
    pub up: TqMat,
    pub down: TqMat,
    pub sh_gate: TqMat,
    pub sh_up: TqMat,
    pub sh_down: TqMat,
}
#[derive(Clone, Copy)]
pub struct PleW {
    pub key: TqMat,
    pub value: TqMat,
    pub norm_key: &'static [f32],
    pub norm_query: &'static [f32],
    pub norm_conv: &'static [f32],
    pub conv: &'static [f32], // [hc*hidden][kernel]
}
#[derive(Clone, Copy)]
pub struct LayerW {
    pub hc_attn: HcW,
    pub hc_ffn: HcW,
    pub gdn: Option<GdnW>,
    pub attn: Option<AttnW>,
    pub moe: MoeW,
    pub ple: Option<PleW>,
}
#[derive(Clone, Copy)]
pub struct HeadW {
    pub hc: HcW,
    pub output: TqMat,
    pub embd: TqMat,
}
/// The MTP draft head (overlay pack): one full-attention layer plus its own output mixer and the
/// two input projections (`x = fc_hidden(gemma_rms(hc_hidden)) per stream + fc_embedding(gemma_rms(embed))`).
#[derive(Clone, Copy)]
pub struct MtpW {
    pub layer: LayerW,
    pub hc: HcW,
    pub fc_emb: TqMat,
    pub fc_hid: TqMat,
    pub enorm: &'static [f32],
    pub hnorm: &'static [f32],
}

fn layer_w(tw: &TileWeights, cfg: &ModelConfig, il: usize) -> LayerW {
    layer_w_kind(tw, cfg, il, cfg.is_recurrent(il))
}

/// `recurrent` overrides the interval rule (the MTP draft layer sits at index n_layer and is full attention).
fn layer_w_kind(tw: &TileWeights, cfg: &ModelConfig, il: usize, recurrent: bool) -> LayerW {
    let p = |n: &str| format!("blk.{il}.{n}");
    let hc = |which: &str| HcW {
        norm: tw.f32(&p(&format!("{which}_norm.weight"))),
        inject: Some(tw.f32(&p(&format!("{which}_inject.weight")))),
        down: *tw.tq(&p(&format!("{which}_down.weight"))),
        up: *tw.tq(&p(&format!("{which}_up.weight"))),
    };
    let gdn = if recurrent {
        Some(GdnW {
            qkv: *tw.tq(&p("attn_qkv.weight")),
            z: *tw.tq(&p("attn_gate.weight")),
            out: *tw.tq(&p("ssm_out.weight")),
            beta: tw.f32(&p("ssm_beta.weight")),
            alpha: tw.f32(&p("ssm_alpha.weight")),
            conv: tw.f32(&p("ssm_conv1d.weight")),
            dt: tw.f32(&p("ssm_dt.bias")),
            a: tw.f32(&p("ssm_a")),
            norm: tw.f32(&p("ssm_norm.weight")),
        })
    } else {
        None
    };
    let attn = if recurrent {
        None
    } else {
        Some(AttnW {
            q: *tw.tq(&p("attn_q.weight")),
            k: *tw.tq(&p("attn_k.weight")),
            v: *tw.tq(&p("attn_v.weight")),
            o: *tw.tq(&p("attn_output.weight")),
            q_norm: tw.f32(&p("attn_q_norm.weight")),
            k_norm: tw.f32(&p("attn_k_norm.weight")),
            idx: if tw.tq.contains_key(&p("indexer.q_proj.weight")) {
                Some(IdxW { q: *tw.tq(&p("indexer.q_proj.weight")), k: *tw.tq(&p("indexer.k_proj.weight")), q_norm: tw.f32(&p("indexer.q_norm.weight")), k_norm: tw.f32(&p("indexer.k_norm.weight")) })
            } else {
                None
            },
        })
    };
    let moe = MoeW {
        router: tw.f32(&p("ffn_gate_inp.weight")),
        shexp_gate: tw.f32(&p("ffn_gate_inp_shexp.weight")),
        gate: *tw.tq(&p("ffn_gate_exps.weight")),
        up: *tw.tq(&p("ffn_up_exps.weight")),
        down: *tw.tq(&p("ffn_down_exps.weight")),
        sh_gate: *tw.tq(&p("ffn_gate_shexp.weight")),
        sh_up: *tw.tq(&p("ffn_up_shexp.weight")),
        sh_down: *tw.tq(&p("ffn_down_shexp.weight")),
    };
    let ple = if cfg.ple_layers.contains(&il) && tw.tq.contains_key(&p("ple_key.weight")) {
        Some(PleW {
            key: *tw.tq(&p("ple_key.weight")),
            value: *tw.tq(&p("ple_value.weight")),
            norm_key: tw.f32(&p("ple_norm_key.weight")),
            norm_query: tw.f32(&p("ple_norm_query.weight")),
            norm_conv: tw.f32(&p("ple_norm_conv.weight")),
            conv: tw.f32(&p("ple_conv1d.weight")),
        })
    } else {
        None
    };
    LayerW { hc_attn: hc("hc_attn"), hc_ffn: hc("hc_ffn"), gdn, attn, moe, ple }
}

// ---------------------------------------------------------------------------------------------
// workspaces

/// Tile-shared scratch (raw slices in tile memory; written by disjoint cores between barriers).
pub struct TileWs {
    pub xn: &'static mut [f32],         // hc*hidden (normed residual, tile-shared)
    pub xq_q: &'static mut [i8],        // 2*hc*hidden (pair-quantised xn)
    pub xq_s: &'static mut [f32],       // 2*hc*hidden/32
    pub xq_sum: &'static mut [i32],
    pub part: &'static mut [f32],       // per-core partials: [cores][2*hc] (sumsq, inject)
    pub gate_local: &'static mut [f32], // hc*hidden/T
    pub mixed_full: &'static mut [f32], // hidden
    pub red: &'static mut [f32],        // hidden (reduced partial sums)
    pub rlog: &'static mut [f32],       // n_expert
    pub qkv: &'static mut [f32],        // c_local
    pub qkvc: &'static mut [f32],       // c_local
    pub z: &'static mut [f32],          // v_local
    pub beta: &'static mut [f32],       // 16
    pub alpha: &'static mut [f32],      // 16
    pub y: &'static mut [f32],          // v_local
    pub qg: &'static mut [f32],         // q heads local * 2 * head_dim
    pub kn: &'static mut [f32],         // head_dim
    pub vn: &'static mut [f32],         // head_dim
    pub o: &'static mut [f32],          // q heads local * head_dim
    pub hg: &'static mut [f32],         // slots * ff_local
    pub hu: &'static mut [f32],
    pub act_q: &'static mut [i8],       // slots * ff_local
    pub act_s: &'static mut [f32],      // slots * ff_local/16
    pub act_sum: &'static mut [i32],
    pub ple_emb: &'static mut [f32],    // hidden (gathered PLE rows, tile-shared)
    pub ple_key: &'static mut [f32],    // hc*hidden/T
    pub ple_val: &'static mut [f32],    // hidden/T
    pub ple_stats: &'static mut [f32],  // 2*hc
    pub idx_qraw: &'static mut [f32],   // idx_heads*idx_dim (gathered indexer q projection)
    pub idx_kraw: &'static mut [f32],   // idx_dim
    pub idx_q: &'static mut [f32],      // idx_heads*idx_dim (normed + roped)
    pub idx_scores: &'static mut [f32], // blocks_max
    pub qn: &'static mut [f32],         // q heads local * head_dim (normed + roped, tile-shared)
    pub sel: &'static mut [u32],        // [0] = n ranges, then (start, len) pairs (QSA selection)
    pub attn_part: &'static mut [f32],  // [qh * cores][head_dim + 16] partial attention outputs
    pub mtp_carry: &'static mut [f32],  // hc*hidden: main-model residual of the last committed position (MTP input)
    pub mtp_res: &'static mut [f32],    // hc*hidden: residual the draft layer produced last (chained drafts)
    pub cand_core: &'static mut [f32],  // [cores][2*CAND_PER_TILE] per-core draft candidates (id bits, logit)
    pub draft_tok: &'static mut [u32],  // [16] the token core 0 sampled in a draft chain
}

/// Per-core private scratch.
pub struct CoreScratch {
    pub hiprec: bool,
    pub res: Vec<f32>,
    pub xn: Vec<f32>,
    pub tmp: Vec<f32>,
    pub lo: Vec<f32>,
    pub vec_v: Vec<f32>,
    pub emb: Vec<f32>,
    pub inject: [f32; 8],
    pub q_hc: QAct,
    pub q_lr: QAct,
    pub q_h: QAct,
    pub q_v: QAct,
    pub ids: Vec<u32>,
    pub w: Vec<f32>,
    pub attn_scratch: Vec<f32>,
    pub head_tmp: Vec<f32>,
    pub sel_idx: Vec<u32>,
    pub sel_ranges: Vec<(u32, u32)>,
    // query-blocked prefill attention
    pub qb_q: Vec<f32>,      // [QB][head_dim]
    pub qb_out: Vec<f32>,    // [QB][head_dim]
    pub qb_scores: Vec<f32>, // [rows][QB]
    pub qb_memb: Vec<u8>,    // [n_kv] membership bits
    pub qb_rows: Vec<u32>,   // union rows
    pub qb_mrows: Vec<u8>,   // membership per union row
    // MoE prefill: one expert group's quantised activations gathered contiguously
    pub grp_q: Vec<i8>,
    pub grp_s: Vec<f32>,
    pub grp_sum: Vec<i32>,
}

impl CoreScratch {
    #[inline]
    fn qz(hiprec: bool, q: &mut QAct, x: &[f32]) {
        if hiprec {
            q.quantize_pair(x);
        } else {
            q.quantize(x);
        }
    }
}

#[derive(Clone, Copy)]
pub struct ScratchDims {
    hc_hidden: usize,
    hc_lr: usize,
    hidden: usize,
    v_local: usize,
    head_dim: usize,
    d_state: usize,
    k_alloc: usize,
    hiprec: bool,
}

impl CoreScratch {
    fn new(d: &ScratchDims) -> CoreScratch {
        CoreScratch {
            hiprec: d.hiprec,
            res: vec![0.0; d.hc_hidden],
            xn: vec![0.0; d.hc_hidden],
            tmp: vec![0.0; d.hc_hidden],
            lo: vec![0.0; d.hc_lr],
            vec_v: vec![0.0; d.v_local],
            emb: vec![0.0; d.hidden],
            inject: [0.0; 8],
            q_hc: QAct::zeros(2, d.hc_hidden, 32),
            q_lr: QAct::zeros(2, d.hc_lr, 32),
            q_h: QAct::zeros(2, d.hidden, 32),
            q_v: QAct::zeros(2, d.v_local, 32),
            ids: vec![0; d.k_alloc],
            w: vec![0.0; d.k_alloc],
            attn_scratch: Vec::new(),
            head_tmp: vec![0.0; d.head_dim.max(d.d_state) * 4],
            sel_idx: Vec::new(),
            sel_ranges: Vec::new(),
            qb_q: vec![0.0; tr_kernels::attn::QB * d.head_dim],
            qb_out: vec![0.0; tr_kernels::attn::QB * d.head_dim],
            qb_scores: Vec::new(),
            qb_memb: Vec::new(),
            qb_rows: Vec::new(),
            qb_mrows: Vec::new(),
            grp_q: Vec::new(),
            grp_s: Vec::new(),
            grp_sum: Vec::new(),
        }
    }
}

pub struct Dump {
    pub entries: Mutex<Vec<(String, Vec<f32>)>>,
}

/// Per-phase wall time accumulated on worker (0,0) across steps (µs). Enabled by TR_PROFILE=1.
pub struct Profile {
    pub enabled: bool,
    pub phases: Mutex<std::collections::BTreeMap<&'static str, (f64, u64)>>,
    /// Per-worker nanoseconds for the batch phases (moe, gdn, attn), indexed by ctx.global.
    pub pw: Vec<[std::sync::atomic::AtomicU64; PW_N]>,
    pub pw_n: [std::sync::atomic::AtomicU64; PW_N],
    pub cores_per_node: usize,
}
pub const PW_MOE: usize = 0;
pub const PW_GDN: usize = 1;
pub const PW_ATTN: usize = 2;
pub const PW_GDN_PROJ: usize = 3;
pub const PW_GDN_W1: usize = 4;
pub const PW_GDN_CONV: usize = 5;
pub const PW_GDN_DELTA: usize = 6;
pub const PW_GDN_OUT: usize = 7;
pub const PW_N: usize = 8;
impl Profile {
    pub fn new(n_workers: usize, cores_per_node: usize) -> Profile {
        Profile {
            enabled: std::env::var("TR_PROFILE").map(|v| v == "1").unwrap_or(false),
            phases: Mutex::new(Default::default()),
            pw: (0..n_workers).map(|_| Default::default()).collect(),
            pw_n: Default::default(),
            cores_per_node,
        }
    }
    /// Record `t0.elapsed()` for this worker in batch phase `ph` (all workers call it).
    #[inline]
    pub(crate) fn pw_add(&self, ctx: &WorkerCtx, ph: usize, t0: std::time::Instant) {
        use std::sync::atomic::Ordering::Relaxed;
        self.pw[ctx.global][ph].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        if ctx.global == 0 {
            self.pw_n[ph].fetch_add(1, Relaxed);
        }
    }
    /// Per-tile summary (min / mean / max over the tile's cores, µs per step) of the batch phases.
    pub fn pw_report(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        let mut s = String::new();
        let nc = self.cores_per_node.max(1);
        for (ph, name) in [(PW_MOE, "b.moe"), (PW_GDN, "b.gdn"), (PW_ATTN, "b.attn"), (PW_GDN_PROJ, "b.gdn.proj"), (PW_GDN_W1, "b.gdn.wait1"), (PW_GDN_CONV, "b.gdn.conv"), (PW_GDN_DELTA, "b.gdn.delta"), (PW_GDN_OUT, "b.gdn.out")] {
            let n = self.pw_n[ph].load(Relaxed).max(1) as f64;
            s.push_str(&format!("  {name}: per tile min/mean/max us per step (cores)\n"));
            for t in 0..self.pw.len() / nc {
                let v: Vec<f64> = (0..nc).map(|c| self.pw[t * nc + c][ph].load(Relaxed) as f64 / n / 1e3).collect();
                let mean = v.iter().sum::<f64>() / nc as f64;
                let (mn, mx) = v.iter().fold((f64::MAX, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
                let cmax = v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
                s.push_str(&format!("    tile {t}: {mn:8.1} {mean:8.1} {mx:8.1}  (max core {cmax})\n"));
            }
        }
        s
    }
    #[inline]
    pub(crate) fn lap(&self, t: &mut std::time::Instant, name: &'static str) {
        if self.enabled {
            let now = std::time::Instant::now();
            let e = &mut *self.phases.lock().unwrap();
            let ent = e.entry(name).or_insert((0.0, 0));
            ent.0 += (now - *t).as_secs_f64() * 1e6;
            ent.1 += 1;
            *t = now;
        }
    }
    /// Drop everything accumulated so far.
    pub fn clear(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.phases.lock().unwrap().clear();
        for w in &self.pw {
            for a in w {
                a.store(0, Relaxed);
            }
        }
        for a in &self.pw_n {
            a.store(0, Relaxed);
        }
    }
    pub fn report(&self) -> String {
        let e = self.phases.lock().unwrap();
        let total: f64 = e.values().map(|v| v.0).sum();
        let mut s = String::new();
        for (k, (us, n)) in e.iter() {
            s.push_str(&format!("  {k:14} {:9.1} us/step  ({:5.1}%)  x{}\n", us / (*n as f64).max(1.0), 100.0 * us / total.max(1e-9), n));
        }
        s
    }
}
impl Dump {
    pub fn new() -> Dump {
        Dump { entries: Mutex::new(Vec::new()) }
    }
    fn push(&self, name: String, v: &[f32]) {
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        let abs: f64 = v.iter().map(|&x| x.abs() as f64).sum();
        let mx = v.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let mut st = vec![sum as f32, abs as f32, mx, v.len() as f32];
        st.extend(v.iter().take(8));
        self.entries.lock().unwrap().push((name, st));
    }
}

fn _il_is_zero(il: usize) -> bool {
    il == 0
}
fn c_local_is_zero(ctx: &WorkerCtx) -> bool {
    ctx.local == 0
}
thread_local! {
    static PROF: std::cell::Cell<Option<(*const Profile, std::time::Instant)>> = const { std::cell::Cell::new(None) };
}
#[inline]
fn plap(name: &'static str) {
    PROF.with(|c| {
        if let Some((p, t)) = c.get() {
            let mut t = t;
            unsafe { (*p).lap(&mut t, name) };
            c.set(Some((p, t)));
        }
    });
}
thread_local! {
    static TILE_DUMP: std::cell::Cell<Option<(*const Dump, usize)>> = const { std::cell::Cell::new(None) };
}
/// Dump from core 0 of every tile (layer 0 only), name prefixed by tile.
fn dmp_tile(name: &str, v: &[f32]) {
    TILE_DUMP.with(|c| {
        if let Some((d, t)) = c.get() {
            unsafe { (*d).push(format!("t{t}:{name}"), v) };
        }
    });
}
thread_local! {
    static DUMP_CTX: std::cell::Cell<Option<(*const Dump, usize)>> = const { std::cell::Cell::new(None) };
}
/// Dump a vector from inside a block (worker (0,0) only, layer stored in the thread-local).
fn dmp_local(name: &str, v: &[f32]) {
    DUMP_CTX.with(|c| {
        if let Some((d, il)) = c.get() {
            unsafe { (*d).push(format!("{name}-{il}"), v) };
        }
    });
}

pub(crate) struct SyncCell<T>(pub(crate) UnsafeCell<T>);
unsafe impl<T> Sync for SyncCell<T> {}
unsafe impl<T> Send for SyncCell<T> {}

pub struct Model {
    pub(crate) sequences: Option<crate::sequence::Sequences>,
    pub(crate) segments: Vec<crate::sequence::Segment>,
    pub(crate) output_rows: Vec<usize>,
    pub cfg: ModelConfig,
    pub manifest: Manifest,
    pub tiles: Vec<TileWeights>,
    pub layers: Vec<Vec<LayerW>>, // [layer][tile]
    pub heads: Vec<HeadW>,        // [tile]
    /// MTP draft head per tile (overlay pack loaded).
    pub mtp: Option<Vec<MtpW>>,
    /// Vision encoder (overlay pack loaded): weights, workspaces and the last image's embeddings.
    pub vision: Option<crate::vision::Vision>,
    pub mbox: Mailboxes,
    pub(crate) ws: Vec<SyncCell<TileWs>>,
    pub(crate) state: Vec<SyncCell<TileState>>,
    pub(crate) batch_ws: Vec<SyncCell<crate::exec_batch::BatchWs>>,
    scratch: Vec<SyncCell<Option<CoreScratch>>>,
    scratch_dims: ScratchDims,
    pub pool: Option<Pool>,
    pub rope: RopeTable,
    pub vocab_per_tile: usize,
    pub ple_rows_per_tile: usize,
    pub n_past: usize,
    pub prev_tokens: Vec<u32>,
    /// Rope position of a text token at cell p is p + rope_delta: an image of nx*ny tokens
    /// advances the rope position by max(nx, ny) only (M-RoPE), so the delta is <= 0 after images.
    pub rope_delta: i64,
    pub load: LoadReport,
    pub profile: Profile,
    /// TR_QSA_DEBUG=1: score blocks even for dense queries (oracle checks of indexer_score).
    pub qsa_debug: bool,
    /// TR_QSA_OFF=1: dense attention at every length (A/B against the sparse selection).
    pub qsa_off: bool,
    /// AMX-BF16 GEMMs for the dense prefill projections (TR_AMX=0 disables).
    pub amx: bool,
    /// Minimum batch rows for the AMX path (TR_AMX_MIN_M; below it the VNNI kernels win).
    pub amx_min_m: usize,
    pub tiles_per_socket: usize,
    /// Draft tokens per speculative round the state was sized for (0 = off).
    pub spec_k: usize,
    /// K/V cache element type the state was allocated with.
    pub kv: KvType,
    /// Hashes of the loaded overlay packs (part of the prefix cache identity).
    pub mtp_hash: Option<String>,
    pub vision_hash: Option<String>,
    pub(crate) verify_pos0: usize,
    pub(crate) verify_toks: Vec<u32>,
    /// Index of the last committed verify row (bookkeeping for the MTP carry).
    pub last_commit: usize,
    /// The draft head's carry (main-model residual of the last committed position) is row j of
    /// `BatchWs::res_keep` rather than `TileWs::mtp_carry` (set by `commit`, consumed by the next
    /// draft chain; None after a decode step or a prefill, which write the carry directly).
    pub(crate) mtp_carry_row: Option<usize>,
    /// Drafts and their distributions from the last in-pool draft chain (written by worker 0).
    pub(crate) draft_out: std::sync::Mutex<Vec<(u32, crate::sampler::Dist)>>,
    /// Expert routing policy: None = the model's fixed top-k (`expert_used_count`), Some = cumulative
    /// router mass (`set_moe`). Read at every MoE block, so it can change between steps.
    moe: Option<MoePolicy>,
    /// Routing statistics: (token, MoE layer) pairs executed and experts selected in them.
    pub moe_stats: MoeStats,
}

/// Counters for the mean number of experts per token and layer (worker 0 counts), plus an optional
/// profile of the cumulative router mass at every k (fixed point, 1e-6).
#[derive(Default)]
pub struct MoeStats {
    pub rows: AtomicU64,
    pub experts: AtomicU64,
    pub profile: std::sync::atomic::AtomicBool,
    prof_rows: AtomicU64,
    cum: [AtomicU64; MOE_K_MAX],
}
impl MoeStats {
    /// Start (or restart) profiling the cumulative router mass.
    pub fn profile_start(&self) {
        self.prof_rows.store(0, Ordering::Relaxed);
        for c in &self.cum {
            c.store(0, Ordering::Relaxed);
        }
        self.profile.store(true, Ordering::Relaxed);
    }
    /// Mean cumulative mass of the best k experts, k = 1..=MOE_K_MAX (empty when nothing was profiled).
    pub fn profile_report(&self) -> Vec<f64> {
        let n = self.prof_rows.load(Ordering::Relaxed);
        if n == 0 {
            return Vec::new();
        }
        self.cum.iter().map(|c| c.load(Ordering::Relaxed) as f64 * 1e-6 / n as f64).collect()
    }
    pub(crate) fn profile_row(&self, logits: &[f32]) {
        if !self.profile.load(Ordering::Relaxed) {
            return;
        }
        let mut cum = [0f32; MOE_K_MAX];
        tr_kernels::router::mass_profile(logits, &mut cum);
        for (c, &m) in self.cum.iter().zip(&cum) {
            c.fetch_add((m as f64 * 1e6) as u64, Ordering::Relaxed);
        }
        self.prof_rows.fetch_add(1, Ordering::Relaxed);
    }
    pub fn snapshot(&self) -> (u64, u64) {
        (self.rows.load(Ordering::Relaxed), self.experts.load(Ordering::Relaxed))
    }
    /// Mean experts per (token, layer) since `since` (a `snapshot`).
    pub fn mean_since(&self, since: (u64, u64)) -> f64 {
        let (r, e) = self.snapshot();
        if r > since.0 { (e - since.1) as f64 / (r - since.0) as f64 } else { 0.0 }
    }
    pub(crate) fn add(&self, rows: u64, experts: u64) {
        self.rows.fetch_add(rows, Ordering::Relaxed);
        self.experts.fetch_add(experts, Ordering::Relaxed);
    }
}

const LAYOUT_LO_PAD: usize = 16;

impl Model {
    pub fn load(pack: &Path, ctx_max: usize, cores_per_node: Option<usize>) -> Result<Model> {
        Self::load_with(pack, ctx_max, cores_per_node, &LoadOptions::default())
    }
    /// Expert routing policy in effect (None = the model's fixed top-k).
    pub fn moe_policy(&self) -> Option<MoePolicy> {
        self.moe
    }
    /// Switch the expert routing policy (validated against the expert count; workspaces hold up to
    /// `MOE_K_MAX` experts per token, so no reallocation).
    pub fn set_moe(&mut self, p: Option<MoePolicy>) -> Result<()> {
        if let Some(p) = &p {
            p.validate(self.cfg.n_expert).map_err(|e| anyhow::anyhow!(e))?;
        }
        self.moe = p;
        Ok(())
    }
    /// Route one token's expert logits into `ids`/`w` (descending score, weights renormalised) and
    /// return the number of experts selected. Every core of a tile calls this on the same logits and
    /// gets the same answer (no cross-core state).
    #[inline]
    pub(crate) fn route(&self, logits: &[f32], ids: &mut [u32], w: &mut [f32]) -> usize {
        match &self.moe {
            Some(p) => tr_kernels::router::route_mass(logits, p, ids, w),
            None => {
                let k = self.cfg.n_expert_used;
                tr_kernels::router::route_topk(logits, k, ids, w);
                k
            }
        }
    }
    pub fn load_with(pack: &Path, ctx_max: usize, cores_per_node: Option<usize>, opts: &LoadOptions) -> Result<Model> {
        let manifest = Manifest::load(pack)?;
        let cfg = ModelConfig::from_manifest(&manifest)?;
        if let Some(p) = &opts.moe {
            p.validate(cfg.n_expert).map_err(|e| anyhow::anyhow!(e))?;
        }
        let topo = tr_sys::topology::Topology::discover()?;
        // XTILEDATA permission is per process and inherited by threads: request it before the pool
        let amx = tr_kernels::amx::init();
        let mut pool = Pool::new(&topo, cores_per_node)?;
        let t = cfg.n_tiles;
        let n_attn = cfg.n_layer / cfg.full_interval + 1;
        let kv_bytes = 2 * ctx_max * cfg.head_dim * opts.kv.elem_bytes() * n_attn + cfg.qsa_blocks_max(ctx_max) * cfg.idx_dim * 4 * n_attn + (1 << 20);
        let overlay = match &opts.mtp {
            Some(dir) => Some(Manifest::load(dir).with_context(|| format!("MTP overlay pack {}", dir.display()))?),
            None => None,
        };
        let n_ckpt = if overlay.is_some() || opts.spec_k > 0 { opts.spec_k + 1 } else { 0 };
        let (vision_man, vision_cfg) = match &opts.vision {
            Some(dir) => {
                let vm = Manifest::load(dir).with_context(|| format!("vision overlay pack {}", dir.display()))?;
                let vc = crate::vision::VisionCfg::from_manifest(&vm, t, opts.image_min_tokens, opts.image_max_tokens)?;
                if vc.proj != cfg.hidden {
                    anyhow::bail!("vision pack projects to {} but the model's hidden size is {}", vc.proj, cfg.hidden);
                }
                if opts.batch_max < 2 {
                    anyhow::bail!("the vision encoder needs the batch path (--batch >= 2)");
                }
                if !amx {
                    anyhow::bail!("the vision encoder needs AMX (bf16 weights); TR_AMX=0 is not supported with --vision");
                }
                (Some(vm), Some(vc))
            }
            None => (None, None),
        };
        let vision_bytes = vision_cfg.as_ref().map(|vc| crate::vision::VisionWs::bytes(vc, opts.batch_max.max(1), pool.cores_per_node())).unwrap_or(0);
        let mtp_kv_bytes = if overlay.is_some() { 2 * ctx_max * cfg.head_dim * opts.kv.elem_bytes() + cfg.qsa_blocks_max(ctx_max) * cfg.idx_dim * 4 + (1 << 20) } else { 0 };
        let bm = opts.batch_max.max(1);
        let batch_bytes = if bm > 1 { bm * (cfg.hc * cfg.hidden * 4 * 6 + cfg.hidden * 4 * 8 + (MOE_K_MAX + 1) * (cfg.hidden * 4 + 3 * cfg.n_ff / t)) + (16 << 20) + pool.cores_per_node() * crate::exec_batch::AMX_BUF_BYTES } else { 0 };
        let state_bytes = kv_bytes + 40 * (cfg.n_v_heads / t) * cfg.d_state * cfg.d_state * 4 + (64 << 20) + batch_bytes + n_ckpt * (TileState::ckpt_bytes(&cfg) + (1 << 20)) + mtp_kv_bytes + vision_bytes;
        let extra: Vec<&Manifest> = vision_man.iter().collect();
        let (mut tiles, load) = load_weights(&manifest, &cfg, &mut pool, state_bytes, opts, overlay.as_ref(), &extra)?;
        let vision = match vision_cfg {
            Some(vc) => Some(Self::vision_init(&mut tiles, vc, opts.batch_max.max(1), pool.cores_per_node())?),
            None => None,
        };
        // Subset packs (tests) carry only some layers / experts; run exactly what is packed.
        let layer_ids = manifest.layer_ids.clone();
        let layers: Vec<Vec<LayerW>> = layer_ids.iter().map(|&il| tiles.iter().map(|tw| layer_w(tw, &cfg, il)).collect()).collect();
        let heads: Vec<HeadW> = tiles
            .iter()
            .map(|tw| HeadW {
                hc: HcW { norm: tw.f32("output_hc_norm.weight"), inject: None, down: *tw.tq("output_hc_down.weight"), up: *tw.tq("output_hc_up.weight") },
                output: *tw.tq("output.weight"),
                embd: *tw.tq("token_embd.weight"),
            })
            .collect();
        let mtp: Option<Vec<MtpW>> = overlay.as_ref().map(|_| {
            let il = cfg.mtp_layer();
            tiles
                .iter()
                .map(|tw| MtpW {
                    layer: layer_w_kind(tw, &cfg, il, false),
                    hc: HcW { norm: tw.f32("mtp_hc_norm.weight"), inject: None, down: *tw.tq("mtp_hc_down.weight"), up: *tw.tq("mtp_hc_up.weight") },
                    fc_emb: *tw.tq("mtp_fc_embedding.weight"),
                    fc_hid: *tw.tq("mtp_fc_hidden.weight"),
                    enorm: tw.f32("mtp_enorm.weight"),
                    hnorm: tw.f32("mtp_hnorm.weight"),
                })
                .collect()
        });
        let vocab_per_tile = heads[0].output.rows;
        let ple_rows_per_tile = tiles[0].raw("per_layer_token_embd.weight").map(|r| r.rows).unwrap_or(0);
        let batch_max = opts.batch_max.max(1);
        let idx_per_tile = (cfg.idx_heads * cfg.idx_dim + cfg.idx_dim) / t;
        // tiles sharing a socket (SLIT distance <= 14 from tile 0's node); the batch all-reduce keeps
        // most traffic inside a socket. Falls back to one socket if the grouping is not contiguous.
        let tiles_per_socket = {
            let same: Vec<usize> = (0..t).filter(|&u| topo.nodes.first().and_then(|n0| n0.distances.get(u)).map(|&d| d <= 14).unwrap_or(true)).collect();
            let tps = same.len().max(1);
            let contiguous = t % tps == 0 && (0..t).all(|u| (u / tps == 0) == same.contains(&u));
            if contiguous { tps } else { t }
        };
        let layout = MailboxLayout::new(cfg.hidden, cfg.hc, cfg.hc_lr, cfg.n_expert, t, vocab_per_tile, idx_per_tile, batch_max, t / tiles_per_socket, n_ckpt);
        let mut arenas: Vec<&mut tr_sys::numa::Arena> = tiles.iter_mut().map(|tw| &mut tw.arena).collect();
        let mbox = Mailboxes::new(&mut arenas, layout)?;
        let mut ws = Vec::new();
        let mut state = Vec::new();
        let mut batch_ws = Vec::new();
        let c_local = (2 * cfg.n_k_heads / t + cfg.n_v_heads / t) * cfg.d_state;
        let v_local = cfg.n_v_heads / t * cfg.d_state;
        let qh_local = cfg.n_head / t;
        let ff_local = cfg.n_ff / t;
        let slots = MOE_K_MAX + 1; // routing policies may select up to MOE_K_MAX experts
        for a in arenas.iter_mut() {
            let w = TileWs {
                xn: a.alloc_slice(cfg.hc * cfg.hidden)?,
                xq_q: a.alloc_slice(2 * cfg.hc * cfg.hidden)?,
                xq_s: a.alloc_slice(2 * cfg.hc * cfg.hidden / 32)?,
                xq_sum: a.alloc_slice(2 * cfg.hc * cfg.hidden / 32)?,
                part: a.alloc_slice(32 * 2 * 8)?,
                gate_local: a.alloc_slice(cfg.hc * cfg.hidden / t)?,
                mixed_full: a.alloc_slice(cfg.hidden)?,
                red: a.alloc_slice(cfg.hidden)?,
                rlog: a.alloc_slice(cfg.n_expert)?,
                qkv: a.alloc_slice(c_local)?,
                qkvc: a.alloc_slice(c_local)?,
                z: a.alloc_slice(v_local)?,
                beta: a.alloc_slice(16)?,
                alpha: a.alloc_slice(16)?,
                y: a.alloc_slice(v_local)?,
                qg: a.alloc_slice(qh_local * 2 * cfg.head_dim)?,
                kn: a.alloc_slice(cfg.head_dim)?,
                vn: a.alloc_slice(cfg.head_dim)?,
                o: a.alloc_slice(qh_local * cfg.head_dim)?,
                hg: a.alloc_slice(slots * ff_local)?,
                hu: a.alloc_slice(slots * ff_local)?,
                act_q: a.alloc_slice(2 * slots * ff_local)?,
                act_s: a.alloc_slice(2 * slots * ff_local / 16)?,
                act_sum: a.alloc_slice(2 * slots * ff_local / 16)?,
                ple_emb: a.alloc_slice(cfg.hidden)?,
                ple_key: a.alloc_slice(cfg.hc * cfg.hidden / t)?,
                ple_val: a.alloc_slice(cfg.hidden / t)?,
                ple_stats: a.alloc_slice(2 * cfg.hc + 16)?,
                idx_qraw: a.alloc_slice(cfg.idx_heads * cfg.idx_dim)?,
                idx_kraw: a.alloc_slice(cfg.idx_dim)?,
                idx_q: a.alloc_slice(cfg.idx_heads * cfg.idx_dim)?,
                idx_scores: a.alloc_slice(cfg.qsa_blocks_max(ctx_max))?,
                qn: a.alloc_slice(qh_local * cfg.head_dim)?,
                sel: a.alloc_slice(2 * (cfg.idx_top_k + 8) + 2)?,
                attn_part: a.alloc_slice(qh_local * 16 * (cfg.head_dim + 16))?,
                mtp_carry: a.alloc_slice(cfg.hc * cfg.hidden)?,
                mtp_res: a.alloc_slice(cfg.hc * cfg.hidden)?,
                cand_core: a.alloc_slice(pool.cores_per_node() * 2 * crate::exchange::CAND_PER_TILE)?,
                draft_tok: a.alloc_slice(16)?,
            };
            ws.push(SyncCell(UnsafeCell::new(w)));
            state.push(SyncCell(UnsafeCell::new(TileState::new(&cfg, a, ctx_max, opts.kv, n_ckpt, mtp.is_some())?)));
            if batch_max > 1 {
                batch_ws.push(SyncCell(UnsafeCell::new(crate::exec_batch::BatchWs::new(a, batch_max, cfg.hidden, cfg.hc, t, cfg.hc_lr, cfg.n_expert, MOE_K_MAX, c_local, v_local, qh_local, cfg.head_dim, ff_local, cfg.idx_heads * cfg.idx_dim, cfg.idx_dim, cfg.qsa_blocks_max(ctx_max), pool.cores_per_node())?)));
            }
        }
        let hiprec = std::env::var("TR_HIPREC").map(|v| v != "0").unwrap_or(true);
        let (n_workers, cpn) = (pool.n_workers(), pool.cores_per_node());
        let scratch_dims = ScratchDims { hc_hidden: cfg.hc * cfg.hidden, hc_lr: cfg.hc_lr, hidden: cfg.hidden, v_local: v_local.max(qh_local * cfg.head_dim), head_dim: cfg.head_dim, d_state: cfg.d_state, k_alloc: MOE_K_MAX, hiprec };
        let scratch: Vec<SyncCell<Option<CoreScratch>>> = (0..pool.n_workers()).map(|_| SyncCell(UnsafeCell::new(None))).collect();
        let rope = RopeTable::new_mrope(cfg.rope_dim, cfg.rope_base, cfg.rope_sections);
        let mut cfg = cfg;
        cfg.n_expert_packed = manifest.n_expert_packed;
        let amx_min_m = std::env::var("TR_AMX_MIN_M").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        Ok(Model { sequences: None, segments: Vec::new(), output_rows: Vec::new(), cfg, manifest, tiles, layers, heads, mtp, vision, mbox, ws, state, batch_ws, scratch, scratch_dims, pool: Some(pool), rope, vocab_per_tile, ple_rows_per_tile, n_past: 0, prev_tokens: Vec::new(), rope_delta: 0, load, profile: Profile::new(n_workers, cpn), qsa_debug: std::env::var("TR_QSA_DEBUG").map(|v| v == "1").unwrap_or(false), qsa_off: std::env::var("TR_QSA_OFF").map(|v| v == "1").unwrap_or(false), amx, amx_min_m, tiles_per_socket, spec_k: opts.spec_k, kv: opts.kv, mtp_hash: overlay.as_ref().map(|o| o.hash.clone()), vision_hash: vision_man.as_ref().map(|v| v.hash.clone()), verify_pos0: 0, verify_toks: Vec::new(), last_commit: 0, mtp_carry_row: None, draft_out: std::sync::Mutex::new(Vec::new()), moe: opts.moe, moe_stats: MoeStats::default() })
    }

    pub fn reset(&mut self) {
        for s in &self.state {
            unsafe { (*s.0.get()).reset() };
        }
        self.n_past = 0;
        self.prev_tokens.clear();
        self.verify_toks.clear();
        self.mtp_carry_row = None;
        self.rope_delta = 0;
    }

    /// M-RoPE position triple of the text token at cell `cell`.
    #[inline]
    pub fn rpos(&self, cell: usize) -> [u32; 3] {
        let p = (cell as i64 + self.rope_delta).max(0) as u32;
        [p, p, p]
    }

    /// Run one token through the model at position `n_past`; returns the full logits.
    pub fn step(&mut self, tok: u32, dump: Option<&Dump>) -> Vec<f32> {
        let pos = self.n_past;
        self.mtp_carry_row = None; // the worker writes the carry itself
        let prev: Vec<u32> = self.prev_tokens.clone();
        {
            let mut pool = self.pool.take().expect("pool");
            let me: &Model = &*self;
            let prev_ref = &prev;
            pool.run(move |ctx| me.worker(ctx, tok, pos, prev_ref, dump));
            self.pool = Some(pool);
        }
        // gather logits
        let mut logits = vec![0f32; self.vocab_per_tile * self.cfg.n_tiles];
        for t in 0..self.cfg.n_tiles {
            let src = self.mbox.slot_ro(t, self.mbox.layout.logits);
            logits[t * self.vocab_per_tile..(t + 1) * self.vocab_per_tile].copy_from_slice(&src[..self.vocab_per_tile]);
        }
        logits.truncate(self.cfg.n_vocab);
        self.n_past += 1;
        self.prev_tokens.push(tok);
        let keep = self.cfg.ple_ngram.saturating_sub(1);
        if self.prev_tokens.len() > keep {
            let drop = self.prev_tokens.len() - keep;
            self.prev_tokens.drain(..drop);
        }
        logits
    }

    pub(crate) fn core_scratch(&self, ctx: &WorkerCtx) -> &mut CoreScratch {
        unsafe {
            let slot = &mut *self.scratch[ctx.global].0.get();
            if slot.is_none() {
                // allocated (and first-touched) by the worker itself: lands on its own tile
                *slot = Some(CoreScratch::new(&self.scratch_dims));
            }
            slot.as_mut().unwrap()
        }
    }

    /// Final hyper-connection mix + lm_head for the residual in `cs.res`; logits land in the
    /// LOGITS mailbox rows of every tile. Ends with a global barrier.
    pub(crate) fn head_step(&self, ctx: &WorkerCtx, cs: &mut CoreScratch, t: usize) {
        let hc = self.heads[t].hc;
        self.head_step_with(ctx, cs, t, &hc);
    }
    /// `head_step` with an explicit output mixer (the MTP draft head has its own, sharing `output.weight`).
    pub(crate) fn head_step_with(&self, ctx: &WorkerCtx, cs: &mut CoreScratch, t: usize, mixer: &HcW) {
        let cfg = &self.cfg;
        let (hh, tiles) = (cfg.hidden, cfg.n_tiles);
        let ws: &mut TileWs = unsafe { &mut *self.ws[t].0.get() };
        let hw = &self.heads[t];
        let lay = &self.mbox.layout;
        self.hc_mix_a(ctx, ws, cs, mixer, t);
        ctx.barrier();
        self.hc_mix_b(ctx, ws, cs, mixer, t);
        ctx.barrier();
        self.mbox.gather(ctx, lay.mixed, hh / tiles, ws.mixed_full);
        CoreScratch::qz(cs.hiprec, &mut cs.q_h, ws.mixed_full);
        let r = ctx.range_local(hw.output.n_strips());
        let out = self.mbox.slot(t, lay.logits);
        unsafe {
            gemv_tq(hw.output.ptr.add(r.start * hw.output.strip_bytes()), hw.output.k, hw.output.codec, r.start, r.end, cs.q_h.as_ref(), out.as_mut_ptr(), 0, false);
        }
        ctx.barrier();
    }

    // -----------------------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn worker(&self, ctx: &WorkerCtx, tok: u32, pos: usize, prev: &[u32], dump: Option<&Dump>) {
        let cfg = &self.cfg;
        let t = ctx.node;
        let c = ctx.local;
        let nc = ctx.cores_per_node;
        let hh = cfg.hidden;
        let hc = cfg.hc;
        let tiles = cfg.n_tiles;
        let ws: &mut TileWs = unsafe { &mut *self.ws[t].0.get() };
        let st: &mut TileState = unsafe { &mut *self.state[t].0.get() };
        let cs: &mut CoreScratch = self.core_scratch(ctx);
        let mb = &self.mbox;
        let lay = &mb.layout;
        let is_dumper = dump.is_some() && ctx.global == 0;
        let prof = ctx.global == 0 && self.profile.enabled;
        let mut tm = std::time::Instant::now();
        PROF.with(|c| c.set(if prof { Some((&self.profile as *const Profile, tm)) } else { None }));
        macro_rules! lap { ($n:expr) => { if prof { self.profile.lap(&mut tm, $n); PROF.with(|c| c.set(Some((&self.profile as *const Profile, tm)))); } } }
        let dmp = |name: &str, v: &[f32]| {
            if is_dumper {
                dump.unwrap().push(name.to_string(), v);
            }
        };

        // ---- token embedding: owner tile publishes the row
        let owner = tok as usize / self.vocab_per_tile;
        if t == owner && c == 0 {
            let slot = mb.slot(t, lay.emb);
            dequant_tq_row(&self.heads[t].embd, tok as usize % self.vocab_per_tile, &mut slot[..hh]);
        }
        ctx.barrier();
        {
            let e = &mb.slot_ro(owner, lay.emb)[..hh];
            for s in 0..hc {
                cs.res[s * hh..(s + 1) * hh].copy_from_slice(e);
            }
        }
        dmp("inp_embd", &cs.res[..hh]);
        lap!("embed");

        let dump_layer: Option<usize> = std::env::var("TR_DUMP_LAYER").ok().and_then(|v| v.parse().ok());
        let dump_path = std::env::var("TR_DUMP_VEC").ok();
        let wr = |name: &str, v: &[f32]| {
            if let Some(path) = &dump_path {
                let _ = std::fs::write(format!("{path}.{name}.bin"), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
            }
        };
        for (li, &il) in self.manifest.layer_ids.iter().enumerate() {
            let lw = &self.layers[li][t];
            let dl = ctx.global == 0 && dump_layer == Some(il) && pos == 0;
            if dl {
                wr("res_in", &cs.res);
            }
            DUMP_CTX.with(|c| c.set(if is_dumper { Some((dump.unwrap() as *const Dump, il)) } else { None }));
            TILE_DUMP.with(|c| c.set(if dump.is_some() && c_local_is_zero(ctx) && il == 0 { Some((dump.unwrap() as *const Dump, t)) } else { None }));
            if let Some(ple) = &lw.ple {
                self.ple_block(ctx, ws, st, cs, ple, tok, prev, il);
                lap!("ple");
            }
            // ---- S0/S1: hc_attn mix part 1 + hc_down -> LO
            self.hc_mix_a(ctx, ws, cs, &lw.hc_attn, t);
            lap!("hc_a");
            ctx.barrier(); // A
            lap!("bar_A");
            // ---- S2: lo gather, silu, hc_up -> gate_local; mixed slice -> MIXED
            self.hc_mix_b(ctx, ws, cs, &lw.hc_attn, t);
            lap!("hc_b");
            ctx.barrier(); // B
            lap!("bar_B");
            // ---- S3: gather mixed, token mixer -> PART
            mb.gather(ctx, lay.mixed, hh / tiles, ws.mixed_full);
            if il == 0 {
                dmp("hc_mixed-0", ws.mixed_full);
            }
            if dl {
                wr("mixed_attn", ws.mixed_full);
                wr("inject_attn", &cs.inject[..hc]);
            }
            lap!("gather");
            if ctx.global == 0 && (il == 3 || il == 0) {
                if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                    let _ = std::fs::write(format!("{path}.l{il}_in.p{pos}.bin"), ws.mixed_full.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
            if let Some(g) = &lw.gdn {
                self.gdn_block(ctx, ws, st, cs, g, il, t, pos);
                lap!("gdn");
            } else {
                self.attn_block(ctx, ws, st, cs, lw.attn.as_ref().unwrap(), il, pos, t);
                lap!("attn");
            }
            ctx.barrier(); // C
            lap!("bar_C");
            // ---- S4: reduce, combine, hc_ffn mix part 1 -> LO
            mb.reduce(ctx, lay.part, hh, ws.red);
            lap!("reduce");
            if ctx.global == 0 && il == 3 {
                if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                    let _ = std::fs::write(format!("{path}.l3_out.p{pos}.bin"), ws.red.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
            dmp(&format!("attn_out-{il}"), ws.red);
            dmp(&format!("inject_attn-{il}"), &cs.inject[..hc]);
            if dl {
                wr("mixer_out", ws.red);
            }
            hc_combine(&mut cs.res, ws.red, &cs.inject, hc, hh);
            dmp(&format!("hc_combine-{il}"), &cs.res);
            for s in 1..hc {
                dmp(&format!("hc_combine_s{s}-{il}"), &cs.res[s * hh..s * hh + 8]);
            }
            self.hc_mix_a(ctx, ws, cs, &lw.hc_ffn, t);
            for s in 0..hc {
                dmp(&format!("hc_norm_ffn_s{s}-{il}"), &ws.xn[s * hh..s * hh + 8]);
            }
            lap!("hc_a");
            ctx.barrier(); // D
            lap!("bar_D");
            // ---- S5: lo gather, hc_up, mixed slice -> MIXED, router partial -> RLOG
            self.hc_mix_b(ctx, ws, cs, &lw.hc_ffn, t);
            dmp(&format!("hc_lo_ffn-{il}"), &cs.lo[..8]);
            dmp(&format!("hc_gate_ffn-{il}"), &ws.gate_local[..8]);
            {
                // router partial from the tile's mixed slice (private copy in cs.tmp[..hh/tiles])
                let hl = hh / tiles;
                let r = ctx.range_local(cfg.n_expert);
                let out = mb.slot(t, lay.rlog);
                for e in r {
                    out[e] = dot(&lw.moe.router[e * hl..(e + 1) * hl], &cs.tmp[..hl]);
                }
            }
            lap!("hc_b");
            ctx.barrier(); // E
            lap!("bar_E");
            // ---- S6: gather mixed + router, MoE -> PART
            mb.gather(ctx, lay.mixed, hh / tiles, ws.mixed_full);
            mb.reduce(ctx, lay.rlog, cfg.n_expert, ws.rlog);
            dmp(&format!("hc_mixed_ffn-{il}"), ws.mixed_full);
            dmp(&format!("inject_ffn-{il}"), &cs.inject[..hc]);
            if dl {
                wr("res_mid", &cs.res);
                wr("mixed_ffn", ws.mixed_full);
                wr("inject_ffn", &cs.inject[..hc]);
                wr("rlog", ws.rlog);
            }
            lap!("gather");
            if ctx.global == 0 && il == 0 && pos == 0 {
                if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                    let _ = std::fs::write(format!("{path}.moe_in.bin"), ws.mixed_full.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                    let _ = std::fs::write(format!("{path}.rlog.bin"), ws.rlog.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
            self.moe_block(ctx, ws, cs, &lw.moe, il, t, nc);
            lap!("moe");
            ctx.barrier(); // G
            lap!("bar_G");
            // ---- S7: reduce, combine
            mb.reduce(ctx, lay.part, hh, ws.red);
            if ctx.global == 0 && il == 0 && pos == 0 {
                if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                    let _ = std::fs::write(format!("{path}.moe_out.bin"), ws.red.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
            dmp(&format!("ffn_out-{il}"), ws.red);
            if dl {
                wr("ffn_out", ws.red);
            }
            hc_combine(&mut cs.res, ws.red, &cs.inject, hc, hh);
            dmp(&format!("l_last-{il}"), &cs.res);
            if dl {
                wr("res_out", &cs.res);
            }
            lap!("reduce");
        }

        DUMP_CTX.with(|c| c.set(None));
        if ctx.global == 0 {
            if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                let _ = std::fs::write(format!("{path}.head_res.p{pos}.bin"), cs.res.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
        }
        if self.mtp.is_some() && c == 0 {
            // the draft head's input for the next position (every core holds the same residual)
            ws.mtp_carry.copy_from_slice(&cs.res);
        }
        // ---- head
        let hw = &self.heads[t];
        self.hc_mix_a(ctx, ws, cs, &hw.hc, t);
        ctx.barrier();
        self.hc_mix_b(ctx, ws, cs, &hw.hc, t);
        ctx.barrier();
        mb.gather(ctx, lay.mixed, hh / tiles, ws.mixed_full);
        dmp("result_norm", ws.mixed_full);
        CoreScratch::qz(cs.hiprec, &mut cs.q_h, ws.mixed_full);
        {
            let r = ctx.range_local(hw.output.n_strips());
            let out = mb.slot(t, lay.logits);
            unsafe {
                gemv_tq(hw.output.ptr.add(r.start * hw.output.strip_bytes()), hw.output.k, hw.output.codec, r.start, r.end, cs.q_h.as_ref(), out.as_mut_ptr(), 0, false);
            }
        }
        lap!("head");
        ctx.barrier();
        lap!("bar_head");
        if ctx.global == 0 {
            if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                let mut all = Vec::with_capacity(cfg.n_vocab);
                for u in 0..tiles {
                    all.extend_from_slice(&mb.slot_ro(u, lay.logits)[..self.vocab_per_tile]);
                }
                let _ = std::fs::write(format!("{path}.logits.p{pos}.bin"), all.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
        }
        if is_dumper {
            let mut all = Vec::with_capacity(cfg.n_vocab);
            for u in 0..tiles {
                all.extend_from_slice(&mb.slot_ro(u, lay.logits)[..self.vocab_per_tile]);
            }
            all.truncate(cfg.n_vocab);
            dmp("result_output", &all);
        }
    }

    /// Part 1 of a hyper-connection mix, split across the tile's cores: per-stream rmsnorm * gamma
    /// into the tile-shared `xn`, inject partials, pair-quantised `xn`, then this core's strips of the
    /// low-rank down projection straight into the LO mailbox. Two node barriers.
    fn hc_mix_a(&self, ctx: &WorkerCtx, ws: &mut TileWs, cs: &mut CoreScratch, w: &HcW, t: usize) {
        let cfg = &self.cfg;
        let (hh, hc) = (cfg.hidden, cfg.hc);
        let c = ctx.local;
        let nc = ctx.cores_per_node;
        // block-aligned element range of each stream for this core
        let nblk = hh / 32;
        let br = ctx.range_local(nblk);
        let r = br.start * 32..br.end * 32;
        // phase 1: partial sums of squares per stream
        for s in 0..hc {
            ws.part[c * 2 * hc + s] = elem::sumsq(&cs.res[s * hh + r.start..s * hh + r.end]);
        }
        ctx.node_barrier();
        let mut rstd = [0f32; 8];
        for s in 0..hc {
            let mut ss = 0f32;
            for cc in 0..nc {
                ss += ws.part[cc * 2 * hc + s];
            }
            rstd[s] = 1.0 / (ss / hh as f32 + cfg.rms_eps).sqrt();
        }
        // phase 2: xn on my range, inject partials, quantise my blocks
        for s in 0..hc {
            let base = s * hh;
            let src = &cs.res[base + r.start..base + r.end];
            let wn = &w.norm[base + r.start..base + r.end];
            let dst = &mut ws.xn[base + r.start..base + r.end];
            for i in 0..src.len() {
                dst[i] = src[i] * rstd[s] * wn[i];
            }
        }
        for s in 0..hc {
            let mut acc = 0f32;
            if let Some(inj) = w.inject {
                for s2 in 0..hc {
                    let base = s2 * hh;
                    acc += dot(&inj[s * hc * hh + base + r.start..s * hc * hh + base + r.end], &ws.xn[base + r.start..base + r.end]);
                }
            }
            ws.part[c * 2 * hc + hc + s] = acc;
        }
        ctx.node_barrier();
        // quantise this tile's K-slice of xn (the input of the K-split down projection)
        let kslice = hc * hh / cfg.n_tiles;
        let ks0 = t * kslice;
        let kblk = kslice / 32;
        let kb_r = ctx.range_local(kblk);
        {
            let xs = &ws.xn[ks0..ks0 + kslice];
            if cs.hiprec {
                tr_kernels::quant::quantize_pair_blocks(xs, kslice, 32, kb_r.start, kb_r.end, ws.xq_q, ws.xq_s, ws.xq_sum);
            } else {
                for b in kb_r.clone() {
                    let (sc, su) = unsafe { tr_kernels::quant::quant_block(&xs[b * 32..(b + 1) * 32], &mut ws.xq_q[b * 32..(b + 1) * 32]) };
                    ws.xq_s[b] = sc;
                    ws.xq_sum[b] = su;
                }
            }
        }
        ctx.node_barrier();
        if w.inject.is_some() {
            for s in 0..hc {
                let mut acc = 0f32;
                for cc in 0..nc {
                    acc += ws.part[cc * 2 * hc + hc + s];
                }
                cs.inject[s] = acc;
            }
        }
        let xq = self.xq_ref(ws, cs.hiprec, kslice, kblk);
        let sr = ctx.range_local(w.down.n_strips());
        let out = self.mbox.slot(t, self.mbox.layout.lo);
        unsafe {
            gemv_tq(w.down.ptr.add(sr.start * w.down.strip_bytes()), w.down.k, w.down.codec, sr.start, sr.end, xq, out.as_mut_ptr(), 0, false);
        }
    }

    #[inline]
    fn xq_ref<'a>(&self, ws: &'a TileWs, hiprec: bool, k: usize, nb: usize) -> QActRef<'a> {
        if hiprec {
            QActRef { m: 2, k, kb: 32, q: &ws.xq_q[..2 * k], scale: &ws.xq_s[..2 * nb], sum: &ws.xq_sum[..2 * nb], pair: true }
        } else {
            QActRef { m: 1, k, kb: 32, q: &ws.xq_q[..k], scale: &ws.xq_s[..nb], sum: &ws.xq_sum[..nb], pair: false }
        }
    }

    /// Part 2: gather lo, silu(lo/hc), up projection (sigmoid) on the tile's rows, then the tile's
    /// slice of `mixed` (mean over streams) into the MIXED mailbox and into cs.tmp[..hl].
    fn hc_mix_b(&self, ctx: &WorkerCtx, ws: &mut TileWs, cs: &mut CoreScratch, w: &HcW, t: usize) {
        let cfg = &self.cfg;
        let (hh, hc, tiles) = (cfg.hidden, cfg.hc, cfg.n_tiles);
        let lay = &self.mbox.layout;
        let lr = cfg.hc_lr;
        cs.lo[..lr].copy_from_slice(&self.mbox.slot_ro(0, lay.lo)[..lr]);
        for u in 1..tiles {
            let src = self.mbox.slot_ro(u, lay.lo);
            for j in 0..lr {
                cs.lo[j] += src[j];
            }
        }
        for j in 0..lr {
            cs.lo[j] = silu(cs.lo[j] / hc as f32);
        }
        CoreScratch::qz(cs.hiprec, &mut cs.q_lr, &cs.lo);
        let r = ctx.range_local(w.up.n_strips());
        unsafe {
            gemv_tq(w.up.ptr.add(r.start * w.up.strip_bytes()), w.up.k, w.up.codec, r.start, r.end, cs.q_lr.as_ref(), ws.gate_local.as_mut_ptr(), 0, false);
        }
        let rows = &mut ws.gate_local[r.start * 16..(r.end * 16).min(w.up.rows)];
        for g in rows.iter_mut() {
            *g = sigmoid(*g);
        }
        ctx.node_barrier();
        let hl = hh / tiles;
        let inv = 1.0 / hc as f32;
        for cidx in 0..hl {
            let mut acc = 0f32;
            for s in 0..hc {
                acc += ws.xn[s * hh + t * hl + cidx] * ws.gate_local[s * hl + cidx];
            }
            cs.tmp[cidx] = acc * inv;
        }
        if ctx.local == 0 {
            self.mbox.slot(t, lay.mixed)[..hl].copy_from_slice(&cs.tmp[..hl]);
        }
        let _ = LAYOUT_LO_PAD;
    }

    fn gdn_block(&self, ctx: &WorkerCtx, ws: &mut TileWs, st: &mut TileState, cs: &mut CoreScratch, g: &GdnW, il: usize, t: usize, pos: usize) {
        let cfg = &self.cfg;
        let tiles = cfg.n_tiles;
        let dk = cfg.d_state;
        let kh = cfg.n_k_heads / tiles;
        let vh = cfg.n_v_heads / tiles;
        let c_local = (2 * kh + vh) * dk;
        let v_off = 2 * kh * dk;
        CoreScratch::qz(cs.hiprec, &mut cs.q_h, ws.mixed_full);
        // projections: qkv strips then z strips as one task range; beta/alpha on core 0
        let ns_qkv = g.qkv.n_strips();
        let ns_z = g.z.n_strips();
        let r = ctx.range_local(ns_qkv + ns_z);
        for s in r {
            unsafe {
                if s < ns_qkv {
                    gemv_tq(g.qkv.ptr.add(s * g.qkv.strip_bytes()), g.qkv.k, g.qkv.codec, s, s + 1, cs.q_h.as_ref(), ws.qkv.as_mut_ptr(), 0, false);
                } else {
                    let s2 = s - ns_qkv;
                    gemv_tq(g.z.ptr.add(s2 * g.z.strip_bytes()), g.z.k, g.z.codec, s2, s2 + 1, cs.q_h.as_ref(), ws.z.as_mut_ptr(), 0, false);
                }
            }
        }
        if ctx.local == 0 {
            for h in 0..vh {
                ws.beta[h] = dot(&g.beta[h * cfg.hidden..(h + 1) * cfg.hidden], ws.mixed_full);
                ws.alpha[h] = dot(&g.alpha[h * cfg.hidden..(h + 1) * cfg.hidden], ws.mixed_full);
            }
        }
        plap("gdn.proj");
        ctx.node_barrier();
        plap("gdn.proj_bar");
        dmp_local("qkv", &ws.qkv[..8]);
        dmp_local("z", &ws.z[..8]);
        dmp_local("beta_raw", &ws.beta[..6]);
        dmp_local("alpha_raw", &ws.alpha[..6]);
        // conv (per channel, kernel d_conv) + silu, with state update; channels split across cores
        let gs = st.gdn[il].as_mut().unwrap();
        let kc = cfg.d_conv;
        for grp in ctx.range_local(c_local / 16) {
            tr_kernels::conv::conv_silu_seq16(ws.qkv, c_local, 1, g.conv, kc, gs.conv, c_local, grp * 16, ws.qkvc, c_local);
        }
        plap("gdn.conv");
        ctx.node_barrier();
        plap("gdn.conv_bar");
        dmp_local("conv_silu", &ws.qkvc[..8]);
        // delta rule: tasks = (v head, 16-col chunk)
        let chunks = dk / 16;
        let ntask = vh * chunks;
        let r = ctx.range_local(ntask);
        let scale = (dk as f32).powf(-0.5);
        for task in r {
            let h = task / chunks;
            let ch = task % chunks;
            // llama.cpp broadcasts key heads periodically (ggml_repeat): value head H uses key head
            // H % n_k. The tile owns k-heads kh*t..+kh and value heads kh*t + i + n_k*j (local j*kh+i).
            let kh_local = h % kh;
            let (q, k, v) = {
                let q = &ws.qkvc[kh_local * dk..(kh_local + 1) * dk];
                let k = &ws.qkvc[kh * dk + kh_local * dk..kh * dk + (kh_local + 1) * dk];
                let v = &ws.qkvc[v_off + h * dk..v_off + (h + 1) * dk];
                (q, k, v)
            };
            let (qn, kn) = cs.head_tmp.split_at_mut(dk);
            elem::l2norm(q, cfg.rms_eps, &mut qn[..dk]);
            elem::l2norm(k, cfg.rms_eps, &mut kn[..dk]);
            let hg = kh * t + (h % kh) + cfg.n_k_heads * (h / kh);
            let gate = g.a[hg] * softplus(ws.alpha[h] + g.dt[hg]);
            let beta = sigmoid(ws.beta[h]);
            // chunk-major state: [head][chunk][dk][16], so a task's 8 KiB is contiguous (L1)
            let state = &mut gs.ssm[(h * chunks + ch) * dk * 16..(h * chunks + ch + 1) * dk * 16];
            tr_kernels::gdn::gdn_step16(state, dk, &qn[..dk], &kn[..dk], &v[ch * 16..ch * 16 + 16], gate, beta, scale, &mut ws.y[h * dk + ch * 16..h * dk + ch * 16 + 16]);
        }
        plap("gdn.delta");
        ctx.node_barrier();
        plap("gdn.delta_bar");
        dmp_local("gdn_y", &ws.y[..8]);
        {
            let gg: Vec<f32> = (0..vh).map(|h| { let hg = kh * t + (h % kh) + cfg.n_k_heads * (h / kh); g.a[hg] * softplus(ws.alpha[h] + g.dt[hg]) }).collect();
            dmp_local("gate_g", &gg);
        }
        // gated rmsnorm per head (redundant per core), quantise, out-proj partial
        for h in 0..vh {
            let y = &ws.y[h * dk..(h + 1) * dk];
            let (nrm, _) = cs.head_tmp.split_at_mut(dk);
            rmsnorm(y, Some(g.norm), cfg.rms_eps, nrm);
            for i in 0..dk {
                cs.vec_v[h * dk + i] = nrm[i] * sigmoid(ws.z[h * dk + i]);
            }
        }
        let vl = vh * dk;
        dmp_local("final_output", &cs.vec_v[..8]);
        dmp_tile("final_first", &cs.vec_v[..4]);
        if ctx.local == 0 && _il_is_zero(il) {
            if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                let bytes: Vec<u8> = cs.vec_v[..vl].iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{path}.t{t}.p{pos}.bin"), bytes);
                if t == 0 {
                    let _ = std::fs::write(format!("{path}.qkvc.p{pos}.bin"), ws.qkvc.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                    let _ = std::fs::write(format!("{path}.y.p{pos}.bin"), ws.y.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
        }
        dmp_tile("final_last", &cs.vec_v[vl - 4..vl]);
        CoreScratch::qz(cs.hiprec, &mut cs.q_v, &cs.vec_v[..vl]);
        let r = ctx.range_local(g.out.n_strips());
        plap("gdn.norm");
        let out = self.mbox.slot(t, self.mbox.layout.part);
        unsafe {
            gemv_tq(g.out.ptr.add(r.start * g.out.strip_bytes()), g.out.k, g.out.codec, r.start, r.end, cs.q_v.as_ref(), out.as_mut_ptr(), 0, false);
        }
        plap("gdn.out");
        if ctx.local == 0 {
            // self-check of the kernel on real data: first 8 output rows via f32 reference
            let mut refv = [0f32; 8];
            let mut row = vec![0f32; g.out.k];
            for r in 0..8 {
                dequant_tq_row(&g.out, r, &mut row);
                refv[r] = dot(&row, &cs.vec_v[..vl]);
            }
            dmp_tile("part_kernel", &out[..8]);
            dmp_tile("part_ref", &refv);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_block(&self, ctx: &WorkerCtx, ws: &mut TileWs, st: &mut TileState, cs: &mut CoreScratch, a: &AttnW, il: usize, pos: usize, t: usize) {
        let cfg = &self.cfg;
        let hd = cfg.head_dim;
        let qh = cfg.n_head / cfg.n_tiles;
        let tiles = cfg.n_tiles;
        let lay = &self.mbox.layout;
        CoreScratch::qz(cs.hiprec, &mut cs.q_h, ws.mixed_full);
        let (nq, nk, nv) = (a.q.n_strips(), a.k.n_strips(), a.v.n_strips());
        let (niq, nik) = a.idx.map(|i| (i.q.n_strips(), i.k.n_strips())).unwrap_or((0, 0));
        let idx_slot = self.mbox.slot(t, lay.idx);
        let r = ctx.range_local(nq + nk + nv + niq + nik);
        for s in r {
            unsafe {
                if s < nq {
                    gemv_tq(a.q.ptr.add(s * a.q.strip_bytes()), a.q.k, a.q.codec, s, s + 1, cs.q_h.as_ref(), ws.qg.as_mut_ptr(), 0, false);
                } else if s < nq + nk {
                    let s2 = s - nq;
                    gemv_tq(a.k.ptr.add(s2 * a.k.strip_bytes()), a.k.k, a.k.codec, s2, s2 + 1, cs.q_h.as_ref(), ws.kn.as_mut_ptr(), 0, false);
                } else if s < nq + nk + nv {
                    let s2 = s - nq - nk;
                    gemv_tq(a.v.ptr.add(s2 * a.v.strip_bytes()), a.v.k, a.v.codec, s2, s2 + 1, cs.q_h.as_ref(), ws.vn.as_mut_ptr(), 0, false);
                } else if s < nq + nk + nv + niq {
                    let i = a.idx.as_ref().unwrap();
                    let s2 = s - nq - nk - nv;
                    gemv_tq(i.q.ptr.add(s2 * i.q.strip_bytes()), i.q.k, i.q.codec, s2, s2 + 1, cs.q_h.as_ref(), idx_slot.as_mut_ptr(), 0, false);
                } else {
                    let i = a.idx.as_ref().unwrap();
                    let s2 = s - nq - nk - nv - niq;
                    gemv_tq(i.k.ptr.add(s2 * i.k.strip_bytes()), i.k.k, i.k.codec, s2, s2 + 1, cs.q_h.as_ref(), idx_slot.as_mut_ptr().add(niq * 16), 0, false);
                }
            }
        }
        ctx.node_barrier();
        dmp_local("Qcur_full", &ws.qg[..8]);
        dmp_local("Kcur", &ws.kn[..8]);
        dmp_local("Vcur", &ws.vn[..8]);
        let ast = st.attn[il].as_mut().unwrap();
        assert!(pos < st.ctx_max, "context overflow: pos {pos} >= ctx_max {}", st.ctx_max);
        let rp = self.rpos(pos);
        if ctx.local == 0 {
            let kdst = &mut cs.head_tmp[..hd];
            rmsnorm(ws.kn, Some(a.k_norm), cfg.rms_eps, kdst);
            self.rope.apply3(kdst, rp);
            kv_dispatch!(ast.kv, |T| {
                let (k, v) = ast.kv_as::<T>();
                store_row(&mut k[pos * hd..(pos + 1) * hd], kdst);
                store_row(&mut v[pos * hd..(pos + 1) * hd], ws.vn);
            });
        }
        // ---- QSA indexer: gather the projections, update the block keys, score the blocks
        let ratio = cfg.qsa_ratio(il);
        let n_kv = pos + 1;
        let mut sparse = false;
        if let (Some(iw), true) = (&a.idx, ratio > 0) {
            let (nh, d) = (cfg.idx_heads, cfg.idx_dim);
            let (per_q, per_k) = (nh * d / tiles, d / tiles);
            let per = per_q + per_k;
            ctx.barrier(); // I: every tile's indexer slice is published
            for i in ctx.range_local(tiles * per) {
                let (u, j) = (i / per, i % per);
                let v = self.mbox.slot_ro(u, lay.idx)[j];
                if j < per_q {
                    ws.idx_qraw[u * per_q + j] = v;
                } else {
                    ws.idx_kraw[u * per_k + j - per_q] = v;
                }
            }
            ctx.node_barrier();
            if ctx.local == 0 {
                for h in 0..nh {
                    let q = &mut ws.idx_q[h * d..(h + 1) * d];
                    rmsnorm(&ws.idx_qraw[h * d..(h + 1) * d], Some(iw.q_norm), cfg.rms_eps, q);
                    self.rope.apply3(q, rp);
                }
                ast.ik_raw[(pos % ratio) * d..(pos % ratio + 1) * d].copy_from_slice(ws.idx_kraw);
                dmp_local("indexer_q", &ws.idx_q[..8]);
                dmp_local("indexer_k_raw", &ws.idx_kraw[..8]);
                if (pos + 1) % ratio == 0 {
                    let b = pos / ratio;
                    let tmp = &mut cs.head_tmp[..d];
                    tmp.fill(0.0);
                    for i in 0..ratio {
                        elem::axpy(tmp, &ast.ik_raw[i * d..(i + 1) * d], 1.0 / ratio as f32);
                    }
                    let pk = &mut ast.ik_pooled[b * d..(b + 1) * d];
                    rmsnorm(tmp, Some(iw.k_norm), cfg.rms_eps, pk);
                    self.rope.apply(pk, (b * ratio) as u32);
                    dmp_local("indexer_k", &pk[..8]);
                }
            }
            ctx.node_barrier();
            sparse = !self.qsa_off && qsa::is_sparse(pos, ratio, cfg.idx_top_k);
            let nb = qsa::n_complete_blocks(pos, ratio);
            if sparse || (self.qsa_debug && nb > 0) {
                let r = ctx.range_local(nb);
                qsa::score_blocks(ws.idx_q, ast.ik_pooled, d, r.start, r.end, ws.idx_scores);
                ctx.node_barrier();
                dmp_local("indexer_score", &ws.idx_scores[..nb.min(8)]);
            }
        } else {
            ctx.node_barrier();
        }
        let scale = (hd as f32).powf(-0.5);
        let nc = ctx.cores_per_node;
        // q per head (normed + roped) into the tile-shared buffer; core 0 builds the selection
        if ctx.local < qh {
            let h = ctx.local;
            rmsnorm(&ws.qg[h * 2 * hd..h * 2 * hd + hd], Some(a.q_norm), cfg.rms_eps, &mut ws.qn[h * hd..(h + 1) * hd]);
            self.rope.apply3(&mut ws.qn[h * hd..(h + 1) * hd], rp);
            if h == 0 {
                dmp_local("Qcur_roped", &ws.qn[..8]);
                let kc: Vec<f32> = kv_dispatch!(ast.kv, |T| ast.kv_as::<T>().0[pos * hd..pos * hd + 8].iter().map(|x| x.to_f32()).collect());
                dmp_local("Kcache_roped", &kc);
            }
        }
        if ctx.local == nc - 1 {
            if sparse {
                qsa::select_ranges(ws.idx_scores, pos, ratio, cfg.idx_top_k, &mut cs.sel_idx, &mut cs.sel_ranges);
            } else {
                cs.sel_ranges.clear();
                cs.sel_ranges.push((0, n_kv as u32));
            }
            ws.sel[0] = cs.sel_ranges.len() as u32;
            for (i, r) in cs.sel_ranges.iter().enumerate() {
                ws.sel[1 + 2 * i] = r.0;
                ws.sel[2 + 2 * i] = r.1;
            }
        }
        ctx.node_barrier();
        // (head, chunk) tasks: each core takes a window of the selected tokens for one head
        let ns = (nc / qh).max(1);
        let n_sel = ws.sel[0] as usize;
        if ctx.local != nc - 1 {
            cs.sel_ranges.clear();
            cs.sel_ranges.extend((0..n_sel).map(|i| (ws.sel[1 + 2 * i], ws.sel[2 + 2 * i])));
        }
        let ranges: &[(u32, u32)] = &cs.sel_ranges;
        let total: usize = ranges.iter().map(|r| r.1 as usize).sum();
        let pw = hd + 16;
        for task in ctx.range_local(qh * ns) {
            let (h, c) = (task / ns, task % ns);
            let (t0, t1) = (c * total / ns, (c + 1) * total / ns);
            let part = &mut ws.attn_part[task * pw..(task + 1) * pw];
            let (ml, o) = part.split_at_mut(16);
            let (mx, sum) = kv_dispatch!(ast.kv, |T| {
                let (k, v) = ast.kv_as::<T>();
                attend_partial(&ws.qn[h * hd..(h + 1) * hd], k, v, ranges, t0, t1, hd, scale, o, &mut cs.attn_scratch, &mut cs.sel_idx)
            });
            ml[0] = mx;
            ml[1] = sum;
        }
        ctx.node_barrier();
        if ctx.local < qh {
            let h = ctx.local;
            let gate = &ws.qg[h * 2 * hd + hd..(h + 1) * 2 * hd];
            let parts: Vec<(f32, f32)> = (0..ns).map(|c| (ws.attn_part[(h * ns + c) * pw], ws.attn_part[(h * ns + c) * pw + 1])).collect();
            let outs: Vec<&[f32]> = (0..ns).map(|c| &ws.attn_part[(h * ns + c) * pw + 16..(h * ns + c + 1) * pw]).collect();
            let o = &mut cs.head_tmp[..hd];
            merge_partials(&parts, &outs, hd, o);
            if h == 0 {
                dmp_local("attn_pregate", &o[..8]);
                let gs: Vec<f32> = gate[..8].iter().map(|&g| sigmoid(g)).collect();
                dmp_local("gate_sigmoid", &gs);
            }
            for i in 0..hd {
                ws.o[h * hd + i] = o[i] * sigmoid(gate[i]);
            }
        }
        ctx.node_barrier();
        let ol = qh * hd;
        CoreScratch::qz(cs.hiprec, &mut cs.q_v, &ws.o[..ol]);
        let r = ctx.range_local(a.o.n_strips());
        let out = self.mbox.slot(t, self.mbox.layout.part);
        unsafe {
            gemv_tq(a.o.ptr.add(r.start * a.o.strip_bytes()), a.o.k, a.o.codec, r.start, r.end, cs.q_v.as_ref(), out.as_mut_ptr(), 0, false);
        }
    }

    fn moe_block(&self, ctx: &WorkerCtx, ws: &mut TileWs, cs: &mut CoreScratch, m: &MoeW, _il: usize, t: usize, nc: usize) {
        let cfg = &self.cfg;
        let ffl = cfg.n_ff / cfg.n_tiles;
        let k_used = self.route(ws.rlog, &mut cs.ids, &mut cs.w);
        let slots = k_used + 1;
        if cfg.n_expert_packed < cfg.n_expert {
            // subset pack (tests only): fold the routed ids onto the packed experts (distinct per token,
            // as real routing is; the batch path groups entries by expert and sizes for m per group)
            crate::exec_batch::fold_ids(&mut cs.ids[..k_used], cfg.n_expert_packed as u32);
        }
        if ctx.global == 0 {
            self.moe_stats.add(1, k_used as u64);
            self.moe_stats.profile_row(ws.rlog);
        }
        let sh_gate = sigmoid(dot(m.shexp_gate, ws.mixed_full));
        {
            let idsf: Vec<f32> = cs.ids[..k_used].iter().map(|&i| i as f32).collect();
            dmp_local("moe_topk", &idsf);
            dmp_local("moe_w", &cs.w[..k_used]);
            dmp_local("rlog", &ws.rlog[..8]);
            dmp_local("sh_gate", &[sh_gate]);
        }
        CoreScratch::qz(cs.hiprec, &mut cs.q_h, ws.mixed_full);
        // tasks A: (slot, gate|up, strip)
        let ns = m.gate.n_strips();
        let ntask = slots * 2 * ns;
        let mut task = ctx.local;
        while task < ntask {
            let slot = task / (2 * ns);
            let which = (task / ns) % 2;
            let s = task % ns;
            let (mat, dst): (&TqMat, &mut [f32]) = if which == 0 { (if slot < k_used { &m.gate } else { &m.sh_gate }, &mut ws.hg[slot * ffl..(slot + 1) * ffl]) } else { (if slot < k_used { &m.up } else { &m.sh_up }, &mut ws.hu[slot * ffl..(slot + 1) * ffl]) };
            let item = if slot < k_used { cs.ids[slot] as usize } else { 0 };
            unsafe {
                gemv_tq(mat.item(item).add(s * mat.strip_bytes()), mat.k, mat.codec, s, s + 1, cs.q_h.as_ref(), dst.as_mut_ptr(), 0, false);
            }
            task += nc;
        }
        plap("moe.gateup");
        ctx.node_barrier();
        plap("moe.gateup_bar");
        dmp_local("moe_gate_e0", &ws.hg[..8]);
        dmp_local("moe_up_e0", &ws.hu[..8]);
        dmp_local("shexp_gate_h", &ws.hg[k_used * ffl..k_used * ffl + 8]);
        // tasks B: activation + quantisation per slot (kb = down.kb), fold the routing weight in
        let kb = m.down.codec.kb;
        let nb = ffl / kb;
        let mut slot = ctx.local;
        while slot < slots {
            let g = &ws.hg[slot * ffl..(slot + 1) * ffl];
            let u = &ws.hu[slot * ffl..(slot + 1) * ffl];
            let act = &mut cs.tmp[..ffl];
            elem::swiglu_vec(g, u, act);
            let wgt = if slot < k_used { cs.w[slot] } else { sh_gate };
            let q = &mut ws.act_q[2 * slot * ffl..2 * (slot + 1) * ffl];
            let sc = &mut ws.act_s[2 * slot * nb..2 * (slot + 1) * nb];
            let su = &mut ws.act_sum[2 * slot * nb..2 * (slot + 1) * nb];
            if cs.hiprec {
                QActRef::quantize_pair_into(ffl, kb, act, q, sc, su);
            } else {
                QActRef::quantize_into(1, ffl, kb, act, q, sc, su);
            }
            for s in sc.iter_mut() {
                *s *= wgt;
            }
            slot += nc;
        }
        plap("moe.act");
        ctx.node_barrier();
        plap("moe.act_bar");
        // tasks C: down strips (this core's range), accumulating all slots
        let r = ctx.range_local(m.down.n_strips());
        let out = self.mbox.slot(t, self.mbox.layout.part);
        for slot in 0..slots {
            let mat = if slot < k_used { &m.down } else { &m.sh_down };
            let item = if slot < k_used { cs.ids[slot] as usize } else { 0 };
            let (mrows, pair) = if cs.hiprec { (2, true) } else { (1, false) };
            let xq = QActRef { m: mrows, k: ffl, kb, q: &ws.act_q[2 * slot * ffl..2 * slot * ffl + mrows * ffl], scale: &ws.act_s[2 * slot * nb..2 * slot * nb + mrows * nb], sum: &ws.act_sum[2 * slot * nb..2 * slot * nb + mrows * nb], pair };
            unsafe {
                gemv_tq(mat.item(item).add(r.start * mat.strip_bytes()), mat.k, mat.codec, r.start, r.end, xq, out.as_mut_ptr(), 0, slot > 0);
            }
        }
        plap("moe.down");
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_block(&self, ctx: &WorkerCtx, ws: &mut TileWs, st: &mut TileState, cs: &mut CoreScratch, p: &PleW, tok: u32, prev: &[u32], _il: usize) {
        let cfg = &self.cfg;
        let (hh, hc, tiles, t) = (cfg.hidden, cfg.hc, cfg.n_tiles, ctx.node);
        let hl = hh / tiles;
        let nh = cfg.ple_n_heads();
        let wr = |name: &str, v: &[f32]| {
            if ctx.global == 0 {
                if let Ok(path) = std::env::var("TR_DUMP_VEC") {
                    let _ = std::fs::write(format!("{path}.{name}.bin"), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
        };
        wr(&format!("ple_res_in.p{}", self.n_past), &cs.res);
        let hash = tr_kernels::ple::PleHash { ngram: cfg.ple_ngram, per_gram: cfg.ple_per_gram, eos: cfg.ple_eos, mult: cfg.ple_mult.clone(), offsets: cfg.ple_offsets.clone(), vocab: cfg.ple_vocab.clone() };
        let mut rows = vec![0u32; nh];
        hash.rows(tok, prev, &mut rows);
        // gather + dequant the rows once per tile (core h % cores handles head h), then share
        for (h, &row) in rows.iter().enumerate() {
            if h % ctx.cores_per_node != ctx.local {
                continue;
            }
            let owner = row as usize / self.ple_rows_per_tile;
            let raw = self.tiles[owner].raw("per_layer_token_embd.weight").expect("PLE table missing");
            let local = row as usize - raw.start_row;
            let src = unsafe { std::slice::from_raw_parts(raw.ptr.add(local * raw.row_bytes), raw.row_bytes) };
            tr_kernels::iq4nl::dequant_row(src, cfg.ple_dim, &mut ws.ple_emb[h * cfg.ple_dim..(h + 1) * cfg.ple_dim]);
        }
        ctx.node_barrier();
        cs.emb.copy_from_slice(ws.ple_emb);
        dmp_local("ple_embd", &cs.emb[..8]);
        dmp_local("ple_rows", &rows.iter().map(|&r| r as f32).collect::<Vec<_>>());
        wr(&format!("ple_rows.p{}", self.n_past), &rows.iter().map(|&r| r as f32).collect::<Vec<_>>());
        CoreScratch::qz(cs.hiprec, &mut cs.q_h, &cs.emb);
        let (nk, nv) = (p.key.n_strips(), p.value.n_strips());
        let r = ctx.range_local(nk + nv);
        for s in r {
            unsafe {
                if s < nk {
                    gemv_tq(p.key.ptr.add(s * p.key.strip_bytes()), p.key.k, p.key.codec, s, s + 1, cs.q_h.as_ref(), ws.ple_key.as_mut_ptr(), 0, false);
                } else {
                    let s2 = s - nk;
                    gemv_tq(p.value.ptr.add(s2 * p.value.strip_bytes()), p.value.k, p.value.codec, s2, s2 + 1, cs.q_h.as_ref(), ws.ple_val.as_mut_ptr(), 0, false);
                }
            }
        }
        ctx.node_barrier();
        // query = grouped_norm(res, norm_query) (private); partial stats per stream from the tile's key slice
        for s in 0..hc {
            rmsnorm(&cs.res[s * hh..(s + 1) * hh], Some(&p.norm_query[s * hh..(s + 1) * hh]), cfg.rms_eps, &mut cs.xn[s * hh..(s + 1) * hh]);
        }
        if ctx.local == 0 {
            let slot = self.mbox.slot(t, self.mbox.layout.ple);
            for s in 0..hc {
                let key = &ws.ple_key[s * hl..(s + 1) * hl];
                let mut ss = 0f32;
                let mut dp = 0f32;
                for cidx in 0..hl {
                    let kv = key[cidx];
                    ss += kv * kv;
                    dp += kv * p.norm_key[s * hh + t * hl + cidx] * cs.xn[s * hh + t * hl + cidx];
                }
                slot[2 * s] = ss;
                slot[2 * s + 1] = dp;
            }
            slot[2 * hc..2 * hc + hl].copy_from_slice(&ws.ple_val[..hl]);
        }
        ctx.barrier(); // P
        // every core: reduce stats, gather value (private), gate, conv, residual update
        let mut gate = [0f32; 8];
        for s in 0..hc {
            let mut ss = 0f32;
            let mut dp = 0f32;
            for u in 0..tiles {
                let sl = self.mbox.slot_ro(u, self.mbox.layout.ple);
                ss += sl[2 * s];
                dp += sl[2 * s + 1];
            }
            let rms = (ss / hh as f32 + cfg.rms_eps).sqrt();
            let sdot = dp / rms / (hh as f32).sqrt();
            let mag = sdot.abs().max(1e-6).sqrt();
            gate[s] = sigmoid(sdot.signum() * mag);
        }
        for u in 0..tiles {
            let sl = self.mbox.slot_ro(u, self.mbox.layout.ple);
            cs.emb[u * hl..(u + 1) * hl].copy_from_slice(&sl[2 * hc..2 * hc + hl]);
        }
        dmp_local("ple_gate", &gate[..hc]);
        dmp_local("ple_value", &cs.emb[..8]);
        dmp_local("ple_key_local", &ws.ple_key[..8]);
        // gated value per stream -> cs.tmp; normalized -> cs.xn
        for s in 0..hc {
            for i in 0..hh {
                cs.tmp[s * hh + i] = cs.emb[i] * gate[s];
            }
            rmsnorm(&cs.tmp[s * hh..(s + 1) * hh], Some(&p.norm_conv[s * hh..(s + 1) * hh]), cfg.rms_eps, &mut cs.xn[s * hh..(s + 1) * hh]);
        }
        // dilated causal conv over history: out[c] = sum_k w[c][k] * x[t - (K-1-k)*dil]
        let kern = cfg.ple_conv_kernel;
        let dil = cfg.ple_ngram;
        let hist = (kern - 1) * dil;
        let hcd = hc * hh;
        dmp_local("ple_gated_value", &cs.tmp[..8]);
        dmp_local("ple_normalized", &cs.xn[..8]);
        let mut conv_dbg = [0f32; 8];
        for cidx in 0..hcd {
            let mut acc = p.conv[cidx * kern + (kern - 1)] * cs.xn[cidx];
            for k in 0..kern - 1 {
                let back = (kern - 1 - k) * dil; // positions back
                acc += p.conv[cidx * kern + k] * st.ple_hist[(hist - back) * hcd + cidx];
            }
            if cidx < 8 {
                conv_dbg[cidx] = silu(acc);
            }
            cs.res[cidx] += cs.tmp[cidx] + silu(acc);
        }
        dmp_local("ple_conv_out", &conv_dbg);
        dmp_local("ple_res_out", &cs.res[..8]);
        wr(&format!("ple_res_out.p{}", self.n_past), &cs.res);
        ctx.node_barrier();
        if ctx.local == 0 {
            // shift history by one, append normalized
            st.ple_hist.copy_within(hcd.., 0);
            st.ple_hist[(hist - 1) * hcd..].copy_from_slice(&cs.xn[..hcd]);
        }
        ctx.node_barrier();
    }
}

fn hc_combine(res: &mut [f32], out: &[f32], inject: &[f32; 8], hc: usize, hh: usize) {
    for s in 0..hc {
        let w = 2.0 * sigmoid(inject[s] / hc as f32);
        elem::axpy(&mut res[s * hh..(s + 1) * hh], out, w);
    }
}

impl Model {
    pub fn load_summary(&self) -> String {
        let topo = tr_sys::topology::Topology::discover().ok();
        let free: Vec<String> = topo.map(|t| t.nodes.iter().map(|n| format!("{:.1}", n.mem_free_kb as f64 / (1u64 << 20) as f64)).collect()).unwrap_or_default();
        format!("loaded {:.2} GiB in {:.1} s ({:.2} GB/s); {}; per-tile arena {:.2} GiB; node MemFree GiB {}", self.load.bytes as f64 / (1u64 << 30) as f64, self.load.seconds, self.load.bytes as f64 / self.load.seconds / 1e9, tr_sys::procinfo::mem_summary(), self.load.per_tile_bytes[0] as f64 / (1u64 << 30) as f64, free.join(" "))
    }
}

impl Drop for Model {
    fn drop(&mut self) {}
}

pub fn ctx_check(m: &Manifest) -> Result<()> {
    m.validate().context("pack validation")
}
