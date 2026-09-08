//! Per-tile weight tables loaded from a pack into tile-bound arenas.
use crate::config::ModelConfig;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tr_format::codec::Codec;
use tr_format::{Manifest, TensorEntry, TensorKind};
use tr_sys::loader::{parallel_load, DirectFile, Section};
use tr_sys::numa::Arena;
use tr_sys::pool::Pool;

/// A packed TQ matrix (or `count` matrices of `stride` bytes) living in tile memory.
#[derive(Clone, Copy)]
pub struct TqMat {
    pub ptr: *const u8,
    pub rows: usize,
    pub k: usize,
    pub codec: Codec,
    pub count: usize,
    pub stride: usize,
}
unsafe impl Send for TqMat {}
unsafe impl Sync for TqMat {}
impl TqMat {
    pub fn item(&self, i: usize) -> *const u8 {
        debug_assert!(i < self.count);
        unsafe { self.ptr.add(i * self.stride) }
    }
    pub fn n_strips(&self) -> usize {
        tr_format::codec::pad_rows(self.rows) / 16
    }
    pub fn strip_bytes(&self) -> usize {
        self.codec.strip_bytes(self.k)
    }
}

/// bf16 weights in AMX strip layout (see `TensorKind::Bf16Strips`); consumed by `amx::gemm_bf16`
/// without unpacking. `ptr` is 64-byte aligned.
#[derive(Clone, Copy)]
pub struct Bf16Mat {
    pub ptr: *const u16,
    pub rows: usize,
    pub k: usize,
}
unsafe impl Send for Bf16Mat {}
unsafe impl Sync for Bf16Mat {}
impl Bf16Mat {
    pub fn n_strips(&self) -> usize {
        self.rows.div_ceil(16)
    }
    /// u16 elements per strip.
    pub fn strip_len(&self) -> usize {
        16 * self.k
    }
}

#[derive(Clone, Copy)]
pub struct F32Mat {
    pub ptr: *const f32,
    pub len: usize,
    pub shape: [usize; 2],
}
unsafe impl Send for F32Mat {}
unsafe impl Sync for F32Mat {}
impl F32Mat {
    pub fn as_slice(&self) -> &'static [f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[derive(Clone, Copy)]
pub struct RawRows {
    pub ptr: *const u8,
    pub rows: usize,
    pub row_bytes: usize,
    pub start_row: usize,
}
unsafe impl Send for RawRows {}
unsafe impl Sync for RawRows {}

/// Everything one tile holds, addressed by tensor name.
pub struct TileWeights {
    pub tile: usize,
    pub tq: HashMap<String, TqMat>,
    pub f32: HashMap<String, F32Mat>,
    pub raw: HashMap<String, RawRows>,
    pub bf16: HashMap<String, Bf16Mat>,
    pub arena: Arena,
    /// File mappings kept alive for file-backed (page-cache) tensors such as the PLE table.
    pub mmaps: Vec<memmap2::Mmap>,
}

impl TileWeights {
    pub fn tq(&self, name: &str) -> &TqMat {
        self.tq.get(name).unwrap_or_else(|| panic!("tile {}: missing tq tensor {name}", self.tile))
    }
    pub fn f32(&self, name: &str) -> &'static [f32] {
        self.f32.get(name).unwrap_or_else(|| panic!("tile {}: missing f32 tensor {name}", self.tile)).as_slice()
    }
    pub fn bf16(&self, name: &str) -> &Bf16Mat {
        self.bf16.get(name).unwrap_or_else(|| panic!("tile {}: missing bf16 tensor {name}", self.tile))
    }
    pub fn raw(&self, name: &str) -> Option<&RawRows> {
        self.raw.get(name)
    }
}

pub struct LoadReport {
    pub bytes: u64,
    pub seconds: f64,
    pub per_tile_bytes: Vec<u64>,
}

/// Plan arenas, allocate every tensor of every tile (shared tensors replicated per tile), then
/// fill them in parallel from the tile's own workers.
pub struct LoadOptions {
    /// Keep the PLE n-gram table file-backed (mmap, page cache) instead of resident in HBM.
    /// Saves ~3.4 GiB per tile; costs a few page faults per token until the hot rows are cached.
    pub ple_mmap: bool,
    /// Maximum prompt batch (tokens per batched step). 1 disables the batch path.
    pub batch_max: usize,
    /// K/V cache element type.
    pub kv: crate::state::KvType,
    /// MTP overlay pack directory (`trpack mtp`); enables speculative decoding.
    pub mtp: Option<std::path::PathBuf>,
    /// Draft tokens per round (state checkpoints for `spec_k + 1` rows are allocated when > 0).
    pub spec_k: usize,
    /// Vision encoder overlay pack directory (`trpack vision`); enables image input.
    pub vision: Option<std::path::PathBuf>,
    /// Image size limits in merged tokens (the workspace is sized for `image_max_tokens`).
    pub image_min_tokens: usize,
    pub image_max_tokens: usize,
    /// Expert routing policy (None = the model's fixed top-k); see `Model::set_moe`.
    pub moe: Option<tr_kernels::router::MoePolicy>,
}
impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { ple_mmap: false, batch_max: 256, kv: crate::state::KvType::F16, mtp: None, spec_k: 0, vision: None, image_min_tokens: 8, image_max_tokens: 4096, moe: None }
    }
}

/// `overlay`: an optional second manifest (the MTP draft head) whose tensors land in the same
/// tile arenas and name maps; it must have been packed for `m` (`overlay_of == m.hash`).
/// `extra`: further overlays that are independent of the base pack (the vision encoder).
pub fn load_weights(m: &Manifest, cfg: &ModelConfig, pool: &mut Pool, extra_per_tile: usize, opts: &LoadOptions, overlay: Option<&Manifest>, extra: &[&Manifest]) -> Result<(Vec<TileWeights>, LoadReport)> {
    let n_tiles = cfg.n_tiles;
    if pool.n_nodes() != n_tiles {
        bail!("pack has {} tiles but the pool has {} nodes", n_tiles, pool.n_nodes());
    }
    if let Some(o) = overlay {
        if o.overlay_of.as_deref() != Some(m.hash.as_str()) {
            bail!("overlay pack {} was built for base hash {:?}, this pack is {}", o.dir.display(), o.overlay_of, m.hash);
        }
        if o.n_tiles != n_tiles {
            bail!("overlay pack has {} tiles, base {}", o.n_tiles, n_tiles);
        }
    }
    for o in extra {
        if o.n_tiles != n_tiles {
            bail!("overlay pack {} has {} tiles, base {}", o.dir.display(), o.n_tiles, n_tiles);
        }
    }
    let mans: Vec<&Manifest> = std::iter::once(m).chain(overlay).chain(extra.iter().copied()).collect();
    let shared_bytes: u64 = mans.iter().flat_map(|mm| mm.shared_tensors()).map(|t| t.nbytes as u64 + 64).sum();
    // file handles keyed by (manifest index, file name)
    let mut files: HashMap<(usize, String), Arc<DirectFile>> = HashMap::new();
    for (mi, mm) in mans.iter().enumerate() {
        for f in &mm.files {
            files.insert((mi, f.name.clone()), Arc::new(DirectFile::open(&mm.file_path(&f.name))?));
        }
    }
    let mut tiles = Vec::with_capacity(n_tiles);
    let mut sections = Vec::new();
    let mut per_tile_bytes = Vec::new();
    let is_mmapped = |e: &TensorEntry| opts.ple_mmap && e.kind == TensorKind::Iq4nlRows;
    for t in 0..n_tiles {
        let resident: u64 = mans.iter().flat_map(|mm| mm.tensors_on_tile(t)).filter(|e| !is_mmapped(e)).map(|e| e.nbytes as u64).sum();
        let n_ent = mans.iter().map(|mm| mm.tensors_on_tile(t).count() as u64).sum::<u64>();
        let need = resident + shared_bytes + n_ent * 4096 + extra_per_tile as u64;
        per_tile_bytes.push(need);
        let mut arena = Arena::new(t, need as usize + (64 << 20)).with_context(|| format!("arena for tile {t}"))?;
        let mut tw = TileWeights { tile: t, tq: HashMap::new(), f32: HashMap::new(), raw: HashMap::new(), bf16: HashMap::new(), arena: Arena::new(t, 1)?, mmaps: Vec::new() };
        for (mi, mm) in mans.iter().enumerate() {
            for e in mm.tensors_on_tile(t).chain(mm.shared_tensors()) {
                if is_mmapped(e) {
                    let f = std::fs::File::open(mm.file_path(&e.file))?;
                    let map = unsafe { memmap2::MmapOptions::new().offset(e.offset).len(e.nbytes).map(&f)? };
                    let ptr = map.as_ptr() as *mut u8;
                    register(&mut tw, e, ptr);
                    tw.mmaps.push(map);
                    continue;
                }
                let dst = arena.alloc(e.nbytes, 4096)?;
                register(&mut tw, e, dst);
                sections.push(Section { node: t, file: files[&(mi, e.file.clone())].clone(), offset: e.offset, dst, len: e.nbytes });
            }
        }
        tw.arena = arena;
        tiles.push(tw);
    }
    let t0 = std::time::Instant::now();
    let bytes = parallel_load(pool, &sections)?;
    let seconds = t0.elapsed().as_secs_f64();
    Ok((tiles, LoadReport { bytes, seconds, per_tile_bytes }))
}

fn register(tw: &mut TileWeights, e: &TensorEntry, dst: *mut u8) {
    match e.kind {
        TensorKind::Tq => {
            tw.tq.insert(e.name.clone(), TqMat { ptr: dst, rows: e.rows, k: e.k, codec: e.codec(), count: e.count, stride: e.stride });
        }
        TensorKind::F32 => {
            let shape = if e.shape.len() >= 2 { [e.shape[0], e.shape[1..].iter().product()] } else { [1, e.shape.first().copied().unwrap_or(0)] };
            tw.f32.insert(e.name.clone(), F32Mat { ptr: dst as *const f32, len: e.nbytes / 4, shape });
        }
        TensorKind::Iq4nlRows => {
            tw.raw.insert(e.name.clone(), RawRows { ptr: dst, rows: e.rows, row_bytes: e.row_bytes, start_row: e.shard.start });
        }
        TensorKind::Bf16Strips => {
            tw.bf16.insert(e.name.clone(), Bf16Mat { ptr: dst as *const u16, rows: e.rows, k: e.k });
        }
    }
}

/// Dequantise one row of a TQ matrix (item 0) into `out` (len k). Used for embedding rows.
pub fn dequant_tq_row(mat: &TqMat, row: usize, out: &mut [f32]) {
    use tr_format::codec::{block_q, f16_to_f32, STRIP};
    let c = mat.codec;
    let nb = mat.k / c.kb;
    let bb = c.block_bytes();
    let s = row / STRIP;
    let r = row % STRIP;
    for b in 0..nb {
        let blk = unsafe { std::slice::from_raw_parts(mat.ptr.add((s * nb + b) * bb), bb) };
        let d = f16_to_f32(u16::from_le_bytes([blk[2 * r], blk[2 * r + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([blk[32 + 2 * r], blk[32 + 2 * r + 1]]));
        for kk in 0..c.kb {
            out[b * c.kb + kk] = d * block_q(blk, c, r, kk) as f32 + m;
        }
    }
}
