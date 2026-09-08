use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Manifest {
    pub format_version: u32,
    pub hash: String,
    pub identity: serde_json::Value,
    pub source_gguf: String,
    pub shards: Vec<ShardHash>,
    pub config: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub chat_template: String,
    pub n_tiles: usize,
    pub layer_ids: Vec<usize>,
    pub n_expert_packed: usize,
    pub n_vocab: usize,
    pub files: Vec<FileEntry>,
    pub tokenizer: String,
    pub tensors: Vec<TensorEntry>,
    /// Overlay packs (the MTP draft head) reference the base pack they were built for by hash.
    #[serde(default)]
    pub overlay_of: Option<String>,
    #[serde(skip)]
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShardHash {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    #[serde(default)]
    pub fast: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorKind {
    Tq,
    F32,
    Iq4nlRows,
    /// bf16 weights pre-laid out as AMX B-tile strips: [pad16(rows)/16][k/2][16][2] u16
    /// (strip s, pair-row p holds (w[16s+n][2p], w[16s+n][2p+1]) for n in 0..16).
    Bf16Strips,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Shard {
    pub kind: String,
    #[serde(default)]
    pub ranges: Vec<[usize; 2]>,
    #[serde(default)]
    pub start: usize,
    #[serde(default)]
    pub len: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TensorEntry {
    pub name: String,
    pub tile: Option<usize>,
    pub kind: TensorKind,
    pub shard: Shard,
    #[serde(default)]
    pub bits: u8,
    #[serde(default)]
    pub kb: usize,
    #[serde(default = "one")]
    pub count: usize,
    #[serde(default)]
    pub layer: Option<usize>,
    #[serde(default)]
    pub rows: usize,
    #[serde(default)]
    pub k: usize,
    #[serde(default)]
    pub stride: usize,
    pub nbytes: usize,
    pub file: String,
    pub offset: u64,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default)]
    pub row_bytes: usize,
}

fn one() -> usize {
    1
}

impl TensorEntry {
    pub fn codec(&self) -> crate::codec::Codec {
        crate::codec::Codec::new(self.bits, self.kb)
    }
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Manifest> {
        let p = dir.join("manifest.json");
        let text = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let mut m: Manifest = serde_json::from_str(&text).context("parse manifest.json")?;
        m.dir = dir.to_path_buf();
        m.validate()?;
        Ok(m)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION {
            bail!("pack format_version {} != {}", self.format_version, FORMAT_VERSION);
        }
        let canon = serde_json::to_string(&canonical(&self.identity))?;
        let digest = {
            use std::fmt::Write;
            let d = sha256(canon.as_bytes());
            let mut s = String::new();
            for b in &d[..8] {
                write!(s, "{b:02x}").unwrap();
            }
            s
        };
        if digest != self.hash {
            bail!("manifest hash mismatch: {} != computed {}", self.hash, digest);
        }
        let mut sizes: HashMap<&str, u64> = HashMap::new();
        for f in &self.files {
            let meta = std::fs::metadata(self.dir.join(&f.name)).with_context(|| format!("stat {}", f.name))?;
            if meta.len() != f.bytes {
                bail!("{}: size {} != manifest {}", f.name, meta.len(), f.bytes);
            }
            sizes.insert(&f.name, f.bytes);
        }
        for t in &self.tensors {
            let Some(&sz) = sizes.get(t.file.as_str()) else { bail!("{}: unknown file {}", t.name, t.file) };
            if t.offset + t.nbytes as u64 > sz {
                bail!("{}: section beyond end of {}", t.name, t.file);
            }
            if t.kind == TensorKind::Tq {
                let want = t.codec().matrix_bytes(t.rows, t.k) * t.count;
                if want != t.nbytes || t.stride != t.codec().matrix_bytes(t.rows, t.k) {
                    bail!("{}: tq size mismatch (manifest {} vs codec {})", t.name, t.nbytes, want);
                }
                if t.k % t.kb != 0 {
                    bail!("{}: k {} not a multiple of kb {}", t.name, t.k, t.kb);
                }
            }
            if t.kind == TensorKind::F32 && t.nbytes != t.shape.iter().product::<usize>() * 4 {
                bail!("{}: f32 size mismatch", t.name);
            }
            if t.kind == TensorKind::Iq4nlRows && t.nbytes != t.rows * t.row_bytes {
                bail!("{}: iq4nl size mismatch", t.name);
            }
            if t.kind == TensorKind::Bf16Strips && (t.nbytes != t.rows.div_ceil(16) * 16 * t.k * 2 || t.k % 32 != 0) {
                bail!("{}: bf16 strips size mismatch (rows {} k {} nbytes {})", t.name, t.rows, t.k, t.nbytes);
            }
        }
        Ok(())
    }

    pub fn file_path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// Config scalar accessors (`<arch>.<key>`).
    pub fn arch(&self) -> String {
        self.config.get("architecture").and_then(|v| v.as_str()).unwrap_or("").to_string()
    }
    pub fn cfg_u64(&self, key: &str) -> Option<u64> {
        self.config.get(&format!("{}.{}", self.arch(), key)).and_then(|v| v.as_u64())
    }
    pub fn cfg_f64(&self, key: &str) -> Option<f64> {
        self.config.get(&format!("{}.{}", self.arch(), key)).and_then(|v| v.as_f64())
    }
    pub fn cfg_arr_u64(&self, key: &str) -> Vec<u64> {
        self.config
            .get(&format!("{}.{}", self.arch(), key))
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default()
    }

    pub fn tensor(&self, name: &str, tile: Option<usize>) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name && t.tile == tile)
    }
    pub fn tensors_on_tile(&self, tile: usize) -> impl Iterator<Item = &TensorEntry> {
        self.tensors.iter().filter(move |t| t.tile == Some(tile))
    }
    pub fn shared_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors.iter().filter(|t| t.tile.is_none())
    }
    pub fn bytes_on_tile(&self, tile: usize) -> u64 {
        self.tensors_on_tile(tile).map(|t| t.nbytes as u64).sum()
    }
}

/// Canonical JSON (sorted keys, compact) matching Python's json.dumps(sort_keys=True, separators=(",",":")).
fn canonical(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut b = std::collections::BTreeMap::new();
            for (k, v) in m {
                b.insert(k.clone(), canonical(v));
            }
            serde_json::Value::Object(b.into_iter().collect())
        }
        serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(canonical).collect()),
        x => x.clone(),
    }
}

/// Minimal SHA-256 (no external dependency).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut a = h;
        for i in 0..64 {
            let s1 = a[4].rotate_right(6) ^ a[4].rotate_right(11) ^ a[4].rotate_right(25);
            let ch = (a[4] & a[5]) ^ (!a[4] & a[6]);
            let t1 = a[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a[0].rotate_right(2) ^ a[0].rotate_right(13) ^ a[0].rotate_right(22);
            let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
            let t2 = s0.wrapping_add(maj);
            a = [t1.wrapping_add(t2), a[0], a[1], a[2], a[3].wrapping_add(t1), a[4], a[5], a[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(a[i]);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha256_known() {
        let d = sha256(b"abc");
        assert_eq!(d[0], 0xba);
        assert_eq!(d[31], 0xad);
    }
}
