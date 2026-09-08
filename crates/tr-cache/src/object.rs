//! Stored objects: a header (JSON metadata, padded to the I/O alignment) followed by the body.
//! Every object is self-describing, so the index can always be rebuilt from the headers.
use crate::key::PrefixKey;
use crate::role::Role;
use crate::AlignedBuf;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// I/O alignment of headers and bodies (O_DIRECT).
pub const ALIGN: usize = 4096;
/// Object format version; bumped when the header or the body layout changes.
pub const FORMAT: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// K/V + indexer rows of positions `[start, end)`, tagged with a role.
    Rows,
    /// The recurrent state at position `end` (plus the host-side scalars the engine stores).
    Snapshot,
}

impl Kind {
    pub fn ext(self) -> &'static str {
        match self {
            Kind::Rows => "rows",
            Kind::Snapshot => "snap",
        }
    }
    pub fn from_ext(e: &str) -> Option<Kind> {
        match e {
            "rows" => Some(Kind::Rows),
            "snap" => Some(Kind::Snapshot),
            _ => None,
        }
    }
}

/// Metadata of one object; the on-disk header and the index entry are this struct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectMeta {
    pub format: u32,
    /// Root key of the store the object belongs to (engine identity).
    pub root: PrefixKey,
    pub kind: Kind,
    /// Prefix key at `end`: the object's address.
    pub key: PrefixKey,
    /// Prefix key at `start` (rows: the previous chunk's key; snapshot: same as `key`).
    pub parent: PrefixKey,
    pub start: usize,
    pub end: usize,
    /// Rows: the tokens' role. Snapshots: the role of the span that ends at the position (None
    /// for objects written before roles were recorded on snapshots).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// Rows only: the token ids of `[start, end)` (verified against the prompt on restore).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<u32>,
    /// Body length in bytes (the file may be padded beyond it).
    pub bytes: u64,
    /// Unix seconds of the last hit (informational; the LRU order is kept in the index).
    pub last_used: u64,
}

impl ObjectMeta {
    pub fn rows(root: PrefixKey, key: PrefixKey, parent: PrefixKey, start: usize, end: usize, role: Role, tokens: Vec<u32>, bytes: u64) -> ObjectMeta {
        assert_eq!(tokens.len(), end - start);
        ObjectMeta { format: FORMAT, root, kind: Kind::Rows, key, parent, start, end, role: Some(role), tokens, bytes, last_used: now() }
    }
    pub fn snapshot(root: PrefixKey, key: PrefixKey, pos: usize, role: Option<Role>, bytes: u64) -> ObjectMeta {
        ObjectMeta { format: FORMAT, root, kind: Kind::Snapshot, key, parent: key, start: pos, end: pos, role, tokens: Vec::new(), bytes, last_used: now() }
    }
    /// File / blob name: `<key hex>.<rows|snap>`.
    pub fn name(&self) -> String {
        object_name(&self.key, self.kind)
    }
    /// Bytes the object occupies once padded (header + body), for the budget.
    pub fn footprint(&self) -> u64 {
        padded(self.header_len_hint()) as u64 + padded(self.bytes as usize) as u64
    }
    fn header_len_hint(&self) -> usize {
        // the JSON header grows with the token list; a rough bound keeps this cheap
        256 + 12 * self.tokens.len()
    }
    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT {
            bail!("object format {} (this build reads {FORMAT})", self.format);
        }
        if self.start > self.end {
            bail!("object {} has start {} > end {}", self.name(), self.start, self.end);
        }
        match self.kind {
            Kind::Rows => {
                if self.tokens.len() != self.end - self.start {
                    bail!("rows object {} covers {} positions but lists {} tokens", self.name(), self.end - self.start, self.tokens.len());
                }
                if self.role.is_none() {
                    bail!("rows object {} has no role", self.name());
                }
            }
            Kind::Snapshot => {
                if self.start != self.end {
                    bail!("snapshot object {} spans {}..{}", self.name(), self.start, self.end);
                }
            }
        }
        Ok(())
    }
}

pub fn object_name(key: &PrefixKey, kind: Kind) -> String {
    format!("{}.{}", key.hex(), kind.ext())
}

/// `(key, kind)` of an object name, None for anything that is not one.
pub fn parse_name(name: &str) -> Option<(PrefixKey, Kind)> {
    let (k, e) = name.rsplit_once('.')?;
    Some((PrefixKey::parse(k)?, Kind::from_ext(e)?))
}

pub fn padded(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The header bytes: `u64 json_len` little-endian, the JSON, zero padding to a multiple of `ALIGN`.
pub fn encode_header(meta: &ObjectMeta) -> Result<AlignedBuf> {
    let json = serde_json::to_vec(meta)?;
    let mut buf = AlignedBuf::new(padded(8 + json.len()));
    buf[..8].copy_from_slice(&(json.len() as u64).to_le_bytes());
    buf[8..8 + json.len()].copy_from_slice(&json);
    Ok(buf)
}

/// Length of the JSON part from the first 8 bytes.
pub fn header_json_len(first8: &[u8]) -> Result<usize> {
    if first8.len() < 8 {
        bail!("short header");
    }
    let n = u64::from_le_bytes(first8[..8].try_into().unwrap());
    if n == 0 || n > (64 << 20) {
        bail!("implausible header length {n}");
    }
    Ok(n as usize)
}

/// Decode a header from the start of `bytes` (which must hold all of it). Returns the metadata
/// and the padded header length (the body offset).
pub fn decode_header(bytes: &[u8]) -> Result<(ObjectMeta, usize)> {
    let n = header_json_len(bytes)?;
    if bytes.len() < 8 + n {
        bail!("header truncated: {} of {} bytes", bytes.len(), 8 + n);
    }
    let meta: ObjectMeta = serde_json::from_slice(&bytes[8..8 + n]).context("object header")?;
    meta.validate()?;
    Ok((meta, padded(8 + n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::root_key;

    #[test]
    fn header_round_trip_and_padding() {
        let root = root_key(&["r"]);
        let m = ObjectMeta::rows(root, root_key(&["k"]), root, 10, 13, Role::User, vec![5, 6, 7], 77777);
        let h = encode_header(&m).unwrap();
        assert_eq!(h.len() % ALIGN, 0);
        let (back, off) = decode_header(&h).unwrap();
        assert_eq!(back, m);
        assert_eq!(off, h.len());
        assert_eq!(m.name(), format!("{}.rows", root_key(&["k"]).hex()));
        assert_eq!(parse_name(&m.name()), Some((root_key(&["k"]), Kind::Rows)));
        assert_eq!(parse_name("index.json"), None);
        assert_eq!(m.footprint(), 4096 + 77824, "header page + body padded to 4 KiB");
        let s = ObjectMeta::snapshot(root, root_key(&["s"]), 13, Some(Role::User), 4096);
        assert_eq!(s.name(), format!("{}.snap", root_key(&["s"]).hex()));
        s.validate().unwrap();
        // truncated / wrong format are errors
        assert!(decode_header(&h[..100]).is_err());
        let mut bad = m.clone();
        bad.format = 99;
        let hb = encode_header(&bad).unwrap();
        assert!(decode_header(&hb).is_err());
        let mut bad = m.clone();
        bad.tokens.pop();
        assert!(bad.validate().is_err());
    }
}
