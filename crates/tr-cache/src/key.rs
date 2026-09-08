//! Prefix keys: a 128-bit chain hash over everything that determines the engine state at a
//! position. `keys[p]` covers positions `0..p`, so `keys[0]` is the root (model identity only).
use serde::{Deserialize, Serialize};
use std::fmt;
use tr_format::manifest::sha256;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrefixKey(pub [u8; 16]);

impl PrefixKey {
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
    pub fn parse(s: &str) -> Option<PrefixKey> {
        if s.len() != 32 {
            return None;
        }
        let mut k = [0u8; 16];
        for i in 0..16 {
            k[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(PrefixKey(k))
    }
    fn of(data: &[u8]) -> PrefixKey {
        let h = sha256(data);
        let mut k = [0u8; 16];
        k.copy_from_slice(&h[..16]);
        PrefixKey(k)
    }
}

impl fmt::Debug for PrefixKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hex())
    }
}
impl fmt::Display for PrefixKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hex())
    }
}
impl Serialize for PrefixKey {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.hex())
    }
}
impl<'de> Deserialize<'de> for PrefixKey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        PrefixKey::parse(&s).ok_or_else(|| serde::de::Error::custom(format!("bad prefix key {s:?}")))
    }
}

/// The root key of a store: everything about the engine that changes the bytes of the state
/// (pack hash, K/V element type, tile count, overlay packs, the state layout version).
pub fn root_key(material: &[&str]) -> PrefixKey {
    let mut buf = Vec::new();
    for m in material {
        buf.extend_from_slice(&(m.len() as u32).to_le_bytes());
        buf.extend_from_slice(m.as_bytes());
    }
    PrefixKey::of(&buf)
}

/// Chain keys of `toks`: `len + 1` entries, `keys[p]` = hash of the root and positions `0..p`.
/// `images` are `(first row, content hash, rows)`; an image's content is folded in at its first
/// row (its rows all carry the pad token, so the ids alone would not tell images apart).
pub fn prefix_keys(root: &PrefixKey, toks: &[u32], images: &[(usize, u64, usize)]) -> Vec<PrefixKey> {
    let mut keys = Vec::with_capacity(toks.len() + 1);
    keys.push(*root);
    let mut img = images.iter().peekable();
    let mut buf = [0u8; 16 + 4 + 8];
    for (p, &t) in toks.iter().enumerate() {
        let prev = keys[p];
        buf[..16].copy_from_slice(&prev.0);
        buf[16..20].copy_from_slice(&t.to_le_bytes());
        let mut n = 20;
        while let Some(im) = img.peek() {
            if im.0 < p {
                img.next();
                continue;
            }
            if im.0 == p {
                buf[20..28].copy_from_slice(&im.1.to_le_bytes());
                n = 28;
                img.next();
            }
            break;
        }
        keys.push(PrefixKey::of(&buf[..n]));
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_stable_and_sensitive() {
        let root = root_key(&["pack-abc", "f16", "8", "-", "-", "1"]);
        let k1 = prefix_keys(&root, &[1, 2, 3], &[]);
        let k2 = prefix_keys(&root, &[1, 2, 3], &[]);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 4);
        assert_eq!(k1[0], root);
        // a prefix shares its keys
        let k3 = prefix_keys(&root, &[1, 2, 9], &[]);
        assert_eq!(k3[..3], k1[..3]);
        assert_ne!(k3[3], k1[3]);
        // the root changes everything but the sequence stays consistent
        let root2 = root_key(&["pack-abc", "bf16", "8", "-", "-", "1"]);
        assert_ne!(prefix_keys(&root2, &[1, 2, 3], &[])[1], k1[1]);
        // images: same ids, different content -> different keys from the image on
        let a = prefix_keys(&root, &[1, 7, 7, 2], &[(1, 0xabc, 2)]);
        let b = prefix_keys(&root, &[1, 7, 7, 2], &[(1, 0xdef, 2)]);
        assert_eq!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
        assert_ne!(a[4], b[4]);
        assert_ne!(a[2], k1[2]);
    }

    #[test]
    fn hex_round_trip() {
        let k = root_key(&["x"]);
        assert_eq!(PrefixKey::parse(&k.hex()), Some(k));
        let j = serde_json::to_string(&k).unwrap();
        let back: PrefixKey = serde_json::from_str(&j).unwrap();
        assert_eq!(back, k);
        assert!(PrefixKey::parse("zz").is_none());
    }
}
