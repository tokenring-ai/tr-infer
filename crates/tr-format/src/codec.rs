//! TQ block codec layout (see python/trpack/codec.py for the authoritative description).
//!
//! Block = 16 rows x KB k-columns: `d[16] f16 | m[16] f16 | q bytes | (5-bit) high-bit plane`.
//! 4/5-bit q: chunk j (0..KB/8) is 64 bytes = 16 rows x 4 k as dwords; low nibble k = 4j+i,
//! high nibble k = KB/2 + 4j + i. 5-bit plane: per chunk j, 64 bits (low half) then 64 bits
//! (high half), bit index r*4+i. 8-bit q: chunk j (0..KB/4) is 64 bytes, k = 4j+i.
//! w[r][k] = d[r]*q + m[r].

pub const STRIP: usize = 16;
pub const HDR_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Codec {
    pub bits: u8,
    pub kb: usize,
}

impl Codec {
    pub const fn new(bits: u8, kb: usize) -> Codec {
        Codec { bits, kb }
    }
    pub const fn block_bytes(&self) -> usize {
        match self.bits {
            8 => HDR_BYTES + STRIP * self.kb,
            4 => HDR_BYTES + STRIP * self.kb / 2,
            5 => HDR_BYTES + STRIP * self.kb / 2 + 2 * self.kb,
            _ => panic!("bad bits"),
        }
    }
    pub const fn q_bytes(&self) -> usize {
        match self.bits {
            8 => STRIP * self.kb,
            _ => STRIP * self.kb / 2,
        }
    }
    pub const fn strip_bytes(&self, k: usize) -> usize {
        (k / self.kb) * self.block_bytes()
    }
    pub const fn matrix_bytes(&self, rows: usize, k: usize) -> usize {
        pad_rows(rows) / STRIP * self.strip_bytes(k)
    }
}

pub const fn pad_rows(rows: usize) -> usize {
    (rows + STRIP - 1) / STRIP * STRIP
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Reference (scalar) dequant of one packed matrix into row-major f32 [rows][k]. Tests only.
pub fn dequant_ref(buf: &[u8], rows: usize, k: usize, c: Codec) -> Vec<f32> {
    let rp = pad_rows(rows);
    let nb = k / c.kb;
    let bb = c.block_bytes();
    let mut out = vec![0f32; rows * k];
    for s in 0..rp / STRIP {
        for b in 0..nb {
            let blk = &buf[(s * nb + b) * bb..(s * nb + b + 1) * bb];
            for r in 0..STRIP {
                let row = s * STRIP + r;
                if row >= rows {
                    continue;
                }
                let d = f16_to_f32(u16::from_le_bytes([blk[2 * r], blk[2 * r + 1]]));
                let m = f16_to_f32(u16::from_le_bytes([blk[32 + 2 * r], blk[32 + 2 * r + 1]]));
                for kk in 0..c.kb {
                    let q = block_q(blk, c, r, kk);
                    out[row * k + b * c.kb + kk] = d * q as f32 + m;
                }
            }
        }
    }
    out
}

/// Unsigned q value of (row r, column kk) inside one block.
pub fn block_q(blk: &[u8], c: Codec, r: usize, kk: usize) -> u32 {
    let qb = &blk[HDR_BYTES..];
    if c.bits == 8 {
        let j = kk / 4;
        let i = kk % 4;
        return qb[j * 64 + r * 4 + i] as u32;
    }
    let half = c.kb / 2;
    let (h, kh) = if kk < half { (0, kk) } else { (1, kk - half) };
    let j = kh / 4;
    let i = kh % 4;
    let byte = qb[j * 64 + r * 4 + i];
    let mut q = if h == 0 { byte & 0x0F } else { byte >> 4 } as u32;
    if c.bits == 5 {
        let plane = &qb[c.kb / 8 * 64..];
        let bit = r * 4 + i;
        let byte = plane[j * 16 + h * 8 + bit / 8];
        q |= (((byte >> (bit % 8)) & 1) as u32) << 4;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_match_python() {
        assert_eq!(Codec::new(4, 32).block_bytes(), 64 + 256);
        assert_eq!(Codec::new(5, 32).block_bytes(), 64 + 256 + 64);
        assert_eq!(Codec::new(8, 32).block_bytes(), 64 + 512);
        assert_eq!(Codec::new(5, 16).block_bytes(), 64 + 128 + 32);
        assert_eq!(Codec::new(4, 32).matrix_bytes(80, 2560), 5 * 80 * 320);
        assert_eq!(Codec::new(5, 16).matrix_bytes(2560, 80), 160 * 5 * 224);
    }
}

/// Reference (scalar) packer: q [rows*k] unsigned, d/m [rows*(k/kb)] f16 bits -> packed bytes.
/// Rows are zero-padded to a strip. Tests and synthetic data only (the Python packer is the writer).
pub fn pack_ref(rows: usize, k: usize, c: Codec, q: &[u32], d: &[u16], m: &[u16]) -> Vec<u8> {
    let rp = pad_rows(rows);
    let nb = k / c.kb;
    let bb = c.block_bytes();
    let mut out = vec![0u8; rp / STRIP * nb * bb];
    for s in 0..rp / STRIP {
        for b in 0..nb {
            let blk = &mut out[(s * nb + b) * bb..(s * nb + b + 1) * bb];
            for r in 0..STRIP {
                let row = s * STRIP + r;
                if row >= rows {
                    continue;
                }
                blk[2 * r..2 * r + 2].copy_from_slice(&d[row * nb + b].to_le_bytes());
                blk[32 + 2 * r..32 + 2 * r + 2].copy_from_slice(&m[row * nb + b].to_le_bytes());
                for kk in 0..c.kb {
                    let qv = q[row * k + b * c.kb + kk];
                    set_block_q(blk, c, r, kk, qv);
                }
            }
        }
    }
    out
}

fn set_block_q(blk: &mut [u8], c: Codec, r: usize, kk: usize, q: u32) {
    let qb = &mut blk[HDR_BYTES..];
    if c.bits == 8 {
        qb[(kk / 4) * 64 + r * 4 + kk % 4] = q as u8;
        return;
    }
    let half = c.kb / 2;
    let (h, kh) = if kk < half { (0, kk) } else { (1, kk - half) };
    let j = kh / 4;
    let i = kh % 4;
    let idx = j * 64 + r * 4 + i;
    let nib = (q & 0xF) as u8;
    if h == 0 {
        qb[idx] = (qb[idx] & 0xF0) | nib;
    } else {
        qb[idx] = (qb[idx] & 0x0F) | (nib << 4);
    }
    if c.bits == 5 {
        let bit = r * 4 + i;
        let pidx = c.kb / 8 * 64 + j * 16 + h * 8 + bit / 8;
        if (q >> 4) & 1 == 1 {
            qb[pidx] |= 1 << (bit % 8);
        } else {
            qb[pidx] &= !(1 << (bit % 8));
        }
    }
}

#[cfg(test)]
mod pack_tests {
    use super::*;
    #[test]
    fn pack_ref_roundtrip() {
        for &(bits, kb, rows, k) in &[(4u8, 32usize, 40usize, 64usize), (5, 32, 16, 96), (8, 32, 33, 32), (5, 16, 16, 80), (8, 16, 16, 48)] {
            let c = Codec::new(bits, kb);
            let nb = k / kb;
            let mut seed = 12345u32;
            let mut rnd = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed >> 8
            };
            let q: Vec<u32> = (0..rows * k).map(|_| rnd() % (1 << bits)).collect();
            let d: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32((rnd() % 1000) as f32 * 1e-4 + 1e-3).to_bits()).collect();
            let m: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32((rnd() % 2000) as f32 * 1e-3 - 1.0).to_bits()).collect();
            let buf = pack_ref(rows, k, c, &q, &d, &m);
            assert_eq!(buf.len(), c.matrix_bytes(rows, k));
            let w = dequant_ref(&buf, rows, k, c);
            for r in 0..rows {
                for kk in 0..k {
                    let want = f16_to_f32(d[r * nb + kk / kb]) * q[r * k + kk] as f32 + f16_to_f32(m[r * nb + kk / kb]);
                    assert_eq!(w[r * k + kk], want, "bits {bits} kb {kb} [{r}][{kk}]");
                }
            }
        }
    }
}
