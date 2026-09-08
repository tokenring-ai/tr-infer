//! IQ4_NL row dequant (PLE table rows): block of 32 = f16 d + 16 bytes; low nibbles = elems 0..15,
//! high nibbles = 16..31; value = d * KVALUES[q].
pub const KVALUES: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
pub const BLOCK_BYTES: usize = 18;

/// Dequantise `n` elements (multiple of 32) from `src` into `out`.
pub fn dequant_row(src: &[u8], n: usize, out: &mut [f32]) {
    for b in 0..n / 32 {
        let blk = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for i in 0..16 {
            let byte = blk[2 + i];
            out[b * 32 + i] = d * KVALUES[(byte & 0x0F) as usize] as f32;
            out[b * 32 + 16 + i] = d * KVALUES[(byte >> 4) as usize] as f32;
        }
    }
}
