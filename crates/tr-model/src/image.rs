//! Image preprocessing for the vision encoder (Qwen-VL style dynamic resolution).
//!
//! An RGB8 image is resized to a multiple of `patch * merge` (32) per side, keeping the aspect
//! ratio within a token budget ("smart_resize"), with a Pillow-style bicubic filter (fixed-point,
//! a = -0.5, widened when downsampling, the same as llama.cpp's port), normalised with the model's
//! mean/std, and cut into 16x16 patches in the *merge-block* order the encoder's merger expects:
//! for every 2x2 block of patches, (y, x), (y, x+1), (y+1, x), (y+1, x+1). Each patch vector is
//! laid out channel-major (c, ky, kx), the conv kernel's order. The learned 48x48 position table is
//! bilinearly resampled (align corners, as ggml_interpolate does) to the patch grid and reordered
//! the same way.

/// Fixed-point precision of the resampling weights (Pillow's).
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Target size for an image of `w` x `h`: aspect-preserving, both sides multiples of `align`, area
/// within [min_pixels, max_pixels] (transformers' smart_resize; llama.cpp's calc_size_preserved_ratio).
pub fn smart_size(w: usize, h: usize, align: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let a = align as f64;
    let round_by = |x: f64| ((x / a).round() as usize * align).max(align);
    let ceil_by = |x: f64| (x / a).ceil() as usize * align;
    let floor_by = |x: f64| ((x / a).floor() as usize * align).max(align);
    let (wf, hf) = (w as f64, h as f64);
    let mut wb = round_by(wf);
    let mut hb = round_by(hf);
    if max_pixels > 0 && hb * wb > max_pixels {
        // f32 like the reference: the sqrt rounding decides the size for borderline images
        let beta = ((hf * wf) as f32 / max_pixels as f32).sqrt() as f64;
        hb = floor_by(hf / beta);
        wb = floor_by(wf / beta);
    } else if min_pixels > 0 && hb * wb < min_pixels {
        let beta = (min_pixels as f32 / (hf * wf) as f32).sqrt() as f64;
        hb = ceil_by(hf * beta);
        wb = ceil_by(wf * beta);
    }
    (wb, hb)
}

/// How the resized image fills the target rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fit {
    /// Stretch to the target (transformers' behaviour; the target already has nearly the same aspect).
    Stretch,
    /// Scale to fit (ceil), centre, pad with black (llama.cpp's default `PAD_CEIL`).
    PadCeil,
}

/// Pillow-style separable bicubic resize of an RGB8 image (`src` is `sw*sh*3`).
pub fn resize_bicubic(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * 3);
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    let mut cur = src.to_vec();
    let (mut cw, mut ch) = (sw, sh);
    if dw != sw {
        let (ksize, bounds, weights) = precompute_weights(sw, dw);
        let mut out = vec![0u8; dw * ch * 3];
        for y in 0..ch {
            let row = &cur[y * cw * 3..(y + 1) * cw * 3];
            let dst = &mut out[y * dw * 3..(y + 1) * dw * 3];
            for xx in 0..dw {
                let (xmin, xcnt) = (bounds[xx * 2], bounds[xx * 2 + 1]);
                let k = &weights[xx * ksize..xx * ksize + xcnt];
                let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
                for (x, &kw) in k.iter().enumerate() {
                    let p = &row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                    for c in 0..3 {
                        ss[c] = ss[c].wrapping_add(p[c] as i32 * kw);
                    }
                }
                for c in 0..3 {
                    dst[xx * 3 + c] = clip8(ss[c] >> PRECISION_BITS);
                }
            }
        }
        cur = out;
        cw = dw;
    }
    if dh != sh {
        let (ksize, bounds, weights) = precompute_weights(sh, dh);
        let row_elems = cw * 3;
        let mut out = vec![0u8; row_elems * dh];
        let mut acc = vec![0i32; row_elems];
        for yy in 0..dh {
            let (ymin, ycnt) = (bounds[yy * 2], bounds[yy * 2 + 1]);
            let k = &weights[yy * ksize..yy * ksize + ycnt];
            acc.fill(1i32 << (PRECISION_BITS - 1));
            for (y, &kw) in k.iter().enumerate() {
                let row = &cur[(ymin + y) * row_elems..(ymin + y + 1) * row_elems];
                for (a, &p) in acc.iter_mut().zip(row) {
                    *a = a.wrapping_add(p as i32 * kw);
                }
            }
            let dst = &mut out[yy * row_elems..(yy + 1) * row_elems];
            for (d, &a) in dst.iter_mut().zip(&acc) {
                *d = clip8(a >> PRECISION_BITS);
            }
        }
        cur = out;
        ch = dh;
    }
    debug_assert_eq!((cw, ch), (dw, dh));
    cur
}

#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn bicubic(mut x: f64) -> f64 {
    const A: f64 = -0.5;
    if x < 0.0 {
        x = -x;
    }
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Filter taps for one dimension: (ksize, bounds [out*2] = (first, count), weights [out*ksize] fixed-point).
fn precompute_weights(in_size: usize, out_size: usize) -> (usize, Vec<usize>, Vec<i32>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;
    let mut bounds = vec![0usize; out_size * 2];
    let mut weights = vec![0i32; out_size * ksize];
    let mut pre = vec![0f64; ksize];
    let fxp = (1u64 << PRECISION_BITS) as f64;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let cnt = xmax - xmin;
        let mut ww = 0.0;
        for x in 0..cnt {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            pre[x] = w;
            ww += w;
        }
        for x in 0..cnt {
            if ww != 0.0 {
                pre[x] /= ww;
            }
            // Pillow adds +-0.5 and truncates toward zero
            let r = pre[x] * fxp + if pre[x] < 0.0 { -0.5 } else { 0.5 };
            weights[xx * ksize + x] = r as i32;
        }
        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = cnt;
    }
    (ksize, bounds, weights)
}

/// Resize `src` (`sw` x `sh` RGB8) into `dw` x `dh` with the given fit.
pub fn resize_fit(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, fit: Fit) -> Vec<u8> {
    match fit {
        Fit::Stretch => resize_bicubic(src, sw, sh, dw, dh),
        Fit::PadCeil => {
            let scale = (dw as f32 / sw as f32).min(dh as f32 / sh as f32);
            let nw = ((sw as f32 * scale).ceil() as usize).min(dw);
            let nh = ((sh as f32 * scale).ceil() as usize).min(dh);
            let inner = resize_bicubic(src, sw, sh, nw, nh);
            let mut out = vec![0u8; dw * dh * 3];
            let (ox, oy) = ((dw - nw) / 2, (dh - nh) / 2);
            for y in 0..nh {
                out[((oy + y) * dw + ox) * 3..((oy + y) * dw + ox + nw) * 3].copy_from_slice(&inner[y * nw * 3..(y + 1) * nw * 3]);
            }
            out
        }
    }
}

/// Preprocessing parameters (from the vision pack's config).
#[derive(Clone, Debug)]
pub struct PrepParams {
    pub patch: usize,
    pub merge: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub fit: Fit,
}

/// A preprocessed image: normalised patch vectors in merge-block order plus their grid positions.
#[derive(Clone, Debug)]
pub struct Patches {
    /// Grid in patches (width, height); both multiples of `merge`.
    pub gw: usize,
    pub gh: usize,
    /// [n_patches][patch*patch*3] f32
    pub data: Vec<f32>,
    /// Patch row / column of every vector (encoder rope positions).
    pub ypos: Vec<u32>,
    pub xpos: Vec<u32>,
    /// Resized pixel size.
    pub width: usize,
    pub height: usize,
}

impl Patches {
    pub fn n_patches(&self) -> usize {
        self.gw * self.gh
    }
    pub fn patch_len(&self) -> usize {
        self.data.len() / self.n_patches().max(1)
    }
    /// Merged-token grid (width, height) for `merge`.
    pub fn tokens(&self, merge: usize) -> (usize, usize) {
        (self.gw / merge, self.gh / merge)
    }
}

/// Resize, normalise and cut an RGB8 image into patches.
pub fn prepare(rgb: &[u8], w: usize, h: usize, p: &PrepParams) -> Patches {
    assert_eq!(rgb.len(), w * h * 3, "prepare: {w}x{h} rgb of {} bytes", rgb.len());
    let align = p.patch * p.merge;
    let area = align * align;
    let (dw, dh) = smart_size(w, h, align, p.min_tokens * area, p.max_tokens * area);
    let img = resize_fit(rgb, w, h, dw, dh, p.fit);
    let (gw, gh) = (dw / p.patch, dh / p.patch);
    let ps = p.patch;
    let plen = ps * ps * 3;
    let n = gw * gh;
    let mut data = vec![0f32; n * plen];
    let mut ypos = Vec::with_capacity(n);
    let mut xpos = Vec::with_capacity(n);
    let inv = [1.0 / 255.0 / p.std[0], 1.0 / 255.0 / p.std[1], 1.0 / 255.0 / p.std[2]];
    let off = [p.mean[0] / p.std[0], p.mean[1] / p.std[1], p.mean[2] / p.std[2]];
    let mut i = 0;
    for by in (0..gh).step_by(p.merge) {
        for bx in (0..gw).step_by(p.merge) {
            for dy in 0..p.merge {
                for dx in 0..p.merge {
                    let (py, px) = (by + dy, bx + dx);
                    let v = &mut data[i * plen..(i + 1) * plen];
                    for c in 0..3 {
                        for ky in 0..ps {
                            let row = (py * ps + ky) * dw;
                            for kx in 0..ps {
                                let pix = img[(row + px * ps + kx) * 3 + c] as f32;
                                v[c * ps * ps + ky * ps + kx] = pix * inv[c] - off[c];
                            }
                        }
                    }
                    ypos.push(py as u32);
                    xpos.push(px as u32);
                    i += 1;
                }
            }
        }
    }
    Patches { gw, gh, data, ypos, xpos, width: dw, height: dh }
}

/// The learned position table (`n_side` x `n_side` rows of `dim`) resampled to a `gw` x `gh` patch
/// grid (bilinear, align corners) and reordered into merge-block order: [gw*gh][dim].
pub fn pos_rows(table: &[f32], n_side: usize, dim: usize, gw: usize, gh: usize, merge: usize) -> Vec<f32> {
    assert_eq!(table.len(), n_side * n_side * dim);
    let mut out = vec![0f32; gw * gh * dim];
    let coord = |i: usize, n_out: usize| -> (usize, usize, f32) {
        if n_out <= 1 || n_side <= 1 {
            return (0, 0, 0.0);
        }
        let x = i as f32 * (n_side as f32 - 1.0) / (n_out as f32 - 1.0);
        let x0 = (x.floor() as usize).min(n_side - 1);
        let x1 = (x0 + 1).min(n_side - 1);
        (x0, x1, x - x0 as f32)
    };
    let mut i = 0;
    for by in (0..gh).step_by(merge) {
        for bx in (0..gw).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    let (py, px) = (by + dy, bx + dx);
                    let (y0, y1, fy) = coord(py, gh);
                    let (x0, x1, fx) = coord(px, gw);
                    let r00 = &table[(y0 * n_side + x0) * dim..][..dim];
                    let r01 = &table[(y0 * n_side + x1) * dim..][..dim];
                    let r10 = &table[(y1 * n_side + x0) * dim..][..dim];
                    let r11 = &table[(y1 * n_side + x1) * dim..][..dim];
                    let o = &mut out[i * dim..(i + 1) * dim];
                    for d in 0..dim {
                        let top = r00[d] * (1.0 - fx) + r01[d] * fx;
                        let bot = r10[d] * (1.0 - fx) + r11[d] * fx;
                        o[d] = top * (1.0 - fy) + bot * fy;
                    }
                    i += 1;
                }
            }
        }
    }
    out
}

/// An image's place in a token sequence: `n = nx*ny` rows starting at `row` (the pad tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagePlace {
    pub row: usize,
    pub nx: usize,
    pub ny: usize,
}

/// M-RoPE positions of `n_rows` tokens (Qwen-VL): text tokens advance the position by one; the
/// tokens of an image at position p get (p, p + i / nx, p + i % nx) and the text after it
/// continues at p + max(nx, ny). Returns the triple of every row and the position the row
/// after each row starts at. `imgs` must be sorted by row and not overlap.
pub fn mrope_positions(n_rows: usize, imgs: &[ImagePlace]) -> (Vec<[u32; 3]>, Vec<usize>) {
    let mut pos3 = Vec::with_capacity(n_rows);
    let mut after = Vec::with_capacity(n_rows);
    let mut p = 0usize;
    let mut r = 0usize;
    let mut it = imgs.iter().peekable();
    while r < n_rows {
        match it.peek() {
            Some(im) if im.row == r => {
                let n = im.nx * im.ny;
                assert!(im.nx > 0 && im.ny > 0 && r + n <= n_rows, "image rows exceed the sequence");
                let end = p + im.nx.max(im.ny);
                for i in 0..n {
                    pos3.push([p as u32, (p + i / im.nx) as u32, (p + i % im.nx) as u32]);
                    after.push(if i + 1 == n { end } else { p });
                }
                r += n;
                p = end;
                it.next();
            }
            Some(im) if im.row < r => panic!("overlapping image placements"),
            _ => {
                pos3.push([p as u32; 3]);
                p += 1;
                after.push(p);
                r += 1;
            }
        }
    }
    (pos3, after)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrope_positions_follow_qwen_vl() {
        // text text [img 3x2] text
        let (p, after) = mrope_positions(9, &[ImagePlace { row: 2, nx: 3, ny: 2 }]);
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[1], [1, 1, 1]);
        assert_eq!(p[2], [2, 2, 2]);
        assert_eq!(p[3], [2, 2, 3]);
        assert_eq!(p[4], [2, 2, 4]);
        assert_eq!(p[5], [2, 3, 2]);
        assert_eq!(p[7], [2, 3, 4]);
        assert_eq!(p[8], [5, 5, 5]); // 2 + max(3, 2)
        assert_eq!(after[1], 2);
        assert_eq!(after[7], 5);
        assert_eq!(after[8], 6);
        let (p, after) = mrope_positions(3, &[]);
        assert_eq!(p, vec![[0; 3], [1; 3], [2; 3]]);
        assert_eq!(after, vec![1, 2, 3]);
    }

    #[test]
    fn smart_sizes_match_reference() {
        // llama.cpp defaults: 8..4096 tokens of 32x32
        let (lo, hi) = (8 * 1024, 4096 * 1024);
        assert_eq!(smart_size(64, 64, 32, lo, hi), (96, 96)); // below the minimum: scaled up
        assert_eq!(smart_size(224, 224, 32, lo, hi), (224, 224));
        assert_eq!(smart_size(224, 200, 32, lo, hi), (224, 192));
        assert_eq!(smart_size(640, 488, 32, lo, hi), (640, 480));
        assert_eq!(smart_size(4000, 3000, 32, lo, hi), (2336, 1760)); // above the maximum: scaled down
        assert_eq!(smart_size(10, 3000, 32, lo, hi).0, 32);
    }

    #[test]
    fn resize_identity_and_constant() {
        let img: Vec<u8> = (0..8 * 4 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_bicubic(&img, 8, 4, 8, 4), img);
        let flat = vec![77u8; 10 * 6 * 3];
        let out = resize_bicubic(&flat, 10, 6, 32, 32);
        assert!(out.iter().all(|&v| v == 77));
        let out = resize_bicubic(&flat, 10, 6, 4, 2);
        assert!(out.iter().all(|&v| v == 77));
        // a horizontal ramp stays monotone and keeps its ends
        let ramp: Vec<u8> = (0..64).flat_map(|x| [x as u8 * 4; 3]).collect();
        let out = resize_bicubic(&ramp, 64, 1, 32, 1);
        assert!(out.chunks(3).map(|p| p[0]).collect::<Vec<_>>().windows(2).all(|w| w[0] <= w[1]));
        assert!(out[0] < 8 && out[31 * 3] > 240);
        // padding fit: black bars around a white image
        let white = vec![255u8; 64 * 48 * 3];
        let out = resize_fit(&white, 64, 48, 64, 32, Fit::PadCeil);
        assert_eq!(out.len(), 64 * 32 * 3);
        assert_eq!(out[0], 0);
        assert_eq!(out[(10 * 64 + 32) * 3], 255);
    }

    #[test]
    fn patches_block_order_and_layout() {
        let (w, h) = (64, 32);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                rgb[(y * w + x) * 3] = x as u8;
                rgb[(y * w + x) * 3 + 1] = y as u8;
                rgb[(y * w + x) * 3 + 2] = 128;
            }
        }
        let p = PrepParams { patch: 16, merge: 2, min_tokens: 1, max_tokens: 4096, mean: [0.5; 3], std: [0.5; 3], fit: Fit::Stretch };
        let pt = prepare(&rgb, w, h, &p);
        assert_eq!((pt.gw, pt.gh), (4, 2));
        assert_eq!(pt.ypos, vec![0, 0, 1, 1, 0, 0, 1, 1]);
        assert_eq!(pt.xpos, vec![0, 1, 0, 1, 2, 3, 2, 3]);
        // patch 5 = (y 0, x 3): channel 0 = x/255*2-1 for x in 48..64, channel 2 = 128 -> ~0
        let v = &pt.data[5 * 768..6 * 768];
        assert!((v[0] - (48.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-6);
        assert!((v[15] - (63.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-6);
        assert!((v[2 * 256] - (128.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-6);
        assert!((v[256 + 16 * 3 + 2] - (3.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-6); // channel 1 (y) at ky=3
    }

    #[test]
    fn pos_rows_interpolates_align_corners() {
        // 3x3 table with value = 10*y + x in dim 0
        let n = 3;
        let table: Vec<f32> = (0..9).map(|i| (10 * (i / 3) + i % 3) as f32).collect();
        let out = pos_rows(&table, n, 1, 2, 2, 2);
        // 2x2 grid: corners of the table, block order (0,0),(0,1),(1,0),(1,1)
        assert_eq!(out, vec![0.0, 2.0, 20.0, 22.0]);
        let out = pos_rows(&table, n, 1, 4, 2, 2);
        // x positions 0, 2/3, 4/3, 2 ; rows y=0 and y=2
        let xs = [0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0];
        let expect = [xs[0], xs[1], 20.0 + xs[0], 20.0 + xs[1], xs[2], xs[3], 20.0 + xs[2], 20.0 + xs[3]];
        for (a, b) in out.iter().zip(expect) {
            assert!((a - b).abs() < 1e-5, "{out:?}");
        }
        // the native grid is the table itself
        let out = pos_rows(&table, n, 1, 3, 3, 1);
        assert_eq!(out, table);
    }
}
