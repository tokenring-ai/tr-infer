//! Image inputs for chat completions: fetch (data URI or http(s)), decode to RGB8, preprocess
//! into encoder patches, and expand the template's pad token into the image's token rows.

use base64::Engine as _;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tr_model::image::{self, ImagePlace, Patches, PrepParams};

/// Largest encoded image accepted (bytes) and largest decoded pixel count.
pub const MAX_IMAGE_BYTES: usize = 32 << 20;
pub const MAX_PIXELS: usize = 64 << 20;

/// Fetch the bytes of an image source: `data:` URI (base64) or an http(s) URL.
pub fn fetch(src: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = src.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or("malformed data URI")?;
        if !meta.ends_with(";base64") {
            return Err("data URI must be base64".into());
        }
        let clean: String = data.chars().filter(|c| !c.is_whitespace()).collect();
        let bytes = base64::engine::general_purpose::STANDARD.decode(clean.as_bytes()).or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(clean.as_bytes())).map_err(|e| format!("data URI base64: {e}"))?;
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(format!("image larger than {} MiB", MAX_IMAGE_BYTES >> 20));
        }
        return Ok(bytes);
    }
    if src.starts_with("http://") || src.starts_with("https://") {
        let mut resp = ureq::get(src).config().timeout_global(Some(std::time::Duration::from_secs(30))).build().call().map_err(|e| format!("fetch {src}: {e}"))?;
        let bytes = resp.body_mut().with_config().limit(MAX_IMAGE_BYTES as u64).read_to_vec().map_err(|e| format!("fetch {src}: {e}"))?;
        return Ok(bytes);
    }
    Err("image_url must be a data: URI or an http(s) URL".into())
}

/// Decode any format the `image` crate knows to RGB8 (`data`, width, height).
pub fn decode(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let img = ::image::load_from_memory(bytes).map_err(|e| format!("decode image: {e}"))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 || w * h > MAX_PIXELS {
        return Err(format!("image {w}x{h} out of range"));
    }
    Ok((img.to_rgb8().into_raw(), w, h))
}

/// Stable content hash of the *preprocessed* input (what the encoder sees), for the embedding cache.
pub fn patches_hash(p: &Patches) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (p.gw, p.gh).hash(&mut h);
    for v in &p.data {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// A preprocessed image ready for the encoder, plus its content hash.
pub struct Prepared {
    pub patches: Arc<Patches>,
    pub hash: u64,
    /// Merged token grid.
    pub nx: usize,
    pub ny: usize,
}

/// Fetch + decode + preprocess.
pub fn prepare(src: &str, prep: &PrepParams) -> Result<Prepared, String> {
    let bytes = fetch(src)?;
    let (rgb, w, h) = decode(&bytes)?;
    let patches = image::prepare(&rgb, w, h, prep);
    let (nx, ny) = patches.tokens(prep.merge);
    let hash = patches_hash(&patches);
    Ok(Prepared { patches: Arc::new(patches), hash, nx, ny })
}

/// Replace each occurrence of `pad` in `ids` (one per image, in order) by `nx*ny` copies and
/// return the expanded ids with the placements. Errors when the counts differ.
pub fn expand_pads(ids: &[u32], pad: u32, imgs: &[Prepared]) -> Result<(Vec<u32>, Vec<ImagePlace>), String> {
    let n_pad = ids.iter().filter(|&&t| t == pad).count();
    if n_pad != imgs.len() {
        return Err(format!("{} image placeholders in the prompt but {} images", n_pad, imgs.len()));
    }
    let mut out = Vec::with_capacity(ids.len() + imgs.iter().map(|i| i.nx * i.ny).sum::<usize>());
    let mut places = Vec::with_capacity(imgs.len());
    let mut k = 0;
    for &t in ids {
        if t == pad {
            let im = &imgs[k];
            places.push(ImagePlace { row: out.len(), nx: im.nx, ny: im.ny });
            out.extend(std::iter::repeat_n(pad, im.nx * im.ny));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok((out, places))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_uri_and_expand() {
        // 2x2 PNG via the image crate's encoder
        let mut png = Vec::new();
        let img = ::image::RgbImage::from_fn(2, 2, |x, y| ::image::Rgb([x as u8 * 100, y as u8 * 100, 7]));
        ::image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut png), ::image::ImageFormat::Png).unwrap();
        let uri = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png));
        let bytes = fetch(&uri).unwrap();
        let (rgb, w, h) = decode(&bytes).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(&rgb[..3], &[0, 0, 7]);
        assert_eq!(&rgb[3..6], &[100, 0, 7]);
        assert!(fetch("ftp://x").is_err());
        let a = Prepared { patches: Arc::new(image::Patches { gw: 2, gh: 2, data: vec![], ypos: vec![], xpos: vec![], width: 32, height: 32 }), hash: 1, nx: 2, ny: 3 };
        let (ids, places) = expand_pads(&[1, 9, 2, 9], 9, &[Prepared { nx: 1, ny: 2, ..a.clone_shallow() }, a]).unwrap();
        assert_eq!(ids, vec![1, 9, 9, 2, 9, 9, 9, 9, 9, 9]);
        assert_eq!(places, vec![ImagePlace { row: 1, nx: 1, ny: 2 }, ImagePlace { row: 4, nx: 2, ny: 3 }]);
        assert!(expand_pads(&[1, 9], 9, &[]).is_err());
    }

    impl Prepared {
        fn clone_shallow(&self) -> Prepared {
            Prepared { patches: self.patches.clone(), hash: self.hash, nx: self.nx, ny: self.ny }
        }
    }
}
