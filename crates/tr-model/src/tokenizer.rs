use anyhow::{anyhow, Result};
use std::path::Path;

pub struct Tok {
    inner: tokenizers::Tokenizer,
}

impl Tok {
    pub fn load(path: &Path) -> Result<Tok> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(Tok { inner })
    }
    pub fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, special).map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
    /// `encode` plus the byte range of `text` each token came from (added tokens such as
    /// `<|im_start|>` are single tokens with exact ranges).
    pub fn encode_with_offsets(&self, text: &str, special: bool) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let enc = self.inner.encode(text, special).map_err(|e| anyhow!("encode: {e}"))?;
        Ok((enc.get_ids().to_vec(), enc.get_offsets().to_vec()))
    }
    /// Id of an added/special token such as `<|image_pad|>`.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.inner.token_to_id(s)
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
    }
}
