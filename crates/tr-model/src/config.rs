use anyhow::{bail, Result};
use tr_format::Manifest;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layer: usize,
    pub hidden: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    /// Interleaved M-RoPE sections (t, h, w pairs); [rope_dim/2, 0, 0] when the pack has none.
    pub rope_sections: [usize; 3],
    pub rms_eps: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff: usize,
    pub n_ff_shexp: usize,
    pub hc: usize,
    pub hc_lr: usize,
    pub d_conv: usize,
    pub d_state: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub full_interval: usize,
    pub ple_layers: Vec<usize>,
    pub ple_ngram: usize,
    pub ple_per_gram: usize,
    pub ple_conv_kernel: usize,
    pub ple_eos: u32,
    pub ple_dim: usize,
    pub ple_mult: Vec<u64>,
    pub ple_offsets: Vec<u64>,
    pub ple_vocab: Vec<u64>,
    pub eos_id: u32,
    pub bos_id: u32,
    pub n_tiles: usize,
    pub n_expert_packed: usize,
    // QSA indexer (sparse attention over blocks of `compress_ratios[il]` tokens)
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    pub compress_ratios: Vec<usize>,
}

impl ModelConfig {
    pub fn from_manifest(m: &Manifest) -> Result<ModelConfig> {
        if m.arch() != "qwen4exp" {
            bail!("unsupported architecture {}", m.arch());
        }
        let u = |k: &str| -> Result<usize> { m.cfg_u64(k).map(|v| v as usize).ok_or_else(|| anyhow::anyhow!("missing config {k}")) };
        let ple_layers: Vec<usize> = m.cfg_arr_u64("ple.layers").into_iter().map(|v| v as usize).collect();
        let g = |k: &str| m.config.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        Ok(ModelConfig {
            n_layer: u("block_count")?,
            hidden: u("embedding_length")?,
            n_vocab: m.n_vocab,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rope_dim: u("rope.dimension_count")?,
            rope_base: m.cfg_f64("rope.freq_base").unwrap_or(10000.0) as f32,
            rope_sections: {
                let v = m.cfg_arr_u64("rope.dimension_sections");
                if v.len() >= 3 && v[..3].iter().sum::<u64>() > 0 { [v[0] as usize, v[1] as usize, v[2] as usize] } else { [u("rope.dimension_count")? / 2, 0, 0] }
            },
            rms_eps: m.cfg_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff: u("expert_feed_forward_length")?,
            n_ff_shexp: u("expert_shared_feed_forward_length")?,
            hc: u("hyper_connection.count")?,
            hc_lr: u("hyper_connection.low_rank")?,
            d_conv: u("ssm.conv_kernel")?,
            d_state: u("ssm.state_size")?,
            n_k_heads: u("ssm.group_count")?,
            n_v_heads: u("ssm.time_step_rank")?,
            full_interval: m.cfg_u64("full_attention_interval").unwrap_or(4) as usize,
            ple_ngram: m.cfg_u64("ple.ngram_size").unwrap_or(0) as usize,
            ple_per_gram: m.cfg_u64("ple.heads_per_ngram").unwrap_or(0) as usize,
            ple_conv_kernel: m.cfg_u64("ple.conv_kernel").unwrap_or(0) as usize,
            ple_eos: m.cfg_u64("ple.eos_token_id").unwrap_or(0) as u32,
            ple_dim: m.cfg_u64("embedding_length_per_layer_input").unwrap_or(0) as usize,
            ple_mult: m.cfg_arr_u64("ple.layer_multipliers"),
            ple_offsets: m.cfg_arr_u64("ple.head_offsets"),
            ple_vocab: m.cfg_arr_u64("ple.head_vocab_sizes"),
            ple_layers,
            eos_id: g("tokenizer.ggml.eos_token_id"),
            bos_id: g("tokenizer.ggml.bos_token_id"),
            n_tiles: m.n_tiles,
            n_expert_packed: m.n_expert_packed,
            idx_heads: m.cfg_u64("attention.indexer.head_count").unwrap_or(0) as usize,
            idx_dim: m.cfg_u64("attention.indexer.key_length").unwrap_or(0) as usize,
            idx_top_k: m.cfg_u64("attention.indexer.top_k").unwrap_or(0) as usize,
            compress_ratios: m.cfg_arr_u64("attention.compress_ratios").into_iter().map(|v| v as usize).collect(),
        })
    }
    pub fn is_recurrent(&self, il: usize) -> bool {
        (il + 1) % self.full_interval != 0
    }
    /// QSA block size for layer `il` (0 = dense attention). The MTP draft layer (`mtp_layer()`)
    /// is a full-attention layer and uses the ratio of the main attention layers.
    pub fn qsa_ratio(&self, il: usize) -> usize {
        if self.idx_dim == 0 || self.idx_top_k == 0 {
            0
        } else if il == self.mtp_layer() {
            (0..self.n_layer).filter(|&l| !self.is_recurrent(l)).map(|l| self.compress_ratios.get(l).copied().unwrap_or(0)).find(|&r| r > 0).unwrap_or(0)
        } else {
            self.compress_ratios.get(il).copied().unwrap_or(0)
        }
    }
    /// Layer index the MTP draft head is packed under (`blk.{n_layer}.*`).
    pub fn mtp_layer(&self) -> usize {
        self.n_layer
    }
    /// Upper bound on indexer blocks per attention layer for a context of `ctx_max` tokens.
    pub fn qsa_blocks_max(&self, ctx_max: usize) -> usize {
        let r = (0..self.n_layer).map(|il| self.qsa_ratio(il)).filter(|&r| r > 0).min().unwrap_or(0);
        if r == 0 { 0 } else { ctx_max / r + 1 }
    }
    pub fn ple_n_heads(&self) -> usize {
        if self.ple_ngram == 0 { 0 } else { (self.ple_ngram - 1) * self.ple_per_gram }
    }
}
