//! tr-model: the qwen4exp graph on top of tr-sys (tiles, pool) and tr-kernels.
pub mod config;
pub mod weights;
pub mod exchange;
pub mod state;
pub mod exec;
pub mod exec_batch;
pub mod exec_mtp;
pub mod image;
pub mod vision;
pub mod sampler;
pub mod snapshot;
pub mod spec;
pub mod tokenizer;

pub mod sequence;
