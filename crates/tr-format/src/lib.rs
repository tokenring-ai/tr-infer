//! Pack v2 format: manifest schema, TQ codec constants and offset math, reference dequant.
//! The Python packer (`python/trpack`) is the writer; this crate is the reader contract.
pub mod codec;
pub mod manifest;
pub use manifest::{Manifest, Shard, TensorEntry, TensorKind};
