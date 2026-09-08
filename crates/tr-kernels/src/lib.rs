//! tr-kernels: CPU kernels for the qwen4exp graph. Every SIMD kernel has a scalar reference in
//! `reference` used by the tests. All SIMD code assumes Sapphire Rapids (AVX-512 F/BW/VL/VNNI/BF16).
pub mod quant;
pub mod gemv;
pub mod reference;
pub mod elem;
pub mod qsa;
pub mod rope;
pub mod attn;
pub mod conv;
pub mod gdn;
pub mod router;
pub mod iq4nl;
pub mod ple;
pub mod amx;
pub mod smallgemm;
pub mod amx_i8;
