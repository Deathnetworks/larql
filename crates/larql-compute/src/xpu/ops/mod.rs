//! XPU (SYCL) operation dispatch — one file per operation type.
//!
//! Replicates the modular structure of the Metal backend for consistency.
//! Each module handles dispatch for one category of compute operation.

pub mod q4_matvec;
pub mod q4_vecmat;
pub mod q4k_matvec;
pub mod q6k_matvec;
pub mod q8_quantize;
pub mod attn_fused;
pub mod f32_gemv;
pub mod rms_norm;
pub mod silu;
pub mod gelu_tanh;
pub mod rope;
pub mod turboquant;
