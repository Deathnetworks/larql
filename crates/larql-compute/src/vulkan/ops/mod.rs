//! Vulkan operation dispatch — one file per operation type.
//!
//! Replicates the modular structure of the Metal backend for consistency.
//! Each module handles dispatch for one category of compute operation.

pub mod f32_gemv;
pub mod q8_quantize;
pub mod q4k_matvec;
pub mod q6k_matvec;
pub mod attn_fused;
pub mod rms_norm;
pub mod silu;
pub mod rope;
pub mod q4k_ffn_gate_up;
pub mod q4k_qkv_proj;
pub mod q4_vecmat;
pub mod turboquant;
