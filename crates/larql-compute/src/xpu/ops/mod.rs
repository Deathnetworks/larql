//! XPU operation dispatch.

pub mod full_layer;
pub mod full_pipeline;
pub mod kv_cache;
pub mod q4_batched;
pub mod q4_common;
pub mod q4_f32_matvec;
pub mod q4_matvec;
pub mod q4_vecmat;
pub mod rms_norm;
