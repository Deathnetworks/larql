//! Full pipeline orchestrator for Vulkan.

mod buffers;
mod dispatch;
mod dump;
mod kv_copy;
mod stages;

pub use dispatch::{dispatch_full_pipeline, encode_residual_add, encode_rms_norm};
