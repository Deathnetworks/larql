//! Full pipeline orchestrator for Vulkan.

pub mod dispatch;

pub use dispatch::{dispatch_full_pipeline, encode_residual_add, encode_rms_norm};
