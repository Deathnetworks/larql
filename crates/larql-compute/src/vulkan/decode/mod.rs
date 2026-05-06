//! Vulkan decode pipeline orchestrator.
//! TODO: Port from Metal decode/mod.rs (5 fns, 28,618 bytes)

use super::VulkanBackend;

// Submodules — all currently stubbed, see task.md Milestone 8
mod diag;
mod encode_attn;
mod encode_ffn;
mod encode_post_ffn;
mod encode_qkv;
pub mod gpu_timing;
mod moe_combine;
mod moe_interleave;
pub mod profile;
mod setup;

pub use profile::ProfileTimings;
