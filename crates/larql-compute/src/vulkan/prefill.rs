//! Vulkan prefill pipeline: full Q4 inference for seq>1 with KV cache population.

use super::*;

/// Run the prefill pipeline on Vulkan.
pub fn dispatch_prefill(
    _backend: &VulkanBackend,
    _layers: &[FullPipelineLayer],
    _x: &[f32],
    _hidden: usize,
    _inter: usize,
    _q_dim: usize,
    _kv_dim: usize,
    _seq_len: usize,
    _use_qk_norm: bool,
    _softcap: f32,
) -> Vec<f32> {
    unimplemented!("Vulkan prefill is not yet ported from Metal")
}
