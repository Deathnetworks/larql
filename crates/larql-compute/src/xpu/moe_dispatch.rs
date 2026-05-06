//! XPU expert dispatch for per-layer Q4_K MoE models.

use super::*;

pub struct MoeScratch {
    // Scratch buffers and metadata for MoE dispatch
}

impl XpuBackend {
    pub fn decode_token_q4k_moe<F>(
        &self,
        _layers: &[FullPipelineLayer],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        _norm_eps: f32,
        _get_expert: F,
    ) -> Option<Vec<f32>>
    where
        F: Fn(usize, usize) -> Option<(&[u8], &[u8])>,
    {
        unimplemented!("XPU moe_dispatch is not yet ported from Metal")
    }
}
