use super::*;

impl XpuBackend {
    pub fn decode_token(
        &self,
        _kv_cache: &mut (), // Placeholder
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
    ) -> Vec<f32> {
        unimplemented!("XPU decode_token is not yet ported from Metal")
    }
}
