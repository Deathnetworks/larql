//! Hybrid decode for XPU — GPU attention only, returns hidden state for CPU FFN.

use super::*;

impl XpuBackend {
    pub fn decode_attention_layer(
        &self,
        _kv_cache: &mut (), // Placeholder
        _layer: &crate::FullPipelineLayer,
        _layer_idx: usize,
        _x: &[f32],
        _hidden: usize,
        _q_dim: usize,
        _kv_dim: usize,
    ) -> Vec<f32> {
        unimplemented!("XPU decode_attention_layer is not yet ported from Metal")
    }
}
