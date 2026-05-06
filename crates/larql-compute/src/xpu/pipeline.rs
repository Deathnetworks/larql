use super::*;

impl XpuBackend {
    /// Full pipeline: attention + FFN for all layers.
    pub fn full_pipeline(
        &self,
        _layers: &[FullPipelineLayer],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
    ) -> Vec<f32> {
        unimplemented!("XPU full_pipeline is not yet ported from Metal")
    }

    /// Multi-layer Q4 FFN.
    pub fn multi_layer_q4_ffn(
        &self,
        _layers_q4: &[(&[u8], &[u8], &[u8])],
        _x: &[f32],
        _inter: usize,
        _hidden: usize,
    ) -> Vec<f32> {
        unimplemented!("XPU multi_layer_q4_ffn is not yet ported from Metal")
    }
}
