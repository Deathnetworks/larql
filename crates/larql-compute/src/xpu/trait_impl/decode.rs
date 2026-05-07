use super::*;
use crate::backend::DecodeBackend;

impl DecodeBackend for XpuBackend {
    fn has_kv_cache(&self) -> bool {
        true
    }

    fn kv_cache_len(&self) -> usize {
        let kv = self.kv_cache.lock().unwrap();
        kv.as_ref().map(|k| k.current_len()).unwrap_or(0)
    }

    fn reset_kv_cache(&self) {
        let mut kv = self.kv_cache.lock().unwrap();
        if let Some(kv) = kv.as_mut() {
            kv.clear();
        }
    }

    fn preallocate_kv_cache_per_layer(&self, shapes: &[(usize, usize)], max_seq: usize) {
        let mut kv = self.kv_cache.lock().unwrap();
        self.ensure_kv_cache_for_shapes(&mut kv, shapes, max_seq);
    }

    fn decode_token(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        let mut kv_lock = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(&mut kv_lock, layers, DEFAULT_KV_CACHE_MAX_SEQ);
        
        Some(self.decode_token_with_moe_split_fn(
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            None,
            None,
        ))
    }

    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        let mut kv_lock = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(&mut kv_lock, layers, DEFAULT_KV_CACHE_MAX_SEQ);
        
        Some(self.decode_token_with_moe_split_fn(
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            Some(moe_fn),
            None,
        ))
    }
}
