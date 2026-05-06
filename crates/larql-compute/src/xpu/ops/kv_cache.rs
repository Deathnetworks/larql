//! KV cache management for XPU.

use super::*;

pub struct LayerKVCache {
    // XPU buffers for K/V
    pub current_len: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(_num_layers: usize, _max_seq: usize, _num_kv_heads: usize, _head_dim: usize) -> Self {
        unimplemented!("XPU KVCache is not yet ported from Metal")
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.current_len = 0;
        }
    }
}
