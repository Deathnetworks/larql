//! KV cache management for XPU (SYCL backend).
//!
//! Per-layer USM buffers for cached K/V vectors. Mirrors the Metal
//! `kv_cache.rs` structure but uses `XpuBuffer` (SYCL USM) instead of
//! Metal `Buffer`. The `attn_fused` SYCL kernel reads `k_cache`/`v_cache`
//! directly via their raw USM pointers.

use crate::xpu::buffers::XpuBuffer;

/// KV cache for one transformer layer — pre-allocated SYCL USM buffers.
pub struct LayerKVCache {
    /// K cache: [max_seq × num_kv_heads × head_dim] f32 in device memory.
    pub k_cache: XpuBuffer,
    /// V cache: same shape as k_cache.
    pub v_cache: XpuBuffer,
    pub current_len: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKVCache {
    /// Allocate device-side K/V buffers for one layer.
    pub fn new(max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = max_seq * num_kv_heads * head_dim * std::mem::size_of::<f32>();
        Self {
            k_cache: XpuBuffer::new_device(size),
            v_cache: XpuBuffer::new_device(size),
            current_len: 0,
            max_seq,
            num_kv_heads,
            head_dim,
        }
    }

    /// Reset sequence position (for a new prompt).
    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    /// Raw pointer to K-cache device memory (passed to `attn_fused`).
    pub fn k_ptr(&mut self) -> *mut f32 {
        self.k_cache.as_mut_ptr_type()
    }

    /// Raw pointer to V-cache device memory (passed to `attn_fused`).
    pub fn v_ptr(&mut self) -> *mut f32 {
        self.v_cache.as_mut_ptr_type()
    }
}

/// Full KV cache for all transformer layers.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    /// Allocate a uniform KV cache (all layers share the same dims).
    /// Corresponds to Llama / Mistral / Gemma 3 layout.
    pub fn new(num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(max_seq, num_kv_heads, head_dim))
            .collect();
        Self { layers }
    }

    /// Allocate with per-layer shapes — required for models like Gemma 4 31B
    /// that alternate sliding-window (e.g. 16 kv heads) with global (4 kv heads) layers.
    ///
    /// `shapes[i]` is `(num_kv_heads_i, head_dim_i)` for layer `i`.
    pub fn new_per_layer(shapes: &[(usize, usize)], max_seq: usize) -> Self {
        let layers = shapes
            .iter()
            .map(|&(num_kv, hd)| LayerKVCache::new(max_seq, num_kv, hd))
            .collect();
        Self { layers }
    }

    /// Reset all layers (new prompt).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Current sequence length (reads from first layer).
    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}
