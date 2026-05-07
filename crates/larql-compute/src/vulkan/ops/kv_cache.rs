//! KV cache management for Vulkan.

use crate::vulkan::VulkanBackend;
use crate::vulkan::buffers::VulkanBuffer;
use vulkano::buffer::BufferUsage;
use std::sync::Arc;

pub const SHORT_ATTENTION_SPAN: u32 = 1024;

pub fn attention_span(t: u32, window_size: u32) -> u32 {
    if window_size > 0 && t > window_size {
        window_size
    } else {
        t
    }
}

/// KV cache for one layer — pre-allocated Vulkan buffers.
pub struct LayerKVCache {
    pub k_cache: VulkanBuffer,
    pub v_cache: VulkanBuffer,
    pub current_len: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKVCache {
    pub fn new(backend: &VulkanBackend, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = max_seq * num_kv_heads * head_dim * 4;
        let k_cache = VulkanBuffer::new(backend, size, BufferUsage::STORAGE_BUFFER).expect("Failed to allocate K cache");
        let v_cache = VulkanBuffer::new(backend, size, BufferUsage::STORAGE_BUFFER).expect("Failed to allocate V cache");
        Self {
            k_cache,
            v_cache,
            current_len: 0,
            max_seq,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(backend: &VulkanBackend, num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(backend, max_seq, num_kv_heads, head_dim))
            .collect();
        Self { layers }
    }

    pub fn new_per_layer(backend: &VulkanBackend, shapes: &[(usize, usize)], max_seq: usize) -> Self {
        let layers = shapes
            .iter()
            .map(|&(num_kv, hd)| LayerKVCache::new(backend, max_seq, num_kv, hd))
            .collect();
        Self { layers }
    }

    pub fn has_shape_mismatch(&self, shapes: &[(usize, usize)]) -> bool {
        if self.layers.len() != shapes.len() {
            return true;
        }
        for (layer, &(expected_num_kv, expected_hd)) in self.layers.iter().zip(shapes.iter()) {
            if layer.num_kv_heads != expected_num_kv || layer.head_dim != expected_hd {
                return true;
            }
        }
        false
    }

    pub fn grow_to_shapes(&mut self, backend: &VulkanBackend, shapes: &[(usize, usize)], max_seq: usize) {
        while self.layers.len() < shapes.len() {
            let (num_kv, hd) = shapes[self.layers.len()];
            self.layers.push(LayerKVCache::new(backend, max_seq, num_kv, hd));
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}
