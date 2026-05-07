//! Full attention layer orchestration for XPU.
//!
//! Wraps the `attn_fused` SYCL kernel which performs:
//!   1. QKV projection (from pre-projected q/k/v inputs)
//!   2. RoPE at current position
//!   3. KV cache append
//!   4. Scaled dot-product attention with optional sliding window
//!   5. Output projection
//!
//! This single fused dispatch replaces the multi-kernel sequence
//! used in the Metal backend (encode_kv_append + encode_kv_attend).

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;
use crate::xpu::ops::kv_cache::LayerKVCache;

/// Parameters for a single fused attention layer dispatch.
pub struct AttnFusedParams {
    pub head_dim: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub scale: f32,
    /// Sliding window size (0 = full attention / no window).
    pub window_size: u32,
    pub rms_eps: f32,
    pub qk_offset: f32,
    pub rope_base: f32,
    pub rotary_dim: u32,
}

/// Dispatch a full fused attention layer on XPU.
///
/// - `q_in / k_in / v_in`: pre-projected Q/K/V activations `[num_heads * head_dim]`
/// - `q_weight / k_weight`: Q/K norm weights for QK-norm (or zeros if unused)
/// - `cache`: mutable KV cache for this layer (updated in-place by kernel)
/// - `pos`: current token position in sequence
/// - `params`: attention geometry and hyper-parameters
/// - Returns: attention output `[num_q_heads * head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn attn_fused_dispatch(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    cache: &mut LayerKVCache,
    pos: usize,
    params: &AttnFusedParams,
) -> Vec<f32> {
    let out_len = (params.num_q_heads * params.head_dim) as usize;
    let mut out = vec![0.0f32; out_len];

    let q_buf = XpuBuffer::from_slice(q_in, false);
    let k_buf = XpuBuffer::from_slice(k_in, false);
    let v_buf = XpuBuffer::from_slice(v_in, false);
    let qw_buf = XpuBuffer::from_slice(q_weight, false);
    let kw_buf = XpuBuffer::from_slice(k_weight, false);
    let mut out_buf = XpuBuffer::new_device(out_len * std::mem::size_of::<f32>());

    unsafe {
        xpu_ffi::attn_fused(
            q_buf.as_ptr_type(),
            k_buf.as_ptr_type(),
            v_buf.as_ptr_type(),
            cache.k_ptr(),
            cache.v_ptr(),
            out_buf.as_mut_ptr_type(),
            qw_buf.as_ptr_type(),
            kw_buf.as_ptr_type(),
            pos as u32,
            params.head_dim,
            params.num_q_heads,
            params.num_kv_heads,
            params.scale,
            params.window_size,
            params.rms_eps,
            params.qk_offset,
            params.rope_base,
            params.rotary_dim,
        );
    }

    // Advance KV cache position after kernel writes new K/V at `pos`.
    cache.current_len = (cache.current_len + 1).min(cache.max_seq);

    out_buf.copy_to_slice(&mut out);
    out
}
