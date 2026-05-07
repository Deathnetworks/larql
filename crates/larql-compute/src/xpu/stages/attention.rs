//! Fused causal attention stage for XPU.
//!
//! XPU equivalent of Metal's `stages::attention`. Dispatches the fused
//! `attn_fused` SYCL kernel which handles RoPE (optional), QK-norm
//! (optional), KV-cache append, scaled dot-product attention with optional
//! sliding window, all in one kernel call.
//!
//! Unlike Metal which encodes via a ComputeCommandEncoderRef, XPU calls
//! the FFI directly.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;
use crate::xpu::ops::kv_cache::LayerKVCache;

/// Control flags for the fused attention dispatch.
#[derive(Clone, Copy, Default)]
pub struct Flags {
    /// Sliding window size. 0 = full causal attention.
    pub window_size: u32,
    /// RMS eps for QK-norm (0.0 = skip QK-norm).
    pub rms_eps: f32,
    /// Additive offset for QK-norm weight (Gemma: +1.0, others: 0.0).
    pub qk_offset: f32,
    /// RoPE base frequency (e.g. 10000.0 or 500000.0).
    pub rope_base: f32,
    /// Number of dimensions to rotate. 0 = full head_dim.
    pub rotary_dim: u32,
}

/// Dispatch fused attention into the KV cache layer.
pub fn encode(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    cache: &mut LayerKVCache,
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    flags: Flags,
) -> Vec<f32> {
    let q_buf  = XpuBuffer::from_slice(q_in, false);
    let k_buf  = XpuBuffer::from_slice(k_in, false);
    let v_buf  = XpuBuffer::from_slice(v_in, false);
    let qw_buf = XpuBuffer::from_slice(q_weight, false);
    let kw_buf = XpuBuffer::from_slice(k_weight, false);
    let out_len = num_q_heads * head_dim;
    let mut out_buf = XpuBuffer::new_device(out_len * 4);

    encode_buf(
        &q_buf, &k_buf, &v_buf,
        &qw_buf, &kw_buf,
        cache, &mut out_buf,
        pos, num_q_heads, num_kv_heads, head_dim, scale, flags,
    );

    let mut out = vec![0.0f32; out_len];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Fused attention from existing buffers.
#[allow(clippy::too_many_arguments)]
pub fn encode_buf(
    q_in: &XpuBuffer,
    k_in: &XpuBuffer,
    v_in: &XpuBuffer,
    q_weight: &XpuBuffer,
    k_weight: &XpuBuffer,
    cache: &mut LayerKVCache,
    out: &mut XpuBuffer,
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    flags: Flags,
) {
    unsafe {
        xpu_ffi::attn_fused(
            q_in.as_ptr_type(),
            k_in.as_ptr_type(),
            v_in.as_ptr_type(),
            cache.k_ptr(),
            cache.v_ptr(),
            out.as_mut_ptr_type(),
            q_weight.as_ptr_type(),
            k_weight.as_ptr_type(),
            pos as u32,
            head_dim as u32,
            num_q_heads as u32,
            num_kv_heads as u32,
            scale,
            flags.window_size,
            flags.rms_eps,
            flags.qk_offset,
            flags.rope_base,
            flags.rotary_dim,
        );
    }
    cache.current_len = (cache.current_len + 1).min(cache.max_seq);
}
