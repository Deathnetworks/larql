//! Step 1 of the decode pipeline: input norm + fused Q/K/V projection.
//!
//! XPU equivalent of Metal's `decode::encode_qkv`. We route to the
//! functions in `crate::xpu::stages::input_norm` and `crate::xpu::stages::qkv_proj`.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;

/// Buffer references the QKV step reads or writes.
pub(super) struct QkvBufs<'a> {
    // Input
    pub h_in: &'a XpuBuffer,
    // Per-layer weights + scales
    pub input_norm: &'a XpuBuffer,
    pub input_norm_bias: Option<&'a [f32]>,
    pub wq: &'a XpuBuffer,
    pub wk: &'a XpuBuffer,
    pub wv: &'a XpuBuffer,
    pub wq_scales: &'a XpuBuffer,
    pub wk_scales: &'a XpuBuffer,
    pub wv_scales: &'a XpuBuffer,
    // Outputs
    pub norm_out: &'a XpuBuffer,
    pub q_out: &'a XpuBuffer,
    pub k_out: &'a XpuBuffer,
    pub v_out: &'a XpuBuffer,
    // Scratch
    pub ffn_q8: &'a XpuBuffer,
    pub ffn_q8s: &'a XpuBuffer,
}

#[derive(Copy, Clone)]
pub(super) struct QkvDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    pub layer_kv_dim: usize,
    pub eps: f32,
    pub norm_offset: f32,
}

impl XpuBackend {
    pub(super) fn encode_input_norm_and_qkv(
        &self,
        layer: &FullPipelineLayer,
        bufs: QkvBufs<'_>,
        dims: QkvDims,
        uses_q4k: bool,
    ) {
        let QkvDims {
            hidden,
            layer_q_dim,
            layer_kv_dim,
            eps,
            norm_offset,
        } = dims;

        // XPU stages expect raw slices for some operations (based on current signatures).
        // Since XpuBuffer has a copy_to_slice method, we'll download to host temporarily
        // to interface with the current stage APIs. (Optimization pass will lift USM pointers).
        
        let mut h_in_f32 = vec![0.0f32; hidden];
        bufs.h_in.copy_to_slice(&mut h_in_f32);
        
        let mut input_norm_f32 = vec![0.0f32; hidden];
        bufs.input_norm.copy_to_slice(&mut input_norm_f32);

        if uses_q4k {
            // 1. Input Norm -> f32
            crate::xpu::stages::input_norm::encode_f32(
                &h_in_f32,
                &input_norm_f32,
                bufs.norm_out, // XpuBuffer
                hidden,
                eps,
                norm_offset,
            );

            // Fetch normed output to host for the QKV step
            let mut norm_f32 = vec![0.0f32; hidden];
            bufs.norm_out.copy_to_slice(&mut norm_f32);

            // Fetch weights
            let mut wq_bytes = vec![0u8; layer.wq.data_len()];
            bufs.wq.copy_to_slice(&mut wq_bytes);
            let mut wk_bytes = vec![0u8; layer.wk.data_len()];
            bufs.wk.copy_to_slice(&mut wk_bytes);
            let mut wv_bytes = vec![0u8; layer.wv.data_len()];
            bufs.wv.copy_to_slice(&mut wv_bytes);

            // 2. QKV Projection
            let uniform_q4k = layer.wq.format == layer.wk.format
                && layer.wk.format == layer.wv.format
                && layer.wq.format != crate::QuantFormat::Q6_K;
            let mixed_q4k_q6k_v = layer.wq.format == crate::QuantFormat::Q4_K
                && layer.wk.format == crate::QuantFormat::Q4_K
                && layer.wv.format == crate::QuantFormat::Q6_K;

            if uniform_q4k {
                let (q, k, v) = crate::xpu::stages::qkv_proj::encode_fused_f32(
                    &wq_bytes, &wk_bytes, &wv_bytes,
                    &norm_f32,
                    layer_q_dim, layer_kv_dim, hidden,
                );
                bufs.q_out.copy_from_slice(&q);
                bufs.k_out.copy_from_slice(&k);
                bufs.v_out.copy_from_slice(&v);
            } else if mixed_q4k_q6k_v {
                let (q, k, v) = crate::xpu::stages::qkv_proj::encode_fused_q4k_q6k(
                    &wq_bytes, &wk_bytes, &wv_bytes,
                    &norm_f32,
                    layer_q_dim, layer_kv_dim, layer_kv_dim, hidden,
                );
                bufs.q_out.copy_from_slice(&q);
                bufs.k_out.copy_from_slice(&k);
                bufs.v_out.copy_from_slice(&v);
            } else {
                let (q, k, v) = crate::xpu::stages::qkv_proj::encode_per_proj(
                    &wq_bytes, &wk_bytes, &wv_bytes,
                    &norm_f32,
                    layer_q_dim, layer_kv_dim, hidden,
                );
                bufs.q_out.copy_from_slice(&q);
                bufs.k_out.copy_from_slice(&k);
                bufs.v_out.copy_from_slice(&v);
            }
        } else {
            // Q4_0 path: encode_q8
            crate::xpu::stages::input_norm::encode_q8(
                &h_in_f32,
                &input_norm_f32,
                bufs.ffn_q8,
                bufs.ffn_q8s,
                hidden,
                eps,
                norm_offset,
            );
            
            // Q4_0 fallback (per-proj for now as XPU legacy path isn't fused)
            let mut wq_bytes = vec![0u8; layer.wq.data_len()];
            bufs.wq.copy_to_slice(&mut wq_bytes);
            let mut wk_bytes = vec![0u8; layer.wk.data_len()];
            bufs.wk.copy_to_slice(&mut wk_bytes);
            let mut wv_bytes = vec![0u8; layer.wv.data_len()];
            bufs.wv.copy_to_slice(&mut wv_bytes);

            let mut norm_f32 = vec![0.0f32; hidden];
            // Recover f32 norm for q4_vecmat (since XPU Q8 QKV proj isn't fully implemented)
            // Just doing a per-proj fallback to keep compilation clean.
            crate::xpu::stages::input_norm::encode_f32(
                &h_in_f32,
                &input_norm_f32,
                bufs.norm_out,
                hidden,
                eps,
                norm_offset,
            );
            bufs.norm_out.copy_to_slice(&mut norm_f32);
            
            let (q, k, v) = crate::xpu::stages::qkv_proj::encode_per_proj(
                &wq_bytes, &wk_bytes, &wv_bytes,
                &norm_f32,
                layer_q_dim, layer_kv_dim, hidden,
            );
            bufs.q_out.copy_from_slice(&q);
            bufs.k_out.copy_from_slice(&k);
            bufs.v_out.copy_from_slice(&v);
        }
    }
}
