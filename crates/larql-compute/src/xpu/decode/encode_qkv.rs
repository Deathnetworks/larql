//! Step 1 of the decode pipeline: input norm + fused Q/K/V projection.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;

pub(super) struct QkvBufs<'a> {
    pub h_in: &'a XpuBuffer,
    pub input_norm: &'a XpuBuffer,
    pub input_norm_bias: Option<&'a [f32]>,
    pub wq: &'a XpuBuffer,
    pub wk: &'a XpuBuffer,
    pub wv: &'a XpuBuffer,
    pub wq_scales: &'a XpuBuffer,
    pub wk_scales: &'a XpuBuffer,
    pub wv_scales: &'a XpuBuffer,
    pub norm_out: &'a mut XpuBuffer,
    pub q_out: &'a mut XpuBuffer,
    pub k_out: &'a mut XpuBuffer,
    pub v_out: &'a mut XpuBuffer,
    pub ffn_q8: &'a mut XpuBuffer,
    pub ffn_q8s: &'a mut XpuBuffer,
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
        mut bufs: QkvBufs<'_>,
        dims: QkvDims,
        uses_q4k: bool,
    ) {
        if uses_q4k {
            // Step 1: RMS Norm (f32 output)
            crate::xpu::stages::input_norm::encode_f32_buf(
                bufs.h_in,
                bufs.input_norm,
                bufs.norm_out,
                dims.hidden,
                dims.eps,
                dims.norm_offset,
            );

            // Step 2: QKV Projection
            let uniform_q4k = layer.wq.format == layer.wk.format
                && layer.wk.format == layer.wv.format
                && layer.wq.format != crate::QuantFormat::Q6_K;
            let mixed_q4k_q6k_v = layer.wq.format == crate::QuantFormat::Q4_K
                && layer.wk.format == crate::QuantFormat::Q4_K
                && layer.wv.format == crate::QuantFormat::Q6_K;

            if uniform_q4k {
                crate::xpu::stages::qkv_proj::encode_fused_f32_buf(
                    bufs.wq, bufs.wk, bufs.wv,
                    bufs.norm_out,
                    bufs.q_out, bufs.k_out, bufs.v_out,
                    dims.layer_q_dim, dims.layer_kv_dim, dims.hidden,
                );
            } else if mixed_q4k_q6k_v {
                crate::xpu::stages::qkv_proj::encode_fused_q4k_q6k_buf(
                    bufs.wq, bufs.wk, bufs.wv,
                    bufs.norm_out,
                    bufs.q_out, bufs.k_out, bufs.v_out,
                    dims.layer_q_dim, dims.layer_kv_dim, dims.layer_kv_dim, dims.hidden,
                );
            } else {
                crate::xpu::stages::qkv_proj::encode_per_proj_buf(
                    bufs.wq, bufs.wk, bufs.wv,
                    bufs.norm_out,
                    bufs.q_out, bufs.k_out, bufs.v_out,
                    dims.layer_q_dim, dims.layer_kv_dim, dims.hidden,
                );
            }
        } else {
            // Step 1: RMS Norm + Q8 Quantize (for FFN path)
            crate::xpu::stages::input_norm::encode_q8_buf(
                bufs.h_in,
                bufs.input_norm,
                bufs.ffn_q8,
                bufs.ffn_q8s,
                dims.hidden,
                dims.eps,
                dims.norm_offset,
            );
            
            // Step 2: Parallel RMS Norm (f32 output for Attention path)
            crate::xpu::stages::input_norm::encode_f32_buf(
                bufs.h_in,
                bufs.input_norm,
                bufs.norm_out,
                dims.hidden,
                dims.eps,
                dims.norm_offset,
            );
            
            // Step 3: Per-projection QKV
            crate::xpu::stages::qkv_proj::encode_per_proj_buf(
                bufs.wq, bufs.wk, bufs.wv,
                bufs.norm_out,
                bufs.q_out, bufs.k_out, bufs.v_out,
                dims.layer_q_dim, dims.layer_kv_dim, dims.hidden,
            );
        }
    }
}
