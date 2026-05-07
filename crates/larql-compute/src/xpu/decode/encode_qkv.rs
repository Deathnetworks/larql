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
        let mut h_in_f32 = vec![0.0f32; dims.hidden];
        bufs.h_in.copy_to_slice(&mut h_in_f32);
        
        let mut input_norm_f32 = vec![0.0f32; dims.hidden];
        bufs.input_norm.copy_to_slice(&mut input_norm_f32);

        if uses_q4k {
            crate::xpu::stages::input_norm::encode_f32(
                &h_in_f32,
                &input_norm_f32,
                bufs.norm_out,
                dims.hidden,
                dims.eps,
                dims.norm_offset,
            );

            let mut norm_f32 = vec![0.0f32; dims.hidden];
            bufs.norm_out.copy_to_slice(&mut norm_f32);

            let mut wq_bytes = vec![0u8; layer.wq.data.len()];
            bufs.wq.copy_to_slice(&mut wq_bytes);
            let mut wk_bytes = vec![0u8; layer.wk.data.len()];
            bufs.wk.copy_to_slice(&mut wk_bytes);
            let mut wv_bytes = vec![0u8; layer.wv.data.len()];
            bufs.wv.copy_to_slice(&mut wv_bytes);

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
                    dims.layer_q_dim, dims.layer_kv_dim, dims.hidden,
                );
                bufs.q_out.copy_from_slice(&q);
                bufs.k_out.copy_from_slice(&k);
                bufs.v_out.copy_from_slice(&v);
            } else if mixed_q4k_q6k_v {
                let (q, k, v) = crate::xpu::stages::qkv_proj::encode_fused_q4k_q6k(
                    &wq_bytes, &wk_bytes, &wv_bytes,
                    &norm_f32,
                    dims.layer_q_dim, dims.layer_kv_dim, dims.layer_kv_dim, dims.hidden,
                );
                bufs.q_out.copy_from_slice(&q);
                bufs.k_out.copy_from_slice(&k);
                bufs.v_out.copy_from_slice(&v);
            } else {
                let (q, k, v) = crate::xpu::stages::qkv_proj::encode_per_proj(
                    &wq_bytes, &wk_bytes, &wv_bytes,
                    &norm_f32,
                    dims.layer_q_dim, dims.layer_kv_dim, dims.hidden,
                );
                bufs.q_out.copy_from_slice(&q);
                bufs.k_out.copy_from_slice(&k);
                bufs.v_out.copy_from_slice(&v);
            }
        } else {
            crate::xpu::stages::input_norm::encode_q8(
                &h_in_f32,
                &input_norm_f32,
                bufs.ffn_q8,
                bufs.ffn_q8s,
                dims.hidden,
                dims.eps,
                dims.norm_offset,
            );
            
            let mut wq_bytes = vec![0u8; layer.wq.data.len()];
            bufs.wq.copy_to_slice(&mut wq_bytes);
            let mut wk_bytes = vec![0u8; layer.wk.data.len()];
            bufs.wk.copy_to_slice(&mut wk_bytes);
            let mut wv_bytes = vec![0u8; layer.wv.data.len()];
            bufs.wv.copy_to_slice(&mut wv_bytes);

            let mut norm_f32 = vec![0.0f32; dims.hidden];
            crate::xpu::stages::input_norm::encode_f32(
                &h_in_f32,
                &input_norm_f32,
                bufs.norm_out,
                dims.hidden,
                dims.eps,
                dims.norm_offset,
            );
            bufs.norm_out.copy_to_slice(&mut norm_f32);
            
            let (q, k, v) = crate::xpu::stages::qkv_proj::encode_per_proj(
                &wq_bytes, &wk_bytes, &wv_bytes,
                &norm_f32,
                dims.layer_q_dim, dims.layer_kv_dim, dims.hidden,
            );
            bufs.q_out.copy_from_slice(&q);
            bufs.k_out.copy_from_slice(&k);
            bufs.v_out.copy_from_slice(&v);
        }
    }
}
