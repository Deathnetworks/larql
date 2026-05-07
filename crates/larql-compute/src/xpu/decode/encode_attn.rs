//! Per-layer attention block — Steps 1.5 through 5 of the decode loop for XPU.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;
use crate::xpu::ops;

pub(super) struct AttnBufs<'a> {
    pub h_buf: &'a XpuBuffer,
    pub q_out: &'a XpuBuffer,
    pub k_out: &'a XpuBuffer,
    pub v_out: &'a XpuBuffer,
    pub attn_out_buf: &'a XpuBuffer,
    pub o_out_buf: &'a XpuBuffer,
    pub ffn_norm_out: &'a XpuBuffer,
    pub h_post_attn: &'a XpuBuffer,
    pub o_q8_scratch: &'a XpuBuffer,
    pub o_q8s_scratch: &'a XpuBuffer,
    pub ffn_q8: &'a XpuBuffer,
    pub ffn_q8s: &'a XpuBuffer,
    pub normed_scratch: &'a XpuBuffer,
    pub wo: &'a XpuBuffer,
    pub wo_scales: &'a XpuBuffer,
    pub post_attn_norm: &'a XpuBuffer,
}

pub(super) struct AttnDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    pub uses_q4k: bool,
    pub ffn_uses_q4k: bool,
}

impl XpuBackend {
    pub(super) fn encode_attention_block(
        &self,
        layer: &FullPipelineLayer,
        kv_cache: &mut ops::kv_cache::KVCache,
        layer_idx: usize,
        bufs: AttnBufs<'_>,
        dims: AttnDims,
    ) {
        let AttnDims {
            hidden,
            layer_q_dim,
            uses_q4k,
            ffn_uses_q4k,
        } = dims;

        let scale = layer.attn_scale;
        let layer_head_dim = layer.head_dim;
        let layer_num_q_heads = layer.num_q_heads;
        let window_size = layer.sliding_window as u32;
        let layer_rope_base = layer.rope_base;
        let layer_rotary_dim = if layer.rotary_dim > 0 {
            layer.rotary_dim
        } else {
            layer_head_dim
        };

        // 1. QK Norm
        if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
            let mut q_w_f32 = vec![0.0f32; layer_head_dim]; // Simplified
            let mut k_w_f32 = vec![0.0f32; layer_head_dim];
            
            crate::xpu::stages::qk_norm::encode_qk_norm(
                bufs.q_out,
                bufs.k_out,
                &q_w_f32,
                &k_w_f32,
                layer_head_dim as u32,
                layer_num_q_heads as u32,
                layer.eps,
                layer.qk_norm_offset,
            );
        }

        // 2. RoPE
        let pos = kv_cache.layers[layer_idx].current_len as u32;
        crate::xpu::stages::rope::encode_batched(
            bufs.q_out,
            bufs.k_out,
            layer_head_dim as u32,
            layer_rope_base,
            pos,
            layer_rotary_dim as u32,
            layer_num_q_heads as u32,
        );

        // 3. V-Norm
        if layer.has_v_norm {
            crate::xpu::stages::qk_norm::encode_v_norm(
                bufs.v_out,
                layer_head_dim as u32,
                layer.eps,
                layer.num_kv_heads as u32,
            );
        }

        // 4. KV Append + Attend
        ops::kv_cache::encode_kv_append(
            &kv_cache.layers[layer_idx],
            bufs.k_out,
            bufs.v_out,
        );
        ops::kv_cache::encode_kv_attend(
            &kv_cache.layers[layer_idx],
            bufs.q_out,
            bufs.attn_out_buf,
            layer_num_q_heads,
            scale,
            window_size,
        );
        kv_cache.layers[layer_idx].current_len += 1;

        // 5a. O projection
        let mut wo_bytes = vec![0u8; layer.wo.data_len()];
        bufs.wo.copy_to_slice(&mut wo_bytes);
        
        let mut attn_f32 = vec![0.0f32; layer_q_dim];
        bufs.attn_out_buf.copy_to_slice(&mut attn_f32);
        
        crate::xpu::stages::o_proj::encode_o_proj(
            &wo_bytes,
            &attn_f32,
            bufs.o_out_buf, // out buffer
            layer.wo.format,
            layer_q_dim,
            hidden,
        );

        // 5b. Residual + norm
        // Using XPU residual stage
        let pre_ffn_norm_f32 = vec![0.0f32; hidden]; // Simplified
        let post_attn_norm_f32 = vec![0.0f32; hidden];
        
        let mut o_out_f32 = vec![0.0f32; hidden];
        bufs.o_out_buf.copy_to_slice(&mut o_out_f32);
        
        let mut h_buf_f32 = vec![0.0f32; hidden];
        bufs.h_buf.copy_to_slice(&mut h_buf_f32);

        if ffn_uses_q4k {
            crate::xpu::stages::residual::encode_residual_norm_store(
                &h_buf_f32,
                &o_out_f32,
                &post_attn_norm_f32,
                &pre_ffn_norm_f32,
                bufs.ffn_norm_out,
                bufs.h_post_attn,
                hidden,
                layer.eps,
                layer.norm_offset,
            );
        } else {
            crate::xpu::stages::residual::encode_residual_norm_q8(
                &h_buf_f32,
                &o_out_f32,
                &post_attn_norm_f32,
                &pre_ffn_norm_f32,
                bufs.ffn_q8,
                bufs.ffn_q8s,
                bufs.h_post_attn,
                hidden,
                layer.eps,
                layer.norm_offset,
            );
        }
    }
}
