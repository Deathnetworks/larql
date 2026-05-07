//! Per-layer attention block — Steps 1.5 through 5 of the decode loop for XPU.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;
use crate::xpu::ops;

pub(super) struct AttnBufs<'a> {
    pub h_buf: &'a XpuBuffer,
    pub q_out: &'a mut XpuBuffer,
    pub k_out: &'a mut XpuBuffer,
    pub v_out: &'a mut XpuBuffer,
    pub attn_out_buf: &'a mut XpuBuffer,
    pub o_out_buf: &'a mut XpuBuffer,
    pub ffn_norm_out: &'a mut XpuBuffer,
    pub h_post_attn: &'a mut XpuBuffer,
    pub ffn_q8: &'a mut XpuBuffer,
    pub ffn_q8s: &'a mut XpuBuffer,
    pub wo: &'a XpuBuffer,
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
        mut bufs: AttnBufs<'_>,
        dims: AttnDims,
    ) {
        let AttnDims {
            hidden,
            layer_q_dim,
            uses_q4k: _,
            ffn_uses_q4k,
        } = dims;

        let layer_kv_dim = layer.num_kv_heads * layer.head_dim;

        let flags = crate::xpu::stages::attention::Flags {
            window_size: layer.sliding_window as u32,
            rms_eps: if layer.q_norm_weight.is_some() { layer.eps } else { 0.0 },
            qk_offset: layer.qk_norm_offset,
            rope_base: layer.rope_base,
            rotary_dim: if layer.rotary_dim > 0 { layer.rotary_dim as u32 } else { layer.head_dim as u32 },
        };

        let mut q_in_f32 = vec![0.0f32; layer_q_dim];
        let mut k_in_f32 = vec![0.0f32; layer_kv_dim];
        let mut v_in_f32 = vec![0.0f32; layer_kv_dim];
        
        bufs.q_out.copy_to_slice(&mut q_in_f32);
        bufs.k_out.copy_to_slice(&mut k_in_f32);
        bufs.v_out.copy_to_slice(&mut v_in_f32);

        let q_w_f32 = layer.q_norm_weight.unwrap_or(&[]);
        let k_w_f32 = layer.k_norm_weight.unwrap_or(&[]);

        let pos = kv_cache.layers[layer_idx].current_len;

        // Fused Attention (QK-norm, RoPE, Append, Attend)
        let attn_out_f32 = crate::xpu::stages::attention::encode(
            &q_in_f32,
            &k_in_f32,
            &v_in_f32,
            q_w_f32,
            k_w_f32,
            &mut kv_cache.layers[layer_idx],
            pos,
            layer.num_q_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.attn_scale,
            flags,
        );

        bufs.attn_out_buf.copy_from_slice(&attn_out_f32);

        // O projection
        let mut wo_bytes = vec![0u8; layer.wo.data.len()];
        bufs.wo.copy_to_slice(&mut wo_bytes);
        
        let o_out_f32 = if layer.wo.format == crate::QuantFormat::Q6_K {
            crate::xpu::stages::o_proj::encode_q6k(
                &wo_bytes,
                &attn_out_f32,
                layer_q_dim,
                hidden,
            )
        } else {
            crate::xpu::stages::o_proj::encode(
                &wo_bytes,
                &attn_out_f32,
                layer_q_dim,
                hidden,
            )
        };

        bufs.o_out_buf.copy_from_slice(&o_out_f32);

        // Residual + norm
        let pre_ffn_norm_f32 = layer.pre_ffn_norm.unwrap_or(&[]);
        let post_attn_norm_f32 = layer.post_attn_norm;
        
        let mut h_buf_f32 = vec![0.0f32; hidden];
        bufs.h_buf.copy_to_slice(&mut h_buf_f32);

        let (h_post_attn_f32, ffn_norm_out_f32) = crate::xpu::stages::residual::encode_post_attn(
            &h_buf_f32,
            &o_out_f32,
            post_attn_norm_f32,
            pre_ffn_norm_f32,
            1, // seq_len
            hidden,
            layer.eps,
            layer.norm_offset,
            layer.has_post_norms,
        );

        bufs.h_post_attn.copy_from_slice(&h_post_attn_f32);
        bufs.ffn_norm_out.copy_from_slice(&ffn_norm_out_f32);

        if !ffn_uses_q4k {
            // Re-quantize for Q8 FFN
            crate::xpu::stages::input_norm::encode_q8(
                &ffn_norm_out_f32,
                &[], // no additional norm weight needed here
                bufs.ffn_q8,
                bufs.ffn_q8s,
                hidden,
                layer.eps,
                0.0,
            );
        }
    }
}
