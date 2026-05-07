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

        let flags = crate::xpu::stages::attention::Flags {
            window_size: layer.sliding_window as u32,
            rms_eps: if layer.q_norm_weight.is_some() { layer.eps } else { 0.0 },
            qk_offset: layer.qk_norm_offset,
            rope_base: layer.rope_base,
            rotary_dim: if layer.rotary_dim > 0 { layer.rotary_dim as u32 } else { layer.head_dim as u32 },
        };

        let q_norm_w = layer.q_norm_weight.map(|w| self.bufs.get_f32(w)).unwrap_or_else(|| std::sync::Arc::new(XpuBuffer::new_device(0)));
        let k_norm_w = layer.k_norm_weight.map(|w| self.bufs.get_f32(w)).unwrap_or_else(|| std::sync::Arc::new(XpuBuffer::new_device(0)));

        let pos = kv_cache.layers[layer_idx].current_len;

        // Fused Attention (QK-norm, RoPE, Append, Attend)
        crate::xpu::stages::attention::encode_buf(
            bufs.q_out,
            bufs.k_out,
            bufs.v_out,
            &q_norm_w,
            &k_norm_w,
            &mut kv_cache.layers[layer_idx],
            bufs.attn_out_buf,
            pos,
            layer.num_q_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.attn_scale,
            flags,
        );

        // O projection
        if layer.wo.format == crate::QuantFormat::Q6_K {
            crate::xpu::stages::o_proj::encode_q6k_buf(
                bufs.wo,
                bufs.attn_out_buf,
                bufs.o_out_buf,
                layer_q_dim,
                hidden,
            );
        } else {
            crate::xpu::stages::o_proj::encode_buf(
                bufs.wo,
                bufs.attn_out_buf,
                bufs.o_out_buf,
                layer_q_dim,
                hidden,
            );
        }

        // Residual + norm
        let pre_ffn_norm = layer.pre_ffn_norm.map(|w| self.bufs.get_f32(w)).unwrap_or_else(|| std::sync::Arc::new(XpuBuffer::new_device(0)));
        let post_attn_norm = layer.post_attn_norm.map(|w| self.bufs.get_f32(w)).unwrap_or_else(|| std::sync::Arc::new(XpuBuffer::new_device(0)));
        
        crate::xpu::stages::residual::encode_post_attn_buf(
            bufs.h_buf,
            bufs.o_out_buf,
            &post_attn_norm,
            &pre_ffn_norm,
            bufs.h_post_attn,
            bufs.ffn_norm_out,
            1, // seq_len
            hidden,
            layer.eps,
            layer.norm_offset,
            layer.has_post_norms,
        );

        if !ffn_uses_q4k {
            // Re-quantize for Q8 FFN
            crate::xpu::stages::input_norm::encode_q8_buf(
                bufs.ffn_norm_out,
                &XpuBuffer::new_device(0), // No additional norm weight
                bufs.ffn_q8,
                bufs.ffn_q8s,
                hidden,
                layer.eps,
                0.0,
            );
        }
    }
}
