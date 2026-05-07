//! Step 7 of decode loop: post-FFN residual.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;

pub(super) struct PostFfnBufs<'a> {
    pub down_out: &'a mut XpuBuffer,
    pub h_post_attn: &'a mut XpuBuffer,
    pub new_h: &'a mut XpuBuffer,
}

impl XpuBackend {
    pub(super) fn encode_post_ffn_residual(
        &self,
        layer: &FullPipelineLayer,
        mut bufs: PostFfnBufs<'_>,
        hidden: usize,
    ) {
        let mut down_out_f32 = vec![0.0f32; hidden];
        bufs.down_out.copy_to_slice(&mut down_out_f32);
        
        let mut h_post_attn_f32 = vec![0.0f32; hidden];
        bufs.h_post_attn.copy_to_slice(&mut h_post_attn_f32);

        let new_h_f32 = crate::xpu::stages::residual::encode_post_ffn(
            &down_out_f32,
            &h_post_attn_f32,
            layer.post_ffn_norm,
            1, // seq_len
            hidden,
            layer.eps,
            layer.norm_offset,
        );

        bufs.new_h.copy_from_slice(&new_h_f32);
    }
}
