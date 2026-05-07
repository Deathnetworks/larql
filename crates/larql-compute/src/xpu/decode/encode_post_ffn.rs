//! Step 7 of the decode pipeline: post-FFN residual.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;

pub(super) struct PostFfnBufs<'a> {
    pub down_out: &'a XpuBuffer,
    pub h_post_attn: &'a XpuBuffer,
    pub new_h: &'a XpuBuffer,
    pub normed_scratch: &'a XpuBuffer,
}

impl XpuBackend {
    pub(super) fn encode_post_ffn_residual(
        &self,
        _layer: &FullPipelineLayer,
        bufs: PostFfnBufs<'_>,
        hidden: usize,
        _use_fused_post_ffn: bool,
    ) {
        let mut down_out_f32 = vec![0.0f32; hidden];
        bufs.down_out.copy_to_slice(&mut down_out_f32);
        
        let mut h_post_attn_f32 = vec![0.0f32; hidden];
        bufs.h_post_attn.copy_to_slice(&mut h_post_attn_f32);
        
        crate::xpu::stages::residual::encode_post_ffn_residual(
            &h_post_attn_f32,
            &down_out_f32,
            bufs.new_h,
            hidden as u32,
        );
    }
}
