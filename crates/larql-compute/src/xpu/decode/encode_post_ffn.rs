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
        let post_ffn_norm = layer.post_ffn_norm.map(|w| self.bufs.get_f32(w)).unwrap_or_else(|| std::sync::Arc::new(XpuBuffer::new_device(0)));
        
        crate::xpu::stages::residual::encode_post_ffn_buf(
            bufs.down_out,
            bufs.h_post_attn,
            layer.post_ffn_norm.as_ref().map(|_| &*post_ffn_norm),
            bufs.new_h,
            1, // seq_len
            hidden,
            layer.eps,
            layer.norm_offset,
        );
    }
}
