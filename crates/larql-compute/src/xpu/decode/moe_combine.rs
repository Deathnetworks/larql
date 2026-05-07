//! MoE Combine stub for XPU decode pipeline.

use crate::xpu::XpuBackend;
use crate::xpu::buffers::XpuBuffer;

impl XpuBackend {
    pub(super) fn handle_moe_combine(
        &self,
        _h_post_attn: &XpuBuffer,
        _moe_output: &[f32],
        _new_h: &mut XpuBuffer,
        _hidden: usize,
    ) {
        unimplemented!("MoE not yet supported on XPU decode pipeline");
    }
}
