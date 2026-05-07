//! MoE Interleave stub for XPU decode pipeline.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;

pub(super) struct MoeInterleaveCtx<'a> {
    pub layer_idx: usize,
    pub num_layers: usize,
    pub hidden: usize,
    pub inter: usize,
    pub inter_padded: usize,
    pub ffn_uses_q4k: bool,
    pub defer_ffn_for_split: bool,
    pub stage_timing_split: bool,
    pub layer_in_snapshot: Option<&'a [f32]>,
    pub dump_l0_dir: Option<&'a str>,
}

pub(super) struct MoeInterleaveBufs<'a> {
    pub gate_w: &'a XpuBuffer,
    pub up_w: &'a XpuBuffer,
    pub down_w: &'a XpuBuffer,
    pub h_post_attn: &'a XpuBuffer,
    pub ffn_norm_out: &'a XpuBuffer,
    pub ffn_q8: &'a XpuBuffer,
    pub ffn_q8s: &'a XpuBuffer,
    pub gate_out_scratch: &'a XpuBuffer,
    pub up_out: &'a XpuBuffer,
    pub act_buf: &'a XpuBuffer,
    pub down_out: &'a XpuBuffer,
    pub normed_scratch: &'a XpuBuffer,
    pub new_h: &'a XpuBuffer,
}

pub(super) struct MoeCommandState<'a> {
    pub gpu_time: &'a mut super::gpu_timing::TokenGpuTime,
}

impl XpuBackend {
    pub(super) fn handle_moe_interleave(
        &self,
        _layer: &FullPipelineLayer,
        _ctx: MoeInterleaveCtx,
        _bufs: MoeInterleaveBufs,
        _cmd_state: MoeCommandState,
        _moe_fn: &mut Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        _moe_collect_fn: &mut Option<&mut dyn FnMut(usize) -> Vec<f32>>,
    ) {
        unimplemented!("MoE not yet supported on XPU decode pipeline");
    }
}
