//! MoE Interleave for XPU decode pipeline.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;
use super::{encode_ffn, encode_post_ffn, gpu_timing};

pub(super) struct MoeInterleaveCtx<'a> {
    pub layer_idx: usize,
    pub num_layers: usize,
    pub hidden: usize,
    pub inter: usize,
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
    pub down_out: &'a mut XpuBuffer,
    pub new_h: &'a mut XpuBuffer,
}

pub(super) struct MoeCommandState<'a> {
    pub gpu_time: &'a mut gpu_timing::TokenGpuTime,
}

impl XpuBackend {
    pub(super) fn handle_moe_interleave(
        &self,
        layer: &FullPipelineLayer,
        ctx: MoeInterleaveCtx,
        mut bufs: MoeInterleaveBufs,
        cmd_state: MoeCommandState,
        moe_fn: &mut Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        moe_collect_fn: &mut Option<&mut dyn FnMut(usize) -> Vec<f32>>,
    ) {
        if layer.moe.is_none() && !layer.ffn_is_remote {
            return;
        }

        // 1. Ensure GPU is done with attention (by reading h_post_attn)
        let mut attn_host = vec![0.0f32; ctx.hidden];
        bufs.h_post_attn.copy_to_slice(&mut attn_host);

        // 2. Fire MoE or expert callback
        let moe_out = if ctx.defer_ffn_for_split {
            let fire = moe_fn.as_deref_mut().expect("split_mode implies moe_fn");
            fire(ctx.layer_idx, &attn_host);

            // Encode dense FFN while remote trip is in flight
            self.encode_ffn_step(
                layer,
                encode_ffn::FfnBufs {
                    gate_w: bufs.gate_w,
                    up_w: bufs.up_w,
                    down_w: bufs.down_w,
                    ffn_norm_out: bufs.ffn_norm_out,
                    down_out: bufs.down_out,
                },
                encode_ffn::FfnDims {
                    hidden: ctx.hidden,
                    inter: ctx.inter,
                },
            );

            self.encode_post_ffn_residual(
                layer,
                encode_post_ffn::PostFfnBufs {
                    down_out: bufs.down_out,
                    h_post_attn: bufs.h_post_attn,
                    new_h: bufs.new_h,
                },
                ctx.hidden,
            );

            let collect = moe_collect_fn.as_deref_mut().expect("split_mode implies moe_collect_fn");
            let result = collect(ctx.layer_idx);
            
            cmd_state.gpu_time.record_stage(&(), gpu_timing::DecodeStage::GateUp);
            result
        } else if let Some(ref mut f) = moe_fn {
            f(ctx.layer_idx, &attn_host)
        } else {
            // Local expert fallback
            let moe = layer.moe.as_ref().expect("cpu_moe_forward requires moe weights");
            crate::cpu::ops::moe::cpu_moe_forward(&attn_host, moe, layer.norm_offset, layer.eps)
        };

        // 3. Accumulate MoE result back to new_h (CPU-side)
        let mut h_ptr_host = vec![0.0f32; ctx.hidden];
        bufs.new_h.copy_to_slice(&mut h_ptr_host);

        if layer.ffn_is_remote {
            for (i, v) in moe_out.iter().enumerate() {
                h_ptr_host[i] = attn_host[i] + v;
            }
        } else {
            for (i, v) in moe_out.iter().enumerate() {
                h_ptr_host[i] += v;
            }
        }
        bufs.new_h.copy_from_slice(&h_ptr_host);

        // 4. Final combine (outer norm + layer scalar)
        self.handle_moe_combine(layer, bufs.new_h, bufs.h_post_attn, ctx.hidden);
    }
}
