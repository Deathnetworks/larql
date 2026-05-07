//! Step 6 of the decode pipeline: FFN (gate, up, activation, down).

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;

pub(super) struct FfnBufs<'a> {
    pub gate_w: &'a XpuBuffer,
    pub up_w: &'a XpuBuffer,
    pub down_w: &'a XpuBuffer,
    pub ffn_norm_out: &'a XpuBuffer,
    pub ffn_q8: &'a XpuBuffer,
    pub ffn_q8s: &'a XpuBuffer,
    pub gate_out_scratch: &'a XpuBuffer,
    pub up_out: &'a XpuBuffer,
    pub act_buf: &'a XpuBuffer,
    pub down_out: &'a XpuBuffer,
}

#[derive(Copy, Clone)]
pub(super) struct FfnDims {
    pub hidden: usize,
    pub inter: usize,
    pub inter_padded: usize,
}

impl XpuBackend {
    pub(super) fn encode_ffn_step(
        &self,
        layer: &FullPipelineLayer,
        bufs: FfnBufs<'_>,
        dims: FfnDims,
        uses_q4k: bool,
    ) {
        let FfnDims { hidden, inter, .. } = dims;

        let mut gate_w_bytes = vec![0u8; layer.gate.data_len()];
        bufs.gate_w.copy_to_slice(&mut gate_w_bytes);
        let mut up_w_bytes = vec![0u8; layer.up.data_len()];
        bufs.up_w.copy_to_slice(&mut up_w_bytes);
        let mut down_w_bytes = vec![0u8; layer.down.data_len()];
        bufs.down_w.copy_to_slice(&mut down_w_bytes);

        let mut ffn_in_f32 = vec![0.0f32; hidden];
        bufs.ffn_norm_out.copy_to_slice(&mut ffn_in_f32);

        if uses_q4k {
            let (down, gate_out) = crate::xpu::stages::ffn::encode_fused_gate_up_q4k(
                &gate_w_bytes,
                &up_w_bytes,
                &down_w_bytes,
                &ffn_in_f32,
                inter,
                hidden,
            );
            bufs.down_out.copy_from_slice(&down);
            bufs.gate_out_scratch.copy_from_slice(&gate_out);
        } else {
            // Unfused path
            let gate = crate::xpu::stages::ffn::encode_unfused_gate(
                &gate_w_bytes,
                &ffn_in_f32,
                inter,
                hidden,
            );
            let up = crate::xpu::stages::ffn::encode_unfused_up(
                &up_w_bytes,
                &ffn_in_f32,
                inter,
                hidden,
            );
            bufs.gate_out_scratch.copy_from_slice(&gate);
            bufs.up_out.copy_from_slice(&up);
            
            let mut act = vec![0.0f32; inter];
            crate::xpu::stages::ffn::encode_geglu(&gate, &up, &mut act, inter as u32);
            bufs.act_buf.copy_from_slice(&act);
            
            let down = crate::xpu::stages::ffn::encode_unfused_down(
                &down_w_bytes,
                &act,
                hidden,
                inter,
            );
            bufs.down_out.copy_from_slice(&down);
        }
    }
}
