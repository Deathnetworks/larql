//! Step 6 of decode loop: FFN projection.

use crate::xpu::XpuBackend;
use crate::FullPipelineLayer;
use crate::xpu::buffers::XpuBuffer;

pub(super) struct FfnBufs<'a> {
    pub gate_w: &'a XpuBuffer,
    pub up_w: &'a XpuBuffer,
    pub down_w: &'a XpuBuffer,
    pub ffn_norm_out: &'a mut XpuBuffer,
    pub down_out: &'a mut XpuBuffer,
}

pub(super) struct FfnDims {
    pub hidden: usize,
    pub inter: usize,
}

impl XpuBackend {
    pub(super) fn encode_ffn_step(
        &self,
        layer: &FullPipelineLayer,
        mut bufs: FfnBufs<'_>,
        dims: FfnDims,
    ) {
        let mut x_norm = vec![0.0f32; dims.hidden];
        bufs.ffn_norm_out.copy_to_slice(&mut x_norm);

        let mut gate_w_bytes = vec![0u8; layer.gate.data.len()];
        bufs.gate_w.copy_to_slice(&mut gate_w_bytes);
        let mut up_w_bytes = vec![0u8; layer.up.data.len()];
        bufs.up_w.copy_to_slice(&mut up_w_bytes);
        let mut down_w_bytes = vec![0u8; layer.down.data.len()];
        bufs.down_w.copy_to_slice(&mut down_w_bytes);

        let activation = match layer.activation {
            crate::Activation::Silu => crate::xpu::stages::ffn::Activation::SiLU,
            crate::Activation::GeluTanh => crate::xpu::stages::ffn::Activation::GeluTanh,
        };

        let down_f32 = if layer.is_gated() {
            crate::xpu::stages::ffn::encode_gated(
                &gate_w_bytes,
                &up_w_bytes,
                &down_w_bytes,
                &x_norm,
                unsafe { std::mem::transmute(layer.gate.format) },
                unsafe { std::mem::transmute(layer.down.format) },
                activation,
                dims.inter,
                dims.hidden,
            )
        } else {
            crate::xpu::stages::ffn::encode_standard(
                &up_w_bytes,
                &down_w_bytes,
                &x_norm,
                unsafe { std::mem::transmute(layer.up.format) },
                unsafe { std::mem::transmute(layer.down.format) },
                activation,
                dims.inter,
                dims.hidden,
            )
        };

        bufs.down_out.copy_from_slice(&down_f32);
    }
}
