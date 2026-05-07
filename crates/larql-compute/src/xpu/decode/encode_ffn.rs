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
        let activation = match layer.activation {
            crate::Activation::Silu => crate::xpu::stages::ffn::Activation::SiLU,
            crate::Activation::GeluTanh => crate::xpu::stages::ffn::Activation::GeluTanh,
        };

        if layer.is_gated() {
            crate::xpu::stages::ffn::encode_gated_buf(
                bufs.gate_w,
                bufs.up_w,
                bufs.down_w,
                bufs.ffn_norm_out,
                bufs.down_out,
                unsafe { std::mem::transmute(layer.gate.format) },
                unsafe { std::mem::transmute(layer.down.format) },
                activation,
                dims.inter,
                dims.hidden,
            );
        } else {
            crate::xpu::stages::ffn::encode_standard_buf(
                bufs.up_w,
                bufs.down_w,
                bufs.ffn_norm_out,
                bufs.down_out,
                unsafe { std::mem::transmute(layer.up.format) },
                unsafe { std::mem::transmute(layer.down.format) },
                activation,
                dims.inter,
                dims.hidden,
            );
        };
    }
}
