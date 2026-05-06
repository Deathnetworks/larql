//! Feed-forward block — gate+up → activation → down for Vulkan.
//!
//! Two variants depending on `FfnType`:
//!
//! - **Gated** (Llama / Gemma / Qwen / most modern): `out = down(act(gate) ⊙ up)`
//!   with activation = SiLU or GELU-tanh.
//!
//! - **Standard** (StarCoder2): `out = down(act(up))`. Dispatched as
//!   `up_matvec + activation + down_matvec`. No gate.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;
use super::quant_matvec;

/// Activation variant for this layer.
#[derive(Clone, Copy)]
pub enum Activation {
    SiLU,
    GeluTanh,
}

pub struct FusedGegluDown<'a> {
    pub q4k_silu: Option<&'a Arc<ComputePipeline>>,
    pub q4k_gelu_tanh: Option<&'a Arc<ComputePipeline>>,
    pub q6k_silu: Option<&'a Arc<ComputePipeline>>,
    pub q6k_gelu_tanh: Option<&'a Arc<ComputePipeline>>,
}

/// Gated FFN (Llama / Gemma / Qwen): `down(act(gate) * up)`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gated(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipes: &quant_matvec::Pipelines<'_>,
    geglu_silu_pipeline: &Arc<ComputePipeline>,
    geglu_gelu_tanh_pipeline: &Arc<ComputePipeline>,
    _fused_down: FusedGegluDown<'_>, // Fused down not yet implemented in Vulkan
    gate_format: crate::QuantFormat,
    up_format: crate::QuantFormat,
    down_format: crate::QuantFormat,
    activation: Activation,
    gate_buf: &Subbuffer<[u32]>,
    up_buf: &Subbuffer<[u32]>,
    down_buf: &Subbuffer<[u32]>,
    ffn_norm_out: &Subbuffer<[f32]>, // f32 input for Q4_K / Q6_K
    ffn_q8_in: &Subbuffer<[i8]>,    // Q8 input for Q4_0 / Q8_0
    ffn_q8s_in: &Subbuffer<[f32]>,
    gate_scratch: &Subbuffer<[f32]>, // holds per-position `inter` floats
    up_scratch: &Subbuffer<[f32]>,
    act_scratch: &Subbuffer<[f32]>,
    down_out: &Subbuffer<[f32]>,
    seq_len: usize,
    inter: usize,
    hidden: usize,
    h_stride_bytes: u64,
    inter_stride_bytes: u64,
    q8_stride_bytes: u64,
    q8s_stride_bytes: u64,
) {
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;

        let norm_slice = ffn_norm_out.clone().slice(h_off .. h_off + hidden as u64);
        let gate_scratch_slice = gate_scratch.clone().slice(inter_off .. inter_off + inter as u64);
        let up_scratch_slice = up_scratch.clone().slice(inter_off .. inter_off + inter as u64);

        let q8_slice = ffn_q8_in.clone().slice(q8_off .. q8_off + hidden as u64);
        let q8s_slice = ffn_q8s_in.clone().slice(q8s_off .. q8s_off + (hidden / 32) as u64);

        quant_matvec::encode(
            builder, backend, gate_format, gate_buf, &norm_slice, &q8_slice, &q8s_slice,
            &gate_scratch_slice, pipes, inter, hidden,
        );
        quant_matvec::encode(
            builder, backend, up_format, up_buf, &norm_slice, &q8_slice, &q8s_slice,
            &up_scratch_slice, pipes, inter, hidden,
        );
    }

    // Separated path: GEGLU then format-aware down.
    {
        let total_inter = (seq_len * inter) as u32;
        let act_pipe = match activation {
            Activation::GeluTanh => geglu_gelu_tanh_pipeline,
            Activation::SiLU => geglu_silu_pipeline,
        };
        let layout = act_pipe.layout().set_layouts().get(0).unwrap();

        let set = PersistentDescriptorSet::new(
            &backend.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, gate_scratch.clone()),
                WriteDescriptorSet::buffer(1, up_scratch.clone()),
                WriteDescriptorSet::buffer(2, act_scratch.clone()),
            ],
            [],
        ).unwrap();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            total_inter: u32,
            mode: u32,
        }

        let pcs = PushConstants {
            total_inter,
            mode: match activation {
                Activation::SiLU => 0,
                Activation::GeluTanh => 1,
            },
        };

        builder
            .bind_pipeline_compute(act_pipe.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, act_pipe.layout().clone(), 0, set)
            .unwrap()
            .push_constants(act_pipe.layout().clone(), 0, pcs)
            .unwrap()
            .dispatch([total_inter.div_ceil(256), 1, 1])
            .unwrap();
    }

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;

        let act_slice = act_scratch.clone().slice(inter_off .. inter_off + inter as u64);
        let down_slice = down_out.clone().slice(h_off .. h_off + hidden as u64);
        
        // Q8 input buffers are assumed unused for down_proj since down_format is typically Q4_K/Q6_K
        let q8_slice = ffn_q8_in.clone().slice(q8_off .. q8_off + hidden as u64);
        let q8s_slice = ffn_q8s_in.clone().slice(q8s_off .. q8s_off + (hidden / 32) as u64);

        quant_matvec::encode(
            builder, backend, down_format, down_buf, &act_slice, &q8_slice, &q8s_slice,
            &down_slice, pipes, hidden, inter,
        );
    }
}

/// Standard FFN (StarCoder2): `down(act(up))`. No gate.
#[allow(clippy::too_many_arguments)]
pub fn encode_standard(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipes: &quant_matvec::Pipelines<'_>,
    silu_pipeline: &Arc<ComputePipeline>,
    gelu_tanh_pipeline: &Arc<ComputePipeline>,
    up_format: crate::QuantFormat,
    down_format: crate::QuantFormat,
    activation: Activation,
    up_buf: &Subbuffer<[u32]>,
    down_buf: &Subbuffer<[u32]>,
    ffn_norm_out: &Subbuffer<[f32]>,
    ffn_q8_in: &Subbuffer<[i8]>,
    ffn_q8s_in: &Subbuffer<[f32]>,
    up_scratch: &Subbuffer<[f32]>,
    act_scratch: &Subbuffer<[f32]>,
    down_out: &Subbuffer<[f32]>,
    seq_len: usize,
    inter: usize,
    hidden: usize,
    h_stride_bytes: u64,
    inter_stride_bytes: u64,
    q8_stride_bytes: u64,
    q8s_stride_bytes: u64,
) {
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;

        let norm_slice = ffn_norm_out.clone().slice(h_off .. h_off + hidden as u64);
        let up_scratch_slice = up_scratch.clone().slice(inter_off .. inter_off + inter as u64);

        let q8_slice = ffn_q8_in.clone().slice(q8_off .. q8_off + hidden as u64);
        let q8s_slice = ffn_q8s_in.clone().slice(q8s_off .. q8s_off + (hidden / 32) as u64);

        quant_matvec::encode(
            builder, backend, up_format, up_buf, &norm_slice, &q8_slice, &q8s_slice,
            &up_scratch_slice, pipes, inter, hidden,
        );
    }

    {
        let total_inter = (seq_len * inter) as u32;
        let act_pipe = match activation {
            Activation::GeluTanh => gelu_tanh_pipeline,
            Activation::SiLU => silu_pipeline,
        };
        let layout = act_pipe.layout().set_layouts().get(0).unwrap();

        let set = PersistentDescriptorSet::new(
            &backend.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, up_scratch.clone()),
                WriteDescriptorSet::buffer(1, act_scratch.clone()), // Dummy for gate/up format compatibility, or activation shader handles it
            ],
            [],
        ).unwrap();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            n: u32,
            mode: u32,
        }

        let pcs = PushConstants {
            n: total_inter,
            mode: match activation {
                Activation::SiLU => 0,
                Activation::GeluTanh => 1,
            },
        };

        builder
            .bind_pipeline_compute(act_pipe.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, act_pipe.layout().clone(), 0, set)
            .unwrap()
            .push_constants(act_pipe.layout().clone(), 0, pcs)
            .unwrap()
            .dispatch([total_inter.div_ceil(256), 1, 1])
            .unwrap();
    }

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;

        let act_slice = act_scratch.clone().slice(inter_off .. inter_off + inter as u64);
        let down_slice = down_out.clone().slice(h_off .. h_off + hidden as u64);
        
        let q8_slice = ffn_q8_in.clone().slice(q8_off .. q8_off + hidden as u64);
        let q8s_slice = ffn_q8s_in.clone().slice(q8s_off .. q8s_off + (hidden / 32) as u64);

        quant_matvec::encode(
            builder, backend, down_format, down_buf, &act_slice, &q8_slice, &q8s_slice,
            &down_slice, pipes, hidden, inter,
        );
    }
}
