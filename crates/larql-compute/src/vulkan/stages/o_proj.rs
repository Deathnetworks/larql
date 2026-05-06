//! Output projection (`attn_out → h_post_attn_input`) — per position for Vulkan.
//!
//! Thin wrapper over [`super::quant_matvec::encode`] that routes the
//! attention output through the right shader based on the O-weight format:
//!
//! - **Q4_K / Q4_KF / Q6_K**: f32 input directly; single matvec dispatch.
//! - **Q4_0 / Q8_0**: quantise `attn_out` to Q8 first (callers supply a
//!   staging buffer), then Q8 matvec.
//!
//! Single-vector per position. Multi-position prefill loops.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;
use super::quant_matvec;

/// Per-position O projection. Caller owns the encoder lifecycle.
///
/// For Q4_K / Q4_KF / Q6_K this is one dispatch. For Q4_0 / Q8_0 we first
/// quantise `attn_in` to the caller's Q8 staging buffer.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipes: &quant_matvec::Pipelines<'_>,
    q8_quant_pipeline: &Arc<ComputePipeline>,
    format: crate::QuantFormat,
    wo_buf: &Subbuffer<[u32]>,
    attn_in: &Subbuffer<[f32]>,
    attn_in_off: u64,
    q8_stage: &Subbuffer<[i8]>,
    q8_stage_off: u64,
    q8s_stage: &Subbuffer<[f32]>,
    q8s_stage_off: u64,
    o_out: &Subbuffer<[f32]>,
    o_out_off: u64,
    q_dim: usize,
    hidden: usize,
) {
    let is_f32_input = matches!(
        format,
        crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF | crate::QuantFormat::Q6_K
    );

    let dim_val = q_dim as u32;

    let attn_in_slice = attn_in.clone().slice(attn_in_off .. attn_in_off + q_dim as u64);
    let o_out_slice = o_out.clone().slice(o_out_off .. o_out_off + hidden as u64);

    let q8_slice = q8_stage.clone().slice(q8_stage_off .. q8_stage_off + q_dim as u64);
    let q8s_slice = q8s_stage.clone().slice(q8s_stage_off .. q8s_stage_off + (q_dim / 32) as u64);

    if !is_f32_input {
        // Q4_0 / Q8_0: quantise attn_in[q_dim] → Q8 int8 + per-32 f32 scale.
        let q8_layout = q8_quant_pipeline.layout().set_layouts().get(0).unwrap();

        let q8_set = PersistentDescriptorSet::new(
            &backend.descriptor_set_allocator,
            q8_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, attn_in_slice.clone()),
                WriteDescriptorSet::buffer(1, q8_slice.clone()),
                WriteDescriptorSet::buffer(2, q8s_slice.clone()),
            ],
            [],
        ).unwrap();

        let q8_pcs = shaders::quantize_q8::PushConstants {
            K: dim_val,
        };

        builder
            .bind_pipeline_compute(q8_quant_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, q8_quant_pipeline.layout().clone(), 0, q8_set)
            .unwrap()
            .push_constants(q8_quant_pipeline.layout().clone(), 0, q8_pcs)
            .unwrap()
            .dispatch([dim_val.div_ceil(32), 1, 1]) // 1 thread per block of 32
            .unwrap();
    }

    quant_matvec::encode(
        builder,
        backend,
        format,
        wo_buf,
        &attn_in_slice,
        &q8_slice,
        &q8s_slice,
        &o_out_slice,
        pipes,
        hidden,
        q_dim,
    );
}
