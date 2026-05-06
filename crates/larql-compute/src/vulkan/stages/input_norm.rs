//! Input layer norm — the first stage of every transformer layer for Vulkan.
//!
//! Two code paths depending on what the QKV projection wants next:
//!
//! - **f32 output** (`encode_f32`): plain `rms_norm` writing f32 to the
//!   norm-out buffer. Used by Q4_K / Q6_K attention which consume f32 input.
//! - **Fused norm + Q8 quantise** (`encode_q8`): `rms_norm` followed by `quantize_q8`
//!   writing Q8 int8s + per-32 f32-scaled blocks. Used by Q8_0 / Q4_0 attention.
//!
//! Both variants are per-position (single hidden vector per call); the
//! caller loops over positions. The caller owns the AutoCommandBufferBuilder lifecycle.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;

/// f32-output input RMS norm.
///
/// Writes `out[hidden]` as `(x / rms(x)) * (weight + offset)` using the
/// cooperative single-threadgroup `rms_norm` shader.
#[allow(clippy::too_many_arguments)]
pub fn encode_f32(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    h_buf: &Subbuffer<[f32]>,
    h_off: u64,
    norm_weight: &Subbuffer<[f32]>,
    out_buf: &Subbuffer<[f32]>,
    out_off: u64,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) {
    let hidden_val = hidden as u32;
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let h_slice = h_buf.clone().slice(h_off .. h_off + hidden as u64);
    let out_slice = out_buf.clone().slice(out_off .. out_off + hidden as u64);

    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator.clone(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, h_slice.clone()),
            WriteDescriptorSet::buffer(1, norm_weight.clone()),
            WriteDescriptorSet::buffer(2, out_slice.clone()),
        ],
        [],
    ).unwrap();

    let pcs = shaders::rms_norm::PushConstants {
        len: hidden_val,
        eps,
        offset: norm_offset,
    };

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, pcs)
        .unwrap()
        .dispatch([1, 1, 1])
        .unwrap();
}

/// Fused RMS norm + Q8 quantise fallback — writes Q8 int8 values and f32 scales.
/// 
/// Note: On Metal this is a single fused shader. For Vulkan, until a fused 
/// `rms_norm_q8` is written, we emit an `rms_norm` into a scratch buffer 
/// followed by `quantize_q8`.
#[allow(clippy::too_many_arguments)]
pub fn encode_q8(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    rms_pipeline: &Arc<ComputePipeline>,
    q8_pipeline: &Arc<ComputePipeline>,
    scratch_alloc: &mut dyn FnMut(u64) -> Subbuffer<[f32]>,
    h_buf: &Subbuffer<[f32]>,
    h_off: u64,
    norm_weight: &Subbuffer<[f32]>,
    q8_out: &Subbuffer<[i8]>,
    q8_out_off: u64,
    q8s_out: &Subbuffer<[f32]>,
    q8s_out_off: u64,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) {
    let hidden_val = hidden as u32;
    
    // 1. RMS Norm into scratch
    let scratch_f32 = scratch_alloc(hidden as u64); // Need hidden f32 elements
    encode_f32(
        builder, backend, rms_pipeline, 
        h_buf, h_off, 
        norm_weight, 
        &scratch_f32, 0, 
        hidden, eps, norm_offset
    );

    // 2. Quantize Q8
    let q8_layout = q8_pipeline.layout().set_layouts().get(0).unwrap();

    let q8_slice = q8_out.clone().slice(q8_out_off .. q8_out_off + hidden as u64);
    let q8s_slice = q8s_out.clone().slice(q8s_out_off .. q8s_out_off + (hidden / 32) as u64);

    let q8_set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator.clone(),
        q8_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, scratch_f32.clone()),
            WriteDescriptorSet::buffer(1, q8_slice.clone()),
            WriteDescriptorSet::buffer(2, q8s_slice.clone()),
        ],
        [],
    ).unwrap();

    let q8_pcs = shaders::quantize_q8::PushConstants {
        len: hidden_val,
    };

    builder
        .bind_pipeline_compute(q8_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, q8_pipeline.layout().clone(), 0, q8_set)
        .unwrap()
        .push_constants(q8_pipeline.layout().clone(), 0, q8_pcs)
        .unwrap()
        .dispatch([hidden_val.div_ceil(32), 1, 1]) // One thread per block (32 elements)
        .unwrap();
}
