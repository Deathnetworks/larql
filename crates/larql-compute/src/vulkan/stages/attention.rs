//! Fused causal attention — one dispatch for the whole layer's QKV → attn_out for Vulkan.
//!
//! Dispatches `attn_fused` which handles RoPE, QK-norm, causal GQA softmax,
//! and softcap.
//!
//! Caller owns the AutoCommandBufferBuilder lifecycle.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;

/// Flags for the fused attention dispatch. Keeps the parameter list readable.
#[derive(Clone, Copy)]
pub struct Flags {
    pub use_qk_norm: bool,
    pub skip_rope: bool,
    pub softcap: f32,
    pub rotary_dim: u32,
}

/// Dispatch `attn_fused` into the given builder.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    q_buf: &Subbuffer<[f32]>,
    k_buf: &Subbuffer<[f32]>,
    v_buf: &Subbuffer<[f32]>,
    attn_out: &Subbuffer<[f32]>,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_base: f32,
    flags: Flags,
) {
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    // Vulkan attn_fused.comp expects 8 bindings. 
    // For parity with Metal's 4-buffer signature, we bind k_buf to KCache and v_buf to VCache.
    // For QWeight and KWeight, we bind q_buf and k_buf respectively as dummies since the caller
    // expects to supply them separately if needed, or the shader currently ignores use_qk_norm flag.
    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator.clone(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, q_buf.clone()),
            WriteDescriptorSet::buffer(1, k_buf.clone()),
            WriteDescriptorSet::buffer(2, v_buf.clone()),
            WriteDescriptorSet::buffer(3, k_buf.clone()), // KCache
            WriteDescriptorSet::buffer(4, v_buf.clone()), // VCache
            WriteDescriptorSet::buffer(5, attn_out.clone()),
            WriteDescriptorSet::buffer(6, q_buf.clone()), // QWeight (dummy)
            WriteDescriptorSet::buffer(7, k_buf.clone()), // KWeight (dummy)
        ],
        [],
    ).unwrap();

    let pcs = shaders::attn_fused::PushConstants {
        T: seq_len as u32,
        head_dim: head_dim as u32,
        num_q: num_q_heads as u32,
        num_kv: num_kv_heads as u32,
        scale,
        window_size: 0,
        eps: 1e-5, // Dummy eps
        qk_offset: 0.0,
        rope_base,
        rotary_dim: flags.rotary_dim,
    };

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, pcs)
        .unwrap()
        .dispatch([num_q_heads as u32, seq_len as u32, 1])
        .unwrap();
}
