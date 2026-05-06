//! Rotary position embedding (RoPE) — pre-attention when KV cache is used.
//!
//! Applies RoPE to Q and K in-place per head per position for Vulkan. Supports
//! partial rotation (Gemma 4 global layers use `rotary_dim = head_dim / 4`).
//!
//! Caller owns the AutoCommandBufferBuilder lifecycle.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;

/// Apply RoPE to Q and K per head per position.
///
/// `rotary_dim == 0` is treated by the shader as "rotate full head_dim".
/// Partial rotation (Gemma 4 global layers) uses `rotary_dim < head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    q_buf: &Subbuffer<[f32]>,
    k_buf: &Subbuffer<[f32]>,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_base: f32,
) {
    let hd = head_dim as u32;
    let rdim_val = rotary_dim as u32;
    let rdim_effective = if rotary_dim == 0 {
        head_dim as u32
    } else {
        rotary_dim as u32
    };
    let hdim = rdim_effective / 2;

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let total_heads = (num_q_heads + num_kv_heads) as u32;

    let q_pos_elements = num_q_heads * head_dim;
    let k_pos_elements = num_kv_heads * head_dim;

    for pos in 0..seq_len {
        let pos_val = pos as u32;

        let q_off = (pos * q_pos_elements) as u64;
        let k_off = (pos * k_pos_elements) as u64;

        let q_slice = q_buf.clone().slice(q_off .. q_off + q_pos_elements as u64);
        let k_slice = k_buf.clone().slice(k_off .. k_off + k_pos_elements as u64);

        let set = PersistentDescriptorSet::new(
            &backend.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, q_slice.clone()),
                WriteDescriptorSet::buffer(1, k_slice.clone()),
            ],
            [],
        ).unwrap();

        let pcs = shaders::rope::PushConstants {
            head_dim: hd,
            rope_base,
            pos: pos_val,
            rotary_dim: rdim_val,
            num_q: num_q_heads as u32,
            num_kv: num_kv_heads as u32,
            mode: 1, // 1: batched qk
        };

        // local_size_x = 32, local_size_y = 8
        let tg_x = hdim.div_ceil(32);
        let tg_y = total_heads.div_ceil(8);

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, pcs)
            .unwrap()
            .dispatch([tg_x, tg_y, 1])
            .unwrap();
    }
}
