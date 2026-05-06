//! QK-norm and V-norm — per-head RMS norm applied inside attention for Vulkan.
//!
//! Variants differ in:
//!   - Whose buffer they target (Q vs K vs V)
//!   - Which weight they multiply (learned q_norm / k_norm / all-ones)
//!   - The norm offset
//!
//! The Caller owns the AutoCommandBufferBuilder lifecycle.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;

/// Compute the threadgroup width for a `head_dim`-long cooperative reduction.
/// In Vulkan, the shader currently uses a fixed local_size_x of 256. We dispatch
/// exactly 1 group per head, and the shader loops over `head_dim`.
fn _tg_width(_head_dim: usize) -> u32 {
    1 // Group count per head is 1
}

/// Per-head RMS norm on Q and K (pre-RoPE, Gemma 3 / Gemma 4).
#[allow(clippy::too_many_arguments)]
pub fn encode_qk_norm(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    q_buf: &Subbuffer<[f32]>,
    q_w_buf: &Subbuffer<[f32]>,
    k_buf: &Subbuffer<[f32]>,
    k_w_buf: &Subbuffer<[f32]>,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    qk_norm_offset: f32,
) {
    let hd_val = head_dim as u32;
    let nq_val = num_q_heads as u32;
    let nkv_val = num_kv_heads as u32;
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let q_head_bytes = num_q_heads * head_dim;
    let kv_head_bytes = num_kv_heads * head_dim;

    for pos in 0..seq_len {
        let q_buf_off = (pos * q_head_bytes) as u64;
        let q_slice = q_buf.clone().slice(q_buf_off .. q_buf_off + q_head_bytes as u64);

        let q_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, q_slice.clone()), // X
                WriteDescriptorSet::buffer(1, q_slice.clone()), // Out (in-place)
                WriteDescriptorSet::buffer(2, q_w_buf.clone()), // W
            ],
            [],
        ).unwrap();

        let q_pcs = shaders::qk_norm::PushConstants {
            head_dim: hd_val,
            num_heads: nq_val,
            eps,
            offset: qk_norm_offset,
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, q_set)
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, q_pcs)
            .unwrap()
            .dispatch([nq_val, 1, 1])
            .unwrap();

        let k_buf_off = (pos * kv_head_bytes) as u64;
        let k_slice = k_buf.clone().slice(k_buf_off .. k_buf_off + kv_head_bytes as u64);

        let k_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, k_slice.clone()),
                WriteDescriptorSet::buffer(1, k_slice.clone()),
                WriteDescriptorSet::buffer(2, k_w_buf.clone()),
            ],
            [],
        ).unwrap();

        let k_pcs = shaders::qk_norm::PushConstants {
            head_dim: hd_val,
            num_heads: nkv_val,
            eps,
            offset: qk_norm_offset,
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, k_set)
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, k_pcs)
            .unwrap()
            .dispatch([nkv_val, 1, 1])
            .unwrap();
    }
}

/// Parameter-free per-head RMS norm on V (Gemma 4).
#[allow(clippy::too_many_arguments)]
pub fn encode_v_norm(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    v_buf: &Subbuffer<[f32]>,
    ones_buf: &Subbuffer<[f32]>,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let hd_val = head_dim as u32;
    let nkv_val = num_kv_heads as u32;
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let kv_head_bytes = num_kv_heads * head_dim;

    for pos in 0..seq_len {
        let v_buf_off = (pos * kv_head_bytes) as u64;
        let v_slice = v_buf.clone().slice(v_buf_off .. v_buf_off + kv_head_bytes as u64);

        let v_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, v_slice.clone()),
                WriteDescriptorSet::buffer(1, v_slice.clone()),
                WriteDescriptorSet::buffer(2, ones_buf.clone()),
            ],
            [],
        ).unwrap();

        let v_pcs = shaders::v_norm::PushConstants {
            head_dim: hd_val,
            num_heads: nkv_val,
            eps,
            offset: 0.0,
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, v_set)
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, v_pcs)
            .unwrap()
            .dispatch([nkv_val, 1, 1])
            .unwrap();
    }
}
