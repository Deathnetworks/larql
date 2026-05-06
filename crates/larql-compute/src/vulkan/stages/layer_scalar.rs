//! Per-layer residual scalar — Gemma 4's learned stabiliser for Vulkan.
//!
//! Multiplies the layer's final residual by a per-layer scalar.
//! Mirrors `apply_layer_scalar` on the CPU path.
//!
//! Scoped to positions 0..seq_len for multi-position prefill; decode
//! calls with seq_len = 1.
//!
//! Caller owns the AutoCommandBufferBuilder lifecycle.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;

/// If `scalar` is non-zero, scale the f32 residual at each position by `scalar`.
///
/// * `h_buf` is the residual buffer holding `seq_len × hidden` f32s.
/// * `pipeline` must be the pipeline for the `residual_ops` shader.
pub fn encode(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    h_buf: &Subbuffer<[f32]>,
    seq_len: usize,
    hidden: usize,
    scalar: f32,
) {
    if scalar == 0.0 {
        return;
    }

    let hidden_val = hidden as u32;
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    for pos in 0..seq_len {
        let offset = (pos * hidden) as u64;
        let h_slice = h_buf.clone().slice(offset..offset + hidden as u64);

        let push_constants = shaders::residual_ops::PushConstants {
            len: hidden_val,
            scalar,
            mode: 2, // 2: scale
        };

        // For scaling, 'A' is the input to scale, 'B' is unused, and 'Output' is overwritten.
        let set = PersistentDescriptorSet::new(
            &backend.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, h_slice.clone()),
                WriteDescriptorSet::buffer(1, h_slice.clone()), // Dummy for unused
                WriteDescriptorSet::buffer(2, h_slice.clone()),
            ],
            [],
        ).unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, push_constants)
            .unwrap()
            .dispatch([hidden_val.div_ceil(256), 1, 1])
            .unwrap();
    }
}
