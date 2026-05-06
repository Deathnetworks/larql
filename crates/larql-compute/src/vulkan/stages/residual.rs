//! Post-attention and post-FFN residual + norm fusions for Vulkan.
//!
//! Two block-level helpers that sit between the matmul-heavy stages:
//!
//! - [`encode_post_attn`] fuses the post-attention residual add, the
//!   pre-FFN RMS norm, and (for Q4_0 / Q8_0 FFN) the Q8 quantisation of
//!   the norm output.
//! - [`encode_post_ffn`] fuses the post-FFN residual add with the
//!   optional post-FFN RMS norm (Gemma post-norm architectures).
//!
//! Caller owns the AutoCommandBufferBuilder lifecycle.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;

/// Post-attention residual + pre-FFN norm (+ optional Q8 quant).
#[allow(clippy::too_many_arguments)]
pub fn encode_post_attn(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    rms_norm_pipeline: &Arc<ComputePipeline>,
    residual_add_pipeline: &Arc<ComputePipeline>,
    q8_quant_pipeline: &Arc<ComputePipeline>,
    scratch_alloc: &mut dyn FnMut(u64) -> Subbuffer<[u8]>,
    h_buf: &Subbuffer<[u8]>,
    o_out: &Subbuffer<[u8]>,
    h_post_attn: &Subbuffer<[u8]>,
    ffn_norm_out: &Subbuffer<[u8]>,
    post_attn_norm_buf: &Subbuffer<[u8]>,
    pre_ffn_weight_buf: &Subbuffer<[u8]>,
    ffn_q8_buf: &Subbuffer<[u8]>,
    ffn_q8s_buf: &Subbuffer<[u8]>,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
    has_post_norms: bool,
    ffn_needs_q8: bool,
    h_stride_bytes: u64,
    q8_stride_bytes: u64,
    q8s_stride_bytes: u64,
) {
    let hidden_val = hidden as u32;
    let tg_threads = 256.min(hidden as u32);
    let blocks = hidden_val.div_ceil(tg_threads);

    let rms_layout = rms_norm_pipeline.layout().set_layouts().get(0).unwrap();
    let res_layout = residual_add_pipeline.layout().set_layouts().get(0).unwrap();
    let q8_layout = q8_quant_pipeline.layout().set_layouts().get(0).unwrap();

    let hidden_bytes = (hidden * 4) as u64;

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;

        let h_slice = h_buf.clone().slice(h_off .. h_off + hidden_bytes);
        let o_slice = o_out.clone().slice(h_off .. h_off + hidden_bytes);
        let h_post_slice = h_post_attn.clone().slice(h_off .. h_off + hidden_bytes);

        if has_post_norms {
            // Post-norm: norm(O) first, then residual add.
            let normed = scratch_alloc(hidden_bytes);

            let rms_set = PersistentDescriptorSet::new(
                backend.descriptor_set_allocator.clone(),
                rms_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, o_slice.clone()),
                    WriteDescriptorSet::buffer(1, post_attn_norm_buf.clone()),
                    WriteDescriptorSet::buffer(2, normed.clone()),
                ],
                [],
            ).unwrap();

            let rms_pcs = shaders::rms_norm::PushConstants {
                len: hidden_val,
                eps,
                offset: norm_offset,
            };

            builder
                .bind_pipeline_compute(rms_norm_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, rms_norm_pipeline.layout().clone(), 0, rms_set)
                .unwrap()
                .push_constants(rms_norm_pipeline.layout().clone(), 0, rms_pcs)
                .unwrap()
                .dispatch([1, 1, 1]) // Metal used (1,1,1) grid and tg_threads block, meaning only 1 block. Let's map to Vulkan correctly: if we need hidden threads, grid is div_ceil.
                .unwrap();
            // Actually Metal used dispatch_thread_groups(MTLSize::new(1,1,1), MTLSize::new(tg_threads,1,1)). 
            // Wait, if it only dispatched 1 threadgroup, then only 256 threads run. If hidden > 256, it relies on a loop inside the shader.
            // In Vulkan, the grid is [1, 1, 1] if the shader implements the loop, which rms_norm.comp usually does.

            let res_set = PersistentDescriptorSet::new(
                backend.descriptor_set_allocator.clone(),
                res_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, h_slice.clone()),
                    WriteDescriptorSet::buffer(1, normed.clone()),
                    WriteDescriptorSet::buffer(2, h_post_slice.clone()),
                ],
                [],
            ).unwrap();

            let res_pcs = shaders::residual_ops::PushConstants {
                len: hidden_val,
                scalar: 0.0,
                mode: 1, // add
            };

            builder
                .bind_pipeline_compute(residual_add_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, residual_add_pipeline.layout().clone(), 0, res_set)
                .unwrap()
                .push_constants(residual_add_pipeline.layout().clone(), 0, res_pcs)
                .unwrap()
                .dispatch([blocks, 1, 1]) // Metal used dispatch_threads, meaning 1 thread per element.
                .unwrap();
        } else {
            // Pre-norm: residual add first (h + O).
            let res_set = PersistentDescriptorSet::new(
                backend.descriptor_set_allocator.clone(),
                res_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, h_slice.clone()),
                    WriteDescriptorSet::buffer(1, o_slice.clone()),
                    WriteDescriptorSet::buffer(2, h_post_slice.clone()),
                ],
                [],
            ).unwrap();

            let res_pcs = shaders::residual_ops::PushConstants {
                len: hidden_val,
                scalar: 0.0,
                mode: 1, // add
            };

            builder
                .bind_pipeline_compute(residual_add_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, residual_add_pipeline.layout().clone(), 0, res_set)
                .unwrap()
                .push_constants(residual_add_pipeline.layout().clone(), 0, res_pcs)
                .unwrap()
                .dispatch([blocks, 1, 1])
                .unwrap();
        }

        // Pre-FFN rms_norm on h_post_attn → ffn_norm_out.
        let ffn_norm_slice = ffn_norm_out.clone().slice(h_off .. h_off + hidden_bytes);

        let rms_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator.clone(),
            rms_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, h_post_slice.clone()),
                WriteDescriptorSet::buffer(1, pre_ffn_weight_buf.clone()),
                WriteDescriptorSet::buffer(2, ffn_norm_slice.clone()),
            ],
            [],
        ).unwrap();

        let rms_pcs = shaders::rms_norm::PushConstants {
            len: hidden_val,
            eps,
            offset: norm_offset,
        };

        builder
            .bind_pipeline_compute(rms_norm_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, rms_norm_pipeline.layout().clone(), 0, rms_set)
            .unwrap()
            .push_constants(rms_norm_pipeline.layout().clone(), 0, rms_pcs)
            .unwrap()
            .dispatch([1, 1, 1]) // Loop inside shader
            .unwrap();

        // Q8-quantise
        if ffn_needs_q8 {
            let q8_slice = ffn_q8_buf.clone().slice(q8_off .. q8_off + hidden_val as u64); // i8 uses 1 byte
            // Q8 blocks have scales (f32). Hidden / 32 blocks.
            let q8s_bytes = (hidden_val / 32 * 4) as u64; 
            let q8s_slice = ffn_q8s_buf.clone().slice(q8s_off .. q8s_off + q8s_bytes);

            let q8_set = PersistentDescriptorSet::new(
                backend.descriptor_set_allocator.clone(),
                q8_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, ffn_norm_slice.clone()),
                    WriteDescriptorSet::buffer(1, q8_slice.clone()),
                    WriteDescriptorSet::buffer(2, q8s_slice.clone()),
                ],
                [],
            ).unwrap();

            let q8_pcs = shaders::quantize_q8::PushConstants {
                len: hidden_val,
            };

            builder
                .bind_pipeline_compute(q8_quant_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, q8_quant_pipeline.layout().clone(), 0, q8_set)
                .unwrap()
                .push_constants(q8_quant_pipeline.layout().clone(), 0, q8_pcs)
                .unwrap()
                .dispatch([hidden_val.div_ceil(32) as u32, 1, 1]) // One thread per block (32 elements)
                .unwrap();
        }
    }
}

/// Post-FFN residual + optional post-FFN RMS norm.
#[allow(clippy::too_many_arguments)]
pub fn encode_post_ffn(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    rms_norm_pipeline: &Arc<ComputePipeline>,
    residual_add_pipeline: &Arc<ComputePipeline>,
    scratch_alloc: &mut dyn FnMut(u64) -> Subbuffer<[u8]>,
    down_out: &Subbuffer<[u8]>,
    h_post_attn: &Subbuffer<[u8]>,
    h_next: &Subbuffer<[u8]>,
    post_ffn_norm_buf: Option<&Subbuffer<[u8]>>,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
    has_post_norms: bool,
    h_stride_bytes: u64,
) {
    let hidden_val = hidden as u32;
    let tg_threads = 256.min(hidden as u32);
    let blocks = hidden_val.div_ceil(tg_threads);

    let rms_layout = rms_norm_pipeline.layout().set_layouts().get(0).unwrap();
    let res_layout = residual_add_pipeline.layout().set_layouts().get(0).unwrap();
    let hidden_bytes = (hidden * 4) as u64;

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;

        let down_slice = down_out.clone().slice(h_off .. h_off + hidden_bytes);
        let h_post_slice = h_post_attn.clone().slice(h_off .. h_off + hidden_bytes);
        let h_next_slice = h_next.clone().slice(h_off .. h_off + hidden_bytes);

        if has_post_norms {
            if let Some(post_ffn_buf) = post_ffn_norm_buf {
                let normed = scratch_alloc(hidden_bytes);

                let rms_set = PersistentDescriptorSet::new(
                    backend.descriptor_set_allocator.clone(),
                    rms_layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, down_slice.clone()),
                        WriteDescriptorSet::buffer(1, post_ffn_buf.clone()),
                        WriteDescriptorSet::buffer(2, normed.clone()),
                    ],
                    [],
                ).unwrap();

                let rms_pcs = shaders::rms_norm::PushConstants {
                    len: hidden_val,
                    eps,
                    offset: norm_offset,
                };

                builder
                    .bind_pipeline_compute(rms_norm_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(PipelineBindPoint::Compute, rms_norm_pipeline.layout().clone(), 0, rms_set)
                    .unwrap()
                    .push_constants(rms_norm_pipeline.layout().clone(), 0, rms_pcs)
                    .unwrap()
                    .dispatch([1, 1, 1])
                    .unwrap();

                let res_set = PersistentDescriptorSet::new(
                    backend.descriptor_set_allocator.clone(),
                    res_layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, h_post_slice.clone()),
                        WriteDescriptorSet::buffer(1, normed.clone()),
                        WriteDescriptorSet::buffer(2, h_next_slice.clone()),
                    ],
                    [],
                ).unwrap();

                let res_pcs = shaders::residual_ops::PushConstants {
                    len: hidden_val,
                    scalar: 0.0,
                    mode: 1, // add
                };

                builder
                    .bind_pipeline_compute(residual_add_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(PipelineBindPoint::Compute, residual_add_pipeline.layout().clone(), 0, res_set)
                    .unwrap()
                    .push_constants(residual_add_pipeline.layout().clone(), 0, res_pcs)
                    .unwrap()
                    .dispatch([blocks, 1, 1])
                    .unwrap();
                continue;
            }
        }

        // Pre-norm or post-norm-without-post_ffn_norm: plain residual.
        let res_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator.clone(),
            res_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, h_post_slice.clone()),
                WriteDescriptorSet::buffer(1, down_slice.clone()),
                WriteDescriptorSet::buffer(2, h_next_slice.clone()),
            ],
            [],
        ).unwrap();

        let res_pcs = shaders::residual_ops::PushConstants {
            len: hidden_val,
            scalar: 0.0,
            mode: 1, // add
        };

        builder
            .bind_pipeline_compute(residual_add_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, residual_add_pipeline.layout().clone(), 0, res_set)
            .unwrap()
            .push_constants(residual_add_pipeline.layout().clone(), 0, res_pcs)
            .unwrap()
            .dispatch([blocks, 1, 1])
            .unwrap();
    }
}
