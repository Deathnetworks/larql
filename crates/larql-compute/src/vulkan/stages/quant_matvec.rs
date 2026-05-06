//! Format-aware single-vector matvec dispatch for Vulkan.
//!
//! One entry point, `encode`, that routes to the right shader based on the
//! weight's quantization format.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;

/// Vulkan shader pipelines this stage may dispatch, in one bundle.
pub struct Pipelines<'a> {
    pub q4k_matvec: &'a Arc<ComputePipeline>,
    pub q6k_matvec: &'a Arc<ComputePipeline>,
    pub q4_vecmat: &'a Arc<ComputePipeline>,
}

/// Single-vector matvec dispatch.
#[allow(clippy::too_many_arguments)]
fn dispatch_vulkan(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    w_buf: &Subbuffer<[u32]>,
    f32_in: &Subbuffer<[f32]>,
    out_buf: &Subbuffer<[f32]>,
    n: u32,
    k: u32,
    rows_per_tg: u32,
) {
    let num_tgs = n.div_ceil(rows_per_tg);
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator.clone(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, w_buf.clone()),
            WriteDescriptorSet::buffer(1, f32_in.clone()),
            WriteDescriptorSet::buffer(2, out_buf.clone()),
        ],
        [],
    ).unwrap();

    // The push constants for q4k_matvec and q6k_matvec
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct PushConstants {
        n: u32,
        k: u32,
    }

    let pcs = PushConstants { n, k };

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, pcs)
        .unwrap()
        .dispatch([num_tgs, 1, 1])
        .unwrap();
}

/// Single-vector matvec dispatch for Q4_0 / Q8_0 which use Q8 inputs.
#[allow(clippy::too_many_arguments)]
fn dispatch_q8_input(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    w_buf: &Subbuffer<[u32]>,
    q8_in: &Subbuffer<[i8]>,
    q8s_in: &Subbuffer<[f32]>,
    out_buf: &Subbuffer<[f32]>,
    n: u32,
    k: u32,
    rows_per_tg: u32,
) {
    let num_tgs = n.div_ceil(rows_per_tg);
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator.clone(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, w_buf.clone()),
            WriteDescriptorSet::buffer(1, q8_in.clone()),
            WriteDescriptorSet::buffer(2, q8s_in.clone()),
            WriteDescriptorSet::buffer(3, out_buf.clone()),
        ],
        [],
    ).unwrap();

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct PushConstants {
        n: u32,
        k: u32,
    }

    let pcs = PushConstants { n, k };

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, pcs)
        .unwrap()
        .dispatch([num_tgs, 1, 1])
        .unwrap();
}

/// Encode a single-vector matvec `out[N] = W[N×K] · x[K]`.
///
/// Does not call `end_encoding` — the caller owns the AutoCommandBufferBuilder lifecycle.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    format: crate::QuantFormat,
    w_buf: &Subbuffer<[u32]>,
    f32_in: &Subbuffer<[f32]>,
    q8_in: &Subbuffer<[i8]>,
    q8s_in: &Subbuffer<[f32]>,
    out_buf: &Subbuffer<[f32]>,
    pipes: &Pipelines<'_>,
    num_rows: usize,
    hidden: usize,
) {
    let n = num_rows as u32;
    let k = hidden as u32;

    match format {
        crate::QuantFormat::Q4_KF | crate::QuantFormat::Q4_K => {
            // q4k_matvec in Vulkan uses 4 rows per threadgroup
            dispatch_vulkan(builder, backend, pipes.q4k_matvec, w_buf, f32_in, out_buf, n, k, 4);
        }
        crate::QuantFormat::Q6_K => {
            // q6k_matvec in Vulkan uses 4 rows per threadgroup
            dispatch_vulkan(builder, backend, pipes.q6k_matvec, w_buf, f32_in, out_buf, n, k, 4);
        }
        crate::QuantFormat::Q4_0 | crate::QuantFormat::Q8_0 => {
            // q4_vecmat expected to use Q8 inputs. We assume 4 rows per tg.
            dispatch_q8_input(builder, backend, pipes.q4_vecmat, w_buf, q8_in, q8s_in, out_buf, n, k, 4);
        }
        crate::QuantFormat::BF16 | crate::QuantFormat::F16 | crate::QuantFormat::F32 => {
            // Handled elsewhere
        }
    }
}
