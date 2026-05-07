//! Q4_0 vecmat dispatch for Vulkan.

use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PipelineBindPoint};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::buffers::VulkanBuffer;
use crate::vulkan::shaders;

/// Dispatch a single Q4_0 vecmat (x @ W) on Vulkan GPU.
pub fn dispatch(
    backend: &VulkanBackend,
    x: &[f32],
    w4: &[u8],
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    let kernel = backend.q4_vecmat_pipeline.clone();
    let pipeline = &kernel.pipeline;
    let queue = backend.queue();

    let mut out = vec![0.0f32; k];
    
    let x_buf = VulkanBuffer::from_slice(backend, x, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let w_buf = VulkanBuffer::from_slice(backend, w4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let out_buf = VulkanBuffer::new(backend, k * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let layout = pipeline.layout().set_layouts().get(0)?;
    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, x_buf.inner().clone()),
            WriteDescriptorSet::buffer(1, w_buf.inner().clone()),
            WriteDescriptorSet::buffer(2, out_buf.inner().clone()),
        ],
        [],
    ).ok()?;

    let push_constants = shaders::q4_vecmat::PushConstants { 
        N: n as u32, 
        K: k as u32 
    };

    let mut builder = AutoCommandBufferBuilder::primary(
        backend.command_buffer_allocator(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).ok()?;

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
        .dispatch([k as u32 / 256 + 1, 1, 1]) // Vecmat is typically 1 thread per output column
        .unwrap();

    let command_buffer = builder.build().ok()?;
    let _ = vulkano::sync::now(backend.device().clone())
        .then_execute(queue.clone(), command_buffer).ok()?
        .then_signal_fence_and_flush().ok()?
        .wait(None);

    out_buf.copy_to_slice(&mut out);
    Some(out)
}
