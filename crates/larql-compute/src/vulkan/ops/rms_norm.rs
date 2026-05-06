//! RMSNorm dispatch for Vulkan.

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;

pub fn dispatch(
    backend: &VulkanBackend,
    x: &[f32],
    weight: &[f32],
    eps: f32,
) -> Option<Vec<f32>> {
    let device = backend.device();
    let queue = backend.queue();
    
    let shader = shaders::rms_norm::load(device.clone()).ok()?;
    let pipeline = VulkanBackend::create_compute_pipeline(device, &shader);

    let mut out = vec![0.0f32; x.len()];
    let len = x.len() as u32;
    
    let x_buf = VulkanBuffer::from_slice(device.clone(), x, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let w_buf = VulkanBuffer::from_slice(device.clone(), weight, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let out_buf = VulkanBuffer::new(device.clone(), out.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

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
        .dispatch([1, 1, 1]) // One WG for the whole vector
        .unwrap();

    let command_buffer = builder.build().ok()?;
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .ok()?
        .then_signal_fence_and_flush()
        .ok()?
        .wait(None)
        .ok()?;

    out_buf.copy_to_slice(&mut out);
    Some(out)
}
