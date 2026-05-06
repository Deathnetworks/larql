//! Q4_K matvec dispatch for Vulkan.

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;

pub fn dispatch(
    backend: &VulkanBackend,
    w4k: &[u8],
    x: &[f32],
    n: u32,
    k: u32,
) -> Option<Vec<f32>> {
    let device = backend.device();
    let queue = backend.queue();
    
    let shader = shaders::q4k_matvec::load(device.clone()).ok()?;
    let pipeline = VulkanBackend::create_compute_pipeline(device, &shader);

    let mut out = vec![0.0f32; n as usize];
    
    let w_buf = VulkanBuffer::from_slice(device.clone(), w4k, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let x_buf = VulkanBuffer::from_slice(device.clone(), x, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let out_buf = VulkanBuffer::new(device.clone(), out.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let layout = pipeline.layout().set_layouts().get(0)?;
    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, w_buf.inner().clone()),
            WriteDescriptorSet::buffer(1, x_buf.inner().clone()),
            WriteDescriptorSet::buffer(2, out_buf.inner().clone()),
        ],
        [],
    ).ok()?;

    let push_constants = shaders::q4k_matvec::PushConstants { N: n, K: k };

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
        .dispatch([(n + 255) / 256, 1, 1]) // Simplified dispatch
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
