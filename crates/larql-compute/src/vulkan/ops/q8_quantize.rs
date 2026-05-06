//! Q8 quantization dispatch for Vulkan.

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;

pub fn dispatch(
    backend: &VulkanBackend,
    input: &[f32],
    k: u32,
) -> Option<(Vec<i8>, Vec<f32>)> {
    let device = backend.device();
    let queue = backend.queue();
    
    let shader = shaders::quantize_q8::load(device.clone()).ok()?;
    let pipeline = VulkanBackend::create_compute_pipeline(device, &shader);

    let num_blocks = (k / 32) as usize;
    let mut q8_out = vec![0i8; k as usize];
    let mut scales = vec![0.0f32; num_blocks];
    
    let input_buf = VulkanBuffer::from_slice(device.clone(), input, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let q8_out_buf = VulkanBuffer::new(device.clone(), q8_out.len(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let scales_buf = VulkanBuffer::new(device.clone(), scales.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let layout = pipeline.layout().set_layouts().get(0)?;
    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, input_buf.inner().clone()),
            WriteDescriptorSet::buffer(1, q8_out_buf.inner().clone()),
            WriteDescriptorSet::buffer(2, scales_buf.inner().clone()),
        ],
        [],
    ).ok()?;

    let push_constants = shaders::quantize_q8::PushConstants { K: k };

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
        .dispatch([num_blocks as u32, 1, 1])
        .unwrap();

    let command_buffer = builder.build().ok()?;
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .ok()?
        .then_signal_fence_and_flush()
        .ok()?
        .wait(None)
        .ok()?;

    q8_out_buf.copy_to_slice(&mut q8_out);
    scales_buf.copy_to_slice(&mut scales);
    Some((q8_out, scales))
}
