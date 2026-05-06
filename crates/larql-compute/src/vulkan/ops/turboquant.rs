//! TurboQuant dispatch for Vulkan.

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;

pub fn encode_dispatch(
    backend: &VulkanBackend,
    input: &[f32],
    d: u32,
    batch: u32,
) -> Option<(Vec<f32>, Vec<u8>)> {
    let device = backend.device();
    let queue = backend.queue();
    
    let shader = shaders::turboquant_encode::load(device.clone()).ok()?;
    let pipeline = VulkanBackend::create_compute_pipeline(device, &shader);

    let mut norms = vec![0.0f32; batch as usize];
    let mut packed = vec![0u8; (batch * d / 2) as usize];
    
    let input_buf = VulkanBuffer::from_slice(device.clone(), input, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let norms_buf = VulkanBuffer::new(device.clone(), norms.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let packed_buf = VulkanBuffer::new(device.clone(), packed.len(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let layout = pipeline.layout().set_layouts().get(0)?;
    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, input_buf.inner().clone()),
            WriteDescriptorSet::buffer(1, norms_buf.inner().clone()),
            WriteDescriptorSet::buffer(2, packed_buf.inner().clone()),
        ],
        [],
    ).ok()?;

    let push_constants = shaders::turboquant_encode::PushConstants { d, batch };

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
        .dispatch([batch, 1, 1])
        .unwrap();

    let command_buffer = builder.build().ok()?;
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .ok()?
        .then_signal_fence_and_flush()
        .ok()?
        .wait(None)
        .ok()?;

    norms_buf.copy_to_slice(&mut norms);
    packed_buf.copy_to_slice(&mut packed);
    Some((norms, packed))
}

pub fn decode_dispatch(
    backend: &VulkanBackend,
    norms: &[f32],
    packed: &[u8],
    d: u32,
    batch: u32,
) -> Option<Vec<f32>> {
    let device = backend.device();
    let queue = backend.queue();
    
    let shader = shaders::turboquant_decode::load(device.clone()).ok()?;
    let pipeline = VulkanBackend::create_compute_pipeline(device, &shader);

    let mut output = vec![0.0f32; (batch * d) as usize];
    
    let norms_buf = VulkanBuffer::from_slice(device.clone(), norms, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let packed_buf = VulkanBuffer::from_slice(device.clone(), packed, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let output_buf = VulkanBuffer::new(device.clone(), output.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let layout = pipeline.layout().set_layouts().get(0)?;
    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, norms_buf.inner().clone()),
            WriteDescriptorSet::buffer(1, packed_buf.inner().clone()),
            WriteDescriptorSet::buffer(2, output_buf.inner().clone()),
        ],
        [],
    ).ok()?;

    let push_constants = shaders::turboquant_decode::PushConstants { d, batch };

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
        .dispatch([batch, 1, 1])
        .unwrap();

    let command_buffer = builder.build().ok()?;
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .ok()?
        .then_signal_fence_and_flush()
        .ok()?
        .wait(None)
        .ok()?;

    output_buf.copy_to_slice(&mut output);
    Some(output)
}
