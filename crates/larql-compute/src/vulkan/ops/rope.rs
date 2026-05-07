//! RoPE dispatch for Vulkan.

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;

pub fn dispatch(
    backend: &VulkanBackend,
    q: &mut [f32],
    k: &mut [f32],
    head_dim: u32,
    rope_base: f32,
    pos: u32,
    rotary_dim: u32,
    num_q: u32,
    num_kv: u32,
) -> Option<()> {
    let device = backend.device();
    let queue = backend.queue();
    
    let shader = shaders::rope::load(device.clone()).ok()?;
    let pipeline = VulkanBackend::create_compute_pipeline(device, &shader);

    let total_heads = num_q + num_kv;
    let rdim = if rotary_dim == 0 { head_dim } else { rotary_dim.min(head_dim) };
    let hdim = rdim / 2;

    let q_buf = VulkanBuffer::from_slice(backend, q, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let k_buf = VulkanBuffer::from_slice(backend, k, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let layout = pipeline.layout().set_layouts().get(0)?;
    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, q_buf.inner().clone()),
            WriteDescriptorSet::buffer(1, k_buf.inner().clone()),
        ],
        [],
    ).ok()?;

    let push_constants = shaders::rope::PushConstants {
        head_dim,
        rope_base,
        pos,
        rotary_dim: rdim,
        num_q,
        num_kv,
        mode: 0,
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
        .dispatch([total_heads, hdim, 1])
        .unwrap();

    let command_buffer = builder.build().ok()?;
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .ok()?
        .then_signal_fence_and_flush()
        .ok()?
        .wait(None)
        .ok()?;

    q_buf.copy_to_slice(q);
    k_buf.copy_to_slice(k);
    Some(())
}
