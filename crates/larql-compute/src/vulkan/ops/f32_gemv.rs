//! F32 matrix-vector dispatch for Vulkan.

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;

pub fn dispatch(
    backend: &VulkanBackend,
    w: &[f32],
    x: &[f32],
    n: u32,
    k: u32,
) -> Option<Vec<f32>> {
    let device = backend.device();
    let queue = backend.queue();
    println!("Vulkan Gemv: n={}, k={}", n, k);
    
    let pipeline = backend.f32_gemv_pipeline.clone();
    println!("Pipeline retrieved.");

    let mut out = vec![0.0f32; n as usize];
    
    println!("Creating buffers...");
    let w_buf = VulkanBuffer::from_slice(backend, w, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let x_buf = VulkanBuffer::from_slice(backend, x, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let out_buf = VulkanBuffer::new(backend, out.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    println!("Buffers created.");

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    struct PushConstants {
        n: u32,
        k: u32,
    }

    let push_constants = PushConstants { n, k };

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
        .dispatch([n, 1, 1])
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
