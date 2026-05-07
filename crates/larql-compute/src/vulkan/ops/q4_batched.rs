//! Batched Q4 operations for Vulkan.

use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PipelineBindPoint};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::buffers::VulkanBuffer;
use crate::vulkan::shaders;
use super::q4_common::quantize_to_q8;

/// Batched gate+up for ALL seq positions in ONE GPU submission.
pub fn pair_batch(
    backend: &VulkanBackend,
    gate_q4: &[u8],
    up_q4: &[u8],
    x_matrix: &[f32],
    seq_len: usize,
    num_rows: usize,
    hidden: usize,
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let kernel = backend.q4_matvec_pipeline.clone();
    let pipeline = &kernel.pipeline;
    let queue = backend.queue();

    let mut builder = AutoCommandBufferBuilder::primary(
        backend.command_buffer_allocator(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).ok()?;

    let buf_gate = VulkanBuffer::from_slice(backend, gate_q4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_up = VulkanBuffer::from_slice(backend, up_q4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    
    let mut gate_results_bufs = Vec::with_capacity(seq_len);
    let mut up_results_bufs = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        let x_slice = &x_matrix[s * hidden..(s + 1) * hidden];
        let (q8_x, q8_scales) = quantize_to_q8(x_slice);
        
        let x8_u32: Vec<u32> = unsafe {
            let (prefix, middle, suffix) = q8_x.align_to::<u32>();
            assert!(prefix.is_empty() && suffix.is_empty());
            middle.to_vec()
        };

        let x_buf = VulkanBuffer::from_slice(backend, &x8_u32, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let sx_buf = VulkanBuffer::from_slice(backend, &q8_scales, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let out_g = VulkanBuffer::new(backend, num_rows * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let out_u = VulkanBuffer::new(backend, num_rows * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        // Gate
        let layout = pipeline.layout().set_layouts().get(0)?;
        let set_g = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_gate.inner().clone()),
                WriteDescriptorSet::buffer(1, x_buf.inner().clone()),
                WriteDescriptorSet::buffer(2, sx_buf.inner().clone()),
                WriteDescriptorSet::buffer(3, out_g.inner().clone()),
            ],
            [],
        ).ok()?;

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set_g)
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, shaders::q4_matvec::PushConstants { N: num_rows as u32, K: hidden as u32 })
            .unwrap()
            .dispatch([num_rows as u32 / kernel.rows_per_tg, 1, 1])
            .unwrap();

        // Up
        let set_u = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_up.inner().clone()),
                WriteDescriptorSet::buffer(1, x_buf.inner().clone()),
                WriteDescriptorSet::buffer(2, sx_buf.inner().clone()),
                WriteDescriptorSet::buffer(3, out_u.inner().clone()),
            ],
            [],
        ).ok()?;

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set_u)
            .unwrap()
            .push_constants(pipeline.layout().clone(), 0, shaders::q4_matvec::PushConstants { N: num_rows as u32, K: hidden as u32 })
            .unwrap()
            .dispatch([num_rows as u32 / kernel.rows_per_tg, 1, 1])
            .unwrap();

        gate_results_bufs.push(out_g);
        up_results_bufs.push(out_u);
    }

    let command_buffer = builder.build().ok()?;
    vulkano::sync::now(backend.device().clone())
        .then_execute(queue.clone(), command_buffer).ok()?
        .then_signal_fence_and_flush().ok()?
        .wait(None).ok()?;

    let mut gate_results = Vec::with_capacity(seq_len);
    let mut up_results = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        let mut g = vec![0.0f32; num_rows];
        let mut u = vec![0.0f32; num_rows];
        gate_results_bufs[s].copy_to_slice(&mut g);
        up_results_bufs[s].copy_to_slice(&mut u);
        gate_results.push(g);
        up_results.push(u);
    }

    Some((gate_results, up_results))
}

/// Multi-layer Q4 FFN in ONE command buffer.
pub fn multi_layer_ffn(
    backend: &VulkanBackend,
    layers_q4: &[(&[u8], &[u8], &[u8])], // [(gate, up, down_t)]
    x: &[f32],
    inter: usize,
    hidden: usize,
) -> Option<Vec<f32>> {
    let kernel = backend.q4_matvec_pipeline.clone();
    let f32_kernel = backend.q4_f32_matvec_pipeline.clone();
    let queue = backend.queue();

    let (q8_init, q8s_init) = quantize_to_q8(x);
    
    let mut builder = AutoCommandBufferBuilder::primary(
        backend.command_buffer_allocator(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).ok()?;

    // Pre-allocate buffers for ALL layers to keep them on GPU
    let mut q8_bufs = Vec::with_capacity(layers_q4.len() + 1);
    let mut q8s_bufs = Vec::with_capacity(layers_q4.len() + 1);
    
    let x8_u32: Vec<u32> = unsafe {
        let (prefix, middle, suffix) = q8_init.align_to::<u32>();
        assert!(prefix.is_empty() && suffix.is_empty());
        middle.to_vec()
    };
    q8_bufs.push(VulkanBuffer::from_slice(backend, &x8_u32, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);
    q8s_bufs.push(VulkanBuffer::from_slice(backend, &q8s_init, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);

    let mut gate_outs = Vec::with_capacity(layers_q4.len());
    let mut up_outs = Vec::with_capacity(layers_q4.len());
    let mut act_outs = Vec::with_capacity(layers_q4.len());
    let mut down_outs = Vec::with_capacity(layers_q4.len());

    for _ in 0..layers_q4.len() {
        gate_outs.push(VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);
        up_outs.push(VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);
        act_outs.push(VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);
        down_outs.push(VulkanBuffer::new(backend, hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);
        q8_bufs.push(VulkanBuffer::new(backend, hidden, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);
        q8s_bufs.push(VulkanBuffer::new(backend, (hidden / 32) * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?);
    }

    for (l, (gate_w, up_w, down_w)) in layers_q4.iter().enumerate() {
        let buf_gate = VulkanBuffer::from_slice(backend, gate_w, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_up = VulkanBuffer::from_slice(backend, up_w, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_down = VulkanBuffer::from_slice(backend, down_w, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        // 1. Gate MatVec
        let layout = kernel.pipeline.layout().set_layouts().get(0)?;
        let set_g = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_gate.inner().clone()),
                WriteDescriptorSet::buffer(1, q8_bufs[l].inner().clone()),
                WriteDescriptorSet::buffer(2, q8s_bufs[l].inner().clone()),
                WriteDescriptorSet::buffer(3, gate_outs[l].inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(kernel.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, kernel.pipeline.layout().clone(), 0, set_g).unwrap()
            .push_constants(kernel.pipeline.layout().clone(), 0, shaders::q4_matvec::PushConstants { N: inter as u32, K: hidden as u32 })
            .unwrap().dispatch([inter as u32 / kernel.rows_per_tg, 1, 1]).unwrap();

        // 2. Up MatVec
        let set_u = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_up.inner().clone()),
                WriteDescriptorSet::buffer(1, q8_bufs[l].inner().clone()),
                WriteDescriptorSet::buffer(2, q8s_bufs[l].inner().clone()),
                WriteDescriptorSet::buffer(3, up_outs[l].inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(kernel.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, kernel.pipeline.layout().clone(), 0, set_u).unwrap()
            .push_constants(kernel.pipeline.layout().clone(), 0, shaders::q4_matvec::PushConstants { N: inter as u32, K: hidden as u32 })
            .unwrap().dispatch([inter as u32 / kernel.rows_per_tg, 1, 1]).unwrap();

        // 3. GEGLU Activation (Fused)
        let set_act = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            backend.geglu_pipeline.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, gate_outs[l].inner().clone()),
                WriteDescriptorSet::buffer(1, up_outs[l].inner().clone()),
                WriteDescriptorSet::buffer(2, act_outs[l].inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(backend.geglu_pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, backend.geglu_pipeline.layout().clone(), 0, set_act).unwrap()
            .push_constants(backend.geglu_pipeline.layout().clone(), 0, shaders::geglu::PushConstants { N: inter as u32 })
            .unwrap().dispatch([(inter as u32).div_ceil(256), 1, 1]).unwrap();

        // 4. Down MatVec (f32 input)
        let layout_f32 = f32_kernel.pipeline.layout().set_layouts().get(0)?;
        let set_d = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            layout_f32.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_down.inner().clone()),
                WriteDescriptorSet::buffer(1, act_outs[l].inner().clone()),
                WriteDescriptorSet::buffer(2, down_outs[l].inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(f32_kernel.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, f32_kernel.pipeline.layout().clone(), 0, set_d).unwrap()
            .push_constants(f32_kernel.pipeline.layout().clone(), 0, shaders::q4_f32_matvec::PushConstants { N: hidden as u32, K: inter as u32 })
            .unwrap().dispatch([hidden as u32 / f32_kernel.rows_per_tg, 1, 1]).unwrap();

        // 5. Q8 Quantize for next layer
        if l + 1 < layers_q4.len() {
             let set_q8 = PersistentDescriptorSet::new(
                backend.descriptor_set_allocator(),
                backend.quantize_q8_pipeline.layout().set_layouts().get(0)?.clone(),
                [
                    WriteDescriptorSet::buffer(0, down_outs[l].inner().clone()),
                    WriteDescriptorSet::buffer(1, q8_bufs[l+1].inner().clone()),
                    WriteDescriptorSet::buffer(2, q8s_bufs[l+1].inner().clone()),
                ],
                [],
            ).ok()?;

            builder.bind_pipeline_compute(backend.quantize_q8_pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, backend.quantize_q8_pipeline.layout().clone(), 0, set_q8).unwrap()
                .push_constants(backend.quantize_q8_pipeline.layout().clone(), 0, shaders::quantize_q8::PushConstants { N: hidden as u32 })
                .unwrap().dispatch([(hidden as u32).div_ceil(256), 1, 1]).unwrap();
        }
    }

    let command_buffer = builder.build().ok()?;
    vulkano::sync::now(backend.device().clone())
        .then_execute(queue.clone(), command_buffer).ok()?
        .then_signal_fence_and_flush().ok()?
        .wait(None).ok()?;

    let mut final_out = vec![0.0f32; hidden];
    down_outs.last()?.copy_to_slice(&mut final_out);
    Some(final_out)
}
