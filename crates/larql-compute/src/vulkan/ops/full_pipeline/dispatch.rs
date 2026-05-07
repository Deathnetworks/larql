//! Full pipeline: ALL Q4 (attention + FFN) in ONE Vulkan command buffer.

use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PipelineBindPoint};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;
use std::sync::Arc;

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;
use crate::vulkan::ops::kv_cache::KVCache;

/// Run all layers in ONE Vulkan command buffer with correct norms and residuals.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_pipeline(
    backend: &VulkanBackend,
    kv_cache: &mut KVCache,
    layers: &[crate::FullPipelineLayer],
    x: &[f32],
    hidden: usize,
    inter: usize,
    seq_len: usize,
    softcap: f32,
) -> Option<Vec<f32>> {
    let queue = backend.queue();
    let device = backend.device();
    let num_layers = layers.len();

    // 1. Initial input buffer
    let mut current_h_buf = VulkanBuffer::from_slice(backend, x, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let mut builder = AutoCommandBufferBuilder::primary(
        backend.command_buffer_allocator(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).ok()?;

    // Pipelines
    let sgemm_pipe = backend.sgemm_transb_pipeline.clone();
    let sgemm_layout = sgemm_pipe.pipeline.layout().set_layouts().get(0)?;
    let attn_pipe = backend.attn_fused_pipeline.clone();
    let q8_pipe = backend.quantize_q8_pipeline.clone();
    let q4_pipe = backend.q4_matvec_pipeline.clone();
    let q4_layout = q4_pipe.pipeline.layout().set_layouts().get(0)?;
    let geglu_pipe = backend.geglu_pipeline.clone();
    let q4_f32_pipe = backend.q4_f32_matvec_pipeline.clone();
    let q4_f32_layout = q4_f32_pipe.pipeline.layout().set_layouts().get(0)?;
    let rms_pipe = backend.rms_norm_pipeline.clone();
    let res_ops_pipe = backend.residual_ops_pipeline.clone();

    for l in 0..num_layers {
        let layer = &layers[l];
        let kv_layer = &mut kv_cache.layers[l];

        // --- Per-layer intermediate buffers ---
        let buf_q = VulkanBuffer::new(backend, seq_len * layer.num_q_heads * layer.head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_k = VulkanBuffer::new(backend, seq_len * layer.num_kv_heads * layer.head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_v = VulkanBuffer::new(backend, seq_len * layer.num_kv_heads * layer.head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_attn_out = VulkanBuffer::new(backend, seq_len * layer.num_q_heads * layer.head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_o_out = VulkanBuffer::new(backend, seq_len * hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        
        let buf_gate_out = VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_up_out = VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_act_out = VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_ffn_out = VulkanBuffer::new(backend, hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        
        let h_norm_buf = VulkanBuffer::new(backend, seq_len * hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let h_post_attn_buf = VulkanBuffer::new(backend, seq_len * hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let h_post_attn_norm_buf = VulkanBuffer::new(backend, seq_len * hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let next_h_buf = VulkanBuffer::new(backend, seq_len * hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        // Q8 staging for FFN
        let buf_q8_x = VulkanBuffer::new(backend, seq_len * hidden, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_q8_s = VulkanBuffer::new(backend, seq_len * (hidden / 32) * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        // Weight buffers (Simplified: assuming they are available as slices)
        let buf_wq = VulkanBuffer::from_slice(backend, layer.wq.as_f32()?, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_wk = VulkanBuffer::from_slice(backend, layer.wk.as_f32()?, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_wv = VulkanBuffer::from_slice(backend, layer.wv.as_f32()?, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_wo = VulkanBuffer::from_slice(backend, layer.wo.as_f32()?, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        
        let buf_input_norm = VulkanBuffer::from_slice(backend, layer.input_norm_weight, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_post_attn_norm = VulkanBuffer::from_slice(backend, layer.post_attn_norm_weight.as_ref().unwrap_or(&layer.input_norm_weight), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        
        let buf_q_norm = VulkanBuffer::from_slice(backend, layer.q_norm_weight?, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_k_norm = VulkanBuffer::from_slice(backend, layer.k_norm_weight?, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        
        let buf_gate = VulkanBuffer::from_slice(backend, layer.gate.as_bytes(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_up = VulkanBuffer::from_slice(backend, layer.up.as_bytes(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let buf_down = VulkanBuffer::from_slice(backend, layer.down.as_bytes(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        // --- 1. Input Norm ---
        let set_in_norm = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            rms_pipe.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, current_h_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_input_norm.inner().clone()),
                WriteDescriptorSet::buffer(2, h_norm_buf.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(rms_pipe.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, rms_pipe.layout().clone(), 0, set_in_norm).unwrap()
            .push_constants(rms_pipe.layout().clone(), 0, shaders::rms_norm::PushConstants { 
                len: (seq_len * hidden) as u32, eps: layer.eps, offset: 0.0 
            })
            .unwrap().dispatch([1, 1, 1]).unwrap();

        // --- 2. QKV Projections ---
        for (buf_w, buf_out, out_dim) in [
            (&buf_wq, &buf_q, layer.num_q_heads * layer.head_dim),
            (&buf_wk, &buf_k, layer.num_kv_heads * layer.head_dim),
            (&buf_wv, &buf_v, layer.num_kv_heads * layer.head_dim)
        ] {
            let set = PersistentDescriptorSet::new(
                backend.descriptor_set_allocator(),
                sgemm_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, h_norm_buf.inner().clone()),
                    WriteDescriptorSet::buffer(1, buf_w.inner().clone()),
                    WriteDescriptorSet::buffer(2, buf_out.inner().clone()),
                ],
                [],
            ).ok()?;

            builder.bind_pipeline_compute(sgemm_pipe.pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, sgemm_pipe.pipeline.layout().clone(), 0, set).unwrap()
                .push_constants(sgemm_pipe.pipeline.layout().clone(), 0, shaders::sgemm_transb::PushConstants { 
                    M: seq_len as u32, N: out_dim as u32, K: hidden as u32 
                })
                .unwrap().dispatch([out_dim as u32 / sgemm_pipe.rows_per_tg, seq_len as u32, 1]).unwrap();
        }

        // --- 3. Fused Attention ---
        let attn_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            attn_pipe.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_q.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_k.inner().clone()),
                WriteDescriptorSet::buffer(2, buf_v.inner().clone()),
                WriteDescriptorSet::buffer(3, kv_layer.k_cache.inner().clone()),
                WriteDescriptorSet::buffer(4, kv_layer.v_cache.inner().clone()),
                WriteDescriptorSet::buffer(5, buf_attn_out.inner().clone()),
                WriteDescriptorSet::buffer(6, buf_q_norm.inner().clone()),
                WriteDescriptorSet::buffer(7, buf_k_norm.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(attn_pipe.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, attn_pipe.layout().clone(), 0, attn_set).unwrap()
            .push_constants(attn_pipe.layout().clone(), 0, shaders::attn_fused::PushConstants {
                T: (kv_layer.current_len + seq_len) as u32,
                head_dim: layer.head_dim as u32,
                num_q: layer.num_q_heads as u32,
                num_kv: layer.num_kv_heads as u32,
                scale: layer.attn_scale,
                window_size: 0,
                eps: layer.eps,
                qk_offset: 0.0,
                rope_base: layer.rope_base,
                rotary_dim: layer.head_dim as u32,
            })
            .unwrap().dispatch([layer.num_q_heads as u32, 1, 1]).unwrap();

        // --- 4. O Projection ---
        let set_o = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            sgemm_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_attn_out.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_wo.inner().clone()),
                WriteDescriptorSet::buffer(2, buf_o_out.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(sgemm_pipe.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, sgemm_pipe.pipeline.layout().clone(), 0, set_o).unwrap()
            .push_constants(sgemm_pipe.pipeline.layout().clone(), 0, shaders::sgemm_transb::PushConstants { 
                M: seq_len as u32, N: hidden as u32, K: (layer.num_q_heads * layer.head_dim) as u32 
            })
            .unwrap().dispatch([hidden as u32 / sgemm_pipe.rows_per_tg, seq_len as u32, 1]).unwrap();

        // --- 5. Residual Add + Post-Attn Norm ---
        let set_res_attn = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            res_ops_pipe.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, current_h_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_o_out.inner().clone()),
                WriteDescriptorSet::buffer(2, h_post_attn_buf.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(res_ops_pipe.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, res_ops_pipe.layout().clone(), 0, set_res_attn).unwrap()
            .push_constants(res_ops_pipe.layout().clone(), 0, shaders::residual_ops::PushConstants { 
                len: (seq_len * hidden) as u32, scalar: 1.0, mode: 1 // Add
            })
            .unwrap().dispatch([(seq_len * hidden) as u32 / 256 + 1, 1, 1]).unwrap();

        let set_post_attn_norm = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            rms_pipe.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, h_post_attn_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_post_attn_norm.inner().clone()),
                WriteDescriptorSet::buffer(2, h_post_attn_norm_buf.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(rms_pipe.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, rms_pipe.layout().clone(), 0, set_post_attn_norm).unwrap()
            .push_constants(rms_pipe.layout().clone(), 0, shaders::rms_norm::PushConstants { 
                len: (seq_len * hidden) as u32, eps: layer.eps, offset: 0.0 
            })
            .unwrap().dispatch([1, 1, 1]).unwrap();

        // --- 6. FFN ---
        // Quantize
        let q8_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            q8_pipe.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, h_post_attn_norm_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_q8_x.inner().clone()),
                WriteDescriptorSet::buffer(2, buf_q8_s.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(q8_pipe.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, q8_pipe.layout().clone(), 0, q8_set).unwrap()
            .push_constants(q8_pipe.layout().clone(), 0, shaders::quantize_q8::PushConstants { K: (seq_len * hidden) as u32 })
            .unwrap().dispatch([(seq_len * hidden) as u32 / 32, 1, 1]).unwrap();

        // Gate/Up
        for (buf_w, buf_out) in [(&buf_gate, &buf_gate_out), (&buf_up, &buf_up_out)] {
            let set = PersistentDescriptorSet::new(
                backend.descriptor_set_allocator(),
                q4_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, buf_w.inner().clone()),
                    WriteDescriptorSet::buffer(1, buf_q8_x.inner().clone()),
                    WriteDescriptorSet::buffer(2, buf_q8_s.inner().clone()),
                    WriteDescriptorSet::buffer(3, buf_out.inner().clone()),
                ],
                [],
            ).ok()?;

            builder.bind_pipeline_compute(q4_pipe.pipeline.clone()).unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, q4_pipe.pipeline.layout().clone(), 0, set).unwrap()
                .push_constants(q4_pipe.pipeline.layout().clone(), 0, shaders::q4_matvec::PushConstants { N: inter as u32, K: hidden as u32 })
                .unwrap().dispatch([inter as u32 / q4_pipe.rows_per_tg, 1, 1]).unwrap();
        }

        // Activation
        let geglu_set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            geglu_pipe.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_gate_out.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_up_out.inner().clone()),
                WriteDescriptorSet::buffer(2, buf_act_out.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(geglu_pipe.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, geglu_pipe.layout().clone(), 0, geglu_set).unwrap()
            .push_constants(geglu_pipe.layout().clone(), 0, shaders::geglu::PushConstants { N: inter as u32 })
            .unwrap().dispatch([(inter as u32).div_ceil(256), 1, 1]).unwrap();

        // Down
        let set_down = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            q4_f32_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_down.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_act_out.inner().clone()),
                WriteDescriptorSet::buffer(2, buf_ffn_out.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(q4_f32_pipe.pipeline.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, q4_f32_pipe.pipeline.layout().clone(), 0, set_down).unwrap()
            .push_constants(q4_f32_pipe.pipeline.layout().clone(), 0, shaders::q4_f32_matvec::PushConstants { N: hidden as u32, K: inter as u32 })
            .unwrap().dispatch([hidden as u32 / q4_f32_pipe.rows_per_tg, 1, 1]).unwrap();

        // Final Residual Add
        let set_res_ffn = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            res_ops_pipe.layout().set_layouts().get(0)?.clone(),
            [
                WriteDescriptorSet::buffer(0, h_post_attn_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, buf_ffn_out.inner().clone()),
                WriteDescriptorSet::buffer(2, next_h_buf.inner().clone()),
            ],
            [],
        ).ok()?;

        builder.bind_pipeline_compute(res_ops_pipe.clone()).unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, res_ops_pipe.layout().clone(), 0, set_res_ffn).unwrap()
            .push_constants(res_ops_pipe.layout().clone(), 0, shaders::residual_ops::PushConstants { 
                len: (seq_len * hidden) as u32, scalar: 1.0, mode: 1 // Add
            })
            .unwrap().dispatch([(seq_len * hidden) as u32 / 256 + 1, 1, 1]).unwrap();

        current_h_buf = next_h_buf;
        kv_layer.current_len += seq_len;
    }

    let command_buffer = builder.build().ok()?;
    vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer).ok()?
        .then_signal_fence_and_flush().ok()?
        .wait(None).ok()?;

    let mut result = vec![0.0f32; hidden];
    current_h_buf.copy_to_slice(&mut result);
    Some(result)
}
