//! Full layer pipeline: attention + FFN in one Vulkan command buffer.

use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PipelineBindPoint};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;
use std::sync::Arc;

use crate::vulkan::{VulkanBackend, shaders};
use crate::vulkan::buffers::VulkanBuffer;
use crate::vulkan::ops::kv_cache::LayerKVCache;
use crate::vulkan::ops::q4_common::quantize_to_q8;

/// Run a full transformer layer on Vulkan: attention + FFN, one command buffer.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    backend: &VulkanBackend,
    kv_cache: &mut LayerKVCache,
    // Attention weights (f32)
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    // FFN weights (Q4)
    gate_q4: &[u8],
    up_q4: &[u8],
    down_t_q4: &[u8],
    // Input
    x: &[f32],
    seq_len: usize,
    hidden: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
) -> Option<Vec<f32>> {
    let queue = backend.queue();
    let device = backend.device();

    // 1. Prepare transient buffers for input
    let buf_x = VulkanBuffer::from_slice(backend, x, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    
    // 2. Prepare weight buffers (these should ideally be cached in a real impl, but following Metal template)
    let buf_wq = VulkanBuffer::from_slice(backend, w_q, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_wk = VulkanBuffer::from_slice(backend, w_k, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_wv = VulkanBuffer::from_slice(backend, w_v, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_wo = VulkanBuffer::from_slice(backend, w_o, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_q_norm = VulkanBuffer::from_slice(backend, q_norm_w, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_k_norm = VulkanBuffer::from_slice(backend, k_norm_w, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    
    let buf_gate = VulkanBuffer::from_slice(backend, gate_q4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_up = VulkanBuffer::from_slice(backend, up_q4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_down = VulkanBuffer::from_slice(backend, down_t_q4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    // 3. Intermediate buffers
    let buf_q = VulkanBuffer::new(backend, seq_len * num_q_heads * head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_k = VulkanBuffer::new(backend, seq_len * num_kv_heads * head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_v = VulkanBuffer::new(backend, seq_len * num_kv_heads * head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_attn_out = VulkanBuffer::new(backend, seq_len * num_q_heads * head_dim * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_o_out = VulkanBuffer::new(backend, seq_len * hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    
    let buf_gate_out = VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_up_out = VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_act_out = VulkanBuffer::new(backend, inter * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_final = VulkanBuffer::new(backend, hidden * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    // Q8 buffers for FFN input
    let buf_q8_x = VulkanBuffer::new(backend, hidden, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
    let buf_q8_s = VulkanBuffer::new(backend, (hidden / 32) * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

    let mut builder = AutoCommandBufferBuilder::primary(
        backend.command_buffer_allocator(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).ok()?;

    // --- ATTENTION PHASE ---

    // Q, K, V Projections (SGEMM TransB)
    let sgemm_pipe = backend.sgemm_transb_pipeline.clone();
    let sgemm_layout = sgemm_pipe.pipeline.layout().set_layouts().get(0)?;

    for (buf_w, buf_out, out_dim) in [
        (&buf_wq, &buf_q, num_q_heads * head_dim),
        (&buf_wk, &buf_k, num_kv_heads * head_dim),
        (&buf_wv, &buf_v, num_kv_heads * head_dim)
    ] {
        let set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            sgemm_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buf_x.inner().clone()),
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

    // Fused Attention (Norm + RoPE + Cache + Softmax + V-Acc)
    let attn_pipe = backend.attn_fused_pipeline.clone();
    let attn_set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        attn_pipe.layout().set_layouts().get(0)?.clone(),
        [
            WriteDescriptorSet::buffer(0, buf_q.inner().clone()),
            WriteDescriptorSet::buffer(1, buf_k.inner().clone()),
            WriteDescriptorSet::buffer(2, buf_v.inner().clone()),
            WriteDescriptorSet::buffer(3, kv_cache.k_cache.inner().clone()),
            WriteDescriptorSet::buffer(4, kv_cache.v_cache.inner().clone()),
            WriteDescriptorSet::buffer(5, buf_attn_out.inner().clone()),
            WriteDescriptorSet::buffer(6, buf_q_norm.inner().clone()),
            WriteDescriptorSet::buffer(7, buf_k_norm.inner().clone()),
        ],
        [],
    ).ok()?;

    builder.bind_pipeline_compute(attn_pipe.clone()).unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, attn_pipe.layout().clone(), 0, attn_set).unwrap()
        .push_constants(attn_pipe.layout().clone(), 0, shaders::attn_fused::PushConstants {
            T: (kv_cache.current_len + seq_len) as u32,
            head_dim: head_dim as u32,
            num_q: num_q_heads as u32,
            num_kv: num_kv_heads as u32,
            scale: attn_scale,
            window_size: 0,
            eps: eps,
            qk_offset: 0.0,
            rope_base: rope_base,
            rotary_dim: head_dim as u32,
        })
        .unwrap().dispatch([num_q_heads as u32, 1, 1]).unwrap();

    // O Projection
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
            M: seq_len as u32, N: hidden as u32, K: (num_q_heads * head_dim) as u32 
        })
        .unwrap().dispatch([hidden as u32 / sgemm_pipe.rows_per_tg, seq_len as u32, 1]).unwrap();

    // --- FFN PHASE ---

    // 1. Q8 Quantize attention output
    let q8_pipe = backend.quantize_q8_pipeline.clone();
    let q8_set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        q8_pipe.layout().set_layouts().get(0)?.clone(),
        [
            WriteDescriptorSet::buffer(0, buf_o_out.inner().clone()),
            WriteDescriptorSet::buffer(1, buf_q8_x.inner().clone()),
            WriteDescriptorSet::buffer(2, buf_q8_s.inner().clone()),
        ],
        [],
    ).ok()?;

    builder.bind_pipeline_compute(q8_pipe.clone()).unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, q8_pipe.layout().clone(), 0, q8_set).unwrap()
        .push_constants(q8_pipe.layout().clone(), 0, shaders::quantize_q8::PushConstants { N: hidden as u32 })
        .unwrap().dispatch([(hidden as u32).div_ceil(256), 1, 1]).unwrap();

    // 2. Gate + Up MatVec (Q4)
    let q4_pipe = backend.q4_matvec_pipeline.clone();
    let q4_layout = q4_pipe.pipeline.layout().set_layouts().get(0)?;

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

    // 3. GEGLU Fused Activation
    let geglu_pipe = backend.geglu_pipeline.clone();
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

    // 4. Down MatVec (Q4 with f32 act_out input)
    let q4_f32_pipe = backend.q4_f32_matvec_pipeline.clone();
    let q4_f32_layout = q4_f32_pipe.pipeline.layout().set_layouts().get(0)?;
    
    let set_down = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator(),
        q4_f32_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, buf_down.inner().clone()),
            WriteDescriptorSet::buffer(1, buf_act_out.inner().clone()),
            WriteDescriptorSet::buffer(2, buf_final.inner().clone()),
        ],
        [],
    ).ok()?;

    builder.bind_pipeline_compute(q4_f32_pipe.pipeline.clone()).unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, q4_f32_pipe.pipeline.layout().clone(), 0, set_down).unwrap()
        .push_constants(q4_f32_pipe.pipeline.layout().clone(), 0, shaders::q4_f32_matvec::PushConstants { N: hidden as u32, K: inter as u32 })
        .unwrap().dispatch([hidden as u32 / q4_f32_pipe.rows_per_tg, 1, 1]).unwrap();

    // --- SUBMIT ---
    let command_buffer = builder.build().ok()?;
    vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer).ok()?
        .then_signal_fence_and_flush().ok()?
        .wait(None).ok()?;

    kv_cache.current_len += seq_len;

    let mut result = vec![0.0f32; hidden];
    buf_final.copy_to_slice(&mut result);
    Some(result)
}
