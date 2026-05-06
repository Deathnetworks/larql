//! Q + K + V projections — one call per position for Vulkan.
//!
//! Three code paths depending on the weight format + mix:
//!
//! - **Fused f32-input** (`encode_fused_f32`): all three projections share
//!   the same format (Q4_K) and we dispatch `q4k_qkv_proj` in one go.
//! - **Per-projection f32-input** (`encode_per_proj`): mixed formats
//!   (e.g. Gemma 4 Q4_K Q/K + Q6_K V). Three separate shader dispatches.
//! - **Fused Q8-input** (`encode_fused_q8`): `Q8_0` attention layers.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use std::sync::Arc;

use crate::vulkan::VulkanBackend;
use crate::vulkan::shaders;
use super::quant_matvec;

/// Per-projection format + weight tuple used by the mixed-format path.
pub struct Proj<'a> {
    pub format: crate::QuantFormat,
    pub w_buf: &'a Subbuffer<[u32]>,
    pub out_buf: &'a Subbuffer<[f32]>,
    pub out_off: u64,
    pub rows: usize,
}

#[derive(Clone, Copy)]
pub enum FusedQkvKernel {
    Q4k,
    Q4kQ6k,
}

impl FusedQkvKernel {
    fn rows_per_tg(self) -> u32 {
        match self {
            Self::Q4k => 8,
            Self::Q4kQ6k => 8,
        }
    }
}

/// Fused Q4_K QKV — all three projections same format.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_f32(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipeline: &Arc<ComputePipeline>,
    kernel: FusedQkvKernel,
    wq_buf: &Subbuffer<[u32]>,
    wk_buf: &Subbuffer<[u32]>,
    wv_buf: &Subbuffer<[u32]>,
    f32_in: &Subbuffer<[f32]>,
    f32_in_off: u64,
    q_out: &Subbuffer<[f32]>,
    q_off: u64,
    k_out: &Subbuffer<[f32]>,
    k_off: u64,
    v_out: &Subbuffer<[f32]>,
    v_off: u64,
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) {
    let total_rows = (q_rows + kv_rows + kv_rows) as u32;
    let q_rows_val = q_rows as u32;
    let k_rows_val = kv_rows as u32;
    let v_rows_val = kv_rows as u32;
    let k_val = hidden as u32;

    let num_tgs = total_rows.div_ceil(kernel.rows_per_tg());
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let in_slice = f32_in.clone().slice(f32_in_off .. f32_in_off + hidden as u64);
    let q_slice = q_out.clone().slice(q_off .. q_off + q_rows as u64);
    let k_slice = k_out.clone().slice(k_off .. k_off + kv_rows as u64);
    let v_slice = v_out.clone().slice(v_off .. v_off + kv_rows as u64);

    let set = PersistentDescriptorSet::new(
        backend.descriptor_set_allocator.clone(),
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, wq_buf.clone()),
            WriteDescriptorSet::buffer(1, wk_buf.clone()),
            WriteDescriptorSet::buffer(2, wv_buf.clone()),
            WriteDescriptorSet::buffer(3, in_slice.clone()),
            WriteDescriptorSet::buffer(4, q_slice.clone()),
            WriteDescriptorSet::buffer(5, k_slice.clone()),
            WriteDescriptorSet::buffer(6, v_slice.clone()),
        ],
        [],
    ).unwrap();

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct PushConstants {
        q_rows: u32,
        k_rows: u32,
        v_rows: u32,
        k: u32,
        mode: u32,
    }

    let pcs = PushConstants {
        q_rows: q_rows_val,
        k_rows: k_rows_val,
        v_rows: v_rows_val,
        k: k_val,
        mode: 0, // 0: fused QKV
    };

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, pcs)
        .unwrap()
        .dispatch([num_tgs, 1, 1])
        .unwrap();
}

/// Per-projection f32-input QKV — mixed formats (Gemma 4 Q4_K + Q6_K).
#[allow(clippy::too_many_arguments)]
pub fn encode_per_proj(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    backend: &VulkanBackend,
    pipes: &quant_matvec::Pipelines<'_>,
    f32_in: &Subbuffer<[f32]>,
    f32_in_off: u64,
    q8_in: &Subbuffer<[i8]>,
    q8_in_off: u64,
    q8s_in: &Subbuffer<[f32]>,
    q8s_in_off: u64,
    projections: [Proj<'_>; 3],
    hidden: usize,
) {
    let f32_slice = f32_in.clone().slice(f32_in_off .. f32_in_off + hidden as u64);
    let q8_slice = q8_in.clone().slice(q8_in_off .. q8_in_off + hidden as u64);
    let q8s_slice = q8s_in.clone().slice(q8s_in_off .. q8s_in_off + (hidden / 32) as u64);

    for p in projections {
        let out_slice = p.out_buf.clone().slice(p.out_off .. p.out_off + p.rows as u64);
        quant_matvec::encode(
            builder,
            backend,
            p.format,
            p.w_buf,
            &f32_slice,
            &q8_slice,
            &q8s_slice,
            &out_slice,
            pipes,
            p.rows,
            hidden,
        );
    }
}

/// Fused Q8-input QKV — for Q8_0 attention.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_q8(
    _builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    _backend: &VulkanBackend,
    _pipeline: &Arc<ComputePipeline>,
    _wq_buf: &Subbuffer<[u32]>,
    _wq_scale: &Subbuffer<[f32]>,
    _wk_buf: &Subbuffer<[u32]>,
    _wk_scale: &Subbuffer<[f32]>,
    _wv_buf: &Subbuffer<[u32]>,
    _wv_scale: &Subbuffer<[f32]>,
    _q8_in: &Subbuffer<[i8]>,
    _q8_in_off: u64,
    _q8s_in: &Subbuffer<[f32]>,
    _q8s_in_off: u64,
    _q_out: &Subbuffer<[f32]>,
    _q_off: u64,
    _k_out: &Subbuffer<[f32]>,
    _k_off: u64,
    _v_out: &Subbuffer<[f32]>,
    _v_off: u64,
    _q_rows: usize,
    _kv_rows: usize,
    _hidden: usize,
) {
    unimplemented!("Q8 fused QKV is not yet implemented in Vulkan");
}
