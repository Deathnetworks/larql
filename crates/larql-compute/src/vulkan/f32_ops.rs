//! f32 matmul operations via Vulkan compute shaders.

use std::sync::Arc;
use ndarray::{Array2, ArrayView2};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};
use vulkano::device::{Device, Queue};

use crate::vulkan::kernel::handle::KernelHandle;
use crate::vulkan::shaders;
use crate::vulkan::buffers::VulkanBuffer;

pub struct F32Ops {
    sgemm_pipeline: KernelHandle,
    transb_pipeline: KernelHandle,
}

impl F32Ops {
    pub fn new(device: &Arc<Device>, _queue: &Arc<Queue>) -> Option<Self> {
        let sgemm_pipeline = KernelHandle::new(super::VulkanBackend::create_compute_pipeline(device, &shaders::sgemm::load(device.clone()).ok()?), "main");
        let transb_pipeline = KernelHandle::new(super::VulkanBackend::create_compute_pipeline(device, &shaders::sgemm_transb::load(device.clone()).ok()?), "main");

        Some(Self {
            sgemm_pipeline,
            transb_pipeline,
        })
    }

    pub fn dispatch_notrans(
        &self,
        backend: &super::VulkanBackend,
        a_data: &[f32],
        b_data: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        self.dispatch_internal(backend, &self.sgemm_pipeline, a_data, b_data, m, n, k)
    }

    pub fn dispatch_transb(
        &self,
        backend: &super::VulkanBackend,
        a_data: &[f32],
        b_data: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        self.dispatch_internal(backend, &self.transb_pipeline, a_data, b_data, m, n, k)
    }

    fn dispatch_internal(
        &self,
        backend: &super::VulkanBackend,
        kernel: &KernelHandle,
        a_data: &[f32],
        b_data: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let pipeline = &kernel.pipeline;
        let device = backend.device();
        let queue = backend.queue();

        let a_buf = VulkanBuffer::from_slice(backend, a_data, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let b_buf = VulkanBuffer::from_slice(backend, b_data, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let out_buf = VulkanBuffer::new(backend, m * n * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        let layout = pipeline.layout().set_layouts().get(0)?;
        let set = PersistentDescriptorSet::new(
            backend.descriptor_set_allocator(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, b_buf.inner().clone()),
                WriteDescriptorSet::buffer(2, out_buf.inner().clone()),
            ],
            [],
        ).ok()?;

        let mut builder = AutoCommandBufferBuilder::primary(
            backend.command_buffer_allocator(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).ok()?;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
        struct PushConstants {
            m: u32,
            n: u32,
            k: u32,
        }
        let push_constants = PushConstants { m: m as u32, n: n as u32, k: k as u32 };

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
            .dispatch([n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1])
            .unwrap();

        let command_buffer = builder.build().ok()?;
        sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .ok()?
            .then_signal_fence_and_flush()
            .ok()?
            .wait(None)
            .ok()?;

        let mut c = vec![0.0f32; m * n];
        out_buf.copy_to_slice(&mut c);
        Some(c)
    }

    pub fn matmul(
        &self,
        backend: &super::VulkanBackend,
        a: ArrayView2<f32>,
        b: ArrayView2<f32>,
        flop_threshold: usize,
    ) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        if 2 * m * n * k < flop_threshold {
            return a.dot(&b);
        }

        let a_owned;
        let a_data: &[f32] = match a.as_slice() {
            Some(s) => s,
            None => {
                a_owned = a.as_standard_layout().into_owned();
                a_owned.as_slice().unwrap()
            }
        };
        let b_owned;
        let b_data: &[f32] = match b.as_slice() {
            Some(s) => s,
            None => {
                b_owned = b.as_standard_layout().into_owned();
                b_owned.as_slice().unwrap()
            }
        };

        let c = self.dispatch_notrans(backend, a_data, b_data, m, n, k).unwrap();
        Array2::from_shape_vec((m, n), c).unwrap()
    }

    pub fn matmul_transb(
        &self,
        backend: &super::VulkanBackend,
        a: ArrayView2<f32>,
        b: ArrayView2<f32>,
        flop_threshold: usize,
    ) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[0];
        if 2 * m * n * k < flop_threshold {
            return a.dot(&b.t());
        }

        let a_owned;
        let a_data: &[f32] = match a.as_slice() {
            Some(s) => s,
            None => {
                a_owned = a.as_standard_layout().into_owned();
                a_owned.as_slice().unwrap()
            }
        };
        let b_owned;
        let b_data: &[f32] = match b.as_slice() {
            Some(s) => s,
            None => {
                b_owned = b.as_standard_layout().into_owned();
                b_owned.as_slice().unwrap()
            }
        };

        let c = self.dispatch_transb(backend, a_data, b_data, m, n, k).unwrap();
        Array2::from_shape_vec((m, n), c).unwrap()
    }
}
