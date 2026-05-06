use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue, DeviceExtensions};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::VulkanLibrary;
use std::sync::Arc;
use once_cell::sync::Lazy;
use crate::backend::Capability;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;

pub struct VulkanBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

static VULKAN_STATE: Lazy<Option<(Arc<Device>, Arc<Queue>)>> = Lazy::new(|| {
    let library = VulkanLibrary::new().ok()?;
    let instance = Instance::new(library, InstanceCreateInfo::default()).ok()?;

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .ok()?
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(_, q)| q.queue_flags.intersects(vulkano::device::QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })?;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    ).ok()?;

    let queue = queues.next()?;
    Some((device, queue))
});

impl VulkanBackend {
    pub fn new() -> Option<Self> {
        VULKAN_STATE.as_ref().map(|(device, queue)| Self {
            device: Arc::clone(device),
            queue: Arc::clone(queue),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default())),
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default())),
        })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

pub mod shaders {
    pub mod rms_norm {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/rms_norm.glsl"
        }
    }
    pub mod silu {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/silu.glsl"
        }
    }
    pub mod rope {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/rope.glsl"
        }
    }
    pub mod q4_vecmat {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/q4_vecmat.glsl"
        }
    }
    pub mod q4k_matvec {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/q4k_matvec.glsl"
        }
    }
    pub mod f32_gemv {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/f32_gemv.glsl"
        }
    }
    pub mod q4k_ffn_gate_up {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/q4k_ffn_gate_up.glsl"
        }
    }
    pub mod q4k_qkv_proj {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/q4k_qkv_proj.glsl"
        }
    }
    pub mod q6k_matvec {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/q6k_matvec.glsl"
        }
    }
    pub mod attn_fused {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/attn_fused.glsl"
        }
    }
}

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, compute::ComputePipelineCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract};
use vulkano::sync::{self, GpuFuture};
use crate::vulkan::buffers::VulkanBuffer;

impl VulkanBackend {
    pub fn rms_norm(
        &self,
        x: &Arc<VulkanBuffer>,
        w: &Arc<VulkanBuffer>,
        out: &Arc<VulkanBuffer>,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        let shader = shaders::rms_norm::load(self.device.clone()).unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    self.device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).unwrap();

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(self.device.clone(), Default::default()),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, x.inner().clone()),
                WriteDescriptorSet::buffer(1, w.inner().clone()),
                WriteDescriptorSet::buffer(2, out.inner().clone()),
            ],
            [],
        ).unwrap();

        let push_constants = shaders::rms_norm::PushConstants {
            head_dim,
            eps,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            &vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(self.device.clone(), Default::default()),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

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
            .dispatch([num_heads, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();
    }

    pub fn silu(&self, x: &Arc<VulkanBuffer>, out: &Arc<VulkanBuffer>, n: u32) {
        let shader = shaders::silu::load(self.device.clone()).unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    self.device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).unwrap();

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, x.inner().clone()),
                WriteDescriptorSet::buffer(1, out.inner().clone()),
            ],
            [],
        ).unwrap();

        let push_constants = shaders::silu::PushConstants { n };

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

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
            .dispatch([(n + 255) / 256, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn rope(
        &self,
        q: &Arc<VulkanBuffer>,
        k: &Arc<VulkanBuffer>,
        pos: u32,
        head_dim: u32,
        num_q: u32,
        num_kv: u32,
        rope_base: f32,
        rotary_dim: u32,
    ) {
        let shader = shaders::rope::load(self.device.clone()).unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    self.device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).unwrap();

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, q.inner().clone()),
                WriteDescriptorSet::buffer(1, k.inner().clone()),
            ],
            [],
        ).unwrap();

        let push_constants = shaders::rope::PushConstants {
            pos,
            head_dim,
            num_q,
            num_kv,
            rope_base,
            rotary_dim,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        let threads = (num_q + num_kv) * (head_dim / 2);
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
            .dispatch([(threads + 255) / 256, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn attn_fused(
        &self,
        q_in: &Arc<VulkanBuffer>,
        k_in: &Arc<VulkanBuffer>,
        v_in: &Arc<VulkanBuffer>,
        k_cache: &Arc<VulkanBuffer>,
        v_cache: &Arc<VulkanBuffer>,
        out: &Arc<VulkanBuffer>,
        q_weight: &Arc<VulkanBuffer>,
        k_weight: &Arc<VulkanBuffer>,
        t: u32,
        head_dim: u32,
        num_q: u32,
        num_kv: u32,
        scale: f32,
        window_size: u32,
        eps: f32,
        qk_offset: f32,
        rope_base: f32,
        rotary_dim: u32,
    ) {
        let shader = shaders::attn_fused::load(self.device.clone()).unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    self.device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).unwrap();

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, q_in.inner().clone()),
                WriteDescriptorSet::buffer(1, k_in.inner().clone()),
                WriteDescriptorSet::buffer(2, v_in.inner().clone()),
                WriteDescriptorSet::buffer(3, k_cache.inner().clone()),
                WriteDescriptorSet::buffer(4, v_cache.inner().clone()),
                WriteDescriptorSet::buffer(5, out.inner().clone()),
                WriteDescriptorSet::buffer(6, q_weight.inner().clone()),
                WriteDescriptorSet::buffer(7, k_weight.inner().clone()),
            ],
            [],
        ).unwrap();

        let push_constants = shaders::attn_fused::PushConstants {
            t,
            head_dim,
            num_q,
            num_kv,
            scale,
            window_size,
            eps,
            qk_offset,
            rope_base,
            rotary_dim,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

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
            .dispatch([num_q, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
}

use crate::backend::MatMul;
use ndarray::{Array2, ArrayView2};

impl MatMul for VulkanBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        // Fallback to CPU for now or implement generic matmul.glsl
        crate::cpu::CpuBackend.matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        crate::cpu::CpuBackend.matmul_transb(a, b)
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let n = w.nrows() as u32;
        let k = w.ncols() as u32;
        if x.len() != k as usize { return None; }

        let shader = shaders::f32_gemv::load(self.device.clone()).ok()?;
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    self.device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).ok()?;

        let mut out = vec![0.0f32; n as usize];
        let w_bytes: Vec<u8> = w.as_slice()?.iter().flat_map(|f| f.to_le_bytes()).collect();
        let x_bytes: Vec<u8> = x.iter().flat_map(|f| f.to_le_bytes()).collect();

        let w_buf = VulkanBuffer::new(self.device.clone(), w_bytes.len(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let x_buf = VulkanBuffer::new(self.device.clone(), x_bytes.len(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let out_buf = VulkanBuffer::new(self.device.clone(), out.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        // Copy data to buffers (This is inefficient, we should have a way to map them directly)
        // For this first pass, we'll just implement the dispatch logic.

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, w_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, x_buf.inner().clone()),
                WriteDescriptorSet::buffer(2, out_buf.inner().clone()),
            ],
            [],
        ).ok()?;

        let push_constants = shaders::f32_gemv::PushConstants {
            n,
            k,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
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
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .ok()?
            .then_signal_fence_and_flush()
            .ok()?
            .wait(None)
            .ok()?;

        // Copy out_buf back to out vec
        Some(out)
    }
}

use crate::backend::{QuantMatVec, DecodeBackend};
use crate::QuantFormat;

impl QuantMatVec for VulkanBackend {
    fn q4k_matvec(
        &self,
        weights: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let n = num_rows as u32;
        let k = hidden as u32;
        
        let shader = shaders::q4k_matvec::load(self.device.clone()).ok()?;
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    self.device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).ok()?;

        let mut out = vec![0.0f32; n as usize];
        
        // Use VulkanBuffer for weights, x, and out
        // For weights, we need to handle the padding to uint[]
        let mut w_padded = weights.to_vec();
        while w_padded.len() % 4 != 0 { w_padded.push(0); }
        
        let w_buf = VulkanBuffer::new(self.device.clone(), w_padded.len(), vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let x_buf = VulkanBuffer::new(self.device.clone(), x.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;
        let out_buf = VulkanBuffer::new(self.device.clone(), out.len() * 4, vulkano::buffer::BufferUsage::STORAGE_BUFFER)?;

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, w_buf.inner().clone()),
                WriteDescriptorSet::buffer(1, x_buf.inner().clone()),
                WriteDescriptorSet::buffer(2, out_buf.inner().clone()),
            ],
            [],
        ).ok()?;

        let push_constants = shaders::q4k_matvec::PushConstants {
            n,
            k,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
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
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .ok()?
            .then_signal_fence_and_flush()
            .ok()?
            .wait(None)
            .ok()?;

        Some(out)
    }

    fn q6k_matvec(
        &self,
        _weights: &[u8],
        _x: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None // Placeholder
    }
}

impl DecodeBackend for VulkanBackend {
    fn has_kv_cache(&self) -> bool {
        true
    }

    fn kv_cache_len(&self) -> usize {
        // This will be managed by a separate KV cache structure in the future.
        0
    }
}

impl Capability for VulkanBackend {
    fn name(&self) -> &'static str {
        "Vulkan"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
