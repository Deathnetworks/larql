//! Vulkan compute backend — Cross-platform GPU acceleration.
//!
//! All operations go through the [`ComputeBackend`] trait. Vulkan-specific
//! optimisations: Persistent descriptor sets, command buffer recycling,
//! and GLSL-based compute shaders.
//!
//! ## Modules
//!
//! - `shaders/`:   GLSL source files and vulkano-shader macros.
//! - `ops/`:       GPU dispatch — modular dispatchers for each operation.
//! - `trait_impl/`: Backend trait implementations (ComputeBackend, MatMul, etc.).
//! - `buffers`:    Vulkan buffer management.
//!
//! ## Requirements
//!
//! - Vulkan SDK installed and `VULKAN_SDK` environment variable set.
//! - Compatible Vulkan 1.2+ GPU.

use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue, DeviceExtensions};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::VulkanLibrary;
use std::sync::Arc;
use once_cell::sync::Lazy;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;

pub mod buffers;
pub mod ops;
mod trait_impl;
pub mod f32_ops;
pub mod calibrate;
pub mod kernel;
pub mod decode;
pub mod stages;

use f32_ops::F32Ops;

/// Vulkan compute backend.
pub struct VulkanBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    f32_ops: F32Ops,
    flop_threshold: std::sync::atomic::AtomicUsize,
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
    /// Create a new Vulkan backend.
    ///
    /// Initializes the Vulkan instance and selects the best available 
    /// physical device (preferring Discrete GPUs).
    pub fn new() -> Option<Self> {
        VULKAN_STATE.as_ref().map(|(device, queue)| {
            let f32_ops = F32Ops::new_internal(device, queue).expect("Failed to initialize F32Ops");
            Self {
                device: Arc::clone(device),
                queue: Arc::clone(queue),
                descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default())),
                command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default())),
                f32_ops,
                flop_threshold: std::sync::atomic::AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
            }
        })
    }

    pub fn flop_threshold(&self) -> usize {
        self.flop_threshold.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_flop_threshold(&self, threshold: usize) {
        self.flop_threshold.store(
            threshold.max(calibrate::MIN_FLOP_FLOOR),
            std::sync::atomic::Ordering::Relaxed
        );
    }

    pub fn calibrate(&self) {
        let best = calibrate::calibrate(self, &self.f32_ops);
        self.set_flop_threshold(best);
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn descriptor_set_allocator(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.descriptor_set_allocator
    }

    pub fn command_buffer_allocator(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.command_buffer_allocator
    }
}

impl F32Ops {
    fn new_internal(device: &Arc<Device>, _queue: &Arc<Queue>) -> Option<Self> {
        let sgemm_shader = shaders::sgemm::load(device.clone()).ok()?;
        let sgemm_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                sgemm_shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).ok()?;

        let transb_shader = shaders::sgemm_transb::load(device.clone()).ok()?;
        let transb_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                transb_shader.entry_point("main").unwrap(),
                PipelineLayout::new(
                    device.clone(),
                    vulkano::pipeline::layout::PipelineLayoutCreateInfo::default(),
                ).unwrap(),
            ),
        ).ok()?;

        Some(Self {
            sgemm_pipeline,
            transb_pipeline,
        })
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
    pub mod quantize_q8 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/quantize_q8.glsl"
        }
    }
    pub mod turboquant_encode {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/turboquant_encode.glsl"
        }
    }
    pub mod turboquant_decode {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/turboquant_decode.glsl"
        }
    }
    pub mod sgemm {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/sgemm.glsl"
        }
    }
    pub mod sgemm_transb {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/vulkan/shaders/sgemm_transb.glsl"
        }
    }
}
