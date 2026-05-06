//! Vulkan compute backend — Cross-platform GPU acceleration.

use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue, DeviceExtensions};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::VulkanLibrary;
use std::sync::Arc;
use once_cell::sync::Lazy;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};

pub mod buffers;
pub mod calibrate;
mod decode;
mod decode_hybrid;
pub mod diag;
mod direct_ops;
pub mod f32_ops;
pub mod kernel;
mod moe_dispatch;
pub mod ops;
mod pipeline;
mod prefill;
pub mod shaders;
pub mod stages;
mod trait_impl;

use f32_ops::F32Ops;
use ops::q4_common::Q4Pipelines;

/// Vulkan compute backend.
pub struct VulkanBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub f32_ops: F32Ops,
    pub q4: Q4Pipelines,
    // Shaders
    pub rms_norm_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub silu_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub rope_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q4_vecmat_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub f32_gemv_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q4k_matvec_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q4k_ffn_gate_up_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub attn_fused_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q4k_qkv_proj_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q4k_q6k_qkv_proj_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q6k_matvec_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub quantize_q8_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub layer_norm_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub residual_ops_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub graph_walk_knn_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub v_norm_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub qk_norm_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub qk_norm_rope_fused_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub post_attn_residual_norm_store_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub post_ffn_norm_residual_add_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub turboquant_encode_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub turboquant_decode_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub sgemm_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub sgemm_transb_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q4_sparse_matvec_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q4k_matvec_stride32_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    pub q8_matvec_pipeline: Arc<vulkano::pipeline::ComputePipeline>,
    
    flop_threshold: std::sync::atomic::AtomicUsize,
    kv_cache: std::sync::Mutex<Option<ops::kv_cache::KVCache>>,
    moe_scratch: std::sync::Mutex<Option<()>>,
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
    pub fn new() -> Option<Self> {
        VULKAN_STATE.as_ref().map(|(device, queue)| {
            let f32_ops = F32Ops::new(device, queue).expect("Failed to initialize F32Ops");
            let q4 = Q4Pipelines { /* stubs */ };

            // Initialize all pipelines
            let rms_norm_pipeline = Self::create_compute_pipeline(device, &shaders::rms_norm::load(device.clone()).expect("rms_norm shader"));
            let silu_pipeline = Self::create_compute_pipeline(device, &shaders::silu::load(device.clone()).expect("silu shader"));
            let rope_pipeline = Self::create_compute_pipeline(device, &shaders::rope::load(device.clone()).expect("rope shader"));
            let q4_vecmat_pipeline = Self::create_compute_pipeline(device, &shaders::q4_vecmat::load(device.clone()).expect("q4_vecmat shader"));
            let f32_gemv_pipeline = Self::create_compute_pipeline(device, &shaders::f32_gemv::load(device.clone()).expect("f32_gemv shader"));
            let q4k_matvec_pipeline = Self::create_compute_pipeline(device, &shaders::q4k_matvec::load(device.clone()).expect("q4k_matvec shader"));
            let q4k_ffn_gate_up_pipeline = Self::create_compute_pipeline(device, &shaders::q4k_ffn_gate_up::load(device.clone()).expect("q4k_ffn_gate_up shader"));
            let attn_fused_pipeline = Self::create_compute_pipeline(device, &shaders::attn_fused::load(device.clone()).expect("attn_fused shader"));
            let q4k_qkv_proj_pipeline = Self::create_compute_pipeline(device, &shaders::q4k_qkv_proj::load(device.clone()).expect("q4k_qkv_proj shader"));
            let q4k_q6k_qkv_proj_pipeline = Self::create_compute_pipeline(device, &shaders::q4k_q6k_qkv_proj::load(device.clone()).expect("q4k_q6k_qkv_proj shader"));
            let q6k_matvec_pipeline = Self::create_compute_pipeline(device, &shaders::q6k_matvec::load(device.clone()).expect("q6k_matvec shader"));
            let quantize_q8_pipeline = Self::create_compute_pipeline(device, &shaders::quantize_q8::load(device.clone()).expect("quantize_q8 shader"));
            let layer_norm_pipeline = Self::create_compute_pipeline(device, &shaders::layer_norm::load(device.clone()).expect("layer_norm shader"));
            let residual_ops_pipeline = Self::create_compute_pipeline(device, &shaders::residual_ops::load(device.clone()).expect("residual_ops shader"));
            let graph_walk_knn_pipeline = Self::create_compute_pipeline(device, &shaders::graph_walk_knn::load(device.clone()).expect("graph_walk_knn shader"));
            let v_norm_pipeline = Self::create_compute_pipeline(device, &shaders::v_norm::load(device.clone()).expect("v_norm shader"));
            let qk_norm_pipeline = Self::create_compute_pipeline(device, &shaders::qk_norm::load(device.clone()).expect("qk_norm shader"));
            let qk_norm_rope_fused_pipeline = Self::create_compute_pipeline(device, &shaders::qk_norm_rope_fused::load(device.clone()).expect("qk_norm_rope_fused shader"));
            let post_attn_residual_norm_store_pipeline = Self::create_compute_pipeline(device, &shaders::post_attn_residual_norm_store::load(device.clone()).expect("post_attn_residual_norm_store shader"));
            let post_ffn_norm_residual_add_pipeline = Self::create_compute_pipeline(device, &shaders::post_ffn_norm_residual_add::load(device.clone()).expect("post_ffn_norm_residual_add shader"));
            let turboquant_encode_pipeline = Self::create_compute_pipeline(device, &shaders::turboquant_encode::load(device.clone()).expect("turboquant_encode shader"));
            let turboquant_decode_pipeline = Self::create_compute_pipeline(device, &shaders::turboquant_decode::load(device.clone()).expect("turboquant_decode shader"));
            let sgemm_pipeline = Self::create_compute_pipeline(device, &shaders::sgemm::load(device.clone()).expect("sgemm shader"));
            let sgemm_transb_pipeline = Self::create_compute_pipeline(device, &shaders::sgemm_transb::load(device.clone()).expect("sgemm_transb shader"));
            let q4_sparse_matvec_pipeline = Self::create_compute_pipeline(device, &shaders::q4_sparse_matvec::load(device.clone()).expect("q4_sparse_matvec shader"));
            let q4k_matvec_stride32_pipeline = Self::create_compute_pipeline(device, &shaders::q4k_matvec_stride32::load(device.clone()).expect("q4k_matvec_stride32 shader"));
            let q8_matvec_pipeline = Self::create_compute_pipeline(device, &shaders::q8_matvec::load(device.clone()).expect("q8_matvec shader"));

            Self {
                device: Arc::clone(device),
                queue: Arc::clone(queue),
                descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default())),
                command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default())),
                f32_ops,
                q4,
                rms_norm_pipeline,
                silu_pipeline,
                rope_pipeline,
                q4_vecmat_pipeline,
                f32_gemv_pipeline,
                q4k_matvec_pipeline,
                q4k_ffn_gate_up_pipeline,
                attn_fused_pipeline,
                q4k_qkv_proj_pipeline,
                q4k_q6k_qkv_proj_pipeline,
                q6k_matvec_pipeline,
                quantize_q8_pipeline,
                layer_norm_pipeline,
                residual_ops_pipeline,
                graph_walk_knn_pipeline,
                v_norm_pipeline,
                qk_norm_pipeline,
                qk_norm_rope_fused_pipeline,
                post_attn_residual_norm_store_pipeline,
                post_ffn_norm_residual_add_pipeline,
                turboquant_encode_pipeline,
                turboquant_decode_pipeline,
                sgemm_pipeline,
                sgemm_transb_pipeline,
                q4_sparse_matvec_pipeline,
                q4k_matvec_stride32_pipeline,
                q8_matvec_pipeline,
                flop_threshold: std::sync::atomic::AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
                kv_cache: std::sync::Mutex::new(None),
                moe_scratch: std::sync::Mutex::new(None),
            }
        })
    }

    /// Create a compute pipeline from a loaded shader module using vulkano 0.34 API.
    fn create_compute_pipeline(device: &Arc<Device>, shader_module: &Arc<vulkano::shader::ShaderModule>) -> Arc<ComputePipeline> {
        let entry_point = shader_module.entry_point("main").expect("Shader missing 'main' entry point");
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .expect("Failed to create pipeline layout info"),
        ).expect("Failed to create pipeline layout");

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        ).expect("Failed to create compute pipeline")
    }

    // ── Accessor methods (used by all ops/* and stages/* files) ──

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

    pub fn calibrate(&self) {
        let best = calibrate::calibrate(self, &self.f32_ops);
        self.set_flop_threshold(best);
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
}
