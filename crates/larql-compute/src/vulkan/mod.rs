//! Vulkan compute backend — Cross-platform GPU acceleration.

use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue, DeviceExtensions};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::VulkanLibrary;
use std::sync::Arc;
use once_cell::sync::Lazy;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::pipeline::{ComputePipeline, layout::PipelineLayout, ComputePipelineCreateInfo, layout::PipelineLayoutCreateInfo, ComputeShaderStageCreateInfo};
use vulkano::shader::EntryPoint;

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
use buffers::BufferCache;
use ops::q4_common::Q4Pipelines;

/// Vulkan compute backend.
pub struct VulkanBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub bufs: BufferCache,
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
    moe_scratch: std::sync::Mutex<Option<moe_dispatch::MoeScratch>>,
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
            let f32_ops = F32Ops::new_internal(device, queue).expect("Failed to initialize F32Ops");
            let bufs = BufferCache::new(device.clone());
            let q4 = Q4Pipelines { /* stubs */ };

            // Initialize all pipelines
            let rms_norm_shader = shaders::rms_norm::load(device.clone()).expect("Failed to load rms_norm shader");
            let silu_shader = shaders::silu::load(device.clone()).expect("Failed to load silu shader");
            let rope_shader = shaders::rope::load(device.clone()).expect("Failed to load rope shader");
            let q4_vecmat_shader = shaders::q4_vecmat::load(device.clone()).expect("Failed to load q4_vecmat shader");
            let f32_gemv_shader = shaders::f32_gemv::load(device.clone()).expect("Failed to load f32_gemv shader");
            let q4k_matvec_shader = shaders::q4k_matvec::load(device.clone()).expect("Failed to load q4k_matvec shader");
            let q4k_ffn_gate_up_shader = shaders::q4k_ffn_gate_up::load(device.clone()).expect("Failed to load q4k_ffn_gate_up shader");
            let attn_fused_shader = shaders::attn_fused::load(device.clone()).expect("Failed to load attn_fused shader");
            let q4k_qkv_proj_shader = shaders::q4k_qkv_proj::load(device.clone()).expect("Failed to load q4k_qkv_proj shader");
            let q4k_q6k_qkv_proj_shader = shaders::q4k_q6k_qkv_proj::load(device.clone()).expect("Failed to load q4k_q6k_qkv_proj shader");
            let q6k_matvec_shader = shaders::q6k_matvec::load(device.clone()).expect("Failed to load q6k_matvec shader");
            let quantize_q8_shader = shaders::quantize_q8::load(device.clone()).expect("Failed to load quantize_q8 shader");
            let layer_norm_shader = shaders::layer_norm::load(device.clone()).expect("Failed to load layer_norm shader");
            let residual_ops_shader = shaders::residual_ops::load(device.clone()).expect("Failed to load residual_ops shader");
            let graph_walk_knn_shader = shaders::graph_walk_knn::load(device.clone()).expect("Failed to load graph_walk_knn shader");
            let v_norm_shader = shaders::v_norm::load(device.clone()).expect("Failed to load v_norm shader");
            let qk_norm_shader = shaders::qk_norm::load(device.clone()).expect("Failed to load qk_norm shader");
            let qk_norm_rope_fused_shader = shaders::qk_norm_rope_fused::load(device.clone()).expect("Failed to load qk_norm_rope_fused shader");
            let post_attn_residual_norm_store_shader = shaders::post_attn_residual_norm_store::load(device.clone()).expect("Failed to load post_attn_residual_norm_store shader");
            let post_ffn_norm_residual_add_shader = shaders::post_ffn_norm_residual_add::load(device.clone()).expect("Failed to load post_ffn_norm_residual_add shader");
            let turboquant_encode_shader = shaders::turboquant_encode::load(device.clone()).expect("Failed to load turboquant_encode shader");
            let turboquant_decode_shader = shaders::turboquant_decode::load(device.clone()).expect("Failed to load turboquant_decode shader");
            let sgemm_shader = shaders::sgemm::load(device.clone()).expect("Failed to load sgemm shader");
            let sgemm_transb_shader = shaders::sgemm_transb::load(device.clone()).expect("Failed to load sgemm_transb shader");
            let q4_sparse_matvec_shader = shaders::q4_sparse_matvec::load(device.clone()).expect("Failed to load q4_sparse_matvec shader");
            let q4k_matvec_stride32_shader = shaders::q4k_matvec_stride32::load(device.clone()).expect("Failed to load q4k_matvec_stride32 shader");
            let q8_matvec_shader = shaders::q8_matvec::load(device.clone()).expect("Failed to load q8_matvec shader");

            let rms_norm_pipeline = Self::create_compute_pipeline(device, rms_norm_shader.entry_point("main").unwrap());
            let silu_pipeline = Self::create_compute_pipeline(device, silu_shader.entry_point("main").unwrap());
            let rope_pipeline = Self::create_compute_pipeline(device, rope_shader.entry_point("main").unwrap());
            let q4_vecmat_pipeline = Self::create_compute_pipeline(device, q4_vecmat_shader.entry_point("main").unwrap());
            let f32_gemv_pipeline = Self::create_compute_pipeline(device, f32_gemv_shader.entry_point("main").unwrap());
            let q4k_matvec_pipeline = Self::create_compute_pipeline(device, q4k_matvec_shader.entry_point("main").unwrap());
            let q4k_ffn_gate_up_pipeline = Self::create_compute_pipeline(device, q4k_ffn_gate_up_shader.entry_point("main").unwrap());
            let attn_fused_pipeline = Self::create_compute_pipeline(device, attn_fused_shader.entry_point("main").unwrap());
            let q4k_qkv_proj_pipeline = Self::create_compute_pipeline(device, q4k_qkv_proj_shader.entry_point("main").unwrap());
            let q4k_q6k_qkv_proj_pipeline = Self::create_compute_pipeline(device, q4k_q6k_qkv_proj_shader.entry_point("main").unwrap());
            let q6k_matvec_pipeline = Self::create_compute_pipeline(device, q6k_matvec_shader.entry_point("main").unwrap());
            let quantize_q8_pipeline = Self::create_compute_pipeline(device, quantize_q8_shader.entry_point("main").unwrap());
            let layer_norm_pipeline = Self::create_compute_pipeline(device, layer_norm_shader.entry_point("main").unwrap());
            let residual_ops_pipeline = Self::create_compute_pipeline(device, residual_ops_shader.entry_point("main").unwrap());
            let graph_walk_knn_pipeline = Self::create_compute_pipeline(device, graph_walk_knn_shader.entry_point("main").unwrap());
            let v_norm_pipeline = Self::create_compute_pipeline(device, v_norm_shader.entry_point("main").unwrap());
            let qk_norm_pipeline = Self::create_compute_pipeline(device, qk_norm_shader.entry_point("main").unwrap());
            let qk_norm_rope_fused_pipeline = Self::create_compute_pipeline(device, qk_norm_rope_fused_shader.entry_point("main").unwrap());
            let post_attn_residual_norm_store_pipeline = Self::create_compute_pipeline(device, post_attn_residual_norm_store_shader.entry_point("main").unwrap());
            let post_ffn_norm_residual_add_pipeline = Self::create_compute_pipeline(device, post_ffn_norm_residual_add_shader.entry_point("main").unwrap());
            let turboquant_encode_pipeline = Self::create_compute_pipeline(device, turboquant_encode_shader.entry_point("main").unwrap());
            let turboquant_decode_pipeline = Self::create_compute_pipeline(device, turboquant_decode_shader.entry_point("main").unwrap());
            let sgemm_pipeline = Self::create_compute_pipeline(device, sgemm_shader.entry_point("main").unwrap());
            let sgemm_transb_pipeline = Self::create_compute_pipeline(device, sgemm_transb_shader.entry_point("main").unwrap());
            let q4_sparse_matvec_pipeline = Self::create_compute_pipeline(device, q4_sparse_matvec_shader.entry_point("main").unwrap());
            let q4k_matvec_stride32_pipeline = Self::create_compute_pipeline(device, q4k_matvec_stride32_shader.entry_point("main").unwrap());
            let q8_matvec_pipeline = Self::create_compute_pipeline(device, q8_matvec_shader.entry_point("main").unwrap());

            Self {
                device: Arc::clone(device),
                queue: Arc::clone(queue),
                descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default())),
                command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default())),
                bufs,
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

    fn create_compute_pipeline(device: &Arc<Device>, entry_point: EntryPoint) -> Arc<ComputePipeline> {
        let stage = ComputeShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo::from_stages(&[stage.clone()]).expect("Failed to create pipeline layout"),
        ).expect("Failed to create pipeline layout");

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        ).expect("Failed to create compute pipeline")
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
