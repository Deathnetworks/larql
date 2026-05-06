use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue, DeviceExtensions};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::VulkanLibrary;
use std::sync::Arc;
use once_cell::sync::Lazy;
use crate::backend::Capability;

pub struct VulkanBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
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
}

impl Capability for VulkanBackend {
    fn name(&self) -> &'static str {
        "Vulkan"
    }
}
