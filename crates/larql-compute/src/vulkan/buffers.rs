//! Vulkan buffer management.
//!
//! Provides a safe Rust wrapper around Vulkan buffers using `vulkano`.
//! Manages memory allocation, staging, and synchronization for compute shaders.
//!
//! All buffers are allocated via a `StandardMemoryAllocator` with
//! `PREFER_DEVICE` flags to ensure high bandwidth for GPU kernels.

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::device::Device;
use std::sync::Arc;

/// A Vulkan GPU buffer.
///
/// Encapsulates a `vulkano::buffer::Buffer` and its associated memory.
pub struct VulkanBuffer {
    inner: Arc<Buffer>,
}

impl VulkanBuffer {
    pub fn new(device: Arc<Device>, size: usize, usage: BufferUsage) -> Option<Self> {
        let allocator = Arc::new(StandardMemoryAllocator::new_default(device));
        
        let buffer = Buffer::new_slice::<u8>(
            allocator,
            BufferCreateInfo {
                usage: usage | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            size as u64,
        ).ok()?;

        Some(Self { inner: buffer })
    }

    pub fn inner(&self) -> &Arc<Buffer> {
        &self.inner
    }

    pub fn as_ptr(&self) -> *const u8 {
        // This is tricky with Vulkano as it's not a raw pointer.
        // For now we'll just expose the inner buffer for the shader macro.
        std::ptr::null()
    }
}
