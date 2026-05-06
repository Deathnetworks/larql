//! Vulkan buffer management.
//!
//! Provides a safe Rust wrapper around Vulkan buffers using `vulkano`.
//! Manages memory allocation, staging, and synchronization for compute shaders.
//!
//! All buffers are allocated via a `StandardMemoryAllocator` with
//! `PREFER_DEVICE` flags to ensure high bandwidth for GPU kernels.

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::device::Device;
use std::sync::Arc;

/// A Vulkan GPU buffer backed by host-visible device memory.
pub struct VulkanBuffer {
    inner: Subbuffer<[u8]>,
}

impl VulkanBuffer {
    /// Allocate a new buffer of `size` bytes.
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

    /// Create a buffer from a byte slice, copying data into GPU memory.
    pub fn from_data(device: Arc<Device>, data: &[u8], usage: BufferUsage) -> Option<Self> {
        let buf = Self::new(device, data.len(), usage)?;
        {
            let mut write = buf.inner.write().ok()?;
            write[..data.len()].copy_from_slice(data);
        }
        Some(buf)
    }

    /// Create a buffer from a typed slice (f32, u32, etc.).
    pub fn from_slice<T: bytemuck::Pod>(device: Arc<Device>, data: &[T], usage: BufferUsage) -> Option<Self> {
        let bytes = bytemuck::cast_slice(data);
        Self::from_data(device, bytes, usage)
    }

    /// Copy buffer contents back to a typed CPU slice.
    pub fn copy_to_slice<T: bytemuck::Pod>(&self, dst: &mut [T]) {
        let read = self.inner.read().expect("Failed to read Vulkan buffer");
        let byte_dst = bytemuck::cast_slice_mut(dst);
        let len = byte_dst.len().min(read.len());
        byte_dst[..len].copy_from_slice(&read[..len]);
    }

    /// Return the inner vulkano Subbuffer for descriptor set binding.
    pub fn inner(&self) -> &Subbuffer<[u8]> {
        &self.inner
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> u64 {
        self.inner.size()
    }
}
