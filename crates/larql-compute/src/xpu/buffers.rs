//! XPU (SYCL) USM (Unified Shared Memory) buffer management.
//!
//! Provides a safe Rust wrapper around SYCL memory allocations. 
//! Supports both `Device` memory (GPU-only, high bandwidth) and 
//! `Shared` memory (migratable between CPU/GPU).
//!
//! USM allows zero-copy pointers to be shared between the Rust FFI
//! boundary and the SYCL kernels in the Bridge-to-DLL.

use std::collections::HashMap;
use std::sync::Mutex;
use super::ffi::ffi as xpu_ffi;

/// Cache key: (pointer address, byte length) of the source data.
type CacheKey = (usize, usize);

/// A SYCL USM buffer.
pub enum XpuBuffer {
    Device {
        ptr: *mut u8,
        size: usize,
    },
    Shared {
        ptr: *mut u8,
        size: usize,
    },
}

unsafe impl Send for XpuBuffer {}
unsafe impl Sync for XpuBuffer {}

impl XpuBuffer {
    pub fn new_device(size: usize) -> Self {
        let ptr = unsafe { xpu_ffi::allocate_device(size) };
        assert!(!ptr.is_null(), "Failed to allocate XPU device memory");
        Self::Device { ptr, size }
    }

    pub fn new_shared(size: usize) -> Self {
        let ptr = unsafe { xpu_ffi::allocate_shared(size) };
        assert!(!ptr.is_null(), "Failed to allocate XPU shared memory");
        Self::Shared { ptr, size }
    }

    pub fn from_slice<T: Copy>(slice: &[T], shared: bool) -> Self {
        let size = slice.len() * std::mem::size_of::<T>();
        let mut buf = if shared {
            Self::new_shared(size)
        } else {
            Self::new_device(size)
        };
        buf.copy_from_slice(slice);
        buf
    }

    pub fn copy_from_slice<T: Copy>(&mut self, slice: &[T]) {
        let size = slice.len() * std::mem::size_of::<T>();
        assert!(size <= self.size());
        unsafe {
            xpu_ffi::copy_h2d(self.as_mut_ptr(), slice.as_ptr() as *const u8, size);
        }
    }

    pub fn copy_to_slice<T: Copy>(&self, slice: &mut [T]) {
        let size = slice.len() * std::mem::size_of::<T>();
        assert!(size <= self.size());
        unsafe {
            xpu_ffi::copy_d2h(slice.as_mut_ptr() as *mut u8, self.as_ptr(), size);
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Device { ptr, .. } => *ptr as *const u8,
            Self::Shared { ptr, .. } => *ptr as *const u8,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Device { ptr, .. } => *ptr,
            Self::Shared { ptr, .. } => *ptr,
        }
    }

    pub fn as_ptr_type<T>(&self) -> *const T {
        self.as_ptr() as *const T
    }

    pub fn as_mut_ptr_type<T>(&mut self) -> *mut T {
        self.as_mut_ptr() as *mut T
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Device { size, .. } => *size,
            Self::Shared { size, .. } => *size,
        }
    }
}

impl Drop for XpuBuffer {
    fn drop(&mut self) {
        unsafe {
            xpu_ffi::free_memory(self.as_mut_ptr());
        }
    }
}

/// Buffer cache for XPU USM buffers.
pub struct BufferCache {
    cache: Mutex<HashMap<CacheKey, std::sync::Arc<XpuBuffer>>>,
    scratch_pool: Mutex<HashMap<usize, Vec<XpuBuffer>>>,
}

impl BufferCache {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            scratch_pool: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_f32(&self, data: &[f32]) -> std::sync::Arc<XpuBuffer> {
        let key: CacheKey = (data.as_ptr() as usize, data.len());
        let mut cache = self.cache.lock().unwrap();
        if let Some(buf) = cache.get(&key) {
            return buf.clone();
        }

        let buf = XpuBuffer::from_slice(data, false);
        let arc = std::sync::Arc::new(buf);
        cache.insert(key, arc.clone());
        arc
    }

    pub fn get_bytes(&self, data: &[u8]) -> std::sync::Arc<XpuBuffer> {
        let key: CacheKey = (data.as_ptr() as usize, data.len());
        let mut cache = self.cache.lock().unwrap();
        if let Some(buf) = cache.get(&key) {
            return buf.clone();
        }

        let buf = XpuBuffer::from_slice(data, false);
        let arc = std::sync::Arc::new(buf);
        cache.insert(key, arc.clone());
        arc
    }

    pub fn output(&self, bytes: usize) -> XpuBuffer {
        let mut pool = self.scratch_pool.lock().unwrap();
        if let Some(buf) = pool.entry(bytes).or_default().pop() {
            return buf;
        }
        XpuBuffer::new_device(bytes)
    }

    pub fn recycle(&self, buf: XpuBuffer) {
        let bytes = buf.size();
        self.scratch_pool.lock().unwrap().entry(bytes).or_default().push(buf);
    }
}

pub struct ScratchGuard<'a> {
    bufs: Vec<XpuBuffer>,
    cache: &'a BufferCache,
}

impl<'a> ScratchGuard<'a> {
    pub fn new(cache: &'a BufferCache) -> Self {
        Self {
            bufs: Vec::new(),
            cache,
        }
    }

    pub fn track(&mut self, buf: XpuBuffer) -> &XpuBuffer {
        self.bufs.push(buf);
        self.bufs.last().unwrap()
    }
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        for buf in self.bufs.drain(..) {
            self.cache.recycle(buf);
        }
    }
}
