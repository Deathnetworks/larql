use std::ptr;
use super::ffi::ffi as xpu_ffi;

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
