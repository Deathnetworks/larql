//! Vulkan kernel traits — defines the interface for compute kernels.

/// A flat-dispatch compute kernel driven by `dispatch` with fixed geometry.
pub trait ShaderKernel {
    /// Shader entry point name (usually "main").
    const KERNEL_NAME: &'static str = "main";
}

/// A tiled compute kernel that needs explicit workgroup sizes for dispatch.
pub trait TiledKernel {
    /// Shader entry point name.
    const KERNEL_NAME: &'static str = "main";
    /// Output rows the kernel covers per threadgroup.
    const ROWS_PER_TG: u32;
    /// Threads per threadgroup (workgroup size X * Y * Z).
    const THREADS_PER_TG: u32;
}
