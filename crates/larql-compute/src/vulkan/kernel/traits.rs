//! Vulkan kernel traits — defines the interface for compute kernels.
//! TODO: Port from Metal kernel/traits.rs (1 fn, 2,501 bytes)

/// Trait for a compute shader kernel that can be dispatched.
pub trait ShaderKernel {
    fn name(&self) -> &str;
}

/// Trait for a tiled compute kernel with explicit workgroup sizes.
pub trait TiledKernel: ShaderKernel {
    fn workgroup_size(&self) -> [u32; 3];
}
