//! Vulkan kernel handle — wraps a compute pipeline with its metadata.

use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;

/// Handle to a compiled compute kernel.
#[derive(Clone)]
pub struct KernelHandle {
    /// The underlying Vulkano pipeline.
    pub pipeline: Arc<ComputePipeline>,
    /// Output rows the kernel covers per threadgroup.
    pub rows_per_tg: u32,
    /// Threads per threadgroup the kernel expects.
    pub threads_per_tg: u32,
    /// Shader entry point name.
    pub name: &'static str,
}

impl KernelHandle {
    /// Create a new handle for a flat-dispatch kernel.
    pub fn new(pipeline: Arc<ComputePipeline>, name: &'static str) -> Self {
        Self {
            pipeline,
            rows_per_tg: 1,
            threads_per_tg: 1,
            name,
        }
    }

    /// Create a new handle with explicit dispatch geometry.
    pub fn with_geometry(
        pipeline: Arc<ComputePipeline>,
        name: &'static str,
        rows_per_tg: u32,
        threads_per_tg: u32,
    ) -> Self {
        Self {
            pipeline,
            name,
            rows_per_tg,
            threads_per_tg,
        }
    }
}
