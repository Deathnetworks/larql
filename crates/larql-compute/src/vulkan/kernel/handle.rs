//! Vulkan kernel handle — wraps a compute pipeline with its metadata.
//! TODO: Port from Metal kernel/handle.rs (2 fns, 3,436 bytes)

use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;

/// Handle to a compiled compute kernel.
pub struct KernelHandle {
    pub pipeline: Arc<ComputePipeline>,
    pub name: &'static str,
}

impl KernelHandle {
    pub fn new(pipeline: Arc<ComputePipeline>, name: &'static str) -> Self {
        Self { pipeline, name }
    }
}
