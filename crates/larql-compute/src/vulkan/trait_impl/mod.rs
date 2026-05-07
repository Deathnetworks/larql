//! Vulkan trait implementations.

mod decode;
mod matmul;
mod quant_matvec;

use super::*;
use crate::backend::{Capability, ComputeBackend};

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan (GPU)"
    }

    fn device_info(&self) -> String {
        "Vulkan GPU".to_string()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: Capability) -> bool {
        match cap {
            Capability::F32Gemv => true,
            _ => false,
        }
    }
}
