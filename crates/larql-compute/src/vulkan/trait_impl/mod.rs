//! Vulkan backend trait implementations.

mod decode;
pub mod matmul;
pub mod quant_matvec;

use super::*;
use crate::backend::{Capability, ComputeBackend};

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "Vulkan"
    }

    fn device_info(&self) -> String {
        format!("{:?}", self.device().physical_device().properties().device_name)
    }

    fn supports(&self, cap: Capability) -> bool {
        match cap {
            Capability::F32Gemv => true,
            Capability::QuantMatVec => true,
            Capability::Q4VecMat => true,
            Capability::DecodeToken => true,
            _ => false,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
