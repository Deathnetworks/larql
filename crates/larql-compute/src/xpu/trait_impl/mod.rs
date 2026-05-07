//! XPU trait implementations.

mod decode;
mod matmul;
mod quant_matvec;

use super::*;
use crate::backend::{Capability, ComputeBackend};

impl ComputeBackend for XpuBackend {
    fn name(&self) -> &str {
        "xpu (SYCL)"
    }

    fn device_info(&self) -> String {
        "Intel XPU via SYCL".to_string()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: Capability) -> bool {
        match cap {
            Capability::F32Gemv => true,
            Capability::QuantMatVec => true,
            Capability::Q4VecMat => true,
            _ => false,
        }
    }
}
