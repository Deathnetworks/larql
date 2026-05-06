//! XPU backend trait implementations.

mod decode;
mod matmul;
mod quant_matvec;

use super::*;
use crate::backend::{Capability, ComputeBackend};
use crate::xpu::ffi::ffi as xpu_ffi;

impl ComputeBackend for XpuBackend {
    fn name(&self) -> &str {
        "XPU (SYCL)"
    }

    fn device_info(&self) -> String {
        xpu_ffi::get_device_info().to_string()
    }

    fn supports(&self, cap: Capability) -> bool {
        match cap {
            Capability::F32Gemv => true,
            Capability::QuantMatVec => true,
            Capability::Q4VecMat => true,
            Capability::DecodeToken => true, // We have attn_fused
            _ => false,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
