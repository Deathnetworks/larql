//! MatMul implementation for Vulkan.

use ndarray::{Array2, ArrayView2};
use crate::backend::MatMul;
use crate::vulkan::VulkanBackend;

impl MatMul for VulkanBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(self, a, b, self.flop_threshold())
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(self, a, b, self.flop_threshold())
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.nrows(), w.ncols());
        if x.len() != k {
            return None;
        }
        
        // Threshold check: small GEMVs are faster on CPU due to dispatch overhead.
        if 2 * n * k < self.flop_threshold() {
            return None;
        }

        crate::vulkan::ops::f32_gemv::dispatch(self, w.as_slice()?, x, n as u32, k as u32)
    }
}
