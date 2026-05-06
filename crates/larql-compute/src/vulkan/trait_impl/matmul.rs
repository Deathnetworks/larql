//! MatMul implementation for Vulkan.

use ndarray::{Array2, ArrayView2};
use crate::backend::MatMul;
use crate::vulkan::VulkanBackend;
use crate::vulkan::ops::f32_gemv;

impl MatMul for VulkanBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        crate::cpu::ops::f32_matmul::matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        crate::cpu::ops::f32_matmul::matmul_transb(a, b)
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.nrows(), w.ncols());
        if x.len() != k {
            return None;
        }

        f32_gemv::dispatch(self, w.as_slice()?, x, n as u32, k as u32)
    }
}
