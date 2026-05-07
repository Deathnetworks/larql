//! MatMul implementation for XPU.

use ndarray::{Array2, ArrayView2};
use crate::backend::MatMul;
use crate::xpu::XpuBackend;
use crate::xpu::ops::f32_gemv;

impl MatMul for XpuBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.nrows(), w.ncols());
        if x.len() != k {
            return None;
        }

        Some(f32_gemv::dispatch(w.as_slice()?, x, n, k))
    }
}
