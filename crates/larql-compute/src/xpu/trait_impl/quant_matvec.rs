//! QuantMatVec implementation for XPU.

use crate::backend::QuantMatVec;
use crate::xpu::XpuBackend;
use crate::xpu::ops::{q4_vecmat, q4k_matvec, q6k_matvec};

impl QuantMatVec for XpuBackend {
    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        Some(q4_vecmat::dispatch(activation, q4_data, n, k))
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(q4k_matvec::dispatch(q4k_data, x, num_rows, hidden))
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(q6k_matvec::dispatch(q6k_data, x, num_rows, hidden))
    }
}
