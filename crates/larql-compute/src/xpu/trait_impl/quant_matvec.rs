use crate::backend::QuantMatVec;
use crate::xpu::XpuBackend;
use crate::xpu::ops::{q4_matvec, q4k_matvec};

impl QuantMatVec for XpuBackend {
    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        Some(q4_matvec::dispatch(q4_data, activation, n, k))
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
        _q6k_data: &[u8],
        _x: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None // XPU q6k not yet implemented
    }
}
