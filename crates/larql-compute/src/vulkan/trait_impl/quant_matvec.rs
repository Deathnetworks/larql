//! QuantMatVec implementation for Vulkan.

use crate::backend::QuantMatVec;
use crate::vulkan::VulkanBackend;

impl QuantMatVec for VulkanBackend {
    fn q4_vecmat(
        &self,
        _activation: &[f32],
        _q4_data: &[u8],
        _intermediate: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        // TODO: port from Metal — dispatch q4_vecmat shader
        None
    }

    fn q4k_matvec(&self, w4k: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        crate::vulkan::ops::q4k_matvec::dispatch(self, w4k, x, n as u32, k as u32)
    }

    fn q6k_matvec(&self, _w6k: &[u8], _x: &[f32], _n: usize, _k: usize) -> Option<Vec<f32>> {
        // TODO: port from Metal — dispatch q6k_matvec shader
        None
    }
}
