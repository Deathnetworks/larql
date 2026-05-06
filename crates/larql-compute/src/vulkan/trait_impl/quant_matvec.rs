//! QuantMatVec implementation for Vulkan.

use crate::backend::QuantMatVec;
use crate::vulkan::VulkanBackend;
use crate::vulkan::ops::{q4_vecmat, q4k_matvec, q6k_matvec};

impl QuantMatVec for VulkanBackend {
    fn q4_vecmat(&self, q4: &[u8], x: &[f32], m: usize, k: usize) -> Option<Vec<f32>> {
        q4_vecmat::dispatch(self, q4, x, m as u32, k as u32)
    }

    fn q4k_matvec(&self, w4k: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        q4k_matvec::dispatch(self, w4k, x, n as u32, k as u32)
    }

    fn q6k_matvec(&self, w6k: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        q6k_matvec::dispatch(self, w6k, x, n as u32, k as u32)
    }
}
