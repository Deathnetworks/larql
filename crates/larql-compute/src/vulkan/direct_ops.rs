use super::*;

impl VulkanBackend {
    pub fn q4_matvec_direct(
        &self,
        _q4_data: &[u8],
        _q8_x: &[i8],
        _q8_scales: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Vec<f32> {
        unimplemented!("Vulkan q4_matvec_direct is not yet ported from Metal")
    }

    pub fn q4_vecmat_direct(
        &self,
        _activation: &[f32],
        _q4_data: &[u8],
        _intermediate: usize,
        _hidden: usize,
    ) -> Vec<f32> {
        unimplemented!("Vulkan q4_vecmat_direct is not yet ported from Metal")
    }
}
