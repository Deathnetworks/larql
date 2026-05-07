//! Q8 quantization dispatch for XPU.

use super::super::ffi::ffi as xpu_ffi;

/// Quantize f32 to Q8 (int8 + scale) on GPU.
pub fn dispatch(
    input_ptr: *const f32,
    q8_out_ptr: *mut i8,
    scales_ptr: *mut f32,
    k: u32,
) {
    unsafe {
        xpu_ffi::dll_quantize_q8(input_ptr, q8_out_ptr, scales_ptr, k);
    }
}

/// Zero-copy Q8 quantization from existing buffers.
pub fn dispatch_buf(
    input: &super::super::buffers::XpuBuffer,
    q8_out: &mut super::super::buffers::XpuBuffer,
    scales_out: &mut super::super::buffers::XpuBuffer,
    k: u32,
) {
    dispatch(input.as_ptr_type(), q8_out.as_mut_ptr_type(), scales_out.as_mut_ptr_type(), k);
}
