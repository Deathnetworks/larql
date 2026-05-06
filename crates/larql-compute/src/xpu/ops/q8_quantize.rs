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
