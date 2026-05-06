//! TurboQuant dispatch for XPU.

use super::super::ffi::ffi as xpu_ffi;

/// Encode vectors using TurboQuant (WHT + Lloyd-Max) on GPU.
pub fn encode_dispatch(
    input_ptr: *const f32,
    norms_ptr: *mut f32,
    packed_ptr: *mut u8,
    d: u32,
    batch: u32,
) {
    unsafe {
        xpu_ffi::dll_turboquant_encode(input_ptr, norms_ptr, packed_ptr, d, batch);
    }
}

/// Decode vectors using TurboQuant (Inverse WHT + Centroid lookup) on GPU.
pub fn decode_dispatch(
    norms_ptr: *const f32,
    packed_ptr: *const u8,
    output_ptr: *mut f32,
    d: u32,
    batch: u32,
) {
    unsafe {
        xpu_ffi::dll_turboquant_decode(norms_ptr, packed_ptr, output_ptr, d, batch);
    }
}
