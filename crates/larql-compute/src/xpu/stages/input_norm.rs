//! Input layer norm — first stage of every transformer layer.
//!
//! XPU equivalent of Metal's `stages::input_norm`. Instead of encoding
//! into a Metal command encoder, we call the SYCL `rms_norm` FFI directly.
//!
//! Two paths:
//! - `encode_f32`: norm → f32 output (Q4_K / Q6_K paths)
//! - `encode_q8`: norm then quantise → Q8 i8 + scales (Q4_0 / Q8_0 paths)

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// RMS norm with f32 output.
///
/// `x[hidden]` → `out[hidden]` via `rms_norm(x, weight, out, len, eps, offset)`.
pub fn encode_f32(
    x: &[f32],
    norm_weight: &[f32],
    out: &mut XpuBuffer,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) {
    let x_buf = XpuBuffer::from_slice(x, false);
    let w_buf = XpuBuffer::from_slice(norm_weight, false);
    unsafe {
        xpu_ffi::rms_norm(
            x_buf.as_ptr_type(),
            w_buf.as_ptr_type(),
            out.as_mut_ptr_type(),
            hidden,
            eps,
            norm_offset,
        );
    }
}

/// RMS norm + Q8 quantise in two passes.
///
/// Norms `x` to a temporary f32 buffer, then quantises via
/// `dll_quantize_q8` into `q8_out` (i8) + `q8s_out` (f32 scales).
pub fn encode_q8(
    x: &[f32],
    norm_weight: &[f32],
    q8_out: &mut XpuBuffer,
    q8s_out: &mut XpuBuffer,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) {
    let x_buf = XpuBuffer::from_slice(x, false);
    let w_buf = XpuBuffer::from_slice(norm_weight, false);
    let mut normed_buf = XpuBuffer::new_device(hidden * std::mem::size_of::<f32>());

    // Step 1: rms_norm → normed_buf (f32)
    unsafe {
        xpu_ffi::rms_norm(
            x_buf.as_ptr_type(),
            w_buf.as_ptr_type(),
            normed_buf.as_mut_ptr_type(),
            hidden,
            eps,
            norm_offset,
        );
    }

    // Step 2: quantise normed_buf → Q8 i8 + scales
    unsafe {
        xpu_ffi::dll_quantize_q8(
            normed_buf.as_ptr_type(),
            q8_out.as_mut_ptr_type(),
            q8s_out.as_mut_ptr_type(),
            hidden as u32,
        );
    }
}
