//! Q4 × f32 matrix-vector dispatch for XPU.
//!
//! scores[N] = Q4[N, K] @ x[K]
//!
//! Matches the Metal `q4_vecmat` path for f32-input callers.
//! For the quantized Q8-input path, use the `q4_matvec_v4` FFI directly.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Dispatch a Q4 matvec (Q4 weights × f32 input vector) on XPU.
///
/// - `q4`: packed Q4_0 weight bytes `[n * k / 2]`
/// - `x`: f32 input vector `[k]`
/// - `n`: output dimension (number of rows)
/// - `k`: input dimension (hidden size)
/// - Returns: f32 output vector `[n]`
pub fn dispatch(q4: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];

    let q4_buf = XpuBuffer::from_slice(q4, false);
    let x_buf = XpuBuffer::from_slice(x, false);
    let mut out_buf = XpuBuffer::new_device(n * std::mem::size_of::<f32>());

    unsafe {
        xpu_ffi::q4_vecmat(
            q4_buf.as_ptr_type(),
            x_buf.as_ptr_type(),
            out_buf.as_mut_ptr_type(),
            n,
            k,
        );
    }

    out_buf.copy_to_slice(&mut out);
    out
}
