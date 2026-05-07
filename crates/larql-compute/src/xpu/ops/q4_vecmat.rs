//! Q4 vec-mat dispatch for XPU (vector × Q4 matrix).
//!
//! out[N] = x[K] @ Q4[K, N]  (equivalent to Q4[N, K] @ x[K] under transpose)
//!
//! Uses the `q4_vecmat` SYCL kernel which handles Q4-packed weights with
//! an f32 activation vector. The kernel handles dequantisation internally.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Dispatch Q4 vec-mat on XPU.
///
/// - `q4`: packed Q4_0 weight bytes
/// - `x`: f32 activation vector `[k]`
/// - `m`: output dimension (number of output rows / columns in weight)
/// - `k`: input dimension (hidden size)
/// - Returns: f32 output vector `[m]`
pub fn dispatch(q4: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];

    let q4_buf = XpuBuffer::from_slice(q4, false);
    let x_buf = XpuBuffer::from_slice(x, false);
    let mut out_buf = XpuBuffer::new_device(m * std::mem::size_of::<f32>());

    unsafe {
        xpu_ffi::q4_vecmat(
            q4_buf.as_ptr_type(),
            x_buf.as_ptr_type(),
            out_buf.as_mut_ptr_type(),
            m,
            k,
        );
    }

    out_buf.copy_to_slice(&mut out);
    out
}
