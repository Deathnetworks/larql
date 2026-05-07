//! Q4 × f32 matvec convenience wrapper for XPU.
//!
//! Thin wrapper over `q4_matvec::dispatch` for callers that explicitly
//! want the f32-input path (vs the Q8-quantized input path via q4_matvec_v4).
//!
//! Both `q4_f32_matvec` and `q4_matvec` resolve to the same `q4_vecmat`
//! SYCL kernel — this module exists to mirror the Metal backend's
//! source layout and maintain call-site clarity.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Dispatch Q4 × f32 matvec on XPU.
///
/// - `q4`: packed Q4_0 weight bytes `[n * k / 2]`
/// - `x`: f32 input vector `[k]`
/// - `n`: output rows
/// - `k`: input hidden size
/// - Returns: f32 output `[n]`
pub fn dispatch(q4: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let q4_buf = XpuBuffer::from_slice(q4, false);
    let x_buf = XpuBuffer::from_slice(x, false);
    let mut out_buf = XpuBuffer::new_device(n * std::mem::size_of::<f32>());

    dispatch_buf(&q4_buf, &x_buf, &mut out_buf, n, k);

    let mut out = vec![0.0f32; n];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Q4 × f32 matvec from existing buffers.
pub fn dispatch_buf(
    q4: &XpuBuffer,
    x: &XpuBuffer,
    out: &mut XpuBuffer,
    n: usize,
    k: usize,
) {
    unsafe {
        xpu_ffi::q4_vecmat(
            q4.as_ptr_type(),
            x.as_ptr_type(),
            out.as_mut_ptr_type(),
            n,
            k,
        );
    }
}
