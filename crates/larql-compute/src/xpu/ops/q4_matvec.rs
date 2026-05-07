//! Q4 × f32 matrix-vector dispatch for XPU.
//!
//! scores[N] = Q4[N, K] @ x[K]
//!
//! Matches the Metal `q4_vecmat` path for f32-input callers.
//! For the quantized Q8-input path, use the `q4_matvec_v4` FFI directly.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Dispatch a Q4 matvec (Q4 weights × f32 input vector) on XPU.
pub fn dispatch(q4: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let q4_buf = XpuBuffer::from_slice(q4, false);
    let x_buf = XpuBuffer::from_slice(x, false);
    let mut out_buf = XpuBuffer::new_device(n * 4);

    dispatch_buf(&q4_buf, &x_buf, &mut out_buf, n, k);

    let mut out = vec![0.0f32; n];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Q4 matvec from existing buffers.
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
