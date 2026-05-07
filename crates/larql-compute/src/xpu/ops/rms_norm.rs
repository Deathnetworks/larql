//! RMSNorm dispatch for XPU.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

pub fn dispatch(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    offset: f32,
) -> Vec<f32> {
    let n = x.len();
    let x_buf = XpuBuffer::from_slice(x, false);
    let w_buf = XpuBuffer::from_slice(weight, false);
    let mut out_buf = XpuBuffer::new_device(n * 4);

    dispatch_buf(&x_buf, &w_buf, &mut out_buf, n, eps, offset);

    let mut out = vec![0.0f32; n];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy RMSNorm from existing buffers.
pub fn dispatch_buf(
    x: &XpuBuffer,
    weight: &XpuBuffer,
    out: &mut XpuBuffer,
    len: usize,
    eps: f32,
    offset: f32,
) {
    unsafe {
        xpu_ffi::rms_norm(
            x.as_ptr_type(),
            weight.as_ptr_type(),
            out.as_mut_ptr_type(),
            len,
            eps,
            offset,
        );
    }
}
