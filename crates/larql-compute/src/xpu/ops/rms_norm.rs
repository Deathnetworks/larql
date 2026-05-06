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
    let mut out = vec![0.0f32; n];
    
    let x_buf = XpuBuffer::from_slice(x, false);
    let w_buf = XpuBuffer::from_slice(weight, false);
    let mut out_buf = XpuBuffer::new_device(n * 4);

    unsafe {
        xpu_ffi::rms_norm(
            x_buf.as_ptr_type(),
            w_buf.as_ptr_type(),
            out_buf.as_mut_ptr_type(),
            n,
            eps,
            offset,
        );
    }

    out_buf.copy_to_slice(&mut out);
    out
}
