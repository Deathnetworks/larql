//! F32 matrix-vector dispatch for XPU.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Dispatch a single F32 gemv on XPU.
pub fn dispatch(
    w: &[f32],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_rows];
    
    let x_buf = XpuBuffer::from_slice(x, false);
    let w_buf = XpuBuffer::from_slice(w, false);
    let mut out_buf = XpuBuffer::new_device(num_rows * 4);

    unsafe {
        xpu_ffi::f32_gemv(
            x_buf.as_ptr_type(),
            w_buf.as_ptr_type(),
            out_buf.as_mut_ptr_type(),
            num_rows,
            hidden,
        );
    }
    
    out_buf.copy_to_slice(&mut out);
    out
}
