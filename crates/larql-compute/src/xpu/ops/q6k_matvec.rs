//! Q6_K matrix-vector dispatch for XPU.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

pub fn dispatch(
    q6k_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let x_buf = XpuBuffer::from_slice(x, false);
    let w_buf = XpuBuffer::from_slice(q6k_data, false);
    let mut out_buf = XpuBuffer::new_device(num_rows * 4);

    dispatch_buf(&w_buf, &x_buf, &mut out_buf, num_rows, hidden);

    let mut out = vec![0.0f32; num_rows];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Q6_K matvec from existing buffers.
pub fn dispatch_buf(
    q6k: &XpuBuffer,
    x: &XpuBuffer,
    out: &mut XpuBuffer,
    num_rows: usize,
    hidden: usize,
) {
    unsafe {
        xpu_ffi::q6k_matvec(
            q6k.as_ptr(),
            x.as_ptr_type(),
            out.as_mut_ptr_type(),
            num_rows,
            hidden,
        );
    }
}
