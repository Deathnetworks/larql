//! Q4_K matrix-vector dispatch for XPU.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

pub fn dispatch(
    q4k_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_rows];
    
    let x_buf = XpuBuffer::from_slice(x, false);
    let w_buf = XpuBuffer::from_slice(q4k_data, false);
    let mut out_buf = XpuBuffer::new_device(num_rows * 4);

    unsafe {
        xpu_ffi::q4k_matvec_8sg(
            w_buf.as_ptr(),
            x_buf.as_ptr_type(),
            out_buf.as_mut_ptr_type(),
            num_rows,
            hidden,
        );
    }

    out_buf.copy_to_slice(&mut out);
    out
}
