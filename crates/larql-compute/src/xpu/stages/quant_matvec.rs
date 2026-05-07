//! Format-aware single-vector matvec dispatch for XPU.
//!
//! XPU equivalent of Metal's `stages::quant_matvec`. Routes to the
//! correct SYCL FFI based on weight quantization format:
//!
//! | Format        | FFI call       | Input type |
//! |---------------|----------------|------------|
//! | Q4_K / Q4_KF  | `q4k_proj`     | f32        |
//! | Q6_K          | `q6k_matvec`   | f32        |
//! | Q4_0 / Q8_0   | `q4_vecmat`    | f32        |

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Supported quantization formats for XPU matvec routing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Q4K,
    Q4KF,
    Q6K,
    Q4_0,
    Q8_0,
}

/// Dispatch a single-vector matvec by weight format.
pub fn encode(w: &[u8], x: &[f32], n: usize, k: usize, format: QuantFormat) -> Vec<f32> {
    let w_buf   = XpuBuffer::from_slice(w, false);
    let x_buf   = XpuBuffer::from_slice(x, false);
    let mut out_buf = XpuBuffer::new_device(n * 4);

    encode_buf(&w_buf, &x_buf, &mut out_buf, n, k, format);

    let mut out = vec![0.0f32; n];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Format-aware matvec from existing buffers.
pub fn encode_buf(
    w: &XpuBuffer,
    x: &XpuBuffer,
    out: &mut XpuBuffer,
    n: usize,
    k: usize,
    format: QuantFormat,
) {
    match format {
        QuantFormat::Q4K | QuantFormat::Q4KF => unsafe {
            xpu_ffi::q4k_proj(
                w.as_ptr_type(),
                x.as_ptr_type(),
                out.as_mut_ptr_type(),
                n,
                k,
            );
        },
        QuantFormat::Q6K => unsafe {
            xpu_ffi::q6k_matvec(
                w.as_ptr_type(),
                x.as_ptr_type(),
                out.as_mut_ptr_type(),
                n,
                k,
            );
        },
        QuantFormat::Q4_0 | QuantFormat::Q8_0 => unsafe {
            xpu_ffi::q4_vecmat(
                w.as_ptr_type(),
                x.as_ptr_type(),
                out.as_mut_ptr_type(),
                n,
                k,
            );
        },
    }
}
