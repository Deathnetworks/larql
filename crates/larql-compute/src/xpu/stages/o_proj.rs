//! Output projection (attn_out → h) for XPU.
//!
//! XPU equivalent of Metal's `stages::o_proj`. Routes attention output
//! through the right weight format: Q4_K (f32 input) or Q6_K.
//!
//! For Q4_0 / Q8_0, callers should quantise `attn_out` before calling.
//! This module handles the Q4_K (most common) f32-input path directly.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// O-projection: Q4_K weights × f32 attention output → f32 hidden.
///
/// - `wo`: packed Q4K weight bytes `[hidden × q_dim / 2]`
/// - `attn_out`: f32 `[q_dim]`
/// - `q_dim`: attention output dimension
/// - `hidden`: output (hidden) dimension
/// Returns f32 `[hidden]`.
pub fn encode(wo: &[u8], attn_out: &[f32], q_dim: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];

    let w_buf   = XpuBuffer::from_slice(wo, false);
    let in_buf  = XpuBuffer::from_slice(attn_out, false);
    let mut out_buf = XpuBuffer::new_device(hidden * 4);

    unsafe {
        xpu_ffi::q4k_proj(
            w_buf.as_ptr_type(),
            in_buf.as_ptr_type(),
            out_buf.as_mut_ptr_type(),
            hidden,
            q_dim,
        );
    }

    out_buf.copy_to_slice(&mut out);
    out
}

/// O-projection using Q6_K weights (Gemma 4 Q6K V path).
///
/// Uses `q6k_matvec` FFI directly.
pub fn encode_q6k(wo: &[u8], attn_out: &[f32], q_dim: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];

    let w_buf   = XpuBuffer::from_slice(wo, false);
    let in_buf  = XpuBuffer::from_slice(attn_out, false);
    let mut out_buf = XpuBuffer::new_device(hidden * 4);

    unsafe {
        xpu_ffi::q6k_matvec(
            w_buf.as_ptr_type(),
            in_buf.as_ptr_type(),
            out_buf.as_mut_ptr_type(),
            hidden,
            q_dim,
        );
    }

    out_buf.copy_to_slice(&mut out);
    out
}
