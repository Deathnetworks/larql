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
pub fn encode(wo: &[u8], attn_out: &[f32], q_dim: usize, hidden: usize) -> Vec<f32> {
    let w_buf   = XpuBuffer::from_slice(wo, false);
    let in_buf  = XpuBuffer::from_slice(attn_out, false);
    let mut out_buf = XpuBuffer::new_device(hidden * 4);

    encode_buf(&w_buf, &in_buf, &mut out_buf, q_dim, hidden);

    let mut out = vec![0.0f32; hidden];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy O-projection from existing buffers.
pub fn encode_buf(
    wo: &XpuBuffer,
    attn_out: &XpuBuffer,
    out: &mut XpuBuffer,
    q_dim: usize,
    hidden: usize,
) {
    unsafe {
        xpu_ffi::q4k_proj(
            wo.as_ptr_type(),
            attn_out.as_ptr_type(),
            out.as_mut_ptr_type(),
            hidden,
            q_dim,
        );
    }
}

/// O-projection using Q6_K weights (Gemma 4 Q6K V path).
pub fn encode_q6k(wo: &[u8], attn_out: &[f32], q_dim: usize, hidden: usize) -> Vec<f32> {
    let w_buf   = XpuBuffer::from_slice(wo, false);
    let in_buf  = XpuBuffer::from_slice(attn_out, false);
    let mut out_buf = XpuBuffer::new_device(hidden * 4);

    encode_q6k_buf(&w_buf, &in_buf, &mut out_buf, q_dim, hidden);

    let mut out = vec![0.0f32; hidden];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Q6_K O-projection from existing buffers.
pub fn encode_q6k_buf(
    wo: &XpuBuffer,
    attn_out: &XpuBuffer,
    out: &mut XpuBuffer,
    q_dim: usize,
    hidden: usize,
) {
    unsafe {
        xpu_ffi::q6k_matvec(
            wo.as_ptr_type(),
            attn_out.as_ptr_type(),
            out.as_mut_ptr_type(),
            hidden,
            q_dim,
        );
    }
}
