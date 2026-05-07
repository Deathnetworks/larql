//! Post-attention and post-FFN residual stages for XPU.
//!
//! XPU equivalent of Metal's `stages::residual`. Performs element-wise
//! add and optional RMS norm for post-attention and post-FFN residuals,
//! using `dll_residual_ops` (mode 0 = add) and `rms_norm` FFI.
//!
//! Mode constants for `dll_residual_ops`:
//!   0 = add (a + b * scalar)

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

const RESIDUAL_ADD_MODE: u32 = 0;

/// Post-attention residual + pre-FFN norm.
///
/// For each position in `seq_len`:
///   1. `h_post_attn = h + o_out` (pre-norm) or `h + norm(o_out, post_attn_norm)` (post-norm).
///   2. `ffn_norm_out = rms_norm(h_post_attn, pre_ffn_weight)`.
///
/// Returns `(h_post_attn[seq_len * hidden], ffn_norm_out[seq_len * hidden])`.
#[allow(clippy::too_many_arguments)]
pub fn encode_post_attn(
    h: &[f32],           // residual stream [seq_len × hidden]
    o_out: &[f32],       // attention output [seq_len × hidden]
    post_attn_norm: &[f32],  // post-attn norm weights [hidden]
    pre_ffn_weight: &[f32],  // pre-FFN norm weights [hidden]
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
    has_post_norms: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut h_post_attn = vec![0.0f32; seq_len * hidden];
    let mut ffn_norm_out = vec![0.0f32; seq_len * hidden];

    let w_post_attn = XpuBuffer::from_slice(post_attn_norm, false);
    let w_pre_ffn   = XpuBuffer::from_slice(pre_ffn_weight, false);

    for pos in 0..seq_len {
        let off = pos * hidden;
        let h_slice   = &h[off..off + hidden];
        let o_slice   = &o_out[off..off + hidden];
        let out_slice = &mut h_post_attn[off..off + hidden];

        if has_post_norms {
            // Post-norm: norm(o_out) first, then add to h.
            let o_buf = XpuBuffer::from_slice(o_slice, false);
            let mut normed_o = XpuBuffer::new_device(hidden * 4);
            unsafe {
                xpu_ffi::rms_norm(
                    o_buf.as_ptr_type(),
                    w_post_attn.as_ptr_type(),
                    normed_o.as_mut_ptr_type(),
                    hidden, eps, norm_offset,
                );
            }
            let h_buf  = XpuBuffer::from_slice(h_slice, false);
            let mut add_out = XpuBuffer::new_device(hidden * 4);
            unsafe {
                xpu_ffi::dll_residual_ops(
                    h_buf.as_ptr_type(),
                    normed_o.as_ptr_type(),
                    add_out.as_mut_ptr_type(),
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
            add_out.copy_to_slice(out_slice);
        } else {
            // Pre-norm: plain residual add h + o_out.
            let h_buf = XpuBuffer::from_slice(h_slice, false);
            let o_buf = XpuBuffer::from_slice(o_slice, false);
            let mut add_out = XpuBuffer::new_device(hidden * 4);
            unsafe {
                xpu_ffi::dll_residual_ops(
                    h_buf.as_ptr_type(),
                    o_buf.as_ptr_type(),
                    add_out.as_mut_ptr_type(),
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
            add_out.copy_to_slice(out_slice);
        }

        // Pre-FFN rms_norm on h_post_attn → ffn_norm_out.
        let hpa_buf = XpuBuffer::from_slice(out_slice as &[f32], false);
        let mut ffn_out_buf = XpuBuffer::new_device(hidden * 4);
        unsafe {
            xpu_ffi::rms_norm(
                hpa_buf.as_ptr_type(),
                w_pre_ffn.as_ptr_type(),
                ffn_out_buf.as_mut_ptr_type(),
                hidden, eps, norm_offset,
            );
        }
        ffn_out_buf.copy_to_slice(&mut ffn_norm_out[off..off + hidden]);
    }

    (h_post_attn, ffn_norm_out)
}

/// Post-FFN residual + optional post-FFN RMS norm.
///
/// Returns `h_next[seq_len × hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_post_ffn(
    down_out: &[f32],          // FFN down output [seq_len × hidden]
    h_post_attn: &[f32],       // residual from post-attn stage [seq_len × hidden]
    post_ffn_norm: Option<&[f32]>, // Some = post-norm model (Gemma), None = pre-norm
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) -> Vec<f32> {
    let mut h_next = vec![0.0f32; seq_len * hidden];

    let w_post_ffn: Option<XpuBuffer> =
        post_ffn_norm.map(|w| XpuBuffer::from_slice(w, false));

    for pos in 0..seq_len {
        let off = pos * hidden;
        let d_slice = &down_out[off..off + hidden];
        let h_slice = &h_post_attn[off..off + hidden];
        let out_slice = &mut h_next[off..off + hidden];

        let d_buf = XpuBuffer::from_slice(d_slice, false);
        let h_buf = XpuBuffer::from_slice(h_slice, false);
        let mut add_out = XpuBuffer::new_device(hidden * 4);

        if let Some(ref w_buf) = w_post_ffn {
            // Post-norm: norm(down_out) then add.
            let mut normed_d = XpuBuffer::new_device(hidden * 4);
            unsafe {
                xpu_ffi::rms_norm(
                    d_buf.as_ptr_type(),
                    w_buf.as_ptr_type(),
                    normed_d.as_mut_ptr_type(),
                    hidden, eps, norm_offset,
                );
                xpu_ffi::dll_residual_ops(
                    h_buf.as_ptr_type(),
                    normed_d.as_ptr_type(),
                    add_out.as_mut_ptr_type(),
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
        } else {
            // Pre-norm: plain add h_post_attn + down_out.
            unsafe {
                xpu_ffi::dll_residual_ops(
                    h_buf.as_ptr_type(),
                    d_buf.as_ptr_type(),
                    add_out.as_mut_ptr_type(),
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
        }

        add_out.copy_to_slice(out_slice);
    }

    h_next
}
