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
pub fn encode_post_attn(
    h: &[f32],
    o_out: &[f32],
    post_attn_norm: &[f32],
    pre_ffn_weight: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
    has_post_norms: bool,
) -> (Vec<f32>, Vec<f32>) {
    let h_buf = XpuBuffer::from_slice(h, false);
    let o_buf = XpuBuffer::from_slice(o_out, false);
    let pan_buf = XpuBuffer::from_slice(post_attn_norm, false);
    let pfn_buf = XpuBuffer::from_slice(pre_ffn_weight, false);
    let mut hpa_buf = XpuBuffer::new_device(seq_len * hidden * 4);
    let mut ffn_buf = XpuBuffer::new_device(seq_len * hidden * 4);

    encode_post_attn_buf(
        &h_buf, &o_buf, &pan_buf, &pfn_buf,
        &mut hpa_buf, &mut ffn_buf,
        seq_len, hidden, eps, norm_offset, has_post_norms,
    );

    let mut h_post_attn = vec![0.0f32; seq_len * hidden];
    let mut ffn_norm_out = vec![0.0f32; seq_len * hidden];
    hpa_buf.copy_to_slice(&mut h_post_attn);
    ffn_buf.copy_to_slice(&mut ffn_norm_out);
    (h_post_attn, ffn_norm_out)
}

/// Zero-copy Post-attention residual + pre-FFN norm from existing buffers.
pub fn encode_post_attn_buf(
    h: &XpuBuffer,
    o_out: &XpuBuffer,
    post_attn_norm: &XpuBuffer,
    pre_ffn_weight: &XpuBuffer,
    h_post_attn: &mut XpuBuffer,
    ffn_norm_out: &mut XpuBuffer,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
    has_post_norms: bool,
) {
    for pos in 0..seq_len {
        let off = pos * hidden;
        let h_ptr = unsafe { h.as_ptr_type::<f32>().add(off) };
        let o_ptr = unsafe { o_out.as_ptr_type::<f32>().add(off) };
        let hpa_ptr = unsafe { h_post_attn.as_mut_ptr_type::<f32>().add(off) };

        if has_post_norms {
            let mut normed_o = XpuBuffer::new_device(hidden * 4);
            unsafe {
                xpu_ffi::rms_norm(
                    o_ptr as *const u8,
                    post_attn_norm.as_ptr_type(),
                    normed_o.as_mut_ptr_type(),
                    hidden, eps, norm_offset,
                );
                xpu_ffi::dll_residual_ops(
                    h_ptr as *const u8,
                    normed_o.as_ptr_type(),
                    hpa_ptr as *mut u8,
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
        } else {
            unsafe {
                xpu_ffi::dll_residual_ops(
                    h_ptr as *const u8,
                    o_ptr as *const u8,
                    hpa_ptr as *mut u8,
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
        }

        let ffn_ptr = unsafe { ffn_norm_out.as_mut_ptr_type::<f32>().add(off) };
        unsafe {
            xpu_ffi::rms_norm(
                hpa_ptr as *const u8,
                pre_ffn_weight.as_ptr_type(),
                ffn_ptr as *mut u8,
                hidden, eps, norm_offset,
            );
        }
    }
}

/// Post-FFN residual + optional post-FFN RMS norm.
pub fn encode_post_ffn(
    down_out: &[f32],
    h_post_attn: &[f32],
    post_ffn_norm: Option<&[f32]>,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) -> Vec<f32> {
    let d_buf = XpuBuffer::from_slice(down_out, false);
    let h_buf = XpuBuffer::from_slice(h_post_attn, false);
    let pfn_buf = post_ffn_norm.map(|w| XpuBuffer::from_slice(w, false));
    let mut next_buf = XpuBuffer::new_device(seq_len * hidden * 4);

    encode_post_ffn_buf(
        &d_buf, &h_buf, pfn_buf.as_ref(),
        &mut next_buf,
        seq_len, hidden, eps, norm_offset,
    );

    let mut h_next = vec![0.0f32; seq_len * hidden];
    next_buf.copy_to_slice(&mut h_next);
    h_next
}

/// Zero-copy Post-FFN residual from existing buffers.
pub fn encode_post_ffn_buf(
    down_out: &XpuBuffer,
    h_post_attn: &XpuBuffer,
    post_ffn_norm: Option<&XpuBuffer>,
    h_next: &mut XpuBuffer,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) {
    for pos in 0..seq_len {
        let off = pos * hidden;
        let d_ptr = unsafe { down_out.as_ptr_type::<f32>().add(off) };
        let h_ptr = unsafe { h_post_attn.as_ptr_type::<f32>().add(off) };
        let next_ptr = unsafe { h_next.as_mut_ptr_type::<f32>().add(off) };

        if let Some(w_buf) = post_ffn_norm {
            let mut normed_d = XpuBuffer::new_device(hidden * 4);
            unsafe {
                xpu_ffi::rms_norm(
                    d_ptr as *const u8,
                    w_buf.as_ptr_type(),
                    normed_d.as_mut_ptr_type(),
                    hidden, eps, norm_offset,
                );
                xpu_ffi::dll_residual_ops(
                    h_ptr as *const u8,
                    normed_d.as_ptr_type(),
                    next_ptr as *mut u8,
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
        } else {
            unsafe {
                xpu_ffi::dll_residual_ops(
                    h_ptr as *const u8,
                    d_ptr as *const u8,
                    next_ptr as *mut u8,
                    hidden as u32, 1.0, RESIDUAL_ADD_MODE,
                );
            }
        }
    }
}
