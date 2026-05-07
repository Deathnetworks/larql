//! QK-norm and V-norm — per-head RMS norm for XPU.
//!
//! XPU equivalent of Metal's `stages::qk_norm`. Uses the SYCL
//! `dll_qk_norm_rope_fused` FFI for QK-norm and `dll_v_norm` for V-norm.
//!
//! Applied pre-RoPE for Gemma 3 / Gemma 4 models.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Per-head RMS norm on Q and K in-place.
///
/// Uses `dll_qk_norm_rope_fused` which applies QK-norm and optional RoPE
/// in one kernel. Set `rope_base = 0.0` to skip RoPE (norm-only).
///
/// Mutates `q` and `k` in-place.
#[allow(clippy::too_many_arguments)]
pub fn encode_qk_norm(
    q: &mut [f32],
    k: &mut [f32],
    q_weight: &[f32],
    k_weight: &[f32],
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    qk_norm_offset: f32,
    rope_base: f32,
    pos: usize,
    rotary_dim: usize,
) {
    for s in 0..seq_len {
        let q_off = s * num_q_heads * head_dim;
        let k_off = s * num_kv_heads * head_dim;
        let cur_pos = (pos + s) as u32;

        let mut q_buf = XpuBuffer::from_slice(&q[q_off..q_off + num_q_heads * head_dim], false);
        let mut k_buf = XpuBuffer::from_slice(&k[k_off..k_off + num_kv_heads * head_dim], false);
        let qw_buf = XpuBuffer::from_slice(q_weight, false);
        let kw_buf = XpuBuffer::from_slice(k_weight, false);

        unsafe {
            xpu_ffi::dll_qk_norm_rope_fused(
                q_buf.as_mut_ptr_type(),
                k_buf.as_mut_ptr_type(),
                qw_buf.as_ptr_type(),
                kw_buf.as_ptr_type(),
                head_dim as u32,
                num_q_heads as u32,
                eps,
                qk_norm_offset,
                rope_base,
                cur_pos,
                rotary_dim as u32,
            );
        }

        q_buf.copy_to_slice(&mut q[q_off..q_off + num_q_heads * head_dim]);
        k_buf.copy_to_slice(&mut k[k_off..k_off + num_kv_heads * head_dim]);
    }
}

/// Per-head RMS norm on V in-place (parameter-free, Gemma 4).
///
/// Uses `dll_v_norm` which normalises each KV head independently.
#[allow(clippy::too_many_arguments)]
pub fn encode_v_norm(
    v: &mut [f32],
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    batched: bool,
) {
    for s in 0..seq_len {
        let off = s * num_kv_heads * head_dim;
        let mut v_buf =
            XpuBuffer::from_slice(&v[off..off + num_kv_heads * head_dim], false);

        unsafe {
            xpu_ffi::dll_v_norm(
                v_buf.as_ptr_type(),
                v_buf.as_mut_ptr_type(),
                head_dim as u32,
                num_kv_heads as u32,
                eps,
                batched,
            );
        }

        v_buf.copy_to_slice(&mut v[off..off + num_kv_heads * head_dim]);
    }
}
