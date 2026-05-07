//! Rotary position embedding (RoPE) — pre-attention stage.
//!
//! XPU equivalent of Metal's `stages::rope`. Applies RoPE to Q and K
//! in-place per position via `rope_at_pos_batched_qk` SYCL kernel.
//!
//! The SYCL kernel handles both Q and K heads in one dispatch, unlike
//! Metal which loops per-head per-position in the encoder.
//!
//! `rotary_dim = 0` means rotate the full `head_dim`.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Apply RoPE to Q and K in-place for a sequence of positions.
///
/// - `q`: mutable f32 slice `[seq_len × num_q_heads × head_dim]`
/// - `k`: mutable f32 slice `[seq_len × num_kv_heads × head_dim]`
/// - `pos`: sequence position of the *first* token in the slice
/// - `rotary_dim`: 0 = full head_dim, otherwise partial rotation
#[allow(clippy::too_many_arguments)]
pub fn encode(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_base: f32,
) {
    let rdim = if rotary_dim == 0 { head_dim } else { rotary_dim };

    // The SYCL kernel is batched over Q and K heads for a single position.
    // For seq_len > 1 (prefill), we loop per position.
    let seq_len = q.len() / (num_q_heads * head_dim);

    for s in 0..seq_len {
        let q_off = s * num_q_heads * head_dim;
        let k_off = s * num_kv_heads * head_dim;
        let cur_pos = pos + s;

        let mut q_buf = XpuBuffer::from_slice(&q[q_off..q_off + num_q_heads * head_dim], false);
        let mut k_buf = XpuBuffer::from_slice(&k[k_off..k_off + num_kv_heads * head_dim], false);

        unsafe {
            xpu_ffi::rope_at_pos_batched_qk(
                q_buf.as_mut_ptr_type(),
                k_buf.as_mut_ptr_type(),
                head_dim,
                rope_base,
                cur_pos,
                rdim,
                num_q_heads,
                num_kv_heads,
            );
        }

        // Copy results back (in-place update)
        q_buf.copy_to_slice(&mut q[q_off..q_off + num_q_heads * head_dim]);
        k_buf.copy_to_slice(&mut k[k_off..k_off + num_kv_heads * head_dim]);
    }
}
