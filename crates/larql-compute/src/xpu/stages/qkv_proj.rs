//! Q + K + V projections for XPU.
//!
//! XPU equivalent of Metal's `stages::qkv_proj`. Three paths:
//!
//! - **Fused f32-input** (`encode_fused_f32`): all three projections share
//!   Q4_K format — one `q4k_qkv_proj` FFI call handles all three.
//! - **Fused Q4_K + Q6_K** (`encode_fused_q4k_q6k`): mixed Q4/Q6 weights
//!   via `dll_q4k_q6k_qkv_proj`.
//! - **Per-projection** (`encode_per_proj`): three separate matvec calls
//!   for fully mixed formats (e.g. Q4_K Q, Q6_K K/V).

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Fused Q4_K QKV projection — one kernel for Q, K, V.
///
/// `wq/wk/wv`: packed Q4_K weights. `x`: f32 input `[hidden]`.
/// Returns `(q_out[q_rows], k_out[kv_rows], v_out[kv_rows])`.
pub fn encode_fused_f32(
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    x: &[f32],
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q_out = vec![0.0f32; q_rows];
    let mut k_out = vec![0.0f32; kv_rows];
    let mut v_out = vec![0.0f32; kv_rows];

    let wq_buf = XpuBuffer::from_slice(wq, false);
    let wk_buf = XpuBuffer::from_slice(wk, false);
    let wv_buf = XpuBuffer::from_slice(wv, false);
    let x_buf  = XpuBuffer::from_slice(x, false);
    let mut q_buf = XpuBuffer::new_device(q_rows  * 4);
    let mut k_buf = XpuBuffer::new_device(kv_rows * 4);
    let mut v_buf = XpuBuffer::new_device(kv_rows * 4);

    unsafe {
        xpu_ffi::q4k_qkv_proj(
            wq_buf.as_ptr_type(),
            wk_buf.as_ptr_type(),
            wv_buf.as_ptr_type(),
            x_buf.as_ptr_type(),
            q_buf.as_mut_ptr_type(),
            k_buf.as_mut_ptr_type(),
            v_buf.as_mut_ptr_type(),
            q_rows  as u32,
            kv_rows as u32,
            kv_rows as u32,
            hidden  as u32,
        );
    }

    q_buf.copy_to_slice(&mut q_out);
    k_buf.copy_to_slice(&mut k_out);
    v_buf.copy_to_slice(&mut v_out);
    (q_out, k_out, v_out)
}

/// Fused Q4_K Q/K + Q6_K V QKV projection.
///
/// Used by Gemma 4 and similar mixed-quant models.
pub fn encode_fused_q4k_q6k(
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    x: &[f32],
    q_rows: usize,
    kv_rows: usize,
    v_rows: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q_out = vec![0.0f32; q_rows];
    let mut k_out = vec![0.0f32; kv_rows];
    let mut v_out = vec![0.0f32; v_rows];

    let wq_buf = XpuBuffer::from_slice(wq, false);
    let wk_buf = XpuBuffer::from_slice(wk, false);
    let wv_buf = XpuBuffer::from_slice(wv, false);
    let x_buf  = XpuBuffer::from_slice(x, false);
    let mut q_buf = XpuBuffer::new_device(q_rows  * 4);
    let mut k_buf = XpuBuffer::new_device(kv_rows * 4);
    let mut v_buf = XpuBuffer::new_device(v_rows  * 4);

    unsafe {
        xpu_ffi::dll_q4k_q6k_qkv_proj(
            wq_buf.as_ptr_type(),
            wk_buf.as_ptr_type(),
            wv_buf.as_ptr_type(),
            x_buf.as_ptr_type(),
            q_buf.as_mut_ptr_type(),
            k_buf.as_mut_ptr_type(),
            v_buf.as_mut_ptr_type(),
            q_rows  as u32,
            kv_rows as u32,
            v_rows  as u32,
            hidden  as u32,
        );
    }

    q_buf.copy_to_slice(&mut q_out);
    k_buf.copy_to_slice(&mut k_out);
    v_buf.copy_to_slice(&mut v_out);
    (q_out, k_out, v_out)
}

/// Per-projection QKV via separate `q4_vecmat` calls.
///
/// For fully mixed formats or when a fused path isn't available.
/// Each weight is Q4-packed f32-input path.
pub fn encode_per_proj(
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    x: &[f32],
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x_buf = XpuBuffer::from_slice(x, false);

    let dispatch = |w: &[u8], rows: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        let w_buf = XpuBuffer::from_slice(w, false);
        let mut out_buf = XpuBuffer::new_device(rows * 4);
        unsafe {
            xpu_ffi::q4_vecmat(
                w_buf.as_ptr_type(),
                x_buf.as_ptr_type(),
                out_buf.as_mut_ptr_type(),
                rows,
                hidden,
            );
        }
        out_buf.copy_to_slice(&mut out);
        out
    };

    let q_out = dispatch(wq, q_rows);
    let k_out = dispatch(wk, kv_rows);
    let v_out = dispatch(wv, kv_rows);
    (q_out, k_out, v_out)
}
