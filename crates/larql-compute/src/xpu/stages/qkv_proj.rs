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
pub fn encode_fused_f32(
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    x: &[f32],
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let wq_buf = XpuBuffer::from_slice(wq, false);
    let wk_buf = XpuBuffer::from_slice(wk, false);
    let wv_buf = XpuBuffer::from_slice(wv, false);
    let x_buf  = XpuBuffer::from_slice(x, false);
    let mut q_buf = XpuBuffer::new_device(q_rows  * 4);
    let mut k_buf = XpuBuffer::new_device(kv_rows * 4);
    let mut v_buf = XpuBuffer::new_device(kv_rows * 4);

    encode_fused_f32_buf(
        &wq_buf, &wk_buf, &wv_buf,
        &x_buf,
        &mut q_buf, &mut k_buf, &mut v_buf,
        q_rows, kv_rows, hidden,
    );

    let mut q_out = vec![0.0f32; q_rows];
    let mut k_out = vec![0.0f32; kv_rows];
    let mut v_out = vec![0.0f32; kv_rows];
    q_buf.copy_to_slice(&mut q_out);
    k_buf.copy_to_slice(&mut k_out);
    v_buf.copy_to_slice(&mut v_out);
    (q_out, k_out, v_out)
}

/// Zero-copy Fused Q4_K QKV projection from existing buffers.
pub fn encode_fused_f32_buf(
    wq: &XpuBuffer,
    wk: &XpuBuffer,
    wv: &XpuBuffer,
    x: &XpuBuffer,
    q_out: &mut XpuBuffer,
    k_out: &mut XpuBuffer,
    v_out: &mut XpuBuffer,
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) {
    unsafe {
        xpu_ffi::q4k_qkv_proj(
            wq.as_ptr_type(),
            wk.as_ptr_type(),
            wv.as_ptr_type(),
            x.as_ptr_type(),
            q_out.as_mut_ptr_type(),
            k_out.as_mut_ptr_type(),
            v_out.as_mut_ptr_type(),
            q_rows  as u32,
            kv_rows as u32,
            kv_rows as u32,
            hidden  as u32,
        );
    }
}

/// Fused Q4_K Q/K + Q6_K V QKV projection.
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
    let wq_buf = XpuBuffer::from_slice(wq, false);
    let wk_buf = XpuBuffer::from_slice(wk, false);
    let wv_buf = XpuBuffer::from_slice(wv, false);
    let x_buf  = XpuBuffer::from_slice(x, false);
    let mut q_buf = XpuBuffer::new_device(q_rows  * 4);
    let mut k_buf = XpuBuffer::new_device(kv_rows * 4);
    let mut v_buf = XpuBuffer::new_device(v_rows  * 4);

    encode_fused_q4k_q6k_buf(
        &wq_buf, &wk_buf, &wv_buf,
        &x_buf,
        &mut q_buf, &mut k_buf, &mut v_buf,
        q_rows, kv_rows, v_rows, hidden,
    );

    let mut q_out = vec![0.0f32; q_rows];
    let mut k_out = vec![0.0f32; kv_rows];
    let mut v_out = vec![0.0f32; v_rows];
    q_buf.copy_to_slice(&mut q_out);
    k_buf.copy_to_slice(&mut k_out);
    v_buf.copy_to_slice(&mut v_out);
    (q_out, k_out, v_out)
}

/// Zero-copy Fused Q4_K Q/K + Q6_K V projection from existing buffers.
pub fn encode_fused_q4k_q6k_buf(
    wq: &XpuBuffer,
    wk: &XpuBuffer,
    wv: &XpuBuffer,
    x: &XpuBuffer,
    q_out: &mut XpuBuffer,
    k_out: &mut XpuBuffer,
    v_out: &mut XpuBuffer,
    q_rows: usize,
    kv_rows: usize,
    v_rows: usize,
    hidden: usize,
) {
    unsafe {
        xpu_ffi::dll_q4k_q6k_qkv_proj(
            wq.as_ptr_type(),
            wk.as_ptr_type(),
            wv.as_ptr_type(),
            x.as_ptr_type(),
            q_out.as_mut_ptr_type(),
            k_out.as_mut_ptr_type(),
            v_out.as_mut_ptr_type(),
            q_rows  as u32,
            kv_rows as u32,
            v_rows  as u32,
            hidden  as u32,
        );
    }
}

/// Per-projection QKV via separate `q4_vecmat` calls.
pub fn encode_per_proj(
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    x: &[f32],
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let wq_buf = XpuBuffer::from_slice(wq, false);
    let wk_buf = XpuBuffer::from_slice(wk, false);
    let wv_buf = XpuBuffer::from_slice(wv, false);
    let x_buf  = XpuBuffer::from_slice(x, false);
    let mut q_buf = XpuBuffer::new_device(q_rows  * 4);
    let mut k_buf = XpuBuffer::new_device(kv_rows * 4);
    let mut v_buf = XpuBuffer::new_device(kv_rows * 4);

    encode_per_proj_buf(
        &wq_buf, &wk_buf, &wv_buf,
        &x_buf,
        &mut q_buf, &mut k_buf, &mut v_buf,
        q_rows, kv_rows, hidden,
    );

    let mut q_out = vec![0.0f32; q_rows];
    let mut k_out = vec![0.0f32; kv_rows];
    let mut v_out = vec![0.0f32; kv_rows];
    q_buf.copy_to_slice(&mut q_out);
    k_buf.copy_to_slice(&mut k_out);
    v_buf.copy_to_slice(&mut v_out);
    (q_out, k_out, v_out)
}

/// Zero-copy Per-projection QKV from existing buffers.
pub fn encode_per_proj_buf(
    wq: &XpuBuffer,
    wk: &XpuBuffer,
    wv: &XpuBuffer,
    x: &XpuBuffer,
    q_out: &mut XpuBuffer,
    k_out: &mut XpuBuffer,
    v_out: &mut XpuBuffer,
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) {
    let mut dispatch = |w: &XpuBuffer, rows: usize, out: &mut XpuBuffer| {
        unsafe {
            xpu_ffi::q4_vecmat(
                w.as_ptr_type(),
                x.as_ptr_type(),
                out.as_mut_ptr_type(),
                rows,
                hidden,
            );
        }
    };

    dispatch(wq, q_rows, q_out);
    dispatch(wk, kv_rows, k_out);
    dispatch(wv, kv_rows, v_out);
}
