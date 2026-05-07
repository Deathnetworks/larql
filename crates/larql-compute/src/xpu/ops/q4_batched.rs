//! Batched Q4 FFN operations for XPU.
//!
//! Uses the `q4k_ffn_gate_up` SYCL kernel to dispatch gate+up projections
//! in a single fused kernel call — amortising dispatch overhead vs two
//! separate `q4_vecmat` dispatches.
//!
//! Mirrors the Metal `pair_batch` but leverages the SYCL fused entrypoint
//! instead of encoding two separate command encoder dispatches.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Fused gate+up FFN projection using Q4K weights.
///
/// Dispatches both `gate_q4 @ x` and `up_q4 @ x` in a single SYCL
/// kernel invocation via `q4k_ffn_gate_up`.
///
/// - `gate_q4`: packed Q4K gate weight bytes `[n * k / 2]`
/// - `up_q4`:   packed Q4K up-projection weight bytes `[n * k / 2]`
/// - `x`:       f32 activation vector `[k]`
/// - `n`:       output rows (intermediate FFN dim)
/// - `k`:       input hidden size
/// - Returns: `(gate_out[n], up_out[n])` as f32 vecs
pub fn ffn_gate_up_fused(
    gate_q4: &[u8],
    up_q4: &[u8],
    x: &[f32],
    n: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let gate_buf = XpuBuffer::from_slice(gate_q4, false);
    let up_buf = XpuBuffer::from_slice(up_q4, false);
    let x_buf = XpuBuffer::from_slice(x, false);
    let mut g_out_buf = XpuBuffer::new_device(n * 4);
    let mut u_out_buf = XpuBuffer::new_device(n * 4);

    pair_batch_buf(
        &gate_buf, &up_buf, &x_buf,
        &mut g_out_buf, &mut u_out_buf,
        1, n, k
    );

    let mut gate_out = vec![0.0f32; n];
    let mut up_out = vec![0.0f32; n];
    g_out_buf.copy_to_slice(&mut gate_out);
    u_out_buf.copy_to_slice(&mut up_out);

    (gate_out, up_out)
}

/// Zero-copy Batched gate+up from existing buffers.
///
/// - `gate_q4 / up_q4`: Q4K weights
/// - `x_matrix`: `[seq_len * hidden]` in device memory
/// - `gate_results / up_results`: `[seq_len * num_rows]` in device memory
pub fn pair_batch_buf(
    gate_q4: &XpuBuffer,
    up_q4: &XpuBuffer,
    x_matrix: &XpuBuffer,
    gate_results: &mut XpuBuffer,
    up_results: &mut XpuBuffer,
    seq_len: usize,
    num_rows: usize,
    hidden: usize,
) {
    for s in 0..seq_len {
        let x_ptr = unsafe { x_matrix.as_ptr_type::<f32>().add(s * hidden) };
        let g_ptr = unsafe { gate_results.as_mut_ptr_type::<f32>().add(s * num_rows) };
        let u_ptr = unsafe { up_results.as_mut_ptr_type::<f32>().add(s * num_rows) };

        unsafe {
            xpu_ffi::q4k_ffn_gate_up(
                gate_q4.as_ptr_type(),
                up_q4.as_ptr_type(),
                x_ptr as *const f32,
                g_ptr as *mut f32,
                u_ptr as *mut f32,
                num_rows,
                hidden,
            );
        }
    }
}

/// Batched gate+up for multiple sequence positions.
pub fn pair_batch(
    gate_q4: &[u8],
    up_q4: &[u8],
    x_matrix: &[f32],
    seq_len: usize,
    num_rows: usize,
    hidden: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let gate_buf = XpuBuffer::from_slice(gate_q4, false);
    let up_buf = XpuBuffer::from_slice(up_q4, false);
    let x_buf = XpuBuffer::from_slice(x_matrix, false);
    
    let mut g_out_buf = XpuBuffer::new_device(seq_len * num_rows * 4);
    let mut u_out_buf = XpuBuffer::new_device(seq_len * num_rows * 4);

    pair_batch_buf(
        &gate_buf, &up_buf, &x_buf,
        &mut g_out_buf, &mut u_out_buf,
        seq_len, num_rows, hidden
    );

    let mut gate_results = Vec::with_capacity(seq_len);
    let mut up_results = Vec::with_capacity(seq_len);
    
    let mut g_flat = vec![0.0f32; seq_len * num_rows];
    let mut u_flat = vec![0.0f32; seq_len * num_rows];
    g_out_buf.copy_to_slice(&mut g_flat);
    u_out_buf.copy_to_slice(&mut u_flat);

    for s in 0..seq_len {
        gate_results.push(g_flat[s * num_rows..(s + 1) * num_rows].to_vec());
        up_results.push(u_flat[s * num_rows..(s + 1) * num_rows].to_vec());
    }

    (gate_results, up_results)
}
