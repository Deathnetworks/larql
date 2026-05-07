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
    let out_size = n * std::mem::size_of::<f32>();

    let gate_buf = XpuBuffer::from_slice(gate_q4, false);
    let up_buf = XpuBuffer::from_slice(up_q4, false);
    let x_buf = XpuBuffer::from_slice(x, false);
    let mut g_out_buf = XpuBuffer::new_device(out_size);
    let mut u_out_buf = XpuBuffer::new_device(out_size);

    unsafe {
        xpu_ffi::q4k_ffn_gate_up(
            gate_buf.as_ptr_type(),
            up_buf.as_ptr_type(),
            x_buf.as_ptr_type(),
            g_out_buf.as_mut_ptr_type(),
            u_out_buf.as_mut_ptr_type(),
            n,
            k,
        );
    }

    let mut gate_out = vec![0.0f32; n];
    let mut up_out = vec![0.0f32; n];
    g_out_buf.copy_to_slice(&mut gate_out);
    u_out_buf.copy_to_slice(&mut up_out);

    (gate_out, up_out)
}

/// Batched gate+up for multiple sequence positions.
///
/// Loops over `seq_len` positions in `x_matrix` and calls `ffn_gate_up_fused`
/// per position. Unlike Metal which encodes all into one command buffer,
/// SYCL's fused kernel handles ordering internally — the loop here batches
/// at the Rust dispatch level.
///
/// - `gate_q4 / up_q4`: Q4K weights (shared across all positions)
/// - `x_matrix`: `[seq_len * k]` flattened row-major
/// - Returns: `(gate_results[seq_len][n], up_results[seq_len][n])`
pub fn pair_batch(
    gate_q4: &[u8],
    up_q4: &[u8],
    x_matrix: &[f32],
    seq_len: usize,
    num_rows: usize,
    hidden: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let out_size = num_rows * std::mem::size_of::<f32>();

    // Pre-upload weight buffers (reused across positions)
    let gate_buf = XpuBuffer::from_slice(gate_q4, false);
    let up_buf = XpuBuffer::from_slice(up_q4, false);

    let mut gate_results = Vec::with_capacity(seq_len);
    let mut up_results = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        let x_slice = &x_matrix[s * hidden..(s + 1) * hidden];
        let x_buf = XpuBuffer::from_slice(x_slice, false);
        let mut g_out_buf = XpuBuffer::new_device(out_size);
        let mut u_out_buf = XpuBuffer::new_device(out_size);

        unsafe {
            xpu_ffi::q4k_ffn_gate_up(
                gate_buf.as_ptr_type(),
                up_buf.as_ptr_type(),
                x_buf.as_ptr_type(),
                g_out_buf.as_mut_ptr_type(),
                u_out_buf.as_mut_ptr_type(),
                num_rows,
                hidden,
            );
        }

        let mut g = vec![0.0f32; num_rows];
        let mut u = vec![0.0f32; num_rows];
        g_out_buf.copy_to_slice(&mut g);
        u_out_buf.copy_to_slice(&mut u);
        gate_results.push(g);
        up_results.push(u);
    }

    (gate_results, up_results)
}
