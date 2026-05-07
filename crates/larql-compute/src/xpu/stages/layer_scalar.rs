//! Per-layer residual scalar — Gemma 4's learned stabiliser.
//!
//! XPU equivalent of Metal's `stages::layer_scalar`. Multiplies the
//! residual stream by a per-layer scalar using `dll_residual_ops`
//! mode 2 (scale in-place).
//!
//! Used by Gemma 4 models which have a learned per-layer weight
//! typically in the range 0.02–0.8 to stabilise residual magnitudes.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;

/// Scalar mode for dll_residual_ops: multiply in-place (a * scalar → out).
const SCALE_MODE: u32 = 2;

/// Scale the f32 residual at each position by `scalar`.
///
/// No-ops when `scalar == 0.0`.
///
/// - `h`: mutable residual slice `[seq_len × hidden]`
/// - `seq_len` / `hidden`: shape
/// - `scalar`: per-layer learned scale factor
pub fn encode(h: &mut [f32], seq_len: usize, hidden: usize, scalar: f32) {
    if scalar == 0.0 {
        return;
    }

    for pos in 0..seq_len {
        let off = pos * hidden;
        let slice = &mut h[off..off + hidden];

        let in_buf = XpuBuffer::from_slice(slice as &[f32], false);
        // Use same buffer as both src and dst via a dummy second arg.
        let mut out_buf = XpuBuffer::new_device(hidden * std::mem::size_of::<f32>());

        unsafe {
            xpu_ffi::dll_residual_ops(
                in_buf.as_ptr_type(),
                in_buf.as_ptr_type(), // second arg unused in scale mode
                out_buf.as_mut_ptr_type(),
                hidden as u32,
                scalar,
                SCALE_MODE,
            );
        }

        out_buf.copy_to_slice(slice);
    }
}
