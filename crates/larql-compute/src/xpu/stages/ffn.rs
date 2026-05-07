//! Feed-forward block for XPU.
//!
//! XPU equivalent of Metal's `stages::ffn`. Two variants:
//!
//! - **Gated** (`encode_gated`): `down(SiLU(gate) ⊙ up)` — Llama / Gemma / Qwen.
//!   Uses the fused `q4k_ffn_gate_up` + `dll_geglu_silu` + `q4k_proj` path.
//! - **Standard** (`encode_standard`): `down(act(up))` — StarCoder2.
//!   Uses `q4_vecmat` → `dll_silu` → `q4_vecmat`.
//!
//! All ops are single-position; callers loop for multi-position prefill.

use crate::xpu::ffi::ffi as xpu_ffi;
use crate::xpu::buffers::XpuBuffer;
use super::quant_matvec::{self, QuantFormat};

/// Activation function for FFN.
#[derive(Clone, Copy)]
pub enum Activation {
    SiLU,
    GeluTanh,
}

/// Gated FFN (Llama / Gemma / Qwen): `down(act(gate) * up)`.
///
/// Uses the fused `q4k_ffn_gate_up` kernel for gate+up in one dispatch,
/// then `dll_geglu_silu` / `dll_geglu_gelu_tanh` for activation,
/// then `q4k_proj` for down projection.
///
/// - `gate/up/down_w`: quantized weight bytes
/// - `x_norm`: f32 normed activation `[hidden]`
/// - `gate_fmt / down_fmt`: format for routing down dispatch
/// - `inter`: intermediate FFN dimension
/// - `hidden`: output dimension
/// - Returns: f32 output `[hidden]`
#[allow(clippy::too_many_arguments)]
pub fn encode_gated(
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    x_norm: &[f32],
    gate_fmt: QuantFormat,
    down_fmt: QuantFormat,
    activation: Activation,
    inter: usize,
    hidden: usize,
) -> Vec<f32> {
    // 1. Gate + Up via fused kernel
    let xg_buf = XpuBuffer::from_slice(gate_w, false);
    let xu_buf = XpuBuffer::from_slice(up_w, false);
    let x_buf  = XpuBuffer::from_slice(x_norm, false);
    let mut g_out = XpuBuffer::new_device(inter * 4);
    let mut u_out = XpuBuffer::new_device(inter * 4);

    // Use fused path for Q4K gate/up, fall back to two q4_vecmat calls otherwise
    if matches!(gate_fmt, QuantFormat::Q4K | QuantFormat::Q4KF) {
        unsafe {
            xpu_ffi::q4k_ffn_gate_up(
                xg_buf.as_ptr_type(),
                xu_buf.as_ptr_type(),
                x_buf.as_ptr_type(),
                g_out.as_mut_ptr_type(),
                u_out.as_mut_ptr_type(),
                inter,
                x_norm.len(),
            );
        }
    } else {
        unsafe {
            xpu_ffi::q4_vecmat(
                xg_buf.as_ptr_type(), x_buf.as_ptr_type(),
                g_out.as_mut_ptr_type(), inter, x_norm.len(),
            );
            xpu_ffi::q4_vecmat(
                xu_buf.as_ptr_type(), x_buf.as_ptr_type(),
                u_out.as_mut_ptr_type(), inter, x_norm.len(),
            );
        }
    }

    // 2. Activation (SiLU-GEGLU or GELU-tanh-GEGLU)
    let mut act_out = XpuBuffer::new_device(inter * 4);
    unsafe {
        match activation {
            Activation::SiLU =>
                xpu_ffi::dll_geglu_silu(
                    g_out.as_ptr_type(), u_out.as_ptr_type(),
                    act_out.as_mut_ptr_type(), inter,
                ),
            Activation::GeluTanh =>
                xpu_ffi::dll_geglu_gelu_tanh(
                    g_out.as_ptr_type(), u_out.as_ptr_type(),
                    act_out.as_mut_ptr_type(), inter,
                ),
        }
    }

    // 3. Down projection via format-aware dispatch
    let mut act_slice = vec![0.0f32; inter];
    act_out.copy_to_slice(&mut act_slice);
    quant_matvec::encode(down_w, &act_slice, hidden, inter, down_fmt)
}

/// Standard (non-gated) FFN: `down(act(up))` — StarCoder2.
#[allow(clippy::too_many_arguments)]
pub fn encode_standard(
    up_w: &[u8],
    down_w: &[u8],
    x_norm: &[f32],
    up_fmt: QuantFormat,
    down_fmt: QuantFormat,
    activation: Activation,
    inter: usize,
    hidden: usize,
) -> Vec<f32> {
    // 1. Up projection
    let up_out = quant_matvec::encode(up_w, x_norm, inter, x_norm.len(), up_fmt);

    // 2. Activation in-place
    let mut act_slice = vec![0.0f32; inter];
    {
        let up_buf = XpuBuffer::from_slice(&up_out, false);
        let dummy  = XpuBuffer::from_slice(&up_out, false);
        let mut act_out = XpuBuffer::new_device(inter * 4);
        unsafe {
            match activation {
                Activation::SiLU =>
                    xpu_ffi::dll_geglu_silu(
                        up_buf.as_ptr_type(), dummy.as_ptr_type(),
                        act_out.as_mut_ptr_type(), inter,
                    ),
                Activation::GeluTanh =>
                    xpu_ffi::dll_geglu_gelu_tanh(
                        up_buf.as_ptr_type(), dummy.as_ptr_type(),
                        act_out.as_mut_ptr_type(), inter,
                    ),
            }
        }
        act_out.copy_to_slice(&mut act_slice);
    }

    // 3. Down projection
    quant_matvec::encode(down_w, &act_slice, hidden, inter, down_fmt)
}
