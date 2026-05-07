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
    let xg_buf = XpuBuffer::from_slice(gate_w, false);
    let xu_buf = XpuBuffer::from_slice(up_w, false);
    let xd_buf = XpuBuffer::from_slice(down_w, false);
    let x_buf  = XpuBuffer::from_slice(x_norm, false);
    let mut out_buf = XpuBuffer::new_device(hidden * 4);

    encode_gated_buf(
        &xg_buf, &xu_buf, &xd_buf, &x_buf, &mut out_buf,
        gate_fmt, down_fmt, activation, inter, hidden,
    );

    let mut out = vec![0.0f32; hidden];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Gated FFN from existing buffers.
#[allow(clippy::too_many_arguments)]
pub fn encode_gated_buf(
    gate_w: &XpuBuffer,
    up_w: &XpuBuffer,
    down_w: &XpuBuffer,
    x_norm: &XpuBuffer,
    out: &mut XpuBuffer,
    gate_fmt: QuantFormat,
    down_fmt: QuantFormat,
    activation: Activation,
    inter: usize,
    hidden: usize,
) {
    // 1. Gate + Up
    let mut g_out = XpuBuffer::new_device(inter * 4);
    let mut u_out = XpuBuffer::new_device(inter * 4);

    if matches!(gate_fmt, QuantFormat::Q4K | QuantFormat::Q4KF) {
        unsafe {
            xpu_ffi::q4k_ffn_gate_up(
                gate_w.as_ptr_type(),
                up_w.as_ptr_type(),
                x_norm.as_ptr_type(),
                g_out.as_mut_ptr_type(),
                u_out.as_mut_ptr_type(),
                inter,
                x_norm.size() / 4,
            );
        }
    } else {
        unsafe {
            xpu_ffi::q4_vecmat(
                gate_w.as_ptr_type(), x_norm.as_ptr_type(),
                g_out.as_mut_ptr_type(), inter, x_norm.size() / 4,
            );
            xpu_ffi::q4_vecmat(
                up_w.as_ptr_type(), x_norm.as_ptr_type(),
                u_out.as_mut_ptr_type(), inter, x_norm.size() / 4,
            );
        }
    }

    // 2. Activation
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

    // 3. Down projection
    quant_matvec::encode_buf(down_w, &act_out, out, hidden, inter, down_fmt);
}

/// Standard (non-gated) FFN: `down(act(up))` — StarCoder2.
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
    let xu_buf = XpuBuffer::from_slice(up_w, false);
    let xd_buf = XpuBuffer::from_slice(down_w, false);
    let x_buf  = XpuBuffer::from_slice(x_norm, false);
    let mut out_buf = XpuBuffer::new_device(hidden * 4);

    encode_standard_buf(
        &xu_buf, &xd_buf, &x_buf, &mut out_buf,
        up_fmt, down_fmt, activation, inter, hidden,
    );

    let mut out = vec![0.0f32; hidden];
    out_buf.copy_to_slice(&mut out);
    out
}

/// Zero-copy Standard FFN from existing buffers.
#[allow(clippy::too_many_arguments)]
pub fn encode_standard_buf(
    up_w: &XpuBuffer,
    down_w: &XpuBuffer,
    x_norm: &XpuBuffer,
    out: &mut XpuBuffer,
    up_fmt: QuantFormat,
    down_fmt: QuantFormat,
    activation: Activation,
    inter: usize,
    hidden: usize,
) {
    // 1. Up projection
    let mut up_out = XpuBuffer::new_device(inter * 4);
    quant_matvec::encode_buf(up_w, x_norm, &mut up_out, inter, x_norm.size() / 4, up_fmt);

    // 2. Activation
    let mut act_out = XpuBuffer::new_device(inter * 4);
    let dummy = XpuBuffer::new_device(0); // Geglu needs two inputs, standard ffn has one
    unsafe {
        match activation {
            Activation::SiLU =>
                xpu_ffi::dll_geglu_silu(
                    up_out.as_ptr_type(), dummy.as_ptr_type(),
                    act_out.as_mut_ptr_type(), inter,
                ),
            Activation::GeluTanh =>
                xpu_ffi::dll_geglu_gelu_tanh(
                    up_out.as_ptr_type(), dummy.as_ptr_type(),
                    act_out.as_mut_ptr_type(), inter,
                ),
        }
    }

    // 3. Down projection
    quant_matvec::encode_buf(down_w, &act_out, out, hidden, inter, down_fmt);
}
