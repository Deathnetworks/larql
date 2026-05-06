use crate::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp, QuantMatVec};
use crate::pipeline::{QuantFormat, QuantWeight};
use std::any::Any;
use ndarray::{Array2, ArrayView2};

mod ffi;
pub mod buffers;

use ffi::ffi as xpu_ffi;
use buffers::XpuBuffer;

pub struct XpuBackend {}

impl XpuBackend {
    pub fn new() -> Option<Self> {
        if xpu_ffi::init_xpu() {
            Some(Self {})
        } else {
            None
        }
    }
}

impl ComputeBackend for XpuBackend {
    fn name(&self) -> &str {
        "XPU (SYCL)"
    }

    fn device_info(&self) -> String {
        xpu_ffi::get_device_info()
    }

    fn supports(&self, cap: Capability) -> bool {
        match cap {
            Capability::F32Gemv => true,
            Capability::QuantMatVec => true,
            Capability::Q4VecMat => true,
            _ => false,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MatMul for XpuBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        crate::cpu::ops::f32_matmul::matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        crate::cpu::ops::f32_matmul::matmul_transb(a, b)
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.nrows(), w.ncols());
        if x.len() != k {
            return None;
        }

        let mut out = vec![0.0f32; n];
        
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut w_buf = XpuBuffer::from_slice(w.as_slice()?, false);
        let mut out_buf = XpuBuffer::new_device(n * 4);

        unsafe {
            xpu_ffi::f32_gemv(
                x_buf.as_ptr_type(),
                w_buf.as_ptr_type(),
                out_buf.as_mut_ptr_type(),
                n,
                k,
            );
        }
        
        out_buf.copy_to_slice(&mut out);
        Some(out)
    }
}

impl QuantMatVec for XpuBackend {
    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let mut out = vec![0.0f32; k];
        
        let mut act_buf = XpuBuffer::from_slice(activation, false);
        let mut q4_buf = XpuBuffer::from_slice(q4_data, false);
        let mut out_buf = XpuBuffer::new_device(k * 4);

        unsafe {
            xpu_ffi::q4_vecmat(
                act_buf.as_ptr_type(),
                q4_buf.as_ptr(),
                out_buf.as_mut_ptr_type(),
                n,
                k,
            );
        }

        out_buf.copy_to_slice(&mut out);
        Some(out)
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let mut out = vec![0.0f32; num_rows];
        
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut w_buf = XpuBuffer::from_slice(q4k_data, false);
        let mut out_buf = XpuBuffer::new_device(num_rows * 4);

        unsafe {
            xpu_ffi::q4k_matvec_8sg(
                w_buf.as_ptr(),
                x_buf.as_ptr_type(),
                out_buf.as_mut_ptr_type(),
                num_rows,
                hidden,
            );
        }

        out_buf.copy_to_slice(&mut out);
        Some(out)
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let mut out = vec![0.0f32; num_rows];
        
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut w_buf = XpuBuffer::from_slice(q6k_data, false);
        let mut out_buf = XpuBuffer::new_device(num_rows * 4);

        unsafe {
            xpu_ffi::q6k_matvec(
                w_buf.as_ptr(),
                x_buf.as_ptr_type(),
                out_buf.as_mut_ptr_type(),
                num_rows,
                hidden,
            );
        }

        out_buf.copy_to_slice(&mut out);
        Some(out)
    }
}

impl DecodeBackend for XpuBackend {}

impl XpuBackend {
    pub fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        offset: f32,
    ) -> Vec<f32> {
        let n = x.len();
        let mut out = vec![0.0f32; n];
        
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut w_buf = XpuBuffer::from_slice(weight, false);
        let mut out_buf = XpuBuffer::new_device(n * 4);

        unsafe {
            xpu_ffi::rms_norm(
                x_buf.as_ptr_type(),
                w_buf.as_ptr_type(),
                out_buf.as_mut_ptr_type(),
                n,
                eps,
                offset,
            );
        }

        out_buf.copy_to_slice(&mut out);
        out
    }

    pub fn silu(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mut out = vec![0.0f32; n];
        
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut out_buf = XpuBuffer::new_device(n * 4);

        unsafe {
            xpu_ffi::silu(x_buf.as_ptr_type(), out_buf.as_mut_ptr_type(), n);
        }

        out_buf.copy_to_slice(&mut out);
        out
    }

    pub fn gelu_tanh(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mut out = vec![0.0f32; n];
        
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut out_buf = XpuBuffer::new_device(n * 4);

        unsafe {
            xpu_ffi::gelu_tanh(x_buf.as_ptr_type(), out_buf.as_mut_ptr_type(), n);
        }

        out_buf.copy_to_slice(&mut out);
        out
    }

    pub fn rope_at_pos(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        head_dim: usize,
        rope_base: f32,
        pos: usize,
        rotary_dim: usize,
    ) {
        let num_q = q.len() / head_dim;
        let num_kv = k.len() / head_dim;

        let mut q_buf = XpuBuffer::from_slice(q, false);
        let mut k_buf = XpuBuffer::from_slice(k, false);

        unsafe {
            xpu_ffi::rope_at_pos_batched_qk(
                q_buf.as_mut_ptr_type(),
                k_buf.as_mut_ptr_type(),
                head_dim,
                rope_base,
                pos,
                rotary_dim,
                num_q,
                num_kv,
            );
        }

        q_buf.copy_to_slice(q);
        k_buf.copy_to_slice(k);
    }

    pub fn q4k_ffn_gate_up(
        &self,
        wg: &[u8],
        wu: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut g_out = vec![0.0f32; num_rows];
        let mut u_out = vec![0.0f32; num_rows];

        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut wg_buf = XpuBuffer::from_slice(wg, false);
        let mut wu_buf = XpuBuffer::from_slice(wu, false);
        let mut g_out_buf = XpuBuffer::new_device(num_rows * 4);
        let mut u_out_buf = XpuBuffer::new_device(num_rows * 4);

        unsafe {
            xpu_ffi::q4k_ffn_gate_up(
                wg_buf.as_ptr(),
                wu_buf.as_ptr(),
                x_buf.as_ptr_type(),
                g_out_buf.as_mut_ptr_type(),
                u_out_buf.as_mut_ptr_type(),
                num_rows,
                hidden,
            );
        }

        g_out_buf.copy_to_slice(&mut g_out);
        u_out_buf.copy_to_slice(&mut u_out);
        (g_out, u_out)
    }

    pub fn q4_matvec(
        &self,
        q4: &[u8],
        q8: &[i8],
        q8s: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; num_rows];
        let mut q4_buf = XpuBuffer::from_slice(q4, false);
        let mut q8_buf = XpuBuffer::from_slice(q8, false);
        let mut q8s_buf = XpuBuffer::from_slice(q8s, false);
        let mut out_buf = XpuBuffer::new_device(num_rows * 4);

        unsafe {
            xpu_ffi::q4_matvec_v4(
                q4_buf.as_ptr(),
                q8_buf.as_ptr() as *const i8,
                q8s_buf.as_ptr() as *const f32,
                out_buf.as_mut_ptr_type(),
                num_rows,
                hidden,
            );
        }
        out_buf.copy_to_slice(&mut out);
        out
    }

    pub fn q4k_qkv_proj(
        &self,
        wq: &[u8],
        wk: &[u8],
        wv: &[u8],
        x: &[f32],
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
        hidden: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q_out = vec![0.0f32; q_rows];
        let mut k_out = vec![0.0f32; k_rows];
        let mut v_out = vec![0.0f32; v_rows];

        let mut wq_buf = XpuBuffer::from_slice(wq, false);
        let mut wk_buf = XpuBuffer::from_slice(wk, false);
        let mut wv_buf = XpuBuffer::from_slice(wv, false);
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut q_out_buf = XpuBuffer::new_device(q_rows * 4);
        let mut k_out_buf = XpuBuffer::new_device(k_rows * 4);
        let mut v_out_buf = XpuBuffer::new_device(v_rows * 4);

        unsafe {
            xpu_ffi::q4k_qkv_proj(
                wq_buf.as_ptr(),
                wk_buf.as_ptr(),
                wv_buf.as_ptr(),
                x_buf.as_ptr_type(),
                q_out_buf.as_mut_ptr_type(),
                k_out_buf.as_mut_ptr_type(),
                v_out_buf.as_mut_ptr_type(),
                q_rows as u32,
                k_rows as u32,
                v_rows as u32,
                hidden as u32,
            );
        }

        q_out_buf.copy_to_slice(&mut q_out);
        k_out_buf.copy_to_slice(&mut k_out);
        v_out_buf.copy_to_slice(&mut v_out);
        (q_out, k_out, v_out)
    }

    pub fn q4k_proj(
        &self,
        w4k: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; num_rows];
        let mut w4k_buf = XpuBuffer::from_slice(w4k, false);
        let mut x_buf = XpuBuffer::from_slice(x, false);
        let mut out_buf = XpuBuffer::new_device(num_rows * 4);

        unsafe {
            xpu_ffi::q4k_proj(
                w4k_buf.as_ptr(),
                x_buf.as_ptr_type(),
                out_buf.as_mut_ptr_type(),
                num_rows,
                hidden,
            );
        }
        out_buf.copy_to_slice(&mut out);
        out
    }
}

impl DecodeBackend for XpuBackend {
    fn decode_token(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        q_weight: &[f32],
        k_weight: &[f32],
        t: usize,
        head_dim: usize,
        num_q: usize,
        num_kv: usize,
        scale: f32,
        window_size: usize,
        eps: f32,
        qk_offset: f32,
        rope_base: f32,
        rotary_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; num_q * head_dim];
        
        let mut q_in_buf = XpuBuffer::from_slice(q, false);
        let mut k_in_buf = XpuBuffer::from_slice(k, false);
        let mut v_in_buf = XpuBuffer::from_slice(v, false);
        let mut k_cache_buf = XpuBuffer::from_slice(k_cache, false);
        let mut v_cache_buf = XpuBuffer::from_slice(v_cache, false);
        let mut out_buf = XpuBuffer::new_device(out.len() * 4);
        let mut q_weight_buf = XpuBuffer::from_slice(q_weight, false);
        let mut k_weight_buf = XpuBuffer::from_slice(k_weight, false);

        unsafe {
            xpu_ffi::attn_fused(
                q_in_buf.as_ptr_type(),
                k_in_buf.as_ptr_type(),
                v_in_buf.as_ptr_type(),
                k_cache_buf.as_mut_ptr_type(),
                v_cache_buf.as_mut_ptr_type(),
                out_buf.as_mut_ptr_type(),
                q_weight_buf.as_ptr_type(),
                k_weight_buf.as_ptr_type(),
                t as u32,
                head_dim as u32,
                num_q as u32,
                num_kv as u32,
                scale,
                window_size as u32,
                eps,
                qk_offset,
                rope_base,
                rotary_dim as u32,
            );
        }

        k_cache_buf.copy_to_slice(k_cache);
        v_cache_buf.copy_to_slice(v_cache);
        out_buf.copy_to_slice(&mut out);
        out
    }
}
