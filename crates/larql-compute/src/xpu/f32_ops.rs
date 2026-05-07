//! f32 matmul operations via XPU (SYCL) compute kernels.

use ndarray::{Array2, ArrayView2};
use super::buffers::{BufferCache, XpuBuffer};
use super::ffi::ffi as xpu_ffi;

/// Dispatch parameters for f32 matmul.
pub struct F32Ops {}

impl F32Ops {
    pub fn new() -> Self {
        Self {}
    }

    /// C = A × B  (A: [m,k], B: [k,n], C: [m,n])
    pub fn dispatch_notrans(
        &self,
        bufs: &BufferCache,
        a_data: &[f32],
        b_data: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let buf_a = bufs.get_f32(a_data);
        let buf_b = bufs.get_f32(b_data);
        let mut buf_c = bufs.output(m * n * 4);

        self.dispatch_notrans_buf(&buf_a, &buf_b, &mut buf_c, m, n, k);

        let mut c = vec![0.0f32; m * n];
        buf_c.copy_to_slice(&mut c);
        bufs.recycle(buf_c);
        c
    }

    /// Zero-copy C = A × B
    pub fn dispatch_notrans_buf(
        &self,
        a: &XpuBuffer,
        b: &XpuBuffer,
        c: &mut XpuBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        unsafe {
            xpu_ffi::dll_sgemm(
                a.as_ptr_type(),
                b.as_ptr_type(),
                c.as_mut_ptr_type(),
                m as u32,
                n as u32,
                k as u32,
            );
        }
    }

    /// C = A × B^T  (A: [m,k], B: [n,k], C: [m,n])
    pub fn dispatch_transb(
        &self,
        bufs: &BufferCache,
        a_data: &[f32],
        b_data: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let buf_a = bufs.get_f32(a_data);
        let buf_b = bufs.get_f32(b_data);
        let mut buf_c = bufs.output(m * n * 4);

        self.dispatch_transb_buf(&buf_a, &buf_b, &mut buf_c, m, n, k);

        let mut c = vec![0.0f32; m * n];
        buf_c.copy_to_slice(&mut c);
        bufs.recycle(buf_c);
        c
    }

    /// Zero-copy C = A × B^T
    pub fn dispatch_transb_buf(
        &self,
        a: &XpuBuffer,
        b: &XpuBuffer,
        c: &mut XpuBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        unsafe {
            xpu_ffi::dll_sgemm_transb(
                a.as_ptr_type(),
                b.as_ptr_type(),
                c.as_mut_ptr_type(),
                m as u32,
                n as u32,
                k as u32,
            );
        }
    }

    /// f32 matmul with automatic GPU/CPU routing.
    pub fn matmul(
        &self,
        bufs: &BufferCache,
        a: ArrayView2<f32>,
        b: ArrayView2<f32>,
        flop_threshold: usize,
    ) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        if 2 * m * n * k < flop_threshold {
            return a.dot(&b);
        }

        let a_owned;
        let a_data: &[f32] = match a.as_slice() {
            Some(s) => s,
            None => {
                a_owned = a.as_standard_layout().into_owned();
                a_owned.as_slice().unwrap()
            }
        };
        let b_owned;
        let b_data: &[f32] = match b.as_slice() {
            Some(s) => s,
            None => {
                b_owned = b.as_standard_layout().into_owned();
                b_owned.as_slice().unwrap()
            }
        };

        let c = self.dispatch_notrans(bufs, a_data, b_data, m, n, k);
        Array2::from_shape_vec((m, n), c).unwrap()
    }

    /// f32 matmul_transb with automatic GPU/CPU routing.
    pub fn matmul_transb(
        &self,
        bufs: &BufferCache,
        a: ArrayView2<f32>,
        b: ArrayView2<f32>,
        flop_threshold: usize,
    ) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[0];
        if 2 * m * n * k < flop_threshold {
            return a.dot(&b.t());
        }

        let a_owned;
        let a_data: &[f32] = match a.as_slice() {
            Some(s) => s,
            None => {
                a_owned = a.as_standard_layout().into_owned();
                a_owned.as_slice().unwrap()
            }
        };
        let b_owned;
        let b_data: &[f32] = match b.as_slice() {
            Some(s) => s,
            None => {
                b_owned = b.as_standard_layout().into_owned();
                b_owned.as_slice().unwrap()
            }
        };

        let c = self.dispatch_transb(bufs, a_data, b_data, m, n, k);
        Array2::from_shape_vec((m, n), c).unwrap()
    }
}
