use crate::backend::QuantMatVec;
use crate::xpu::XpuBackend;
use crate::xpu::ops::{q4_matvec, q4k_matvec, q6k_matvec};

impl QuantMatVec for XpuBackend {
    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let buf_x = self.bufs.get_f32(activation);
        let buf_w = self.bufs.get_u8(q4_data);
        let mut buf_out = self.bufs.output(n * 4);

        q4_matvec::dispatch_buf(&buf_w, &buf_x, &mut buf_out, n, k);

        let mut out = vec![0.0f32; n];
        buf_out.copy_to_slice(&mut out);
        self.bufs.recycle(buf_out);
        Some(out)
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let buf_x = self.bufs.get_f32(x);
        let buf_w = self.bufs.get_u8(q4k_data);
        let mut buf_out = self.bufs.output(num_rows * 4);

        q4k_matvec::dispatch_buf(&buf_w, &buf_x, &mut buf_out, num_rows, hidden);

        let mut out = vec![0.0f32; num_rows];
        buf_out.copy_to_slice(&mut out);
        self.bufs.recycle(buf_out);
        Some(out)
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let buf_x = self.bufs.get_f32(x);
        let buf_w = self.bufs.get_u8(q6k_data);
        let mut buf_out = self.bufs.output(num_rows * 4);

        q6k_matvec::dispatch_buf(&buf_w, &buf_x, &mut buf_out, num_rows, hidden);

        let mut out = vec![0.0f32; num_rows];
        buf_out.copy_to_slice(&mut out);
        self.bufs.recycle(buf_out);
        Some(out)
    }
}
