//! MoE Combine for XPU decode pipeline.

use crate::xpu::XpuBackend;
use crate::xpu::buffers::XpuBuffer;
use crate::FullPipelineLayer;
use crate::cpu::ops::outer_combine::{apply_layer_scalar_in_place, outer_post_norm_residual};

impl XpuBackend {
    pub(super) fn handle_moe_combine(
        &self,
        layer: &FullPipelineLayer,
        new_h: &mut XpuBuffer,
        h_post_attn: &XpuBuffer,
        hidden: usize,
    ) {
        if std::env::var("SKIP_OUTER_NORM").is_ok() {
            return;
        }

        // 1. Sync and read to host
        let mut new_h_host = vec![0.0f32; hidden];
        let mut h_pa_host = vec![0.0f32; hidden];
        new_h.copy_to_slice(&mut new_h_host);
        h_post_attn.copy_to_slice(&mut h_pa_host);

        // 2. Step A: Outer post-FFN norm
        if layer.moe_combined_output_norm {
            let outer_w_id = layer.moe_outer_post_norm.or(layer.post_ffn_norm);
            let outer_w = outer_w_id.map(|w| self.bufs.get_f32(w));
            
            let h1_plus_h2: Vec<f32> = new_h_host
                .iter()
                .zip(h_pa_host.iter())
                .map(|(&n, &ha)| n - ha)
                .collect();

            // We need the weight slice from the XpuBuffer.
            // Since we are on CPU here, we'd need to copy the weight to host too.
            // For now, let's assume we can get it from the original slice if it's available,
            // but XpuBackend.bufs.get_f32(w) returns an Arc<XpuBuffer>.
            // I'll add a helper to XpuBuffer to get a host slice if it's not already there.
            // Or just copy_to_slice into a temporary vec.
            
            let mut w_host = vec![0.0f32; hidden];
            if let Some(w_buf) = outer_w {
                w_buf.copy_to_slice(&mut w_host);
            }

            let combined = outer_post_norm_residual(
                &h_pa_host,
                &h1_plus_h2,
                if outer_w_id.is_some() { Some(&w_host) } else { None },
                layer.norm_offset,
                layer.eps,
            );
            new_h_host.copy_from_slice(&combined);
        }

        // 3. Step B: Layer scalar
        apply_layer_scalar_in_place(&mut new_h_host, layer.layer_scalar);

        // 4. Copy back to device
        new_h.copy_from_slice(&new_h_host);
    }
}
