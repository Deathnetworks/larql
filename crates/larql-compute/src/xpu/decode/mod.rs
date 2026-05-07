use super::*;
use crate::xpu::buffers::XpuBuffer;

mod encode_attn;
mod encode_ffn;
mod encode_post_ffn;
mod encode_qkv;
pub mod gpu_timing;
mod moe_combine;
mod moe_interleave;
pub mod profile;

pub use profile::ProfileTimings;

pub(crate) const DEFAULT_KV_CACHE_MAX_SEQ: usize = 4096;

impl XpuBackend {
    pub fn create_kv_cache(
        &self,
        num_layers: usize,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(num_layers, max_seq, num_kv_heads, head_dim)
    }

    pub fn create_kv_cache_per_layer(
        &self,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new_per_layer(shapes, max_seq)
    }

    pub(crate) fn kv_shapes_for_layers(
        layers: &[crate::FullPipelineLayer<'_>],
    ) -> Vec<(usize, usize)> {
        layers
            .iter()
            .map(|layer| (layer.num_kv_heads, layer.head_dim))
            .collect()
    }

    pub(crate) fn ensure_kv_cache_for_layers<'a>(
        &self,
        cache: &'a mut Option<ops::kv_cache::KVCache>,
        layers: &[crate::FullPipelineLayer<'_>],
        max_seq: usize,
    ) -> &'a mut ops::kv_cache::KVCache {
        let shapes = Self::kv_shapes_for_layers(layers);
        self.ensure_kv_cache_for_shapes(cache, &shapes, max_seq)
    }

    pub(crate) fn ensure_kv_cache_for_shapes<'a>(
        &self,
        cache: &'a mut Option<ops::kv_cache::KVCache>,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> &'a mut ops::kv_cache::KVCache {
        let needs_rebuild = cache
            .as_ref()
            .is_none_or(|kv| kv.has_shape_mismatch(shapes));

        if needs_rebuild {
            *cache = Some(self.create_kv_cache_per_layer(shapes, max_seq));
        }

        let kv = cache.as_mut().expect("KV cache initialized above");
        kv.grow_to_shapes(shapes, max_seq);
        kv
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn decode_token_with_moe_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        moe_fn: Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
    ) -> Vec<f32> {
        self.decode_token_with_moe_split_fn(
            kv_cache,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            _num_q_heads,
            _num_kv_heads,
            _head_dim,
            _rope_base,
            moe_fn,
            None,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn decode_token_with_moe_split_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        mut moe_fn: Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        mut moe_collect_fn: Option<&mut dyn FnMut(usize) -> Vec<f32>>,
    ) -> Vec<f32> {
        let _gpu_time_token_start = std::time::Instant::now();
        let mut gpu_time = gpu_timing::TokenGpuTime::default();

        let num_layers = layers.len();
        
        let mut h_init = XpuBuffer::from_slice(x, false);
        
        // Use ScratchGuard for intermediate buffers (auto-recycled)
        let mut guard = buffers::ScratchGuard::new(&self.bufs);
        
        let mut h_a = self.bufs.output(hidden * 4);
        let mut h_b = self.bufs.output(hidden * 4);
        
        let mut norm_f32_buf = self.bufs.output(hidden * 4);
        let mut q_out = self.bufs.output(q_dim * 4);
        let mut k_out = self.bufs.output(kv_dim * 4);
        let mut v_out = self.bufs.output(kv_dim * 4);
        let mut attn_out_buf = self.bufs.output(q_dim * 4);
        let mut o_out_buf = self.bufs.output(hidden * 4);
        let mut h_post_attn = self.bufs.output(hidden * 4);
        let mut ffn_norm_out = self.bufs.output(hidden * 4);
        let mut down_out = self.bufs.output(hidden * 4);
        
        let mut ffn_q8 = self.bufs.output(hidden);
        let mut ffn_q8s = self.bufs.output(hidden * 4);

        let mut h_buf_ref = &mut h_init;
        let split_mode = moe_fn.is_some() && moe_collect_fn.is_some();

        for l in 0..num_layers {
            let layer = &layers[l];
            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let uses_q4k = layer.wq.format.is_q4k_family();
            let layer_q_dim = layer.num_q_heads * layer.head_dim;
            let layer_kv_dim = layer.num_kv_heads * layer.head_dim;
            let ffn_uses_q4k = layer.gate.format.is_q4k_family();

            let input_norm_w = self.bufs.get_f32(layer.input_norm);
            let wq_w = self.bufs.get_bytes(&layer.wq.data);
            let wk_w = self.bufs.get_bytes(&layer.wk.data);
            let wv_w = self.bufs.get_bytes(&layer.wv.data);
            
            self.encode_input_norm_and_qkv(
                layer,
                encode_qkv::QkvBufs {
                    h_in: h_buf_ref,
                    input_norm: &input_norm_w,
                    input_norm_bias: None,
                    wq: &wq_w,
                    wk: &wk_w,
                    wv: &wv_w,
                    wq_scales: &wq_w,
                    wk_scales: &wk_w,
                    wv_scales: &wv_w,
                    norm_out: &mut norm_f32_buf,
                    q_out: &mut q_out,
                    k_out: &mut k_out,
                    v_out: &mut v_out,
                    ffn_q8: &mut ffn_q8,
                    ffn_q8s: &mut ffn_q8s,
                },
                encode_qkv::QkvDims {
                    hidden,
                    layer_q_dim,
                    layer_kv_dim,
                    eps,
                    norm_offset,
                },
                uses_q4k,
            );

            let wo_w = self.bufs.get_bytes(&layer.wo.data);

            self.encode_attention_block(
                layer,
                kv_cache,
                l,
                encode_attn::AttnBufs {
                    h_buf: h_buf_ref,
                    q_out: &mut q_out,
                    k_out: &mut k_out,
                    v_out: &mut v_out,
                    attn_out_buf: &mut attn_out_buf,
                    o_out_buf: &mut o_out_buf,
                    ffn_norm_out: &mut ffn_norm_out,
                    h_post_attn: &mut h_post_attn,
                    ffn_q8: &mut ffn_q8,
                    ffn_q8s: &mut ffn_q8s,
                    wo: &wo_w,
                },
                encode_attn::AttnDims {
                    hidden,
                    layer_q_dim,
                    uses_q4k,
                    ffn_uses_q4k,
                },
            );

            let defer_ffn_for_split = split_mode && layer.moe.is_some();

            if !defer_ffn_for_split && !layer.ffn_is_remote {
                let gate_w = self.bufs.get_bytes(&layer.gate.data);
                let up_w = self.bufs.get_bytes(&layer.up.data);
                let down_w = self.bufs.get_bytes(&layer.down.data);

                self.encode_ffn_step(
                    layer,
                    encode_ffn::FfnBufs {
                        gate_w: &gate_w,
                        up_w: &up_w,
                        down_w: &down_w,
                        ffn_norm_out: &mut ffn_norm_out,
                        down_out: &mut down_out,
                    },
                    encode_ffn::FfnDims {
                        hidden,
                        inter,
                    },
                );

                self.encode_post_ffn_residual(
                    layer,
                    encode_post_ffn::PostFfnBufs {
                        down_out: &mut down_out,
                        h_post_attn: &mut h_post_attn,
                        new_h: if l % 2 == 0 { &mut h_a } else { &mut h_b },
                    },
                    hidden,
                );
            }

            if layer.moe.is_some() || layer.ffn_is_remote {
                let gate_w = self.bufs.get_bytes(&layer.gate.data);
                let up_w = self.bufs.get_bytes(&layer.up.data);
                let down_w = self.bufs.get_bytes(&layer.down.data);

                self.handle_moe_interleave(
                    layer,
                    moe_interleave::MoeInterleaveCtx {
                        layer_idx: l,
                        num_layers,
                        hidden,
                        inter,
                        ffn_uses_q4k,
                        defer_ffn_for_split,
                        stage_timing_split: false,
                        layer_in_snapshot: None,
                        dump_l0_dir: None,
                    },
                    moe_interleave::MoeInterleaveBufs {
                        gate_w: &gate_w,
                        up_w: &up_w,
                        down_w: &down_w,
                        h_post_attn: &h_post_attn,
                        ffn_norm_out: &ffn_norm_out,
                        down_out: &mut down_out,
                        new_h: if l % 2 == 0 { &mut h_a } else { &mut h_b },
                    },
                    moe_interleave::MoeCommandState {
                        gpu_time: &mut gpu_time,
                    },
                    &mut moe_fn,
                    &mut moe_collect_fn,
                );
            }
            
            h_buf_ref = if l % 2 == 0 { &mut h_a } else { &mut h_b };
        }

        let mut result = vec![0.0f32; hidden];
        h_buf_ref.copy_to_slice(&mut result);

        // Recycle buffers back to pool
        self.bufs.recycle(h_a);
        self.bufs.recycle(h_b);
        self.bufs.recycle(norm_f32_buf);
        self.bufs.recycle(q_out);
        self.bufs.recycle(k_out);
        self.bufs.recycle(v_out);
        self.bufs.recycle(attn_out_buf);
        self.bufs.recycle(o_out_buf);
        self.bufs.recycle(h_post_attn);
        self.bufs.recycle(ffn_norm_out);
        self.bufs.recycle(down_out);
        self.bufs.recycle(ffn_q8);
        self.bufs.recycle(ffn_q8s);

        let wall_ms = _gpu_time_token_start.elapsed().as_secs_f64() * 1000.0;
        gpu_time.print_if_enabled(wall_ms);

        result
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_token(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Vec<f32> {
        self.decode_token_with_moe_fn(
            kv_cache,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            None,
        )
    }
}
