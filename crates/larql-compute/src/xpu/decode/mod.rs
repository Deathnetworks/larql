use super::*;

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
        
        // Setup initial buffer
        let h_init = ops::buffers::XpuBuffer::from_slice(x, false);
        
        let mut h_a = ops::buffers::XpuBuffer::new_device(hidden * 4);
        let mut h_b = ops::buffers::XpuBuffer::new_device(hidden * 4);
        
        // Scratch buffers (reused across layers)
        let norm_f32_buf = ops::buffers::XpuBuffer::new_device(hidden * 4);
        let q_out = ops::buffers::XpuBuffer::new_device(q_dim * 4);
        let k_out = ops::buffers::XpuBuffer::new_device(kv_dim * 4);
        let v_out = ops::buffers::XpuBuffer::new_device(kv_dim * 4);
        let attn_out_buf = ops::buffers::XpuBuffer::new_device(q_dim * 4);
        let o_out_buf = ops::buffers::XpuBuffer::new_device(hidden * 4);
        let h_post_attn = ops::buffers::XpuBuffer::new_device(hidden * 4);
        let ffn_norm_out = ops::buffers::XpuBuffer::new_device(hidden * 4);
        
        let mut inter_padded = inter;
        if inter_padded % 256 != 0 {
            inter_padded = (inter_padded / 256 + 1) * 256;
        }
        
        let gate_out_scratch = ops::buffers::XpuBuffer::new_device(inter_padded * 4);
        let up_out = ops::buffers::XpuBuffer::new_device(inter_padded * 4);
        let act_buf = ops::buffers::XpuBuffer::new_device(inter_padded * 4);
        let down_out = ops::buffers::XpuBuffer::new_device(hidden * 4);
        
        let ffn_q8 = ops::buffers::XpuBuffer::new_device(hidden);
        let ffn_q8s = ops::buffers::XpuBuffer::new_device(hidden * 4);
        let o_q8_scratch = ops::buffers::XpuBuffer::new_device(q_dim);
        let o_q8s_scratch = ops::buffers::XpuBuffer::new_device(q_dim * 4);
        let normed_scratch = ops::buffers::XpuBuffer::new_device(hidden * 4);

        let mut h_buf_ref = &h_init;
        let split_mode = moe_fn.is_some() && moe_collect_fn.is_some();

        for l in 0..num_layers {
            let layer = &layers[l];
            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let layer_head_dim = layer.head_dim;
            let layer_num_q_heads = layer.num_q_heads;
            let layer_num_kv_heads = layer.num_kv_heads;
            let uses_q4k = layer.wq.format.is_q4k_family();
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let ffn_uses_q4k = layer.gate.format.is_q4k_family();

            let input_norm_w = self.bufs.get_f32(layer.input_norm_weight.unwrap());
            let wq_w = self.bufs.get_bytes(&layer.wq.data);
            let wk_w = self.bufs.get_bytes(&layer.wk.data);
            let wv_w = self.bufs.get_bytes(&layer.wv.data);
            
            // Step 1: Input norm + QKV
            self.encode_input_norm_and_qkv(
                layer,
                encode_qkv::QkvBufs {
                    h_in: h_buf_ref,
                    input_norm: &input_norm_w,
                    input_norm_bias: None,
                    wq: &wq_w,
                    wk: &wk_w,
                    wv: &wv_w,
                    wq_scales: &wq_w, // stub
                    wk_scales: &wk_w, // stub
                    wv_scales: &wv_w, // stub
                    norm_out: &norm_f32_buf,
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                    ffn_q8: &ffn_q8,
                    ffn_q8s: &ffn_q8s,
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
            let post_attn_w = self.bufs.get_f32(layer.post_attn_norm_weight.unwrap());

            // Steps 1.5 - 5: Attention Block
            self.encode_attention_block(
                layer,
                kv_cache,
                l,
                encode_attn::AttnBufs {
                    h_buf: h_buf_ref,
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                    attn_out_buf: &attn_out_buf,
                    o_out_buf: &o_out_buf,
                    ffn_norm_out: &ffn_norm_out,
                    h_post_attn: &h_post_attn,
                    o_q8_scratch: &o_q8_scratch,
                    o_q8s_scratch: &o_q8s_scratch,
                    ffn_q8: &ffn_q8,
                    ffn_q8s: &ffn_q8s,
                    normed_scratch: &normed_scratch,
                    wo: &wo_w,
                    wo_scales: &wo_w, // stub
                    post_attn_norm: &post_attn_w,
                },
                encode_attn::AttnDims {
                    hidden,
                    layer_q_dim,
                    uses_q4k,
                    ffn_uses_q4k,
                },
            );

            let new_h_ref = if l % 2 == 0 { &mut h_a } else { &mut h_b };
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
                        ffn_norm_out: &ffn_norm_out,
                        ffn_q8: &ffn_q8,
                        ffn_q8s: &ffn_q8s,
                        gate_out_scratch: &gate_out_scratch,
                        up_out: &up_out,
                        act_buf: &act_buf,
                        down_out: &down_out,
                    },
                    encode_ffn::FfnDims {
                        hidden,
                        inter,
                        inter_padded,
                    },
                    ffn_uses_q4k,
                );

                self.encode_post_ffn_residual(
                    layer,
                    encode_post_ffn::PostFfnBufs {
                        down_out: &down_out,
                        h_post_attn: &h_post_attn,
                        new_h: new_h_ref,
                        normed_scratch: &normed_scratch,
                    },
                    hidden,
                    true,
                );
            }
            
            h_buf_ref = new_h_ref;
        }

        let mut result = vec![0.0f32; hidden];
        h_buf_ref.copy_to_slice(&mut result);

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
