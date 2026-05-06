#include "kernels.hpp"
#include <iostream>

#if defined(_MSC_VER)
#pragma runtime_checks("", off)
#pragma strict_gs_check(off)
#endif

static std::unique_ptr<sycl::queue> g_queue = nullptr;

bool init_xpu() {
    if (g_queue) return true;
    try {
        g_queue = std::make_unique<sycl::queue>(sycl::default_selector_v);
        return true;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Error: " << e.what() << std::endl;
        return false;
    }
}

rust::String get_device_info() {
    if (!init_xpu()) return rust::String("No XPU Device");
    auto device = g_queue->get_device();
    std::string name = device.get_info<sycl::info::device::name>();
    return rust::String(name);
}

uint8_t* allocate_device(size_t size) {
    if (!init_xpu()) return nullptr;
    return static_cast<uint8_t*>(sycl::malloc_device(size, *g_queue));
}

uint8_t* allocate_shared(size_t size) {
    if (!init_xpu()) return nullptr;
    return static_cast<uint8_t*>(sycl::malloc_shared(size, *g_queue));
}

void free_memory(uint8_t* ptr) {
    if (!g_queue || !ptr) return;
    sycl::free(ptr, *g_queue);
}

void copy_h2d(uint8_t* dst, const uint8_t* src, size_t size) {
    if (!init_xpu()) return;
    g_queue->memcpy(dst, src, size).wait();
}

void copy_d2h(uint8_t* dst, const uint8_t* src, size_t size) {
    if (!init_xpu()) return;
    g_queue->memcpy(dst, src, size).wait();
}

static inline float decode_f16(uint16_t bits) {
    sycl::half h = sycl::bit_cast<sycl::half>(bits);
    return (float)h;
}

void f32_gemv(
    const float* x,
    const float* a,
    float* y,
    size_t m,
    size_t k
) {
    if (!init_xpu()) return;
    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(m), [=](sycl::id<1> idx) {
            size_t row = idx[0];
            float sum = 0.0f;
            for (size_t col = 0; col < k; ++col) {
                sum += x[col] * a[row * k + col];
            }
            y[row] = sum;
        });
    }).wait();
}

void q4_vecmat(
    const float* activation,
    const uint8_t* q4,
    float* out,
    size_t n,
    size_t k
) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(k), [=](sycl::id<1> idx) {
        size_t tid = idx[0];
        uint32_t blocks_per_row = k / 32;
        uint32_t bytes_per_row = blocks_per_row * 18;
        uint32_t block_idx = tid / 32;
        uint32_t elem_in_block = tid % 32;
        uint32_t nibble_idx = elem_in_block / 2;
        bool is_high = (elem_in_block & 1) != 0;

        float acc = 0.0f;
        for (uint32_t row = 0; row < n; row++) {
            float act = activation[row];
            if (sycl::fabs(act) < 1e-10f) continue;

            const uint8_t* block = q4 + row * bytes_per_row + block_idx * 18;
            uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float q4_scale = decode_f16(scale_bits);
            uint8_t byte = block[2 + nibble_idx];
            int q_val = is_high ? (int(byte >> 4) - 8) : (int(byte & 0x0F) - 8);
            acc += (float)q_val * q4_scale * act;
        }
        out[tid] = acc;
    }).wait();
}

void q4k_matvec_8sg(
    const uint8_t* w4k,
    const float* x,
    float* out,
    size_t n,
    size_t k
) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 8;
    const uint32_t BLOCK_SIZE = 144;
    const uint32_t THREADS_PER_TG = 256;

    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((n / ROWS_PER_TG) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t tg_id = item.get_group(0);
            uint32_t tid_in_tg = item.get_local_id(0);
            uint32_t sg_id = tid_in_tg / 32;
            uint32_t lane = tid_in_tg % 32;

            uint32_t row_idx = tg_id * ROWS_PER_TG + sg_id;
            if (row_idx >= n) return;

            const uint32_t superblocks = k / 256;
            const uint32_t bytes_per_row = superblocks * BLOCK_SIZE;
            const uint8_t* row_w = w4k + row_idx * bytes_per_row;

            const uint32_t ix = lane & 1;
            const uint32_t tid = lane >> 1;
            const uint32_t j = tid >> 1;
            const uint32_t sh = tid & 1;
            const bool hi = (j & 1) != 0;
            const uint32_t group = j >> 1;

            float acc = 0.0f;

            for (uint32_t sb = ix; sb < superblocks; sb += 2) {
                const uint8_t* block = row_w + sb * BLOCK_SIZE;
                uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                uint16_t dmin_bits = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
                float d = decode_f16(d_bits);
                float dmin = decode_f16(dmin_bits);

                const uint8_t* sb_bytes = block + 4;
                uint32_t sc, mn;
                if (j < 4) {
                    sc = (uint32_t)sb_bytes[j] & 0x3F;
                    mn = (uint32_t)sb_bytes[j + 4] & 0x3F;
                } else {
                    sc = ((uint32_t)sb_bytes[j + 4] & 0x0F) | (((uint32_t)sb_bytes[j - 4] >> 6) << 4);
                    mn = ((uint32_t)sb_bytes[j + 4] >> 4) | (((uint32_t)sb_bytes[j] >> 6) << 4);
                }
                float scale = d * (float)sc;
                float mmin = dmin * (float)mn;

                const uint32_t x_base = sb * 256 + j * 32 + sh * 16;
                float sumy = 0.0f;
                float dot_acc = 0.0f;

                const uint8_t* qs = block + 16 + group * 32 + sh * 16;

                for (uint32_t l = 0; l < 16; l++) {
                    float xi = x[x_base + l];
                    sumy += xi;
                    uint8_t byte = qs[l];
                    float nib = hi ? (float)((byte >> 4) & 0x0F) : (float)(byte & 0x0F);
                    dot_acc += nib * xi;
                }
                acc += scale * dot_acc - mmin * sumy;
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());

            if (lane == 0) out[row_idx] = acc;
        });
    }).wait();
}

void q6k_matvec(
    const uint8_t* w6k,
    const float* x,
    float* out,
    size_t n,
    size_t k
) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 4;
    const uint32_t BLOCK_SIZE = 210;
    const uint32_t THREADS_PER_TG = 128;

    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((n / ROWS_PER_TG) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t tg_id = item.get_group(0);
            uint32_t tid_in_tg = item.get_local_id(0);
            uint32_t sg_id = tid_in_tg / 32;
            uint32_t lane = tid_in_tg % 32;

            uint32_t row_idx = tg_id * ROWS_PER_TG + sg_id;
            if (row_idx >= n) return;

            const uint32_t superblocks = k / 256;
            const uint32_t bytes_per_row = superblocks * BLOCK_SIZE;
            const uint8_t* row = w6k + row_idx * bytes_per_row;

            const uint32_t ix = lane & 1;
            const uint32_t tid = lane >> 1;
            const uint32_t base = tid << 2;
            const uint32_t sc_base = tid >> 2;

            float acc = 0.0f;

            for (uint32_t i = ix; i < superblocks; i += 2) {
                const uint8_t* block = row + i * BLOCK_SIZE;
                const uint8_t* ql = block;
                const uint8_t* qh = block + 128;
                const int8_t* sc = (const int8_t*)(block + 192);
                uint16_t d_bits = (uint16_t)block[208] | ((uint16_t)block[209] << 8);
                float d = decode_f16(d_bits);

                const uint32_t xb = i * 256 + base;
                float xl[16];
                xl[0] = x[xb];     xl[1] = x[xb + 1];
                xl[2] = x[xb + 2]; xl[3] = x[xb + 3];
                xl[4] = x[xb + 64]; xl[5] = x[xb + 65];
                xl[6] = x[xb + 66]; xl[7] = x[xb + 67];
                xl[8] = x[xb + 128]; xl[9] = x[xb + 129];
                xl[10] = x[xb + 130]; xl[11] = x[xb + 131];
                xl[12] = x[xb + 192]; xl[13] = x[xb + 193];
                xl[14] = x[xb + 194]; xl[15] = x[xb + 195];

                // Pass 0
                {
                    const uint32_t b = base;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 0];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * xl[0] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * xl[1] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * xl[2] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * xl[3]);
                }
                // Pass 1
                {
                    const uint32_t b = base + 64;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 4];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * xl[4] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * xl[5] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * xl[6] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * xl[7]);
                }
                // Pass 2
                {
                    const uint32_t b = base + 128;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 8];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * xl[8] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * xl[9] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * xl[10] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * xl[11]);
                }
                // Pass 3
                {
                    const uint32_t b = base + 192;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 12];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * xl[12] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * xl[13] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * xl[14] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * xl[15]);
                }
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());

            if (lane == 0) out[row_idx] = acc;
        });
    }).wait();
}

void q4k_ffn_gate_up(
    const uint8_t* wg,
    const uint8_t* wu,
    const float* x,
    float* g_out,
    float* u_out,
    size_t n,
    size_t k
) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 4;
    const uint32_t BLOCK_SIZE = 144;
    const uint32_t THREADS_PER_TG = 128;

    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(2 * ((n + ROWS_PER_TG - 1) / ROWS_PER_TG) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t tg_id = item.get_group(0);
            uint32_t tid_in_tg = item.get_local_id(0);
            uint32_t sg_id = tid_in_tg / 32;
            uint32_t lane = tid_in_tg % 32;

            uint32_t tgs_per_mat = (n + ROWS_PER_TG - 1) / ROWS_PER_TG;
            bool is_up = (tg_id >= tgs_per_mat);
            uint32_t mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

            uint32_t row_idx = mat_tg * ROWS_PER_TG + sg_id;
            if (row_idx >= n) return;

            const uint8_t* w = is_up ? wu : wg;
            float* out_buf = is_up ? u_out : g_out;

            const uint32_t superblocks = k / 256;
            const uint32_t bytes_per_row = superblocks * BLOCK_SIZE;
            const uint8_t* row_w = w + row_idx * bytes_per_row;

            const uint32_t ix = lane & 1;
            const uint32_t tid = lane >> 1;
            const uint32_t j = tid >> 1;
            const uint32_t sh = tid & 1;
            const bool hi = (j & 1) != 0;
            const uint32_t group = j >> 1;

            float acc = 0.0f;

            for (uint32_t sb = ix; sb < superblocks; sb += 2) {
                const uint8_t* block = row_w + sb * BLOCK_SIZE;
                uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                uint16_t dmin_bits = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
                float d = decode_f16(d_bits);
                float dmin = decode_f16(dmin_bits);

                const uint8_t* sb_bytes = block + 4;
                uint32_t sc, mn;
                if (j < 4) {
                    sc = (uint32_t)sb_bytes[j] & 0x3F;
                    mn = (uint32_t)sb_bytes[j + 4] & 0x3F;
                } else {
                    sc = ((uint32_t)sb_bytes[j + 4] & 0x0F) | (((uint32_t)sb_bytes[j - 4] >> 6) << 4);
                    mn = ((uint32_t)sb_bytes[j + 4] >> 4) | (((uint32_t)sb_bytes[j] >> 6) << 4);
                }
                float scale = d * (float)sc;
                float mmin = dmin * (float)mn;

                const uint32_t x_base = sb * 256 + j * 32 + sh * 16;
                float xl[16];
                for (uint32_t l = 0; l < 16; l++) { xl[l] = x[x_base + l]; }

                const uint8_t* qs = block + 16 + group * 32 + sh * 16;

                float sumy = 0.0f;
                for (uint32_t l = 0; l < 16; l++) { sumy += xl[l]; }

                float dot_acc = 0.0f;
                for (uint32_t l = 0; l < 16; l++) {
                    uint8_t byte = qs[l];
                    float nib = hi ? (float)((byte >> 4) & 0x0F) : (float)(byte & 0x0F);
                    dot_acc += nib * xl[l];
                }
                acc += scale * dot_acc - mmin * sumy;
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
            if (lane == 0) out_buf[row_idx] = acc;
        });
    }).wait();
}

void rms_norm(
    const float* x,
    const float* weight,
    float* out,
    size_t len,
    float eps,
    float offset
) {
    if (!init_xpu()) return;
    size_t wg_size = 256; 
    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(wg_size), sycl::range<1>(wg_size)), [=](sycl::nd_item<1> item) {
            auto g = item.get_group();
            uint32_t tid = item.get_local_id(0);
            uint32_t tg_sz = item.get_local_range(0);

            float partial = 0.0f;
            for (uint32_t i = tid; i < len; i += tg_sz) {
                partial += x[i] * x[i];
            }

            float sum_sq = sycl::reduce_over_group(g, partial, sycl::plus<>());
            float rms = 1.0f / sycl::sqrt(sum_sq / float(len) + eps);

            for (uint32_t i = tid; i < len; i += tg_sz) {
                out[i] = x[i] * (weight[i] + offset) * rms;
            }
        });
    }).wait();
}

void silu(
    const float* input,
    float* out,
    size_t n
) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        float x = input[i];
        out[i] = x / (1.0f + sycl::exp(-x));
    }).wait();
}

void attn_fused(
    const float* q_in,
    const float* k_in,
    const float* v_in,
    float* k_cache,
    float* v_cache,
    float* out,
    const float* q_weight,
    const float* k_weight,
    uint32_t t,
    uint32_t head_dim,
    uint32_t num_q,
    uint32_t num_kv,
    float scale,
    uint32_t window_size,
    float eps,
    float qk_offset,
    float rope_base,
    uint32_t rotary_dim
) {
    if (!init_xpu()) return;
    const uint32_t THREADS_PER_TG = 256; 

    g_queue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> tg_q(sycl::range<1>(256), h);
        sycl::local_accessor<float, 1> tg_k_normed(sycl::range<1>(256), h);
        sycl::local_accessor<float, 1> tg_red(sycl::range<1>(8), h);
        sycl::local_accessor<float, 1> tg_scores(sycl::range<1>(1024), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_q * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t head = item.get_group(0);
            uint32_t tid = item.get_local_id(0);
            uint32_t tg_sz = item.get_local_range(0);
            uint32_t lane = tid % 32;
            uint32_t sg_id = tid / 32;
            uint32_t n_sg = (tg_sz + 31) / 32;

            uint32_t kv_head = head / (num_q / num_kv);
            uint32_t pos = t - 1;

            uint32_t rdim = (rotary_dim == 0) ? head_dim : std::min(rotary_dim, head_dim);
            uint32_t hdim = rdim / 2;

            // Phase 1: Parallel RMS for Q and K
            float partial_q = 0.0f;
            float partial_k = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += tg_sz) {
                float vq = q_in[head * head_dim + d];
                float vk = k_in[kv_head * head_dim + d];
                partial_q += vq * vq;
                partial_k += vk * vk;
            }

            auto sub_g = item.get_sub_group();
            float sg_q = sycl::reduce_over_group(sub_g, partial_q, sycl::plus<>());
            float sg_k = sycl::reduce_over_group(sub_g, partial_k, sycl::plus<>());
            
            if (lane == 0) tg_red[sg_id] = sg_q;
            item.barrier(sycl::access::fence_space::local_space);
            float ss_q = 0.0f;
            if (tid == 0) {
                for (uint32_t i = 0; i < n_sg; i++) ss_q += tg_red[i];
                tg_red[0] = ss_q; 
            }
            item.barrier(sycl::access::fence_space::local_space);
            ss_q = tg_red[0];

            if (lane == 0) tg_red[sg_id] = sg_k;
            item.barrier(sycl::access::fence_space::local_space);
            float ss_k = 0.0f;
            if (tid == 0) {
                for (uint32_t i = 0; i < n_sg; i++) ss_k += tg_red[i];
                tg_red[0] = ss_k;
            }
            item.barrier(sycl::access::fence_space::local_space);
            ss_k = tg_red[0];

            float inv_rms_q = 1.0f / sycl::sqrt(ss_q / float(head_dim) + eps);
            float inv_rms_k = 1.0f / sycl::sqrt(ss_k / float(head_dim) + eps);

            // Phase 2: Write normed Q,K to TG memory
            for (uint32_t d = tid; d < head_dim; d += tg_sz) {
                float vq = q_in[head * head_dim + d];
                float vk = k_in[kv_head * head_dim + d];
                tg_q[d] = (vq * inv_rms_q) * (qk_offset + q_weight[d]);
                tg_k_normed[d] = (vk * inv_rms_k) * (qk_offset + k_weight[d]);
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Phase 3: Shared RoPE
            uint32_t cache_off = pos * num_kv * head_dim + kv_head * head_dim;
            for (uint32_t d = tid; d < hdim; d += tg_sz) {
                float freq = 1.0f / sycl::pow(rope_base, float(2 * d) / float(rdim));
                float angle = float(pos) * freq;
                float cos_a = sycl::cos(angle);
                float sin_a = sycl::sin(angle);

                float qr = tg_q[d];
                float qi = tg_q[d + hdim];
                tg_q[d] = qr * cos_a - qi * sin_a;
                tg_q[d + hdim] = qr * sin_a + qi * cos_a;

                float kr = tg_k_normed[d];
                float ki = tg_k_normed[d + hdim];
                k_cache[cache_off + d] = kr * cos_a - ki * sin_a;
                k_cache[cache_off + d + hdim] = kr * sin_a + ki * cos_a;
            }
            for (uint32_t d = tid + rdim; d < head_dim; d += tg_sz) {
                k_cache[cache_off + d] = tg_k_normed[d];
            }

            // Phase 4: Stream V
            for (uint32_t d = tid; d < head_dim; d += tg_sz) {
                v_cache[cache_off + d] = v_in[kv_head * head_dim + d];
            }
            item.barrier(sycl::access::fence_space::global_space);

            // Phase 5: Scores
            uint32_t t_start = (window_size > 0 && t > window_size) ? t - window_size : 0;
            float local_max = -1e30f;
            for (uint32_t it = t_start + tid; it < t; it += tg_sz) {
                const float* k_ptr = k_cache + it * num_kv * head_dim + kv_head * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += tg_q[d] * k_ptr[d];
                }
                dot *= scale;
                tg_scores[it - t_start] = dot;
                local_max = std::max(local_max, dot);
            }

            float sg_max = sycl::reduce_over_group(sub_g, local_max, sycl::maximum<>());
            if (lane == 0) tg_red[sg_id] = sg_max;
            item.barrier(sycl::access::fence_space::local_space);
            float global_max = -1e30f;
            if (tid == 0) {
                for (uint32_t i = 0; i < n_sg; i++) global_max = std::max(global_max, tg_red[i]);
                tg_red[0] = global_max;
            }
            item.barrier(sycl::access::fence_space::local_space);
            global_max = tg_red[0];

            // Phase 6: Softmax sum
            float local_sum = 0.0f;
            for (uint32_t it = t_start + tid; it < t; it += tg_sz) {
                float w = sycl::exp(tg_scores[it - t_start] - global_max);
                tg_scores[it - t_start] = w;
                local_sum += w;
            }

            float sg_sum = sycl::reduce_over_group(sub_g, local_sum, sycl::plus<>());
            if (lane == 0) tg_red[sg_id] = sg_sum;
            item.barrier(sycl::access::fence_space::local_space);
            float global_sum = 0.0f;
            if (tid == 0) {
                for (uint32_t i = 0; i < n_sg; i++) global_sum += tg_red[i];
                tg_red[0] = global_sum;
            }
            item.barrier(sycl::access::fence_space::local_space);
            global_sum = tg_red[0];
            float inv_sum = 1.0f / global_sum;

            for (uint32_t it = t_start + tid; it < t; it += tg_sz) {
                tg_scores[it - t_start] *= inv_sum;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Phase 7: V sum
            float* out_head = out + head * head_dim;
            for (uint32_t d = tid; d < head_dim; d += tg_sz) {
                float acc = 0.0f;
                for (uint32_t it = t_start; it < t; it++) {
                    acc += tg_scores[it - t_start] * v_cache[it * num_kv * head_dim + kv_head * head_dim + d];
                }
                out_head[d] = acc;
            }
        });
    }).wait();
}

void q4k_qkv_proj(
    const uint8_t* wq,
    const uint8_t* wk,
    const uint8_t* wv,
    const float* x,
    float* q_out,
    float* k_out,
    float* v_out,
    uint32_t q_rows,
    uint32_t k_rows,
    uint32_t v_rows,
    uint32_t k
) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 8;
    const uint32_t BLOCK_SIZE = 144;
    const uint32_t THREADS_PER_TG = 256;

    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((q_rows + k_rows + v_rows + ROWS_PER_TG - 1) / ROWS_PER_TG) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t tg_id = item.get_group(0);
            uint32_t tid_in_tg = item.get_local_id(0);
            uint32_t sg_id = tid_in_tg / 32;
            uint32_t lane = tid_in_tg % 32;

            uint32_t total_rows = q_rows + k_rows + v_rows;
            uint32_t global_row = tg_id * ROWS_PER_TG + sg_id;
            if (global_row >= total_rows) return;

            const uint8_t* w_ptr;
            float* out_ptr;
            uint32_t local_row;
            if (global_row < q_rows) {
                w_ptr = wq; out_ptr = q_out; local_row = global_row;
            } else if (global_row < q_rows + k_rows) {
                w_ptr = wk; out_ptr = k_out; local_row = global_row - q_rows;
            } else {
                w_ptr = wv; out_ptr = v_out; local_row = global_row - q_rows - k_rows;
            }

            uint32_t superblocks = k / 256;
            uint32_t bytes_per_row = superblocks * BLOCK_SIZE;
            const uint8_t* row = w_ptr + local_row * bytes_per_row;

            float acc = 0.0f;
            for (uint32_t sb = lane; sb < superblocks; sb += 32) {
                const uint8_t* block = row + sb * BLOCK_SIZE;

                uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                uint16_t dmin_bits = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
                float d = decode_f16(d_bits);
                float dmin = decode_f16(dmin_bits);

                const uint8_t* sb_bytes = block + 4;
                uint32_t scales[8], mins[8];
                for (uint32_t j = 0; j < 4; j++) {
                    scales[j] = (uint32_t)sb_bytes[j] & 0x3F;
                    mins[j] = (uint32_t)sb_bytes[j + 4] & 0x3F;
                }
                for (uint32_t j = 4; j < 8; j++) {
                    scales[j] = ((uint32_t)sb_bytes[j + 4] & 0x0F) | (((uint32_t)sb_bytes[j - 4] >> 6) << 4);
                    mins[j] = ((uint32_t)sb_bytes[j + 4] >> 4) | (((uint32_t)sb_bytes[j] >> 6) << 4);
                }

                const uint8_t* qs = block + 16;
                uint32_t x_base = sb * 256;
                float sb_acc = 0.0f;
                for (uint32_t g = 0; g < 4; g++) {
                    uint32_t sub_lo = 2 * g;
                    uint32_t sub_hi = 2 * g + 1;
                    float sc_lo = d * (float)scales[sub_lo];
                    float sc_hi = d * (float)scales[sub_hi];
                    float mn_lo = dmin * (float)mins[sub_lo];
                    float mn_hi = dmin * (float)mins[sub_hi];
                    float dot_lo = 0.0f, sum_lo = 0.0f;
                    float dot_hi = 0.0f, sum_hi = 0.0f;
                    for (uint32_t l = 0; l < 32; l++) {
                        uint8_t byte = qs[g * 32 + l];
                        float nib_lo = (float)(byte & 0x0F);
                        float nib_hi = (float)((byte >> 4) & 0x0F);
                        float xlo = x[x_base + sub_lo * 32 + l];
                        float xhi = x[x_base + sub_hi * 32 + l];
                        dot_lo += nib_lo * xlo;
                        sum_lo += xlo;
                        dot_hi += nib_hi * xhi;
                        sum_hi += xhi;
                    }
                    sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
                    sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
                }
                acc += sb_acc;
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
            if (lane == 0) out_ptr[local_row] = acc;
        });
    }).wait();
}

void q4k_proj(
    const uint8_t* w4k,
    const float* x,
    float* out,
    size_t n,
    size_t k
) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 8;
    const uint32_t BLOCK_SIZE = 144;
    const uint32_t THREADS_PER_TG = 256;

    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((n + ROWS_PER_TG - 1) / ROWS_PER_TG) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t tg_id = item.get_group(0);
            uint32_t tid_in_tg = item.get_local_id(0);
            uint32_t sg_id = tid_in_tg / 32;
            uint32_t lane = tid_in_tg % 32;

            uint32_t row_idx = tg_id * ROWS_PER_TG + sg_id;
            if (row_idx >= n) return;

            uint32_t superblocks = k / 256;
            uint32_t bytes_per_row = superblocks * BLOCK_SIZE;
            const uint8_t* row = w4k + row_idx * bytes_per_row;

            float acc = 0.0f;
            for (uint32_t sb = lane; sb < superblocks; sb += 32) {
                const uint8_t* block = row + sb * BLOCK_SIZE;

                uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                uint16_t dmin_bits = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
                float d = decode_f16(d_bits);
                float dmin = decode_f16(dmin_bits);

                const uint8_t* sb_bytes = block + 4;
                uint32_t scales[8], mins[8];
                for (uint32_t j = 0; j < 4; j++) {
                    scales[j] = (uint32_t)sb_bytes[j] & 0x3F;
                    mins[j] = (uint32_t)sb_bytes[j + 4] & 0x3F;
                }
                for (uint32_t j = 4; j < 8; j++) {
                    scales[j] = ((uint32_t)sb_bytes[j + 4] & 0x0F) | (((uint32_t)sb_bytes[j - 4] >> 6) << 4);
                    mins[j] = ((uint32_t)sb_bytes[j + 4] >> 4) | (((uint32_t)sb_bytes[j] >> 6) << 4);
                }

                const uint8_t* qs = block + 16;
                uint32_t x_base = sb * 256;
                float sb_acc = 0.0f;
                for (uint32_t g = 0; g < 4; g++) {
                    uint32_t sub_lo = 2 * g;
                    uint32_t sub_hi = 2 * g + 1;
                    float sc_lo = d * (float)scales[sub_lo];
                    float sc_hi = d * (float)scales[sub_hi];
                    float mn_lo = dmin * (float)mins[sub_lo];
                    float mn_hi = dmin * (float)mins[sub_hi];
                    float dot_lo = 0.0f, sum_lo = 0.0f;
                    float dot_hi = 0.0f, sum_hi = 0.0f;
                    for (uint32_t l = 0; l < 32; l++) {
                        uint8_t byte = qs[g * 32 + l];
                        float nib_lo = (float)(byte & 0x0F);
                        float nib_hi = (float)((byte >> 4) & 0x0F);
                        float xlo = x[x_base + sub_lo * 32 + l];
                        float xhi = x[x_base + sub_hi * 32 + l];
                        dot_lo += nib_lo * xlo;
                        sum_lo += xlo;
                        dot_hi += nib_hi * xhi;
                        sum_hi += xhi;
                    }
                    sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
                    sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
                }
                acc += sb_acc;
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
            if (lane == 0) out[row_idx] = acc;
        });
    }).wait();
}

void q4k_q6k_qkv_proj(
    const uint8_t* wq,
    const uint8_t* wk,
    const uint8_t* wv,
    const float* x,
    float* q_out,
    float* k_out,
    float* v_out,
    uint32_t q_rows,
    uint32_t k_rows,
    uint32_t v_rows,
    uint32_t k
) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 4;
    const uint32_t Q4K_BLOCK_SIZE = 144;
    const uint32_t Q6K_BLOCK_SIZE = 210;
    const uint32_t THREADS_PER_TG = 128;

    g_queue->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((q_rows + k_rows + v_rows + ROWS_PER_TG - 1) / ROWS_PER_TG) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t tg_id = item.get_group(0);
            uint32_t tid_in_tg = item.get_local_id(0);
            uint32_t sg_id = tid_in_tg / 32;
            uint32_t lane = tid_in_tg % 32;

            uint32_t total_rows = q_rows + k_rows + v_rows;
            uint32_t global_row = tg_id * ROWS_PER_TG + sg_id;
            if (global_row >= total_rows) return;

            const uint32_t superblocks = k / 256;
            float acc = 0.0f;

            if (global_row < q_rows + k_rows) {
                const uint8_t* w_ptr;
                float* out_ptr;
                uint32_t local_row;
                if (global_row < q_rows) {
                    w_ptr = wq; out_ptr = q_out; local_row = global_row;
                } else {
                    w_ptr = wk; out_ptr = k_out; local_row = global_row - q_rows;
                }

                const uint32_t bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
                const uint8_t* row = w_ptr + local_row * bytes_per_row;

                const uint32_t ix = lane & 1;
                const uint32_t tid = lane >> 1;
                const uint32_t j = tid >> 1;
                const uint32_t sh = tid & 1;
                const bool hi = (j & 1) != 0;
                const uint32_t group = j >> 1;

                for (uint32_t sb = ix; sb < superblocks; sb += 2) {
                    const uint8_t* block = row + sb * Q4K_BLOCK_SIZE;
                    uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                    uint16_t dmin_bits = (uint16_t)block[2] | ((uint16_t)block[3] << 8);
                    float d = decode_f16(d_bits);
                    float dmin = decode_f16(dmin_bits);

                    const uint8_t* sb_bytes = block + 4;
                    uint32_t sc, mn;
                    if (j < 4) {
                        sc = (uint32_t)sb_bytes[j] & 0x3F;
                        mn = (uint32_t)sb_bytes[j + 4] & 0x3F;
                    } else {
                        sc = ((uint32_t)sb_bytes[j + 4] & 0x0F) | (((uint32_t)sb_bytes[j - 4] >> 6) << 4);
                        mn = ((uint32_t)sb_bytes[j + 4] >> 4) | (((uint32_t)sb_bytes[j] >> 6) << 4);
                    }
                    float scale = d * (float)sc;
                    float mmin = dmin * (float)mn;

                    const uint32_t x_base = sb * 256 + j * 32 + sh * 16;
                    float xl[16];
                    for (uint32_t l = 0; l < 16; l++) { xl[l] = x[x_base + l]; }

                    const uint8_t* qs = block + 16 + group * 32 + sh * 16;
                    float dot_acc = 0.0f, sum_acc = 0.0f;
                    for (uint32_t l = 0; l < 16; l++) {
                        uint8_t byte = qs[l];
                        float nib = hi ? (float)((byte >> 4) & 0x0F) : (float)(byte & 0x0F);
                        dot_acc += nib * xl[l];
                        sum_acc += xl[l];
                    }
                    acc += scale * dot_acc - mmin * sum_acc;
                }

                auto sub_g = item.get_sub_group();
                acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
                if (lane == 0) out_ptr[local_row] = acc;

            } else {
                uint32_t local_row = global_row - q_rows - k_rows;
                const uint32_t bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
                const uint8_t* row = wv + local_row * bytes_per_row;

                const uint32_t ix6 = lane & 1;
                const uint32_t tid6 = lane >> 1;
                const uint32_t base = tid6 << 2;
                const uint32_t sc_base = tid6 >> 2;

                for (uint32_t sb = ix6; sb < superblocks; sb += 2) {
                    const uint8_t* block = row + sb * Q6K_BLOCK_SIZE;
                    const uint8_t* ql = block;
                    const uint8_t* qh = block + 128;
                    const int8_t* sc = (const int8_t*)(block + 192);
                    uint16_t d_bits = (uint16_t)block[208] | ((uint16_t)block[209] << 8);
                    float d = decode_f16(d_bits);

                    const uint32_t xb = sb * 256 + base;
                    float xl[16];
                    xl[0] = x[xb]; xl[1] = x[xb + 1];
                    xl[2] = x[xb + 2]; xl[3] = x[xb + 3];
                    xl[4] = x[xb + 64]; xl[5] = x[xb + 65];
                    xl[6] = x[xb + 66]; xl[7] = x[xb + 67];
                    xl[8] = x[xb + 128]; xl[9] = x[xb + 129];
                    xl[10] = x[xb + 130]; xl[11] = x[xb + 131];
                    xl[12] = x[xb + 192]; xl[13] = x[xb + 193];
                    xl[14] = x[xb + 194]; xl[15] = x[xb + 195];

                    auto compute_dot = [&](uint32_t b, uint32_t sb_off) {
                        uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                        float _sc = d * (float)sc[sc_base + sb_off];
                        return _sc * (
                            (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * xl[sb_off + 0] +
                            (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * xl[sb_off + 1] +
                            (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * xl[sb_off + 2] +
                            (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * xl[sb_off + 3]
                        );
                    };

                    acc += compute_dot(base, 0);
                    acc += compute_dot(base + 64, 4);
                    acc += compute_dot(base + 128, 8);
                    acc += compute_dot(base + 192, 12);
                }

                auto sub_g = item.get_sub_group();
                acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
                if (lane == 0) v_out[local_row] = acc;
            }
        });
    }).wait();
}

void gelu_tanh(
    const float* input,
    float* out,
    size_t n
) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        float x = input[i];
        float c = 0.7978845608f; 
        float y = c * (x + 0.044715f * x * x * x);
        y = sycl::clamp(y, -15.0f, 15.0f);
        float t = sycl::tanh(y);
        out[i] = 0.5f * x * (1.0f + t);
    }).wait();
}

void q4_matvec_v4(
    const uint8_t* q4,
    const int8_t* q8,
    const float* q8s,
    float* out,
    size_t n,
    size_t k
) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 8;
    const uint32_t THREADS_PER_TG = 256;

    g_queue->submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> tg_q8(sycl::range<1>(8192), h);
        sycl::local_accessor<float, 1> tg_q8s(sycl::range<1>(256), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((n / ROWS_PER_TG) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), [=](sycl::nd_item<1> item) {
            uint32_t tg_id = item.get_group(0);
            uint32_t tid_in_tg = item.get_local_id(0);
            uint32_t sg_id = tid_in_tg / 32;
            uint32_t lane = tid_in_tg % 32;

            uint32_t blocks = k / 32;
            uint32_t bytes_per_row = blocks * 18;

            for (uint32_t i = tid_in_tg; i < k; i += THREADS_PER_TG) tg_q8[i] = q8[i];
            for (uint32_t i = tid_in_tg; i < blocks; i += THREADS_PER_TG) tg_q8s[i] = q8s[i];
            item.barrier(sycl::access::fence_space::local_space);

            uint32_t row_idx = tg_id * ROWS_PER_TG + sg_id;
            if (row_idx >= n) return;

            const uint8_t* row = q4 + row_idx * bytes_per_row;

            float acc = 0.0f;
            for (uint32_t b = lane; b < blocks; b += 32) {
                const uint8_t* block = row + b * 18;

                uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                float combined_scale = decode_f16(scale_bits) * tg_q8s[b];

                const uint8_t* qb = block + 2;
                uint32_t w0 = (uint32_t)qb[0] | ((uint32_t)qb[1] << 8) | ((uint32_t)qb[2] << 16) | ((uint32_t)qb[3] << 24);
                uint32_t w1 = (uint32_t)qb[4] | ((uint32_t)qb[5] << 8) | ((uint32_t)qb[6] << 16) | ((uint32_t)qb[7] << 24);
                uint32_t w2 = (uint32_t)qb[8] | ((uint32_t)qb[9] << 8) | ((uint32_t)qb[10] << 16) | ((uint32_t)qb[11] << 24);
                uint32_t w3 = (uint32_t)qb[12] | ((uint32_t)qb[13] << 8) | ((uint32_t)qb[14] << 16) | ((uint32_t)qb[15] << 24);

                int isum = 0;
                auto nibble = [](uint32_t w, uint32_t shift) { return (int)((w >> shift) & 0x0F) - 8; };

                isum += nibble(w0, 0) * (int)tg_q8[b * 32 + 0];
                isum += nibble(w0, 4) * (int)tg_q8[b * 32 + 1];
                isum += nibble(w0, 8) * (int)tg_q8[b * 32 + 2];
                isum += nibble(w0, 12) * (int)tg_q8[b * 32 + 3];
                isum += nibble(w0, 16) * (int)tg_q8[b * 32 + 4];
                isum += nibble(w0, 20) * (int)tg_q8[b * 32 + 5];
                isum += nibble(w0, 24) * (int)tg_q8[b * 32 + 6];
                isum += nibble(w0, 28) * (int)tg_q8[b * 32 + 7];

                isum += nibble(w1, 0) * (int)tg_q8[b * 32 + 8];
                isum += nibble(w1, 4) * (int)tg_q8[b * 32 + 9];
                isum += nibble(w1, 8) * (int)tg_q8[b * 32 + 10];
                isum += nibble(w1, 12) * (int)tg_q8[b * 32 + 11];
                isum += nibble(w1, 16) * (int)tg_q8[b * 32 + 12];
                isum += nibble(w1, 20) * (int)tg_q8[b * 32 + 13];
                isum += nibble(w1, 24) * (int)tg_q8[b * 32 + 14];
                isum += nibble(w1, 28) * (int)tg_q8[b * 32 + 15];

                isum += nibble(w2, 0) * (int)tg_q8[b * 32 + 16];
                isum += nibble(w2, 4) * (int)tg_q8[b * 32 + 17];
                isum += nibble(w2, 8) * (int)tg_q8[b * 32 + 18];
                isum += nibble(w2, 12) * (int)tg_q8[b * 32 + 19];
                isum += nibble(w2, 16) * (int)tg_q8[b * 32 + 20];
                isum += nibble(w2, 20) * (int)tg_q8[b * 32 + 21];
                isum += nibble(w2, 24) * (int)tg_q8[b * 32 + 22];
                isum += nibble(w2, 28) * (int)tg_q8[b * 32 + 23];

                isum += nibble(w3, 0) * (int)tg_q8[b * 32 + 24];
                isum += nibble(w3, 4) * (int)tg_q8[b * 32 + 25];
                isum += nibble(w3, 8) * (int)tg_q8[b * 32 + 26];
                isum += nibble(w3, 12) * (int)tg_q8[b * 32 + 27];
                isum += nibble(w3, 16) * (int)tg_q8[b * 32 + 28];
                isum += nibble(w3, 20) * (int)tg_q8[b * 32 + 29];
                isum += nibble(w3, 24) * (int)tg_q8[b * 32 + 30];
                isum += nibble(w3, 28) * (int)tg_q8[b * 32 + 31];

                acc += (float)isum * combined_scale;
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
            if (lane == 0) out[row_idx] = acc;
        });
    }).wait();
}

void rope_at_pos_batched_qk(
    float* q,
    float* k,
    size_t head_dim,
    float rope_base,
    size_t pos,
    size_t rotary_dim,
    size_t num_q,
    size_t num_kv
) {
    if (!init_xpu()) return;
    size_t total_heads = num_q + num_kv;
    size_t rdim = (rotary_dim == 0) ? head_dim : std::min(rotary_dim, head_dim);
    size_t hdim = rdim / 2;

    g_queue->parallel_for(sycl::range<2>(total_heads, hdim), [=](sycl::id<2> idx) {
        size_t h = idx[0];
        size_t d = idx[1];

        bool is_q = (h < num_q);
        size_t local_h = is_q ? h : (h - num_q);
        float* x = is_q ? q : k;
        size_t base_idx = local_h * head_dim;

        float freq = 1.0f / sycl::pow(rope_base, float(2 * d) / float(rdim));
        float angle = float(pos) * freq;
        float cos_a = sycl::cos(angle);
        float sin_a = sycl::sin(angle);

        float re = x[base_idx + d];
        float im = x[base_idx + d + hdim];
        x[base_idx + d] = re * cos_a - im * sin_a;
        x[base_idx + d + hdim] = re * sin_a + im * cos_a;
    }).wait();
}
