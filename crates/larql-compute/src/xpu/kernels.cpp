#ifndef SYCL_DLL_BUILD
#include "kernels.hpp"
#else
#include <sycl/sycl.hpp>
#include <memory>
#include <iostream>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#endif

#include <iostream>

#if defined(_MSC_VER) && !defined(SYCL_DLL_BUILD)
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

#ifndef SYCL_DLL_BUILD
rust::String get_device_info() {
    if (!init_xpu()) return rust::String("No XPU Device");
    auto device = g_queue->get_device();
    std::string name = device.get_info<sycl::info::device::name>();
    return rust::String(name);
}
#endif

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

// Functor Definitions

struct F32GemvFunctor {
    const float* x;
    const float* a;
    float* y;
    uint32_t k;

    void operator()(sycl::id<1> idx) const {
        size_t row = idx[0];
        float sum = 0.0f;
        for (size_t col = 0; col < k; ++col) {
            sum += x[col] * a[row * k + col];
        }
        y[row] = sum;
    }
};

struct Q4MatVecV4Functor {
    const uint8_t* q4;
    const float* x;
    float* out;
    uint32_t n;
    uint32_t k;

    void operator()(sycl::id<1> idx) const {
        size_t tid = idx[0];
        if (tid >= n) return;
        const uint32_t bytes_per_row = (k / 32) * 18;
        const uint8_t* row_q4 = q4 + tid * bytes_per_row;
        float acc = 0.0f;
        for (uint32_t j = 0; j < k; j++) {
            uint32_t block_idx = j / 32;
            uint32_t nibble_idx = (j % 32) / 2;
            bool is_high = (j % 2) != 0;
            const uint8_t* block = row_q4 + block_idx * 18;
            uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float q4_scale = decode_f16(scale_bits);
            uint8_t byte = block[2 + nibble_idx];
            int q_val = is_high ? (int(byte >> 4) - 8) : (int(byte & 0x0F) - 8);
            acc += (float)q_val * q4_scale * x[j];
        }
        out[tid] = acc;
    }
};

struct Q4VecMatFunctor {
    const uint8_t* q4;
    const float* x;
    float* out;
    uint32_t m;
    uint32_t k;

    void operator()(sycl::id<1> idx) const {
        size_t tid = idx[0];
        if (tid >= k) return;
        float acc = 0.0f;
        const uint32_t bytes_per_row = (k / 32) * 18;
        uint32_t block_idx = tid / 32;
        uint32_t nibble_idx = (tid % 32) / 2;
        bool is_high = (tid % 2) != 0;
        for (uint32_t row = 0; row < m; row++) {
            float act = x[row];
            const uint8_t* block = q4 + row * bytes_per_row + block_idx * 18;
            uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float q4_scale = decode_f16(scale_bits);
            uint8_t byte = block[2 + nibble_idx];
            int q_val = is_high ? (int(byte >> 4) - 8) : (int(byte & 0x0F) - 8);
            acc += (float)q_val * q4_scale * act;
        }
        out[tid] = acc;
    }
};

struct Q4kMatVec8sgFunctor {
    const uint8_t* w4k;
    const float* x;
    float* out;
    uint32_t n;
    uint32_t k;

    void operator()(sycl::nd_item<1> item) const {
        uint32_t tg_id = item.get_group(0);
        uint32_t tid_in_tg = item.get_local_id(0);
        uint32_t sg_id = tid_in_tg / 32;
        uint32_t lane = tid_in_tg % 32;

        uint32_t row_idx = tg_id * 8 + sg_id;
        if (row_idx >= n) return;

        const uint32_t superblocks = k / 256;
        const uint32_t BLOCK_SIZE = 144;
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
            const uint8_t* qs = block + 16 + group * 32 + sh * 16;

            float sumy = 0.0f;
            float dot_acc = 0.0f;
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
    }
};

struct Q6kMatVecFunctor {
    const uint8_t* w6k;
    const float* x;
    float* out;
    uint32_t n;
    uint32_t k;

    void operator()(sycl::nd_item<1> item) const {
        uint32_t tg_id = item.get_group(0);
        uint32_t tid_in_tg = item.get_local_id(0);
        uint32_t sg_id = tid_in_tg / 32;
        uint32_t lane = tid_in_tg % 32;

        uint32_t row_idx = tg_id * 4 + sg_id;
        if (row_idx >= n) return;

        const uint32_t superblocks = k / 256;
        const uint32_t BLOCK_SIZE = 210;
        const uint8_t* row = w6k + row_idx * (superblocks * BLOCK_SIZE);

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
            // Unrolled vector dot product for Q6_K
            {
                const uint32_t b = base;
                uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                float _sc = d * (float)sc[sc_base + 0];
                acc += _sc * (
                    (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb] +
                    (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 1] +
                    (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 2] +
                    (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 3]);
            }
            {
                const uint32_t b = base + 64;
                uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                float _sc = d * (float)sc[sc_base + 4];
                acc += _sc * (
                    (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb + 64] +
                    (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 65] +
                    (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 66] +
                    (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 67]);
            }
            {
                const uint32_t b = base + 128;
                uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                float _sc = d * (float)sc[sc_base + 8];
                acc += _sc * (
                    (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb + 128] +
                    (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 129] +
                    (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 130] +
                    (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 131]);
            }
            {
                const uint32_t b = base + 192;
                uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                float _sc = d * (float)sc[sc_base + 12];
                acc += _sc * (
                    (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb + 192] +
                    (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 193] +
                    (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 194] +
                    (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 195]);
            }
        }

        auto sub_g = item.get_sub_group();
        acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
        if (lane == 0) out[row_idx] = acc;
    }
};

struct Q4kFfnGateUpFunctor {
    const uint8_t* wg;
    const uint8_t* wu;
    const float* x;
    float* g_out;
    float* u_out;
    uint32_t n;
    uint32_t k;
    uint32_t tgs_per_mat;

    void operator()(sycl::nd_item<1> item) const {
        uint32_t tg_id = item.get_group(0);
        uint32_t tid_in_tg = item.get_local_id(0);
        uint32_t sg_id = tid_in_tg / 32;
        uint32_t lane = tid_in_tg % 32;

        bool is_up = (tg_id >= tgs_per_mat);
        uint32_t mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

        const uint32_t ROWS_PER_TG = 4;
        uint32_t row_idx = mat_tg * ROWS_PER_TG + sg_id;
        if (row_idx >= n) return;

        const uint8_t* w = is_up ? wu : wg;
        float* out_buf = is_up ? u_out : g_out;

        const uint32_t superblocks = k / 256;
        const uint32_t BLOCK_SIZE = 144;
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
            const uint8_t* qs = block + 16 + group * 32 + sh * 16;

            float sumy = 0.0f;
            float dot_acc = 0.0f;
            for (uint32_t l = 0; l < 16; l++) {
                float xv = x[x_base + l];
                sumy += xv;
                uint8_t byte = qs[l];
                float nib = hi ? (float)((byte >> 4) & 0x0F) : (float)(byte & 0x0F);
                dot_acc += nib * xv;
            }
            acc += scale * dot_acc - mmin * sumy;
        }

        auto sub_g = item.get_sub_group();
        acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
        if (lane == 0) out_buf[row_idx] = acc;
    }
};

struct SiluFunctor {
    const float* input;
    float* out;

    void operator()(sycl::id<1> idx) const {
        size_t i = idx[0];
        float x = input[i];
        out[i] = x / (1.0f + sycl::exp(-x));
    }
};

struct RmsNormFunctor {
    const float* x;
    const float* weight;
    float* out;
    size_t len;
    float eps;
    float offset;

    void operator()(sycl::nd_item<1> item) const {
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
    }
};

struct AttnFusedFunctor {
    const float* q_in;
    const float* k_in;
    const float* v_in;
    float* k_cache;
    float* v_cache;
    float* out;
    const float* q_weight;
    const float* k_weight;
    uint32_t t;
    uint32_t head_dim;
    uint32_t num_q;
    uint32_t num_kv;
    float scale;
    uint32_t window_size;
    float eps;
    float qk_offset;
    float rope_base;
    uint32_t rotary_dim;

    sycl::local_accessor<float, 1> tg_q;
    sycl::local_accessor<float, 1> tg_k_normed;
    sycl::local_accessor<float, 1> tg_red;
    sycl::local_accessor<float, 1> tg_scores;

    void operator()(sycl::nd_item<1> item) const {
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
            float acc_v = 0.0f;
            for (uint32_t it = t_start; it < t; it++) {
                acc_v += tg_scores[it - t_start] * v_cache[it * num_kv * head_dim + kv_head * head_dim + d];
            }
            out_head[d] = acc_v;
        }
    }
};

struct Q4kQkvProjFunctor {
    const uint8_t* wq;
    const uint8_t* wk;
    const uint8_t* wv;
    const float* x;
    float* q_out;
    float* k_out;
    float* v_out;
    uint32_t q_rows;
    uint32_t k_rows;
    uint32_t v_rows;
    uint32_t k;

    void operator()(sycl::nd_item<1> item) const {
        uint32_t tg_id = item.get_group(0);
        uint32_t tid_in_tg = item.get_local_id(0);
        uint32_t sg_id = tid_in_tg / 32;
        uint32_t lane = tid_in_tg % 32;

        uint32_t total_rows = q_rows + k_rows + v_rows;
        uint32_t global_row = tg_id * 8 + sg_id;
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
        const uint32_t BLOCK_SIZE = 144;
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
    }
};

struct Q4kProjFunctor {
    const uint8_t* w4k;
    const float* x;
    float* out;
    uint32_t n;
    uint32_t k;

    void operator()(sycl::nd_item<1> item) const {
        uint32_t tg_id = item.get_group(0);
        uint32_t tid_in_tg = item.get_local_id(0);
        uint32_t sg_id = tid_in_tg / 32;
        uint32_t lane = tid_in_tg % 32;

        uint32_t row_idx = tg_id * 8 + sg_id;
        if (row_idx >= n) return;

        uint32_t superblocks = k / 256;
        const uint32_t BLOCK_SIZE = 144;
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
    }
};

struct Q4kQ6kQkvProjFunctor {
    const uint8_t* wq;
    const uint8_t* wk;
    const uint8_t* wv;
    const float* x;
    float* q_out;
    float* k_out;
    float* v_out;
    uint32_t q_rows;
    uint32_t k_rows;
    uint32_t v_rows;
    uint32_t k;

    void operator()(sycl::nd_item<1> item) const {
        uint32_t tg_id = item.get_group(0);
        uint32_t tid_in_tg = item.get_local_id(0);
        uint32_t sg_id = tid_in_tg / 32;
        uint32_t lane = tid_in_tg % 32;

        uint32_t total_rows = q_rows + k_rows + v_rows;
        uint32_t global_row = tg_id * 4 + sg_id;
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

            const uint32_t Q4K_BLOCK_SIZE = 144;
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
                const uint8_t* qs = block + 16 + group * 32 + sh * 16;
                float dot_acc = 0.0f, sum_acc = 0.0f;
                for (uint32_t l = 0; l < 16; l++) {
                    float xi = x[x_base + l];
                    uint8_t byte = qs[l];
                    float nib = hi ? (float)((byte >> 4) & 0x0F) : (float)(byte & 0x0F);
                    dot_acc += nib * xi;
                    sum_acc += xi;
                }
                acc += scale * dot_acc - mmin * sum_acc;
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
            if (lane == 0) out_ptr[local_row] = acc;

        } else {
            uint32_t local_row = global_row - q_rows - k_rows;
            const uint32_t Q6K_BLOCK_SIZE = 210;
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
                {
                    const uint32_t b = base;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 0];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 1] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 2] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 3]);
                }
                {
                    const uint32_t b = base + 64;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 4];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb + 64] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 65] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 66] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 67]);
                }
                {
                    const uint32_t b = base + 128;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 8];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb + 128] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 129] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 130] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 131]);
                }
                {
                    const uint32_t b = base + 192;
                    uint8_t la = ql[b >> 1], lb = ql[(b >> 1) + 1], hi = qh[b >> 2];
                    float _sc = d * (float)sc[sc_base + 12];
                    acc += _sc * (
                        (float)((int8_t)((la & 0x0F) | ((hi & 0x03) << 4)) - 32) * x[xb + 192] +
                        (float)((int8_t)(((la >> 4) & 0x0F) | ((hi & 0x0C) << 2)) - 32) * x[xb + 193] +
                        (float)((int8_t)((lb & 0x0F) | ((hi & 0x30))) - 32) * x[xb + 194] +
                        (float)((int8_t)(((lb >> 4) & 0x0F) | ((hi & 0xC0) >> 2)) - 32) * x[xb + 195]);
                }
            }

            auto sub_g = item.get_sub_group();
            acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
            if (lane == 0) v_out[local_row] = acc;
        }
    }
};

struct GeluTanhFunctor {
    const float* input;
    float* out;

    void operator()(sycl::id<1> idx) const {
        size_t i = idx[0];
        float x = input[i];
        float c = 0.7978845608f; 
        float y = c * (x + 0.044715f * x * x * x);
        y = sycl::clamp(y, -15.0f, 15.0f);
        float t = sycl::tanh(y);
        out[i] = 0.5f * x * (1.0f + t);
    }
};

struct Q4MatVecQ8Functor {
    const uint8_t* q4;
    const int8_t* q8;
    const float* q8s;
    float* out;
    uint32_t n;
    uint32_t k;
    sycl::local_accessor<int8_t, 1> tg_q8;
    sycl::local_accessor<float, 1> tg_q8s;

    void operator()(sycl::nd_item<1> item) const {
        uint32_t tg_id = item.get_group(0);
        uint32_t tid_in_tg = item.get_local_id(0);
        uint32_t sg_id = tid_in_tg / 32;
        uint32_t lane = tid_in_tg % 32;

        uint32_t blocks = k / 32;
        uint32_t bytes_per_row = blocks * 18;

        for (uint32_t i = tid_in_tg; i < k; i += 256) tg_q8[i] = q8[i];
        for (uint32_t i = tid_in_tg; i < blocks; i += 256) tg_q8s[i] = q8s[i];
        item.barrier(sycl::access::fence_space::local_space);

        uint32_t row_idx = tg_id * 8 + sg_id;
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
            isum += (int)((w0 >> 0) & 0x0F) - 8 * (int)tg_q8[b * 32 + 0];
            isum += (int)((w0 >> 4) & 0x0F) - 8 * (int)tg_q8[b * 32 + 1];
            isum += (int)((w0 >> 8) & 0x0F) - 8 * (int)tg_q8[b * 32 + 2];
            isum += (int)((w0 >> 12) & 0x0F) - 8 * (int)tg_q8[b * 32 + 3];
            isum += (int)((w0 >> 16) & 0x0F) - 8 * (int)tg_q8[b * 32 + 4];
            isum += (int)((w0 >> 20) & 0x0F) - 8 * (int)tg_q8[b * 32 + 5];
            isum += (int)((w0 >> 24) & 0x0F) - 8 * (int)tg_q8[b * 32 + 6];
            isum += (int)((w0 >> 28) & 0x0F) - 8 * (int)tg_q8[b * 32 + 7];

            isum += (int)((w1 >> 0) & 0x0F) - 8 * (int)tg_q8[b * 32 + 8];
            isum += (int)((w1 >> 4) & 0x0F) - 8 * (int)tg_q8[b * 32 + 9];
            isum += (int)((w1 >> 8) & 0x0F) - 8 * (int)tg_q8[b * 32 + 10];
            isum += (int)((w1 >> 12) & 0x0F) - 8 * (int)tg_q8[b * 32 + 11];
            isum += (int)((w1 >> 16) & 0x0F) - 8 * (int)tg_q8[b * 32 + 12];
            isum += (int)((w1 >> 20) & 0x0F) - 8 * (int)tg_q8[b * 32 + 13];
            isum += (int)((w1 >> 24) & 0x0F) - 8 * (int)tg_q8[b * 32 + 14];
            isum += (int)((w1 >> 28) & 0x0F) - 8 * (int)tg_q8[b * 32 + 15];

            isum += (int)((w2 >> 0) & 0x0F) - 8 * (int)tg_q8[b * 32 + 16];
            isum += (int)((w2 >> 4) & 0x0F) - 8 * (int)tg_q8[b * 32 + 17];
            isum += (int)((w2 >> 8) & 0x0F) - 8 * (int)tg_q8[b * 32 + 18];
            isum += (int)((w2 >> 12) & 0x0F) - 8 * (int)tg_q8[b * 32 + 19];
            isum += (int)((w2 >> 16) & 0x0F) - 8 * (int)tg_q8[b * 32 + 20];
            isum += (int)((w2 >> 20) & 0x0F) - 8 * (int)tg_q8[b * 32 + 21];
            isum += (int)((w2 >> 24) & 0x0F) - 8 * (int)tg_q8[b * 32 + 22];
            isum += (int)((w2 >> 28) & 0x0F) - 8 * (int)tg_q8[b * 32 + 23];

            isum += (int)((w3 >> 0) & 0x0F) - 8 * (int)tg_q8[b * 32 + 24];
            isum += (int)((w3 >> 4) & 0x0F) - 8 * (int)tg_q8[b * 32 + 25];
            isum += (int)((w3 >> 8) & 0x0F) - 8 * (int)tg_q8[b * 32 + 26];
            isum += (int)((w3 >> 12) & 0x0F) - 8 * (int)tg_q8[b * 32 + 27];
            isum += (int)((w3 >> 16) & 0x0F) - 8 * (int)tg_q8[b * 32 + 28];
            isum += (int)((w3 >> 20) & 0x0F) - 8 * (int)tg_q8[b * 32 + 29];
            isum += (int)((w3 >> 24) & 0x0F) - 8 * (int)tg_q8[b * 32 + 30];
            isum += (int)((w3 >> 28) & 0x0F) - 8 * (int)tg_q8[b * 32 + 31];

            acc += (float)isum * combined_scale;
        }

        auto sub_g = item.get_sub_group();
        acc = sycl::reduce_over_group(sub_g, acc, sycl::plus<>());
        if (lane == 0) out[row_idx] = acc;
    }
};

struct RopeAtPosBatchedQkFunctor {
    float* q;
    float* k;
    uint32_t head_dim;
    float rope_base;
    uint32_t pos;
    uint32_t rdim;
    uint32_t hdim;
    uint32_t num_q;

    void operator()(sycl::id<2> idx) const {
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
    }
};

struct QuantizeQ8Functor {
    const float* input;
    int8_t* q8_out;
    float* scales;
    uint32_t k;

    void operator()(sycl::id<1> idx) const {
        uint32_t block = idx[0];
        uint32_t num_blocks = k / 32;
        if (block >= num_blocks) return;
        uint32_t off = block * 32;
        float amax = 0.0f;
        for (uint32_t j = 0; j < 32; j++) {
            float v = sycl::fabs(input[off + j]);
            if (v > amax) amax = v;
        }
        float scale = amax / 127.0f;
        float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        scales[block] = scale;
        for (uint32_t j = 0; j < 32; j++) {
            float v = input[off + j] * inv;
            v = sycl::clamp(v, -128.0f, 127.0f);
            q8_out[off + j] = (int8_t)sycl::round(v);
        }
    }
};

const float tq4_centroids[16] = {
    -0.1089, -0.0782, -0.0588, -0.0427,
    -0.0283, -0.0148, -0.0050,  0.0050,
     0.0148,  0.0283,  0.0427,  0.0588,
     0.0782,  0.1089,  0.1500,  0.2000
};

const float tq4_boundaries[15] = {
    -0.0936, -0.0685, -0.0508, -0.0355,
    -0.0216, -0.0099,  0.0000,  0.0099,
     0.0216,  0.0355,  0.0508,  0.0685,
     0.0936,  0.1295,  0.1750
};

static inline bool tq_sign_flip(uint32_t i) {
    return ((i * 2654435761u) >> 16) & 1u;
}

struct TurboQuantEncodeFunctor {
    const float* input;
    float* norms;
    uint8_t* packed;
    uint32_t d;
    uint32_t batch;
    sycl::local_accessor<float, 1> shared;

    void operator()(sycl::nd_item<1> it) const {
        uint32_t elem = it.get_local_id(0);
        uint32_t vec_idx = it.get_group(0);
        if (vec_idx >= batch || elem >= d) return;

        uint32_t base = vec_idx * d;
        shared[elem] = input[base + elem];
        it.barrier(sycl::access::fence_space::local_space);

        // Step 1: L2 norm (thread 0 computes, all read)
        if (elem == 0) {
            float sum_sq = 0.0f;
            for (uint32_t i = 0; i < d; i++) {
                sum_sq += shared[i] * shared[i];
            }
            norms[vec_idx] = sycl::sqrt(sum_sq);
        }
        it.barrier(sycl::access::fence_space::local_space);

        float norm = norms[vec_idx];
        float inv_norm = (norm > 1e-12f) ? (1.0f / norm) : 0.0f;

        // Step 2: Normalize
        shared[elem] *= inv_norm;
        it.barrier(sycl::access::fence_space::local_space);

        // Step 3: Sign flips
        if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
        it.barrier(sycl::access::fence_space::local_space);

        // Step 4: WHT
        for (uint32_t hstep = 1; hstep < d; hstep *= 2) {
            uint32_t blk = hstep * 2;
            uint32_t blk_idx = elem / blk;
            uint32_t within = elem % blk;
            if (within < hstep) {
                uint32_t j = blk_idx * blk + within;
                float a = shared[j];
                float b = shared[j + hstep];
                shared[j] = a + b;
                shared[j + hstep] = a - b;
            }
            it.barrier(sycl::access::fence_space::local_space);
        }

        shared[elem] *= 1.0f / sycl::sqrt((float)d);

        // Step 5: Sign flips again
        if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
        it.barrier(sycl::access::fence_space::local_space);

        // Step 6: Quantize
        float y = shared[elem];
        uint32_t idx = 0;
        for (uint32_t b = 0; b < 15; b++) {
            if (y > tq4_boundaries[b]) idx = b + 1;
        }

        // Step 7: Pack 4-bit
        uint32_t pack_offset = vec_idx * (d / 2) + elem / 2;
        if (elem % 2 == 0) {
            packed[pack_offset] = (uint8_t)(idx & 0x0F);
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (elem % 2 == 1) {
            packed[pack_offset] |= (uint8_t)((idx & 0x0F) << 4);
        }
    }
};

struct TurboQuantDecodeFunctor {
    const float* norms;
    const uint8_t* packed;
    float* output;
    uint32_t d;
    uint32_t batch;
    sycl::local_accessor<float, 1> shared;

    void operator()(sycl::nd_item<1> it) const {
        uint32_t elem = it.get_local_id(0);
        uint32_t vec_idx = it.get_group(0);
        if (vec_idx >= batch || elem >= d) return;

        // Step 1: Unpack 4-bit index
        uint32_t pack_offset = vec_idx * (d / 2) + elem / 2;
        uint8_t byte_val = packed[pack_offset];
        uint32_t idx = (elem % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);

        // Step 2: Centroid lookup
        shared[elem] = tq4_centroids[idx];
        it.barrier(sycl::access::fence_space::local_space);

        // Step 3: Sign flips
        if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
        it.barrier(sycl::access::fence_space::local_space);

        // Step 4: WHT
        for (uint32_t hstep = 1; hstep < d; hstep *= 2) {
            uint32_t blk = hstep * 2;
            uint32_t blk_idx = elem / blk;
            uint32_t within = elem % blk;
            if (within < hstep) {
                uint32_t j = blk_idx * blk + within;
                float a = shared[j];
                float b = shared[j + hstep];
                shared[j] = a + b;
                shared[j + hstep] = a - b;
            }
            it.barrier(sycl::access::fence_space::local_space);
        }

        shared[elem] *= 1.0f / sycl::sqrt((float)d);

        // Step 5: Sign flips again
        if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
        it.barrier(sycl::access::fence_space::local_space);

        // Step 6: Rescale by norm
        output[vec_idx * d + elem] = shared[elem] * norms[vec_idx];
    }
};

        }
    }
};

struct SgemmFunctor {
    const float* a;
    const float* b;
    float* c;
    uint32_t m;
    uint32_t n;
    uint32_t k;

    void operator()(sycl::nd_item<2> it) const {
        uint32_t row = it.get_global_id(1);
        uint32_t col = it.get_global_id(0);

        if (row < m && col < n) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < k; i++) {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
};

struct SgemmTransBFunctor {
    const float* a;
    const float* b;
    float* c;
    uint32_t m;
    uint32_t n;
    uint32_t k;

    void operator()(sycl::nd_item<2> it) const {
        uint32_t row = it.get_global_id(1);
        uint32_t col = it.get_global_id(0);

        if (row < m && col < n) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < k; i++) {
                sum += a[row * k + i] * b[col * k + i];
            }
            c[row * n + col] = sum;
        }
    }
};

void check_sycl() {
    std::cout << "Checking SYCL..." << std::endl;
    if (!init_xpu()) {
        std::cout << "SYCL Init Failed!" << std::endl;
        return;
    }
    std::cout << "SYCL Init Success. Device: " << g_queue->get_device().get_info<sycl::info::device::name>() << std::endl;
}

// Function Wrappers

class F32GemvKernel;

void f32_gemv(const float* x, const float* a, float* y, size_t m, size_t k) {
    std::cout << "f32_gemv start: m=" << m << ", k=" << k << std::endl;
    if (!init_xpu()) return;
    std::cout << "f32_gemv: dispatching parallel_for..." << std::endl;
    try {
        g_queue->parallel_for<F32GemvKernel>(sycl::range<1>(m), F32GemvFunctor{x, a, y, (uint32_t)k}).wait();
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Kernel Error: " << e.what() << std::endl;
    }
    std::cout << "f32_gemv end" << std::endl;
}

void q4_matvec_v4(const uint8_t* q4, const float* x, float* out, size_t n, size_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), Q4MatVecV4Functor{q4, x, out, (uint32_t)n, (uint32_t)k}).wait();
}

void q4_vecmat(const uint8_t* q4, const float* x, float* out, size_t m, size_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(k), Q4VecMatFunctor{q4, x, out, (uint32_t)m, (uint32_t)k}).wait();
}

void q4k_matvec_8sg(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) {
    if (!init_xpu()) return;
    const uint32_t THREADS_PER_TG = 256;
    g_queue->parallel_for(sycl::nd_range<1>(sycl::range<1>((n / 8) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), 
        Q4kMatVec8sgFunctor{w4k, x, out, (uint32_t)n, (uint32_t)k}).wait();
}

void q6k_matvec(const uint8_t* w6k, const float* x, float* out, size_t n, size_t k) {
    if (!init_xpu()) return;
    const uint32_t THREADS_PER_TG = 128;
    g_queue->parallel_for(sycl::nd_range<1>(sycl::range<1>((n / 4) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), 
        Q6kMatVecFunctor{w6k, x, out, (uint32_t)n, (uint32_t)k}).wait();
}

void q4k_ffn_gate_up(const uint8_t* wg, const uint8_t* wu, const float* x, float* g_out, float* u_out, size_t n, size_t k) {
    if (!init_xpu()) return;
    const uint32_t ROWS_PER_TG = 4;
    const uint32_t THREADS_PER_TG = 128;
    uint32_t tgs_per_mat = (uint32_t)((n + ROWS_PER_TG - 1) / ROWS_PER_TG);
    g_queue->parallel_for(sycl::nd_range<1>(sycl::range<1>(2 * tgs_per_mat * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), 
        Q4kFfnGateUpFunctor{wg, wu, x, g_out, u_out, (uint32_t)n, (uint32_t)k, tgs_per_mat}).wait();
}

void silu(const float* input, float* out, size_t n) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), SiluFunctor{input, out}).wait();
}

void rms_norm(const float* x, const float* weight, float* out, size_t len, float eps, float offset) {
    if (!init_xpu()) return;
    size_t wg_size = 256; 
    g_queue->parallel_for(sycl::nd_range<1>(sycl::range<1>(wg_size), sycl::range<1>(wg_size)), RmsNormFunctor{x, weight, out, len, eps, offset}).wait();
}

void attn_fused(const float* q_in, const float* k_in, const float* v_in, float* k_cache, float* v_cache, float* out, const float* q_weight, const float* k_weight, uint32_t t, uint32_t head_dim, uint32_t num_q, uint32_t num_kv, float scale, uint32_t window_size, float eps, float qk_offset, float rope_base, uint32_t rotary_dim) {
    if (!init_xpu()) return;
    const uint32_t THREADS_PER_TG = 256; 
    g_queue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> tg_q(sycl::range<1>(256), h);
        sycl::local_accessor<float, 1> tg_k_normed(sycl::range<1>(256), h);
        sycl::local_accessor<float, 1> tg_red(sycl::range<1>(8), h);
        sycl::local_accessor<float, 1> tg_scores(sycl::range<1>(1024), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_q * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), 
            AttnFusedFunctor{q_in, k_in, v_in, k_cache, v_cache, out, q_weight, k_weight, t, head_dim, num_q, num_kv, scale, window_size, eps, qk_offset, rope_base, rotary_dim, tg_q, tg_k_normed, tg_red, tg_scores});
    }).wait();
}

void q4k_qkv_proj(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x, float* q_out, float* k_out, float* v_out, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t k) {
    if (!init_xpu()) return;
    const uint32_t THREADS_PER_TG = 256;
    g_queue->parallel_for(sycl::nd_range<1>(sycl::range<1>(((q_rows + k_rows + v_rows + 7) / 8) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), 
        Q4kQkvProjFunctor{wq, wk, wv, x, q_out, k_out, v_out, (uint32_t)q_rows, (uint32_t)k_rows, (uint32_t)v_rows, (uint32_t)k}).wait();
}

void q4k_proj(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) {
    if (!init_xpu()) return;
    const uint32_t THREADS_PER_TG = 256;
    g_queue->parallel_for(sycl::nd_range<1>(sycl::range<1>(((n + 7) / 8) * THREADS_PER_TG), sycl::range<1>(THREADS_PER_TG)), 
        Q4kProjFunctor{w4k, x, out, (uint32_t)n, (uint32_t)k}).wait();
}

void gelu_tanh(const float* input, float* out, size_t n) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), GeluTanhFunctor{input, out}).wait();
}

void q4_matvec_v4(const uint8_t* q4, const int8_t* q8, const float* q8s, float* out, size_t n, size_t k) {
    if (!init_xpu()) return;
    g_queue->submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> tg_q8(sycl::range<1>(8192), h);
        sycl::local_accessor<float, 1> tg_q8s(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((n / 8) * 256), sycl::range<1>(256)), 
            Q4MatVecQ8Functor{q4, q8, q8s, out, (uint32_t)n, (uint32_t)k, tg_q8, tg_q8s});
    }).wait();
}

void rope_at_pos_batched_qk(float* q, float* k, size_t head_dim, float rope_base, size_t pos, size_t rotary_dim, size_t num_q, size_t num_kv) {
    if (!init_xpu()) return;
    size_t total_heads = num_q + num_kv;
    size_t rdim = (rotary_dim == 0) ? head_dim : std::min(rotary_dim, head_dim);
    size_t hdim = rdim / 2;
    g_queue->parallel_for(sycl::range<2>(total_heads, hdim), 
        RopeAtPosBatchedQkFunctor{q, k, (uint32_t)head_dim, rope_base, (uint32_t)pos, (uint32_t)rdim, (uint32_t)hdim, (uint32_t)num_q}).wait();
}

void quantize_q8(const float* input, int8_t* q8_out, float* scales, uint32_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(k / 32), QuantizeQ8Functor{input, q8_out, scales, k}).wait();
}

void turboquant_encode(const float* input, float* norms, uint8_t* packed, uint32_t d, uint32_t batch) {
    if (!init_xpu()) return;
    g_queue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(d), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(batch * d), sycl::range<1>(d)), TurboQuantEncodeFunctor{input, norms, packed, d, batch, shared});
    }).wait();
}

void turboquant_decode(const float* norms, const uint8_t* packed, float* output, uint32_t d, uint32_t batch) {
    if (!init_xpu()) return;
    g_queue->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(d), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(batch * d), sycl::range<1>(d)), TurboQuantDecodeFunctor{norms, packed, output, d, batch, shared});
    }).wait();
}

void sgemm(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::nd_range<2>(sycl::range<2>((n + 31) / 32 * 32, (m + 31) / 32 * 32), sycl::range<2>(32, 32)), SgemmFunctor{a, b, c, m, n, k}).wait();
}

void sgemm_transb(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::nd_range<2>(sycl::range<2>((n + 31) / 32 * 32, (m + 31) / 32 * 32), sycl::range<2>(32, 32)), SgemmTransBFunctor{a, b, c, m, n, k}).wait();
}

void q8_matvec(const int8_t* w8, const int8_t* x8, const float* w8s, const float* x8s, float* out, uint32_t n, uint32_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        uint32_t row = idx[0];
        uint32_t blocks = k / 32;
        float row_acc = 0.0f;
        for (uint32_t b = 0; b < blocks; b++) {
            float combined_scale = w8s[row * blocks + b] * x8s[b];
            const int8_t* wb = w8 + row * k + b * 32;
            const int8_t* xb = x8 + b * 32;
            int32_t isum = 0;
            for (uint32_t i = 0; i < 32; i++) {
                isum += (int32_t)wb[i] * (int32_t)xb[i];
            }
            row_acc += (float)isum * combined_scale;
        }
        out[row] = row_acc;
    }).wait();
}

void q4_sparse_matvec(const uint8_t* q4, const int8_t* q8, const float* q8s, const uint32_t* indices, float* out, uint32_t k_selected, uint32_t hidden) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(k_selected), [=](sycl::id<1> idx) {
        uint32_t tid = idx[0];
        uint32_t row_idx = indices[tid];
        uint32_t blocks = hidden / 32;
        uint32_t bytes_per_row = blocks * 18;
        const uint8_t* row_w = q4 + row_idx * bytes_per_row;

        float acc = 0.0f;
        for (uint32_t b = 0; b < blocks; b++) {
            const uint8_t* blk = row_w + b * 18;
            uint16_t sb_bits = blk[0] | (blk[1] << 8);
            float combined_scale = decode_f16(sb_bits) * q8s[b];
            const uint8_t* qb = blk + 2;
            const int8_t* q8_ptr = q8 + b * 32;

            int32_t isum = 0;
            for (uint32_t i = 0; i < 16; i++) {
                uint8_t byte = qb[i];
                int32_t v_lo = (int32_t)(byte & 0x0F) - 8;
                int32_t v_hi = (int32_t)(byte >> 4) - 8;
                isum += v_lo * (int32_t)q8_ptr[i];
                isum += v_hi * (int32_t)q8_ptr[i + 16];
            }
            acc += (float)isum * combined_scale;
        }
        out[tid] = acc;
    }).wait();
}

void q4k_matvec_stride32(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        uint32_t row_idx = idx[0];
        uint32_t superblocks = k / 256;
        uint32_t bytes_per_row = superblocks * 144;
        const uint8_t* row_w = w4k + row_idx * bytes_per_row;

        float row_acc = 0.0f;
        for (uint32_t sb = 0; sb < superblocks; sb++) {
            const uint8_t* block = row_w + sb * 144;
            uint16_t d_bits = block[0] | (block[1] << 8);
            uint16_t dmin_bits = block[2] | (block[3] << 8);
            float d = decode_f16(d_bits);
            float dmin = decode_f16(dmin_bits);
            const uint8_t* sb_bytes = block + 4;

            float sb_acc = 0.0f;
            for (uint32_t sub = 0; sub < 8; sub++) {
                uint32_t sc, mn;
                if (sub < 4) {
                    sc = sb_bytes[sub] & 0x3F;
                    mn = sb_bytes[sub + 4] & 0x3F;
                } else {
                    sc = (sb_bytes[sub + 4] & 0x0F) | ((sb_bytes[sub - 4] >> 6) << 4);
                    mn = (sb_bytes[sub + 4] >> 4) | ((sb_bytes[sub] >> 6) << 4);
                }
                float scale = d * (float)sc;
                float mmin = dmin * (float)mn;

                uint32_t group = sub >> 1;
                bool hi = (sub & 1) != 0;
                const uint8_t* qs = block + 16 + group * 32;

                for (uint32_t i = 0; i < 32; i++) {
                    uint8_t byte = qs[i];
                    float nib = hi ? (float)((byte >> 4) & 0x0F) : (float)(byte & 0x0F);
                    uint32_t x_idx = sb * 256 + sub * 32 + i;
                    sb_acc += (scale * nib - mmin) * x[x_idx];
                }
            }
            row_acc += sb_acc;
        }
        out[row_idx] = row_acc;
    }).wait();
}

#ifdef SYCL_DLL_BUILD
extern "C" {
    EXPORT bool dll_init_xpu() { return init_xpu(); }
    EXPORT void dll_get_device_info(char* buf, int max_len) {
        if (!init_xpu()) { strncpy(buf, "None", max_len); return; }
        auto name = g_queue->get_device().get_info<sycl::info::device::name>();
        strncpy(buf, name.c_str(), max_len);
    }
    EXPORT uint8_t* dll_allocate_device(size_t size) { return allocate_device(size); }
    EXPORT uint8_t* dll_allocate_shared(size_t size) { return allocate_shared(size); }
    EXPORT void dll_free_memory(uint8_t* ptr) { free_memory(ptr); }
    EXPORT void dll_copy_h2d(uint8_t* dst, const uint8_t* src, size_t size) { copy_h2d(dst, src, size); }
    EXPORT void dll_copy_d2h(uint8_t* dst, const uint8_t* src, size_t size) { copy_d2h(dst, src, size); }
    EXPORT void dll_f32_gemv(const float* x, const float* a, float* y, size_t m, size_t k) { f32_gemv(x, a, y, m, k); }
    EXPORT void dll_rms_norm(const float* x, const float* weight, float* out, size_t len, float eps, float offset) { rms_norm(x, weight, out, len, eps, offset); }
    EXPORT void dll_silu(const float* input, float* out, size_t n) { silu(input, out, n); }
    EXPORT void dll_gelu_tanh(const float* input, float* out, size_t n) { gelu_tanh(input, out, n); }
    EXPORT void dll_rope_at_pos_batched_qk(float* q, float* k, size_t head_dim, float rope_base, size_t pos, size_t rotary_dim, size_t num_q, size_t num_kv) { 
        rope_at_pos_batched_qk(q, k, head_dim, rope_base, pos, rotary_dim, num_q, num_kv); 
    }
    EXPORT void dll_q4_vecmat(const uint8_t* q4, const float* x, float* out, size_t m, size_t k) { q4_vecmat(q4, x, out, m, k); }
    EXPORT void dll_q4k_matvec_8sg(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) { q4k_matvec_8sg(w4k, x, out, n, k); }
    EXPORT void dll_q6k_matvec(const uint8_t* w6k, const float* x, float* out, size_t n, size_t k) { q6k_matvec(w6k, x, out, n, k); }
    EXPORT void dll_q4k_ffn_gate_up(const uint8_t* wg, const uint8_t* wu, const float* x, float* g_out, float* u_out, size_t n, size_t k) {
        q4k_ffn_gate_up(wg, wu, x, g_out, u_out, n, k);
    }
    EXPORT void dll_q4_matvec_v4(const uint8_t* q4, const int8_t* q8, const float* q8s, float* out, size_t n, size_t k) {
        q4_matvec_v4(q4, q8, q8s, out, n, k);
    }
    EXPORT void dll_attn_fused(const float* q_in, const float* k_in, const float* v_in, float* k_cache, float* v_cache, float* out, const float* q_weight, const float* k_weight, uint32_t t, uint32_t head_dim, uint32_t num_q, uint32_t num_kv, float scale, uint32_t window_size, float eps, float qk_offset, float rope_base, uint32_t rotary_dim) {
        attn_fused(q_in, k_in, v_in, k_cache, v_cache, out, q_weight, k_weight, t, head_dim, num_q, num_kv, scale, window_size, eps, qk_offset, rope_base, rotary_dim);
    }
    EXPORT void dll_q4k_qkv_proj(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x, float* q_out, float* k_out, float* v_out, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t k) {
        q4k_qkv_proj(wq, wk, wv, x, q_out, k_out, v_out, q_rows, k_rows, v_rows, k);
    }
    EXPORT void dll_q4k_proj(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) { q4k_proj(w4k, x, out, n, k); }
    EXPORT void dll_quantize_q8(const float* input, int8_t* q8_out, float* scales, uint32_t k) { quantize_q8(input, q8_out, scales, k); }
    EXPORT void dll_turboquant_encode(const float* input, float* norms, uint8_t* packed, uint32_t d, uint32_t batch) { turboquant_encode(input, norms, packed, d, batch); }
    EXPORT void dll_turboquant_decode(const float* norms, const uint8_t* packed, float* output, uint32_t d, uint32_t batch) { turboquant_decode(norms, packed, output, d, batch); }
    EXPORT void dll_q8_matvec(const int8_t* w8, const int8_t* x8, const float* w8s, const float* x8s, float* out, uint32_t n, uint32_t k) {
        q8_matvec(w8, x8, w8s, x8s, out, n, k);
    }
    EXPORT void dll_q4_sparse_matvec(const uint8_t* q4, const int8_t* q8, const float* q8s, const uint32_t* indices, float* out, uint32_t k_selected, uint32_t hidden) {
        q4_sparse_matvec(q4, q8, q8s, indices, out, k_selected, hidden);
    }
    EXPORT void dll_q4k_matvec_stride32(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) {
        q4k_matvec_stride32(w4k, x, out, n, k);
    }
    EXPORT void dll_sgemm(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k) { sgemm(a, b, c, m, n, k); }
    EXPORT void dll_sgemm_transb(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k) { sgemm_transb(a, b, c, m, n, k); }
    EXPORT void dll_check_sycl() { check_sycl(); }
}
#endif
