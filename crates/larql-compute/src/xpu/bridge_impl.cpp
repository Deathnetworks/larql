#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>

static std::unique_ptr<sycl::queue> g_queue = nullptr;

bool init_xpu() {
    if (g_queue) return true;
    try {
        g_queue = std::make_unique<sycl::queue>(sycl::default_selector_v);
        return true;
    } catch (const sycl::exception& e) {
        return false;
    }
}

rust::String get_device_info() {
    if (!init_xpu()) return rust::String("None");
    auto d_name = g_queue->get_device().get_info<sycl::info::device::name>();
    return rust::String(d_name);
}

uint8_t* allocate_device(size_t size) {
    if (!init_xpu()) return nullptr;
    return (uint8_t*)sycl::malloc_device(size, *g_queue);
}

uint8_t* allocate_shared(size_t size) {
    if (!init_xpu()) return nullptr;
    return (uint8_t*)sycl::malloc_shared(size, *g_queue);
}

void free_memory(uint8_t* ptr) {
    if (g_queue && ptr) sycl::free(ptr, *g_queue);
}

void copy_h2d(uint8_t* dst, const uint8_t* src, size_t size) {
    if (g_queue) g_queue->memcpy(dst, src, size).wait();
}

void copy_d2h(uint8_t* dst, const uint8_t* src, size_t size) {
    if (g_queue) g_queue->memcpy(dst, src, size).wait();
}

// Kernels

struct F32GemvKernel {
    const float* x;
    const float* a;
    float* y;
    uint32_t k;
    void operator()(sycl::id<1> idx) const {
        size_t row = idx[0];
        float sum = 0.0f;
        for (size_t col = 0; col < k; ++col) sum += x[col] * a[row * k + col];
        y[row] = sum;
    }
};

void f32_gemv(const float* x, const float* a, float* y, size_t m, size_t k) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(m), F32GemvKernel{x, a, y, (uint32_t)k}).wait();
}

struct RmsNormKernel {
    const float* x;
    const float* w;
    float* out;
    size_t len;
    float eps;
    void operator()(sycl::id<1> idx) const {
        float ss = 0.0f;
        for (size_t i = 0; i < len; ++i) ss += x[i] * x[i];
        float inv_rms = 1.0f / std::sqrt(ss / len + eps);
        for (size_t i = 0; i < len; ++i) out[i] = x[i] * inv_rms * w[i];
    }
};

void rms_norm(const float* x, const float* weight, float* out, size_t len, float eps, float offset) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(1), RmsNormKernel{x, weight, out, len, eps}).wait();
}

struct SiluKernel {
    const float* in;
    float* out;
    void operator()(sycl::id<1> idx) const {
        float v = in[idx[0]];
        out[idx[0]] = v / (1.0f + std::exp(-v));
    }
};

void silu(const float* input, float* out, size_t n) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), SiluKernel{input, out}).wait();
}

void check_sycl() {
    init_xpu();
}

// Implementations for direct names
void gelu_tanh(const float*, float*, size_t) {}
void rope_at_pos_batched_qk(float*, float*, size_t, float, size_t, size_t, size_t, size_t) {}
void q4_vecmat(const uint8_t*, const float*, float*, size_t, size_t) {}
void q4k_matvec_8sg(const uint8_t*, const float*, float*, size_t, size_t) {}
void q6k_matvec(const uint8_t*, const float*, float*, size_t, size_t) {}
void q4k_ffn_gate_up(const uint8_t*, const uint8_t*, const float*, float*, float*, size_t, size_t) {}
void q4_matvec_v4(const uint8_t*, const int8_t*, const float*, float*, size_t, size_t) {}
void attn_fused(const float*, const float*, const float*, float*, float*, float*, const float*, const float*, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, float, float, float, uint32_t) {}
void q4k_qkv_proj(const uint8_t*, const uint8_t*, const uint8_t*, const float*, float*, float*, float*, uint32_t, uint32_t, uint32_t, uint32_t) {}
void q4k_proj(const uint8_t*, const float*, float*, size_t, size_t) {}

// Implementations matching ffi.rs dll_* prefixes where used
// NOTE: We don't use extern "C" here because cxx expects these to be in the global C++ namespace
// but ffi.rs calls them 'dll_quantize_q8' etc. in its bridge.

void dll_quantize_q8(const float*, int8_t*, float*, uint32_t) {}
void dll_turboquant_encode(const float*, float*, uint8_t*, uint32_t, uint32_t) {}
void dll_turboquant_decode(const float*, const uint8_t*, float*, uint32_t, uint32_t) {}
void dll_sgemm(const float*, const float*, float*, uint32_t, uint32_t, uint32_t) {}
void dll_sgemm_transb(const float*, const float*, float*, uint32_t, uint32_t, uint32_t) {}
void dll_layer_norm(const float*, const float*, const float*, float*, uint32_t, float, float, bool) {}
void dll_v_norm(const float*, float*, uint32_t, uint32_t, float, bool) {}
void dll_qk_norm_rope_fused(float*, float*, const float*, const float*, uint32_t, uint32_t, float, float, float, uint32_t, uint32_t) {}
void dll_q4k_q6k_qkv_proj(const uint8_t*, const uint8_t*, const uint8_t*, const float*, float*, float*, float*, uint32_t, uint32_t, uint32_t, uint32_t) {}
void dll_q8_matvec(const int8_t*, const int8_t*, const float*, const float*, float*, uint32_t, uint32_t) {}
void dll_q4_sparse_matvec(const uint8_t*, const int8_t*, const float*, const uint32_t*, float*, uint32_t, uint32_t) {}
void dll_q4k_matvec_stride32(const uint8_t*, const float*, float*, size_t, size_t) {}
struct GegluSiluKernel {
    const float* gate;
    const float* up;
    float* out;
    void operator()(sycl::id<1> idx) const {
        float g = gate[idx[0]];
        float u = up[idx[0]];
        out[idx[0]] = (g / (1.0f + std::exp(-g))) * u;
    }
};

void dll_geglu_silu(const float* gate, const float* up, float* out, size_t n) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), GegluSiluKernel{gate, up, out}).wait();
}

struct GegluGeluTanhKernel {
    const float* gate;
    const float* up;
    float* out;
    void operator()(sycl::id<1> idx) const {
        float g = gate[idx[0]];
        float u = up[idx[0]];
        float gelu = 0.5f * g * (1.0f + std::tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
        out[idx[0]] = gelu * u;
    }
};

void dll_geglu_gelu_tanh(const float* gate, const float* up, float* out, size_t n) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(n), GegluGeluTanhKernel{gate, up, out}).wait();
}

struct ResidualOpsKernel {
    const float* a;
    const float* b;
    float* out;
    float scalar;
    uint32_t mode;
    void operator()(sycl::id<1> idx) const {
        float va = a[idx[0]];
        float vb = b[idx[0]];
        if (mode == 0) {
            out[idx[0]] = va + vb * scalar;
        } else if (mode == 1) {
            out[idx[0]] = va * vb * scalar;
        }
    }
};

void dll_residual_ops(const float* a, const float* b, float* out, uint32_t len, float scalar, uint32_t mode) {
    if (!init_xpu()) return;
    g_queue->parallel_for(sycl::range<1>(len), ResidualOpsKernel{a, b, out, scalar, mode}).wait();
}
