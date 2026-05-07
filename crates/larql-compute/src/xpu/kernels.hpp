#pragma once
#include <string>
#include <memory>
#include <cstdint>
#include <cstddef>
#ifndef SYCL_BRIDGE_ONLY
#include <sycl/sycl.hpp>
#include "cxx.h"
#else
#include "rust/cxx.h"
#endif

// Core SYCL functions
bool init_xpu();
rust::String get_device_info();

// DLL Exports (C Linkage)
extern "C" {
    bool dll_init_xpu();
    void dll_get_device_info(char* buf, int max_len);
    
    uint8_t* dll_allocate_device(size_t size);
    uint8_t* dll_allocate_shared(size_t size);
    void dll_free_memory(uint8_t* ptr);
    void dll_copy_h2d(uint8_t* dst, const uint8_t* src, size_t size);
    void dll_copy_d2h(uint8_t* dst, const uint8_t* src, size_t size);

    void dll_f32_gemv(const float* x, const float* a, float* y, size_t m, size_t k);
    void dll_rms_norm(const float* x, const float* weight, float* out, size_t len, float eps, float offset);
    void dll_silu(const float* input, float* out, size_t n);
    void dll_gelu_tanh(const float* input, float* out, size_t n);

    void dll_quantize_q8(const float* input, int8_t* q8_out, float* scales, uint32_t k);
    void dll_turboquant_encode(const float* input, float* norms, uint8_t* packed, uint32_t d, uint32_t batch);
    void dll_turboquant_decode(const float* norms, const uint8_t* packed, float* output, uint32_t d, uint32_t batch);
    void dll_sgemm(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k);
    void dll_sgemm_transb(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k);
    
    void dll_rope_at_pos_batched_qk(float* q, float* k, size_t head_dim, float rope_base, size_t pos, size_t rotary_dim, size_t num_q, size_t num_kv);
    void dll_q4_vecmat(const uint8_t* q4, const float* x, float* out, size_t m, size_t k);
    void dll_q4k_matvec_8sg(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);
    void dll_q6k_matvec(const uint8_t* w6k, const float* x, float* out, size_t n, size_t k);
    void dll_q4k_ffn_gate_up(const uint8_t* wg, const uint8_t* wu, const float* x, float* g_out, float* u_out, size_t n, size_t k);
    void dll_q4_matvec_v4(const uint8_t* q4, const int8_t* q8, const float* q8s, float* out, size_t n, size_t k);
    void dll_attn_fused(const float* q_in, const float* k_in, const float* v_in, float* k_cache, float* v_cache, float* out, const float* q_weight, const float* k_weight, uint32_t t, uint32_t head_dim, uint32_t num_q, uint32_t num_kv, float scale, uint32_t window_size, float eps, float qk_offset, float rope_base, uint32_t rotary_dim);
    void dll_q4k_qkv_proj(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x, float* q_out, float* k_out, float* v_out, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t k);
    void dll_q4k_proj(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);
    
    void dll_layer_norm(const float* x, const float* weight, const float* bias, float* out, uint32_t len, float eps, float offset, bool has_bias);
    void dll_v_norm(const float* x, float* out, uint32_t head_dim, uint32_t num_heads, float eps, bool batched);
    void dll_qk_norm_rope_fused(float* q, float* k, const float* q_weight, const float* k_weight, uint32_t head_dim, uint32_t num_q, float eps, float offset, float rope_base, uint32_t pos, uint32_t rotary_dim);
    void dll_q4k_q6k_qkv_proj(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x, float* q_out, float* k_out, float* v_out, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t k);

    void dll_geglu_silu(const float* gate, const float* up, float* out, size_t n);
    void dll_geglu_gelu_tanh(const float* gate, const float* up, float* out, size_t n);
    void dll_residual_ops(const float* a, const float* b, float* out, uint32_t len, float scalar, uint32_t mode);
    
    void dll_q8_matvec(const int8_t* w8, const int8_t* x8, const float* w8s, const float* x8s, float* out, uint32_t n, uint32_t k);
    void dll_q4_sparse_matvec(const uint8_t* q4, const int8_t* q8, const float* q8s, const uint32_t* indices, float* out, uint32_t k_selected, uint32_t hidden);
    void dll_q4k_matvec_stride32(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);
}

// C++ internal versions (for kernels.cpp)
uint8_t* allocate_device(size_t size);
uint8_t* allocate_shared(size_t size);
void free_memory(uint8_t* ptr);
void copy_h2d(uint8_t* dst, const uint8_t* src, size_t size);
void copy_d2h(uint8_t* dst, const uint8_t* src, size_t size);

void f32_gemv(const float* x, const float* a, float* y, size_t m, size_t k);
void rms_norm(const float* x, const float* weight, float* out, size_t len, float eps, float offset);
void silu(const float* input, float* out, size_t n);
void gelu_tanh(const float* input, float* out, size_t n);
void rope_at_pos_batched_qk(float* q, float* k, size_t head_dim, float rope_base, size_t pos, size_t rotary_dim, size_t num_q, size_t num_kv);
void q4_vecmat(const uint8_t* q4, const float* x, float* out, size_t m, size_t k);
void q4k_matvec_8sg(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);
void q6k_matvec(const uint8_t* w6k, const float* x, float* out, size_t n, size_t k);
void q4k_ffn_gate_up(const uint8_t* wg, const uint8_t* wu, const float* x, float* g_out, float* u_out, size_t n, size_t k);
void q4_matvec_v4(const uint8_t* q4, const int8_t* q8, const float* q8s, float* out, size_t n, size_t k);
void attn_fused(const float* q_in, const float* k_in, const float* v_in, float* k_cache, float* v_cache, float* out, const float* q_weight, const float* k_weight, uint32_t t, uint32_t head_dim, uint32_t num_q, uint32_t num_kv, float scale, uint32_t window_size, float eps, float qk_offset, float rope_base, uint32_t rotary_dim);
void q4k_qkv_proj(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x, float* q_out, float* k_out, float* v_out, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t k);
void q4k_proj(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);
void quantize_q8(const float* input, int8_t* q8_out, float* scales, uint32_t k);
void turboquant_encode(const float* input, float* norms, uint8_t* packed, uint32_t d, uint32_t batch);
void turboquant_decode(const float* norms, const uint8_t* packed, float* output, uint32_t d, uint32_t batch);
void sgemm(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k);
void sgemm_transb(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k);
void q8_matvec(const int8_t* w8, const int8_t* x8, const float* w8s, const float* x8s, float* out, uint32_t n, uint32_t k);
void q4_sparse_matvec(const uint8_t* q4, const int8_t* q8, const float* q8s, const uint32_t* indices, float* out, uint32_t k_selected, uint32_t hidden);
void q4k_matvec_stride32(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);

void check_sycl();
