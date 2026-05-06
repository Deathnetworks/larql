#pragma once
#include <string>
#include <memory>
#include <sycl/sycl.hpp>
#include "rust/cxx.h"

bool init_xpu();
rust::String get_device_info();

uint8_t* allocate_device(size_t size);
uint8_t* allocate_shared(size_t size);
void free_memory(uint8_t* ptr);
void copy_h2d(uint8_t* dst, const uint8_t* src, size_t size);
void copy_d2h(uint8_t* dst, const uint8_t* src, size_t size);

void f32_gemv(
    const float* x,
    const float* a,
    float* y,
    size_t m,
    size_t k
);

void rms_norm(
    const float* x,
    const float* weight,
    float* out,
    size_t len,
    float eps,
    float offset
);

void silu(
    const float* input,
    float* out,
    size_t n
);

void gelu_tanh(
    const float* input,
    float* out,
    size_t n
);

void rope_at_pos_batched_qk(
    float* q,
    float* k,
    size_t head_dim,
    float rope_base,
    size_t pos,
    size_t rotary_dim,
    size_t num_q,
    size_t num_kv
);

void q4_vecmat(
    const float* activation,
    const uint8_t* q4,
    float* out,
    size_t n,
    size_t k
);

void q4k_matvec_8sg(
    const uint8_t* w4k,
    const float* x,
    float* out,
    size_t n,
    size_t k
);

void q6k_matvec(
    const uint8_t* w6k,
    const float* x,
    float* out,
    size_t n,
    size_t k
);

void q4k_ffn_gate_up(
    const uint8_t* wg,
    const uint8_t* wu,
    const float* x,
    float* g_out,
    float* u_out,
    size_t n,
    size_t k
);

void q4_matvec_v4(
    const uint8_t* q4,
    const int8_t* q8,
    const float* q8s,
    float* out,
    size_t n,
    size_t k
);

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
);

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
);

void q4k_proj(
    const uint8_t* w4k,
    const float* x,
    float* out,
    size_t n,
    size_t k
);
