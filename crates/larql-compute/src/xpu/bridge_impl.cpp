#include "kernels.hpp"
#include <vector>

// External DLL functions
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
    void dll_rope_at_pos_batched_qk(float* q, float* k, size_t head_dim, float rope_base, size_t pos, size_t rotary_dim, size_t num_q, size_t num_kv);
    void dll_q4_vecmat(const uint8_t* q4, const float* x, float* out, size_t m, size_t k);
    void dll_q4k_matvec_8sg(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);
    void dll_q6k_matvec(const uint8_t* w6k, const float* x, float* out, size_t n, size_t k);
    void dll_q4k_ffn_gate_up(const uint8_t* wg, const uint8_t* wu, const float* x, float* g_out, float* u_out, size_t n, size_t k);
    void dll_q4_matvec_v4(const uint8_t* q4, const int8_t* q8, const float* q8s, float* out, size_t n, size_t k);
    void dll_attn_fused(const float* q_in, const float* k_in, const float* v_in, float* k_cache, float* v_cache, float* out, const float* q_weight, const float* k_weight, uint32_t t, uint32_t head_dim, uint32_t num_q, uint32_t num_kv, float scale, uint32_t window_size, float eps, float qk_offset, float rope_base, uint32_t rotary_dim);
    void dll_q4k_qkv_proj(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x, float* q_out, float* k_out, float* v_out, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t k);
    void dll_q4k_proj(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k);
    void dll_quantize_q8(const float* input, int8_t* q8_out, float* scales, uint32_t k);
    void dll_turboquant_encode(const float* input, float* norms, uint8_t* packed, uint32_t d, uint32_t batch);
    void dll_turboquant_decode(const float* norms, const uint8_t* packed, float* output, uint32_t d, uint32_t batch);
    void dll_sgemm(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k);
    void dll_sgemm_transb(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k);
    void dll_check_sycl();
}

bool init_xpu() { return dll_init_xpu(); }

rust::String get_device_info() {
    char buf[256];
    dll_get_device_info(buf, 256);
    return rust::String(buf);
}

uint8_t* allocate_device(size_t size) { return dll_allocate_device(size); }
uint8_t* allocate_shared(size_t size) { return dll_allocate_shared(size); }
void free_memory(uint8_t* ptr) { dll_free_memory(ptr); }
void copy_h2d(uint8_t* dst, const uint8_t* src, size_t size) { dll_copy_h2d(dst, src, size); }
void copy_d2h(uint8_t* dst, const uint8_t* src, size_t size) { dll_copy_d2h(dst, src, size); }

void f32_gemv(const float* x, const float* a, float* y, size_t m, size_t k) { dll_f32_gemv(x, a, y, m, k); }
void rms_norm(const float* x, const float* weight, float* out, size_t len, float eps, float offset) { dll_rms_norm(x, weight, out, len, eps, offset); }
void silu(const float* input, float* out, size_t n) { dll_silu(input, out, n); }
void gelu_tanh(const float* input, float* out, size_t n) { dll_gelu_tanh(input, out, n); }
void rope_at_pos_batched_qk(float* q, float* k, size_t head_dim, float rope_base, size_t pos, size_t rotary_dim, size_t num_q, size_t num_kv) {
    dll_rope_at_pos_batched_qk(q, k, head_dim, rope_base, pos, rotary_dim, num_q, num_kv);
}
void q4_vecmat(const uint8_t* q4, const float* x, float* out, size_t m, size_t k) { dll_q4_vecmat(q4, x, out, m, k); }
void q4k_matvec_8sg(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) { dll_q4k_matvec_8sg(w4k, x, out, n, k); }
void q6k_matvec(const uint8_t* w6k, const float* x, float* out, size_t n, size_t k) { dll_q6k_matvec(w6k, x, out, n, k); }
void q4k_ffn_gate_up(const uint8_t* wg, const uint8_t* wu, const float* x, float* g_out, float* u_out, size_t n, size_t k) {
    dll_q4k_ffn_gate_up(wg, wu, x, g_out, u_out, n, k);
}
void q4_matvec_v4(const uint8_t* q4, const int8_t* q8, const float* q8s, float* out, size_t n, size_t k) {
    dll_q4_matvec_v4(q4, q8, q8s, out, n, k);
}
void attn_fused(const float* q_in, const float* k_in, const float* v_in, float* k_cache, float* v_cache, float* out, const float* q_weight, const float* k_weight, uint32_t t, uint32_t head_dim, uint32_t num_q, uint32_t num_kv, float scale, uint32_t window_size, float eps, float qk_offset, float rope_base, uint32_t rotary_dim) {
    dll_attn_fused(q_in, k_in, v_in, k_cache, v_cache, out, q_weight, k_weight, t, head_dim, num_q, num_kv, scale, window_size, eps, qk_offset, rope_base, rotary_dim);
}
void q4k_qkv_proj(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x, float* q_out, float* k_out, float* v_out, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t k) {
    dll_q4k_qkv_proj(wq, wk, wv, x, q_out, k_out, v_out, q_rows, k_rows, v_rows, k);
}
void q4k_proj(const uint8_t* w4k, const float* x, float* out, size_t n, size_t k) { dll_q4k_proj(w4k, x, out, n, k); }
void quantize_q8(const float* input, int8_t* q8_out, float* scales, uint32_t k) { dll_quantize_q8(input, q8_out, scales, k); }
void turboquant_encode(const float* input, float* norms, uint8_t* packed, uint32_t d, uint32_t batch) { dll_turboquant_encode(input, norms, packed, d, batch); }
void turboquant_decode(const float* norms, const uint8_t* packed, float* output, uint32_t d, uint32_t batch) { dll_turboquant_decode(norms, packed, output, d, batch); }
void sgemm(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k) { dll_sgemm(a, b, c, m, n, k); }
void sgemm_transb(const float* a, const float* b, float* c, uint32_t m, uint32_t n, uint32_t k) { dll_sgemm_transb(a, b, c, m, n, k); }
void check_sycl() { dll_check_sycl(); }
