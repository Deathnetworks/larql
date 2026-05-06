#include <sycl/sycl.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static std::unique_ptr<sycl::queue> g_queue = nullptr;

extern "C" {

EXPORT bool sycl_init() {
    if (g_queue) return true;
    try {
        g_queue = std::make_unique<sycl::queue>(sycl::default_selector_v);
        return true;
    } catch (const sycl::exception& e) {
        return false;
    }
}

EXPORT void sycl_get_device_name(char* name, int max_len) {
    if (!sycl_init()) {
        strncpy(name, "None", max_len);
        return;
    }
    auto d_name = g_queue->get_device().get_info<sycl::info::device::name>();
    strncpy(name, d_name.c_str(), max_len);
}

EXPORT void* sycl_malloc_device(size_t size) {
    if (!sycl_init()) return nullptr;
    return sycl::malloc_device(size, *g_queue);
}

EXPORT void* sycl_malloc_shared(size_t size) {
    if (!sycl_init()) return nullptr;
    return sycl::malloc_shared(size, *g_queue);
}

EXPORT void sycl_free(void* ptr) {
    if (g_queue && ptr) sycl::free(ptr, *g_queue);
}

EXPORT void sycl_memcpy(void* dst, const void* src, size_t size) {
    if (g_queue) g_queue->memcpy(dst, src, size).wait();
}

// Kernels (Simplified Functor Dispatch)

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

EXPORT void sycl_f32_gemv(const float* x, const float* a, float* y, size_t m, size_t k) {
    if (!sycl_init()) return;
    g_queue->parallel_for(sycl::range<1>(m), F32GemvKernel{x, a, y, (uint32_t)k}).wait();
}

// ... other kernels will be added here
}
