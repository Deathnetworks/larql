#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { float data[]; } x;
layout(set = 0, binding = 1) readonly buffer W { float data[]; } w;
layout(set = 0, binding = 2) readonly buffer B { float data[]; } b;
layout(set = 0, binding = 3) writeonly buffer Out { float data[]; } out_buf;

layout(push_constant) uniform PushConstants {
    uint len;
    float eps;
    float offset;
} pc;

shared float ss[256];
shared float ss2[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint n = pc.len;

    // Phase 1: Mean
    float partial_sum = 0.0;
    for (uint i = tid; i < n; i += 256) {
        partial_sum += x.data[i];
    }
    ss[tid] = partial_sum;
    barrier();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) ss[tid] += ss[tid + s];
        barrier();
    }
    float mean = ss[0] / float(n);
    barrier();

    // Phase 2: Variance
    float partial_var = 0.0;
    for (uint i = tid; i < n; i += 256) {
        float diff = x.data[i] - mean;
        partial_var += diff * diff;
    }
    ss2[tid] = partial_var;
    barrier();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) ss2[tid] += ss2[tid + s];
        barrier();
    }
    float inv_std = 1.0 / sqrt(ss2[0] / float(n) + pc.eps);

    // Phase 3: Final scaling
    for (uint i = tid; i < n; i += 256) {
        out_buf.data[i] = (x.data[i] - mean) * inv_std * (w.data[i] + pc.offset) + b.data[i];
    }
}
