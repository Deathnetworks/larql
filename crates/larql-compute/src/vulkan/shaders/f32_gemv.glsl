#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer W { float data[]; } w;
layout(set = 0, binding = 1) buffer X { float data[]; } x;
layout(set = 0, binding = 2) buffer Out { float data[]; } out_buf;

layout(push_constant) block PushConstants {
    uint n;
    uint k;
} pc;

shared float ss[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint k = pc.k;

    float acc = 0.0;
    for (uint i = tid; i < k; i += 256) {
        acc += w.data[row * k + i] * x.data[i];
    }
    ss[tid] = acc;
    barrier();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            ss[tid] += ss[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        out_buf.data[row] = ss[0];
    }
}
