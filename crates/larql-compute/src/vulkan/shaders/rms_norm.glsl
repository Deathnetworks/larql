#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer X { float data[]; } x;
layout(set = 0, binding = 1) buffer W { float data[]; } w;
layout(set = 0, binding = 2) buffer Out { float data[]; } out_buf;

layout(push_constant) block PushConstants {
    uint head_dim;
    float eps;
} pc;

shared float ss[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint head = gl_WorkGroupID.x;
    uint head_dim = pc.head_dim;
    float eps = pc.eps;

    float partial = 0.0;
    for (uint i = tid; i < head_dim; i += 256) {
        float val = x.data[head * head_dim + i];
        partial += val * val;
    }
    ss[tid] = partial;
    barrier();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            ss[tid] += ss[tid + s];
        }
        barrier();
    }

    float inv_rms = 1.0 / sqrt(ss[0] / float(head_dim) + eps);

    for (uint i = tid; i < head_dim; i += 256) {
        uint idx = head * head_dim + i;
        out_buf.data[idx] = x.data[idx] * inv_rms * w.data[i];
    }
}
