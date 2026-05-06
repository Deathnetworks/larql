#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer X { float data[]; } x;
layout(set = 0, binding = 1) buffer Out { float data[]; } out_buf;
layout(set = 0, binding = 2) readonly buffer W { float data[]; } w;

layout(push_constant) uniform PushConstants {
    uint head_dim;
    uint num_heads;
    float eps;
    float offset;
} pc;

shared float ss[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint head = gl_WorkGroupID.x;
    if (head >= pc.num_heads) return;
    
    uint base = head * pc.head_dim;

    float partial = 0.0;
    for (uint i = tid; i < pc.head_dim; i += 256) {
        float val = x.data[base + i];
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

    float rms = sqrt(ss[0] / float(pc.head_dim) + pc.eps);

    for (uint i = tid; i < pc.head_dim; i += 256) {
        out_buf.data[base + i] = (x.data[base + i] / rms) * (pc.offset + w.data[i]);
    }
}
