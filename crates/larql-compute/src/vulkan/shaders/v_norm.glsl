#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer InputX { float X[]; };
layout(set = 0, binding = 1) writeonly buffer Output { float Out[]; };

layout(push_constant) uniform PushConstants {
    uint head_dim;
    uint num_heads;
    float eps;
    uint mode; // 0: single, 1: batched
} pcs;

shared float tg_red;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint sg_id = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint tg_id = gl_WorkGroupID.x;

    if (pcs.mode == 0) {
        // Single vector RMSNorm
        float partial = 0.0f;
        for (uint i = tid; i < pcs.head_dim; i += 256) {
            float v = X[i];
            partial += v * v;
        }
        float sg_sum = subgroupSum(partial);
        if (lane == 0) {
            if (sg_id == 0) tg_red = 0.0;
        }
        barrier();
        if (lane == 0) atomicAdd(tg_red, sg_sum);
        barrier();
        float rms = 1.0f / sqrt(tg_red / float(pcs.head_dim) + pcs.eps);
        for (uint i = tid; i < pcs.head_dim; i += 256) {
            Out[i] = X[i] * rms;
        }
    } else {
        // Batched per-head RMSNorm
        if (tg_id >= pcs.num_heads) return;
        uint base = tg_id * pcs.head_dim;
        float partial = 0.0f;
        for (uint i = tid; i < pcs.head_dim; i += 256) {
            float v = X[base + i];
            partial += v * v;
        }
        float sg_sum = subgroupSum(partial);
        if (lane == 0) {
            if (sg_id == 0) tg_red = 0.0;
        }
        barrier();
        if (lane == 0) atomicAdd(tg_red, sg_sum);
        barrier();
        float rms = 1.0f / sqrt(tg_red / float(pcs.head_dim) + pcs.eps);
        for (uint i = tid; i < pcs.head_dim; i += 256) {
            Out[base + i] = X[base + i] * rms;
        }
    }
}
