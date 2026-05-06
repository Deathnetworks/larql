#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer X { float data[]; } x;
layout(set = 0, binding = 1) buffer Out { float data[]; } out_buf;

layout(push_constant) block PushConstants {
    uint n;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;

    float val = x.data[i];
    out_buf.data[i] = val / (1.0 + exp(-val));
}
