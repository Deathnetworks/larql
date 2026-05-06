#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Q { float data[]; } q;
layout(set = 0, binding = 1) buffer K { float data[]; } k;

layout(push_constant) block PushConstants {
    uint pos;
    uint head_dim;
    uint num_q;
    uint num_kv;
    float rope_base;
    uint rotary_dim;
} pc;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint head = tid / (pc.head_dim / 2);
    uint d = tid % (pc.head_dim / 2);
    
    if (head >= (pc.num_q + pc.num_kv)) return;
    if (d * 2 >= pc.rotary_dim) return;

    float freq = 1.0 / pow(pc.rope_base, float(2 * d) / float(pc.rotary_dim));
    float angle = float(pc.pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    bool is_q = (head < pc.num_q);
    uint h_idx = is_q ? head : (head - pc.num_q);
    uint base = h_idx * pc.head_dim + d;
    uint half = pc.rotary_dim / 2;

    if (is_q) {
        float r = q.data[base];
        float i = q.data[base + half];
        q.data[base] = r * cos_a - i * sin_a;
        q.data[base + half] = r * sin_a + i * cos_a;
    } else {
        float r = k.data[base];
        float i = k.data[base + half];
        k.data[base] = r * cos_a - i * sin_a;
        k.data[base + half] = r * sin_a + i * cos_a;
    }
}
