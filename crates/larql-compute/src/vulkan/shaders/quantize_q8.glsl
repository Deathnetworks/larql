#version 450

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) buffer Input { float input_data[]; };
layout(set = 0, binding = 1) buffer Q8Out { int q8_out[]; };
layout(set = 0, binding = 2) buffer Scales { float scales[]; };

layout(push_constant) uniform PushConstants {
    uint k;
} pcs;

void main() {
    uint block = gl_GlobalInvocationID.x;
    uint num_blocks = pcs.k / 32;
    if (block >= num_blocks) return;

    uint off = block * 32;
    float amax = 0.0;
    for (uint j = 0; j < 32; j++) {
        float v = abs(input_data[off + j]);
        if (v > amax) amax = v;
    }

    float scale = amax / 127.0;
    float inv = (scale > 0.0) ? (1.0 / scale) : 0.0;
    scales[block] = scale;

    for (uint j = 0; j < 32; j++) {
        float v = input_data[off + j] * inv;
        v = clamp(v, -128.0, 127.0);
        // GLSL doesn't have int8_t directly in all versions, 
        // using int and the caller handles the storage.
        q8_out[off + j] = int(round(v));
    }
}
