#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Input { float input_data[]; };
layout(set = 0, binding = 1) buffer Norms { float norms[]; };
layout(set = 0, binding = 2) buffer Packed { uint packed_data[]; };

layout(push_constant) uniform PushConstants {
    uint d;
    uint batch;
} pcs;

shared float shared_data[256];

const float tq4_boundaries[15] = float[15](
    -0.0936, -0.0685, -0.0508, -0.0355,
    -0.0216, -0.0099,  0.0000,  0.0099,
     0.0216,  0.0355,  0.0508,  0.0685,
     0.0936,  0.1295,  0.1750
);

bool tq_sign_flip(uint i) {
    return ((i * 2654435761u) >> 16) & 1u != 0;
}

void main() {
    uint elem = gl_LocalInvocationID.x;
    uint vec_idx = gl_WorkGroupID.x;
    uint d = pcs.d;
    if (vec_idx >= pcs.batch || elem >= d) return;

    uint base = vec_idx * d;
    shared_data[elem] = input_data[base + elem];
    barrier();

    // L2 Norm
    if (elem == 0) {
        float sum_sq = 0.0;
        for (uint i = 0; i < d; i++) {
            sum_sq += shared_data[i] * shared_data[i];
        }
        norms[vec_idx] = sqrt(sum_sq);
    }
    barrier();

    float norm = norms[vec_idx];
    float inv_norm = (norm > 1e-12) ? (1.0 / norm) : 0.0;
    shared_data[elem] *= inv_norm;
    barrier();

    // Sign flip
    if (tq_sign_flip(elem)) shared_data[elem] = -shared_data[elem];
    barrier();

    // WHT
    for (uint hstep = 1; hstep < d; hstep *= 2) {
        uint blk = hstep * 2;
        uint blk_idx = elem / blk;
        uint within = elem % blk;
        if (within < hstep) {
            uint j = blk_idx * blk + within;
            float a = shared_data[j];
            float b = shared_data[j + hstep];
            shared_data[j] = a + b;
            shared_data[j + hstep] = a - b;
        }
        barrier();
    }

    shared_data[elem] *= 1.0 / sqrt(float(d));

    if (tq_sign_flip(elem)) shared_data[elem] = -shared_data[elem];
    barrier();

    // Quantize
    float y = shared_data[elem];
    uint idx = 0;
    for (uint b = 0; b < 15; b++) {
        if (y > tq4_boundaries[b]) idx = b + 1;
    }

    // Pack
    uint pack_offset = vec_idx * (d / 2) + elem / 2;
    // Vulkan packing is a bit different since we work with uint32 often.
    // For simplicity, we'll assume the caller handles the 8-bit packing or we use atomicOr.
    // Here we'll use a local shared buffer to pack and then write.
    // Actually, let's just use the same logic as Metal but be careful with alignment.
}
