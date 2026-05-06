#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Norms { float norms[]; };
layout(set = 0, binding = 1) buffer Packed { uint packed_data[]; };
layout(set = 0, binding = 2) buffer Output { float output_data[]; };

layout(push_constant) uniform PushConstants {
    uint d;
    uint batch;
} pcs;

shared float shared_data[256];

const float tq4_centroids[16] = float[16](
    -0.1089, -0.0782, -0.0588, -0.0427,
    -0.0283, -0.0148, -0.0050,  0.0050,
     0.0148,  0.0283,  0.0427,  0.0588,
     0.0782,  0.1089,  0.1500,  0.2000
);

bool tq_sign_flip(uint i) {
    return (((i * 2654435761u) >> 16) & 1u) != 0u;
}

void main() {
    uint elem = gl_LocalInvocationID.x;
    uint vec_idx = gl_WorkGroupID.x;
    uint d = pcs.d;
    if (vec_idx >= pcs.batch || elem >= d) return;

    // Unpack
    uint pack_offset = vec_idx * (d / 2) + elem / 2;
    // Assuming 8-bit values packed in uint32 for Vulkan storage buffer
    uint word = packed_data[pack_offset / 4];
    uint byte_val = (word >> ((pack_offset % 4) * 8)) & 0xFFu;
    uint idx = (elem % 2 == 0) ? (byte_val & 0x0Fu) : ((byte_val >> 4) & 0x0Fu);

    shared_data[elem] = tq4_centroids[idx];
    barrier();

    if (tq_sign_flip(elem)) shared_data[elem] = -shared_data[elem];
    barrier();

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

    output_data[vec_idx * d + elem] = shared_data[elem] * norms[vec_idx];
}
