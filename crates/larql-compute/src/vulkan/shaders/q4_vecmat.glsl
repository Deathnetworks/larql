#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) buffer Q4 { uint data[]; } q4;
layout(set = 0, binding = 1) buffer X { float data[]; } x;
layout(set = 0, binding = 2) buffer Out { float data[]; } out_buf;

layout(push_constant) block PushConstants {
    uint n;
    uint k;
} pc;

float decode_f16(uint bits) {
    uint s = (bits >> 15) & 0x1u;
    uint e = (bits >> 10) & 0x1Fu;
    uint m = bits & 0x3FFu;

    if (e == 0) {
        if (m == 0) return (s != 0) ? -0.0 : 0.0;
        return (s != 0 ? -1.0 : 1.0) * pow(2.0, -14.0) * (float(m) / 1024.0);
    } else if (e == 31) {
        return (m == 0) ? ((s != 0) ? -1.0/0.0 : 1.0/0.0) : 0.0/0.0;
    }
    return (s != 0 ? -1.0 : 1.0) * pow(2.0, float(e) - 15.0) * (1.0 + float(m) / 1024.0);
}

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint k = pc.k;
    
    uint blocks = k / 32;
    uint uints_per_row = blocks * 5; // 18 bytes = 4.5 uints, padded to 5 or similar.
    // GGUF Q4_0: 2 bytes scale (f16) + 16 bytes nibbles = 18 bytes.
    // 18 bytes is not uint-aligned. We need to handle byte offsets.
    
    // For simplicity in this first draft, let's assume K is a multiple of 32.
    // 18 bytes per 32 elements.
    
    float acc = 0.0;
    for (uint b = tid; b < blocks; b += 32) {
        uint byte_off = row * (blocks * 18) + b * 18;
        
        // Load scale (first 2 bytes)
        uint s_idx = byte_off / 4;
        uint s_shift = (byte_off % 4) * 8;
        uint s_bits;
        if (s_shift <= 16) {
            s_bits = (q4.data[s_idx] >> s_shift) & 0xFFFFu;
        } else {
            s_bits = (q4.data[s_idx] >> 24) | ((q4.data[s_idx+1] & 0xFFu) << 8);
        }
        float scale = decode_f16(s_bits);
        
        // Load nibbles (16 bytes)
        uint qs_off = byte_off + 2;
        for (uint i = 0; i < 8; i++) {
            uint q_idx = (qs_off + i * 2) / 4;
            uint q_shift = ((qs_off + i * 2) % 4) * 8;
            uint pair;
            if (q_shift <= 16) {
                pair = (q4.data[q_idx] >> q_shift) & 0xFFFFu;
            } else {
                pair = (q4.data[q_idx] >> 24) | ((q4.data[q_idx+1] & 0xFFu) << 8);
            }
            
            // Extract 4 nibbles from 2 bytes
            uint b0 = pair & 0xFFu;
            uint b1 = (pair >> 8) & 0xFFu;
            
            float n0 = float(int(b0 & 0xFu) - 8);
            float n1 = float(int(b0 >> 4) - 8);
            float n2 = float(int(b1 & 0xFu) - 8);
            float n3 = float(int(b1 >> 4) - 8);
            
            acc += n0 * scale * x.data[b * 32 + i * 4 + 0];
            acc += n1 * scale * x.data[b * 32 + i * 4 + 1];
            acc += n2 * scale * x.data[b * 32 + i * 4 + 2];
            acc += n3 * scale * x.data[b * 32 + i * 4 + 3];
        }
    }
    
    // Local reduction (subgroup sum would be better but keeping it portable for now)
    // Actually GLSL has subgroup support in modern Vulkan.
}
