#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) buffer W { uint data[]; } w;
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
    if (e == 0) return (s != 0 ? -1.0 : 1.0) * pow(2.0, -14.0) * (float(m) / 1024.0);
    return (s != 0 ? -1.0 : 1.0) * pow(2.0, float(e) - 15.0) * (1.0 + float(m) / 1024.0);
}

shared float ss[32];

void main() {
    uint global_row = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;
    uint k = pc.k;
    uint superblocks = k / 256;
    
    uint n = pc.n;
    uint row_idx = global_row % n;
    bool is_up = (global_row >= n);
    
    // Each row of GATE/UP is superblocks * 144 bytes = 36 uints.
    // UP follows GATE in the weight buffer.
    uint uints_per_row = superblocks * 36;
    uint row_start_uint = is_up ? (n * uints_per_row + row_idx * uints_per_row) : (row_idx * uints_per_row);

    uint ix = lane & 1u;
    uint tid = lane >> 1u;
    uint j = tid >> 1u;
    uint sh = tid & 1u;
    bool hi = (j & 1u) != 0u;
    uint group = j >> 1u;

    float acc = 0.0;
    for (uint sb = ix; sb < superblocks; sb += 2u) {
        uint sb_off = row_start_uint + sb * 36;
        
        uint d_bits = w.data[sb_off];
        float d = decode_f16(d_bits & 0xFFFFu);
        float dmin = decode_f16(d_bits >> 16);
        
        uint sc, mn;
        if (j < 4) {
            uint byte = (w.data[sb_off + 1] >> (j * 8)) & 0xFFu;
            sc = byte & 0x3Fu;
            uint byte_mn = (w.data[sb_off + 2] >> (j * 8)) & 0xFFu;
            mn = byte_mn & 0x3Fu;
        } else {
            uint j_idx = j - 4;
            uint byte_sc = (w.data[sb_off + 3] >> (j_idx * 8)) & 0x0Fu;
            uint byte_sc_hi = (w.data[sb_off + 1] >> (j_idx * 8 + 6)) & 0x03u;
            sc = byte_sc | (byte_sc_hi << 4);
            
            uint byte_mn = (w.data[sb_off + 3] >> (j_idx * 8 + 4)) & 0x0Fu;
            uint byte_mn_hi = (w.data[sb_off + 2] >> (j_idx * 8 + 6)) & 0x03u;
            mn = byte_mn | (byte_mn_hi << 4);
        }
        
        float scale = d * float(sc);
        float mmin = dmin * float(mn);
        
        uint x_base = sb * 256 + j * 32 + sh * 16;
        float dot_acc = 0.0, sum_acc = 0.0;
        
        uint qs_idx = sb_off + 4 + group * 8 + sh * 4;
        for (uint l = 0; l < 4; l++) {
            uint val = w.data[qs_idx + l];
            for (uint b = 0; b < 4; b++) {
                uint byte = (val >> (b * 8)) & 0xFFu;
                float nib = hi ? float((byte >> 4) & 0xFu) : float(byte & 0xFu);
                float xv = x.data[x_base + l * 4 + b];
                dot_acc += nib * xv;
                sum_acc += xv;
            }
        }
        acc += scale * dot_acc - mmin * sum_acc;
    }
    
    ss[lane] = acc;
    barrier();

    for (uint s = 16; s > 0; s >>= 1) {
        if (lane < s) {
            ss[lane] += ss[lane + s];
        }
        barrier();
    }

    if (lane == 0) {
        out_buf.data[global_row] = ss[0];
    }
}
