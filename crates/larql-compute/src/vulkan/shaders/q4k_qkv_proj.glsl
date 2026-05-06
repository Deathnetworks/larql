#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) buffer WQ { uint data[]; } wq;
layout(set = 0, binding = 1) buffer WK { uint data[]; } wk;
layout(set = 0, binding = 2) buffer WV { uint data[]; } wv;
layout(set = 0, binding = 3) buffer X { float data[]; } x;
layout(set = 0, binding = 4) buffer Q_Out { float data[]; } q_out;
layout(set = 0, binding = 5) buffer K_Out { float data[]; } k_out;
layout(set = 0, binding = 6) buffer V_Out { float data[]; } v_out;

layout(push_constant) block PushConstants {
    uint q_rows;
    uint k_rows;
    uint v_rows;
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
    uint tg_id = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;
    uint k = pc.k;
    uint superblocks = k / 256;
    uint uints_per_row = superblocks * 36;

    uint total_rows = pc.q_rows + pc.k_rows + pc.v_rows;
    uint global_row = tg_id;
    if (global_row >= total_rows) return;

    // Determine which buffer and local row
    uint local_row;
    uint buffer_type; // 0=Q, 1=K, 2=V
    if (global_row < pc.q_rows) {
        local_row = global_row; buffer_type = 0;
    } else if (global_row < pc.q_rows + pc.k_rows) {
        local_row = global_row - pc.q_rows; buffer_type = 1;
    } else {
        local_row = global_row - pc.q_rows - pc.k_rows; buffer_type = 2;
    }

    uint ix = lane & 1u;
    uint tid = lane >> 1u;
    uint j = tid >> 1u;
    uint sh = tid & 1u;
    bool hi = (j & 1u) != 0u;
    uint group = j >> 1u;

    float acc = 0.0;
    for (uint sb = ix; sb < superblocks; sb += 2u) {
        uint sb_off = local_row * uints_per_row + sb * 36;
        
        uint d_bits;
        if (buffer_type == 0) d_bits = wq.data[sb_off];
        else if (buffer_type == 1) d_bits = wk.data[sb_off];
        else d_bits = wv.data[sb_off];

        float d = decode_f16(d_bits & 0xFFFFu);
        float dmin = decode_f16(d_bits >> 16);
        
        uint sc, mn;
        uint s_idx_1, s_idx_2, s_idx_3;
        if (buffer_type == 0) { s_idx_1 = wq.data[sb_off+1]; s_idx_2 = wq.data[sb_off+2]; s_idx_3 = wq.data[sb_off+3]; }
        else if (buffer_type == 1) { s_idx_1 = wk.data[sb_off+1]; s_idx_2 = wk.data[sb_off+2]; s_idx_3 = wk.data[sb_off+3]; }
        else { s_idx_1 = wv.data[sb_off+1]; s_idx_2 = wv.data[sb_off+2]; s_idx_3 = wv.data[sb_off+3]; }

        if (j < 4) {
            sc = (s_idx_1 >> (j * 8)) & 0x3Fu;
            mn = (s_idx_2 >> (j * 8)) & 0x3Fu;
        } else {
            uint j_idx = j - 4;
            uint byte_sc = (s_idx_3 >> (j_idx * 8)) & 0x0Fu;
            uint byte_sc_hi = (s_idx_1 >> (j_idx * 8 + 6)) & 0x03u;
            sc = byte_sc | (byte_sc_hi << 4);
            uint byte_mn = (s_idx_3 >> (j_idx * 8 + 4)) & 0x0Fu;
            uint byte_mn_hi = (s_idx_2 >> (j_idx * 8 + 6)) & 0x03u;
            mn = byte_mn | (byte_mn_hi << 4);
        }
        
        float scale = d * float(sc);
        float mmin = dmin * float(mn);
        
        uint x_base = sb * 256 + j * 32 + sh * 16;
        float dot_acc = 0.0, sum_acc = 0.0;
        
        uint qs_idx = sb_off + 4 + group * 8 + sh * 4;
        for (uint l = 0; l < 4; l++) {
            uint val;
            if (buffer_type == 0) val = wq.data[qs_idx + l];
            else if (buffer_type == 1) val = wk.data[qs_idx + l];
            else val = wv.data[qs_idx + l];

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
        if (lane < s) ss[lane] += ss[lane + s];
        barrier();
    }
    if (lane == 0) {
        if (buffer_type == 0) q_out.data[local_row] = ss[0];
        else if (buffer_type == 1) k_out.data[local_row] = ss[0];
        else v_out.data[local_row] = ss[0];
    }
}
