#version 450

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

void main() {
    uint row = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;
    uint k = pc.k;
    uint superblocks = k / 256;
    
    // GGUF Q4_K: 144 bytes per superblock.
    // 144 bytes = 36 uints.
    uint uints_per_row = superblocks * 36;
    
    uint ix = lane & 1u;
    uint tid = lane >> 1u;
    uint j = tid >> 1u;
    uint sh = tid & 1u;
    bool hi = (j & 1u) != 0u;
    uint group = j >> 1u;

    float acc = 0.0;
    for (uint sb = ix; sb < superblocks; sb += 2u) {
        uint sb_off = row * uints_per_row + sb * 36;
        
        // d and dmin are at the start (4 bytes each -> 1 uint each)
        float d = decode_f16(w.data[sb_off] & 0xFFFFu);
        float dmin = decode_f16(w.data[sb_off] >> 16);
        
        // Unpack scales and mins (12 bytes -> 3 uints)
        // This is complex due to bit-packing. 
        // For simplicity, let's just do a basic version first.
        uint sb_bytes_off = 1; // start of scales/mins uints
        uint sc, mn;
        if (j < 4) {
            uint val = w.data[sb_off + 1 + (j / 4)]; // scales/mins start at w.data[sb_off + 1]
            // j=0..3: bytes 4,5,6,7 of superblock
            // w.data[sb_off + 1] contains bytes 4,5,6,7
            uint byte = (w.data[sb_off + 1] >> (j * 8)) & 0xFFu;
            sc = byte & 0x3Fu;
            
            uint byte_mn = (w.data[sb_off + 2] >> (j * 8)) & 0xFFu;
            mn = byte_mn & 0x3Fu;
        } else {
            // j=4..7: complex packing
            sc = 0; mn = 0; // Placeholder for now
        }
        
        float scale = d * float(sc);
        float mmin = dmin * float(mn);
        
        uint x_base = sb * 256 + j * 32 + sh * 16;
        float dot_acc = 0.0, sum_acc = 0.0;
        
        // Nibbles start at byte 16 (uint index 4)
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
    
    // Subgroup reduction
    // ...
}
