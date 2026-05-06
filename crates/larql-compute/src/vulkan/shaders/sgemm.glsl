#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer A_buf { float A[]; };
layout(set = 0, binding = 1) buffer B_buf { float B[]; };
layout(set = 0, binding = 2) buffer C_buf { float C[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint N;
    uint K;
} pc;

shared float As[32][32];
shared float Bs[32][32];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint tidx = gl_LocalInvocationID.x;
    uint tidy = gl_LocalInvocationID.y;

    float acc = 0.0;
    uint tiles = (pc.K + 31) / 32;

    for (uint t = 0; t < tiles; t++) {
        uint ac = t * 32 + tidx;
        uint br = t * 32 + tidy;

        As[tidy][tidx] = (row < pc.M && ac < pc.K) ? A[row * pc.K + ac] : 0.0;
        Bs[tidy][tidx] = (br < pc.K && col < pc.N) ? B[br * pc.N + col] : 0.0;

        barrier();
        for (uint i = 0; i < 32; i++) {
            acc += As[tidy][i] * Bs[i][tidx];
        }
        barrier();
    }

    if (row < pc.M && col < pc.N) {
        C[row * pc.N + col] = acc;
    }
}
