#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Q_In { float data[]; } q_in;
layout(set = 0, binding = 1) buffer K_In { float data[]; } k_in;
layout(set = 0, binding = 2) buffer V_In { float data[]; } v_in;
layout(set = 0, binding = 3) buffer K_Cache { float data[]; } k_cache;
layout(set = 0, binding = 4) buffer V_Cache { float data[]; } v_cache;
layout(set = 0, binding = 5) buffer Out { float data[]; } out_buf;
layout(set = 0, binding = 6) buffer Q_Weight { float data[]; } q_weight;
layout(set = 0, binding = 7) buffer K_Weight { float data[]; } k_weight;

layout(push_constant) block PushConstants {
    uint t;
    uint head_dim;
    uint num_q;
    uint num_kv;
    float scale;
    uint window_size;
    float eps;
    float qk_offset;
    float rope_base;
    uint rotary_dim;
} pc;

shared float tg_q[256];
shared float tg_k_normed[256];
shared float tg_red[8];
shared float tg_scores[1024];

void main() {
    uint head = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint tg_sz = gl_WorkGroupSize.x;
    uint lane = tid % 32;
    uint sg_id = tid / 32;
    uint n_sg = (tg_sz + 31) / 32;

    uint kv_head = head / (pc.num_q / pc.num_kv);
    uint pos = pc.t - 1;

    uint rdim = (pc.rotary_dim == 0) ? pc.head_dim : min(pc.rotary_dim, pc.head_dim);
    uint hdim = rdim / 2;

    // Phase 1: Parallel RMS for Q and K
    float partial_q = 0.0;
    float partial_k = 0.0;
    for (uint d = tid; d < pc.head_dim; d += tg_sz) {
        float vq = q_in.data[head * pc.head_dim + d];
        float vk = k_in.data[kv_head * pc.head_dim + d];
        partial_q += vq * vq;
        partial_k += vk * vk;
    }

    // Manual reduction for Q
    tg_q[tid] = partial_q; // Using tg_q as scratch for first reduction
    barrier();
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) tg_q[tid] += tg_q[tid + s];
        barrier();
    }
    float ss_q = tg_q[0];

    // Manual reduction for K
    tg_k_normed[tid] = partial_k; // Using tg_k_normed as scratch
    barrier();
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) tg_k_normed[tid] += tg_k_normed[tid + s];
        barrier();
    }
    float ss_k = tg_k_normed[0];

    float inv_rms_q = 1.0 / sqrt(ss_q / float(pc.head_dim) + pc.eps);
    float inv_rms_k = 1.0 / sqrt(ss_k / float(pc.head_dim) + pc.eps);

    // Phase 2: Write normed Q,K to TG memory
    for (uint d = tid; d < pc.head_dim; d += tg_sz) {
        float vq = q_in.data[head * pc.head_dim + d];
        float vk = k_in.data[kv_head * pc.head_dim + d];
        tg_q[d] = (vq * inv_rms_q) * (pc.qk_offset + q_weight.data[d]);
        tg_k_normed[d] = (vk * inv_rms_k) * (pc.qk_offset + k_weight.data[d]);
    }
    barrier();

    // Phase 3: Shared RoPE
    uint cache_off = pos * pc.num_kv * pc.head_dim + kv_head * pc.head_dim;
    for (uint d = tid; d < hdim; d += tg_sz) {
        float freq = 1.0 / pow(pc.rope_base, float(2 * d) / float(rdim));
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float qr = tg_q[d];
        float qi = tg_q[d + hdim];
        tg_q[d] = qr * cos_a - qi * sin_a;
        tg_q[d + hdim] = qr * sin_a + qi * cos_a;

        float kr = tg_k_normed[d];
        float ki = tg_k_normed[d + hdim];
        k_cache.data[cache_off + d] = kr * cos_a - ki * sin_a;
        k_cache.data[cache_off + d + hdim] = kr * sin_a + ki * cos_a;
    }
    for (uint d = tid + rdim; d < pc.head_dim; d += tg_sz) {
        k_cache.data[cache_off + d] = tg_k_normed[d];
    }

    // Phase 4: Stream V
    for (uint d = tid; d < pc.head_dim; d += tg_sz) {
        v_cache.data[cache_off + d] = v_in.data[kv_head * pc.head_dim + d];
    }
    barrier();

    // Phase 5: Scores
    uint t_start = (pc.window_size > 0 && pc.t > pc.window_size) ? pc.t - pc.window_size : 0;
    float local_max = -1e30;
    for (uint it = t_start + tid; it < pc.t; it += tg_sz) {
        uint k_off = it * pc.num_kv * pc.head_dim + kv_head * pc.head_dim;
        float dot = 0.0;
        for (uint d = 0; d < pc.head_dim; d++) {
            dot += tg_q[d] * k_cache.data[k_off + d];
        }
        dot *= pc.scale;
        tg_scores[it - t_start] = dot;
        local_max = max(local_max, dot);
    }

    // Reduction for max
    tg_red[tid] = local_max;
    barrier();
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) tg_red[tid] = max(tg_red[tid], tg_red[tid + s]);
        barrier();
    }
    float global_max = tg_red[0];

    // Phase 6: Softmax sum
    float local_sum = 0.0;
    for (uint it = t_start + tid; it < pc.t; it += tg_sz) {
        float w = exp(tg_scores[it - t_start] - global_max);
        tg_scores[it - t_start] = w;
        local_sum += w;
    }

    // Reduction for sum
    tg_red[tid] = local_sum;
    barrier();
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) tg_red[tid] += tg_red[tid + s];
        barrier();
    }
    float global_sum = tg_red[0];
    float inv_sum = 1.0 / global_sum;

    for (uint it = t_start + tid; it < pc.t; it += tg_sz) {
        tg_scores[it - t_start] *= inv_sum;
    }
    barrier();

    // Phase 7: V sum
    for (uint d = tid; d < pc.head_dim; d += tg_sz) {
        float acc_v = 0.0;
        for (uint it = t_start; it < pc.t; it++) {
            acc_v += tg_scores[it - t_start] * v_cache.data[it * pc.num_kv * pc.head_dim + kv_head * pc.head_dim + d];
        }
        out_buf.data[head * pc.head_dim + d] = acc_v;
    }
}
