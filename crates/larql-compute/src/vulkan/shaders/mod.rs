//! Vulkan shader registry — GLSL sources compiled via vulkano-shaders.

pub mod rms_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/rms_norm.comp", vulkan_version: "1.2"
    }
}
pub mod silu {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/activation.comp", vulkan_version: "1.2"
    }
}
pub mod rope {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/rope.comp", vulkan_version: "1.2"
    }
}
pub mod q4_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4_matvec.comp", vulkan_version: "1.2"
    }
}
pub mod q4_f32_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4_f32_matvec.comp", vulkan_version: "1.2"
    }
}
pub mod q4_vecmat {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4_vecmat.comp", vulkan_version: "1.2"
    }
}
pub mod q4k_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_matvec.comp", vulkan_version: "1.2"
    }
}
pub mod f32_gemv {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/f32_gemv.comp", vulkan_version: "1.2"
    }
}
pub mod q4k_ffn_gate_up {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_ffn_gate_up.comp", vulkan_version: "1.2"
    }
}
pub mod q4k_qkv_proj {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_qkv_proj.comp", vulkan_version: "1.2"
    }
}
pub mod q4k_q6k_qkv_proj {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_q6k_qkv_proj.comp", vulkan_version: "1.2"
    }
}
pub mod q6k_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q6k_matvec.comp", vulkan_version: "1.2"
    }
}
pub mod attn_fused {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/attn_fused.comp", vulkan_version: "1.2"
    }
}
pub mod quantize_q8 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/quantize_q8.comp", vulkan_version: "1.2"
    }
}
pub mod turboquant_encode {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/turboquant_encode.glsl", vulkan_version: "1.2"
    }
}
pub mod turboquant_decode {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/turboquant_decode.glsl", vulkan_version: "1.2"
    }
}
pub mod sgemm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/sgemm.comp", vulkan_version: "1.2"
    }
}
pub mod sgemm_transb {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/sgemm_transb.comp", vulkan_version: "1.2"
    }
}
pub mod layer_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/layer_norm.comp", vulkan_version: "1.2"
    }
}
pub mod residual_ops {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/residual_ops.comp", vulkan_version: "1.2"
    }
}
pub mod v_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/v_norm.glsl", vulkan_version: "1.2"
    }
}
pub mod qk_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/qk_norm.glsl", vulkan_version: "1.2"
    }
}
pub mod q4_sparse_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4_sparse_matvec.comp", vulkan_version: "1.2"
    }
}
pub mod q4k_matvec_stride32 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_matvec_stride32.comp", vulkan_version: "1.2"
    }
}
pub mod q8_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q8_matvec.comp", vulkan_version: "1.2"
    }
}
// ── Previously missing shader modules (shaders exist, weren't registered) ──
pub mod graph_walk_knn {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/graph_walk_knn.comp", vulkan_version: "1.2"
    }
}
pub mod qk_norm_rope_fused {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/qk_norm_rope_fused.comp", vulkan_version: "1.2"
    }
}
pub mod post_attn_residual_norm_store {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/post_attn_residual_norm_store.comp", vulkan_version: "1.2"
    }
}
pub mod post_ffn_norm_residual_add {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/post_ffn_norm_residual_add.comp", vulkan_version: "1.2"
    }
}
pub mod geglu {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/geglu.comp", vulkan_version: "1.2"
    }
}
