//! Vulkan shader registry — GLSL sources compiled via vulkano-shaders.

pub mod rms_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/rms_norm.comp"
    }
}
pub mod silu {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/activation.comp"
    }
}
pub mod rope {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/rope.comp"
    }
}
pub mod q4_vecmat {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4_vecmat.comp"
    }
}
pub mod q4k_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_matvec.comp"
    }
}
pub mod f32_gemv {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/f32_gemv.comp"
    }
}
pub mod q4k_ffn_gate_up {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_ffn_gate_up.comp"
    }
}
pub mod q4k_qkv_proj {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_qkv_proj.comp"
    }
}
pub mod q4k_q6k_qkv_proj {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q4k_q6k_qkv_proj.comp"
    }
}
pub mod q6k_matvec {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/q6k_matvec.comp"
    }
}
pub mod attn_fused {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/attn_fused.comp"
    }
}
pub mod quantize_q8 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/quantize_q8.comp"
    }
}
pub mod turboquant_encode {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/turboquant_encode.glsl"
    }
}
pub mod turboquant_decode {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/turboquant_decode.glsl"
    }
}
pub mod sgemm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/sgemm.comp"
    }
}
pub mod sgemm_transb {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/sgemm_transb.comp"
    }
}
pub mod layer_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/layer_norm.comp"
    }
}
pub mod residual_ops {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/residual_ops.comp"
    }
}
pub mod v_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/v_norm.glsl"
    }
}
pub mod qk_norm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/vulkan/shaders/qk_norm.glsl"
    }
}
