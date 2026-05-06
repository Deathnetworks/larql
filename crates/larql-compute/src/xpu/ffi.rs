#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("larql-compute/src/xpu/kernels.hpp");

        fn init_xpu() -> bool;
        fn get_device_info() -> String;

        unsafe fn allocate_device(size: usize) -> *mut u8;
        unsafe fn allocate_shared(size: usize) -> *mut u8;
        unsafe fn free_memory(ptr: *mut u8);
        unsafe fn copy_h2d(dst: *mut u8, src: *const u8, size: usize);
        unsafe fn copy_d2h(dst: *mut u8, src: *const u8, size: usize);

        unsafe fn f32_gemv(
            x: *const f32,
            a: *const f32,
            y: *mut f32,
            m: usize,
            k: usize,
        );

        unsafe fn rms_norm(
            x: *const f32,
            weight: *const f32,
            out: *mut f32,
            len: usize,
            eps: f32,
            offset: f32,
        );

        unsafe fn silu(input: *const f32, out: *mut f32, n: usize);
        unsafe fn gelu_tanh(input: *const f32, out: *mut f32, n: usize);

        unsafe fn rope_at_pos_batched_qk(
            q: *mut f32,
            k: *mut f32,
            head_dim: usize,
            rope_base: f32,
            pos: usize,
            rotary_dim: usize,
            num_q: usize,
            num_kv: usize,
        );

        unsafe fn q4_vecmat(
            activation: *const f32,
            q4: *const u8,
            out: *mut f32,
            n: usize,
            k: usize,
        );

        unsafe fn q4k_matvec_8sg(
            w4k: *const u8,
            x: *const f32,
            out: *mut f32,
            n: usize,
            k: usize,
        );

        unsafe fn q6k_matvec(
            w6k: *const u8,
            x: *const f32,
            out: *mut f32,
            n: usize,
            k: usize,
        );

        unsafe fn q4k_ffn_gate_up(
            wg: *const u8,
            wu: *const u8,
            x: *const f32,
            g_out: *mut f32,
            u_out: *mut f32,
            n: usize,
            k: usize,
        );

        unsafe fn q4_matvec_v4(
            q4: *const u8,
            q8: *const i8,
            q8s: *const f32,
            out: *mut f32,
            n: usize,
            k: usize,
        );

        unsafe fn attn_fused(
            q_in: *const f32,
            k_in: *const f32,
            v_in: *const f32,
            k_cache: *mut f32,
            v_cache: *mut f32,
            out: *mut f32,
            q_weight: *const f32,
            k_weight: *const f32,
            t: u32,
            head_dim: u32,
            num_q: u32,
            num_kv: u32,
            scale: f32,
            window_size: u32,
            eps: f32,
            qk_offset: f32,
            rope_base: f32,
            rotary_dim: u32,
        );

        unsafe fn q4k_qkv_proj(
            wq: *const u8,
            wk: *const u8,
            wv: *const u8,
            x: *const f32,
            q_out: *mut f32,
            k_out: *mut f32,
            v_out: *mut f32,
            q_rows: u32,
            k_rows: u32,
            v_rows: u32,
            k: u32,
        );

        unsafe fn q4k_proj(
            w4k: *const u8,
            x: *const f32,
            out: *mut f32,
            n: usize,
            k: usize,
        );
    }
}
