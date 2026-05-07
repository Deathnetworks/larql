//! XPU shader bench and pipeline inventory.

use crate::xpu::XpuBackend;

pub fn run_bench(backend: &XpuBackend) {
    println!("XPU Shader Bench (SYCL)");
    println!("Device: {}", backend.device_info());
    println!();
    
    // Target shapes (Gemma 4 31B approx)
    let hidden = 6144;
    let inter = 16384;
    let x = vec![1.0f32; hidden];
    let w_f32 = vec![1.0f32; hidden * hidden];
    let w_q4 = vec![0u8; hidden * inter / 2];
    let w_q6 = vec![0u8; hidden * inter * 6 / 8];
    
    // 1. F32 GEMV
    let t = std::time::Instant::now();
    for _ in 0..100 {
        let _ = backend.f32_gemv(ndarray::ArrayView2::from_shape((hidden, hidden), &w_f32).unwrap(), &x);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("f32_gemv ({}x{}): {:.3}ms", hidden, hidden, ms);

    // 2. RMSNorm
    let t = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = crate::xpu::ops::rms_norm::dispatch(&x, &x[..hidden], 1e-6, 0.0);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("rms_norm ({}): {:.3}ms", hidden, ms);

    // 3. Q4 MatVec
    let t = std::time::Instant::now();
    for _ in 0..100 {
        let _ = backend.q4_vecmat(&x, &w_q4, inter, hidden);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("q4_matvec ({}x{}): {:.3}ms", inter, hidden, ms);

    // 4. Gated FFN (fused)
    let t = std::time::Instant::now();
    let act = crate::xpu::stages::ffn::Activation::SiLU;
    let fmt = crate::xpu::stages::quant_matvec::QuantFormat::Q4_0;
    for _ in 0..100 {
        let _ = crate::xpu::stages::ffn::encode_gated(&w_q4, &w_q4, &w_q4, &x, fmt, fmt, act, inter, hidden);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("gated_ffn_fused ({}x{}): {:.3}ms", inter, hidden, ms);
    // 5. Q6 MatVec (MoE Down)
    let t = std::time::Instant::now();
    for _ in 0..100 {
        let _ = backend.q6k_matvec(&w_q6, &x, hidden, inter);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("q6k_matvec ({}x{}): {:.3}ms", hidden, inter, ms);

    // 6. Fused FFN (Q4_K, MoE Gate/Up)
    let t = std::time::Instant::now();
    let fmt_q4k = crate::xpu::stages::quant_matvec::QuantFormat::Q4K;
    for _ in 0..100 {
        let _ = crate::xpu::stages::ffn::encode_gated(&w_q4, &w_q4, &w_q4, &x, fmt_q4k, fmt_q4k, act, inter, hidden);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("gated_ffn_q4k_fused ({}x{}): {:.3}ms", inter, hidden, ms);

    // 7. Q4_K MatVec
    let t = std::time::Instant::now();
    for _ in 0..100 {
        let _ = backend.q4k_matvec(&w_q4, &x, hidden, inter);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("q4k_matvec ({}x{}): {:.3}ms", hidden, inter, ms);
}
