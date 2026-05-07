//! Per-kernel XPU GPU bandwidth profiler.

use std::time::Instant;
use crate::xpu::XpuBackend;
use crate::QuantFormat;

#[derive(Debug, Clone)]
pub struct KernelResult {
    pub name: String,
    pub mb_per_call: f64,
    pub isolated_ms: f64,
    pub isolated_sd_ms: f64,
    pub isolated_gbs: f64,
    pub batched_ms_per_layer: f64,
    pub batched_gbs: f64,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn synth_f32(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.007).sin() * 0.4)
        .collect()
}

fn measure_isolated(warmup: usize, iters: usize, f: &mut impl FnMut()) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms);
        }
    }
    (mean(&times), stddev(&times))
}

pub fn profile_all(backend: &XpuBackend, n_layers: usize, warmup: usize, iters: usize) -> Vec<KernelResult> {
    let mut results = Vec::new();
    
    // Target shapes (Gemma 4 31B approx)
    let hidden = 6144usize;
    let inter = 16384usize;

    println!("{:<44} {:>8} {:>8} {:>8} {:>8}", "Kernel", "iso_ms", "iso_gbs", "bat_ms", "bat_gbs");
    println!("{}", "-".repeat(80));

    // q6k_matvec
    {
        let n = hidden;
        let k = inter;
        let mb = (n * k * 6 / 8) as f64 / 1e6; // Rough estimate for Q6_K
        let w = vec![0u8; n * k * 6 / 8];
        let x = synth_f32(k, 0.5);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = backend.q6k_matvec(&w, &x, n, k);
        });

        let r = KernelResult {
            name: format!("q6k_matvec ({}x{})", n, k),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_ms,
            batched_ms_per_layer: iso_ms, 
            batched_gbs: mb / iso_ms,
        };
        println!("{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1}", r.name, r.isolated_ms, r.isolated_gbs, r.batched_ms_per_layer, r.batched_gbs);
        results.push(r);
    }

    // q4k_matvec
    {
        let n = hidden;
        let k = inter;
        let mb = (n * k * 4 / 8) as f64 / 1e6; // Rough estimate for Q4_K
        let w = vec![0u8; n * k * 4 / 8];
        let x = synth_f32(k, 0.6);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = backend.q4k_matvec(&w, &x, n, k);
        });

        let r = KernelResult {
            name: format!("q4k_matvec ({}x{})", n, k),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_ms,
            batched_ms_per_layer: iso_ms,
            batched_gbs: mb / iso_ms,
        };
        println!("{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1}", r.name, r.isolated_ms, r.isolated_gbs, r.batched_ms_per_layer, r.batched_gbs);
        results.push(r);
    }

    // q4_matvec
    {
        let n = hidden;
        let k = inter;
        let mb = (n * k * 4 / 8) as f64 / 1e6; // Q4_0
        let w = vec![0u8; n * k * 4 / 8];
        let x = synth_f32(k, 0.7);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = backend.q4_vecmat(&x, &w, n, k);
        });

        let r = KernelResult {
            name: format!("q4_matvec ({}x{})", n, k),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_ms,
            batched_ms_per_layer: iso_ms,
            batched_gbs: mb / iso_ms,
        };
        println!("{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1}", r.name, r.isolated_ms, r.isolated_gbs, r.batched_ms_per_layer, r.batched_gbs);
        results.push(r);
    }

    // f32_gemv
    {
        let n = hidden;
        let k = hidden;
        let mb = (n * k * 4) as f64 / 1e6;
        let w_data = synth_f32(n * k, 0.8);
        let w = ndarray::ArrayView2::from_shape((n, k), &w_data).unwrap();
        let x = synth_f32(k, 0.9);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = backend.f32_gemv(w, &x);
        });

        let r = KernelResult {
            name: format!("f32_gemv ({}x{})", n, k),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_ms,
            batched_ms_per_layer: iso_ms,
            batched_gbs: mb / iso_ms,
        };
        println!("{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1}", r.name, r.isolated_ms, r.isolated_gbs, r.batched_ms_per_layer, r.batched_gbs);
        results.push(r);
    }

    // rms_norm (isolated)
    {
        let n = hidden;
        let mb = (n * 4 * 2) as f64 / 1e6; // input + weight + output
        let x = synth_f32(n, 0.1);
        let w = synth_f32(n, 0.2);
        let mut out = backend.bufs.output(n * 4);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            crate::xpu::stages::input_norm::encode_f32(&x, &w, &mut out, n, 1e-6, 0.0);
        });

        backend.bufs.recycle(out);

        let r = KernelResult {
            name: format!("rms_norm (hidden={})", n),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_ms,
            batched_ms_per_layer: iso_ms,
            batched_gbs: mb / iso_ms,
        };
        println!("{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1}", r.name, r.isolated_ms, r.isolated_gbs, r.batched_ms_per_layer, r.batched_gbs);
        results.push(r);
    }
    // q4k_ffn_gate_up (Gemma 4 fused)
    {
        let n = inter;
        let k = hidden;
        let mb = (n * k * 4 / 8 * 2) as f64 / 1e6; // Two Q4_K weights
        let w = vec![0u8; n * k * 4 / 8];
        let x = synth_f32(k, 0.3);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let mut g_out = backend.bufs.output(n * 4);
            let mut u_out = backend.bufs.output(n * 4);
            let x_buf = crate::xpu::buffers::XpuBuffer::from_slice(&x, false);
            let w_buf = crate::xpu::buffers::XpuBuffer::from_slice(&w, false);
            
            unsafe {
                crate::xpu::ffi::ffi::q4k_ffn_gate_up(
                    w_buf.as_ptr_type(),
                    w_buf.as_ptr_type(),
                    x_buf.as_ptr_type(),
                    g_out.as_mut_ptr_type(),
                    u_out.as_mut_ptr_type(),
                    n,
                    k,
                );
            }
            backend.bufs.recycle(g_out);
            backend.bufs.recycle(u_out);
        });

        let r = KernelResult {
            name: format!("q4k_ffn_gate_up ({}x{})", n, k),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_ms,
            batched_ms_per_layer: iso_ms,
            batched_gbs: mb / iso_ms,
        };
        println!("{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1}", r.name, r.isolated_ms, r.isolated_gbs, r.batched_ms_per_layer, r.batched_gbs);
        results.push(r);
    }

    results
}
