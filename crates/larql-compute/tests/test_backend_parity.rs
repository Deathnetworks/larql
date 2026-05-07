use larql_compute::prelude::*;
use larql_compute::{default_backend, cpu_backend};
use larql_compute::pipeline::{QuantFormat, QuantWeight};
use ndarray::{Array2, ArrayView2};

/// A generalized Numerical Parity Test Harness for validating XPU and Vulkan backends
/// against the CpuBackend. This ensures that variable warp sizes (16, 32, 64) 
/// don't cause PPL drift or structural corruption when migrating Metal shaders.
#[test]
fn parity_harness_f32_gemv() {
    let cpu = cpu_backend();
    let gpu = default_backend();

    // If default_backend falls back to CPU, skip the test.
    if gpu.name().contains("cpu") {
        println!("No XPU, Vulkan, or Metal backend found. Skipping parity test.");
        return;
    }

    println!("Running numerical parity harness on backend: {}", gpu.device_info());

    let m = 256;
    let k = 1024;
    
    // Create random F32 inputs
    println!("Creating input data...");
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
    let x_data: Vec<f32> = (0..k).map(|i| (i as f32).cos()).collect();
    
    let a = ArrayView2::from_shape((m, k), &a_data).unwrap();
    
    println!("Running CPU baseline (pure Rust)...");
    let mut cpu_y = vec![0.0f32; m];
    for r in 0..m {
        let mut sum = 0.0f32;
        for c in 0..k {
            sum += x_data[c] * a_data[r * k + c];
        }
        cpu_y[r] = sum;
    }
    let x_view = ArrayView2::from_shape((1, k), &x_data).unwrap(); // Still need it for fallback path if used

    println!("Running GPU implementation...");
    let gpu_y = if gpu.supports(Capability::F32Gemv) {
        gpu.f32_gemv(a, &x_data).expect("f32_gemv failed")
    } else {
        let gpu_y_array = gpu.matmul_transb(x_view, a);
        gpu_y_array.as_slice().unwrap().to_vec()
    };
    println!("GPU implementation finished.");

    // Exact byte / precision comparison
    let mut max_err = 0.0f32;
    for (c, g) in cpu_y.iter().zip(gpu_y.iter()) {
        let err = (c - g).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // F32 math can have tiny floating point variations depending on reduction order,
    // but the tolerance should be extremely tight.
    assert!(
        max_err < 1e-3, 
        "Numerical Parity Failed! Max drift between CPU and {} was {}. Check your sub_group padding.", 
        gpu.name(), max_err
    );
}
