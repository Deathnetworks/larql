fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");

    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");
    build.opt_level(3);

    #[cfg(target_arch = "aarch64")]
    build.flag_if_supported("-march=armv8.2-a+dotprod");

    #[cfg(target_arch = "x86_64")]
    build.flag_if_supported("-mavx2");

    build.compile("q4_dot");

    #[cfg(feature = "xpu")]
    {
        use std::process::Command;
        use std::env;
        use std::path::PathBuf;

        println!("cargo:rerun-if-changed=src/xpu/ffi.rs");
        println!("cargo:rerun-if-changed=src/xpu/kernels.cpp");
        println!("cargo:rerun-if-changed=src/xpu/kernels.hpp");
        println!("cargo:rerun-if-changed=src/xpu/bridge_impl.cpp");

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        
        // 1. Build the standalone SYCL DLL
        let status = Command::new("icpx")
            .arg("-fsycl")
            .arg("-shared")
            .arg("-DSYCL_DLL_BUILD")
            .arg("-o").arg(out_dir.join("larql_xpu.dll"))
            .arg("-I").arg(&manifest_dir)
            .arg("-I").arg(manifest_dir.parent().unwrap())
            .arg("src/xpu/kernels.cpp")
            .status()
            .expect("Failed to run icpx for DLL");

        if !status.success() {
            panic!("icpx DLL build failed");
        }

        // 2. Build the CXX bridge using bridge_impl.cpp (which calls the DLL)
        cxx_build::bridge("src/xpu/ffi.rs")
            .file("src/xpu/bridge_impl.cpp")
            .define("SYCL_BRIDGE_ONLY", None)
            .include(&manifest_dir)
            .include(manifest_dir.parent().unwrap())
            .include(manifest_dir.parent().unwrap().parent().unwrap())
            .flag_if_supported("/GS-")
            .compile("larql-xpu-bridge");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=dylib=larql_xpu");
        
        println!("cargo:rustc-link-search=native=C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib");
        println!("cargo:rustc-link-lib=sycl8");
        println!("cargo:rustc-link-lib=sycl-devicelib-host");
        println!("cargo:rustc-link-lib=libircmt");
        println!("cargo:rustc-link-lib=svml_dispmd");
        println!("cargo:rustc-link-lib=libmmd");
    }
}
