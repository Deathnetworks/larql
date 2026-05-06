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
        println!("cargo:rerun-if-changed=src/xpu/ffi.rs");
        println!("cargo:rerun-if-changed=src/xpu/kernels.cpp");
        println!("cargo:rerun-if-changed=src/xpu/kernels.hpp");

        cxx_build::bridge("src/xpu/ffi.rs")
            .file("src/xpu/kernels.cpp")
            .flag_if_supported("-fsycl") // Enable SYCL
            .compiler("icpx")           // Use Intel oneAPI compiler
            .flag("-fno-stack-protector") // Disable stack security checks
            .flag("-g")                   // Debug info
            .compile("larql-xpu");

        println!("cargo:rustc-link-search=native=C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2025.3\\lib");
        println!("cargo:rustc-link-lib=sycl8");
        println!("cargo:rustc-link-lib=sycl-devicelib-host");
        println!("cargo:rustc-link-lib=libircmt");
        println!("cargo:rustc-link-lib=svml_dispmd");
        println!("cargo:rustc-link-lib=libmmd");
    }
}
