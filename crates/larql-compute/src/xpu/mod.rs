//! XPU (SYCL) compute backend — Intel Arc / Xe Graphics.
//!
//! All operations go through the [`ComputeBackend`] trait. XPU-specific
//! optimisations: USM (Unified Shared Memory) zero-copy buffers, SYCL 
//! subgroup reductions, and Intel ICPX device bundling via a dedicated DLL.
//!
//! ## Modules
//!
//! - `kernels.cpp`: SYCL C++ kernels — compiled into `larql_xpu.dll`.
//! - `ops/`:       GPU dispatch — modular dispatchers for each operation.
//! - `trait_impl/`: Backend trait implementations (ComputeBackend, MatMul, etc.).
//! - `buffers`:    SYCL USM buffer management.
//! - `ffi`:        Low-level bridge to the SYCL DLL.
//!
//! ## Requirements
//!
//! - Intel oneAPI Base Toolkit (setvars.bat must be sourced).
//! - Intel Arc Pro / Xe2 GPU or compatible SYCL device.

pub mod buffers;
mod decode;
mod ffi;
pub mod ops;
mod trait_impl;

use ffi::ffi as xpu_ffi;

/// XPU (SYCL) compute backend.
///
/// Encapsulates the SYCL runtime and manages kernel dispatch to Intel GPUs.
/// Built as a Bridge-to-DLL architecture to ensure stability on Windows.
pub struct XpuBackend {}

impl XpuBackend {
    /// Create a new XPU backend.
    ///
    /// Initializes the SYCL queue and verifies device availability.
    /// Returns `None` if no SYCL-compatible device is found or if the 
    /// ICPX DLL fails to load.
    pub fn new() -> Option<Self> {
        xpu_ffi::check_sycl();
        if xpu_ffi::init_xpu() {
            Some(Self {})
        } else {
            None
        }
    }
}
