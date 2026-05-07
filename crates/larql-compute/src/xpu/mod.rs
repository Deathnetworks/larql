//! XPU (SYCL) compute backend — Intel Arc / Xe Graphics.

pub mod buffers;
pub mod calibrate;
mod decode;
mod decode_hybrid;
pub mod diag;
mod direct_ops;
mod ffi;
pub mod f32_ops;
pub mod kernel;
mod moe_dispatch;
pub mod ops;
mod pipeline;
mod prefill;
// pub mod shaders;
pub mod stages;
mod trait_impl;

use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicUsize;
use ffi::ffi as xpu_ffi;
use buffers::BufferCache;
use f32_ops::F32Ops;
use ops::q4_common::Q4Pipelines;

/// XPU (SYCL) compute backend.
pub struct XpuBackend {
    pub bufs: BufferCache,
    pub f32_ops: F32Ops,
    pub q4: Q4Pipelines,
    flop_threshold: AtomicUsize,
    kv_cache: Mutex<Option<ops::kv_cache::KVCache>>,
    moe_scratch: Mutex<Option<moe_dispatch::MoeScratch>>,
}

impl XpuBackend {
    /// Create a new XPU backend.
    pub fn new() -> Option<Self> {
        xpu_ffi::check_sycl();
        if xpu_ffi::init_xpu() {
            let bufs = BufferCache::new();
            let f32_ops = F32Ops::new();
            let q4 = Q4Pipelines { /* stubs */ };
            let backend = Self {
                bufs,
                f32_ops,
                q4,
                flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
                kv_cache: Mutex::new(None),
                moe_scratch: Mutex::new(None),
            };
            backend.calibrate();
            Some(backend)
        } else {
            None
        }
    }

    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.f32_ops, &self.bufs);
        self.flop_threshold.store(threshold, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn flop_threshold(&self) -> usize {
        self.flop_threshold.load(std::sync::atomic::Ordering::Relaxed)
    }
}
