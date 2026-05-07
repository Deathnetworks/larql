//! Profiling stub for XPU decode pipeline.

pub struct ProfileTimings {
    pub attn_ms: f64,
    pub gate_up_ms: f64,
    pub down_ms: f64,
}

pub fn split_profile_requested() -> bool {
    false
}

pub fn store_last_split_timings(_timings: ProfileTimings) {}
