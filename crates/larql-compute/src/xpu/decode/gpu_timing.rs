//! GPU Timing stub for XPU decode pipeline.

#[derive(Default)]
pub struct TokenGpuTime {
    pub attn_ms: f64,
    pub gate_up_ms: f64,
    pub down_ms: f64,
}

pub enum DecodeStage {
    Attention,
    GateUp,
    Down,
}

impl TokenGpuTime {
    pub fn record_stage(&mut self, _cmd: &(), _stage: DecodeStage) {}
    pub fn record(&mut self, _cmd: &()) {}
    pub fn print_if_enabled(&self, _wall_ms: f64) {}
}
