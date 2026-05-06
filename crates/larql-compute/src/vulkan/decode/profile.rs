// TODO: Port from Metal decode/profile.rs (7 fns, 5,521 bytes)

/// Per-layer timing data from a decode pass.
#[derive(Debug, Default, Clone)]
pub struct ProfileTimings {
    pub attn_ms: f64,
    pub ffn_ms: f64,
    pub residual_ms: f64,
}
