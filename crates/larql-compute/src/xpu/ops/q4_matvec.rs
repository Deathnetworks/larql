//! Q4 matrix-vector dispatch for XPU.

pub fn dispatch(
    _q4: &[u8],
    _x: &[f32],
    _n: usize,
    _k: usize,
) -> Vec<f32> {
    unimplemented!("XPU q4_matvec not yet ported")
}
