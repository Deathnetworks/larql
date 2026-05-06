//! XPU kernel handle and trait system.

pub mod handle;
pub mod traits;

pub use handle::KernelHandle;
pub use traits::{ShaderKernel, TiledKernel};
