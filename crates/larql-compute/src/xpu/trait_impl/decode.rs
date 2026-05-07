use super::*;
use crate::backend::DecodeBackend;

impl DecodeBackend for XpuBackend {
    fn has_kv_cache(&self) -> bool {
        false
    }
}
