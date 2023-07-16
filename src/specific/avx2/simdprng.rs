use std::arch::x86_64::*;

use super::vecs::*;

pub trait SimdPrng {
    fn next_m256i(&mut self, vector: &mut __m256i);
    fn next_u64x4(&mut self, vector: &mut U64x4);
    fn next_f64x4(&mut self, vector: &mut F64x4);
}