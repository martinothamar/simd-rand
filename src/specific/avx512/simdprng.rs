use std::arch::x86_64::*;

use super::vecs::*;

pub trait SimdPrng {
    fn next_m512i(&mut self, vector: &mut __m512i);
    fn next_m512d(&mut self, vector: &mut __m512d);
    fn next_u64x8(&mut self, vector: &mut U64x8);
    fn next_f64x8(&mut self, vector: &mut F64x8);
}
