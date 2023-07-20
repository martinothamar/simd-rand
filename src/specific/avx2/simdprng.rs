use std::{arch::x86_64::*, mem::transmute};

use super::vecs::*;

pub trait SimdPrng {
    fn next_m256i(&mut self, vector: &mut __m256i);
    
    // Unfortunately this is not fast enough,
    // since there is no direct intrinsic for u64 -> f64 conversion (other than in avx512)
    // TODO: if avx512, we can do better
    #[inline(always)]
    fn next_m256d(&mut self, result: &mut __m256d) {
        // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        unsafe {
            let mut v = Default::default();
            self.next_f64x4(&mut v);
            
            *result = _mm256_load_pd(v.as_ptr());
        }
    }

    #[inline(always)]
    fn next_u64x4(&mut self, vector: &mut U64x4) {
        unsafe {
            let mut v = _mm256_setzero_si256();
            self.next_m256i(&mut v);
            _mm256_store_si256(transmute::<_, *mut __m256i>(vector), v);
        }
    }

    #[inline(always)]
    fn next_f64x4(&mut self, vector: &mut F64x4) {
        let mut v = Default::default();
        self.next_u64x4(&mut v);

        // TODO - convert from vector
        vector[0] = (v[0] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[1] = (v[1] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[2] = (v[2] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[3] = (v[3] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    }

}

#[inline(always)]
fn next_m256d_pure_avx2(v: &mut __m256i, result: &mut __m256d) {
    // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    unsafe {
        let lhs1 = _mm256_srl_epi64(*v, _mm_cvtsi32_si128(11));
        let lhs2 = super::u64_to_f64(lhs1);

        let rhs = _mm256_set1_pd(1.1102230246251565E-16);
        *result = _mm256_mul_pd(lhs2, rhs)
    }
}
