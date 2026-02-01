use core::{arch::x86_64::*, mem::transmute};

use super::vecs::*;

pub trait SimdRand {
    fn next_m256i(&mut self) -> __m256i;

    #[allow(clippy::items_after_statements)]
    #[inline(always)]
    fn next_m256d(&mut self) -> __m256d {
        unsafe {
            let v = self.next_m256i();

            let lhs = m256i_to_m256d(_mm256_srli_epi64::<11>(v));

            // PERF: This is precomputed based on the constants from the formula above
            // I found no other efficient (and succint) constant way of representing the RHS.
            // setzero and constants used in shifts like 11 above are automatically constant folded.
            // Writing out the actual formula ended up not being constant folded by the compiler.
            const RHS_FACTOR: [f64; 4] = [1.1102230246251565E-16; 4];
            const RHS: __m256d = unsafe { transmute::<[f64; 4], __m256d>(RHS_FACTOR) };

            _mm256_mul_pd(lhs, RHS)
        }
    }

    #[inline(always)]
    fn next_u64x4(&mut self) -> U64x4 {
        unsafe {
            let v = self.next_m256i();
            let mut vector = U64x4::default();
            _mm256_store_si256(core::ptr::from_mut(&mut vector).cast::<__m256i>(), v);
            vector
        }
    }

    #[inline(always)]
    fn next_f64x4(&mut self) -> F64x4 {
        unsafe {
            let v = self.next_m256d();
            let mut vector = F64x4::default();
            _mm256_store_pd(core::ptr::from_mut(&mut vector).cast::<f64>(), v);
            vector
        }
    }
}

#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512dq", target_feature = "avx512vl"))]
unsafe fn m256i_to_m256d(v: __m256i) -> __m256d {
    unsafe { _mm256_cvtepu64_pd(v) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512dq", target_feature = "avx512vl")))]
unsafe fn m256i_to_m256d(v: __m256i) -> __m256d {
    // No direct conv intrinsic in AVX2, this "hack" is from
    // https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
    // Unfortunately not faster than just loading and operating on the scalars
    // SAFETY: relies on AVX2 being enabled for the build.
    unsafe {
        let magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);
        let magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);
        let magic_i_all = _mm256_set1_epi64x(0x4530000000100000);
        let magic_d_all = _mm256_castsi256_pd(magic_i_all);

        let v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
        let v_hi = _mm256_srli_epi64::<32>(v);
        let v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);
        let v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
        let result = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
        result
    }
}
