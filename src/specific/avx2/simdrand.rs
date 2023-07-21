use std::{
    arch::{asm, x86_64::*},
    mem::transmute,
};

use super::vecs::*;

pub trait SimdRand {
    fn next_m256i(&mut self, vector: &mut __m256i);

    // Unfortunately this is not fast enough,
    // since there is no direct intrinsic for u64 -> f64 conversion (other than in avx512)
    // TODO: if avx512, we can do better
    #[inline(always)]
    fn next_m256d(&mut self, result: &mut __m256d) {
        // (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        unsafe {
            let mut v = _mm256_setzero_si256();
            self.next_m256i(&mut v);

            let lhs = m256i_to_m256d(_mm256_srl_epi64(v, _mm_cvtsi32_si128(11)));

            // PERF: This is precomputed based on the constants from the formula above
            // I found no other efficient (and succint) constant way of representing the RHS.
            // setzero and constants used in shifts like 11 above are automatically constant folded.
            // Writing out the actual formula ended up not being constant folded by the compiler.
            const RHS_FACTOR: [f64; 4] = [1.1102230246251565E-16; 4];
            const RHS: __m256d = unsafe { transmute::<[f64; 4], __m256d>(RHS_FACTOR) };

            *result = _mm256_mul_pd(lhs, RHS)
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
        unsafe {
            let mut v = _mm256_setzero_pd();
            self.next_m256d(&mut v);
            _mm256_store_pd(vector.as_mut_ptr(), v);
        }
    }
}

#[inline(always)]
unsafe fn m256i_to_m256d(v: __m256i) -> __m256d {
    if is_x86_feature_detected!("avx512dq") || is_x86_feature_detected!("avx512vl") {
        // With AVX512 DQ/VL we can use the below instruction
        // with both 512bit and 256bit vectors
        // see https://www.felixcloutier.com/x86/vcvtuqq2pd
        let mut dst: __m256d;
        asm!(
            "vcvtuqq2pd {1}, {0}",
            in(ymm_reg) v,
            out(ymm_reg) dst,
        );
        dst
    } else {
        // No direct conv intrinsic in AVX2, this "hack" is from
        // https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
        // Unfortunately not faster than just loading and operating on the scalars
        let magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);
        let magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);
        let magic_i_all = _mm256_set1_epi64x(0x4530000000100000);
        let magic_d_all = _mm256_castsi256_pd(magic_i_all);

        let v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
        let v_hi = _mm256_srli_epi64(v, 32);
        let v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);
        let v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
        let result = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
        result
    }
}
