use std::arch::x86_64::*;

pub use shishua::*;
pub use simdprng::*;
pub use vecs::*;
pub use xoshiro256plusplus::*;

mod shishua;
mod simdprng;
mod vecs;
mod xoshiro256plusplus;

// No direct conv intrinsic in AVX2, this hack is from
// https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
// Unfortunately not faster than just loading and operating on the scalars
#[cfg_attr(dasm, inline(never))]
#[cfg_attr(not(dasm), inline(always))]
unsafe fn u64_to_f64(v: __m256i) -> __m256d {
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
