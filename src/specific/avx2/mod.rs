use std::{arch::x86_64::*, mem::{self, transmute}};

pub use shishua::*;
pub use simdprng::*;
pub use vecs::*;
pub use xoshiro256plusplus::*;
pub use xoshiro256plus::*;

mod shishua;
mod simdprng;
mod vecs;
mod xoshiro256plusplus;
mod xoshiro256plus;

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

#[inline(always)]
fn read_u64_into_vec(src: &[u8], dst: &mut __m256i) {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 4);
    unsafe {
        *dst = _mm256_set_epi64x(
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap(),
            )),
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap(),
            )),
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap(),
            )),
            transmute::<_, i64>(u64::from_le_bytes(
                src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap(),
            )),
        )
    }
}

#[cfg_attr(dasm, inline(never))]
#[cfg_attr(not(dasm), inline(always))]
fn rotate_left<const K: i32>(x: __m256i) -> __m256i {
    unsafe {
        // rotl: (x << k) | (x >> (64 - k))
        let left = _mm256_sll_epi64(x, _mm_cvtsi32_si128(K));
        let right = _mm256_srl_epi64(x, _mm_cvtsi32_si128(64 - K));
        _mm256_or_si256(left, right)
    }
}
