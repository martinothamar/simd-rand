use std::{
    arch::x86_64::*,
    mem::{self, transmute},
};

pub use shishua::*;
pub use simdrand::*;
pub use vecs::*;
pub use xoshiro256plus::*;
pub use xoshiro256plusplus::*;

mod shishua;
mod simdrand;
mod vecs;
mod xoshiro256plus;
mod xoshiro256plusplus;

#[inline(always)]
fn read_u64_into_vec(src: &[u8]) -> __m256i {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 4);
    unsafe {
        _mm256_set_epi64x(
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap())),
        )
    }
}

#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
fn rotate_left<const K: i32>(x: __m256i) -> __m256i {
    // rotl: (x << k) | (x >> (64 - k))
    unsafe { _mm256_rol_epi64::<K>(x) }
}

#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl")))]
fn rotate_left<const K: i32>(x: __m256i) -> __m256i {
    // rotl: (x << k) | (x >> (64 - k))
    unsafe {
        let left = _mm256_sll_epi64(x, _mm_cvtsi32_si128(K));
        let right = _mm256_srl_epi64(x, _mm_cvtsi32_si128(64 - K));
        _mm256_or_si256(left, right)
    }
}
