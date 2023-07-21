use std::{
    arch::x86_64::*,
    mem::{self, transmute},
};

// pub use shishua::*;
pub use simdrand::*;
pub use vecs::*;
pub use xoshiro256plusplus::*;
pub use xoshiro256plus::*;

// mod shishua;
mod simdrand;
mod vecs;
mod xoshiro256plusplus;
mod xoshiro256plus;

#[inline(always)]
fn read_u64_into_vec(src: &[u8], dst: &mut __m512i) {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 8);
    unsafe {
        *dst = _mm512_set_epi64(
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 4)..(SIZE * 5)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 5)..(SIZE * 6)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 6)..(SIZE * 7)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 7)..(SIZE * 8)].try_into().unwrap())),
        )
    }
}

#[inline(always)]
fn rotate_left<const K: i32>(x: __m512i) -> __m512i {
    unsafe {
        // rotl: (x << k) | (x >> (64 - k))
        let left = _mm512_sll_epi64(x, _mm_cvtsi32_si128(K));
        let right = _mm512_srl_epi64(x, _mm_cvtsi32_si128(64 - K));
        _mm512_or_si512(left, right)
    }
}
