use std::{arch::x86_64::*, mem};

// pub use shishua::*;
pub use simdrand::*;
pub use vecs::*;
pub use xoshiro256plus::*;
pub use xoshiro256plusplus::*;

// mod shishua;
mod simdrand;
mod vecs;
mod xoshiro256plus;
mod xoshiro256plusplus;

#[allow(clippy::identity_op, clippy::erasing_op)]
#[inline(always)]
fn read_u64_into_vec(src: &[u8]) -> __m512i {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 8);
    unsafe {
        _mm512_set_epi64(
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 4)..(SIZE * 5)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 5)..(SIZE * 6)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 6)..(SIZE * 7)].try_into().unwrap())),
            u64::cast_signed(u64::from_le_bytes(src[(SIZE * 7)..(SIZE * 8)].try_into().unwrap())),
        )
    }
}
