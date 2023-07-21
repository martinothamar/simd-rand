use std::{mem, simd::u64x4};

pub use simdrand::*;
pub use xoshiro256plusx4::*;

mod simdrand;
mod xoshiro256plusx4;

#[inline(always)]
fn read_u64_into_vec(src: &[u8], dst: &mut u64x4) {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 4);

    dst[0] = u64::from_le_bytes(src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap());
    dst[1] = u64::from_le_bytes(src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap());
    dst[2] = u64::from_le_bytes(src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap());
    dst[3] = u64::from_le_bytes(src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap());
}

#[inline(always)]
fn rotate_left<const K: u64>(x: u64x4) -> u64x4 {
    let k = u64x4::splat(K);
    // rotl: (x << k) | (x >> (64 - k))
    let left = x << k;
    let right = x >> u64x4::splat(64 - K);
    return left | right;
}
