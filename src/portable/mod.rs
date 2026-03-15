use core::mem;
use core::{
    ops::{BitOr, Shl, Shr},
    simd::Simd,
};

pub use frandx4::*;
pub use frandx8::*;
pub use simdrand::*;
pub use xoshiro256plusplusx4::*;
pub use xoshiro256plusplusx8::*;
pub use xoshiro256plusx4::*;
pub use xoshiro256plusx8::*;

mod frandx4;
mod frandx8;
mod simdrand;
mod xoshiro256plusplusx4;
mod xoshiro256plusplusx8;
mod xoshiro256plusx4;
mod xoshiro256plusx8;

#[inline(always)]
fn read_u64_into_vec<const N: usize>(src: &[u8]) -> Simd<u64, N> {
    Simd::<u64, N>::from_array(read_u64_array(src))
}

#[inline(always)]
fn read_u64_array<const N: usize>(src: &[u8]) -> [u64; N] {
    assert!(src.len() == mem::size_of::<u64>() * N);

    let (chunks, remainder) = src.as_chunks::<8>();
    assert!(remainder.is_empty());
    assert!(chunks.len() == N);

    let mut values = [0; N];
    for (dst, chunk) in values.iter_mut().zip(chunks) {
        *dst = u64::from_le_bytes(*chunk);
    }

    values
}

#[inline(always)]
// Multiple trait bounds on the SIMD value are required; clippy sees them as repetition
#[allow(clippy::type_repetition_in_bounds)]
fn rotate_left<const N: usize>(x: Simd<u64, N>, k: u64) -> Simd<u64, N>
where
    Simd<u64, N>: Shl<Simd<u64, N>, Output = Simd<u64, N>>,
    Simd<u64, N>: Shr<Simd<u64, N>, Output = Simd<u64, N>>,
    Simd<u64, N>: BitOr<Simd<u64, N>, Output = Simd<u64, N>>,
{
    let kv = Simd::<u64, N>::splat(k);
    let left = x << kv;
    let right = x >> Simd::<u64, N>::splat(64 - k);
    left | right
}
