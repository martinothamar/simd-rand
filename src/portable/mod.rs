use std::{
    fmt::Debug,
    mem,
    ops::{BitOr, Shl, Shr, Sub},
    simd::{Simd, SimdElement},
};

pub use simdrand::*;
pub use xoshiro256plusplusx4::*;
pub use xoshiro256plusplusx8::*;
pub use xoshiro256plusx4::*;
pub use xoshiro256plusx8::*;

mod simdrand;
mod xoshiro256plusplusx4;
mod xoshiro256plusplusx8;
mod xoshiro256plusx4;
mod xoshiro256plusx8;

#[inline(always)]
fn read_u64_into_vec<const N: usize>(src: &[u8]) -> Simd<u64, N> {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * N);

    let mut scalars: [u64; N] = [0; N];

    for i in 0..N {
        scalars[i] = u64::from_le_bytes(src[(SIZE * i)..(SIZE * (i + 1))].try_into().unwrap());
    }

    Simd::<u64, N>::from_array(scalars)
}

#[inline(always)]
// Generics in rust is great
fn rotate_left<T, const N: usize>(x: Simd<T, N>, k: T) -> Simd<T, N>
where
    T: SimdElement + Sub<T, Output = T>,
    usize: TryInto<T>,
    <usize as TryInto<T>>::Error: Debug,
    Simd<T, N>: Shl<Simd<T, N>, Output = Simd<T, N>>,
    Simd<T, N>: Shr<Simd<T, N>, Output = Simd<T, N>>,
    Simd<T, N>: BitOr<Simd<T, N>, Output = Simd<T, N>>,
{
    let bitsize = mem::size_of::<T>() * 8;
    let kv = Simd::<T, N>::splat(k);
    let left = x << kv;
    let right = x >> Simd::<T, N>::splat(bitsize.try_into().unwrap() - k);
    left | right
}
